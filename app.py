# DermaAI v2 — FastAPI Web Application
# Run with: uvicorn app:app --reload

import os
import uuid
import queue
import json
import asyncio
import threading
import tempfile
import time
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

app = FastAPI(title="DermaAI v2")

# ── Static files and templates ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ── In-memory session store ───────────────────────────────────────────────────
# Each session holds the full state for one diagnosis run.
SESSIONS: dict[str, dict] = {}


def _new_session() -> dict:
    return {
        "status": "intake",          # intake | clarifying | analyzing | review | approved
        "patient_text": "",
        "enriched_text": "",
        "pending_questions": [],     # questions waiting for the patient's answers
        "image_path": "",
        "progress_queue": queue.Queue(),
        "derma_crew": None,
        "result": None,
        "audit": None,
        "error": None,
        "pdf_paths": {},             # { "doctor": path, "patient": path, "audit": path }
        "_created_at": time.time(),  # used by the cleanup job
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _audit_to_dict(audit) -> dict:
    """Serialise an AuditTrail to a plain dict the frontend can consume."""
    if audit is None:
        return {}

    def _pyd(obj):
        if obj is None:
            return None
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return str(obj)

    return {
        "patient_text": audit.patient_text,
        "image_path": audit.image_path,
        "vision_colour_raw": audit.vision_colour_raw,
        "vision_texture_raw": audit.vision_texture_raw,
        "vision_levelling_raw": audit.vision_levelling_raw,
        "vision_border_raw": audit.vision_border_raw,
        "vision_shape_raw": audit.vision_shape_raw,
        "vision_pattern_raw": audit.vision_pattern_raw,
        "biodata_summary": audit.biodata_summary,
        "colour_output": _pyd(audit.colour_output),
        "texture_output": _pyd(audit.texture_output),
        "levelling_output": _pyd(audit.levelling_output),
        "border_output": _pyd(audit.border_output),
        "shape_output": _pyd(audit.shape_output),
        "pattern_output": _pyd(getattr(audit, "pattern_output", None)),
        "decomposition_output": _pyd(audit.decomposition_output),
        "research_output": _pyd(audit.research_output),
        "differential_output": _pyd(audit.differential_output),
        "mimic_resolution_output": _pyd(audit.mimic_resolution_output),
        "visual_differential_review_output": _pyd(getattr(audit, "visual_differential_review_output", None)),
        "cmo_output": _pyd(audit.cmo_output),
        "treatment_output": _pyd(audit.treatment_output),
        "final_diagnosis": _pyd(audit.final_diagnosis),
        "raw_outputs": audit.raw_outputs,
        "adapter_status": audit.adapter_status,
        "adapter_errors": audit.adapter_errors,
        "feedback_history": audit.feedback_history,
        "run_count": audit.run_count,
    }


def _result_to_dict(result) -> dict:
    if result is None:
        return {}
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {}


def _make_task_callback(session_id: str):
    """Return a task_callback that pushes progress events into the session queue."""
    def callback(task_output):
        sess = SESSIONS.get(session_id)
        if sess is None:
            return
        try:
            agent_name = getattr(task_output, "agent", "Agent")
            raw = getattr(task_output, "raw", "") or ""
            summary = raw[:120].replace("\n", " ") + ("…" if len(raw) > 120 else "")
            sess["progress_queue"].put({
                "type": "task_done",
                "agent": str(agent_name),
                "summary": summary,
            })
        except Exception:
            pass
    return callback


def _run_analysis_thread(session_id: str, is_rerun: bool = False,
                          feedback: str = "", scope: str = "full"):
    """Background thread: runs DermaCrew and pushes events to the session queue."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        return

    q: queue.Queue = sess["progress_queue"]
    callback = _make_task_callback(session_id)

    try:
        crew = sess["derma_crew"]

        if is_rerun:
            print(f"\n[App] Session {session_id[:8]} — Starting RE-RUN (scope: {scope})")
            result, audit = crew.rerun(
                feedback=feedback,
                scope=scope,
                task_callback=callback,
            )
        else:
            print(f"\n[App] Session {session_id[:8]} — Starting full analysis run")
            result, audit = crew.run(
                task_callback=callback,
                skip_clarification=True,   # clarification done before this call
            )

        sess["result"] = result
        sess["audit"] = audit
        sess["status"] = "review"

        print(f"\n[App] Session {session_id[:8]} — Analysis COMPLETE")
        print(f"[App]   primary_diagnosis : {getattr(result, 'primary_diagnosis', 'N/A')}")
        print(f"[App]   confidence        : {getattr(result, 'confidence', 'N/A')}")
        print(f"[App]   severity          : {getattr(result, 'severity', 'N/A')}")
        print(f"[App]   adapter_status    : {getattr(audit, 'adapter_status', {}).get('final_diagnosis', 'unknown')}")

        q.put({"type": "complete"})

    except Exception as e:
        sess["error"] = str(e)
        sess["status"] = "error"
        q.put({"type": "error", "message": str(e)})
        print(f"[App] Session {session_id[:8]} — Analysis ERROR: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/profile")
async def get_saved_profile():
    """Return the saved patient profile so the UI can pre-fill the intake form."""
    from agents.biodata_agent import load_profile
    try:
        profile = load_profile()
        return JSONResponse(profile.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load profile: {e}")


@app.post("/api/start")
async def start_session(
    profile_json: str = Form(...),
    symptom_text: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    """
    Create a new diagnosis session.
    Saves the patient profile, optionally saves the uploaded image,
    runs the first clarification round and returns any questions.
    """
    # ── Parse and save profile ────────────────────────────────────────────────
    try:
        profile_data = json.loads(profile_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid profile JSON")

    from agents.biodata_agent import PatientProfile, save_profile
    try:
        profile = PatientProfile(**profile_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Profile validation error: {e}")

    save_profile(profile)

    # ── Save uploaded image ───────────────────────────────────────────────────
    image_path = ""
    if image and image.filename:
        suffix = Path(image.filename).suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            raise HTTPException(status_code=400, detail=f"Unsupported image format: {suffix}")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=BASE_DIR / "uploads")
        tmp.write(await image.read())
        tmp.close()
        image_path = tmp.name

    # ── Create session ────────────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    sess = _new_session()
    sess["patient_text"] = symptom_text
    sess["enriched_text"] = symptom_text
    sess["image_path"] = image_path
    SESSIONS[session_id] = sess

    # ── Clarification round 1 (non-blocking — runs the mini-crew synchronously) ──
    # This is acceptable because clarification is fast (2 small agents, no image).
    from utils.clarification_loop_web import run_clarification_round_web
    try:
        _, questions = run_clarification_round_web(symptom_text)
    except Exception as e:
        print(f"[Session {session_id}] Clarification error: {e}. Skipping.")
        questions = []

    sess["pending_questions"] = questions
    if questions:
        sess["status"] = "clarifying"
    else:
        sess["status"] = "ready"

    # ── Instantiate DermaCrew (does not run yet) ──────────────────────────────
    from crew.derma_crew import DermaCrew
    sess["derma_crew"] = DermaCrew(
        image_path=image_path,
        patient_text=symptom_text,
    )

    return JSONResponse({
        "session_id": session_id,
        "questions": questions,
    })


@app.post("/api/{session_id}/clarify")
async def submit_clarification(session_id: str, body: dict):
    """
    Accept the patient's answers to the clarification questions.
    Runs another clarification round and returns remaining questions (if any).
    """
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    answers: list[str] = body.get("answers", [])
    questions = sess.get("pending_questions", [])

    from utils.clarification_loop_web import append_answers_to_text, run_clarification_round_web

    # Append answers to enriched text
    enriched = append_answers_to_text(sess["enriched_text"], questions, answers)
    sess["enriched_text"] = enriched

    # Update the crew's patient_text so it uses enriched text during the main run
    if sess["derma_crew"]:
        sess["derma_crew"].patient_text = enriched

    # Run another clarification round to check if more info is needed
    try:
        _, new_questions = run_clarification_round_web(enriched)
    except Exception as e:
        print(f"[Session {session_id}] Clarification round 2 error: {e}. Skipping.")
        new_questions = []

    sess["pending_questions"] = new_questions
    if not new_questions:
        sess["status"] = "ready"

    return JSONResponse({"questions": new_questions})


@app.post("/api/{session_id}/analyze")
async def start_analysis(session_id: str):
    """Start the main crew analysis in a background thread."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if sess["status"] == "analyzing":
        return JSONResponse({"status": "already_running"})

    sess["status"] = "analyzing"
    # Clear any stale events from a previous run
    while not sess["progress_queue"].empty():
        sess["progress_queue"].get_nowait()

    thread = threading.Thread(
        target=_run_analysis_thread,
        args=(session_id,),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"status": "started"})


@app.get("/api/{session_id}/stream")
async def stream_progress(session_id: str, request: Request):
    """
    Server-Sent Events stream.
    Pushes task_done, complete, and error events as the crew runs.
    """
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    q: queue.Queue = sess["progress_queue"]

    async def event_generator():
        yield "data: {\"type\": \"connected\"}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                event = q.get(timeout=0.5)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("complete", "error"):
                    break
            except queue.Empty:
                # Send a heartbeat so the connection stays alive
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/{session_id}/result")
async def get_result(session_id: str):
    """Return the final result and audit trail once analysis is complete."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if sess["status"] == "analyzing":
        return JSONResponse({"status": "analyzing"})

    if sess["error"]:
        return JSONResponse({"status": "error", "error": sess["error"]})

    return JSONResponse({
        "status": "complete",
        "result": _result_to_dict(sess["result"]),
        "audit": _audit_to_dict(sess["audit"]),
    })


@app.post("/api/{session_id}/approve")
async def approve(session_id: str):
    """Doctor approves the diagnosis. Generates PDFs and marks session approved."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    audit = sess["audit"]
    result = sess["result"]

    if audit:
        audit.feedback_history.append({
            "round": audit.run_count,
            "action": "approved",
            "feedback": "",
        })

    sess["status"] = "approved"

    # Generate PDFs
    pdf_paths = {}
    os.makedirs(BASE_DIR / "reports", exist_ok=True)
    try:
        from pdf_service import save_reports, save_doctor_audit_pdf
        doctor_pdf, patient_pdf = save_reports(result, audit)
        audit_pdf = save_doctor_audit_pdf(audit)
        pdf_paths = {
            "doctor": f"/api/{session_id}/pdf/doctor",
            "patient": f"/api/{session_id}/pdf/patient",
            "audit": f"/api/{session_id}/pdf/audit",
        }
        sess["pdf_paths"] = {
            "doctor": doctor_pdf,
            "patient": patient_pdf,
            "audit": audit_pdf,
        }
    except Exception as e:
        print(f"[Session {session_id}] PDF generation error: {e}")
        pdf_paths = {}

    return JSONResponse({"status": "approved", "pdf_urls": pdf_paths})


@app.post("/api/{session_id}/reject")
async def reject(session_id: str, body: dict):
    """
    Doctor rejects the diagnosis.
    Starts a re-run in the background with the given feedback and scope.
    """
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    feedback: str = body.get("feedback", "No specific feedback provided.")
    scope: str = body.get("scope", "full")
    if scope not in ("full", "post_research", "orchestrator_only"):
        scope = "full"

    # Clear stale events
    while not sess["progress_queue"].empty():
        sess["progress_queue"].get_nowait()

    sess["status"] = "analyzing"
    callback = _make_task_callback(session_id)

    thread = threading.Thread(
        target=_run_analysis_thread,
        args=(session_id, True, feedback, scope),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"status": "rerunning", "scope": scope})


@app.get("/api/{session_id}/pdf/{pdf_type}")
async def download_pdf(session_id: str, pdf_type: str):
    """Serve a generated PDF file."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if pdf_type not in ("doctor", "patient", "audit"):
        raise HTTPException(status_code=400, detail="Invalid PDF type")

    pdf_paths = sess.get("pdf_paths", {})
    path = pdf_paths.get(pdf_type)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="PDF not yet generated")

    filename_map = {
        "doctor": "DermaAI_Doctor_Report.pdf",
        "patient": "DermaAI_Patient_Summary.pdf",
        "audit": "DermaAI_Audit_Trail.pdf",
    }

    return FileResponse(
        path,
        media_type="application/pdf",
        filename=filename_map[pdf_type],
    )


# ── Periodic cleanup ──────────────────────────────────────────────────────────

CLEANUP_MAX_AGE_SECONDS = 2 * 60 * 60  # 2 hours


def _cleanup_old_files():
    """
    Delete uploaded images and generated PDFs that are older than
    CLEANUP_MAX_AGE_SECONDS. Also removes stale completed/errored sessions
    from memory so the SESSIONS dict doesn't grow indefinitely.
    """
    now = time.time()
    deleted_files = 0
    deleted_sessions = 0

    for folder in (BASE_DIR / "uploads", BASE_DIR / "reports"):
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if not f.is_file():
                continue
            try:
                age = now - f.stat().st_mtime
                if age > CLEANUP_MAX_AGE_SECONDS:
                    f.unlink()
                    deleted_files += 1
            except Exception as e:
                print(f"[Cleanup] Could not delete {f}: {e}")

    stale_ids = [
        sid for sid, sess in list(SESSIONS.items())
        if sess.get("status") in ("approved", "error")
        and (now - sess.get("_created_at", now)) > CLEANUP_MAX_AGE_SECONDS
    ]
    for sid in stale_ids:
        SESSIONS.pop(sid, None)
        deleted_sessions += 1

    if deleted_files or deleted_sessions:
        print(
            f"[Cleanup] Removed {deleted_files} file(s) and "
            f"{deleted_sessions} session(s) older than "
            f"{CLEANUP_MAX_AGE_SECONDS // 3600}h"
        )


# ── Ollama watchdog ───────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_WATCHDOG_INTERVAL = 60   # seconds between health checks
_ollama_process: subprocess.Popen | None = None  # handle to a watchdog-spawned process


def _ollama_is_alive() -> bool:
    """Return True if Ollama is responding on its API port."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _start_ollama() -> None:
    """Launch `ollama serve` as a background subprocess."""
    global _ollama_process
    try:
        print("[Watchdog] Starting ollama serve...")
        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=open("/tmp/ollama.log", "a"),
            stderr=subprocess.STDOUT,
        )
        # Wait up to 30 s for it to become responsive
        for _ in range(30):
            time.sleep(1)
            if _ollama_is_alive():
                print("[Watchdog] Ollama is back up.")
                return
        print("[Watchdog] WARNING: Ollama did not respond within 30 s after restart.")
    except FileNotFoundError:
        print("[Watchdog] ERROR: 'ollama' binary not found — is Ollama installed?")
    except Exception as e:
        print(f"[Watchdog] ERROR starting Ollama: {e}")


def _ollama_watchdog_loop() -> None:
    """
    Background thread: checks Ollama every OLLAMA_WATCHDOG_INTERVAL seconds.
    If it stops responding, attempts to restart it automatically.
    """
    # Give the app a moment to finish starting before first check
    time.sleep(10)
    while True:
        if not _ollama_is_alive():
            print("[Watchdog] Ollama is not responding — attempting restart...")
            _start_ollama()
        time.sleep(OLLAMA_WATCHDOG_INTERVAL)


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Returns the live status of the app and the Ollama backend."""
    ollama_ok = _ollama_is_alive()
    return JSONResponse({
        "app": "ok",
        "ollama": "ok" if ollama_ok else "unavailable",
        "ollama_url": OLLAMA_BASE_URL,
        "active_sessions": len(SESSIONS),
    }, status_code=200 if ollama_ok else 503)


# ── Startup: ensure required directories exist ────────────────────────────────

@app.on_event("startup")
async def startup():
    (BASE_DIR / "static" / "css").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "static" / "js").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "templates").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "uploads").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "reports").mkdir(parents=True, exist_ok=True)

    # Start Ollama watchdog — auto-restarts Ollama if it crashes on RunPod
    watchdog = threading.Thread(target=_ollama_watchdog_loop, daemon=True, name="ollama-watchdog")
    watchdog.start()
    print(f"[Watchdog] Ollama watchdog started — checking every {OLLAMA_WATCHDOG_INTERVAL}s.")

    # If Ollama isn't already up, start it now
    if not _ollama_is_alive():
        print("[Watchdog] Ollama not detected on startup — attempting to start it...")
        threading.Thread(target=_start_ollama, daemon=True).start()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _cleanup_old_files,
        trigger="interval",
        hours=2,
        id="file_cleanup",
        replace_existing=True,
    )
    scheduler.start()
    print(f"[Cleanup] Scheduler started — files older than "
          f"{CLEANUP_MAX_AGE_SECONDS // 3600}h will be purged every 2 hours.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
