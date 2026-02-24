# Chapter 16 — Web Application

**Goal:** Replace the CLI entry point (`main.py`) with a FastAPI web application that exposes the full diagnostic pipeline through a browser UI. All agent logic, PDF generation, and the doctor approval loop are unchanged — the web layer wraps them with a REST + SSE API and a single-page frontend.

**Time estimate:** Already implemented. Read this chapter to understand what was built and how to extend it.

---

## Why FastAPI + Vanilla JS?

- **FastAPI** is already in `requirements.txt` and fits naturally alongside the existing Python codebase.
- **Vanilla JS** with no build step means zero npm, zero bundler, zero configuration.
- **Server-Sent Events (SSE)** push agent progress to the browser in real-time without the complexity of WebSockets.
- The CLI (`main.py`) is preserved unchanged — run it any time for debugging without the browser.

---

## File Structure

```
AgenticDermaAI/
├── app.py                          # FastAPI routes, session store, SSE
├── templates/
│   └── index.html                  # Single HTML shell (5 view divs)
├── static/
│   ├── css/
│   │   └── style.css               # Full medical aesthetic
│   └── js/
│       └── app.js                  # SPA view manager + SSE client
├── uploads/                        # Temp image files (auto-created)
├── utils/
│   ├── clarification_loop.py       # CLI version (unchanged)
│   └── clarification_loop_web.py   # NEW: returns questions instead of input()
└── crew/
    └── derma_crew.py               # MODIFIED: task_callback + skip_clarification
```

---

## Architecture

```
Browser (HTML/CSS/JS)
       │
       │  REST (JSON) + SSE
       ▼
FastAPI  app.py
       │
       ├── GET  /api/profile           Return saved patient_profile.json (for form pre-fill)
       ├── POST /api/start            Save profile, run clarification round 1
       ├── POST /api/{id}/clarify     Accept answers, run round 2
       ├── POST /api/{id}/analyze     Start background thread
       ├── GET  /api/{id}/stream      SSE — push task_done / complete / error
       ├── GET  /api/{id}/result      Return FinalDiagnosis + AuditTrail as JSON
       ├── POST /api/{id}/approve     Generate PDFs, mark approved
       ├── POST /api/{id}/reject      Trigger rerun with feedback + scope
       └── GET  /api/{id}/pdf/{type}  Serve PDF file
              │
              ▼
       DermaCrew.run(task_callback, skip_clarification=True)
       DermaCrew.rerun(feedback, scope, task_callback)
```

---

## Session State

Each browser session gets a UUID. The server keeps an in-memory dict:

```python
SESSIONS[session_id] = {
    "status":             "intake|clarifying|ready|analyzing|review|approved",
    "patient_text":       str,          # original symptom text
    "enriched_text":      str,          # after clarification answers appended
    "pending_questions":  list[str],    # questions waiting for the patient
    "image_path":         str,          # path to uploaded temp file (or "")
    "progress_queue":     queue.Queue(),# SSE event queue
    "derma_crew":         DermaCrew,    # instantiated on /api/start
    "result":             FinalDiagnosis | None,
    "audit":              AuditTrail | None,
    "error":              str | None,
    "pdf_paths":          dict,         # { "doctor": path, "patient": path, "audit": path }
}
```

> **Important:** Sessions live in memory. Restarting the server clears them. For production, persist sessions to Redis or a database.

---

## Step 1 — Install New Dependencies

```powershell
pip install uvicorn[standard] python-multipart jinja2
```

Or reinstall from requirements:
```powershell
pip install -r requirements.txt
```

---

## Step 2 — Run the Web App

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

`--reload` watches for code changes and restarts automatically. Remove it in production.

The existing `main.py` CLI still works independently:
```powershell
python main.py
```

---

## Step 3 — The Five UI Views

The frontend is a single-page application. JS swaps which view is visible. The top stepper reflects the current step.

### View 1 — Patient Intake (`view-intake`)

Renders on every page load (fresh session). Collects:
- All `PatientProfile` fields (name, age, sex, skin tone, occupation, etc.)
- Symptom description (required)
- Skin image (optional, drag-and-drop or browse)

**Load Saved Profile button:** Calls `GET /api/profile` and pre-fills all form fields from the previously saved `patient_profile.json`. Useful for returning patients — they only need to update their symptom description, not re-enter demographics every session.

On submit:
1. `POST /api/start` with `profile_json` (form data), `symptom_text`, and optional `image` file.
2. Server saves `patient_profile.json` and runs the first clarification round.
3. If questions come back → View 2. If not → directly to View 3.

### View 2 — Clarification (`view-clarification`)

Shows up to 2 rounds of AI-generated follow-up questions. Each question is an input field.

- **Submit Answers** → `POST /api/{id}/clarify` with the answer array. If more questions, re-renders; else proceeds to analysis.
- **Skip** → submits empty answers, moves to analysis.

The enriched patient text (original + Q&A) is stored server-side and passed to `DermaCrew` before `run()`.

### View 3 — Analysis (`view-analysis`)

Opens an SSE stream immediately: `GET /api/{id}/stream`.

Replaces the old two-panel (agent list + log) layout with a single seamless **pipeline flow**:

- **Progress bar** at the top fills from 0 → 100 % as agents complete (0 / 10 → 10 / 10).
- **Active-agent hero card** (dark gradient) shows the phase label, agent icon, agent name, and an animated three-dot indicator for the currently running stage. Turns green on completion or red on error.
- **Completed-stage grid** — as each agent finishes, a small card slides in below the hero showing the agent's icon, name, and a short human-readable insight extracted from its output (first meaningful string value from JSON, or first `Key: Value` line for biodata text). No timestamps, no raw JSON.

The `appendLog` and `clearLog` helpers are preserved as no-ops so the rejection / re-run path can still call them without errors. Error-class calls to `appendLog` switch the hero into the error state.

SSE event types:

| Type | Meaning |
|---|---|
| `connected` | Stream opened (no UI change — hero already shows first agent) |
| `task_done` | A task completed — hero advances, completed card slides in |
| `complete` | All agents finished — hero turns green, results fetched |
| `error` | Run failed — hero turns red, alert shown |

### View 4 — Doctor Review (`view-review`)

After the analysis completes, `GET /api/{id}/result` fetches the full `FinalDiagnosis` + `AuditTrail` and renders two tabs:

**Doctor tab** shows:
- Diagnosis strip (primary diagnosis, confidence, severity, re-diagnosis badge)
- Lesion profile table (colour, surface, elevation, border)
- Symptom decomposition table
- Research evidence (query, strength, key findings, PMIDs as clickable links to PubMed)
- Differential diagnosis cards (probability, FOR/AGAINST findings, distinguishing test)
- Treatment plan (medication protocol by line, immediate actions, follow-up)
- Orchestrator clinical reasoning + re-diagnosis explanation
- Doctor feedback history from prior runs

**Patient tab** shows:
- Primary diagnosis + severity badge (large)
- Patient summary in plain language
- What to do recommendations (bulleted)
- When to seek urgent care
- Disclaimer

**Approval bar** (sticky at bottom):
- **Approve** → `POST /api/{id}/approve` → generates PDFs → View 5
- **Reject** → expands feedback panel → doctor types notes → selects re-run scope → `POST /api/{id}/reject` → returns to View 3 for the re-run

### View 5 — Complete (`view-complete`)

Three download cards:
- **Doctor Report** PDF — full clinical assessment + treatment + differential
- **Patient Summary** PDF — plain-language report
- **AI Audit Trail** PDF — every agent's output + approval history

**Start New Session** resets all state and returns to View 1, which prompts for fresh patient details.

---

## Step 4 — How `DermaCrew` Was Modified

Two backward-compatible parameters added to `run()`:

```python
def run(
    self,
    task_callback=None,         # callable(task_output) — fires after each task
    skip_clarification=False,   # True when web app handles clarification externally
) -> tuple[FinalDiagnosis, AuditTrail]:
```

And to `rerun()`:
```python
def rerun(self, feedback: str, scope: str, task_callback=None) -> tuple:
```

The `task_callback` is wired into the `Crew` constructor:
```python
crew_kwargs = dict(agents=..., tasks=..., process=..., verbose=True)
if task_callback is not None:
    crew_kwargs["task_callback"] = task_callback
crew = Crew(**crew_kwargs)
```

**Agent logs still print to the terminal** because `verbose=True` is unchanged. The web UI receives structured events through the callback; the terminal receives the full CrewAI output.

---

## Step 5 — How Clarification Works in the Web Context

`utils/clarification_loop_web.py` provides two functions:

```python
def run_clarification_round_web(patient_text: str) -> tuple[str, list[str]]:
    """Runs decomp + clarification mini-crew. Returns (text, questions)."""

def append_answers_to_text(patient_text: str, questions: list[str], answers: list[str]) -> str:
    """Appends Q&A pairs to patient_text for the next round or main run."""
```

Flow in `app.py`:
1. `POST /api/start` → calls `run_clarification_round_web(symptom_text)` → returns `questions`
2. If questions → client shows them
3. `POST /api/{id}/clarify` → calls `append_answers_to_text` → then `run_clarification_round_web(enriched)` again
4. When `questions = []` → client calls `POST /api/{id}/analyze`
5. `analyze` sets `skip_clarification=True` so `DermaCrew.run()` skips its own pre-pass

---

## Step 6 — How the Doctor Approval Loop Works in the Web Context

The loop is driven by the frontend rather than a Python `while` loop:

```
Frontend:  POST /api/{id}/approve → PDFs generated → View 5
             OR
Frontend:  POST /api/{id}/reject (feedback, scope)
             → Backend: crew.rerun(feedback, scope, task_callback)
             → SSE stream reopened → View 3 replays
             → On complete → View 4 re-rendered with updated result
```

`DOCTOR_FEEDBACK` is set in the environment before `rerun()` calls `run()`, so the Orchestrator task description picks it up automatically (same mechanism as the CLI).

---

## Design System

| Token | Value | Usage |
|---|---|---|
| `--navy` | `#1E3A5F` | Top bar, headings, buttons |
| `--teal` | `#0891B2` | Accent, links, active state |
| `--bg` | `#F0F4F8` | Page background |
| `--surface` | `#FFFFFF` | Cards |
| `--success` | `#16A34A` | Approve, done agents, Mild severity |
| `--warning` | `#D97706` | Moderate severity, monitoring |
| `--danger` | `#DC2626` | Reject, red flags, Severe severity |

Typography: **Inter** (Google Fonts, 300–700). All sizing in `rem`.

Cards use `border-radius: 16px` and a two-level shadow (`var(--shadow)`).

---

## Extending the Web App

### Add authentication
FastAPI supports OAuth2 out of the box. Wrap routes with a `Depends(get_current_user)` dependency.

### Persist sessions across restarts
Replace the in-memory `SESSIONS` dict with a Redis store or SQLite. Serialize `DermaCrew` state or re-hydrate from stored task outputs.

### Stream individual agent thoughts (not just task completion)
Replace `task_callback` with `step_callback` on the Crew. Each agent thought and tool call becomes an SSE event. The log panel is already wired to display any `task_done` or future step events.

### Add patient login / history
Each patient creates an account. Their profile, reports, and diagnosis history are stored in a database. The intake form pre-fills from their profile on login.

---

## Checkpoint ✅

- [ ] `app.py` serves the SPA at `/` and all `/api/*` routes
- [ ] `templates/index.html` has all 5 view divs
- [ ] `static/css/style.css` and `static/js/app.js` are in place
- [ ] `utils/clarification_loop_web.py` exists with `run_clarification_round_web` and `append_answers_to_text`
- [ ] `crew/derma_crew.py` accepts `task_callback` and `skip_clarification` in `run()`
- [ ] Page load always shows View 1 (Patient Intake) — fresh profile every time
- [ ] Submitting the form saves `patient_profile.json` and triggers clarification
- [ ] Analysis view shows live agent progress via SSE
- [ ] Agent logs still print to the terminal console (`verbose=True` unchanged)
- [ ] Doctor tab shows full audit trail; Patient tab shows friendly summary
- [ ] Approve generates PDFs and shows download links
- [ ] Reject with feedback triggers a re-run and returns to the analysis view
- [ ] `uvicorn app:app --reload` starts the server

---

*Next → `17_VOICE_INPUT.md` (voice input adapted for the web UI)*
