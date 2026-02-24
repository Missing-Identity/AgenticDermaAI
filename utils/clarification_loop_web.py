# Web-friendly clarification loop.
# Returns questions to the caller instead of blocking on input().
# Used by app.py to drive the Q&A flow through the web UI.
#
# Biodata optimisation: the patient profile is a static JSON file for the
# lifetime of a session. Rather than spinning up a Biodata LLM crew on every
# clarification round, we load the profile in pure Python and embed it inline
# in the Decomposition task description.  The main crew still runs the Biodata
# agent normally (its output is needed in the audit trail and as a CrewAI
# context object for downstream tasks).

from crewai import Crew, Process
from agents.decomposition_agent import create_decomposition_agent, create_decomposition_task
from agents.clarification_agent import create_clarification_agent, create_clarification_task
from agents.decomposition_agent import DecompositionOutput
from agents.clarification_agent import ClarificationOutput
from agents.biodata_agent import load_profile, profile_to_context_string
from utils.schema_adapter import adapt_to_model


def _critical_fields_present(decomp_result) -> bool:
    """
    Python-level check: return True if the two highest-priority clinical fields
    are already populated in the decomposition output.

    This guard prevents the Clarification Agent from re-asking questions that
    were already answered in a previous round, which can happen because MedGemma
    (VISION_LLM) occasionally fails to read structured context correctly.

    body_location and time_days are the fields we always ask about first.
    Once they are present, we let the pipeline proceed rather than risk the
    LLM asking about them again.
    """
    if decomp_result is None:
        return False
    has_location = bool(decomp_result.body_location)
    has_duration = decomp_result.time_days is not None
    return has_location and has_duration


def _get_biodata_text() -> str:
    """Load the saved patient profile and format it as a plain string.
    No LLM call — pure Python file read + formatting."""
    try:
        profile = load_profile()
        return profile_to_context_string(profile)
    except Exception as e:
        print(f"[Clarification-Web] Could not load profile: {e}")
        return ""


def run_clarification_round_web(patient_text: str) -> tuple[str, list[str]]:
    """
    Run one round of Decomposition + Clarification without blocking on user input.

    Strategy (two-stage):
      1. Load patient biodata in pure Python (no LLM). Embed it inline in the
         Decomposition task description — no Biodata crew needed.
      2. Run Decomposition and inspect the output in Python.
         If the two critical fields (body_location, time_days) are already
         populated, skip the Clarification Agent entirely and return no questions.
      3. Only if critical fields are missing, run the Clarification Agent to
         generate targeted follow-up questions.

    Returns:
        (patient_text, questions)
        - patient_text is returned unchanged (caller appends answers externally).
        - questions is a list of strings the AI wants answered.
          An empty list means no further clarification is needed.
    """
    # ── Load biodata in Python — no LLM call ─────────────────────────────────
    biodata_text = _get_biodata_text()

    # ── Stage 1: Decomposition only ──────────────────────────────────────────
    decomp_agent = create_decomposition_agent()
    decomp_task  = create_decomposition_task(
        decomp_agent,
        patient_text,
        biodata_text=biodata_text,   # inline profile string, no task object needed
    )

    try:
        Crew(
            agents=[decomp_agent],
            tasks=[decomp_task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()
    except Exception as e:
        print(f"[Clarification-Web] Decomposition error: {e}. Skipping clarification.")
        return patient_text, []

    decomp_raw = decomp_task.output.raw if decomp_task.output else ""
    decomp_result, _ = adapt_to_model(decomp_raw, DecompositionOutput, "clarification_decomposition")

    # Python-level guard: if both priority fields are present, no need to ask
    if _critical_fields_present(decomp_result):
        print("[Clarification-Web] Critical fields present — skipping clarification agent.")
        return patient_text, []

    # ── Stage 2: Clarification Agent (only if fields are missing) ────────────
    clarif_agent = create_clarification_agent()
    clarif_task  = create_clarification_task(
        clarif_agent,
        decomp_task,
        biodata_task=None,   # no task object; profile context is already in decomp output
    )

    try:
        Crew(
            agents=[clarif_agent],
            tasks=[clarif_task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()
    except Exception as e:
        print(f"[Clarification-Web] Clarification error: {e}. Skipping clarification.")
        return patient_text, []

    clarif_raw = clarif_task.output.raw if clarif_task.output else ""
    clarif_result, _ = adapt_to_model(clarif_raw, ClarificationOutput, "clarification_output")

    if clarif_result is None or not clarif_result.needs_clarification:
        return patient_text, []

    return patient_text, clarif_result.questions or []


def append_answers_to_text(patient_text: str, questions: list[str], answers: list[str]) -> str:
    """
    Merge patient answers back into the enriched text, ready for the next round
    or for the main crew run.
    """
    if not answers:
        return patient_text

    qa_pairs = []
    for q, a in zip(questions, answers):
        if a and a.strip():
            qa_pairs.append(f"Q: {q}\nA: {a.strip()}")

    if not qa_pairs:
        return patient_text

    return patient_text + "\n\nAdditional information:\n" + "\n".join(qa_pairs)
