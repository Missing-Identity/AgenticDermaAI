# Chapter 13 — Doctor Approval Loop

**Goal:** Add a review stage where a doctor sees the full AI audit trail, approves or rejects the diagnosis, and — if rejected — provides specific feedback that triggers a targeted re-run. The system loops until the doctor approves.

**Time estimate:** 45–60 minutes

---

## Why This Matters Clinically and Legally

An AI system that produces a diagnosis and immediately delivers it to the patient is irresponsible. A doctor must remain in the loop. This chapter implements that loop properly:

- The doctor sees **everything** the AI did — not just the conclusion
- The doctor can push back on any specific part
- The system re-diagnoses accordingly
- The patient report is only generated **after explicit doctor approval**

This also produces an audit trail of the doctor's decisions, which is important for medical liability.

---

## Architecture of the Loop

```
DermaCrew.run() → AuditTrail + FinalDiagnosis
      │
      ▼
Doctor sees: Full audit trail PDF (generated immediately after first run)
      │
      ├─► "APPROVE" ──────────────────────────────────────────────────► Generate patient PDF + TTS
      │
      └─► "REJECT" with typed feedback
                │
                ▼
          Classify feedback:
            ├─ INTERPRETATION (reasoning wrong) → Re-run Orchestrator only
            └─ AGENT OUTPUT WRONG → Re-run from the flagged agent + all downstream
                │
                ▼
          New AuditTrail + FinalDiagnosis
                │
                └─► Back to Doctor review (loop)
```

---

## Step 1 — Define the AuditTrail Data Structure

**`audit_trail.py`** exists in the project root. This is a pure data container — no LLM, no agents.

```python
# audit_trail.py  (complete — no changes needed)
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AuditTrail:
    # Inputs
    patient_text: str = ""
    image_path: str = ""

    # Vision pre-run (raw model outputs before any agent sees them)
    vision_colour_raw: str = ""
    vision_texture_raw: str = ""
    vision_levelling_raw: str = ""
    vision_shape_raw: str = ""

    # Agent outputs (pydantic objects populated after crew.kickoff())
    biodata_summary: str = ""
    colour_output: Optional[object] = None
    texture_output: Optional[object] = None
    levelling_output: Optional[object] = None
    shape_output: Optional[object] = None
    decomposition_output: Optional[object] = None
    research_output: Optional[object] = None
    differential_output: Optional[object] = None
    treatment_output: Optional[object] = None
    final_diagnosis: Optional[object] = None    # populated in derma_crew.py

    # Doctor review history — one entry per rejection round
    feedback_history: list[dict] = field(default_factory=list)
    # Each entry: {"round": int, "feedback": str, "rerun_scope": str}
    #             or {"round": int, "action": "approved", "feedback": ""}

    run_count: int = 1   # increments on each rerun()
```

---

## Step 2 — Update DermaCrew to Populate the AuditTrail

These changes are already applied in **`crew/derma_crew.py`**. Key points for reference:

- `import os` and `from audit_trail import AuditTrail` are at the top.
- `__init__` creates `self.audit = AuditTrail(patient_text=..., image_path=...)`.
- The audit trail collection block runs **after** `crew.kickoff()` using the real task objects. Every field is guarded against `None` (lesion tasks are optional when no image is provided).
- `self.audit.final_diagnosis` is explicitly populated from `orchestrator_task.output.pydantic` — this is the field that `show_audit_summary_cli()` reads to display the orchestrator's final synthesis.
- `run()` returns `(self._result, self.audit)` — a tuple, not a single value.

```python
# Audit trail collection block (inside run(), after crew.kickoff())
if self.image_path:
    self.audit.vision_colour_raw    = vision["colour"]
    self.audit.vision_texture_raw   = vision["texture"]
    self.audit.vision_levelling_raw = vision["levelling"]
    self.audit.vision_shape_raw     = vision["shape"]

self.audit.biodata_summary      = biodata_task.output.raw if biodata_task.output else ""
self.audit.colour_output        = colour_task.output.pydantic if colour_task and colour_task.output else None
self.audit.texture_output       = texture_task.output.pydantic if texture_task and texture_task.output else None
self.audit.levelling_output     = levelling_task.output.pydantic if levelling_task and levelling_task.output else None
self.audit.shape_output         = shape_task.output.pydantic if shape_task and shape_task.output else None
self.audit.decomposition_output = decomp_task.output.pydantic if decomp_task.output else None
self.audit.research_output      = research_task.output.pydantic if research_task.output else None
self.audit.differential_output  = diff_task.output.pydantic if diff_task.output else None
self.audit.treatment_output     = treatment_task.output.pydantic if treatment_task.output else None
self.audit.final_diagnosis      = orchestrator_task.output.pydantic if orchestrator_task.output else None
```

---

## Step 3 — The Approval Loop Module

**`doctor_approval.py`** exists in the project root and is complete. For reference:

```python
# doctor_approval.py
# Manages the doctor review and approval loop.
# The doctor sees the audit trail, approves or provides feedback,
# and the system re-runs until approved.

import os
from audit_trail import AuditTrail


# ── Feedback classification ────────────────────────────────────────────────────

RERUN_OPTIONS = {
    "1": ("Full re-run (everything)", "full"),
    "2": ("Re-run from Differential + Treatment + Orchestrator", "post_research"),
    "3": ("Re-run Orchestrator only (reasoning correction)", "orchestrator_only"),
}


def show_audit_summary_cli(audit: AuditTrail) -> None:
    """
    Print a readable summary of the audit trail to the terminal.
    The doctor reads this (or the full PDF) before deciding.
    """
    sep = "─" * 60

    print(f"\n{'═'*60}")
    print("  DOCTOR REVIEW — AI AUDIT TRAIL")
    print(f"{'═'*60}")

    print(f"\n{sep}")
    print("PATIENT INPUT")
    print(sep)
    print(f"Text: {audit.patient_text[:200]}...")
    print(f"Image: {audit.image_path or 'None provided'}")

    if audit.image_path:
        print(f"\n{sep}")
        print("VISION ANALYSIS (raw model outputs)")
        print(sep)
        print(f"Colour:    {audit.vision_colour_raw[:200]}...")
        print(f"Texture:   {audit.vision_texture_raw[:200]}...")
        print(f"Levelling: {audit.vision_levelling_raw[:200]}...")
        print(f"Shape:     {audit.vision_shape_raw[:200]}...")

    if audit.colour_output:
        print(f"\n{sep}")
        print("LESION AGENT OUTPUTS")
        print(sep)
        print(f"  Colour:    {audit.colour_output.lesion_colour} — {audit.colour_output.reason}")
        print(f"  Surface:   {audit.texture_output.surface} — {audit.texture_output.reason}")
        print(f"  Levelling: {audit.levelling_output.levelling} — {audit.levelling_output.reason}")
        print(f"  Border:    {getattr(audit.shape_output, 'shape_border', 'N/A')} — {audit.shape_output.reason}")

    if audit.decomposition_output:
        d = audit.decomposition_output
        print(f"\n{sep}")
        print("SYMPTOM DECOMPOSITION")
        print(sep)
        print(f"  Symptoms: {', '.join(d.symptoms[:5])}")
        print(f"  Duration: {d.time_days} days | Onset: {d.onset} | Progression: {d.progression}")
        print(f"  Location: {d.body_location} | Occupation: {d.occupational_exposure}")

    if audit.research_output:
        r = audit.research_output
        print(f"\n{sep}")
        print("RESEARCH FINDINGS")
        print(sep)
        print(f"  Query: {r.primary_search_query}")
        print(f"  Evidence strength: {r.evidence_strength}")
        print(f"  Key findings:")
        for f in r.key_findings[:3]:
            print(f"    • {f}")
        print(f"  PMIDs: {', '.join(r.cited_pmids[:5])}")

    if audit.differential_output:
        diff = audit.differential_output
        print(f"\n{sep}")
        print("DIFFERENTIAL DIAGNOSIS")
        print(sep)
        print(f"  Primary: {diff.primary_diagnosis} ({diff.confidence_in_primary} confidence)")
        print(f"  Differentials:")
        for entry in diff.differentials:
            print(f"    [{entry.probability.upper()}] {entry.condition}")
        if diff.red_flags:
            print(f"  ⚠️  Red flags: {', '.join(diff.red_flags)}")

    if audit.treatment_output:
        t = audit.treatment_output
        print(f"\n{sep}")
        print("TREATMENT PLAN")
        print(sep)
        print(f"  For: {t.for_diagnosis}")
        print(f"  First-line:")
        for m in t.medications:
            if m.line == "first":
                print(f"    {m.treatment_name} — {m.dose_or_protocol} for {m.duration}")
        print(f"  Follow-up: {t.follow_up}")

    if audit.final_diagnosis:
        fd = audit.final_diagnosis
        print(f"\n{sep}")
        print("ORCHESTRATOR FINAL SYNTHESIS")
        print(sep)
        print(f"  Diagnosis: {fd.primary_diagnosis}")
        print(f"  Confidence: {fd.confidence} | Severity: {fd.severity}")
        print(f"  Re-diagnosis applied: {fd.re_diagnosis_applied}")
        if fd.re_diagnosis_applied:
            print(f"  Reason: {fd.re_diagnosis_reason}")
        print(f"  Clinical reasoning (excerpt): {fd.clinical_reasoning[:400]}...")

    print(f"\n{'═'*60}")
    print(f"  Run #{audit.run_count}")
    if audit.feedback_history:
        print(f"  Previous feedback rounds: {len(audit.feedback_history)}")
    print(f"{'═'*60}")


def get_doctor_decision() -> tuple[str, str, str]:
    """
    Prompt the doctor to approve or reject.
    Returns: (decision, feedback_text, rerun_scope)
    decision: "approve" or "reject"
    feedback_text: doctor's notes (empty if approve)
    rerun_scope: "full" | "post_research" | "orchestrator_only" (empty if approve)
    """
    print("\n" + "─"*60)
    print("DOCTOR REVIEW REQUIRED")
    print("─"*60)
    print("\nPlease review the full audit trail PDF in the reports/ folder.")
    print("Then enter your decision:\n")
    print("  [A] APPROVE — Diagnosis is clinically sound")
    print("  [R] REJECT  — Changes required")

    while True:
        choice = input("\nYour decision (A/R): ").strip().upper()
        if choice == "A":
            return "approve", "", ""
        elif choice == "R":
            break
        print("  Please enter A or R.")

    # Collect feedback
    print("\n" + "─"*60)
    print("REJECTION FEEDBACK")
    print("─"*60)
    print("Describe what is wrong and what should be changed:")
    print("(Be specific — e.g., 'The colour assessment is wrong, the lesion is")
    print(" hyperpigmented not erythematous. The differential should include melanoma.')\n")
    feedback = input("Your feedback: ").strip()
    if not feedback:
        feedback = "No specific feedback provided."

    # Ask re-run scope
    print("\n" + "─"*60)
    print("RE-RUN SCOPE")
    print("─"*60)
    print("How much of the pipeline needs to re-run?\n")
    for key, (label, _) in RERUN_OPTIONS.items():
        print(f"  [{key}] {label}")

    while True:
        scope_choice = input("\nRe-run scope (1/2/3): ").strip()
        if scope_choice in RERUN_OPTIONS:
            _, rerun_scope = RERUN_OPTIONS[scope_choice]
            break
        print("  Please enter 1, 2, or 3.")

    return "reject", feedback, rerun_scope
```

---

## Step 4 — Scoped Re-Runs in DermaCrew

These changes are already applied. `DermaCrew.rerun()` is implemented in **`crew/derma_crew.py`** and the feedback injection is live in **`agents/orchestrator_agent.py`**.

### `DermaCrew.rerun()` (in `crew/derma_crew.py`)

```python
def rerun(self, feedback: str, scope: str) -> tuple:
    """
    Re-run the pipeline with doctor feedback injected.

    scope:
        "full"              — Re-run every agent from scratch.
        "post_research"     — Re-run Differential + Treatment + Orchestrator.
        "orchestrator_only" — Re-run only the Orchestrator synthesis.

    All three scopes call run() with DOCTOR_FEEDBACK set in the environment.
    The Orchestrator task description reads the env var and prepends the feedback
    block, so it explicitly addresses the doctor's concern in its synthesis.
    """
    self.audit.run_count += 1
    self.audit.feedback_history.append({
        "round": self.audit.run_count,
        "feedback": feedback,
        "rerun_scope": scope,
    })

    print(f"\n[Re-run #{self.audit.run_count}] Scope: {scope}")
    print(f"Doctor feedback: {feedback}\n")

    os.environ["DOCTOR_FEEDBACK"] = feedback

    result, audit = self.run()

    # Clear after run so feedback does not leak into future independent runs
    os.environ.pop("DOCTOR_FEEDBACK", None)

    return result, audit
```

> **Note on partial re-runs:** True partial crew execution (re-running only from agent N onward) requires preserving upstream task objects. This is achievable but adds significant complexity. The env-var approach is the recommended first implementation — the Orchestrator picks up the feedback and revises its synthesis accordingly.

### Doctor feedback injection (in `agents/orchestrator_agent.py`)

```python
import os

def create_orchestrator_task(...) -> Task:
    doctor_feedback = os.getenv("DOCTOR_FEEDBACK", "").strip()
    feedback_block = ""
    if doctor_feedback:
        feedback_block = (
            f"DOCTOR FEEDBACK FROM PREVIOUS RUN:\n"
            f"   \"{doctor_feedback}\"\n\n"
            f"You MUST address this feedback in your synthesis. "
            f"If the doctor identified a specific error, correct it explicitly "
            f"and explain what you changed in re_diagnosis_reason. "
            f"Set re_diagnosis_applied = true.\n\n"
        )

    return Task(
        description=(
            feedback_block +    # injected at the very top when set
            "You have received the complete clinical picture from all specialist agents:\n"
            ...
        ),
        ...
    )
```

---

## Step 5 — Wire the Approval Loop into main.py

These changes are already applied in **`main.py`**. The full `main()` approval loop for reference:

```python
from crew.derma_crew import DermaCrew
from doctor_approval import show_audit_summary_cli, get_doctor_decision
from pdf_service import save_reports, save_doctor_audit_pdf

try:
    derma_crew = DermaCrew(image_path=image_path, patient_text=patient_text)

    # ── First run ──────────────────────────────────────────────────────────────
    result, audit = derma_crew.run()   # run() returns a tuple — not a single value

    # ── Doctor approval loop ───────────────────────────────────────────────────
    approved = False
    while not approved:
        # Generate doctor audit PDF after each run (before approval)
        try:
            audit_pdf_path = save_doctor_audit_pdf(audit)
            print(f"\nDoctor audit report saved to: {audit_pdf_path}")
        except Exception as pdf_err:
            print(f"\n[Warning] Could not generate audit PDF: {pdf_err}")

        show_audit_summary_cli(audit)
        decision, feedback, rerun_scope = get_doctor_decision()

        if decision == "approve":
            approved = True
            print("\nDoctor approved the diagnosis.")
            audit.feedback_history.append({
                "round": audit.run_count,
                "action": "approved",
                "feedback": "",
            })
        else:
            print(f"\nRe-running with scope: {rerun_scope}...")
            result, audit = derma_crew.rerun(feedback, rerun_scope)

    # ── Post-approval: display and save ───────────────────────────────────────
    display_result(result)

    os.makedirs("reports", exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/diagnosis_{timestamp}.json"
    with open(output_file, "w") as f:
        f.write(result.model_dump_json(indent=2))
    print(f"\nFull JSON saved to: {output_file}")

    patient_name_input = input("\nEnter patient name for PDF reports (or Enter for 'Patient'): ").strip()
    pdf_patient_name = patient_name_input or "Patient"

    print("Generating PDFs...")
    try:
        doctor_pdf, patient_pdf = save_reports(result, audit, patient_name=pdf_patient_name)
        print(f"Doctor report:   {doctor_pdf}")
        print(f"Patient summary: {patient_pdf}")
    except Exception as pdf_err:
        print(f"PDF generation failed: {pdf_err}")
        print("   JSON result is still saved.")

except KeyboardInterrupt:
    print("\n\nAnalysis interrupted.")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    raise
```

> **Critical:** `derma_crew.run()` returns `(result, audit)` — a tuple. Unpacking it as `result, audit = derma_crew.run()` is mandatory. The old pattern `result = derma_crew.run()` will cause an `AttributeError` when `display_result(result)` tries to access `.primary_diagnosis` on the tuple.

---

## Step 6 — Testing the Approval Loop

Run the full pipeline:
```powershell
python main.py
```

**Test scenario 1 — Approve first time:**
1. Enter image path + symptom text
2. Wait for analysis to complete
3. At the approval prompt, enter `A`
4. Verify patient PDF is generated

**Test scenario 2 — One rejection then approval:**
1. At the approval prompt, enter `R`
2. Type feedback: `"The surface texture assessment should be scaly not smooth. The agent may have misread the image."`
3. Enter scope `2` (post-research re-run)
4. Wait for re-analysis
5. Review the new audit trail — does the orchestrator address the feedback?
6. Enter `A` to approve
7. Verify the feedback appears in `audit.feedback_history`

**What to verify:**
- Doctor audit PDF is generated after EVERY run (before approval)
- Re-run scope options work (all three produce a new diagnosis)
- Doctor feedback appears verbatim in the orchestrator's `re_diagnosis_reason` field after rejection
- `audit.feedback_history` contains the complete history of all rounds
- Patient PDF is only generated AFTER approval

---

## Checkpoint ✅

- [ ] `audit_trail.py` exists with `AuditTrail` dataclass covering all 9 agent outputs + inputs + feedback history
- [ ] `DermaCrew.__init__` creates `self.audit = AuditTrail(...)` on construction
- [ ] `DermaCrew.run()` returns `(FinalDiagnosis, AuditTrail)` — `self.audit.final_diagnosis` is explicitly populated
- [ ] Audit trail collection runs **after** `crew.kickoff()`, uses correct task objects, guards lesion tasks with `if colour_task and colour_task.output`
- [ ] `DermaCrew.rerun()` exists, increments `run_count`, appends to `feedback_history`, sets `os.environ["DOCTOR_FEEDBACK"]`, calls `run()`, clears env var after
- [ ] `doctor_approval.py` has `show_audit_summary_cli()` and `get_doctor_decision()` — no changes needed
- [ ] `agents/orchestrator_agent.py` imports `os` and reads `DOCTOR_FEEDBACK` at the top of `create_orchestrator_task()`; feedback block prepended to task description when set
- [ ] `main.py` unpacks `result, audit = derma_crew.run()` (tuple) and runs the approval loop
- [ ] Patient PDF is only generated **after** doctor approval
- [ ] Audit PDF (`save_doctor_audit_pdf`) is generated after every run, inside a `try/except` so a PDF failure never blocks the review loop
- [ ] Test: rejecting once then approving produces a new diagnosis that addresses the feedback
- [ ] `audit.feedback_history` records every rejection round plus a final `"action": "approved"` entry

### Bugs Fixed During Implementation

| Bug | Location | Fix |
|---|---|---|
| `result = derma_crew.run()` — treating tuple as single value | `main.py` | Changed to `result, audit = derma_crew.run()` |
| `self.audit.final_diagnosis` never populated | `crew/derma_crew.py` | Added `self.audit.final_diagnosis = orchestrator_task.output.pydantic if orchestrator_task.output else None` |
| Audit trail block ran before `vision` was defined (NameError) | `crew/derma_crew.py` | Moved collection block to after `crew.kickoff()` (Chapter 12 fix) |
| `os` not imported | `crew/derma_crew.py`, `agents/orchestrator_agent.py` | Added `import os` to both files |

---

*Next → `14_VOICE_INPUT.md`*
