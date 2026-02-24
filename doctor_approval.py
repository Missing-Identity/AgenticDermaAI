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