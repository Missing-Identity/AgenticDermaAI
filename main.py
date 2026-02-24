# DermaAI v2 — Entry Point
# Run this to start a full diagnostic session.

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def print_header():
    print("\n" + "═"*60)
    print("  DermaAI v2 — Multi-Agent Dermatology Diagnosis")
    print("  Powered by CrewAI + Ollama")
    print("═"*60)


def get_image_path() -> str:
    """Prompt user for image path and validate it."""
    while True:
        path = input("\nEnter the full path to the skin image (or press Enter to skip): ").strip()

        if not path:
            print("No image provided. Proceeding with text-only analysis.")
            return ""

        # Remove surrounding quotes if user added them
        path = path.strip('"').strip("'")

        if not os.path.exists(path):
            print(f"  ❌ File not found: {path}")
            print("  Check the path and try again.")
            continue

        ext = Path(path).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            print(f"  ❌ Unsupported format: {ext}. Use jpg, png, or webp.")
            continue

        print(f"  ✓ Image found: {Path(path).name}")
        return path


def get_patient_text() -> str:
    """Prompt user for symptom description."""
    print("\nDescribe your symptoms in detail (the more you share, the better):")
    print("  - What does the skin look like?")
    print("  - Where is it on your body?")
    print("  - How long have you had it?")
    print("  - What makes it better or worse?")
    print("  - Any other symptoms?")
    print("\n  Note: The AI may ask you 1-2 follow-up questions if it needs more detail.\n")
    text = input("Your description: ").strip()

    if not text:
        print("  ❌ No symptom description provided. Please describe your symptoms.")
        return get_patient_text()

    return text


def display_result(result) -> None:
    """Print the final diagnosis in a readable CLI format."""
    print("\n" + "═"*60)
    print("  DIAGNOSIS COMPLETE")
    print("═"*60)

    print(f"\n🩺 PRIMARY DIAGNOSIS: {result.primary_diagnosis}")
    print(f"   Confidence: {result.confidence.upper()}")
    print(f"   Severity:   {result.severity}")

    if result.re_diagnosis_applied:
        print(f"\n⚠️  RE-DIAGNOSIS APPLIED:")
        print(f"   {result.re_diagnosis_reason}")

    print(f"\n📋 DIFFERENTIAL DIAGNOSES:")
    for i, dx in enumerate(result.differential_diagnoses, 1):
        print(f"   {i}. {dx}")

    print(f"\n🔬 LESION PROFILE:")
    for key, val in result.lesion_profile.items():
        print(f"   {key.capitalize()}: {val}")

    print(f"\n👤 PATIENT SUMMARY:")
    print(f"   {result.patient_summary}")

    print(f"\n📌 WHAT YOU SHOULD DO:")
    for i, rec in enumerate(result.patient_recommendations, 1):
        print(f"   {i}. {rec}")

    print(f"\n🚨 WHEN TO SEEK CARE:")
    print(f"   {result.when_to_seek_care}")

    print(f"\n📚 EVIDENCE BASE:")
    print(f"   {result.literature_support[:300]}...")
    if result.cited_pmids:
        print(f"   PMIDs: {', '.join(result.cited_pmids[:5])}")

    print(f"\n🏥 CLINICAL NOTES (for doctor):")
    print(f"   {result.doctor_notes[:400]}...")

    print(f"\n⚠️  DISCLAIMER:")
    print(f"   {result.disclaimer}")

    print("\n" + "═"*60)
    print("  Reports saved to: reports/")
    print("═"*60 + "\n")


def main():
    print_header()

    # Collect inputs
    image_path = get_image_path()
    patient_text = get_patient_text()

    # Handle text-only mode (no image)
    if not image_path:
        print("\n  Note: Without an image, lesion visual analysis will be skipped.")
        print("  Diagnosis will rely on symptoms and research only.\n")

    confirm = input("Start analysis? This will take several minutes. (y/n): ").strip().lower()
    if confirm != "y":
        print("Analysis cancelled.")
        sys.exit(0)

    # Import here (after validation) to avoid slow import on startup
    from crew.derma_crew import DermaCrew
    from doctor_approval import show_audit_summary_cli, get_doctor_decision
    from pdf_service import save_reports, save_doctor_audit_pdf

    try:
        derma_crew = DermaCrew(image_path=image_path, patient_text=patient_text)

        # ── First run ──────────────────────────────────────────────────────────
        result, audit = derma_crew.run()

        # ── Doctor approval loop ───────────────────────────────────────────────
        approved = False
        while not approved:
            # Generate doctor audit PDF immediately after each run
            try:
                audit_pdf_path = save_doctor_audit_pdf(audit)
                print(f"\nDoctor audit report saved to: {audit_pdf_path}")
                print("   Please open and review this PDF before deciding.\n")
            except Exception as pdf_err:
                print(f"\n[Warning] Could not generate audit PDF: {pdf_err}")

            # Show CLI summary as a quick reference
            show_audit_summary_cli(audit)

            # Get doctor's decision
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

        # ── Post-approval: display and save ───────────────────────────────────
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
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Check that Ollama is running and all models are available.")
        raise


if __name__ == "__main__":
    main()