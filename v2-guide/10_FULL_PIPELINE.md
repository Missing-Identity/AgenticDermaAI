# Chapter 10 — The Full Pipeline (main.py + End-to-End Testing)

**Goal:** Wire everything into `main.py`, run the complete 8-agent pipeline end-to-end with a real image and patient text, understand the output, and diagnose any issues that arise.

**Time estimate:** 45–90 minutes (most time is spent waiting for agents to run)

---

## Before You Start — Pre-Run Checklist

```powershell
# 1. Venv is active
.\.venv\Scripts\Activate.ps1

# 2. Ollama is running (in another terminal)
ollama serve

# 3. All three models are available
ollama list

# 4. Patient profile exists
python patient_setup.py   # run this if you don't have patient_profile.json

# 5. You have a test skin image ready
#    Download a public domain dermatology photo or use one of your own
#    Note its full path (e.g., C:\Users\...\test_rash.jpg)
```

---

## Step 1 — Write main.py

Open `main.py` and replace the placeholder with the real entry point:

```python
# main.py
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

    try:
        derma_crew = DermaCrew(
            image_path=image_path,
            patient_text=patient_text,
        )
        result = derma_crew.run()
        display_result(result)

        # Save raw result to file
        os.makedirs("reports", exist_ok=True)
        import json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"reports/diagnosis_{timestamp}.json"
        with open(output_file, "w") as f:
            f.write(result.model_dump_json(indent=2))
        print(f"Full diagnosis saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Check that Ollama is running and all models are available.")
        raise


if __name__ == "__main__":
    main()
```

---

## Step 2 — Handle Text-Only Mode in DermaCrew

Update `crew/derma_crew.py` to gracefully handle the case where no image is provided. Open the file and modify the `run()` method.

Find the vision pre-run block and the lesion task creation block, and replace them with this guarded version:

```python
# Lesion agents and tasks only run when an image is provided
if self.image_path:
    vision = self._run_vision_analysis()

    colour_task     = create_colour_task(colour_agent, self.image_path, biodata_task, vision_result=vision["colour"])
    texture_task    = create_texture_task(texture_agent, self.image_path, biodata_task, vision_result=vision["texture"])
    levelling_task  = create_levelling_task(levelling_agent, self.image_path, biodata_task, vision_result=vision["levelling"])
    shape_task      = create_shape_task(shape_agent, self.image_path, biodata_task, vision_result=vision["shape"])
    lesion_agents   = [colour_agent, texture_agent, levelling_agent, shape_agent]
    lesion_tasks    = [colour_task, texture_task, levelling_task, shape_task]
else:
    colour_task = texture_task = levelling_task = shape_task = None
    lesion_agents = []
    lesion_tasks  = []
```

And in the `Crew()` constructor:
```python
crew = Crew(
    agents=[biodata_agent] + lesion_agents + [decomp_agent, research_agent, orchestrator],
    tasks=[biodata_task] + lesion_tasks + [decomp_task, research_task, orchestrator_task],
    process=Process.sequential,
    verbose=True,
)
```

Also update the research task call further down to pass the (possibly None) lesion task handles:
```python
research_task = create_research_task(
    research_agent,
    biodata_task=biodata_task,
    colour_task=colour_task,       # None if no image was provided
    texture_task=texture_task,
    levelling_task=levelling_task,
    shape_task=shape_task,
    decomposition_task=decomp_task,
)
```

The task factories already handle `None` context arguments — `[t for t in [...] if t is not None]`.

---

## Step 3 — Run the Full Pipeline

```powershell
python main.py
```

You'll be prompted for:
1. Image path — paste the full path to your test skin image
2. Symptom description — type something realistic
3. Confirmation to start

**Expected runtime:** 8–20 minutes for the full 8-agent pipeline on your machine.

**Watch the verbose output** as agents run. You should see:
1. Biodata agent summarising the patient profile
2. Four lesion agents each analysing the image with their specific focus
3. Decomposition agent extracting structured symptoms
4. Research agent making PubMed calls (you'll see search queries)
5. Orchestrator synthesising everything

---

## Step 4 — Read and Evaluate the Output

After completion, the diagnosis is printed to terminal and saved to `reports/diagnosis_TIMESTAMP.json`.

Open the JSON file and check each section:

**Check `primary_diagnosis`:**
- Is it clinically reasonable for the image + text you provided?
- Does it match what you'd expect?

**Check `confidence`:**
- Is it calibrated? (Should be "low" if you gave vague input, "high" if specific)

**Check `re_diagnosis_applied`:**
- If True: read `re_diagnosis_reason`. Does the revision make sense?
- If the research clearly supported a different condition, this should be True

**Check `patient_summary`:**
- Read it out loud. Could a non-doctor understand it?
- Is it empathetic?

**Check `doctor_notes`:**
- Does it mention suggested investigations?
- Does it reference the specific lesion findings?

**Check `cited_pmids`:**
- Open 2-3 of them on PubMed and confirm they're real and relevant

---

## Step 4b — What Changed Since Chapter 09

By the time you reach the full pipeline, DermaCrew now runs **10 agents** (not 8). The updated order is:

```
[Pre-pass — runs BEFORE the main crew]
  0a. Biodata (silent, for clarification context)
  0b. Decomposition (silent, for gap analysis)
  0c. Clarification Agent → asks patient follow-up questions if needed   ← Chapter 06b

[Main crew — sequential]
  1.  Biodata
  2–5. Colour, Texture, Levelling, Shape (lesion agents — vision pre-run)
  6.  Decomposition  ← now uses enriched patient_text if clarification happened
  7.  Research
  8.  Differential Diagnosis   ← Chapter 12
  9.  Treatment Plan           ← Chapter 12
  10. Orchestrator
```

After the crew finishes, `DermaCrew.run()` returns `(FinalDiagnosis, AuditTrail)` — two values, not one.

The main.py flow is also different from what was shown in Step 1:
- A **clarification pre-pass** runs before the main crew, asking follow-up questions if needed (Chapter 06b)
- A **doctor approval loop** runs between `DermaCrew.run()` and patient PDF generation (Chapter 13)
- The **doctor audit PDF** is generated after every run, before approval (Chapter 11)
- The patient PDF and doctor report are generated only after approval (Chapter 11)
- Voice input is offered at the symptom-description stage (Chapter 14)
- Voice output is offered after approval (Chapter 15)

See the updated `main.py` code in **Chapter 13, Step 5** for the current full implementation.

---

## Step 5 — Common Issues and How to Debug Them

### Issue: Output is None / RuntimeError from DermaCrew
**Cause:** Either (a) the orchestrator itself produced malformed JSON, or (b) an upstream task (most commonly `TreatmentPlanOutput`) failed validation and halted the crew before the orchestrator ever ran.  
**How to diagnose:** Look at the `[Warning]` lines printed above the `RuntimeError`. They show which task's `ValidationError` caused the crew to halt, along with the raw output. The `RuntimeError` message also lists any upstream tasks that had no output (`Upstream tasks with no output: Treatment, ...`).  
**Auto-recovery:** `DermaCrew.run()` detects when `orchestrator_task.output is None` and automatically launches a recovery mini-crew — one agent, one task — passing only the tasks that successfully produced output as context (the failed task is omitted). You will see `[Recovery] Orchestrator did not run — starting isolated recovery pass...` in the console, followed by `[Recovery] Orchestrator recovery run succeeded.`  
**If recovery also fails:** Extend the relevant schema's `ResilientBase.model_validate_json` or add a field validator to cover the new LLM output pattern. The raw output of the failing task is printed in the `[Warning]` lines.

### Issue: Orchestrator fails with ValidationError for `severity` or `confidence`
**Cause:** LLM returned `"mild to moderate"` for `severity`, or `"High"` (capitalised) for `confidence`.  
**Status: Auto-handled.** `FinalDiagnosis` has `@field_validator` normalisers for both fields. `severity` maps any string containing `"severe"/"moderate"/"mild"` to the correct capitalised Literal. `confidence` lowercases whatever the LLM returns.  
**If it still fails:** The LLM returned a completely unrecognisable value — extend the normaliser in `agents/orchestrator_agent.py`.

### Issue: Lesion agent returns wrong Literal value
**Cause:** Model returned a semantically correct but lexically non-matching value (e.g. `"elevated"` instead of `"raised"`, `"fine scaling"` instead of `"scaly"`).  
**Status: Auto-handled.** Each `Literal` field in `SurfaceOutput`, `LevellingOutput`, and `ShapeBorderOutput` has a `@field_validator(mode="before")` that fuzzy-maps common synonyms to the correct value. You should not see `ValidationError` for these fields under normal operation.  
**If it still fails:** Look at `levelling_task.output.raw` to see what the model returned. If it is completely unrecognisable, extend the normaliser in `agents/lesion_agents.py` to cover the new variant.

### Issue: Research agent doesn't call PubMed tool
**Cause:** The agent decided it doesn't need the tool.  
**Fix:** Add `"Your FIRST action must be to call the pubmed_search tool."` to the task description.

### Issue: Research agent loops with `tool_code` blocks and never finishes
**Cause:** `create_research_agent()` is using `TEXT_LLM` (MedGemma) instead of `ORCHESTRATOR_LLM`. MedGemma doesn't support CrewAI's tool-call format.  
**Fix:** Confirm `create_research_agent()` in `agents/research_agent.py` uses `llm=ORCHESTRATOR_LLM`. See Chapter 08 for details.

### Issue: Clarification pre-pass crashes with ValidationError or "trailing characters"
**Cause:** The LLM returned malformed JSON (e.g. text after the closing `}`) or `null` for a list field.  
**Status: Auto-handled.** `mini_crew.kickoff()` is wrapped in a `try/except` in `utils/clarification_loop.py`. If the clarification output cannot be parsed, the loop breaks and the pipeline proceeds with whatever data was already extracted. You will see `[Clarification] Output parsing error: ...` in the console.  
**If it triggers repeatedly:** Check which LLM is powering the clarification agent (`VISION_LLM`). Switching to `ORCHESTRATOR_LLM` (qwen2.5:7b) generally produces more reliable JSON output.

### Issue: Clarification agent always triggers questions even for complete input
**Cause:** Overly conservative gap detection, or model hallucinating missing fields.  
**Debug:** Print `clarif_task.output.pydantic.reasoning` to see why it triggered.  
**Fix:** Make the task description more explicit: add "If all four critical fields are present (body_location, time_days, onset, progression), you MUST set needs_clarification = false regardless of other gaps."

### Issue: Very long runtime (>30 minutes)
**Cause:** Model loading/swap time on 4050 VRAM.  
**Fix:** Run `ollama ps` in another terminal to see which model is loaded. Consider reducing `MAX_FREE_TURNS` to 0 if testing orchestration logic only.

### Issue: Hallucinated PMIDs
**Cause:** Research agent generating citations from memory instead of tool output.  
**Fix:** Add to research task: "ONLY include PMIDs that appeared in the pubmed_search tool output. Cross-check every PMID you cite against the tool results."

---

## Step 6 — What a Great Output Looks Like

Here's an example of what well-functioning pipeline output looks like for "itchy red bumpy rash, forearm, painter":

```
🩺 PRIMARY DIAGNOSIS: Allergic Contact Dermatitis
   Confidence: HIGH
   Severity:   Moderate

📋 DIFFERENTIAL DIAGNOSES:
   1. Irritant Contact Dermatitis
   2. Atopic Dermatitis
   3. Nummular Eczema

🔬 LESION PROFILE:
   Colour: Erythematous with faint violaceous tinge on medium skin tone
   Texture: Papular with fine scaling
   Levelling: Raised
   Border: Irregular, not well-defined

👤 PATIENT SUMMARY:
   You appear to have an allergic skin reaction, most likely caused by contact
   with a chemical in your workplace. Your skin is red, bumpy, and itchy because
   your immune system is reacting to something it touched. The good news is this
   type of rash usually improves once you identify and avoid the trigger.

📌 WHAT YOU SHOULD DO:
   1. Avoid further contact with the new solvent brand until seen by a doctor
   2. Apply a mild corticosteroid cream (e.g., hydrocortisone 1%) twice daily
   3. Keep the area moisturised with a fragrance-free cream
   4. See a dermatologist within 1 week if no improvement

🚨 WHEN TO SEEK CARE:
   Go to A&E immediately if the rash spreads to your face, you have difficulty
   breathing, or swelling develops. See a doctor within 2-3 days if the rash
   spreads significantly or you develop fever.

📚 EVIDENCE BASE:
   Literature strongly supports allergic contact dermatitis in occupational
   painter settings (PMID: 38421234, 37892341). A 2022 systematic review found
   solvent exposure the leading cause of occupational ACD in construction workers...
```

---

## Checkpoint ✅

- [ ] `main.py` is complete with input collection and result display
- [ ] `DermaCrew` handles the no-image case gracefully
- [ ] Full pipeline runs end-to-end without crashing
- [ ] `reports/diagnosis_TIMESTAMP.json` is created after each run
- [ ] Primary diagnosis is clinically reasonable
- [ ] Patient summary is jargon-free
- [ ] PMIDs in output are verifiable on PubMed
- [ ] You understand how to debug the most common failure modes

---

*Next → `11_PDF_REPORTS.md` (required — generates the audit trail and patient/doctor PDFs)*
