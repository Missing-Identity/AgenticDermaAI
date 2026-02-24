# Chapter 09 — The CMO, Medical Scribe, and Full Crew Assembly

**Goal:** Build the Chief Medical Officer (CMO) Agent that synthesises all upstream outputs and handles re-diagnosis, and the Medical Scribe Agent that produces the final structured diagnosis and patient/doctor reports. Then assemble the `DermaCrew` class that wires everything together and is the main entry point for the pipeline.

**Time estimate:** 60–75 minutes

---

## What the CMO and Scribe Do

The CMO and Scribe are the last agents to run. They receive:
- Patient biodata summary
- 4 lesion analysis outputs (colour, texture, levelling, border)
- Decomposed symptoms and history
- Research summary with cited evidence and any contradictions
- Differential Diagnosis and Mimic Resolution
- Treatment Plan

The **CMO** then:
1. **Synthesises** all clinical findings into a coherent clinical picture
2. **Checks consistency** — do the lesion features match the research-supported diagnosis?
3. **Re-diagnoses if needed** — if research flags a contradiction, it revises
4. **Produces a clinical reasoning decision** in pure JSON.

The **Scribe** then:
1. Takes the CMO's decision and the Treatment Plan.
2. **Produces two reports** — a detailed clinical report for doctors and a plain-language summary for patients.

---

## Step 1 — Design the Final Diagnosis Schemas

**`agents/orchestrator_agent.py`**

```python
# agents/orchestrator_agent.py
# The Orchestrator module: synthesises all agent outputs into a final clinical assessment.
# Split into CMO (Reasoning) and Scribe (Reporting)

import os
from typing import Any, Literal
from pydantic import Field, field_validator, model_validator
from crewai import Agent, Task
from config import VISION_LLM, ORCHESTRATOR_LLM
from utils.resilient_base import ResilientBase

# ── 1. CMO Schema (Pure Clinical Reasoning) ───────────────────────────────────

class CMOResult(ResilientBase):
    """
    The structured clinical decision output from the Chief Medical Officer agent.
    This contains no patient-facing text or formatting, only medical logic.
    """

    primary_diagnosis: str = Field(
        default="Unknown",
        description="The final confirmed diagnosis after all conflict resolution"
    )
    confidence: Literal["high", "moderate", "low"] = Field(
        default="moderate",
        description="Confidence in the final diagnosis"
    )
    severity: Literal["Mild", "Moderate", "Severe"] = Field(
        default="Moderate",
        description="Clinical severity of the condition"
    )
    lesion_profile_summary: dict = Field(
        default_factory=dict,
        description="Summary of key visual features that support the diagnosis"
    )
    clinical_reasoning: str = Field(
        default="",
        description="Step-by-step clinical reasoning explaining the final decision"
    )
    re_diagnosis_applied: bool = Field(
        default=False,
        description="True if the CMO revised the diagnosis away from the initial visual/differential assessment"
    )
    re_diagnosis_reason: str = Field(
        default="",
        description="If re_diagnosis_applied is True, explain what was revised and why"
    )
    suggested_investigations: list[str] = Field(
        default=[],
        description="Recommended diagnostic tests or referrals (biopsy, KOH prep, etc.)"
    )
    cited_pmids: list[str] = Field(
        default=[],
        description="PMIDs of the most relevant cited articles"
    )

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_severity(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        lower = v.lower()
        if "severe" in lower: return "Severe"
        if "moderate" in lower: return "Moderate"
        if "mild" in lower: return "Mild"
        return v.capitalize()

    @field_validator("confidence", mode="before")
    @classmethod
    def normalise_confidence(cls, v: str) -> str:
        if isinstance(v, str): return v.lower()
        return v

    @field_validator("suggested_investigations", "cited_pmids", mode="before")
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if v is not None else []

    @field_validator("primary_diagnosis", "clinical_reasoning", "re_diagnosis_reason", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("lesion_profile_summary", mode="before")
    @classmethod
    def coerce_null_dict(cls, v):
        return v if v is not None else {}

    @field_validator("re_diagnosis_applied", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str): return v.strip().lower() == "true"
        return v

# ── 2. Final Output Schema (Scribe) ───────────────────────────────────────────

class FinalDiagnosis(ResilientBase):
    """
    The complete, structured clinical diagnosis produced by the Medical Scribe.
    It combines the CMO's logic with the Treatment Plan into readable reports.
    """
    
    # Pulled directly from CMO (duplicated for API compatibility)
    primary_diagnosis: str = Field(default="Unknown")
    confidence: Literal["high", "moderate", "low"] = Field(default="moderate")
    severity: Literal["Mild", "Moderate", "Severe"] = Field(default="Moderate")
    lesion_profile: dict = Field(default_factory=dict)
    clinical_reasoning: str = Field(default="")
    suggested_investigations: list[str] = Field(default=[])
    cited_pmids: list[str] = Field(default=[])

    # Scribe Generated Content
    patient_summary: str = Field(
        default="",
        description="Plain English explanation of the diagnosis for the patient. No jargon. Empathetic tone."
    )
    patient_recommendations: list[str] = Field(
        default=[],
        description="Actionable steps for the patient based on the treatment plan."
    )
    doctor_notes: str = Field(
        default="",
        description="Technical clinical notes for the physician integrating CMO logic and Treatment Plan."
    )
    treatment_suggestions: list[str] = Field(
        default=[],
        description="Evidence-based treatment options, from first-line to escalation"
    )
    literature_support: str = Field(
        default="",
        description="Summary of PubMed evidence supporting this diagnosis."
    )
    when_to_seek_care: str = Field(
        default="",
        description="Specific, clear advice on when to go to a doctor or emergency room"
    )
    disclaimer: str = Field(
        default=(
            "This assessment is generated by an AI system and is intended for informational "
            "purposes only. It does not constitute a medical diagnosis. Please consult a "
            "qualified dermatologist or healthcare provider for proper evaluation and treatment."
        )
    )

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_severity(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        lower = v.lower()
        if "severe" in lower: return "Severe"
        if "moderate" in lower: return "Moderate"
        if "mild" in lower: return "Mild"
        return v.capitalize()

    @field_validator("confidence", mode="before")
    @classmethod
    def normalise_confidence(cls, v: str) -> str:
        if isinstance(v, str): return v.lower()
        return v

    @field_validator("patient_recommendations", "suggested_investigations", "treatment_suggestions", "cited_pmids", mode="before")
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if v is not None else []

    @field_validator("primary_diagnosis", "clinical_reasoning", "patient_summary", "doctor_notes", "literature_support", "when_to_seek_care", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("lesion_profile", mode="before")
    @classmethod
    def coerce_null_dict(cls, v):
        return v if v is not None else {}
```

> **Resilience layers — four defences against LLM output failures:**
>
> 1. **`ResilientBase.model_validate_json`** (from `utils/resilient_base.py`) — Before Pydantic's JSON parser even starts, the raw LLM string is scanned with `re.search(r'\{.*\}', ...)` to extract only the outermost `{ … }` block. Any self-check narration or backtick characters that the LLM emits adjacent to the brace are silently dropped, preventing `Invalid JSON: key must be a string` and `trailing characters` errors.
> 2. **Literal normalisers and null-safety** on every field — `severity` and `confidence` are fuzzy-matched to their Literal values. All list, string, dict, and bool fields have `default` values and `coerce_null_*` validators so the model always constructs even if fields are missing or `null`.

---

## Step 2 — Build the CMO Agent and Task

```python
# ── 3. CMO Agent ──────────────────────────────────────────────────────────────

def create_cmo_agent() -> Agent:
    return Agent(
        role="Chief Medical Officer (Dermatology)",
        goal=(
            "Review all evidence (visual, text, differentials, mimic resolution, research) "
            "and make the final, authoritative clinical decision on the diagnosis."
        ),
        backstory=(
            "You are the Chief Medical Officer. You receive raw data from specialist agents. "
            "Your job is strictly clinical reasoning. You look at the Mimic Resolution "
            "and Differential, cross-check them against the PubMed research and visual findings, "
            "and declare the final diagnosis. You do not write patient letters; you only output "
            "strict medical logic and validation."
        ),
        llm=VISION_LLM,  # MedGemma for deep clinical reasoning
        verbose=True,
    )

def create_cmo_task(
    agent: Agent,
    biodata_task=None,
    colour_task=None,
    texture_task=None,
    levelling_task=None,
    shape_task=None,
    decomposition_task=None,
    research_task=None,
    differential_task=None,
    mimic_task=None,
) -> Task:
    context = [
        t for t in [
            biodata_task, colour_task, texture_task, levelling_task, shape_task,
            decomposition_task, research_task, differential_task, mimic_task
        ] if t is not None
    ]

    doctor_feedback = os.getenv("DOCTOR_FEEDBACK", "").strip()
    feedback_block = ""
    if doctor_feedback:
        feedback_block = (
            f"DOCTOR FEEDBACK FROM PREVIOUS RUN:\n"
            f"   \"{doctor_feedback}\"\n\n"
            f"You MUST address this feedback. Correct the error explicitly, "
            f"set re_diagnosis_applied = true, and explain in re_diagnosis_reason.\n\n"
        )

    return Task(
        description=(
            feedback_block +
            "Review the complete clinical picture from all specialist agents. "
            "Pay special attention to the Mimic Resolution and Differential tasks.\n\n"
            "STEP 1: Validate the primary diagnosis. Does it align with visuals, history, and literature?\n"
            "STEP 2: Decide if a re-diagnosis is necessary based on research or contradictions.\n"
            "STEP 3: Output the final clinical decision."
        ),
        expected_output=(
            "ONLY a valid JSON object matching the CMOResult schema. "
            "Keys: primary_diagnosis, confidence, severity, lesion_profile_summary, "
            "clinical_reasoning, re_diagnosis_applied, re_diagnosis_reason, "
            "suggested_investigations, cited_pmids."
        ),
        agent=agent,
        output_pydantic=CMOResult,
        context=context,
    )
```

## Step 3 — Build the Medical Scribe Agent and Task

```python
# ── 4. Medical Scribe Agent ───────────────────────────────────────────────────

def create_scribe_agent() -> Agent:
    return Agent(
        role="Medical Scribe & Patient Communicator",
        goal=(
            "Take the CMO's final decision and the Treatment Plan, and format them into "
            "highly readable, strictly formatted JSON reports for doctors and patients."
        ),
        backstory=(
            "You are a Medical Scribe. You excel at taking complex medical logic from the CMO "
            "and translating it into empathetic patient summaries and structured technical "
            "doctor notes. You never invent diagnoses; you only format what you are given."
        ),
        llm=ORCHESTRATOR_LLM,  # Qwen for strict JSON adherence and formatting
        verbose=True,
    )

def create_scribe_task(
    agent: Agent,
    cmo_task,
    treatment_task,
    research_task=None,
) -> Task:
    context = [t for t in [cmo_task, treatment_task, research_task] if t is not None]

    return Task(
        description=(
            "You have received the final clinical decision from the Chief Medical Officer (CMO), "
            "the Treatment Plan, and the PubMed Research.\n\n"
            "Your job is to compile the FinalDiagnosis JSON.\n"
            "1. Copy the primary_diagnosis, severity, confidence, reasoning, etc. directly from the CMO output.\n"
            "2. Write a compassionate, jargon-free `patient_summary`.\n"
            "3. Extract actionable `patient_recommendations` from the Treatment Plan.\n"
            "4. Write technical `doctor_notes` combining the CMO's reasoning with the Treatment Plan's protocol.\n"
            "5. Summarize the `literature_support` from the research task."
        ),
        expected_output=(
            "ONLY a valid JSON object matching the FinalDiagnosis schema. "
            "Keys: primary_diagnosis, confidence, severity, lesion_profile, clinical_reasoning, "
            "suggested_investigations, cited_pmids, patient_summary, patient_recommendations, "
            "doctor_notes, treatment_suggestions, literature_support, when_to_seek_care."
        ),
        agent=agent,
        output_pydantic=FinalDiagnosis,
        context=context,
    )
```

---

## Step 4 — Build the DermaCrew Assembly Class

Now create the file that wires everything together: **`crew/derma_crew.py`**

```python
# crew/derma_crew.py
# Assembles and runs the full DermaAI v2 pipeline.
# This is the main entry point called by main.py.

from crewai import Crew, Process

from agents.biodata_agent import create_biodata_agent, create_biodata_task
from agents.lesion_agents import (
    create_colour_agent, create_colour_task,
    create_texture_agent, create_texture_task,
    create_levelling_agent, create_levelling_task,
    create_shape_agent, create_shape_task,
)
from agents.decomposition_agent import create_decomposition_agent, create_decomposition_task
from agents.research_agent import create_research_agent, create_research_task
from agents.orchestrator_agent import create_orchestrator_agent, create_orchestrator_task, FinalDiagnosis
from tools.image_tool import ImageAnalysisTool
from utils.clarification_loop import run_clarification_loop


class DermaCrew:
    """
    Orchestrates the full multi-agent dermatology diagnosis pipeline.

    Usage:
        crew = DermaCrew(image_path="path/to/image.jpg", patient_text="I have...")
        result = crew.run()
        print(result.primary_diagnosis)
    """

    def __init__(self, image_path: str, patient_text: str):
        self.image_path = image_path
        self.patient_text = patient_text
        self._result: FinalDiagnosis | None = None

    def _run_vision_analysis(self) -> dict:
        """
        Each agent independently examines the image using VISION_LLM (medgemma).
        Returns a dict with keys: colour, texture, levelling, shape.

        The prompt for each call is written from that specialist agent's perspective —
        the model IS the agent looking at the full image and forming its own clinical view.

        This is done BEFORE the crew starts because medgemma does not support CrewAI's
        OpenAI-style tool-calling format. Calling it directly here avoids the infinite
        retry loop that occurs when the model outputs tool_code blocks instead of JSON
        function calls. Architecturally, these calls ARE the agents examining the image.
        """
        print("\n[Vision] Each specialist agent examining image independently...")
        tool = ImageAnalysisTool()

        results = {
            "colour": tool._run(
                self.image_path,
                "You are a Dermatology Colour Analyst directly examining a skin lesion. "
                "Look at the entire image. Describe the complete colour profile of the lesion: "
                "name all shades present using clinical terminology (erythematous, violaceous, "
                "hyperpigmented, hypopigmented, melanotic, etc.), note any colour variation "
                "or gradient across the lesion, and compare the lesion colour to the surrounding skin. "
                "Form your own independent clinical colour assessment.",
            ),
            "texture": tool._run(
                self.image_path,
                "You are a Dermatology Texture and Surface Analyst directly examining a skin lesion. "
                "Look at the entire image. Describe the complete surface texture of the lesion: "
                "note scaling, dryness, crusting, blistering, weeping, smoothness, or any other "
                "surface irregularity you observe. Describe the texture in clinical terms and note "
                "any areas where surface characteristics vary. "
                "Form your own independent clinical texture assessment.",
            ),
            "levelling": tool._run(
                self.image_path,
                "You are a Dermatology Morphology and Elevation Analyst directly examining a skin lesion. "
                "Look at the entire image. Determine the complete elevation profile of the lesion: "
                "is it raised, flat, or depressed relative to surrounding skin? "
                "Describe any shadow patterns at the lesion edge, dome or concave shape, "
                "and 3D visual cues visible in the photograph. "
                "Form your own independent clinical elevation assessment.",
            ),
            "shape": tool._run(
                self.image_path,
                "You are a Dermatology Border and Shape Analyst directly examining a skin lesion. "
                "Look at the entire image. Provide your complete border and shape assessment: "
                "evaluate border regularity (regular/irregular — is the edge smooth or jagged/notched?), "
                "border definition (well-defined/not well-defined — is the edge sharp or does it fade?), "
                "and note any asymmetry, satellite lesions, or ABCDE red flags you observe. "
                "Form your own independent clinical border assessment.",
            ),
        }
        print("[Vision] Independent specialist examination complete.\n")
        return results

    def run(self) -> FinalDiagnosis:
        print("\n" + "="*60)
        print("  DermaAI v2 — Multi-Agent Analysis Starting")
        print("="*60)

        # ── Clarification pre-pass (runs BEFORE the main crew) ────────────────
        # Run a lightweight biodata + decomposition + clarification mini-crew.
        # If the patient's input is missing critical fields, follow-up questions
        # are presented here. self.patient_text is updated with any Q&A answers.
        # If the patient gave complete information, this completes in seconds.
        print("\n[0/3] Running clarification pre-pass...")
        _pre_biodata_agent = create_biodata_agent()
        _pre_biodata_task  = create_biodata_task(_pre_biodata_agent)
        Crew(
            agents=[_pre_biodata_agent],
            tasks=[_pre_biodata_task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()

        self.patient_text, _ = run_clarification_loop(
            patient_text=self.patient_text,
            biodata_agent=_pre_biodata_agent,
            biodata_task=_pre_biodata_task,
        )
        # ──────────────────────────────────────────────────────────────────────

        # ── Phase 1: Create all agents ────────────────────────────────────────
        print("\n[1/3] Initialising agents...")

        biodata_agent    = create_biodata_agent()
        colour_agent     = create_colour_agent()
        texture_agent    = create_texture_agent()
        levelling_agent  = create_levelling_agent()
        shape_agent      = create_shape_agent()
        decomp_agent     = create_decomposition_agent()
        research_agent   = create_research_agent()
        orchestrator     = create_orchestrator_agent()

        # ── Phase 2: Create tasks in dependency order ─────────────────────────
        print("[2/3] Wiring task dependencies...")

        # Biodata runs first — everything else may reference it
        biodata_task     = create_biodata_task(biodata_agent)

        # Each agent independently examines the image (see _run_vision_analysis)
        vision = self._run_vision_analysis()

        # Each lesion task receives that agent's own independent image observations
        colour_task      = create_colour_task(colour_agent, self.image_path, biodata_task, vision_result=vision["colour"])
        texture_task     = create_texture_task(texture_agent, self.image_path, biodata_task, vision_result=vision["texture"])
        levelling_task   = create_levelling_task(levelling_agent, self.image_path, biodata_task, vision_result=vision["levelling"])
        shape_task       = create_shape_task(shape_agent, self.image_path, biodata_task, vision_result=vision["shape"])

        # Decomposition uses self.patient_text — which is now enriched if clarification happened
        decomp_task      = create_decomposition_task(decomp_agent, self.patient_text, biodata_task)

        # Research receives all upstream outputs
        research_task    = create_research_task(
            research_agent,
            biodata_task=biodata_task,
            colour_task=colour_task,
            texture_task=texture_task,
            levelling_task=levelling_task,
            shape_task=shape_task,
            decomposition_task=decomp_task,
        )

        # Orchestrator receives everything
        orchestrator_task = create_orchestrator_task(
            orchestrator,
            biodata_task=biodata_task,
            colour_task=colour_task,
            texture_task=texture_task,
            levelling_task=levelling_task,
            shape_task=shape_task,
            decomposition_task=decomp_task,
            research_task=research_task,
        )

        # ── Phase 3: Assemble and run the crew ───────────────────────────────
        print("[3/3] Running crew (this will take several minutes)...")

        crew = Crew(
            agents=[
                biodata_agent,
                colour_agent,
                texture_agent,
                levelling_agent,
                shape_agent,
                decomp_agent,
                research_agent,
                orchestrator,
            ],
            tasks=[
                biodata_task,
                colour_task,
                texture_task,
                levelling_task,
                shape_task,
                decomp_task,
                research_task,
                orchestrator_task,
            ],
            process=Process.sequential,
            verbose=True,
        )

        crew.kickoff()

        # Extract the structured result
        if orchestrator_task.output and orchestrator_task.output.pydantic:
            self._result = orchestrator_task.output.pydantic
        else:
            raise RuntimeError(
                "Orchestrator did not produce a valid structured output. "
                f"Raw output: {orchestrator_task.output.raw if orchestrator_task.output else 'None'}"
            )

        return self._result

    def get_intermediate_outputs(self, tasks: dict) -> dict:
        """
        Returns intermediate outputs from all tasks for debugging.
        Pass the task objects dict after running.
        """
        return {
            name: (task.output.pydantic or task.output.raw) if task.output else None
            for name, task in tasks.items()
        }
```

---

## Step 5 — Test the Orchestrator with a Small Crew

Before running the full 8-agent pipeline, test just the orchestrator with simulated upstream data. Create **`test_orchestrator.py`**:

```python
# test_orchestrator.py
# Tests orchestrator synthesis with biodata, decomposition, and research only.
# (Skips image-based lesion agents for speed — add them in Chapter 10.)

from crewai import Crew, Process
from agents.biodata_agent import create_biodata_agent, create_biodata_task
from agents.decomposition_agent import create_decomposition_agent, create_decomposition_task
from agents.research_agent import create_research_agent, create_research_task
from agents.orchestrator_agent import create_orchestrator_agent, create_orchestrator_task

PATIENT_TEXT = (
    "Red, itchy rash with small bumps on my left forearm for 4 days. "
    "Spreading. Works as a painter using new solvents. Worse in evenings."
)

# Create agents
biodata_agent   = create_biodata_agent()
decomp_agent    = create_decomposition_agent()
research_agent  = create_research_agent()
orchestrator    = create_orchestrator_agent()

# Create tasks
biodata_task    = create_biodata_task(biodata_agent)
decomp_task     = create_decomposition_task(decomp_agent, PATIENT_TEXT, biodata_task)
research_task   = create_research_task(
    research_agent,
    biodata_task=biodata_task,
    decomposition_task=decomp_task,
)
orch_task = create_orchestrator_task(
    orchestrator,
    biodata_task=biodata_task,
    decomposition_task=decomp_task,
    research_task=research_task,
)

crew = Crew(
    agents=[biodata_agent, decomp_agent, research_agent, orchestrator],
    tasks=[biodata_task, decomp_task, research_task, orch_task],
    process=Process.sequential,
    verbose=True,
)

crew.kickoff()

print("\n" + "="*60)
print("FINAL ORCHESTRATOR OUTPUT")
print("="*60)

if orch_task.output and orch_task.output.pydantic:
    result = orch_task.output.pydantic
    print(f"\nPrimary Diagnosis: {result.primary_diagnosis}")
    print(f"Confidence: {result.confidence}")
    print(f"Severity: {result.severity}")
    print(f"Re-diagnosis Applied: {result.re_diagnosis_applied}")
    if result.re_diagnosis_reason:
        print(f"Re-diagnosis Reason: {result.re_diagnosis_reason}")
    print(f"\nDifferential Diagnoses:")
    for dx in result.differential_diagnoses:
        print(f"  - {dx}")
    print(f"\n--- PATIENT SUMMARY ---")
    print(result.patient_summary)
    print(f"\n--- PATIENT RECOMMENDATIONS ---")
    for i, rec in enumerate(result.patient_recommendations, 1):
        print(f"  {i}. {rec}")
    print(f"\n--- DOCTOR NOTES (excerpt) ---")
    print(result.doctor_notes[:500] + "...")
    print(f"\nCited PMIDs: {result.cited_pmids}")
else:
    print("Raw output:")
    print(orch_task.output.raw if orch_task.output else "No output")
```

Run it:
```powershell
python test_orchestrator.py
```

**This will take 5–10 minutes** — it's running 4 agents with real LLM calls and PubMed queries.

**What to verify:**
- `primary_diagnosis` is clinically reasonable (should be contact dermatitis or similar)
- `patient_summary` uses no medical jargon
- `doctor_notes` is detailed enough to be clinically useful
- `cited_pmids` are real (check one on PubMed)
- `re_diagnosis_applied` is either False (confidence) or True (with a clear reason)
- `disclaimer` is present

Delete when done:
```powershell
Remove-Item test_orchestrator.py
```

---

## Checkpoint ✅

- [ ] `agents/orchestrator_agent.py` exists with `FinalDiagnosis`, agent, and task factory
- [ ] `utils/resilient_base.py` exists; `FinalDiagnosis` inherits `ResilientBase` (not `BaseModel`)
- [ ] `flatten_nested_diagnosis` `@model_validator(mode="before")` is present — lifts nested `"diagnosis"` key to root and maps aliases
- [ ] `coerce_differential_diagnoses` `@field_validator` converts list-of-dicts and plain-dict forms to `list[str]`
- [ ] `confidence` and `severity` have `default=` values so the model never crashes with "Field required"
- [ ] `crew/derma_crew.py` exists with the `DermaCrew` class
- [ ] Running the partial pipeline (without image agents) produces a `FinalDiagnosis` object
- [ ] `primary_diagnosis` makes clinical sense for the test input
- [ ] `patient_summary` is readable by a non-medical person
- [ ] `doctor_notes` is detailed and technically sound
- [ ] `re_diagnosis_applied` field reflects actual orchestrator reasoning
- [ ] `cited_pmids` are verifiable

---

---

## What Comes Next — Chapters 06b, 12–15

This chapter built the 8-agent pipeline. Before running it end-to-end, you need:

| Chapter | What it adds | Impact on DermaCrew |
|---------|-------------|---------------------|
| **06b** | Clarification Agent + clarification loop | `run_clarification_loop()` pre-pass added to `DermaCrew.run()` |
| **12** | Differential, Mimic, and Treatment agents | Adds 3 agents; CMO context grows |
| **13** | Doctor approval loop + AuditTrail collection | `DermaCrew.run()` returns `(result, audit)`; AuditTrail captures clarification data |
| **14** | Voice input (faster-whisper) | New option in `main.py` patient text input |
| **15** | Voice output (ElevenLabs) | Post-approval optional step in `main.py` |
| **11** | Three PDF types incl. full audit PDF | `pdf_service.py` redesigned |

**Do not build `main.py` yet.** The `main.py` shown in Chapter 10 Step 1 will be updated in Chapter 13 Step 5 with the approval loop. Build it then.

---

*Next → `10_FULL_PIPELINE.md`*
