# Chapter 12 — Differential Diagnosis + Treatment Plan Agents

**Goal:** Add two new specialist agents — one that produces a detailed, evidence-grounded differential diagnosis and one that produces a complete treatment protocol. These agents sit between the Research Agent and the Orchestrator, giving the Orchestrator a far richer picture to synthesise from.

**Time estimate:** 50–70 minutes

---

## Why Two New Agents?

The existing Orchestrator already produced a brief `differential_diagnoses` list and `treatment_suggestions` list — but they were generated in a single pass alongside everything else. That is fine for a quick summary but not good enough for a document a doctor would act on.

**What the doctor needs:**
- For each differential: *why* it was considered, *which specific findings* support or argue against it, and *what test* would definitively rule it in or out
- For treatment: tiered protocols (first-line → second-line → escalation), with evidence levels, dosing, duration, and contraindications

These are deep enough tasks that they deserve their own agents with their own LLM prompts and their own structured Pydantic schemas.

**Where they fit in the pipeline:**

```
Biodata
  └─► 4 Lesion Agents
        └─► Decomposition
              └─► Research
                    └─► Differential Diagnosis  ◄── NEW
                          └─► Treatment Plan     ◄── NEW
                                └─► Orchestrator
```

The Orchestrator now receives the differential and treatment as structured context — it synthesises rather than generating from scratch, making its final output more accurate and internally consistent.

---

## Step 1 — Create the Agent File

Create **`agents/clinical_agents.py`**. Both agents live in this file because they are closely related and always used together.

### Step 1a — Imports and Schemas

```python
# agents/clinical_agents.py
# Differential Diagnosis Agent + Treatment Plan Agent
# These run after the Research Agent and before the Orchestrator.

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from crewai import Agent, Task
from config import VISION_LLM   # MedGemma — lesion, differential, treatment agents
from utils.resilient_base import ResilientBase


# ── Differential Diagnosis Schema ─────────────────────────────────────────────

class DifferentialEntry(BaseModel):
    """One alternative diagnosis candidate with full clinical justification."""

    condition: str = Field(
        default="",
        description="Name of the differential diagnosis condition"
    )
    probability: Literal["high", "moderate", "low"] = Field(
        description="Likelihood this is the correct diagnosis given current findings"
    )
    key_features_matching: list[str] = Field(
        default=[],
        description=(
            "Specific visual or symptom features that support this diagnosis. "
            "Reference actual findings from the lesion agents and decomposition."
        )
    )
    key_features_against: list[str] = Field(
        default=[],
        description=(
            "Specific findings that argue AGAINST this diagnosis. "
            "Honest counter-evidence is essential for the doctor's review."
        )
    )
    distinguishing_test: str = Field(
        default="",
        description=(
            "The single most useful investigation or test that would confirm "
            "or rule out this diagnosis (e.g., patch test, skin biopsy, KOH prep)."
        )
    )
    clinical_reasoning: str = Field(
        default="",
        description="2-3 sentence explanation of the clinical logic for this differential"
    )

    @field_validator("key_features_matching", "key_features_against", mode="before")
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if v is not None else []

    @field_validator("condition", "distinguishing_test", "clinical_reasoning", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("probability", mode="before")
    @classmethod
    def normalise_probability(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        lower = v.lower()
        if "high" in lower:
            return "high"
        if "low" in lower:
            return "low"
        return "moderate"


class DifferentialDiagnosisOutput(ResilientBase):
    """
    Complete differential diagnosis list with clinical justification for each entry.
    Produced by the Differential Diagnosis Agent before the Orchestrator runs.
    """

    primary_diagnosis: str = Field(
        default="",
        description="The most likely diagnosis based on all available findings"
    )
    confidence_in_primary: Literal["high", "moderate", "low"] = Field(
        default="moderate",
        description="Confidence level in the primary diagnosis"
    )
    primary_reasoning: str = Field(
        default="",
        description=(
            "Detailed reasoning for why this is the primary diagnosis. "
            "Must reference specific lesion findings, demographics, and research evidence."
        )
    )
    differentials: list[DifferentialEntry] = Field(
        default=[],
        description=(
            "2–4 alternative diagnoses to consider and rule out. "
            "Ordered from most to least likely."
        )
    )
    red_flags: list[str] = Field(
        default=[],
        description=(
            "Any clinical features present that raise concern for malignancy, "
            "systemic disease, or conditions requiring urgent referral."
        )
    )
    requires_urgent_referral: bool = Field(
        default=False,
        description="True if any red flags indicate this should not wait for a routine appointment"
    )

    @field_validator("differentials", "red_flags", mode="before")
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if v is not None else []

    @field_validator("primary_diagnosis", "primary_reasoning", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("confidence_in_primary", mode="before")
    @classmethod
    def normalise_confidence(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        lower = v.lower()
        if "high" in lower:
            return "high"
        if "low" in lower:
            return "low"
        return "moderate"

    @field_validator("requires_urgent_referral", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return v


# ── Treatment Plan Schema ──────────────────────────────────────────────────────

class TreatmentEntry(BaseModel):
    """A single treatment option with clinical detail."""

    line: Literal["first", "second", "third", "adjunct"] = Field(
        description="Treatment line in the escalation protocol"
    )
    treatment_name: str = Field(
        default="",
        description="Name of medication, procedure, or intervention"
    )
    dose_or_protocol: str = Field(
        default="",
        description="Specific dose, frequency, route of administration, or procedure protocol"
    )
    duration: str = Field(
        default="",
        description="How long to continue this treatment"
    )
    rationale: str = Field(
        default="",
        description="Why this treatment is recommended and what evidence supports it"
    )
    monitoring: str = Field(
        default="",
        description="What to monitor during this treatment (side effects, response markers)"
    )

    @field_validator("treatment_name", "dose_or_protocol", "duration", "rationale", "monitoring", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("line", mode="before")
    @classmethod
    def normalise_line(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        lower = v.lower()
        if "first" in lower or "1st" in lower:
            return "first"
        if "second" in lower or "2nd" in lower:
            return "second"
        if "third" in lower or "3rd" in lower:
            return "third"
        if "adjunct" in lower or "supplement" in lower or "add-on" in lower:
            return "adjunct"
        return "first"


class TreatmentPlanOutput(ResilientBase):
    """
    Detailed treatment protocol for the primary diagnosis.
    Produced by the Treatment Plan Agent before the Orchestrator runs.
    """

    for_diagnosis: str = Field(
        default="",
        description="The diagnosis this treatment plan addresses"
    )
    immediate_actions: list[str] = Field(
        default=[],
        description=(
            "Actions the patient or clinician should take immediately — "
            "before any prescription fills or lab results. "
            "Examples: avoid trigger, apply cool compress, cease offending medication."
        )
    )
    medications: list[TreatmentEntry] = Field(
        default=[],
        description="Tiered medication protocol from first-line to escalation"
    )
    non_pharmacological: list[str] = Field(
        default=[],
        description=(
            "Non-drug interventions: lifestyle changes, trigger avoidance, "
            "wound care, phototherapy, diet, occupational adjustments, etc."
        )
    )
    patient_instructions: str = Field(
        default="",
        description=(
            "Plain-language instructions for the patient. No jargon. "
            "What to do, what to avoid, what to expect."
        )
    )
    follow_up: str = Field(
        default="",
        description=(
            "When and how to follow up — timeline, what to assess, "
            "who to see (GP, dermatologist, allergy specialist, etc.)"
        )
    )
    referral_needed: bool = Field(
        default=False,
        description="Whether the patient needs a specialist referral"
    )
    referral_to: str = Field(
        default="",
        description="Type of specialist (if referral_needed is True)"
    )
    contraindications: list[str] = Field(
        default=[],
        description=(
            "Specific contraindications based on this patient's biodata — "
            "allergies, current medications, age-related restrictions, etc."
        )
    )
    evidence_level: Literal["strong", "moderate", "limited", "expert_opinion"] = Field(
        default="moderate",
        description="Strength of evidence supporting this treatment plan"
    )

    @field_validator(
        "immediate_actions", "medications", "non_pharmacological", "contraindications",
        mode="before",
    )
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if v is not None else []

    @field_validator(
        "for_diagnosis", "patient_instructions", "follow_up", "referral_to",
        mode="before",
    )
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("referral_needed", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return v

    @field_validator("evidence_level", mode="before")
    @classmethod
    def normalise_evidence_level(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        lower = v.lower()
        if "strong" in lower:
            return "strong"
        if "limited" in lower or "weak" in lower or "poor" in lower:
            return "limited"
        if "expert" in lower or "opinion" in lower or "consensus" in lower:
            return "expert_opinion"
        return "moderate"
```

> **Model note:** Both agents use `VISION_LLM` (MedGemma) — the same vision model as the lesion agents. Neither agent calls a tool, so there is no tool-calling constraint. MedGemma's strong medical knowledge base makes it the right choice for differential and treatment reasoning.
>
> **Null-safety and Literal normalisers:** Every field carries a `default` value and a `@field_validator(mode="before")` that coerces `null → []`/`""` for list/string fields and maps fuzzy LLM descriptions to the correct Literal. `TreatmentEntry.rationale` in particular has `default=""` — it was previously required, which caused `test.py` to fail when constructing mock data without it. Literal normalisers handle common LLM habits: `"1st-line"` → `"first"`, `"expert consensus"` → `"expert_opinion"`, `"high confidence"` → `"high"`, etc.
>
> **`ResilientBase` — raw-output sanitiser (shared):** `DifferentialDiagnosisOutput` and `TreatmentPlanOutput` both inherit from `ResilientBase` imported from `utils/resilient_base.py`. This shared module is the single source of truth — all agent output classes across the project (lesion agents, clinical agents, orchestrator) use the same class. `ResilientBase` overrides `model_validate_json` to run `re.search(r'\{.*\}', ...)` on the raw string before handing it to Pydantic's JSON parser, stripping any self-check narration or backtick garbage the LLM emits adjacent to the `{` brace. Field-level validators cannot catch this because the JSON parse fails before they are called.

---

### Step 1b — Differential Diagnosis Agent and Task

Add below the schemas:

```python
# ── Agent 1: Differential Diagnosis ───────────────────────────────────────────

def create_differential_agent() -> Agent:
    return Agent(
        role="Dermatology Differential Diagnosis Specialist",
        goal=(
            "Produce a rigorous ranked differential diagnosis from the lesion findings, "
            "symptoms, demographics, and research evidence. "
            "For each condition: state what fits, what does not fit, and what test resolves it. "
            "No commentary beyond clinical diagnostic reasoning."
        ),
        backstory=(
            "You are a consultant dermatologist. "
            "Every diagnosis you make must be justified: what clinical features support it, "
            "what features argue against it, and which single test would confirm or exclude it. "
            "You never anchor on one diagnosis — you always consider at least two alternatives. "
            "Red flags for malignancy (asymmetric borders, multicolour, rapid change, "
            "ulceration, diameter >6mm) are always checked."
        ),
        llm=VISION_LLM,
        verbose=True,
    )


def create_differential_task(
    agent: Agent,
    biodata_task=None,
    colour_task=None,
    texture_task=None,
    levelling_task=None,
    shape_task=None,
    decomposition_task=None,
    research_task=None,
) -> Task:
    context = [
        t for t in [
            biodata_task, colour_task, texture_task,
            levelling_task, shape_task, decomposition_task, research_task,
        ]
        if t is not None
    ]

    return Task(
        description=(
            "You have received the complete clinical picture from all upstream agents:\n"
            "  ✓ Patient biodata (age, sex, skin tone, occupation, history)\n"
            "  ✓ Lesion colour, surface, elevation, and border analysis\n"
            "  ✓ Decomposed patient symptoms and history\n"
            "  ✓ PubMed research findings with cited evidence\n\n"

            "YOUR TASK:\n"
            "1. Identify the PRIMARY diagnosis — the single most likely condition "
            "given ALL the evidence combined.\n"
            "2. Build a DIFFERENTIAL LIST of 2–4 alternative conditions:\n"
            "   - For each: which specific findings support it?\n"
            "   - For each: which specific findings argue AGAINST it?\n"
            "   - For each: what ONE test would confirm or rule it out?\n"
            "3. Flag any RED FLAGS — features suggesting malignancy, systemic disease, "
            "or need for urgent referral.\n\n"

            "CRITICAL RULES:\n"
            "- Reference SPECIFIC findings from the lesion agents (e.g., 'The irregular border "
            "from the shape agent raises concern for...')\n"
            "- Reference SPECIFIC evidence from research (mention PMIDs where relevant)\n"
            "- Do not list a differential unless you can articulate both FOR and AGAINST evidence\n"
            "- Do not diagnose malignancy definitively — flag it as requiring biopsy"
        ),
        expected_output=(
            "ONLY a valid JSON object — no text, no explanation, no markdown. "
            "Start with { and end with }. Do NOT write anything after the closing }. "
            "Use exactly these keys: primary_diagnosis, confidence_in_primary, primary_reasoning, "
            "differentials (list of objects with: condition, probability, key_features_matching, "
            "key_features_against, distinguishing_test, clinical_reasoning), "
            "red_flags (list of strings), requires_urgent_referral (boolean). "
            "Each differential entry must have non-empty key_features_matching, "
            "key_features_against, distinguishing_test, and clinical_reasoning."
        ),
        agent=agent,
        output_pydantic=DifferentialDiagnosisOutput,
        context=context,
    )
```

---

### Step 1c — Treatment Plan Agent and Task

Add below the differential agent:

```python
# ── Agent 2: Treatment Plan ────────────────────────────────────────────────────

def create_treatment_agent() -> Agent:
    return Agent(
        role="Dermatology Treatment Protocol Specialist",
        goal=(
            "Write a complete, patient-specific, evidence-based treatment protocol "
            "for the confirmed diagnosis. "
            "Include tiered medications, immediate actions, and follow-up. "
            "No commentary — only the treatment plan."
        ),
        backstory=(
            "You are a dermatology treatment specialist. "
            "You write treatment protocols: first-line, second-line, escalation. "
            "You check the patient's biodata for allergies, current medications, and age "
            "before recommending anything. Contraindications must be patient-specific, not generic. "
            "Immediate actions come first. Prescriptions and referrals are clearly labelled."
        ),
        llm=VISION_LLM,
        verbose=True,
    )


def create_treatment_task(
    agent: Agent,
    biodata_task=None,
    research_task=None,
    differential_task=None,
) -> Task:
    context = [
        t for t in [biodata_task, research_task, differential_task]
        if t is not None
    ]

    return Task(
        description=(
            "You have the differential diagnosis, patient biodata, and research evidence.\n\n"

            "YOUR TASK — Design a treatment plan for the PRIMARY diagnosis:\n\n"

            "STEP 1 — IMMEDIATE ACTIONS:\n"
            "  What should the patient do RIGHT NOW before any prescription or test?\n"
            "  (e.g., stop using offending product, apply cold compress, avoid scratching)\n\n"

            "STEP 2 — MEDICATION PROTOCOL:\n"
            "  First-line: The most evidence-based, least invasive treatment.\n"
            "  Second-line: If first-line fails after the expected response time.\n"
            "  Third-line: Escalation for refractory or severe cases.\n"
            "  For each: specific drug name, dose, route, frequency, duration, monitoring.\n\n"

            "STEP 3 — NON-PHARMACOLOGICAL INTERVENTIONS:\n"
            "  Trigger avoidance, skincare routine, lifestyle adjustments, wound care.\n\n"

            "STEP 4 — PATIENT INSTRUCTIONS:\n"
            "  Write plain-English instructions the patient can follow without a dictionary.\n\n"

            "STEP 5 — FOLLOW-UP AND REFERRAL:\n"
            "  When to follow up, what to assess at follow-up, whether specialist referral\n"
            "  is needed and to whom.\n\n"

            "PATIENT-SPECIFIC RULES:\n"
            "- Check the biodata for known allergies and current medications — "
            "list any specific contraindications for THIS patient.\n"
            "- Adjust recommendations for patient age (paediatric dosing, geriatric caution).\n"
            "- Reference the research evidence level where possible."
        ),
        expected_output=(
            "ONLY a valid JSON object — no text, no explanation, no markdown. "
            "Start with { and end with }. Do NOT write anything after the closing }. "
            "Use exactly these keys: for_diagnosis, immediate_actions (list), "
            "medications (list of objects with: line, treatment_name, dose_or_protocol, duration, rationale, monitoring), "
            "non_pharmacological (list), patient_instructions, follow_up, "
            "referral_needed (boolean), referral_to, contraindications (list), evidence_level. "
            "medications must have at least a first-line entry with specific dose and duration. "
            "patient_instructions must be in plain English with no unexplained medical terms. "
            "contraindications must be specific to THIS patient's biodata, not generic."
        ),
        agent=agent,
        output_pydantic=TreatmentPlanOutput,
        context=context,
    )
```

---

## Step 2 — Update the Orchestrator to Use These Agents

The Orchestrator's role shifts from generating a differential from scratch to **synthesising and validating** what the new agents already produced.

These changes are already applied in **`agents/orchestrator_agent.py`**. For reference, the completed `create_orchestrator_task()` signature and task description look like this:

```python
def create_orchestrator_task(
    agent: Agent,
    biodata_task=None,
    colour_task=None,
    texture_task=None,
    levelling_task=None,
    shape_task=None,
    decomposition_task=None,
    research_task=None,
    differential_task=None,
    treatment_task=None,
) -> Task:
    context = [
        t for t in [
            biodata_task, colour_task, texture_task,
            levelling_task, shape_task, decomposition_task,
            research_task, differential_task, treatment_task,
        ]
        if t is not None
    ]

    return Task(
        description=(
            "You have received the complete clinical picture from all specialist agents:\n"
            "  ✓ Patient biodata\n"
            "  ✓ Lesion colour, texture, levelling, and border analysis\n"
            "  ✓ Decomposed symptoms and history\n"
            "  ✓ PubMed research findings\n\n"
            "  ✓ Differential diagnosis with clinical justification for each alternative\n"
            "  ✓ Evidence-based treatment protocol\n"

            "STEP 1 — VALIDATE:\n"
            "  Do the differential and treatment agents agree with the visual analysis?\n"
            "  Does the primary diagnosis make sense across ALL data sources?\n"
            "  If the differential agent identified any red flags, does the severity reflect that?\n\n"
            # ... STEP 2 and STEP 3 continue as before
        ),
        ...
    )
```

---

## Step 3 — Test the Two New Agents

Create **`test_clinical_agents.py`** to test them after the research step:

```python
# test_clinical_agents.py
from crewai import Crew, Process
from agents.biodata_agent import create_biodata_agent, create_biodata_task
from agents.decomposition_agent import create_decomposition_agent, create_decomposition_task
from agents.research_agent import create_research_agent, create_research_task
from agents.clinical_agents import (
    create_differential_agent, create_differential_task,
    create_treatment_agent, create_treatment_task,
)

PATIENT_TEXT = (
    "I have had a very itchy, bumpy rash on my left forearm for 4 days. "
    "Spreading. I work as a painter and recently switched solvent brands."
)

biodata_agent  = create_biodata_agent()
decomp_agent   = create_decomposition_agent()
research_agent = create_research_agent()
diff_agent     = create_differential_agent()
treat_agent    = create_treatment_agent()

biodata_task   = create_biodata_task(biodata_agent)
decomp_task    = create_decomposition_task(decomp_agent, PATIENT_TEXT, biodata_task)
research_task  = create_research_task(research_agent, biodata_task=biodata_task, decomposition_task=decomp_task)
diff_task      = create_differential_task(
    diff_agent,
    biodata_task=biodata_task,
    decomposition_task=decomp_task,
    research_task=research_task,
)
treat_task     = create_treatment_task(
    treat_agent,
    biodata_task=biodata_task,
    research_task=research_task,
    differential_task=diff_task,
)

crew = Crew(
    agents=[biodata_agent, decomp_agent, research_agent, diff_agent, treat_agent],
    tasks=[biodata_task, decomp_task, research_task, diff_task, treat_task],
    process=Process.sequential,
    verbose=True,
)

crew.kickoff()

print("\n" + "="*60)
print("DIFFERENTIAL DIAGNOSIS OUTPUT")
print("="*60)
if diff_task.output and diff_task.output.pydantic:
    d = diff_task.output.pydantic
    print(f"\nPrimary: {d.primary_diagnosis} ({d.confidence_in_primary} confidence)")
    print(f"Primary reasoning: {d.primary_reasoning[:300]}...")
    print(f"\nDifferentials ({len(d.differentials)}):")
    for entry in d.differentials:
        print(f"  [{entry.probability.upper()}] {entry.condition}")
        print(f"    FOR: {', '.join(entry.key_features_matching[:2])}")
        print(f"    AGAINST: {', '.join(entry.key_features_against[:2])}")
        print(f"    CONFIRM WITH: {entry.distinguishing_test}")
    if d.red_flags:
        print(f"\n  ⚠️  Red flags: {', '.join(d.red_flags)}")
    print(f"  Urgent referral: {d.requires_urgent_referral}")

print("\n" + "="*60)
print("TREATMENT PLAN OUTPUT")
print("="*60)
if treat_task.output and treat_task.output.pydantic:
    t = treat_task.output.pydantic
    print(f"\nFor: {t.for_diagnosis}")
    print(f"\nImmediate actions:")
    for a in t.immediate_actions:
        print(f"  • {a}")
    print(f"\nMedications:")
    for m in t.medications:
        print(f"  [{m.line.upper()}] {m.treatment_name} — {m.dose_or_protocol} for {m.duration}")
    print(f"\nFollow-up: {t.follow_up}")
    print(f"Referral needed: {t.referral_needed}")
    if t.contraindications:
        print(f"Contraindications (patient-specific): {', '.join(t.contraindications)}")
    print(f"Evidence level: {t.evidence_level}")
```

Run it:
```powershell
python test_clinical_agents.py
```

**What to verify:**

**Differential output:**
- Primary diagnosis is clinically reasonable
- Each differential has specific, non-generic FOR and AGAINST evidence
- `key_features_matching` references actual findings (not "redness" generically)
- `distinguishing_test` is a real clinical test, not "see a doctor"
- `red_flags` is empty or genuinely concerning features — not a generic warning list

**Treatment output:**
- First-line medication is specific: drug name, dose, frequency, duration
- `immediate_actions` are practical and immediate (not "consult a doctor")
- `patient_instructions` reads naturally — no unexplained medical terms
- `contraindications` mentions THIS patient's profile (from biodata), not generic disclaimers
- `follow_up` gives a specific timeline

**Delete `test_clinical_agents.py` when it passes.**

---

## Step 4 — Wire Into DermaCrew

These changes are already applied in **`crew/derma_crew.py`**. The complete Phase 1 → Phase 3 block for reference:

**Phase 1 — agents (at the top of `run()`):**
```python
from agents.clinical_agents import (
    create_differential_agent, create_differential_task,
    create_treatment_agent, create_treatment_task,
)

# ... inside run():
biodata_agent    = create_biodata_agent()
colour_agent     = create_colour_agent()
texture_agent    = create_texture_agent()
levelling_agent  = create_levelling_agent()
shape_agent      = create_shape_agent()
decomp_agent     = create_decomposition_agent()
research_agent   = create_research_agent()
diff_agent       = create_differential_agent()
treatment_agent  = create_treatment_agent()
orchestrator     = create_orchestrator_agent()
```

**Phase 2 — tasks (after the vision block):**
```python
diff_task = create_differential_task(
    diff_agent,
    biodata_task=biodata_task,
    colour_task=colour_task,        # None if no image
    texture_task=texture_task,
    levelling_task=levelling_task,
    shape_task=shape_task,
    decomposition_task=decomp_task,
    research_task=research_task,
)

treatment_task = create_treatment_task(
    treatment_agent,
    biodata_task=biodata_task,
    research_task=research_task,
    differential_task=diff_task,
)

orchestrator_task = create_orchestrator_task(
    orchestrator,
    biodata_task=biodata_task,
    colour_task=colour_task,
    texture_task=texture_task,
    levelling_task=levelling_task,
    shape_task=shape_task,
    decomposition_task=decomp_task,
    research_task=research_task,
    differential_task=diff_task,
    treatment_task=treatment_task,
)
```

**Phase 3 — crew assembly and audit trail:**
```python
crew = Crew(
    agents=[biodata_agent] + lesion_agents + [decomp_agent, research_agent, diff_agent, treatment_agent, orchestrator],
    tasks=[biodata_task] + lesion_tasks + [decomp_task, research_task, diff_task, treatment_task, orchestrator_task],
    process=Process.sequential,
    verbose=True,
)

try:
    crew.kickoff()
except Exception as e:
    print(f"\n[Warning] Crew execution encountered an error: {e}")
    print("[Warning] Attempting to extract partial results...\n")

    # If the orchestrator never ran (upstream task failed), attempt a
    # recovery mini-crew using only the tasks that produced output.
    if orchestrator_task.output is None:
        print("[Recovery] Orchestrator did not run — starting isolated recovery pass...")
        try:
            def _has_output(t):
                return t is not None and getattr(t, "output", None) is not None

            recovery_orch_task = create_orchestrator_task(
                orchestrator,
                biodata_task=biodata_task if _has_output(biodata_task) else None,
                colour_task=colour_task if _has_output(colour_task) else None,
                texture_task=texture_task if _has_output(texture_task) else None,
                levelling_task=levelling_task if _has_output(levelling_task) else None,
                shape_task=shape_task if _has_output(shape_task) else None,
                decomposition_task=decomp_task if _has_output(decomp_task) else None,
                research_task=research_task if _has_output(research_task) else None,
                differential_task=diff_task if _has_output(diff_task) else None,
                treatment_task=None,  # failed — excluded from recovery context
            )
            recovery_crew = Crew(
                agents=[orchestrator],
                tasks=[recovery_orch_task],
                process=Process.sequential,
                verbose=True,
            )
            recovery_crew.kickoff()
            orchestrator_task = recovery_orch_task  # promote for audit collection below
            print("[Recovery] Orchestrator recovery run succeeded.\n")
        except Exception as re_err:
            print(f"[Recovery] Recovery run also failed: {re_err}\n")

# ── Collect audit trail (runs after crew, using correct task objects) ─────────
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
```

> **Audit trail note:** The collection block must run **after** `crew.kickoff()` — not before. Each line reads from the task object that was populated during the run. `colour_task` and the other lesion tasks are guarded with `if colour_task and colour_task.output` because they may be `None` when no image is provided.

---

## Checkpoint ✅

- [ ] `agents/clinical_agents.py` exists with both schemas, both agents, and both task factories
- [ ] `utils/resilient_base.py` is imported (`from utils.resilient_base import ResilientBase`) — no local copy of `ResilientBase` defined in this file
- [ ] `DifferentialDiagnosisOutput` and `TreatmentPlanOutput` inherit from `ResilientBase`, not `BaseModel`
- [ ] `DifferentialEntry` and `TreatmentEntry` (nested models) still inherit `BaseModel` directly — they don't need the sanitiser since they're never the top-level `output_pydantic` target
- [ ] `DifferentialEntry` has all 6 fields (all with `default=""` or `default=[]`); `DifferentialDiagnosisOutput` has all 6 fields (all with defaults)
- [ ] `TreatmentEntry` has all 6 fields (all with `default=""`); `TreatmentPlanOutput` has all 10 fields (all with defaults)
- [ ] All `list[str]` fields have a `coerce_null_to_list` validator (`mode="before"`)
- [ ] All `str` fields have a `coerce_null_str` validator (`mode="before"`)
- [ ] All `bool` fields have a `coerce_bool` validator that handles `"true"`/`"false"` strings
- [ ] All `Literal` fields have a normaliser validator that maps fuzzy LLM descriptions to the correct value
- [ ] Both `expected_output` strings forbid markdown/text outside the JSON object
- [ ] Test passes: differential lists 2–4 conditions with specific FOR/AGAINST per entry
- [ ] Test passes: treatment has at least a first-line medication with dose + duration
- [ ] Both agents use `VISION_LLM` — same model as lesion agents (Decomposition uses qwen2.5:7b)
- [ ] `DermaCrew` updated to include both new agents and tasks
- [ ] Orchestrator's task signature updated to accept `differential_task` and `treatment_task`
- [ ] Audit trail collection block runs **after** `crew.kickoff()` and references each correct task object (not `_pre_biodata_task`)
- [ ] `crew.kickoff()` is wrapped in `try/except Exception` to prevent a single validation error from crashing the whole pipeline
- [ ] When the orchestrator never ran (upstream task halted the crew), a recovery mini-crew is launched automatically with only the tasks that produced output; you see `[Recovery] Orchestrator did not run — starting isolated recovery pass...` in the console
- [ ] The `RuntimeError` message includes a list of upstream tasks with no output so the root cause is immediately visible
- [ ] Full pipeline now runs 10 agents (biodata + 4 lesion + decomp + research + differential + treatment + orchestrator)

### Common Errors Fixed by This Hardening

| Error | Root Cause | Fix Applied |
|---|---|---|
| `ValidationError: Input should be a valid list` | LLM returned `null` for a list field | `coerce_null_to_list` validator on every `list[str]` field |
| `ValidationError: Input should be 'first'` | LLM wrote `"1st-line"` or `"First Line"` | `normalise_line` validator in `TreatmentEntry` |
| `ValidationError: Input should be 'high'` | LLM wrote `"high confidence"` | `normalise_probability` / `normalise_confidence` validators |
| `ValidationError: Input should be 'strong'` | LLM wrote `"strong evidence"` | `normalise_evidence_level` validator |
| `ValidationError: Invalid JSON: trailing characters` | LLM appended prose after the closing `}` | `expected_output` explicitly forbids text outside JSON |
| `ValidationError: Field required (rationale)` | `TreatmentEntry.rationale` had no default | Added `default=""` to `rationale` |
| `Invalid JSON: key must be a string at line 1 column 2` | LLM prepended self-check narration directly after `{` (e.g. `` {`? Yes. `` ) — the JSON parser sees the backtick as a non-string key before any field validator runs | `ResilientBase.model_validate_json` strips everything outside `{ … }` before parsing |

---

*Next → `13_DOCTOR_APPROVAL_LOOP.md`*
