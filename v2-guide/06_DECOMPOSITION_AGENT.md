# Chapter 06 — The Decomposition Agent

**Goal:** Build the agent that extracts structured clinical information from the patient's free-text input — symptoms, onset time, progression, complaints, and more. Test it with realistic patient text inputs.

**Time estimate:** 30–40 minutes

---

## What This Agent Does

When a patient types:

> *"I've had this itchy red bumpy rash on my left forearm for about 4 days. It started as a small dot and has spread. It gets really bad in the evenings. I work as a painter so I'm around chemicals all day."*

The Decomposition Agent reads this and returns:

```json
{
  "symptoms": ["itchy rash", "erythema", "papular eruption"],
  "time_days": 4,
  "onset": "sudden",
  "progression": "spreading",
  "aggravating_factors": ["evening time", "chemical exposure"],
  "relieving_factors": [],
  "body_location": "left forearm",
  "associated_symptoms": [],
  "occupational_exposure": "painter - chemical exposure",
  "patient_description": "small dot that spread"
}
```

This structured output is what the Research Agent uses to form targeted PubMed search queries.

---

## Step 1 — Design the Output Schema

Open **`agents/decomposition_agent.py`** and start with the Pydantic output model:

```python
# agents/decomposition_agent.py
# Extracts structured clinical information from patient free-text input.
# Uses ORCHESTRATOR_LLM (qwen2.5:7b) — strict schema compliance, reliable JSON output.
# No vision or tool calls needed here.

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from crewai import Agent, Task
from config import ORCHESTRATOR_LLM


def _null_to_list(v):
    """Coerce None → [] so the LLM can legally return null for empty list fields."""
    return v if v is not None else []


class DecompositionOutput(BaseModel):
    """
    Structured clinical data extracted from patient's reported symptoms.
    All fields are optional — the agent only populates what the patient mentioned.
    """

    symptoms: list[str] = Field(
        default=[],
        description="List of identified symptoms using clinical terminology where appropriate"
    )
    time_days: Optional[int] = Field(
        default=None,
        description="Duration of the condition in days (convert weeks/months if mentioned)"
    )
    onset: Optional[str] = Field(
        default=None,
        description="How the condition started: sudden / gradual / unknown"
    )
    progression: Optional[str] = Field(
        default=None,
        description="How it has changed: spreading / stable / improving / worsening / fluctuating"
    )
    body_location: list[str] = Field(
        default=[],
        description="Body locations mentioned by patient (can be multiple)"
    )
    aggravating_factors: list[str] = Field(
        default=[],
        description="Things that make the condition worse"
    )
    relieving_factors: list[str] = Field(
        default=[],
        description="Things that make the condition better"
    )
    associated_symptoms: list[str] = Field(
        default=[],
        description="Other symptoms mentioned: fever, fatigue, joint pain, etc."
    )
    occupational_exposure: list[str] = Field(
        default=[],
        description="Work-related exposures relevant to skin condition (can be multiple)"
    )
    recent_exposures: list[str] = Field(
        default=[],
        description="Recent new products, foods, environments, medications, or contacts"
    )
    patient_description: str = Field(
        default="",
        description="The patient's own words describing the lesion appearance, preserved verbatim"
    )
    prior_treatments: list[str] = Field(
        default=[],
        description="Any treatments the patient has already tried"
    )

    @field_validator(
        "symptoms", "body_location", "aggravating_factors", "relieving_factors",
        "associated_symptoms", "occupational_exposure", "recent_exposures", "prior_treatments",
        mode="before",
    )
    @classmethod
    def coerce_null_to_list(cls, v):
        return _null_to_list(v)
```

> **LLM null-safety:** Even though the prompt instructs the LLM to use `[]` for empty lists, local models sometimes return `null` instead. The `@field_validator` with `mode="before"` catches this before Pydantic type-checks the value, converting `null → []`. Without it, Pydantic raises a `ValidationError` and the entire crew crashes.

> **Design consideration:** Why preserve `patient_description` verbatim?  
> Clinical NLP sometimes loses nuance. A patient saying "it looks like someone threw acid on me" is important context that clinical terminology cannot capture. Always keep the patient's own words alongside the structured extraction.

---

## Step 2 — Build the Agent and Task

Add below the schema:

```python
def create_decomposition_agent() -> Agent:
    return Agent(
        role="Clinical Symptom Decomposition Specialist",
        goal=(
            "Extract every clinically relevant detail from the patient's text. "
            "Map lay language to dermatology terminology. "
            "Produce only structured data — no commentary, no diagnosis."
        ),
        backstory=(
            "You are a clinical dermatology informaticist. "
            "Your only job is to read the patient's words and extract structured clinical data. "
            "'Itchy' is pruritus. 'Crusty' is impetiginization. 'Ring-shaped' is annular. "
            "Durations become days (1 week = 7, 1 month = 30). "
            "You extract only what the patient stated. You never invent symptoms."
        ),
        llm=ORCHESTRATOR_LLM,
        verbose=True,
    )


def create_decomposition_task(
    agent: Agent,
    patient_text: str,
    biodata_task=None,
) -> Task:
    """
    Args:
        patient_text: The raw text input from the patient describing their symptoms.
        biodata_task: Optional — if provided, agent gets patient occupation/history context.
    """
    context = [biodata_task] if biodata_task else []

    return Task(
        description=(
            "Extract all clinical information from the following patient statement:\n\n"
            f'"{patient_text}"\n\n'
            "Cross-reference with the patient biodata for occupational context if available.\n"
            "Convert all durations to days (1 week = 7, 1 month = 30).\n"
            "Use clinical terminology for symptoms where appropriate.\n"
            "Do NOT invent any information not present in the patient's text."
        ),
        expected_output=(
            "A JSON object containing all extracted clinical fields. "
            "Only include fields where information was actually mentioned. "
            "For list fields with no data, use empty lists []. "
            "For optional string fields with no data, use null."
        ),
        agent=agent,
        output_pydantic=DecompositionOutput,
        context=context,
    )
```

> **Duration conversion:** The agent converts all time expressions to days. The prompt and backstory explicitly include: `1 day = 1, 1 week = 7, 1 month = 30, 1 year = 365`. The examples (`'4 years' → 1460`) are included because without them, local models occasionally output `null` for year-unit durations. Always include the unit table and at least one year-based example.

> **Model note:** The Decomposition agent uses `ORCHESTRATOR_LLM` (qwen2.5:7b) — not MedGemma. qwen2.5:7b is the top-ranked local model for strict JSON schema compliance. It is significantly better than gemma3 at distinguishing input context data from output schema fields, which prevents the model from accidentally copying upstream agent fields into its own output. `qwen3:8b` was evaluated but its thinking-mode tokens caused tool-call parsing failures in CrewAI.

---

## Step 3 — Test the Decomposition Agent

Create **`test_decomposition.py`** in the project root:

```python
# test_decomposition.py
from crewai import Crew, Process
from agents.biodata_agent import create_biodata_agent, create_biodata_task
from agents.decomposition_agent import create_decomposition_agent, create_decomposition_task

# Test with several different patient texts to see how the agent handles them

TEST_CASES = [
    # Case 1: Rich detail
    (
        "Case 1 — Rich detail",
        "I've had this really itchy, bumpy red rash on my left forearm for about 4 days. "
        "It started as a small patch near my wrist and has been spreading upward. "
        "It gets much worse in the evenings and when I'm at work. I work as a painter "
        "and I recently started using a new type of solvent. I tried some calamine lotion "
        "but it didn't help much."
    ),
    # Case 2: Minimal info
    (
        "Case 2 — Minimal info",
        "I have a rash on my back. It's been there for a while."
    ),
    # Case 3: Multiple symptoms
    (
        "Case 3 — Multiple symptoms",
        "I have circular patches on both my elbows that are very scaly and silver-coloured. "
        "They've been there for about 3 months. My joints also ache, especially my knees. "
        "My father had similar skin patches. The patches get worse in winter."
    ),
]

biodata_agent = create_biodata_agent()
biodata_task  = create_biodata_task(biodata_agent)
decomp_agent  = create_decomposition_agent()

for label, patient_text in TEST_CASES:
    print(f"\n{'='*60}")
    print(f"TESTING: {label}")
    print(f"{'='*60}")
    print(f"INPUT: {patient_text[:80]}...")

    decomp_task = create_decomposition_task(decomp_agent, patient_text, biodata_task)

    crew = Crew(
        agents=[biodata_agent, decomp_agent],
        tasks=[biodata_task, decomp_task],
        process=Process.sequential,
        verbose=False,   # set to True if you want to see the thinking
    )

    result = crew.kickoff()
    print("\nOUTPUT:")
    if decomp_task.output and decomp_task.output.pydantic:
        print(decomp_task.output.pydantic.model_dump_json(indent=2))
    else:
        print(decomp_task.output.raw if decomp_task.output else "No output")
```

Run it:
```powershell
python test_decomposition.py
```

**What to check for each case:**

**Case 1:**
- `symptoms` contains at least ["pruritus", "erythema"] or similar clinical terms
- `time_days` is `4`
- `progression` is "spreading"
- `aggravating_factors` includes "evenings" and something about work/chemical
- `occupational_exposure` mentions painter/solvent
- `prior_treatments` mentions calamine lotion
- `patient_description` preserves their own words

**Case 2:**
- `symptoms` is short — only what was mentioned
- `time_days` is `null` (patient said "a while" — agent shouldn't guess)
- `body_location` is "back"
- Most fields are empty lists or null

**Case 3:**
- `body_location` is "bilateral elbows"
- `symptoms` includes terms for scaly plaques and joint pain
- `time_days` is approximately 90
- `aggravating_factors` includes "winter"
- `associated_symptoms` mentions joint pain

**If the agent invents information:** This is hallucination. Make the `description` more explicit: add "Do NOT add any symptom or detail not explicitly stated by the patient."

---

## Step 4 — Understanding How This Feeds the Research Agent

The Decomposition Output drives the Research Agent's PubMed queries. Here's the logic the Research Agent will use:

```python
# The Research Agent will build queries like:
query = f"{primary_diagnosis_candidate} {symptoms[0]} {body_location}"
# e.g., "contact dermatitis pruritus forearm painter solvent"
```

By making the decomposition structured, you ensure the research queries are targeted and relevant — not generic searches like "skin rash".

---

## Checkpoint ✅

- [ ] `agents/decomposition_agent.py` exists with `DecompositionOutput`, agent, and task factory
- [ ] Case 1 (rich) extracts most fields correctly, including occupational exposure
- [ ] Case 2 (minimal) does NOT invent information — missing fields are null/empty
- [ ] Case 3 (complex) extracts joint symptoms into `associated_symptoms` correctly
- [ ] `patient_description` field preserves the patient's original phrasing
- [ ] Duration conversions work (weeks → days)
- [ ] `body_location` and `occupational_exposure` are typed as `list[str]` (not `Optional[str]`)
- [ ] `@field_validator` with `mode="before"` is applied to all list fields to coerce `null → []`

Delete the test file:
```powershell
Remove-Item test_decomposition.py
```

---

## What Comes Next — Clarification Agent

The Decomposition Agent extracts everything the patient *did* say. But what about what they *didn't* say?

A patient might write *"I have a rash"* — technically a valid input, but almost useless for diagnosis. The **Clarification Agent** (Chapter 06b) reviews the decomposition output, detects critically missing fields, and generates plain-language follow-up questions for the patient.

Key points:
- It runs immediately after this agent, before the Research Agent
- It uses the same `VISION_LLM` (no tool calls needed — just clinical gap analysis)
- It triggers at most **2 rounds** of follow-up questions to avoid overwhelming the patient
- If the patient's original input was complete, it produces `needs_clarification = False` and the pipeline proceeds immediately — zero friction

**→ Continue to `06b_CLARIFICATION_AGENT.md` before moving to `07_PUBMED_TOOL.md`**

---

*Next → `06b_CLARIFICATION_AGENT.md`*
