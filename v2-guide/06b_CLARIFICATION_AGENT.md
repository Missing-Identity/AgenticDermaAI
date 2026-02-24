# Chapter 06b — The Clarification Agent

**Goal:** Build a Clarification Agent that reviews the Decomposition Agent's output, identifies critical missing clinical information, and generates targeted follow-up questions for the patient. Wire a clarification loop into the pipeline so the system can ask the patient up to 2 rounds of follow-up questions before proceeding.

**Time estimate:** 40–55 minutes

---

## Why Ask Follow-Up Questions?

A patient might write: *"I have a rash."*

That gives the Decomposition Agent almost nothing useful. Without body location, duration, or any contextual detail, the Research Agent will produce generic PubMed queries, the Differential Agent will have nothing to reason from, and the Orchestrator will produce a vague, low-confidence output.

Rather than proceeding with an incomplete picture, the system should pause after decomposition and ask the patient targeted follow-up questions — only asking for what's genuinely missing. If the patient gave complete information, no questions are asked and the pipeline proceeds immediately.

This is a **human-in-the-loop clarification** — similar in principle to the Doctor Approval Loop in Chapter 13, but on the patient input side and earlier in the pipeline.

---

## When Clarification Triggers

Clarification is triggered when the Decomposition output is missing **critical clinical fields**:

| Priority | Field | Why It Matters |
|---|---|---|
| Critical | `body_location` | Determines anatomo-clinical differential |
| Critical | `time_days` | Acute vs chronic changes the diagnosis entirely |
| Important | `onset` | Sudden vs gradual guides aetiology |
| Important | `progression` | Spreading vs stable is a red flag indicator |
| Situational | `occupational_exposure` | If symptoms suggest contact/occupational dermatitis |
| Situational | `aggravating_factors` | Helps confirm contact or environmental triggers |

> **Rule:** Ask at most **3 questions per round** to avoid overwhelming the patient. Prioritise critical fields over important ones.

---

## Pipeline Position

```
Patient text input
       │
       ▼
[Decomposition Agent]  ← extracts structured data
       │
       ▼
[Python guard]         ← checks body_location & time_days in Python
       │
  ┌────┴─────────────────────────────────────┐
  │ Both critical fields present?            │
  │                                          │
  ▼ No                                       ▼ Yes
[Clarification Agent]          Proceed to Research Agent
  │
  ├── needs_clarification = true  → Ask patient (max 3 questions)
  │                                 Append answers, re-run (max 2 rounds)
  └── needs_clarification = false → Proceed to Research Agent
```

> **Why the Python guard?** The Clarification Agent (VISION_LLM / MedGemma) can occasionally fail to read the structured decomposition context correctly and re-ask questions about fields that are already populated. The Python guard inspects the decomposition output directly — bypassing the LLM entirely when `body_location` and `time_days` are already present. This eliminates the double-questioning bug without changing the agent's prompts.

---

## Part A — The ClarificationOutput Schema and Agent

**`agents/clarification_agent.py`**

```python
# agents/clarification_agent.py
# Reviews the Decomposition output for critical missing fields.
# Generates targeted follow-up questions for the patient.
# Uses VISION_LLM — clinical reasoning, no tool calls needed.

from pydantic import BaseModel, Field, field_validator
from crewai import Agent, Task
from config import VISION_LLM


class ClarificationOutput(BaseModel):
    """
    Result of the Clarification Agent's gap analysis.
    If needs_clarification is False, questions will be empty and the pipeline proceeds.
    """

    needs_clarification: bool = Field(
        description=(
            "True if critical clinical information is missing and follow-up questions "
            "are needed before the diagnostic pipeline can proceed meaningfully."
        )
    )
    questions: list[str] = Field(
        default=[],
        description=(
            "Targeted follow-up questions for the patient. "
            "Maximum 3 questions per round. "
            "Empty list if needs_clarification is False."
        )
    )
    missing_fields: list[str] = Field(
        default=[],
        description=(
            "Names of the clinical fields that are missing or critically incomplete. "
            "e.g. ['body_location', 'time_days']. Empty if nothing is missing."
        )
    )
    reasoning: str = Field(
        default="",
        description=(
            "Brief explanation of why clarification is or is not needed. "
            "1-2 sentences. Used for the audit trail."
        )
    )

    @field_validator("questions", "missing_fields", mode="before")
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if v is not None else []

    @field_validator("needs_clarification", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        """Accept string 'true'/'false' in addition to actual booleans."""
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return v

    @field_validator("reasoning", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""
```

> **LLM null-safety:** Local models sometimes return `null` for empty list fields or wrap boolean values in quotes (`"true"` instead of `true`). The validators above silently normalise these before Pydantic type-checks them, preventing a `ValidationError` from crashing the clarification pre-pass.

```python
def create_clarification_agent() -> Agent:
    return Agent(
        role="Clinical Information Gap Analyst",
        goal=(
            "Review the structured clinical data extracted from the patient's statement. "
            "Identify which critical pieces of information are missing. "
            "Generate concise, plain-language follow-up questions to fill only the critical gaps. "
            "Never ask for information that was already provided."
        ),
        backstory=(
            "You are a clinical triage specialist. "
            "You read structured patient data and immediately spot what is missing. "
            "You ask the fewest possible questions to fill the most critical gaps. "
            "You speak to patients in plain, reassuring language — not clinical jargon. "
            "You never ask about things the patient already told you. "
            "If the data is sufficient for a diagnosis attempt, you say so and ask nothing."
        ),
        llm=VISION_LLM,
        verbose=True,
    )
```

---

## Part B — The Clarification Task

```python
def create_clarification_task(
    agent: Agent,
    decomposition_task,
    biodata_task=None,
) -> Task:
    """
    Args:
        decomposition_task: The completed Decomposition task (used as context).
        biodata_task: Optional — provides patient demographics as additional context.
    """
    context = [decomposition_task]
    if biodata_task:
        context.append(biodata_task)

    return Task(
        description=(
            "Review the structured clinical data extracted from the patient's statement (in context).\n\n"
            "Determine whether the following critical fields are present and meaningful:\n"
            "  1. body_location — at least one specific body area\n"
            "  2. time_days — how long the patient has had the condition\n"
            "  3. onset — how it started (sudden / gradual)\n"
            "  4. progression — whether it is spreading, stable, improving, or worsening\n\n"
            "Also check: if the symptoms suggest a contact or occupational cause "
            "(e.g. chemicals, new products, workplace exposure), is occupational_exposure populated?\n\n"
            "Rules:\n"
            "  - If ALL critical fields are present, set needs_clarification = false and stop.\n"
            "  - If any critical fields are missing, set needs_clarification = true.\n"
            "  - Generate at most 3 questions. Prioritise body_location and time_days first.\n"
            "  - Write questions in plain English, as if speaking to the patient directly.\n"
            "  - NEVER ask about something already provided in the decomposition data.\n"
            "  - Do NOT ask about diagnosis, treatment, or anything medical beyond what is needed."
        ),
        expected_output=(
            "A JSON object with four keys:\n"
            '"needs_clarification": true or false\n'
            '"questions": list of plain-English questions (empty list if no clarification needed)\n'
            '"missing_fields": list of field names that are missing (empty list if nothing missing)\n'
            '"reasoning": 1-2 sentences explaining your decision\n'
            "Example (clarification needed):\n"
            '{"needs_clarification": true, '
            '"questions": ["Where on your body is the rash?", "How many days have you had it?"], '
            '"missing_fields": ["body_location", "time_days"], '
            '"reasoning": "Body location and duration are missing — essential for narrowing the differential."}\n\n'
            "Example (no clarification needed):\n"
            '{"needs_clarification": false, "questions": [], "missing_fields": [], '
            '"reasoning": "All critical fields are present. Sufficient data to proceed."}'
        ),
        agent=agent,
        output_pydantic=ClarificationOutput,
        context=context,
    )
```

---

## Part C — Testing the Clarification Agent in Isolation

**`test_clarification.py`** (project root):

```python
# test_clarification.py
from crewai import Crew, Process
from agents.biodata_agent import create_biodata_agent, create_biodata_task
from agents.decomposition_agent import create_decomposition_agent, create_decomposition_task
from agents.clarification_agent import create_clarification_agent, create_clarification_task

print("\n" + "="*60)
print("TEST 1 — Sparse input (should trigger clarification)")
print("="*60)

sparse_text = "I have a rash."

biodata_agent  = create_biodata_agent()
biodata_task   = create_biodata_task(biodata_agent)
decomp_agent   = create_decomposition_agent()
decomp_task    = create_decomposition_task(decomp_agent, sparse_text, biodata_task)
clarif_agent   = create_clarification_agent()
clarif_task    = create_clarification_task(clarif_agent, decomp_task, biodata_task)

crew1 = Crew(
    agents=[biodata_agent, decomp_agent, clarif_agent],
    tasks=[biodata_task, decomp_task, clarif_task],
    process=Process.sequential,
    verbose=True,
)
result1 = crew1.kickoff()

output1 = clarif_task.output.pydantic
print(f"\nNeeds clarification: {output1.needs_clarification}")
print(f"Questions: {output1.questions}")
print(f"Missing fields: {output1.missing_fields}")
print(f"Reasoning: {output1.reasoning}")


print("\n" + "="*60)
print("TEST 2 — Complete input (should NOT trigger clarification)")
print("="*60)

complete_text = (
    "I've had this itchy, ring-shaped red rash on my right inner forearm "
    "for about 10 days. It started suddenly and has been spreading outward. "
    "It gets worse when I sweat. I recently started a new laundry detergent."
)

biodata_agent2 = create_biodata_agent()
biodata_task2  = create_biodata_task(biodata_agent2)
decomp_agent2  = create_decomposition_agent()
decomp_task2   = create_decomposition_task(decomp_agent2, complete_text, biodata_task2)
clarif_agent2  = create_clarification_agent()
clarif_task2   = create_clarification_task(clarif_agent2, decomp_task2, biodata_task2)

crew2 = Crew(
    agents=[biodata_agent2, decomp_agent2, clarif_agent2],
    tasks=[biodata_task2, decomp_task2, clarif_task2],
    process=Process.sequential,
    verbose=True,
)
result2 = crew2.kickoff()

output2 = clarif_task2.output.pydantic
print(f"\nNeeds clarification: {output2.needs_clarification}")
print(f"Questions: {output2.questions}")
print(f"Missing fields: {output2.missing_fields}")
print(f"Reasoning: {output2.reasoning}")
```

**What to verify for Test 1:**
- `needs_clarification` is `True`
- At least `body_location` appears in `missing_fields`
- Questions are plain English, not medical jargon

**What to verify for Test 2:**
- `needs_clarification` is `False`
- `questions` is an empty list
- `reasoning` confirms sufficient data

**Delete `test_clarification.py` when both tests pass.**

---

## Part D — The Clarification Loop Helper

Create a standalone helper function that encapsulates the "run → check → ask → repeat" loop. This function will be called from `main.py` and later wired into `DermaCrew`.

**`utils/clarification_loop.py`**

```python
# utils/clarification_loop.py
# Runs the Decomposition + Clarification mini-crew in a loop.
# Asks the patient follow-up questions if critical fields are missing.
# Returns the final patient_text (original + any Q&A appended) and the
# final DecompositionOutput once no further clarification is needed.

from crewai import Crew, Process
from agents.decomposition_agent import (
    create_decomposition_agent,
    create_decomposition_task,
    DecompositionOutput,
)
from agents.clarification_agent import (
    create_clarification_agent,
    create_clarification_task,
    ClarificationOutput,
)

MAX_ROUNDS = 2   # Never ask the patient more than 2 rounds of follow-up questions


def _ask_patient_questions(questions: list[str]) -> str:
    """Present the clarification questions to the patient and collect their answers."""
    print("\n  The AI needs a bit more information to give you an accurate diagnosis.")
    print("  Please answer the following questions:\n")

    answers = []
    for i, question in enumerate(questions, 1):
        answer = input(f"  Q{i}. {question}\n  Your answer: ").strip()
        if answer:
            answers.append(f"Q: {question}\nA: {answer}")

    return "\n".join(answers)


def run_clarification_loop(
    patient_text: str,
    biodata_agent,
    biodata_task,
) -> tuple[str, DecompositionOutput]:
    """
    Runs up to MAX_ROUNDS of Decomposition + Clarification.

    Returns:
        - enriched_patient_text: original text + any Q&A appended
        - final_decomposition: the last DecompositionOutput
    """
    enriched_text = patient_text
    final_decomposition = None

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n[Clarification] Round {round_num} — Extracting clinical data...")

        # Build mini-crew: Decomposition + Clarification only
        decomp_agent  = create_decomposition_agent()
        decomp_task   = create_decomposition_task(decomp_agent, enriched_text, biodata_task)
        clarif_agent  = create_clarification_agent()
        clarif_task   = create_clarification_task(clarif_agent, decomp_task, biodata_task)

        mini_crew = Crew(
            agents=[decomp_agent, clarif_agent],
            tasks=[decomp_task, clarif_task],
            process=Process.sequential,
            verbose=False,   # quiet — main pipeline will show its own output
        )

        try:
            mini_crew.kickoff()
        except Exception as e:
            # LLM output failed Pydantic validation (e.g. trailing JSON characters,
            # null for a list field, bad Literal value).  Grab whatever was parsed
            # and proceed rather than crashing the whole pipeline.
            print(f"[Clarification] Output parsing error: {e}. Proceeding with available data.")
            final_decomposition = getattr(decomp_task.output, "pydantic", None)
            break

        final_decomposition = decomp_task.output.pydantic
        clarif_result: ClarificationOutput = clarif_task.output.pydantic

        # Defensive: if Pydantic validation failed silently, proceed without blocking
        if clarif_result is None:
            print("[Clarification] Could not parse clarification output. Proceeding with available data.")
            break

        if not clarif_result.needs_clarification:
            print(f"[Clarification] Sufficient information. Proceeding with analysis.")
            break

        print(f"\n[Clarification] Missing: {', '.join(clarif_result.missing_fields)}")

        # Ask the patient and append answers to enriched_text
        qna = _ask_patient_questions(clarif_result.questions)
        if qna:
            enriched_text = enriched_text + "\n\nAdditional information:\n" + qna
            # Patient answered — proceed immediately, do not re-run another round
            break

        # Patient gave no answers — if this was the last allowed round, proceed anyway
        if round_num == MAX_ROUNDS:
            print(
                f"\n[Clarification] Maximum clarification rounds reached. "
                f"Proceeding with available information."
            )

    return enriched_text, final_decomposition
```

> **Why a separate `utils/` module?**
>
> The clarification loop is not a CrewAI agent or task — it's an *orchestration utility* that wraps a mini-crew. Keeping it in `utils/` prevents circular imports and makes it testable without running the full pipeline.

---

## Part E — Wiring into DermaCrew

In `crew/derma_crew.py` (Chapter 09), update `DermaCrew.run()` to call the clarification loop before the main crew runs:

```python
# At the top of derma_crew.py — add these imports:
from utils.clarification_loop import run_clarification_loop

# Inside DermaCrew.run(), replace the direct create_decomposition_task call with:

# ── Clarification loop (runs BEFORE the main crew) ────────────────────────────
# Biodata agent is needed for context; run it standalone first.
from crewai import Crew, Process
_biodata_agent = create_biodata_agent()
_biodata_task  = create_biodata_task(_biodata_agent)
Crew(
    agents=[_biodata_agent],
    tasks=[_biodata_task],
    process=Process.sequential,
    verbose=False,
).kickoff()

self.patient_text, pre_decomp = run_clarification_loop(
    patient_text=self.patient_text,
    biodata_agent=_biodata_agent,
    biodata_task=_biodata_task,
)
# ──────────────────────────────────────────────────────────────────────────────

# Then continue as before — create all agents and tasks for the main crew,
# using self.patient_text (now enriched if clarification happened).
# Note: biodata_agent and biodata_task are re-created fresh for the main crew
# (the ones above were only for the clarification pre-pass).
```

> **Why re-create biodata agent/task for the main crew?**
>
> CrewAI task objects carry state from their previous execution. Re-creating them ensures the main crew's biodata task is clean and gets properly output-linked to the downstream agents that depend on it.

---

## Part F — Wiring into main.py

In `main.py` (Chapter 10), the `get_patient_text()` function prompts the patient for their initial description. The clarification loop runs automatically inside `DermaCrew.run()` — no changes needed to `main.py`.

However, you should update `main.py` to tell the patient upfront that the system may ask them follow-up questions:

```python
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
```

---

## Part G — Audit Trail

When you build the `AuditTrail` in Chapter 13, make sure to capture the clarification outcome. Add these fields to the `AuditTrail` dataclass:

```python
@dataclass
class AuditTrail:
    # ... existing fields ...

    # Clarification
    clarification_triggered: bool = False
    clarification_rounds: int = 0
    clarification_questions_asked: list[str] = field(default_factory=list)
    original_patient_text: str = ""
    enriched_patient_text: str = ""   # same as original if no clarification happened
```

Update `DermaCrew.run()` to populate these fields from the return value of `run_clarification_loop()`.

---

## Checkpoint ✅

- [ ] `agents/clarification_agent.py` exists with `ClarificationOutput`, agent, and task factory
- [ ] Test 1 (sparse "I have a rash") triggers `needs_clarification = True` with at least 1 question
- [ ] Test 2 (complete text) produces `needs_clarification = False` with empty `questions`
- [ ] Questions in Test 1 are written in plain English, not medical jargon
- [ ] `utils/clarification_loop.py` exists and correctly appends Q&A to `patient_text`
- [ ] `mini_crew.kickoff()` is wrapped in `try/except` — a bad LLM parse does NOT crash the pipeline
- [ ] The loop exits cleanly after `MAX_ROUNDS = 2` even if clarification is still needed
- [ ] `DermaCrew.run()` calls `run_clarification_loop()` before the main crew starts
- [ ] `main.py` informs the patient that follow-up questions may be asked
- [ ] The `AuditTrail` captures whether clarification was triggered and what was asked

---

*Next → `07_PUBMED_TOOL.md`*
