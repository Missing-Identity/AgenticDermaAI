# Chapter 04 — The Profile / Biodata Agent

**Goal:** Build the patient profile system (a JSON file + a loading mechanism) and the Biodata Agent that serves patient context to every other agent on demand. Test it standalone before anything else depends on it.

**Time estimate:** 30–40 minutes

---

## Why This Agent Comes First

Every other agent may need patient context:
- The Lesion Texture Agent needs age and sex to contextualise findings
- The Decomposition Agent may need occupation to flag occupational exposures
- The Orchestrator needs the full profile to reason about differential diagnoses

The Biodata Agent is the **single source of truth** for patient demographics. We build and test it first so the data contract is solid before downstream agents depend on it.

---

## Step 1 — Design the Patient Profile Schema

Before writing any code, design what information you want to capture. Open a new file:

**`agents/biodata_agent.py`**

Start with the data model at the top of the file. As you type each field, think about why a dermatologist would need it:

```python
# agents/biodata_agent.py

import json
import os
from pydantic import BaseModel, Field
from typing import Optional
from crewai import Agent, Task
from config import TEXT_LLM


class PatientProfile(BaseModel):
    """Structured patient biodata. Every field is optional — not all patients provide all data."""

    name: str = Field(default="Unknown", description="Patient's full name")
    age: Optional[int] = Field(default=None, description="Age in years")
    sex: Optional[str] = Field(default=None, description="Biological sex: Male/Female/Other")
    gender: Optional[str] = Field(default=None, description="Gender identity if different from sex")
    skin_tone: Optional[str] = Field(
        default=None,
        description="Self-reported skin tone: very light, light, medium, medium-dark, dark, very dark"
    )
    occupation: Optional[str] = Field(default=None, description="Current occupation")
    caste: Optional[str] = Field(default=None, description="Caste/ethnicity (relevant for genetic skin conditions)")
    pincode: Optional[str] = Field(default=None, description="Area pincode (for geographic disease patterns)")
    known_allergies: Optional[list[str]] = Field(default=None, description="Known allergies")
    current_medications: Optional[list[str]] = Field(default=None, description="Current medications")
    past_skin_conditions: Optional[list[str]] = Field(default=None, description="Previous skin conditions")
    family_skin_history: Optional[str] = Field(default=None, description="Family history of skin conditions")
    notes: Optional[str] = Field(default=None, description="Any other relevant context")
```

> **Why `Optional` everywhere?** You don't want the system to fail if a patient hasn't provided their occupation. Defaults allow partial profiles to work correctly. The agent will simply say "not provided" for missing fields.

---

## Step 2 — Build the Profile Loader

Add this below the `PatientProfile` class in the same file:

```python
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "patient_profile.json")


def load_profile() -> PatientProfile:
    """
    Load the patient profile from patient_profile.json.
    If the file does not exist, return an empty profile.
    """
    if not os.path.exists(PROFILE_PATH):
        return PatientProfile()

    with open(PROFILE_PATH, "r") as f:
        data = json.load(f)

    return PatientProfile(**data)


def save_profile(profile: PatientProfile) -> None:
    """Save the patient profile to patient_profile.json."""
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile.model_dump(), f, indent=2)
    print(f"Profile saved to {PROFILE_PATH}")


def profile_to_context_string(profile: PatientProfile) -> str:
    """
    Convert the profile to a formatted string for injection into agent prompts.
    Only include fields that have actual values.
    """
    lines = ["PATIENT BIODATA:"]

    if profile.name != "Unknown":
        lines.append(f"  Name: {profile.name}")
    if profile.age:
        lines.append(f"  Age: {profile.age} years")
    if profile.sex:
        lines.append(f"  Sex: {profile.sex}")
    if profile.skin_tone:
        lines.append(f"  Skin Tone: {profile.skin_tone}")
    if profile.occupation:
        lines.append(f"  Occupation: {profile.occupation}")
    if profile.caste:
        lines.append(f"  Ethnicity/Caste: {profile.caste}")
    if profile.pincode:
        lines.append(f"  Location (Pincode): {profile.pincode}")
    if profile.known_allergies:
        lines.append(f"  Known Allergies: {', '.join(profile.known_allergies)}")
    if profile.current_medications:
        lines.append(f"  Current Medications: {', '.join(profile.current_medications)}")
    if profile.past_skin_conditions:
        lines.append(f"  Past Skin Conditions: {', '.join(profile.past_skin_conditions)}")
    if profile.family_skin_history:
        lines.append(f"  Family Skin History: {profile.family_skin_history}")
    if profile.notes:
        lines.append(f"  Notes: {profile.notes}")

    if len(lines) == 1:
        lines.append("  No biodata provided.")

    return "\n".join(lines)
```

---

## Step 3 — Build the Patient Setup CLI

Now create a separate file for the interactive profile setup. Users run this once per session (or once per patient):

**`patient_setup.py`** (in the project root):

```python
# patient_setup.py
# Run this to set up patient biodata before starting a diagnosis session.
# python patient_setup.py

from agents.biodata_agent import PatientProfile, save_profile


def prompt(label: str, hint: str = "") -> str:
    """Helper to ask a question and return stripped input."""
    if hint:
        label = f"{label} ({hint})"
    return input(f"{label}: ").strip()


def prompt_list(label: str) -> list[str]:
    """Ask for comma-separated values, return as a list."""
    raw = input(f"{label} (comma-separated, or press Enter to skip): ").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def collect_profile() -> PatientProfile:
    print("\n" + "="*55)
    print("  DermaAI v2 — Patient Profile Setup")
    print("  All fields are optional. Press Enter to skip any.")
    print("="*55 + "\n")

    profile = PatientProfile(
        name=prompt("Patient Name", "or press Enter for Anonymous") or "Anonymous",
        age=int(a) if (a := prompt("Age", "years")) else None,
        sex=prompt("Biological Sex", "Male / Female / Other") or None,
        gender=prompt("Gender Identity", "if different from sex, else Enter") or None,
        skin_tone=prompt(
            "Skin Tone",
            "very light / light / medium / medium-dark / dark / very dark"
        ) or None,
        occupation=prompt("Occupation") or None,
        caste=prompt("Ethnicity / Caste") or None,
        pincode=prompt("Area Pincode") or None,
        known_allergies=prompt_list("Known Allergies"),
        current_medications=prompt_list("Current Medications"),
        past_skin_conditions=prompt_list("Past Skin Conditions"),
        family_skin_history=prompt("Family History of Skin Conditions") or None,
        notes=prompt("Any other relevant notes") or None,
    )

    return profile


def main():
    profile = collect_profile()

    print("\n--- Profile collected ---")
    print(profile.model_dump_json(indent=2))

    confirm = input("\nSave this profile? (y/n): ").strip().lower()
    if confirm == "y":
        save_profile(profile)
        print("Ready. Run your diagnosis session now.")
    else:
        print("Profile discarded.")


if __name__ == "__main__":
    main()
```

**Test it now:**
```powershell
python patient_setup.py
```

Fill in some details. When you confirm with `y`, a `patient_profile.json` file should appear in your project root.

Open `patient_profile.json` and verify the JSON is correct. You should see exactly the fields you entered.

---

## Step 4 — Build the Biodata Agent

Add this to the bottom of `agents/biodata_agent.py`:

```python
def create_biodata_agent() -> Agent:
    """
    Creates the Biodata Agent.
    This agent loads and serves the patient profile to other agents.
    It does NOT analyse anything — it only holds and reports data.
    """
    profile = load_profile()
    context_string = profile_to_context_string(profile)

    return Agent(
        role="Patient Profile Specialist",
        goal=(
            "Accurately report patient biodata to other agents on request. "
            "Never invent or guess missing information — report it as 'not provided'."
        ),
        backstory=(
            "You are a clinical data administrator at a dermatology clinic. "
            "You maintain precise patient records and provide them to the medical team "
            "whenever needed. You only share what is documented.\n\n"
            f"{context_string}"
        ),
        llm=TEXT_LLM,
        verbose=True,
    )


def create_biodata_task(agent: Agent) -> Task:
    """
    Task that makes the biodata agent summarise the patient profile.
    Other agents use this task's output as context.
    """
    return Task(
        description=(
            "Summarise the patient's biodata in a structured format. "
            "Include all fields that have values. For any field not provided, "
            "state 'not provided'. Do not add any assumptions or inferences."
        ),
        expected_output=(
            "A structured summary of the patient profile with clearly labelled fields. "
            "Each field on its own line. Example:\n"
            "Name: Ravi Kumar\n"
            "Age: 34 years\n"
            "Sex: Male\n"
            "Skin Tone: medium-dark\n"
            "Occupation: Farmer\n"
            "..."
        ),
        agent=agent,
    )
```

---

## Step 5 — Test the Biodata Agent Standalone

Create a test file `test_biodata.py` in your project root:

```python
# test_biodata.py
from crewai import Crew, Process
from agents.biodata_agent import create_biodata_agent, create_biodata_task

def main():
    print("\nTesting Biodata Agent...")

    agent = create_biodata_agent()
    task = create_biodata_task(agent)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "="*50)
    print("BIODATA AGENT OUTPUT:")
    print("="*50)
    print(result)

if __name__ == "__main__":
    main()
```

Run it:
```powershell
python test_biodata.py
```

**What to verify:**
- The agent output lists the same fields you entered during setup
- No invented or hallucinated information appears
- Missing fields say "not provided" (not blank, not a guess)

**Experiment — test with empty profile:**  
Delete `patient_profile.json` and run `test_biodata.py` again. All fields should say "not provided." Then re-run `patient_setup.py` to recreate it.

---

## Step 6 — Understanding How Other Agents Will Use Biodata

In later chapters, when a lesion agent needs to know the patient's skin tone or age, it will declare the biodata task as its `context`:

```python
texture_task = Task(
    description="Analyse the lesion texture, considering the patient's age and sex...",
    expected_output="...",
    agent=texture_agent,
    context=[biodata_task],   # ← this injects the biodata summary automatically
)
```

CrewAI will prepend the biodata task's output to the texture task's prompt. The texture agent then has full access to patient context without needing to query a database or make a separate call.

This is the "shared memory" pattern for sequential crews.

---

## Checkpoint ✅

- [ ] `agents/biodata_agent.py` exists with `PatientProfile`, `load_profile`, `save_profile`, `create_biodata_agent`, `create_biodata_task`
- [ ] `patient_setup.py` exists in the project root
- [ ] Running `python patient_setup.py` creates `patient_profile.json`
- [ ] `patient_profile.json` contains the exact data you entered
- [ ] Running `python test_biodata.py` shows the profile through the agent without hallucination
- [ ] Running with no profile file produces only "not provided" values

Delete the test file when done:
```powershell
Remove-Item test_biodata.py
```

---

*Next → `05_LESION_AGENTS.md`*
