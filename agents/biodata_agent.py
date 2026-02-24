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