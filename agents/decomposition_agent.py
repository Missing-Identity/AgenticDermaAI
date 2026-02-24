# Extracts structured clinical information from patient free-text input.
# Uses ORCHESTRATOR_LLM (qwen2.5:7b) — strict schema compliance, reliable JSON output.
# No vision or tool calls needed here.

from typing import Optional
from pydantic import Field, field_validator
from crewai import Agent, Task
from config import ORCHESTRATOR_LLM
from utils.resilient_base import ResilientBase


def _null_to_list(v):
    """Coerce None → [] so the LLM can legally return null for empty list fields."""
    return v if v is not None else []


class DecompositionOutput(ResilientBase):
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
            "Map lay language to standard dermatological terminology. "
            "Convert all durations to days (1 week = 7, 1 month = 30, 1 year = 365). "
            "Extract ONLY what the patient stated — never invent symptoms or exposures."
        ),
        llm=ORCHESTRATOR_LLM,
        verbose=True,
    )


def create_decomposition_task(
    agent: Agent,
    patient_text: str,
    biodata_task=None,
    biodata_text: str = "",
) -> Task:
    """
    Args:
        patient_text:  The raw text input from the patient describing their symptoms.
        biodata_task:  Optional Task object — used by the main crew so downstream
                       tasks can reference this task as context.
        biodata_text:  Optional pre-formatted profile string (from profile_to_context_string).
                       Used by clarification rounds to avoid an extra LLM call.
                       Ignored when biodata_task is provided.
    """
    context = [biodata_task] if biodata_task else []

    # If no task object is available but a pre-formatted string was supplied,
    # embed it directly in the description so the agent still has patient context.
    inline_biodata = ""
    if not biodata_task and biodata_text:
        inline_biodata = f"\nPatient background (for occupational context):\n{biodata_text}\n"

    return Task(
        description=(
            "Extract all clinical information from the following patient statement:\n\n"
            f'"{patient_text}"\n'
            f"{inline_biodata}\n"
            "Cross-reference with the patient biodata for occupational context if available.\n"
            "Convert all durations to days\n"
            "Use clinical terminology for symptoms where appropriate.\n"
            "Do NOT invent any information not present in the patient's text."
        ),
        expected_output=(
            "A concise free-text structured extraction of symptoms/history covering: symptoms, "
            "duration in days, onset, progression, body locations, aggravating and relieving factors, "
            "associated symptoms, occupational/recent exposures, patient description, and prior treatments."
        ),
        agent=agent,
        context=context,
    )