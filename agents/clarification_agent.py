# Reviews the Decomposition output for critical missing fields.
# Generates targeted follow-up questions for the patient.
# Uses VISION_LLM — clinical reasoning, no tool calls needed.

from pydantic import Field, field_validator, model_validator
from crewai import Agent, Task
from config import VISION_LLM
from utils.resilient_base import ResilientBase


class ClarificationOutput(ResilientBase):
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
            "A concise free-text clarification decision including: whether clarification is needed, "
            "which fields are missing, and up to 3 patient-facing follow-up questions if needed."
        ),
        agent=agent,
        context=context,
    )