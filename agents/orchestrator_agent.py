# The Orchestrator module: synthesises all agent outputs into a final clinical assessment.
# Split into CMO (Reasoning) and Scribe (Reporting)

import os
from typing import Any, Literal
from pydantic import Field, field_validator, model_validator
from crewai import Agent, Task
from config import ORCHESTRATOR_LLM
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
    re_diagnosis_applied: bool = Field(default=False)
    re_diagnosis_reason: str = Field(default="")

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

# ── 3. CMO Agent ──────────────────────────────────────────────────────────────

def create_cmo_agent() -> Agent:
    return Agent(
        role="Chief Medical Officer (Dermatology)",
        goal=(
            "Review all evidence (visual, text, differentials, mimic resolution, research) "
            "and make the final, authoritative clinical decision on the diagnosis."
        ),
        backstory=(
            "You are the Chief Medical Officer. "
            "You review outputs from all specialist agents and make the final authoritative diagnosis. "
            "Your job is purely clinical validation: cross-check the proposed diagnosis against "
            "all visual, historical, and research evidence, then confirm or correct it. "
            "You do not write patient letters — only strict medical reasoning."
        ),
        llm=ORCHESTRATOR_LLM,
        verbose=True,
    )

def create_cmo_task(
    agent: Agent,
    biodata_task=None,
    decomposition_task=None,
    research_task=None,
    differential_task=None,
    mimic_task=None,
    lesion_summary: str = "",
    confirmed_diagnosis: str = "",
    # Legacy keyword args accepted but ignored
    visual_verdict_summary: str = "",
    colour_task=None,
    texture_task=None,
    levelling_task=None,
    border_task=None,
    shape_task=None,
) -> Task:
    # The Debate Resolver has already picked the authoritative diagnosis from the image.
    # The CMO's role is now to build the clinical output around that confirmed diagnosis,
    # not to re-arbitrate between text agents and vision signals.
    context = [
        t for t in [biodata_task, decomposition_task, research_task, differential_task, mimic_task]
        if t is not None
    ]

    doctor_feedback = os.getenv("DOCTOR_FEEDBACK", "").strip()
    feedback_block = ""
    if doctor_feedback:
        feedback_block = (
            f"DOCTOR FEEDBACK FROM PREVIOUS RUN:\n"
            f"   \"{doctor_feedback}\"\n\n"
            f"The doctor has provided feedback. You MUST address it — update the diagnosis "
            f"or reasoning as needed and set re_diagnosis_applied = true in your output.\n\n"
        )

    lesion_block = f"{lesion_summary}\n\n" if lesion_summary else ""

    if confirmed_diagnosis:
        diagnosis_block = (
            f"CONFIRMED DIAGNOSIS (visual debate resolver — authoritative):\n"
            f"   {confirmed_diagnosis}\n\n"
            f"This diagnosis was selected by direct image analysis. "
            f"Do NOT override it unless the doctor feedback above explicitly requires a change.\n\n"
        )
        instructions = (
            "Your task is to build the clinical output for the confirmed diagnosis above.\n\n"
            "1. Set primary_diagnosis to the confirmed diagnosis exactly as written.\n"
            "2. Use the lesion visual summary, patient history, demographics, and research evidence "
            "to construct the clinical reasoning that supports this diagnosis.\n"
            "3. Assign appropriate confidence and severity based on the evidence.\n"
            "4. List suggested investigations and cited PMIDs.\n"
            "5. Set re_diagnosis_applied = false unless doctor feedback requires a change."
        )
    else:
        diagnosis_block = ""
        instructions = (
            "Review all specialist agent outputs and make the final diagnostic decision.\n\n"
            "1. Read the lesion visual summary above first — morphology is the strongest evidence.\n"
            "2. Identify the diagnosis best supported by all available evidence.\n"
            "3. Assign confidence, severity, investigations, and PMIDs.\n"
            "4. Set re_diagnosis_applied = true if you correct the text agents' proposed diagnosis."
        )

    return Task(
        description=(
            feedback_block +
            lesion_block +
            diagnosis_block +
            instructions
        ),
        expected_output=(
            "A concise free-text final clinical decision including: primary diagnosis, confidence, "
            "severity, lesion profile summary, clinical reasoning, re_diagnosis_applied (true/false), "
            "suggested investigations, and cited PMIDs."
        ),
        agent=agent,
        context=context,
    )

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
        llm=ORCHESTRATOR_LLM,
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
            "Your job is to compile the FinalDiagnosis report.\n"
            "1. Copy primary_diagnosis, severity, confidence, clinical_reasoning, "
            "suggested_investigations, and cited_pmids DIRECTLY from the CMO output — do not alter them.\n"
            "2. Write a compassionate, jargon-free patient_summary (2-3 sentences).\n"
            "3. Extract actionable patient_recommendations (list of strings) from the Treatment Plan.\n"
            "4. Write technical doctor_notes combining the CMO's reasoning with the Treatment Plan protocol.\n"
            "5. Summarize literature_support from the research task.\n"
            "6. Write when_to_seek_care with clear, specific guidance."
        ),
        expected_output=(
            "Output a single flat JSON object with these exact top-level keys: "
            "primary_diagnosis, confidence, severity, lesion_profile, clinical_reasoning, "
            "suggested_investigations, cited_pmids, patient_summary, patient_recommendations, "
            "doctor_notes, treatment_suggestions, literature_support, when_to_seek_care.\n"
            "CRITICAL: Do NOT wrap the output inside a 'FinalDiagnosis' key or any other wrapper. "
            "All fields must be at the top level of the JSON object."
        ),
        agent=agent,
        context=context,
    )