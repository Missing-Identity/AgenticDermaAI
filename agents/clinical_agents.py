# Differential Diagnosis Agent + Treatment Plan Agent
# These run after the Research Agent and before the Orchestrator.
#
# Model assignment:
#   ORCHESTRATOR_LLM (Qwen 2.5, num_ctx=16384) — Differential, Mimic, Treatment, CMO.
#   These agents receive 8-10 upstream task outputs as context and never process images.
#   MedGemma's ~2048 token context window would silently truncate most of that context,
#   causing the agents to reason with incomplete evidence. Qwen 2.5 with 16384 tokens
#   comfortably fits the full multi-agent context.

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from crewai import Agent, Task
from config import ORCHESTRATOR_LLM
from utils.resilient_base import ResilientBase


# ── Differential Diagnosis Schema ─────────────────────────────────────────────

class DifferentialEntry(BaseModel):
    """One alternative diagnosis candidate with full clinical justification."""

    condition: str = Field(
        default="",
        description="Name of the differential diagnosis condition"
    )
    probability: Literal["high", "moderate", "low"] = Field(
        default="moderate",
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
    def normalise_probability(cls, v) -> str:
        if v is None:
            return "moderate"
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

class MimicResolutionOutput(ResilientBase):
    """
    Output from the Mimic Resolution Agent comparing the top differentials.
    """
    primary_diagnosis_confirmed: str = Field(
        default="",
        description="The winning diagnosis after resolving the mimic conflict"
    )
    rejected_mimic: str = Field(
        default="",
        description="The closely mimicking condition that was rejected"
    )
    distinguishing_factor: str = Field(
        default="",
        description="The specific historical or visual detail that separated the two conditions"
    )
    mimic_reasoning: str = Field(
        default="",
        description="Clinical reasoning explaining why the primary won and the mimic was rejected"
    )

    @field_validator("primary_diagnosis_confirmed", "rejected_mimic", "distinguishing_factor", "mimic_reasoning", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""


# ── Agent 1: Differential Diagnosis ───────────────────────────────────────────

def create_differential_agent() -> Agent:
    return Agent(
        role="Dermatology Differential Diagnosis Specialist",
        goal=(
            "Produce a rigorous ranked differential diagnosis from all available clinical evidence. "
            "For each condition: state what fits, what does not fit, and what single test resolves it."
        ),
        backstory=(
            "You are a consultant dermatologist. "
            "You reason from the visual lesion findings, patient history, demographics, and research. "
            "You never anchor on the first condition that comes to mind. "
            "The shape and morphology of the lesion are your primary diagnostic signals — "
            "let the visual evidence lead, then refine with history and literature."
        ),
        llm=ORCHESTRATOR_LLM,
        verbose=True,
    )


def create_differential_task(
    agent: Agent,
    biodata_task=None,
    colour_task=None,
    texture_task=None,
    levelling_task=None,
    border_task=None,
    shape_task=None,
    pattern_task=None,
    decomposition_task=None,
    research_task=None,
    medgemma_anchor: str = "",
) -> Task:
    context = [
        t for t in [
            biodata_task, colour_task, texture_task,
            levelling_task, border_task, shape_task, pattern_task,
            decomposition_task, research_task,
        ]
        if t is not None
    ]

    anchor_block = ""
    if medgemma_anchor:
        anchor_block = (
            f"MEDGEMMA PRIMARY DIAGNOSIS (default — treat as correct unless evidence strongly contradicts):\n"
            f"   {medgemma_anchor}\n\n"
        )

    return Task(
        description=(
            anchor_block +
            "Using ALL upstream agent outputs — visual lesion findings (colour, surface, elevation, "
            "border, shape, pattern), patient biodata, decomposed symptoms, and research evidence — "
            "produce a ranked differential diagnosis.\n\n"

            "1. Identify the single most likely PRIMARY diagnosis from the combined evidence.\n"
            "2. Provide 2 to 4 differential alternatives. For each: note which findings support it, "
            "which argue against it, and what one test would confirm or rule it out.\n"
            "3. Flag any red flags that suggest malignancy, systemic involvement, or urgent referral.\n\n"

            "The lesion's shape, border, and morphology are your strongest objective signals — "
            "let them anchor your reasoning before layering in history and demographics. "
            "Do not diagnose malignancy definitively — flag it as requiring biopsy."
        ),
        expected_output=(
            "A concise free-text differential assessment including: primary diagnosis, confidence, "
            "primary reasoning, 2-4 differential alternatives with supporting/against findings, "
            "one distinguishing test per alternative, and any urgent red flags."
        ),
        agent=agent,
        context=context,
    )

# ── Agent 2: Mimic Resolution ─────────────────────────────────────────────────

def create_mimic_resolution_agent() -> Agent:
    return Agent(
        role="Clinical Mimic & Edge-Case Specialist",
        goal=(
            "Cross-examine the top differential diagnoses. "
            "Determine whether the primary diagnosis holds, or whether the closest alternative "
            "fits the evidence better. Do not introduce new diagnoses."
        ),
        backstory=(
            "You are a master diagnostician specialising in conditions that look alike. "
            "You use both visual morphology and patient history to arbitrate between competing diagnoses. "
            "Morphological evidence carries more weight than non-specific symptoms. "
            "No single feature is pathognomonic in isolation — always seek corroborating evidence."
        ),
        llm=ORCHESTRATOR_LLM,
        verbose=True,
    )

def create_mimic_resolution_task(
    agent: Agent,
    differential_task=None,
    colour_task=None,
    texture_task=None,
    levelling_task=None,
    border_task=None,
    shape_task=None,
    pattern_task=None,
    # biodata and decomposition already incorporated by Differential — excluded to reduce context
    biodata_task=None,
    decomposition_task=None,
    research_task=None,
    medgemma_anchor: str = "",
) -> Task:
    context = [
        t for t in [
            differential_task,
            colour_task, texture_task, levelling_task, border_task, shape_task, pattern_task,
            research_task,
        ]
        if t is not None
    ]

    anchor_block = ""
    if medgemma_anchor:
        anchor_block = (
            f"MEDGEMMA PRIMARY DIAGNOSIS (default — treat as correct unless evidence strongly contradicts):\n"
            f"   {medgemma_anchor}\n\n"
        )

    return Task(
        description=(
            anchor_block +
            "Using the Differential Diagnosis output and all visual lesion findings (colour, surface, "
            "elevation, border, shape):\n\n"

            "1. Identify the primary diagnosis and the highest-probability alternative (the mimic).\n"
            "2. List the specific morphological features that distinguish the two conditions.\n"
            "3. Note which features in this case favour the primary and which (if any) favour the mimic.\n\n"

            "IMPORTANT: Do NOT state a final confirmed diagnosis. Do NOT say which condition wins. "
            "Your role is comparison only — describing the distinguishing evidence. "
            "A separate image-based system will make the authoritative final call after your output."
        ),
        expected_output=(
            "A concise free-text comparison of the primary diagnosis vs its closest mimic: "
            "list the key distinguishing morphological features and note which features are "
            "present or absent in this case. Do not state a winner or confirmed diagnosis."
        ),
        agent=agent,
        context=context,
    )

# ── Agent 3: Treatment Plan ────────────────────────────────────────────────────

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
        llm=ORCHESTRATOR_LLM,
        verbose=True,
    )


def create_treatment_task(
    agent: Agent,
    biodata_task=None,
    research_task=None,
    differential_task=None,
    mimic_task=None,
) -> Task:
    context = [
        t for t in [biodata_task, research_task, differential_task, mimic_task]
        if t is not None
    ]

    return Task(
        description=(
            "Design a treatment plan for the CONFIRMED PRIMARY diagnosis "
            "(from Mimic Resolution if available, otherwise from Differential). "
            "Use the patient biodata and research evidence.\n\n"

            "1. IMMEDIATE ACTIONS — what the patient should do right now, tailored to the diagnosis.\n"
            "2. MEDICATION PROTOCOL — first-line, second-line, and escalation. "
            "For each: drug name, dose, route, frequency, duration, and monitoring.\n"
            "3. NON-PHARMACOLOGICAL INTERVENTIONS — lifestyle, skincare, trigger avoidance.\n"
            "4. PATIENT INSTRUCTIONS — plain-language guidance the patient can act on.\n"
            "5. FOLLOW-UP AND REFERRAL — when to review, what to assess, and whether specialist "
            "referral is needed.\n\n"

            "Check biodata for allergies and current medications — "
            "note any contraindications specific to this patient. "
            "Adjust for patient age where relevant."
        ),
        expected_output=(
            "A concise free-text treatment plan covering: diagnosis targeted, immediate actions, "
            "tiered medications, non-pharmacological care, patient instructions, follow-up, "
            "referral needs, contraindications, and evidence level."
        ),
        agent=agent,
        context=context,
    )