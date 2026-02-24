from pydantic import Field, field_validator
from crewai import Agent, Task
from config import VISION_LLM
from utils.resilient_base import ResilientBase


# ── Output schemas ────────────────────────────────────────────────────────────

class ColourOutput(ResilientBase):
    lesion_colour: str = Field(description="Clinical colour description of the lesion")
    reason: str = Field(default="", description="Clinical reasoning behind the colour assessment")

    @field_validator("lesion_colour", "reason", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""


class SurfaceOutput(ResilientBase):
    surface: str = Field(description="Primary surface characteristic of the lesion")
    reason: str = Field(default="", description="Clinical reasoning with reference to patient age/sex if relevant")

    @field_validator("surface", "reason", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""


class LevellingOutput(ResilientBase):
    levelling: str = Field(description="Elevation of the lesion relative to surrounding skin")
    reason: str = Field(default="", description="Clinical reasoning for the elevation assessment")

    @field_validator("levelling", "reason", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("levelling", mode="before")
    @classmethod
    def normalise_levelling(cls, v: str) -> str:
        """Map free-text elevation descriptions to raised / flat / depressed."""
        if not isinstance(v, str):
            return v
        lower = v.lower()
        if any(w in lower for w in ("raised", "elevat", "dome", "papule", "nodule", "plaque", "verruc")):
            return "raised"
        if any(w in lower for w in ("depress", "atrophic", "pitted", "indented", "concave", "sunken")):
            return "depressed"
        if any(w in lower for w in ("flat", "macular", "macule", "level", "flush")):
            return "flat"
        return v


class BorderOutput(ResilientBase):
    border: str = Field(description="Clinical description of the lesion border and edge characteristics")
    reason: str = Field(default="", description="Clinical reasoning for the border assessment")

    @field_validator("border", "reason", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""


class ShapeOutput(ResilientBase):
    shape: str = Field(description="Clinical description of the lesion's geometric shape and outline")
    reason: str = Field(default="", description="Clinical reasoning for the shape assessment")

    @field_validator("shape", "reason", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""


class PatternOutput(ResilientBase):
    pattern: str = Field(description="Overall configuration and pattern of the lesion")
    reason: str = Field(default="", description="Clinical reasoning for the pattern assessment")

    @field_validator("pattern", "reason", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""


# ── Agent 1: Lesion Colour ────────────────────────────────────────────────────

def create_colour_agent() -> Agent:
    return Agent(
        role="Dermatology Colour Analyst",
        goal=(
            "Precisely describe the colour of the skin lesion in clinical terms, "
            "taking into account the patient's skin tone to assess colour contrast accurately."
        ),
        backstory=(
            "You are a specialist dermatologist with expertise in colour assessment "
            "across all skin tones. You know that erythema appears differently on dark "
            "skin than on light skin, and you adjust your clinical descriptions accordingly. "
            "You always use standard dermatology colour terminology: erythematous, "
            "violaceous, hyperpigmented, hypopigmented, melanotic, xanthomatous, etc."
        ),
        llm=VISION_LLM,
        verbose=True,
    )


def create_colour_task(
    agent: Agent,
    image_path: str,
    biodata_task=None,
    vision_result: str = None,
) -> Task:
    context = [biodata_task] if biodata_task else []

    if vision_result:
        description = (
            f"You directly examined the skin lesion image at: {image_path}\n\n"
            f"Your clinical observations from the image:\n{vision_result}\n\n"
            "Using your direct visual examination above and the patient's skin tone from biodata, "
            "provide your final clinical colour assessment using standard dermatology terminology. "
            "If the biodata context changes your interpretation (e.g. skin tone affects how erythema "
            "presents), note that explicitly."
        )
    else:
        description = (
            f"Examine the skin lesion image at: {image_path}\n\n"
            "You are directly analysing this image. "
            "Describe the colour of the lesion in clinical dermatology terms. "
            "Consider the patient's skin tone from the biodata context to assess colour contrast.\n"
            "Report the clinical colour of the lesion (not the surrounding skin)."
        )

    return Task(
        description=description,
        expected_output=(
            "A short free-text clinical colour assessment (2-4 sentences). "
            "Describe the lesion colour and briefly explain your reasoning. "
            "Do not use JSON or markdown."
        ),
        agent=agent,
        context=context,
    )


# ── Agent 2: Lesion Surface / Texture ─────────────────────────────────────────

def create_texture_agent() -> Agent:
    return Agent(
        role="Dermatology Texture and Lesion Surface Analyst",
        goal=(
            "Characterise the surface texture of the lesion with clinical precision. "
            "Account for patient age, occupation and sex when interpreting texture findings."
        ),
        backstory=(
            "You are a specialist in dermoscopy and lesion morphology. "
            "You know that scaling is more pronounced in elderly skin lesion due to reduced sebum production and in manual laborers due to exposure to harsh chemicals."
            "You know that depending on the occupation, the lesion may be more prone to scaling, blistering, crusting, weeping, or smoothness."
            "You know that based on age, the lesion may be more prone to scaling, blistering, crusting, weeping, or smoothness."
            "You know that blistering patterns differ between males and females for certain conditions. "
            "You cross-reference visual findings with patient demographics to give accurate assessments."
        ),
        llm=VISION_LLM,
        verbose=True,
    )


def create_texture_task(
    agent: Agent,
    image_path: str,
    biodata_task=None,
    vision_result: str = None,
) -> Task:
    context = [biodata_task] if biodata_task else []

    if vision_result:
        description = (
            f"You directly examined the skin lesion image at: {image_path}\n\n"
            f"Your clinical observations from the image:\n{vision_result}\n\n"
            "Using your direct visual examination above and the patient's age and sex from biodata, "
            "provide your final clinical surface assessment. "
            "If the patient's demographics alter the clinical significance of what you observed "
            "(e.g. scaling in elderly vs young), note that explicitly."
        )
    else:
        description = (
            f"Examine the skin lesion image at: {image_path}\n\n"
            "You are directly analysing this image. "
            "Look for: scaling, dryness, blistering, crusting, weeping, smoothness, "
            "or any other surface characteristic of the lesion or skin condition.\n"
            "Cross-reference with the patient's age and sex from biodata — "
            "e.g., scaling significance differs with age."
        )

    return Task(
        description=description,
        expected_output=(
            "A short free-text clinical surface/texture assessment (2-4 sentences). "
            "Describe the primary surface characteristic and brief reasoning. "
            "Do not use JSON or markdown."
        ),
        agent=agent,
        context=context,
    )


# ── Agent 3: Lesion Levelling ─────────────────────────────────────────────────

def create_levelling_agent() -> Agent:
    return Agent(
        role="Dermatology Morphology and Elevation Analyst",
        goal=(
            "Determine precisely whether the skin lesion is raised, flat, or depressed "
            "relative to surrounding skin, using visual cues and shadow analysis."
        ),
        backstory=(
            "You specialise in 3D morphological assessment of skin lesions. "
            "You use indirect visual cues like shadows, light reflection, and texture "
            "gradients to determine elevation from a 2D photograph. "
            "You know that hypertrophic scars are raised, atrophic scars are depressed, "
            "and macular lesions are flat. You verify your assessment against patient "
            "demographics — keloid formation is more common in certain ethnicities."
        ),
        llm=VISION_LLM,
        verbose=True,
    )


def create_levelling_task(
    agent: Agent,
    image_path: str,
    biodata_task=None,
    vision_result: str = None,
) -> Task:
    context = [biodata_task] if biodata_task else []

    if vision_result:
        description = (
            f"You directly examined the skin lesion image at: {image_path}\n\n"
            f"Your clinical observations from the image:\n{vision_result}\n\n"
            "Using your direct visual examination above and the patient's ethnicity from biodata, "
            "provide your final elevation assessment. "
            "If the patient's ethnicity is clinically relevant to your finding "
            "(e.g. keloid risk), note that explicitly."
        )
    else:
        description = (
            f"Examine the skin lesion image at: {image_path}\n\n"
            "You are directly analysing this image. "
            "Determine if the lesion is raised above, level with, or depressed below surrounding skin.\n"
            "Use visual cues: shadows at edges (raised), flat appearance (flat), "
            "central indentation or pit (depressed).\n"
            "Reference patient ethnicity from biodata when relevant to the finding."
        )

    return Task(
        description=description,
        expected_output=(
            "A short free-text elevation assessment (2-4 sentences). "
            "State whether the lesion appears raised, flat, or depressed and explain why. "
            "Do not use JSON or markdown."
        ),
        agent=agent,
        context=context,
    )


# ── Agent 4: Lesion Border ────────────────────────────────────────────────────

def create_border_agent() -> Agent:
    return Agent(
        role="Dermatology Border Analyst",
        goal=(
            "Evaluate the border and edge characteristics of the skin lesion in precise "
            "clinical terms, noting any features that carry diagnostic or prognostic significance."
        ),
        backstory=(
            "You are trained in dermoscopic border assessment, including the ABCDE criteria "
            "(Asymmetry, Border, Colour, Diameter, Evolution) used in melanoma screening. "
            "You focus exclusively on the periphery of the lesion — how the edge transitions "
            "to surrounding skin, the contour variation, any notching, and any red flags."
        ),
        llm=VISION_LLM,
        verbose=True,
    )


def create_border_task(
    agent: Agent,
    image_path: str,
    biodata_task=None,
    vision_result: str = None,
) -> Task:
    context = [biodata_task] if biodata_task else []

    if vision_result:
        description = (
            f"You directly examined the skin lesion image at: {image_path}\n\n"
            f"Your clinical observations from the image:\n{vision_result}\n\n"
            "Using your direct visual examination above, describe the border and edge "
            "characteristics of the lesion. Note the edge transition, contour variation, "
            "any notching or asymmetry, and any red flags for malignancy at the periphery."
        )
    else:
        description = (
            f"Examine the skin lesion image at: {image_path}\n\n"
            "You are directly analysing this image. "
            "Describe the border and edge characteristics of the lesion: "
            "how the lesion transitions to surrounding skin, the contour of the edge, "
            "and any asymmetry, notching, satellite lesions, or ABCDE red flags you observe."
        )

    return Task(
        description=description,
        expected_output=(
            "A short free-text border assessment (2-4 sentences). "
            "Describe edge characteristics and brief clinical reasoning. "
            "Do not use JSON or markdown."
        ),
        agent=agent,
        context=context,
    )


# ── Agent 5: Lesion Shape ─────────────────────────────────────────────────────

def create_shape_agent() -> Agent:
    return Agent(
        role="Dermatology Shape Analyst",
        goal=(
            "Determine the geometric shape and overall outline of the skin lesion in precise "
            "clinical terms, noting form, symmetry, and any structural features."
        ),
        backstory=(
            "You specialise in the morphological classification of skin lesions. "
            "You assess the overall geometry of a lesion — whether it is circular, oval, "
            "linear, annular, polycyclic, or has any other distinctive form. "
            "You also note overall symmetry and satellite lesions."
        ),
        llm=VISION_LLM,
        verbose=True,
    )


def create_shape_task(
    agent: Agent,
    image_path: str,
    biodata_task=None,
    vision_result: str = None,
) -> Task:
    context = [biodata_task] if biodata_task else []

    if vision_result:
        description = (
            f"You directly examined the skin lesion image at: {image_path}\n\n"
            f"Your clinical observations from the image:\n{vision_result}\n\n"
            "Using your direct visual examination above, describe the geometric shape "
            "and overall outline of the lesion. Note the overall form, symmetry, and "
            "any notable structural features."
        )
    else:
        description = (
            f"Examine the skin lesion image at: {image_path}\n\n"
            "You are directly analysing this image. "
            "Describe the geometric shape and overall outline of the lesion: "
            "its form (circular, oval, linear, annular, polycyclic, etc.), "
            "overall symmetry, and any distinctive structural characteristics."
        )

    return Task(
        description=description,
        expected_output=(
            "A short free-text shape assessment (2-4 sentences). "
            "Describe the lesion geometry/outline and brief clinical reasoning. "
            "Do not use JSON or markdown."
        ),
        agent=agent,
        context=context,
    )


# ── Agent 6: Lesion Pattern ───────────────────────────────────────────────────

def create_pattern_agent() -> Agent:
    return Agent(
        role="Dermatology Pattern Analyst",
        goal=(
            "Describe the overall configuration and pattern of the skin lesion. "
            "Identify distinctive arrangements that can point to specific diagnoses."
        ),
        backstory=(
            "You specialise in recognising lesion patterns and configurations. "
            "You assess whether the lesion is annular, bullseye/target-like, nummular, "
            "reticular, serpiginous, or has any other distinctive arrangement. "
            "Classic patterns — e.g. target-like morphology — can be pathognomonic. "
            "You describe what you see without forcing a diagnosis."
        ),
        llm=VISION_LLM,
        verbose=True,
    )


def create_pattern_task(
    agent: Agent,
    image_path: str,
    biodata_task=None,
    vision_result: str = None,
) -> Task:
    context = [biodata_task] if biodata_task else []

    if vision_result:
        description = (
            f"You directly examined the skin lesion image at: {image_path}\n\n"
            f"Your clinical observations from the image:\n{vision_result}\n\n"
            "Using your direct visual examination above, describe the overall pattern "
            "and configuration of the lesion. Note any distinctive arrangements "
            "that may have diagnostic significance."
        )
    else:
        description = (
            f"Examine the skin lesion image at: {image_path}\n\n"
            "You are directly analysing this image. "
            "Describe the overall pattern and configuration of the lesion: "
            "its arrangement (annular, bullseye/target-like, nummular, reticular, "
            "or other distinctive forms) and any classic morphologies you observe."
        )

    return Task(
        description=description,
        expected_output=(
            "A short free-text pattern assessment (2-4 sentences). "
            "Describe the lesion configuration and brief clinical reasoning. "
            "Do not use JSON or markdown."
        ),
        agent=agent,
        context=context,
    )
