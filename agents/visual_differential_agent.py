# Visual Differential Review Agent
#
# Contains two vision-based arbitration strategies:
#
#   run_debate_resolver() [PRIMARY — used by derma_crew.py]
#     Sends a single MedGemma call with the full candidate list and the image.
#     MedGemma picks ONE winner directly. No per-candidate calls, no synthesis LLM.
#     The winner is authoritative and bypasses CMO arbitration.
#
#   run_visual_differential_review() [LEGACY — kept for reference]
#     The original approach: one MedGemma call per candidate (YES/NO), then
#     a formatter LLM synthesises the votes into a structured winner.
#     This was replaced because the synthesis step re-introduced text-agent bias.
#
# Both functions run BETWEEN the two crew phases (after Mimic Resolution, before CMO)
# because MedGemma does not support CrewAI's OpenAI-style tool-calling format and
# the differential candidates are only known after the Differential task completes.

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from tools.image_tool import ImageAnalysisTool
from utils.resilient_base import ResilientBase
from utils.schema_adapter import adapt_to_model


# ── Schema ────────────────────────────────────────────────────────────────────

class VisualDifferentialVote(BaseModel):
    """One image-based assessment for a single diagnosis candidate."""

    condition: str = Field(
        default="",
        description="The diagnosis candidate being assessed",
    )
    visually_consistent: bool = Field(
        default=False,
        description="True if the lesion image is consistent with this condition",
    )
    confidence: Literal["high", "moderate", "low"] = Field(
        default="low",
        description="Confidence in this visual assessment",
    )
    visual_reasoning: str = Field(
        default="",
        description=(
            "1-2 sentences citing specific morphological features "
            "(colour, border, shape, texture, elevation) that support or refute this condition"
        ),
    )

    @field_validator("condition", "visual_reasoning", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("confidence", mode="before")
    @classmethod
    def normalise_confidence(cls, v):
        if isinstance(v, str):
            lower = v.lower()
            if "high" in lower:
                return "high"
            if "low" in lower:
                return "low"
            return "moderate"
        return v

    @field_validator("visually_consistent", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        return bool(v) if v is not None else False


class VisualDifferentialReviewOutput(ResilientBase):
    """
    Structured result of MedGemma re-examining the lesion image against
    every differential candidate. Produced between Phase A and Phase B of the crew.
    """

    visual_winner: str = Field(
        default="",
        description="The diagnosis candidate the image most strongly supports",
    )
    visual_confidence: Literal["high", "moderate", "low"] = Field(
        default="low",
        description="Overall confidence in the visual winner selection",
    )
    votes: list[VisualDifferentialVote] = Field(
        default=[],
        description="One vote entry per assessed candidate",
    )
    visual_reasoning_summary: str = Field(
        default="",
        description=(
            "2-3 sentence summary explaining which morphological features were decisive "
            "and why the visual winner was selected over the others"
        ),
    )
    decisive_features: list[str] = Field(
        default=[],
        description="Key visual features (e.g. 'annular border', 'central clearing') that drove the decision",
    )

    @field_validator("visual_winner", "visual_reasoning_summary", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("visual_confidence", mode="before")
    @classmethod
    def normalise_confidence(cls, v):
        if isinstance(v, str):
            lower = v.lower()
            if "high" in lower:
                return "high"
            if "low" in lower:
                return "low"
            return "moderate"
        return v

    @field_validator("votes", "decisive_features", mode="before")
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if v is not None else []


# ── Debate Resolver Schema ────────────────────────────────────────────────────

class DebateResolverOutput(ResilientBase):
    """
    Result of the single-call MedGemma Debate Resolver.
    MedGemma receives the image + full candidate list and picks ONE winner directly.
    This is the authoritative final diagnosis — the CMO does not override it.
    """

    confirmed_diagnosis: str = Field(
        default="",
        description="The diagnosis the image most strongly supports, chosen from the candidate list",
    )
    visual_reasoning: str = Field(
        default="",
        description=(
            "2-3 sentences citing specific morphological features "
            "(colour, border, shape, texture, elevation) that led to this choice"
        ),
    )
    candidates_considered: list[str] = Field(
        default=[],
        description="The full list of candidates that were presented to MedGemma",
    )

    @field_validator("confirmed_diagnosis", "visual_reasoning", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("candidates_considered", mode="before")
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if v is not None else []


# ── Debate Resolver — primary visual arbitration ───────────────────────────────

def run_debate_resolver(
    image_path: str,
    primary_diagnosis: str,
    differentials: list[str],
) -> DebateResolverOutput:
    """
    Single-call MedGemma debate: present the image + all candidate diagnoses,
    ask MedGemma to pick the one that best matches the visual evidence.

    This is the authoritative final diagnosis step. No synthesis LLM is involved;
    the answer is parsed directly from MedGemma's response.

    Args:
        image_path:        Path to the skin lesion image.
        primary_diagnosis: Primary diagnosis proposed by the Differential agent.
        differentials:     List of alternative condition names.

    Returns:
        DebateResolverOutput with confirmed_diagnosis and visual_reasoning.
    """
    if not image_path:
        return DebateResolverOutput()

    # Deduplicate while preserving order; primary goes first
    seen: set[str] = set()
    candidates: list[str] = []
    for c in [primary_diagnosis] + differentials:
        clean = (c or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            candidates.append(clean)

    if not candidates:
        return DebateResolverOutput()

    numbered = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(candidates))
    print(f"\n[DebateResolver] Presenting {len(candidates)} candidate(s) to MedGemma: {', '.join(candidates)}")

    prompt = (
        "You are a consultant dermatologist performing a visual diagnosis.\n"
        "The following diagnoses have been proposed for this skin lesion:\n"
        f"{numbered}\n\n"
        "Look ONLY at the image. Ignore all other context.\n"
        "Which ONE diagnosis from the list above does the image most strongly support?\n\n"
        "Respond in this exact format (no other text):\n"
        "DIAGNOSIS: <copy the exact diagnosis name from the numbered list above>\n"
        "REASONING: <2-3 sentences citing specific visual features you observe: "
        "colour, border characteristics, shape, surface texture, and elevation>"
    )

    tool = ImageAnalysisTool()
    response = tool._run(image_path, prompt)

    # Parse DIAGNOSIS: line directly — no formatter LLM needed
    confirmed = ""
    reasoning = ""

    diag_match = re.search(r"DIAGNOSIS\s*:\s*(.+)", response, re.IGNORECASE)
    reason_match = re.search(r"REASONING\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)

    if diag_match:
        raw_diag = diag_match.group(1).strip().rstrip(".")
        # Match against candidates (case-insensitive) to get the canonical name
        raw_lower = raw_diag.lower()
        for candidate in candidates:
            if candidate.lower() in raw_lower or raw_lower in candidate.lower():
                confirmed = candidate
                break
        if not confirmed:
            confirmed = raw_diag  # use as-is if no fuzzy match

    if reason_match:
        reasoning = reason_match.group(1).strip()

    # Fallback: if parsing failed, use adapt_to_model as a safety net
    if not confirmed:
        print("[DebateResolver] Direct parse failed — falling back to schema adapter.")
        synthesis_prompt = (
            f"MedGemma was shown a skin lesion image and {len(candidates)} diagnosis candidates.\n"
            f"Candidates: {', '.join(candidates)}\n\n"
            f"MedGemma's response:\n{response}\n\n"
            "Extract the chosen diagnosis and the visual reasoning from the response above."
        )
        fallback, meta = adapt_to_model(synthesis_prompt, DebateResolverOutput, "debate_resolver")
        confirmed = fallback.confirmed_diagnosis or (candidates[0] if candidates else "")
        reasoning = fallback.visual_reasoning or reasoning

    print(f"[DebateResolver] Winner: {confirmed}")

    return DebateResolverOutput(
        confirmed_diagnosis=confirmed,
        visual_reasoning=reasoning,
        candidates_considered=candidates,
    )


# ── Initial Holistic MedGemma Diagnosis ───────────────────────────────────────

class MedGemmaInitialDiagnosis(ResilientBase):
    """
    Structured result of the very first MedGemma call — image + patient symptoms,
    free-form response parsed by the formatter LLM into primary diagnosis + reasoning.
    This is the highest-authority anchor for the entire pipeline.
    """

    primary_diagnosis: str = Field(
        default="",
        description="The primary diagnosis MedGemma identified from the image and patient symptoms",
    )
    reasoning: str = Field(
        default="",
        description="MedGemma's clinical reasoning supporting the diagnosis",
    )

    @field_validator("primary_diagnosis", "reasoning", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""


def run_initial_medgemma_diagnosis(
    image_path: str,
    patient_text: str,
) -> MedGemmaInitialDiagnosis:
    """
    First step of the pipeline: MedGemma examines the image alongside whatever
    the patient has typed. No format nudging — MedGemma writes freely.
    The formatter LLM (adapt_to_model) then extracts the structured result.

    Args:
        image_path:   Path to the skin lesion image.
        patient_text: The patient's raw symptom description + profile details.

    Returns:
        MedGemmaInitialDiagnosis with primary_diagnosis and reasoning.
    """
    if not image_path:
        return MedGemmaInitialDiagnosis()

    prompt = (
        "You are a dermatologist. Here is the patient's information:\n\n"
        f"{patient_text}\n\n"
        "What is your diagnosis?"
    )

    tool = ImageAnalysisTool()
    raw = tool._run(image_path, prompt)

    print(f"\n[InitialDiagnosis] MedGemma raw response:\n{raw[:300]}{'...' if len(raw) > 300 else ''}")

    parsed, meta = adapt_to_model(raw, MedGemmaInitialDiagnosis, "medgemma_initial")
    status = meta.get("status", "unknown")
    print(f"[InitialDiagnosis] Formatter status: {status} | Diagnosis: {parsed.primary_diagnosis}")

    return parsed


# ── Legacy: per-candidate Visual Differential Review ──────────────────────────

def run_visual_differential_review(
    image_path: str,
    primary_diagnosis: str,
    differentials: list[str],
) -> tuple[VisualDifferentialReviewOutput, str]:
    """
    Re-examine the lesion image against every differential candidate using MedGemma.

    Args:
        image_path:        Path to the skin lesion image.
        primary_diagnosis: The primary diagnosis proposed by the Differential agent.
        differentials:     List of alternative condition names from DifferentialDiagnosisOutput.

    Returns:
        (VisualDifferentialReviewOutput, raw_combined_text)
        raw_combined_text is the concatenated MedGemma responses for audit logging.
    """
    if not image_path:
        empty = VisualDifferentialReviewOutput()
        return empty, ""

    # Deduplicate while preserving order; primary goes first
    seen: set[str] = set()
    candidates: list[str] = []
    for c in [primary_diagnosis] + differentials:
        clean = (c or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            candidates.append(clean)

    if not candidates:
        empty = VisualDifferentialReviewOutput()
        return empty, ""

    print(f"\n[VisualReview] Examining image against {len(candidates)} candidate(s): {', '.join(candidates)}")

    tool = ImageAnalysisTool()

    def _assess(condition: str) -> tuple[str, str]:
        prompt = (
            f"You are a dermatology specialist examining a skin lesion photograph.\n"
            f"Question: Is this lesion visually consistent with the diagnosis of '{condition}'?\n"
            f"Answer YES or NO on the first line.\n"
            f"Then explain your visual assessment in exactly 2 sentences, "
            f"citing specific morphological features you observe: "
            f"colour, border characteristics, shape, surface texture, and elevation."
        )
        response = tool._run(image_path, prompt)
        return condition, response

    raw_parts: list[str] = []
    per_condition: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        futures = {executor.submit(_assess, c): c for c in candidates}
        for future in as_completed(futures):
            condition, response = future.result()
            per_condition[condition] = response
            raw_parts.append(f"=== {condition} ===\n{response}")

    # Compile into one block for the schema adapter
    raw_combined = "\n\n".join(raw_parts)

    # Build a richer prompt for the formatter that includes the full per-condition responses
    synthesis_prompt = (
        "You are reviewing MedGemma's visual assessments of a skin lesion "
        f"against {len(candidates)} diagnosis candidates.\n\n"
        "Per-candidate assessments:\n"
        + "\n\n".join(
            f"Candidate: {c}\nMedGemma response: {per_condition[c]}"
            for c in candidates
        )
        + "\n\nUsing these assessments, determine:\n"
        "1. Which candidate the image MOST strongly supports (visual_winner)\n"
        "2. Your confidence in that selection (high/moderate/low)\n"
        "3. A vote for each candidate (visually_consistent true/false, confidence, reasoning)\n"
        "4. A 2-3 sentence summary of which features were decisive\n"
        "5. A list of the specific decisive visual features"
    )

    parsed, meta = adapt_to_model(synthesis_prompt, VisualDifferentialReviewOutput, "visual_differential_review")

    status = meta.get("status", "unknown")
    print(f"[VisualReview] Schema adapter status: {status} | Winner: {parsed.visual_winner}")

    return parsed, raw_combined
