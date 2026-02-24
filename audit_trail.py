# Collects all intermediate agent outputs from a DermaCrew run.
# This data feeds the doctor's detailed PDF and the approval loop.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AuditTrail:
    """
    Complete record of everything the AI did to reach its diagnosis.
    Populated by DermaCrew.run() after each crew execution.
    """

    # Inputs
    patient_text: str = ""
    image_path: str = ""

    # Vision pre-run (raw model outputs before any agent sees them)
    vision_colour_raw: str = ""
    vision_texture_raw: str = ""
    vision_levelling_raw: str = ""
    vision_border_raw: str = ""
    vision_shape_raw: str = ""

    # Agent outputs (stored as raw strings + pydantic objects where available)
    biodata_summary: str = ""
    colour_output: Optional[object] = None       # ColourOutput pydantic
    texture_output: Optional[object] = None      # SurfaceOutput pydantic
    levelling_output: Optional[object] = None    # LevellingOutput pydantic
    border_output: Optional[object] = None       # BorderOutput pydantic
    shape_output: Optional[object] = None        # ShapeOutput pydantic
    decomposition_output: Optional[object] = None  # DecompositionOutput pydantic
    research_output: Optional[object] = None     # ResearchSummary pydantic
    differential_output: Optional[object] = None # DifferentialDiagnosisOutput pydantic
    mimic_resolution_output: Optional[object] = None # MimicResolutionOutput pydantic
    visual_differential_review_raw: str = ""      # raw MedGemma text for all differential votes
    visual_differential_review_output: Optional[object] = None  # VisualDifferentialReviewOutput pydantic
    cmo_output: Optional[object] = None          # CMOResult pydantic
    treatment_output: Optional[object] = None    # TreatmentPlanOutput pydantic
    final_diagnosis: Optional[object] = None     # FinalDiagnosis pydantic
    raw_outputs: dict[str, str] = field(default_factory=dict)
    adapter_status: dict[str, str] = field(default_factory=dict)   # ok / recovered / defaulted / missing
    adapter_errors: dict[str, str] = field(default_factory=dict)

    # Doctor review history — one entry per rejection round
    feedback_history: list[dict] = field(default_factory=list)
    # Each entry: {"round": int, "feedback": str, "rerun_scope": str}

    run_count: int = 1   # increments on each re-run