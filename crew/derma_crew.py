# Assembles and runs the full DermaAI v2 pipeline.
# This is the main entry point called by main.py.
#
# Execution is split into two crew phases with a direct MedGemma call between them:
#   Phase A: biodata → lesion agents → decomposition → research → differential → mimic resolution
#   [Between]: Visual Differential Review — MedGemma re-examines image vs every differential
#   Phase B: treatment → CMO (receives lesion summary + visual verdict) → scribe

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from crewai import Crew, Process

from agents.biodata_agent import create_biodata_agent, create_biodata_task
from agents.lesion_agents import (
    create_colour_agent, create_colour_task,
    create_texture_agent, create_texture_task,
    create_levelling_agent, create_levelling_task,
    create_border_agent, create_border_task,
    create_shape_agent, create_shape_task,
    ColourOutput, SurfaceOutput, LevellingOutput, BorderOutput, ShapeOutput,
)
from agents.decomposition_agent import create_decomposition_agent, create_decomposition_task, DecompositionOutput
from agents.research_agent import create_research_agent, create_research_task, ResearchSummary
from agents.orchestrator_agent import (
    create_cmo_agent, create_cmo_task, create_scribe_agent, create_scribe_task,
    CMOResult, FinalDiagnosis,
)
from tools.image_tool import ImageAnalysisTool
from utils.clarification_loop import run_clarification_loop
from agents.clinical_agents import (
    create_differential_agent, create_differential_task,
    create_mimic_resolution_agent, create_mimic_resolution_task,
    create_treatment_agent, create_treatment_task,
    DifferentialDiagnosisOutput, MimicResolutionOutput, TreatmentPlanOutput,
)
from agents.visual_differential_agent import (
    run_debate_resolver, DebateResolverOutput,
    run_visual_differential_review, VisualDifferentialReviewOutput,
    run_initial_medgemma_diagnosis, MedGemmaInitialDiagnosis,
)
from audit_trail import AuditTrail
from utils.schema_adapter import adapt_to_model

class DermaCrew:
    """
    Orchestrates the full multi-agent dermatology diagnosis pipeline.

    Usage:
        crew = DermaCrew(image_path="path/to/image.jpg", patient_text="I have...")
        result = crew.run()
        print(result.primary_diagnosis)
    """

    def __init__(self, image_path: str, patient_text: str):
        self.image_path = image_path
        self.patient_text = patient_text
        self._result: FinalDiagnosis | None = None
        self.audit: AuditTrail = AuditTrail(
            patient_text=patient_text,
            image_path=image_path,
        )

    def _run_vision_analysis(self) -> dict:
        """
        Each specialist agent independently examines the image using VISION_LLM (MedGemma).
        Returns a dict with keys: colour, texture, levelling, border, shape.

        All 5 calls run in parallel via ThreadPoolExecutor — they are fully independent
        and each can take 30-60 s, so parallelizing cuts this phase from ~3 min to ~1 min.

        Called directly (not through CrewAI) because MedGemma outputs tool_code blocks
        instead of OpenAI-style function calls, causing infinite retry loops in CrewAI.
        """
        print("\n[Vision] Specialist agents examining image in parallel...")
        tool = ImageAnalysisTool()

        specs = [
            (
                "colour",
                "You are a Dermatology Colour Analyst examining a skin lesion. "
                "In 2-3 concise sentences, describe the colour of the lesion using clinical dermatology terms. "
                "State what colour(s) are present and how the lesion compares to the surrounding skin.",
            ),
            (
                "texture",
                "You are a Dermatology Texture Analyst examining a skin lesion. "
                "In 2-3 concise sentences, describe the surface texture of the lesion using clinical terms. "
                "Note the key surface characteristics you observe.",
            ),
            (
                "levelling",
                "You are a Dermatology Morphology Analyst examining a skin lesion. "
                "In 2-3 concise sentences, describe the elevation of the lesion relative to surrounding skin. "
                "State whether it is raised, flat, or depressed and any relevant 3D features visible.",
            ),
            (
                "border",
                "You are a Dermatology Border Analyst examining a skin lesion. "
                "In 2-3 concise sentences, describe the border and edge characteristics of the lesion. "
                "Describe how the edge transitions to surrounding skin and any notable edge features.",
            ),
            (
                "shape",
                "You are a Dermatology Shape Analyst examining a skin lesion. "
                "In 2-3 concise sentences, describe the geometric shape and overall outline of the lesion. "
                "State the form, symmetry, and any distinctive structural characteristics.",
            ),
        ]

        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(tool._run, self.image_path, prompt): key
                for key, prompt in specs
            }
            for future in as_completed(futures):
                key = futures[future]
                results[key] = future.result()

        print("[Vision] Parallel specialist examination complete.\n")
        return results

    @staticmethod
    def _build_lesion_summary(vision: dict) -> str:
        """
        Build a compact string of lesion visual findings to inject directly into
        task descriptions, replacing 5 verbose task objects in context lists.
        Reduces downstream agent token consumption by ~60% for the affected tasks.
        """
        parts = ["LESION VISUAL SUMMARY (from MedGemma specialist agents):"]
        for key in ("colour", "texture", "levelling", "border", "shape"):
            value = (vision.get(key) or "").strip()
            if value:
                parts.append(f"- {key.capitalize()}: {value}")
        return "\n".join(parts)

    @staticmethod
    def _build_visual_verdict_summary(review: VisualDifferentialReviewOutput) -> str:
        """
        Build a compact string of the Visual Differential Review result to inject
        into the CMO task description. Gives the CMO the image-based verdict without
        adding another full task object to its context list.
        """
        if not review or not review.visual_winner:
            return ""

        lines = [
            "VISUAL DIFFERENTIAL REVIEW (MedGemma re-examined image vs each differential):",
            f"- Visual winner: {review.visual_winner} (confidence: {review.visual_confidence})",
        ]
        if review.visual_reasoning_summary:
            lines.append(f"- Summary: {review.visual_reasoning_summary}")
        if review.decisive_features:
            lines.append(f"- Decisive features: {', '.join(review.decisive_features)}")
        if review.votes:
            lines.append("- Per-candidate votes:")
            for vote in review.votes:
                consistent = "YES" if vote.visually_consistent else "NO"
                lines.append(
                    f"    {vote.condition}: {consistent} ({vote.confidence}) — {vote.visual_reasoning}"
                )
        return "\n".join(lines)

    def _adapt_task_output(self, key: str, task, model_cls):
        """
        Parse a task's raw free-text output into a target Pydantic schema.
        Always records adapter status in the audit trail and never throws.
        """
        raw = ""
        if task is not None and getattr(task, "output", None) is not None:
            raw = getattr(task.output, "raw", "") or ""

        self.audit.raw_outputs[key] = raw

        if not raw:
            parsed, meta = adapt_to_model("", model_cls, key)
            self.audit.adapter_status[key] = "missing"
            if meta.get("error"):
                self.audit.adapter_errors[key] = meta["error"]
            return parsed

        parsed, meta = adapt_to_model(raw, model_cls, key)
        self.audit.adapter_status[key] = meta.get("status", "unknown")
        if meta.get("error"):
            self.audit.adapter_errors[key] = meta["error"]
        return parsed


    def run(
        self,
        task_callback=None,
        skip_clarification: bool = False,
    ) -> tuple[FinalDiagnosis, AuditTrail]:
        """
        Run the full pipeline.

        Args:
            task_callback: Optional callable(task_output) invoked after each task
                           completes. Used by the web app to push SSE progress events.
                           When None the CLI path runs unchanged.
            skip_clarification: When True the clarification pre-pass is skipped.
                                 Set by the web app, which handles clarification
                                 externally before calling run().
        """
        print("\n" + "="*60)
        print("  DermaAI v2 — Multi-Agent Analysis Starting")
        print("="*60)

        if not skip_clarification:
            print("\n[Phase 0/4] ── Clarification Pre-pass ──────────────────────")
            _pre_biodata_agent = create_biodata_agent()
            _pre_biodata_task  = create_biodata_task(_pre_biodata_agent)
            Crew(
                agents=[_pre_biodata_agent],
                tasks=[_pre_biodata_task],
                process=Process.sequential,
                verbose=True,
            ).kickoff()

            self.patient_text, _ = run_clarification_loop(
                patient_text=self.patient_text,
                biodata_agent=_pre_biodata_agent,
                biodata_task=_pre_biodata_task,
            )

        # ── Phase 1: Create all agents ────────────────────────────────────────
        print("\n[Phase 1/4] ── Initialising Agents ────────────────────────────")

        biodata_agent    = create_biodata_agent()
        colour_agent     = create_colour_agent()
        texture_agent    = create_texture_agent()
        levelling_agent  = create_levelling_agent()
        border_agent     = create_border_agent()
        shape_agent      = create_shape_agent()
        decomp_agent     = create_decomposition_agent()
        research_agent   = create_research_agent()
        diff_agent       = create_differential_agent()
        mimic_agent      = create_mimic_resolution_agent()
        treatment_agent  = create_treatment_agent()
        cmo_agent        = create_cmo_agent()
        scribe_agent     = create_scribe_agent()

        # ── Phase 2: Create tasks in dependency order ─────────────────────────
        print("\n[Phase 2/4] ── Wiring Task Dependencies ───────────────────────")

        # Biodata runs first — everything else may reference it
        biodata_task     = create_biodata_task(biodata_agent)

        # Lesion agents and tasks only run when an image is provided
        vision: dict = {}
        lesion_summary: str = ""
        medgemma_initial: MedGemmaInitialDiagnosis = MedGemmaInitialDiagnosis()
        medgemma_anchor: str = ""
        if self.image_path:
            vision = self._run_vision_analysis()
            lesion_summary = self._build_lesion_summary(vision)

            # ── Initial holistic MedGemma diagnosis (image + patient symptoms) ──
            # MedGemma sees the full image + raw patient text and writes freely.
            # The formatter LLM extracts primary_diagnosis + reasoning from the response.
            # This becomes the highest-authority anchor for all downstream agents.
            print("\n[Phase 2/4] ── Initial MedGemma Diagnosis ─────────────────────")
            try:
                medgemma_initial = run_initial_medgemma_diagnosis(self.image_path, self.patient_text)
                if medgemma_initial.primary_diagnosis:
                    medgemma_anchor = f"{medgemma_initial.primary_diagnosis} — {medgemma_initial.reasoning}"
                    lesion_summary += (
                        f"\n\nMEDGEMMA INITIAL HOLISTIC ASSESSMENT (image + patient symptoms):\n"
                        f"- Primary Diagnosis: {medgemma_initial.primary_diagnosis}\n"
                        f"- Reasoning: {medgemma_initial.reasoning}"
                    )
                    self.audit.raw_outputs["medgemma_initial"] = medgemma_anchor
                    print(f"[Phase 2/4] Anchor diagnosis: '{medgemma_initial.primary_diagnosis}'")
            except Exception as mg_err:
                print(f"[Phase 2/4] WARNING: Initial MedGemma diagnosis failed: {mg_err}")

            colour_task     = create_colour_task(colour_agent, self.image_path, biodata_task, vision_result=vision["colour"])
            texture_task    = create_texture_task(texture_agent, self.image_path, biodata_task, vision_result=vision["texture"])
            levelling_task  = create_levelling_task(levelling_agent, self.image_path, biodata_task, vision_result=vision["levelling"])
            border_task     = create_border_task(border_agent, self.image_path, biodata_task, vision_result=vision["border"])
            shape_task      = create_shape_task(shape_agent, self.image_path, biodata_task, vision_result=vision["shape"])
            lesion_agents   = [colour_agent, texture_agent, levelling_agent, border_agent, shape_agent]
            lesion_tasks    = [colour_task, texture_task, levelling_task, border_task, shape_task]
        else:
            colour_task = texture_task = levelling_task = border_task = shape_task = None
            lesion_agents = []
            lesion_tasks  = []

        # Decomposition uses self.patient_text — enriched if clarification happened
        decomp_task = create_decomposition_task(decomp_agent, self.patient_text, biodata_task)

        # Research: lesion findings arrive via lesion_summary string (not task objects)
        research_task = create_research_task(
            research_agent,
            biodata_task=biodata_task,
            decomposition_task=decomp_task,
            lesion_summary=lesion_summary,
        )

        # Differential: still receives full lesion task objects (it synthesizes them for the first time)
        diff_task = create_differential_task(
            diff_agent,
            biodata_task=biodata_task,
            colour_task=colour_task,
            texture_task=texture_task,
            levelling_task=levelling_task,
            border_task=border_task,
            shape_task=shape_task,
            decomposition_task=decomp_task,
            research_task=research_task,
            medgemma_anchor=medgemma_anchor,
        )

        # Mimic: needs differential + visual evidence; research available as lower-weight context
        mimic_task = create_mimic_resolution_task(
            mimic_agent,
            differential_task=diff_task,
            colour_task=colour_task,
            texture_task=texture_task,
            levelling_task=levelling_task,
            border_task=border_task,
            shape_task=shape_task,
            research_task=research_task,
            medgemma_anchor=medgemma_anchor,
        )

        # ── Phase 3A: Run Phase A crew (up to mimic resolution) ──────────────
        print("\n[Phase 3A/4] ── Running Phase A Crew ──────────────────────────")
        print("  Agents: Biodata → Lesion (×5) → Decomp → Research → Differential → Mimic")

        phase_a_kwargs = dict(
            agents=[biodata_agent] + lesion_agents + [decomp_agent, research_agent, diff_agent, mimic_agent],
            tasks=[biodata_task] + lesion_tasks + [decomp_task, research_task, diff_task, mimic_task],
            process=Process.sequential,
            verbose=True,
        )
        if task_callback is not None:
            phase_a_kwargs["task_callback"] = task_callback

        phase_a_crew = Crew(**phase_a_kwargs)

        from tools.pubmed_tools import reset_pubmed_call_count
        reset_pubmed_call_count()

        phase_a_crew.kickoff()

        print("\n[Phase 3A/4] ── Phase A Complete ──────────────────────────────")

        # ── Between phases: Debate Resolver (single MedGemma call) ───────────
        # MedGemma receives the image + full candidate list and picks ONE winner.
        # This winner is the authoritative confirmed diagnosis — the CMO accepts it.
        confirmed_diagnosis: str = ""
        debate_output: DebateResolverOutput | None = None
        vdr_output: VisualDifferentialReviewOutput | None = None  # kept for audit compat
        vdr_raw: str = ""

        if self.image_path:
            print("\n[Phase 3.5/4] ── Debate Resolver (Image vs Differentials) ─")
            diff_parsed: DifferentialDiagnosisOutput = self._adapt_task_output(
                "differential_output_pre_vdr", diff_task, DifferentialDiagnosisOutput
            )
            candidates = [
                entry.condition for entry in (diff_parsed.differentials or [])
                if entry.condition
            ]
            try:
                # Use the MedGemma anchor as the primary diagnosis so it is always
                # candidate #1 in the debate. Fall back to the differential primary
                # if no anchor was produced.
                debate_primary = medgemma_initial.primary_diagnosis or diff_parsed.primary_diagnosis or ""
                debate_output = run_debate_resolver(
                    self.image_path,
                    debate_primary,
                    candidates,
                )
                confirmed_diagnosis = debate_output.confirmed_diagnosis or diff_parsed.primary_diagnosis or ""
                print(f"[Phase 3.5/4] Debate Resolver → confirmed: '{confirmed_diagnosis}'")
            except Exception as dr_err:
                print(f"[Phase 3.5/4] WARNING: Debate resolver failed: {dr_err}")
                debate_output = DebateResolverOutput()
                confirmed_diagnosis = diff_parsed.primary_diagnosis or ""
                print(f"[Phase 3.5/4] Falling back to differential primary: '{confirmed_diagnosis}'")

        # ── Phase 3B: Create Phase B tasks (treatment, CMO, scribe) ──────────

        treatment_task = create_treatment_task(
            treatment_agent,
            biodata_task=biodata_task,
            research_task=research_task,
            differential_task=diff_task,
            mimic_task=mimic_task,
        )

        # CMO receives compact lesion_summary + confirmed_diagnosis from the Debate Resolver.
        # It also receives the MedGemma initial anchor as the highest-authority default.
        cmo_task = create_cmo_task(
            cmo_agent,
            biodata_task=biodata_task,
            decomposition_task=decomp_task,
            research_task=research_task,
            differential_task=diff_task,
            mimic_task=mimic_task,
            lesion_summary=lesion_summary,
            confirmed_diagnosis=confirmed_diagnosis,
            medgemma_initial_diagnosis=medgemma_anchor,
        )

        scribe_task = create_scribe_task(
            scribe_agent,
            cmo_task=cmo_task,
            treatment_task=treatment_task,
            research_task=research_task,
        )

        print("\n[Phase 3B/4] ── Running Phase B Crew ──────────────────────────")
        print("  Agents: Treatment Protocol → CMO → Medical Scribe")

        phase_b_kwargs = dict(
            agents=[treatment_agent, cmo_agent, scribe_agent],
            tasks=[treatment_task, cmo_task, scribe_task],
            process=Process.sequential,
            verbose=True,
        )
        if task_callback is not None:
            phase_b_kwargs["task_callback"] = task_callback

        phase_b_crew = Crew(**phase_b_kwargs)

        try:
            phase_b_crew.kickoff()
        except Exception as e:
            print(f"\n[Warning] Phase B crew encountered an error: {e}")
            print("[Warning] Attempting to extract partial results...\n")

            # If the scribe/cmo never ran, attempt a recovery mini-crew
            if scribe_task.output is None or cmo_task.output is None:
                print("[Recovery] CMO/Scribe did not run — starting isolated recovery pass...")
                try:
                    def _has_output(t):
                        return t is not None and getattr(t, "output", None) is not None

                    recovery_cmo_task = create_cmo_task(
                        cmo_agent,
                        biodata_task=biodata_task if _has_output(biodata_task) else None,
                        decomposition_task=decomp_task if _has_output(decomp_task) else None,
                        research_task=research_task if _has_output(research_task) else None,
                        differential_task=diff_task if _has_output(diff_task) else None,
                        mimic_task=mimic_task if _has_output(mimic_task) else None,
                        lesion_summary=lesion_summary,
                        confirmed_diagnosis=confirmed_diagnosis,
                    )

                    recovery_scribe_task = create_scribe_task(
                        scribe_agent,
                        cmo_task=recovery_cmo_task,
                        treatment_task=None,
                        research_task=research_task if _has_output(research_task) else None,
                    )

                    recovery_crew = Crew(
                        agents=[cmo_agent, scribe_agent],
                        tasks=[recovery_cmo_task, recovery_scribe_task],
                        process=Process.sequential,
                        verbose=True,
                    )
                    recovery_crew.kickoff()
                    cmo_task = recovery_cmo_task
                    scribe_task = recovery_scribe_task
                    print("[Recovery] CMO/Scribe recovery run succeeded.\n")
                except Exception as re_err:
                    print(f"[Recovery] Recovery run also failed: {re_err}\n")

        print("\n[Phase 3B/4] ── Phase B Complete ──────────────────────────────")

        # ── Phase 4: Collect audit trail ──────────────────────────────────────
        print("\n[Phase 4/4] ── Collecting Results & Building Audit Trail ───────")

        if self.image_path:
            self.audit.vision_colour_raw    = vision["colour"]
            self.audit.vision_texture_raw   = vision["texture"]
            self.audit.vision_levelling_raw = vision["levelling"]
            self.audit.vision_border_raw    = vision["border"]
            self.audit.vision_shape_raw     = vision["shape"]

        self.audit.biodata_summary = biodata_task.output.raw if biodata_task.output else ""

        if self.image_path:
            self.audit.colour_output    = self._adapt_task_output("colour_output", colour_task, ColourOutput)
            self.audit.texture_output   = self._adapt_task_output("texture_output", texture_task, SurfaceOutput)
            self.audit.levelling_output = self._adapt_task_output("levelling_output", levelling_task, LevellingOutput)
            self.audit.border_output    = self._adapt_task_output("border_output", border_task, BorderOutput)
            self.audit.shape_output     = self._adapt_task_output("shape_output", shape_task, ShapeOutput)
        else:
            self.audit.colour_output    = None
            self.audit.texture_output   = None
            self.audit.levelling_output = None
            self.audit.border_output    = None
            self.audit.shape_output     = None

        self.audit.decomposition_output      = self._adapt_task_output("decomposition_output", decomp_task, DecompositionOutput)
        self.audit.research_output           = self._adapt_task_output("research_output", research_task, ResearchSummary)
        self.audit.differential_output       = self._adapt_task_output("differential_output", diff_task, DifferentialDiagnosisOutput)
        self.audit.mimic_resolution_output   = self._adapt_task_output("mimic_resolution_output", mimic_task, MimicResolutionOutput)

        # Debate Resolver — stored in audit trail (reuses visual_differential_review fields for compat)
        if debate_output is not None:
            debate_raw = (
                f"confirmed_diagnosis: {debate_output.confirmed_diagnosis}\n"
                f"visual_reasoning: {debate_output.visual_reasoning}\n"
                f"candidates_considered: {', '.join(debate_output.candidates_considered)}"
            )
            self.audit.visual_differential_review_raw    = debate_raw
            self.audit.visual_differential_review_output = debate_output
            self.audit.raw_outputs["debate_resolver"] = debate_raw
            self.audit.adapter_status["debate_resolver"] = "ok" if debate_output.confirmed_diagnosis else "defaulted"

        self.audit.treatment_output  = self._adapt_task_output("treatment_output", treatment_task, TreatmentPlanOutput)
        self.audit.cmo_output        = self._adapt_task_output("cmo_output", cmo_task, CMOResult)
        self.audit.final_diagnosis   = self._adapt_task_output("final_diagnosis", scribe_task, FinalDiagnosis)

        print(f"\n[Phase 4/4] CMO primary_diagnosis:    '{getattr(self.audit.cmo_output, 'primary_diagnosis', 'N/A')}'")
        print(f"[Phase 4/4] Scribe primary_diagnosis:  '{getattr(self.audit.final_diagnosis, 'primary_diagnosis', 'N/A')}'")
        print(f"[Phase 4/4] Adapter status (final_diagnosis): {self.audit.adapter_status.get('final_diagnosis', 'unknown')}")
        if self.audit.adapter_errors.get("final_diagnosis"):
            print(f"[Phase 4/4] Adapter error: {self.audit.adapter_errors['final_diagnosis']}")

        # Extract the structured result
        if self.audit.final_diagnosis is not None:
            self._result = self.audit.final_diagnosis
        else:
            failed_tasks = [
                name for name, t in [
                    ("Treatment", treatment_task),
                    ("Mimic Res", mimic_task),
                    ("Differential", diff_task),
                    ("Research", research_task),
                    ("Decomposition", decomp_task),
                    ("CMO", cmo_task),
                    ("Scribe", scribe_task),
                ]
                if t is not None and getattr(t, "output", None) is None
            ]
            raise RuntimeError(
                "Could not construct final diagnosis from raw outputs. "
                f"Upstream tasks with no output: {', '.join(failed_tasks) if failed_tasks else 'None'}."
            )

        return self._result, self.audit

    def rerun(self, feedback: str, scope: str, task_callback=None) -> tuple:
        """
        Re-run the pipeline with doctor feedback injected.

        Args:
            feedback: The doctor's rejection notes.
            scope:
                "full"              — Re-run every agent from scratch.
                "post_research"     — Re-run Differential + Treatment + Orchestrator
                                      (keeps visual/research; corrects interpretation).
                "orchestrator_only" — Re-run only the Orchestrator synthesis.

        For all three scopes the full crew is re-run with DOCTOR_FEEDBACK injected into
        the environment so the Orchestrator task description picks it up automatically.
        True partial-crew execution (re-running from agent N onward) is achievable but
        adds significant complexity — the env-var approach is the recommended first impl.
        """
        self.audit.run_count += 1
        self.audit.feedback_history.append({
            "round": self.audit.run_count,
            "feedback": feedback,
            "rerun_scope": scope,
        })

        print(f"\n[Re-run #{self.audit.run_count}] Scope: {scope}")
        print(f"Doctor feedback: {feedback}\n")

        # Inject feedback so the Orchestrator task description includes it
        os.environ["DOCTOR_FEEDBACK"] = feedback

        # All scopes call run() — feedback injection handles the distinction.
        # skip_clarification=True: enriched text is already set from the first run.
        result, audit = self.run(
            task_callback=task_callback,
            skip_clarification=True,
        )

        # Clear the env var after the run to avoid leaking into future runs
        os.environ.pop("DOCTOR_FEEDBACK", None)

        return result, audit

    def get_intermediate_outputs(self, tasks: dict) -> dict:
        """
        Returns intermediate outputs from all tasks for debugging.
        Pass the task objects dict after running.
        """
        return {
            name: (task.output.raw if task.output else None)
            for name, task in tasks.items()
        }
