# Runs the Decomposition + Clarification mini-crew in a loop.
# Asks the patient follow-up questions if critical fields are missing.
# Returns the final patient_text (original + any Q&A appended) and the
# final DecompositionOutput once no further clarification is needed.

from crewai import Crew, Process
from agents.decomposition_agent import (
    create_decomposition_agent,
    create_decomposition_task,
    DecompositionOutput,
)
from agents.clarification_agent import (
    create_clarification_agent,
    create_clarification_task,
    ClarificationOutput,
)
from utils.schema_adapter import adapt_to_model

MAX_ROUNDS = 2   # Never ask the patient more than 2 rounds of follow-up questions


def _ask_patient_questions(questions: list[str]) -> str:
    """Present the clarification questions to the patient and collect their answers."""
    print("\n  The AI needs a bit more information to give you an accurate diagnosis.")
    print("  Please answer the following questions:\n")

    answers = []
    for i, question in enumerate(questions, 1):
        answer = input(f"  Q{i}. {question}\n  Your answer: ").strip()
        if answer:
            answers.append(f"Q: {question}\nA: {answer}")

    return "\n".join(answers)


def run_clarification_loop(
    patient_text: str,
    biodata_agent,
    biodata_task,
) -> tuple[str, DecompositionOutput]:
    """
    Runs up to MAX_ROUNDS of Decomposition + Clarification.

    Returns:
        - enriched_patient_text: original text + any Q&A appended
        - final_decomposition: the last DecompositionOutput
    """
    enriched_text = patient_text
    final_decomposition = None

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n[Clarification] Round {round_num} — Extracting clinical data...")

        # Build mini-crew: Decomposition + Clarification only
        decomp_agent  = create_decomposition_agent()
        decomp_task   = create_decomposition_task(decomp_agent, enriched_text, biodata_task)
        clarif_agent  = create_clarification_agent()
        clarif_task   = create_clarification_task(clarif_agent, decomp_task, biodata_task)

        mini_crew = Crew(
            agents=[decomp_agent, clarif_agent],
            tasks=[decomp_task, clarif_task],
            process=Process.sequential,
            verbose=False,   # quiet — main pipeline will show its own output
        )

        try:
            mini_crew.kickoff()
        except Exception as e:
            # LLM output failed Pydantic validation (e.g. trailing JSON characters,
            # null for a list field, bad Literal value).  Grab whatever was parsed
            # and proceed rather than crashing the whole pipeline.
            print(f"[Clarification] Output parsing error: {e}. Proceeding with available data.")
            raw = decomp_task.output.raw if decomp_task.output else ""
            final_decomposition, _ = adapt_to_model(raw, DecompositionOutput, "clarification_decomposition")
            break

        decomp_raw = decomp_task.output.raw if decomp_task.output else ""
        clarif_raw = clarif_task.output.raw if clarif_task.output else ""
        final_decomposition, _ = adapt_to_model(decomp_raw, DecompositionOutput, "clarification_decomposition")
        clarif_result, _ = adapt_to_model(clarif_raw, ClarificationOutput, "clarification_output")

        # Defensive: if Pydantic validation failed silently, proceed without blocking
        if clarif_result is None:
            print("[Clarification] Could not parse clarification output. Proceeding with available data.")
            break

        if not clarif_result.needs_clarification:
            print(f"[Clarification] Sufficient information. Proceeding with analysis.")
            break

        print(f"\n[Clarification] Missing: {', '.join(clarif_result.missing_fields)}")

        # Ask the patient and append answers to enriched_text
        qna = _ask_patient_questions(clarif_result.questions)
        if qna:
            enriched_text = enriched_text + "\n\nAdditional information:\n" + qna
            # Patient answered — proceed immediately, do not re-run another round
            break

        # Patient gave no answers — if this was the last allowed round, proceed anyway
        if round_num == MAX_ROUNDS:
            print(
                f"\n[Clarification] Maximum clarification rounds reached. "
                f"Proceeding with available information."
            )

    return enriched_text, final_decomposition

