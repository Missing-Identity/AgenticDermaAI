# Research Agent: queries PubMed to find evidence-based literature
# relevant to the patient's presentation and lesion findings.
#
# NOTE: Lesion, differential, and treatment agents use VISION_LLM (MedGemma).
# The Decomposition agent uses ORCHESTRATOR_LLM (qwen2.5:7b) for reliable JSON output.
# The Research Agent also uses ORCHESTRATOR_LLM because it must CALL a tool
# (PubMedSearchTool), and MedGemma does not output the OpenAI-style JSON function-call
# format that CrewAI requires — it outputs tool_code blocks instead, causing an
# infinite retry loop.
#
# qwen2.5:7b (ORCHESTRATOR_LLM) supports CrewAI tool calling correctly.
# This is a tool-compatibility constraint, not a medical knowledge choice.

from pydantic import Field, field_validator
from crewai import Agent, Task
from config import ORCHESTRATOR_LLM
from tools.pubmed_tools import PubMedSearchTool
from utils.resilient_base import ResilientBase


class ResearchSummary(ResilientBase):
    """
    Structured summary of PubMed research findings relevant to the case.
    """

    primary_search_query: str = Field(
        default="",
        description="The main search query used for PubMed"
    )
    secondary_search_query: str = Field(
        default="",
        description="A secondary/broader search query if primary returned few results"
    )
    articles_found: int = Field(
        default=0,
        description="Total number of relevant articles found"
    )
    key_findings: list[str] = Field(
        default=[],
        description=(
            "List of key clinical findings from the literature. "
            "Each item is one sentence summarising a distinct finding. "
            "Minimum 3 findings, maximum 5."
        )
    )
    supported_diagnoses: list[str] = Field(
        default=[],
        description=(
            "Diagnoses mentioned in the literature that match the patient presentation. "
            "Most likely first."
        )
    )
    contradicted_findings: list[str] = Field(
        default=[],
        description=(
            "Any literature findings that CONTRADICT the current lesion analysis. "
            "Important for the orchestrator's re-diagnosis decision."
        )
    )
    evidence_strength: str = Field(
        default="limited",
        description=(
            "Overall strength of literature support: "
            "strong / moderate / limited / conflicting"
        )
    )
    cited_pmids: list[str] = Field(
        default=[],
        description="List of PubMed IDs (PMIDs) of relevant articles found"
    )
    research_notes: str = Field(
        default="",
        description=(
            "Any important caveats, gaps in literature, or special notes "
            "the orchestrator should be aware of."
        )
    )

    @field_validator(
        "key_findings", "supported_diagnoses", "contradicted_findings", "cited_pmids",
        mode="before",
    )
    @classmethod
    def coerce_null_to_list(cls, v):
        return v if v is not None else []

    @field_validator(
        "primary_search_query", "secondary_search_query", "evidence_strength", "research_notes",
        mode="before",
    )
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("articles_found", mode="before")
    @classmethod
    def coerce_articles_found(cls, v):
        """Accept string '4' or '4 articles found' as well as plain int."""
        if v is None:
            return 0
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            # Extract leading digits
            import re
            m = re.search(r"\d+", v)
            return int(m.group()) if m else 0
        return 0

def create_research_agent() -> Agent:
    return Agent(
        role="Dermatology Research Analyst",
        goal=(
            "Search PubMed for peer-reviewed evidence directly relevant to this skin condition. "
            "Return only findings that bear on the lesion, diagnosis, or patient demographics. "
            "No general commentary."
        ),
        backstory=(
            "You are a dermatology literature specialist. "
            "You have only one tool: pubmed_search. You produce your final summary as text output, not by calling another tool. "
            "You construct precise PubMed queries from lesion features and patient data. "
            "You retrieve, read, and summarise abstracts. You do at most 2 searches, then write your summary. "
            "You cite only PMIDs that appeared in the tool output — never from memory. "
            "You flag any literature that contradicts the visual analysis."
        ),
        llm=ORCHESTRATOR_LLM,
        tools=[PubMedSearchTool()],
        max_iter=4,
        verbose=True,
    )

def create_research_task(
    agent: Agent,
    biodata_task=None,
    decomposition_task=None,
    lesion_summary: str = "",
    # Legacy keyword args accepted but ignored — lesion tasks replaced by lesion_summary
    colour_task=None,
    texture_task=None,
    levelling_task=None,
    border_task=None,
    shape_task=None,
) -> Task:
    """
    The Research Agent uses a compact lesion summary (injected into the description)
    plus biodata and decomposition context to build targeted PubMed queries.

    Individual lesion task objects are no longer passed as context — their content
    is condensed into lesion_summary to avoid injecting hundreds of redundant tokens.
    """

    # Only biodata and decomposition — lesion findings arrive via lesion_summary string
    context = [t for t in [biodata_task, decomposition_task] if t is not None]

    summary_block = f"{lesion_summary}\n\n" if lesion_summary else ""

    return Task(
        description=(
            summary_block +
            "Using the lesion visual summary above (if provided), patient biodata, "
            "and decomposed symptoms from upstream agents:\n\n"

            "1. Read all upstream findings and identify the key clinical descriptors "
            "(shape, border, colour, texture, patient demographics).\n"
            "2. Build a SHORT PubMed search query of 2-4 clinical terms only. "
            "Use the primary diagnosis candidate or lesion morphology as the first term, "
            "then add 1-2 highly specific modifiers (e.g. body site or key symptom). "
            "Good examples: 'tinea corporis hand pruritus', 'granuloma annulare', "
            "'erythema annulare centrifugum'. Bad examples: 'annular erythematous plaque "
            "central clearing border elevation patient demographics'.\n"
            "3. Run the pubmed_search tool ONCE. If it returns 3+ articles, STOP searching and go to step 5. "
            "If it returns fewer than 3, run ONE broader secondary query (pass exclude_pmids with PMIDs from the first search to avoid duplicates), then STOP. "
            "You must never make more than 2 pubmed_search calls total.\n"
            "4. When summarising, deduplicate by PMID — if the same PMID appeared in multiple searches, count and cite it only once.\n"
            "5. Summarise the findings: which diagnoses are supported, which are contradicted.\n"
            "6. Cite only PMIDs that appeared in the tool output — never from memory.\n\n"
            "IMPORTANT: You have ONLY the pubmed_search tool. Produce your final research summary as your TEXT OUTPUT. "
            "Do NOT call any other tool such as 'final_analysis' — it does not exist. Your task ends when you write your summary."
        ),
        expected_output=(
            "A concise free-text research summary. Include: primary query, optional secondary query, "
            "article count, supported diagnoses, contradictions, evidence strength, and cited PMIDs. "
            "Do not use JSON or markdown."
        ),
        agent=agent,
        context=context,
    )