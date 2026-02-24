# Chapter 08 — The Research Agent

**Goal:** Build the Research Agent that takes the combined outputs from the lesion agents and decomposition agent, formulates targeted PubMed queries, retrieves relevant literature, and summarises the evidence. Test it with context from previous agents.

**Time estimate:** 40–50 minutes

---

## What This Agent Does and Why It Matters

Without a research agent, your diagnosis is based only on:
- What the model "knows" from training data (possibly outdated)
- The patient's current presentation

With the research agent, your diagnosis is grounded in:
- Recent peer-reviewed literature (post-2015)
- Specific evidence for the combination of lesion features + patient demographics
- Articles that may identify rare conditions the LLM wouldn't consider

The research agent's output is one of the most important inputs the Orchestrator uses when re-evaluating or confirming the diagnosis.

---

## Step 1 — Design the Research Output Schema

**`agents/research_agent.py`**

```python
# agents/research_agent.py
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

from pydantic import BaseModel, Field, field_validator
from crewai import Agent, Task
from config import ORCHESTRATOR_LLM
from tools.pubmed_tools import PubMedSearchTool


class ResearchSummary(BaseModel):
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
            "Minimum 3 findings, maximum 8."
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
            import re
            m = re.search(r"\d+", v)
            return int(m.group()) if m else 0
        return 0
```

> **Null-safety and defaults:** All required fields now carry `default` values. All list fields have a `null → []` validator, all string fields have a `null → ""` validator, and `articles_found` accepts strings like `"4 articles found"` — a common LLM formatting habit. Without these, any `null` from the LLM causes a `ValidationError` that crashes the crew.

---

## Step 2 — Build the Agent

Add below the schema:

```python
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
            "You construct precise PubMed queries from lesion features and patient data. "
            "You retrieve, read, and summarise abstracts. "
            "You cite only PMIDs that appeared in the tool output — never from memory. "
            "You flag any literature that contradicts the visual analysis."
        ),
        llm=ORCHESTRATOR_LLM,
        tools=[PubMedSearchTool()],
        verbose=True,
    )
```

---

## Step 3 — Build the Research Task

The research task receives context from ALL previous tasks. This is its most important design aspect.

```python
def create_research_task(
    agent: Agent,
    biodata_task=None,
    colour_task=None,
    texture_task=None,
    levelling_task=None,
    shape_task=None,
    decomposition_task=None,
) -> Task:
    """
    The Research Agent synthesises all upstream task outputs to form
    targeted PubMed search queries.

    All arguments are optional — the agent adapts to whatever context is provided.
    """

    # Build context list from whatever tasks were provided
    context = [
        t for t in [
            biodata_task,
            colour_task,
            texture_task,
            levelling_task,
            shape_task,
            decomposition_task,
        ]
        if t is not None
    ]

    return Task(
        description=(
            "You have received the following clinical findings from the analysis agents:\n"
            "- Patient biodata (age, sex, skin tone, occupation, etc.)\n"
            "- Lesion colour analysis\n"
            "- Lesion surface/texture analysis\n"
            "- Lesion elevation analysis\n"
            "- Lesion border/shape analysis\n"
            "- Decomposed patient symptoms and history\n\n"

            "Your job:\n"
            "1. Synthesise these findings into 1-2 precise PubMed search queries\n"
            "   - Primary query: combine the most distinctive lesion features + likely diagnosis\n"
            "   - Secondary query (if needed): broader or alternative angle\n"
            "2. Use the pubmed_search tool with your primary query\n"
            "3. If primary results are weak (<3 relevant articles), run secondary query\n"
            "4. Read the abstracts carefully\n"
            "5. Identify which diagnoses are supported by the literature\n"
            "6. Identify any contradictions with the visual analysis\n"
            "7. Summarise your findings structured as the required output\n\n"

            "IMPORTANT: If literature suggests a condition the visual analysis did NOT identify, "
            "flag it in contradicted_findings — the orchestrator will decide whether to re-diagnose."
        ),
        expected_output=(
            "ONLY a valid JSON object — no text, no explanation, no markdown. "
            "Start with { and end with }. Use exactly these keys:\n"
            "  primary_search_query: string — the PubMed query you ran\n"
            "  secondary_search_query: string — second query if used, else empty string\n"
            "  articles_found: integer — number of relevant articles retrieved\n"
            "  key_findings: list of 3-8 strings — one sentence per finding from the abstracts\n"
            "  supported_diagnoses: list of strings — diagnoses mentioned in the literature\n"
            "  contradicted_findings: list of strings — any contradictions with visual analysis\n"
            "  evidence_strength: string — exactly one of: strong / moderate / limited / conflicting\n"
            "  cited_pmids: list of strings — actual PMID numbers from the tool output only\n"
            "  research_notes: string — caveats or gaps for the orchestrator\n"
            "Example: {\"primary_search_query\": \"contact dermatitis solvent forearm\", "
            "\"secondary_search_query\": \"\", \"articles_found\": 4, "
            "\"key_findings\": [\"finding 1\", \"finding 2\", \"finding 3\"], "
            "\"supported_diagnoses\": [\"allergic contact dermatitis\"], "
            "\"contradicted_findings\": [], \"evidence_strength\": \"moderate\", "
            "\"cited_pmids\": [\"12345678\", \"23456789\"], "
            "\"research_notes\": \"Limited studies on painter solvent exposure.\"}"
        ),
        agent=agent,
        output_pydantic=ResearchSummary,
        context=context,
    )
```

---

## Step 4 — Test the Research Agent with Simulated Context

Since you've already built and tested the upstream agents, you can now test the research agent with real upstream outputs. Create **`test_research_agent.py`**:

```python
# test_research_agent.py
# Tests the Research Agent with a realistic set of upstream task outputs.

from crewai import Crew, Process
from agents.biodata_agent import create_biodata_agent, create_biodata_task
from agents.decomposition_agent import create_decomposition_agent, create_decomposition_task
from agents.research_agent import create_research_agent, create_research_task

# Simulated patient input
PATIENT_TEXT = (
    "I have had a very itchy, bumpy rash on my left forearm for 4 days. "
    "It started near my wrist and has been spreading. "
    "I work as a painter and recently switched to a new solvent brand. "
    "It gets worse in the evenings. I tried antihistamine cream but no improvement."
)

print("Setting up agents and tasks...")

# Create all agents
biodata_agent   = create_biodata_agent()
decomp_agent    = create_decomposition_agent()
research_agent  = create_research_agent()

# Create tasks in dependency order
biodata_task    = create_biodata_task(biodata_agent)
decomp_task     = create_decomposition_task(decomp_agent, PATIENT_TEXT, biodata_task)
research_task   = create_research_task(
    research_agent,
    biodata_task=biodata_task,
    decomposition_task=decomp_task,
    # Note: We're skipping lesion tasks for this test (no image in this test)
    # In the full pipeline, you'll include all 4 lesion tasks here
)

crew = Crew(
    agents=[biodata_agent, decomp_agent, research_agent],
    tasks=[biodata_task, decomp_task, research_task],
    process=Process.sequential,
    verbose=True,
)

print("Running research pipeline (this will make real PubMed calls)...")
result = crew.kickoff()

print("\n" + "="*60)
print("RESEARCH AGENT OUTPUT")
print("="*60)
if research_task.output and research_task.output.pydantic:
    output = research_task.output.pydantic
    print(f"\nPrimary Query: {output.primary_search_query}")
    print(f"Secondary Query: {output.secondary_search_query}")
    print(f"Articles Found: {output.articles_found}")
    print(f"Evidence Strength: {output.evidence_strength}")
    print("\nKey Findings:")
    for i, finding in enumerate(output.key_findings, 1):
        print(f"  {i}. {finding}")
    print("\nSupported Diagnoses:")
    for dx in output.supported_diagnoses:
        print(f"  - {dx}")
    if output.contradicted_findings:
        print("\nContradictions / Flags:")
        for flag in output.contradicted_findings:
            print(f"  ⚠️ {flag}")
    print(f"\nCited PMIDs: {output.cited_pmids}")
    print(f"\nResearch Notes: {output.research_notes}")
else:
    print(result)
```

Run it:
```powershell
python test_research_agent.py
```

**What to verify:**

- The agent constructs a logical search query (should mention contact dermatitis + painter/solvent)
- Real PMIDs appear in `cited_pmids` — verify one at pubmed.ncbi.nlm.nih.gov
- `key_findings` are specific, cited from the abstracts (not invented)
- `supported_diagnoses` includes "contact dermatitis" or "allergic contact dermatitis"
- `evidence_strength` is appropriate ("strong" or "moderate" — contact dermatitis has good literature)

**If `cited_pmids` contains made-up numbers:** The agent is hallucinating. Add this to the task description: "You MUST use the pubmed_search tool and ONLY cite PMIDs that appear in the tool's output. Never write a PMID from memory."

**If the agent doesn't call the tool:** Make the task description more directive: "Your FIRST action must be to call the pubmed_search tool."

**If you see the agent looping with `tool_code` blocks and never completing:** You are likely using `TEXT_LLM` (MedGemma) which does not support tool calling. Make sure `create_research_agent()` uses `ORCHESTRATOR_LLM` as shown above — `qwen2.5:7b` supports the JSON function-call format that CrewAI requires.

**If you get a `ValidationError` with `{'description': '...'}` as the input value:** The model returned its own schema description as a JSON value instead of the actual data. This happens when `expected_output` is too vague — the model mirrors the Pydantic `Field(description=...)` text it sees in the schema and wraps it in a single `description` key. The fix (already applied above) is to spell out every required key name, its type, and a concrete JSON example directly in `expected_output`. Never use a vague phrase like "a structured JSON object with all fields populated" — always enumerate the keys explicitly.

---

## Step 5 — Understanding the Re-Diagnosis Hook

The `contradicted_findings` field is specifically designed for the Orchestrator. In Chapter 09 you'll see how the Orchestrator uses it:

```python
# Orchestrator logic (simplified):
if research_output.contradicted_findings:
    # Ask the relevant lesion agent to re-examine
    # Present the contradiction as additional context
    # Make a final decision with both the original and revised analysis
```

This is the **agentic re-diagnosis loop** — one of the most powerful features of this architecture. The system can self-correct based on evidence.

---

## Checkpoint ✅

- [ ] `agents/research_agent.py` exists with `ResearchSummary`, agent, and task factory
- [ ] The agent actually calls the `pubmed_search` tool (you see it in verbose output)
- [ ] `cited_pmids` contains real PMIDs you can verify on PubMed
- [ ] `key_findings` are grounded in the article abstracts (not hallucinated)
- [ ] `supported_diagnoses` makes clinical sense for the patient input
- [ ] If PMIDs are hallucinated, you've added the explicit "only cite tool output" instruction

Delete the test file:
```powershell
Remove-Item test_research_agent.py
```

---

*Next → `09_ORCHESTRATOR.md`*
