# DermaAI v2 — Architecture Overview

This guide walks you through building a **multi-agent dermatology diagnosis system** from scratch using **CrewAI**, **Ollama**, and a **PubMed research tool**. You will build it as a CLI application first. A web UI comes last.

---

## What You Are Building

A crew of specialised AI agents that each do one job very well, share information with each other, and are coordinated by an orchestrator. The final output is two clinical reports — one for the doctor, one for the patient.

---

## The Agent Roster

| # | Agent | Primary Job | Model |
|---|-------|-------------|-------|
| 1 | **Profile / Biodata Agent** | Holds and serves patient context (age, sex, occupation, etc.) | `MedAIBase/MedGemma1.5:4b` |
| 2 | **Lesion Colour Agent** | Analyses lesion colour vs patient complexion from image | `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` |
| 3 | **Lesion Surface/Texture Agent** | Checks for dryness, scaling, blistering, crusting | `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` |
| 4 | **Lesion Levelling Agent** | Determines raised / flat / depressed morphology | `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` |
| 5 | **Lesion Border Agent** | Evaluates the edge/border characteristics of the lesion | `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` |
| 6 | **Lesion Shape Agent** | Determines the geometric shape and outline of the lesion | `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` |
| 7 | **Decomposition Agent** | Extracts symptoms, onset, progression from patient text | `qwen2.5:7b` |
| 8 | **Research Agent** | Queries PubMed for relevant medical literature | `qwen2.5:7b` |
| 9 | **Differential Agent** | Produces ranked differential list with full justification | `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` |
| 10| **Mimic Resolution Agent** | Cross-examines top mimics to find distinguishing factors | `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` |
| 11| **Treatment Agent** | Generates tiered treatment protocol for the confirmed diagnosis | `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` |
| 12| **CMO Agent** | Final clinical authority. Reviews all data and confirms diagnosis | `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` |
| 13| **Medical Scribe** | Compiles CMO reasoning and treatment plan into patient/doctor reports | `qwen2.5:7b` |

---

## How Data Flows — Step by Step

```
Patient Input
 ├── Image file path (skin photo)
 └── Free-text symptoms + complaints
        │
        ▼
 ┌─────────────────────────┐
 │   Profile/Biodata Agent │  ← Loaded once at start from user setup
 │   (serves context to    │
 │    all other agents)    │
 └────────────┬────────────┘
              │ context (age, sex, occupation, caste, pincode)
              ▼
 ┌────────────────────────────────────────────────────────────┐
 │              PARALLEL ANALYSIS PHASE                       │
 │                                                            │
 │  [Lesion Colour Agent]   → JSON: lesion_colour, reason     │
 │  [Lesion Texture Agent]  → JSON: surface, reason           │
 │  [Lesion Levelling Agent]→ JSON: levelling, reason         │
 │  [Lesion Border Agent]   → JSON: border, reason            │
 │  [Lesion Shape Agent]    → JSON: shape, reason             │
 │  [Decomposition Agent]   → JSON: symptoms[], time, etc.    │
 └────────────────────────┬───────────────────────────────────┘
                          │ all six outputs
                          ▼
              ┌───────────────────────┐
              │    Research Agent     │
              │  Queries PubMed with  │
              │  combined findings    │
              └───────────┬───────────┘
                          │ relevant articles + evidence
                          ▼
              ┌──────────────────────────┐
              │   Differential Agent     │
              │  Generates list with     │
              │  FOR/AGAINST reasoning   │
              └───────────┬──────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │ Mimic Resolution Agent   │
              │ Cross-examines top 2-3   │
              │ to confirm primary dx    │
              └───────────┬──────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │    Treatment Agent       │
              │  Writes protocol for     │
              │  confirmed diagnosis     │
              └───────────┬──────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │ Chief Medical Officer    │
              │  • Validates pipeline    │
              │  • Output final clinical │
              │    reasoning (strict JSON)│
              └───────────┬──────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │   Medical Scribe Agent   │
              │  • Drafts patient summary│
              │  • Drafts doctor notes   │
              └──────────────────────────┘
                          │
                          ▼
              Doctor Report (PDF / text)
              Patient Summary (PDF / text)
```

---

## Key Design Decisions (Read Before You Code)

### Why separate agents for each lesion property?
Each visual property (colour, texture, level, border) requires a different way of "looking" at the image. A single model asked to do all four tends to cut corners. Separate agents keep each analysis focused and its output independently verifiable.

### Why does the Research Agent come AFTER the lesion agents?
The research agent needs to know what to search for. It takes the combined lesion profile + decomposed symptoms as its search query. This way PubMed searches are targeted, not generic.

### Why separate the CMO and the Scribe?
Large Language Models suffer from "Lost in the Middle" syndrome when asked to perform deep clinical reasoning *and* heavy text formatting simultaneously. By separating them, the Medgemma-powered CMO focuses strictly on logic, while the Qwen-powered Scribe handles formatting and empathy.

### Why add a Mimic Resolution Agent?
Many skin conditions look identical visually (e.g., Ringworm vs. Eczema). The Differential Agent lists them both, but the Mimic Resolution Agent is explicitly instructed to cross-examine patient history and demographics to find the *one* distinguishing factor that proves one and disproves the other.

---

## Tools You Will Build

| Tool | Purpose | Library |
|------|---------|---------|
| `ImageAnalysisTool` | Loads an image file, base64 encodes it, sends to Ollama vision model with a specific prompt | `httpx`, `base64` |
| `PubMedSearchTool` | Searches PubMed by query string, returns titles + abstracts | `biopython` (Entrez) |
| `BiodataLookupTool` | Lets an agent ask the Biodata Agent for a specific patient field | internal |

---

## Project Directory Structure (Target)

By the end of this guide, your project will look like this:

```
dermaai-v2/
├── requirements.txt
├── .env
├── patient_profile.json        ← written during user setup step
│
├── tools/
│   ├── __init__.py
│   ├── image_tool.py           ← ImageAnalysisTool
│   └── pubmed_tool.py          ← PubMedSearchTool
│
├── agents/
│   ├── __init__.py
│   ├── biodata_agent.py
│   ├── lesion_agents.py        ← all 5 lesion agents in one file
│   ├── decomposition_agent.py
│   ├── research_agent.py
│   └── orchestrator_agent.py
│
├── crew/
│   ├── __init__.py
│   └── derma_crew.py           ← assembles and runs the full crew
│
├── reports/                    ← generated output lands here
│
└── main.py                     ← entry point
```

---

## Guide Chapters

| File | What You Will Do |
|------|-----------------|
| `01_SETUP.md` | Create the project, virtualenv, install packages |
| `02_CREWAI_CONCEPTS.md` | Understand Agent / Task / Crew / Tool before writing any |
| `03_OLLAMA_CONNECTION.md` | Connect CrewAI to Ollama, run a sanity-check agent |
| `04_BIODATA_AGENT.md` | Build the patient profile system + biodata agent |
| `05_LESION_AGENTS.md` | Build the ImageAnalysisTool + all 4 lesion agents |
| `06_DECOMPOSITION_AGENT.md` | Build the symptom extraction agent |
| `07_PUBMED_TOOL.md` | Build and test the PubMed custom tool |
| `08_RESEARCH_AGENT.md` | Build the research agent using the PubMed tool |
| `09_ORCHESTRATOR.md` | Build the CMO, Medical Scribe, and assemble the full crew |
| `10_FULL_PIPELINE.md` | Run everything end-to-end, read and understand the output |
| `11_PDF_REPORTS.md` | Add PDF generation to the orchestrator output |

---

## Before You Start — Prerequisites Checklist

Make sure these are in place before opening chapter 01:

- [ ] Python 3.11 or 3.12 installed (check with `python --version`)
- [ ] `uv` installed (check with `uv --version`)
- [ ] Ollama running (`ollama serve` in a terminal)
- [ ] The following models already pulled:
  - `ollama pull dcarrascosa/medgemma-1.5-4b-it:Q4_K_M`
  - `ollama pull MedAIBase/MedGemma1.5:4b`
  - `ollama pull qwen2.5:7b`
- [ ] An NCBI account (free) and API key from https://www.ncbi.nlm.nih.gov/account/ — needed for PubMed in chapter 07
- [ ] A test image of a skin condition saved somewhere on your machine

**Verify Ollama is running before every session:**
```powershell
ollama list
# should show all three models above
```

---

*Start with → `01_SETUP.md`*
