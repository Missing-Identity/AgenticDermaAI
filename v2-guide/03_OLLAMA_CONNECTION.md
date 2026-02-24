# Chapter 03 — Connecting CrewAI to Ollama

**Goal:** Understand exactly how CrewAI talks to Ollama, create a reusable LLM configuration module, and run a real multi-model test to confirm all three Ollama models respond correctly.

**Time estimate:** 20–30 minutes

---

## How CrewAI Talks to Ollama

CrewAI does NOT call Ollama directly. The call chain is:

```
CrewAI Agent
     │
     ▼
  LiteLLM          ← universal LLM router library
     │
     ▼
 Ollama API         ← http://localhost:11434
     │
     ▼
  Local Model       ← qwen2.5:7b, MedGemma, etc.
```

**LiteLLM** is what makes this possible. It translates CrewAI's OpenAI-style API calls into Ollama's format. You just need to use the right prefix and base URL.

The pattern is always:
```python
from crewai import LLM

llm = LLM(
    model="ollama/MODEL_NAME_EXACTLY_AS_IN_OLLAMA_LIST",
    base_url="http://localhost:11434",
)
```

> **Critical:** The model name after `ollama/` must match exactly what `ollama list` shows. A single character difference causes a 404 error.

---

## Step 1 — Verify Your Models Are Running

Before writing any code, confirm Ollama is serving and your models are present:

```powershell
# In a SEPARATE terminal window (keep Ollama running here)
ollama serve

# In your main terminal:
ollama list
```

You should see all three:
```
NAME                                     SIZE
qwen2.5:7b                               4.7 GB
dcarrascosa/medgemma-1.5-4b-it:Q4_K_M   3.3 GB
MedAIBase/MedGemma1.5:4b                7.8 GB
```

Also run a quick raw test to confirm Ollama responds:
```powershell
curl http://localhost:11434/api/tags
```
This should return JSON listing your models. If it fails, Ollama isn't running.

---

## Step 2 — Create the LLM Configuration Module

Create `config.py` in your project root. This is where all three model handles live:

```python
# config.py
# Central LLM configuration for all agents.
# Import from here — never hardcode model names in agent files.

import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Model handles ─────────────────────────────────────────────────────────────

# Vision model — used by all 4 lesion analysis agents + clarification +
# differential + treatment.  Supports image input via base64.
#
# repeat_penalty=1.25 / repeat_last_n=64: guard against the deterministic
# repetition loops that MedGemma 1.5b exhibits at low temperature.
VISION_LLM = LLM(
    model="ollama/dcarrascosa/medgemma-1.5-4b-it:Q4_K_M",
    base_url=OLLAMA_BASE_URL,
    extra_body={"options": {"repeat_penalty": 1.25, "repeat_last_n": 64}},
)

# Text medical model — Biodata agent only (data formatting)
TEXT_LLM = LLM(
    model="ollama/MedAIBase/MedGemma1.5:4b",
    base_url=OLLAMA_BASE_URL,
)

# Orchestration model — Research, Decomposition, and Orchestrator agents
# qwen2.5:7b: best-in-class strict schema compliance + reliable tool calling
ORCHESTRATOR_LLM = LLM(
    model="ollama/qwen2.5:7b",
    base_url=OLLAMA_BASE_URL,
)
```

> **Why centralise this?** If Ollama releases an update, or you want to swap a model for testing, you change it in ONE place. No hunting through 8 different agent files.

---

## Step 3 — Test Each Model with a Real CrewAI Call

Create a test file in your project root called `test_models.py`. You will write one test per model. **Write them one at a time, run each one before moving to the next.**

### Test A — Orchestrator (qwen2.5:7b)

```python
# test_models.py
from crewai import Agent, Task, Crew, Process
from config import ORCHESTRATOR_LLM

def test_orchestrator():
    print("\n" + "="*50)
    print("TEST A: Orchestrator LLM (qwen2.5:7b)")
    print("="*50)

    agent = Agent(
        role="Test Agent",
        goal="Answer a simple medical question concisely",
        backstory="You are a general knowledge assistant being tested.",
        llm=ORCHESTRATOR_LLM,
        verbose=True,
    )

    task = Task(
        description=(
            "Name three common inflammatory skin conditions. "
            "Be brief — one word each."
        ),
        expected_output="A numbered list of exactly three skin condition names.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()

    print("\n✅ Orchestrator result:")
    print(result)
    return result

test_orchestrator()
```

Run it:
```powershell
python test_models.py
```

**What to look for:**
- CrewAI prints the agent working through the task
- The final output lists 3 skin conditions
- No connection errors

**If it fails with a connection error:** Make sure `ollama serve` is running in another terminal.  
**If it fails with a 404:** Run `ollama list` and check the exact model name.

---

### Test B — Text Medical Model (MedAIBase/MedGemma1.5:4b)

Add this function to `test_models.py` and call it:

```python
from config import TEXT_LLM

def test_text_model():
    print("\n" + "="*50)
    print("TEST B: Text Medical LLM (MedAIBase/MedGemma1.5:4b)")
    print("="*50)

    agent = Agent(
        role="Medical Text Analyst",
        goal="Extract key clinical information from patient text",
        backstory=(
            "You are a medical AI trained to identify symptoms, "
            "duration, and progression from patient-reported text."
        ),
        llm=TEXT_LLM,
        verbose=True,
    )

    task = Task(
        description=(
            "A patient says: 'I have had an itchy red rash on my left arm "
            "for about 5 days. It started as a small patch but has spread. "
            "It gets worse in the morning.' "
            "Extract the main symptom, body location, duration, and one aggravating factor."
        ),
        expected_output=(
            "Four clearly labelled items: symptom, location, duration, aggravating factor."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()

    print("\n✅ Text model result:")
    print(result)
    return result

test_text_model()
```

Run again. Note whether the output quality feels medically coherent.

---

### Test C — Vision Model (dcarrascosa) — Text Only First

Before testing with an actual image, confirm the vision model responds to text queries:

```python
from config import VISION_LLM

def test_vision_model_text():
    print("\n" + "="*50)
    print("TEST C: Vision LLM — text mode (dcarrascosa)")
    print("="*50)

    agent = Agent(
        role="Dermatology Visual Analyst",
        goal="Analyse skin lesion characteristics precisely",
        backstory=(
            "You are a dermatology specialist trained to analyse skin lesion "
            "images. You describe findings clinically and objectively."
        ),
        llm=VISION_LLM,
        verbose=True,
    )

    task = Task(
        description=(
            "Describe what clinical features you would look for when assessing "
            "lesion colour in a patient with medium-brown skin tone. "
            "Be specific about colour terminology."
        ),
        expected_output=(
            "A concise clinical description of colour assessment approach, "
            "mentioning 3–4 specific colour terms used in dermatology."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff()

    print("\n✅ Vision model (text mode) result:")
    print(result)
    return result

test_vision_model_text()
```

> **Why text mode first?** Image input requires a custom Tool (Chapter 05). For now, confirm the model itself is accessible before adding the complexity of image encoding.

---

## Step 4 — Understanding What LiteLLM Does Behind the Scenes

Run this quick Python snippet (not in a file — paste directly in your Python REPL):

```python
import litellm
litellm.set_verbose = True

from config import ORCHESTRATOR_LLM
# Now re-run test_orchestrator() — you'll see the raw HTTP calls LiteLLM makes to Ollama
```

You'll see LiteLLM translate the OpenAI-format call into Ollama's `/api/chat` endpoint. This is valuable to understand — if something breaks, reading these raw calls tells you exactly what's going wrong.

Turn it off when done:
```python
litellm.set_verbose = False
```

---

## Step 5 — Common Error Reference

Bookmark this. You'll encounter these:

| Error | Cause | Fix |
|-------|-------|-----|
| `Connection refused on localhost:11434` | Ollama not running | `ollama serve` in a terminal |
| `model 'ollama/qwen2.5:7b' not found` | Model not pulled or wrong name | `ollama list` then check spelling |
| `litellm.exceptions.AuthenticationError` | LiteLLM thinks you're calling OpenAI | Check your `base_url` is set correctly in `LLM()` |
| `Timeout` | Model loading from disk (first call) | Wait up to 2 minutes, especially for 7.8 GB model |
| `Output does not match expected_output format` | Model ignored the format instruction | Make `expected_output` more specific, or add format instructions to `description` |

---

## Checkpoint ✅

- [ ] `config.py` exists with all three LLM handles
- [ ] Test A (qwen2.5:7b) produces a list of 3 skin conditions
- [ ] Test B (MedGemma text) correctly extracts symptom / location / duration / aggravating factor
- [ ] Test C (vision model text mode) produces dermatology colour terminology
- [ ] You understand what LiteLLM does between CrewAI and Ollama
- [ ] You know how to read `ollama list` to get the exact model name

Delete `test_models.py` when all tests pass — it was scaffolding:
```powershell
Remove-Item test_models.py
```

---

*Next → `04_BIODATA_AGENT.md`*
