# Chapter 01 — Project Setup

**Goal:** Create the project directory, set up a clean Python environment, and install all dependencies. By the end of this chapter you will have a working skeleton that imports without errors.

**Time estimate:** 15–20 minutes

---

## Step 1 — Create the Project Directory

Open a fresh PowerShell window. Navigate to where you want the project to live (Desktop is fine):

```powershell
cd C:\Users\Unmilan\Desktop
mkdir dermaai-v2
cd dermaai-v2
```

> **Why a fresh directory?**  
> This v2 project is architecturally different from v1. Mixing them will cause import confusion. Keep them completely separate.

Verify you are inside it:
```powershell
pwd
# Should show: C:\Users\Unmilan\Desktop\dermaai-v2
```

---

## Step 2 — Create a Virtual Environment with uv

```powershell
uv venv --python 3.12 .venv
```

Activate it:
```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(dermaai-v2)` or `(.venv)` at the start of your prompt.

> **Always activate this venv** before working on this project. Every terminal session needs it.

Verify the Python version:
```powershell
python --version
# Should show: Python 3.12.x
```

---

## Step 3 — Create the requirements.txt

In your project root (`dermaai-v2/`), create a file called `requirements.txt` and type in the following. **Don't copy-paste blindly — read each line and understand what it is for:**

```
# Core agentic framework
crewai==0.80.0
crewai-tools==0.17.0

# Ollama HTTP client (CrewAI uses LiteLLM internally which calls Ollama)
litellm>=1.40.0

# PubMed access
biopython>=1.84

# HTTP requests (for our custom image tool)
httpx>=0.27.0

# Image handling
pillow>=10.0.0

# PDF generation
reportlab>=4.0.0

# Environment variable management
python-dotenv>=1.0.0

# Data validation
pydantic>=2.0.0
```

**What each group does:**
- `crewai` — the agentic orchestration framework we are building on
- `crewai-tools` — pre-built tool wrappers (we use these as reference, not as black boxes)
- `litellm` — CrewAI routes all LLM calls through LiteLLM, which supports Ollama
- `biopython` — gives us the `Entrez` module to query NCBI/PubMed
- `httpx` — async HTTP client for our custom Ollama image calls
- `pillow` — for reading and validating image files
- `reportlab` — PDF generation at the very end
- `python-dotenv` — loads secrets from a `.env` file so they never appear in code
- `pydantic` — type-safe data models (you'll use these for agent outputs)

---

## Step 4 — Install the Dependencies

```powershell
uv pip install -r requirements.txt
```

This will take 2–4 minutes. Watch the output — if anything fails, note the exact error.

**Common issues:**
- If `crewai` version 0.80.0 isn't found, remove the `==0.80.0` pin and install the latest with `uv pip install crewai crewai-tools`
- If you get a `biopython` build error, run `uv pip install biopython --no-binary biopython`

After install, verify core packages:
```powershell
python -c "import crewai; print(crewai.__version__)"
python -c "import Bio; print(Bio.__version__)"
python -c "import litellm; print(litellm.__version__)"
```

All three should print a version number without errors.

---

## Step 5 — Create the .env File

In the project root, create a file called `.env`:

```
# Ollama base URL (leave as-is unless you changed the port)
OLLAMA_BASE_URL=http://localhost:11434

# Your NCBI API key — get one free at https://www.ncbi.nlm.nih.gov/account/
# Without it, PubMed limits you to 3 requests/second (still usable for dev)
NCBI_API_KEY=your_key_here

# Your email (NCBI requires this for Entrez queries — it is never shared publicly)
NCBI_EMAIL=your@email.com
```

> **Important:** Add `.env` to a `.gitignore` file immediately if you ever use git.  
> Never commit API keys.

---

## Step 6 — Create the Folder Skeleton

Run these one by one in your terminal. Creating them manually teaches you the structure:

```powershell
mkdir tools
mkdir agents
mkdir crew
mkdir reports

# Create __init__.py files (makes each folder a Python package)
New-Item tools\__init__.py -ItemType File
New-Item agents\__init__.py -ItemType File
New-Item crew\__init__.py -ItemType File
```

Your directory should now look like:
```
dermaai-v2/
├── .env
├── requirements.txt
├── tools/
│   └── __init__.py
├── agents/
│   └── __init__.py
├── crew/
│   └── __init__.py
└── reports/
```

Verify it:
```powershell
Get-ChildItem -Recurse -Name
```

---

## Step 7 — Create the Entry Point

Create a file called `main.py` in the project root. For now it will just be a placeholder — you'll fill it in later:

```python
# main.py
# Entry point for DermaAI v2
# This file will be completed in Chapter 10.

def main():
    print("DermaAI v2 — Multi-Agent Dermatology System")
    print("Run chapters 01-09 first to build all components.")

if __name__ == "__main__":
    main()
```

Run it to confirm Python is working:
```powershell
python main.py
# Should print: DermaAI v2 — Multi-Agent Dermatology System
```

---

## Checkpoint ✅

Before moving to the next chapter, confirm all of these:

- [ ] `python --version` shows 3.12.x
- [ ] `.venv` is activated (you see it in your prompt)
- [ ] `python -c "import crewai"` runs without error
- [ ] `python -c "import Bio"` runs without error  
- [ ] The folder structure (`tools/`, `agents/`, `crew/`, `reports/`) exists
- [ ] `python main.py` prints the welcome message
- [ ] `.env` file exists with your NCBI credentials

**If any of these fail, fix them before proceeding. Later chapters assume this baseline is solid.**

---

*Next → `02_CREWAI_CONCEPTS.md`*
