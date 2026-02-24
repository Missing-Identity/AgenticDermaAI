# Chapter 05 — The Image Analysis Tool + Five Lesion Agents

**Goal:** Build one shared `ImageAnalysisTool` that sends an image to the Ollama vision model, then build all five lesion agents (Colour, Surface/Texture, Levelling, Border, Shape) that use it. Test each agent individually before combining them.

**Time estimate:** 60–90 minutes

---

## Why One Tool, Five Agents?

All five lesion agents need to independently examine the image. Each specialist looks at the same photograph but with a completely different clinical focus — the Colour Analyst assesses pigmentation, the Texture Analyst assesses surface characteristics, the Border Analyst focuses on the lesion edge, and the Shape Analyst focuses on the overall geometric form. Each forms its own independent conclusion.

```
ImageAnalysisTool (called directly in Python — NOT via CrewAI tool-calling)
  ├── Colour Analyst prompt    → agent's own colour observations → Colour Agent task
  ├── Texture Analyst prompt   → agent's own texture observations → Texture Agent task
  ├── Elevation Analyst prompt → agent's own elevation observations → Elevation Agent task
  ├── Border Analyst prompt    → agent's own border observations → Border Agent task
  └── Shape Analyst prompt     → agent's own shape observations → Shape Agent task
```

The `ImageAnalysisTool._run()` call **is the agent examining the image.** The same `VISION_LLM` (medgemma) that powers the agent IS the vision model inside the tool. Each call uses a comprehensive, specialist-specific prompt so the model forms a complete independent clinical view of the image — it is not writing a generic description for someone else to interpret. The agent then validates and structures its own observations into the required JSON schema.

> **Important — Why the vision call is made in Python, not inside the CrewAI agent loop:**
>
> Local Ollama vision models (including `dcarrascosa/medgemma-1.5-4b-it`) do **not** support the OpenAI-style function/tool-calling format that CrewAI requires. If you give these models a tool and ask them to invoke it, they output a ` ```tool_code ``` ` block (Google's notebook format) instead of a proper JSON tool call. CrewAI cannot parse this and retries endlessly — causing an **infinite loop**.
>
> **The pattern used throughout this project:**
> 1. Call `ImageAnalysisTool._run()` directly in Python with a **comprehensive, first-person specialist prompt** — the model IS the agent, looking at the image from their clinical specialty
> 2. The agent's task description presents the result as **"Your clinical observations from the image"** — because it IS the agent's own examination
> 3. The agent then validates the clinical observations against biodata context and formats the output as structured JSON
>
> This preserves full clinical independence for each agent while completely sidestepping the tool-calling incompatibility.

---

## Part A — The ImageAnalysisTool

### Step 1 — Understand What the Tool Must Do

The tool needs to:
1. Accept an image file path and a clinical analysis prompt
2. Read the image file from disk
3. Encode it to base64 (Ollama's required format)
4. Send it to the Ollama vision model with a POST request to `/api/chat`
5. Return the model's text response

This is a **synchronous** tool — CrewAI's `BaseTool._run()` is synchronous. We use `httpx` in sync mode.

### Step 2 — Create the Tool File

**`tools/image_tool.py`**

Type this out yourself, section by section:

```python
# tools/image_tool.py
# ImageAnalysisTool: sends a skin image to the Ollama vision model
# and returns a clinical analysis based on the provided prompt.

import base64
import os
import httpx
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = "dcarrascosa/medgemma-1.5-4b-it:Q4_K_M"


class ImageAnalysisInput(BaseModel):
    image_path: str = Field(
        description="Absolute or relative path to the skin image file (jpg, png, webp)"
    )
    clinical_prompt: str = Field(
        description=(
            "The specific clinical question to answer about the image. "
            "Be precise about what aspect to analyse."
        )
    )


class ImageAnalysisTool(BaseTool):
    name: str = "image_analysis"
    description: str = (
        "Analyses a skin image using a vision AI model. "
        "Provide the image file path and a specific clinical prompt about what to observe. "
        "Returns a clinical description of the requested visual feature. "
        "Use this tool whenever you need to examine a skin lesion photograph."
    )
    args_schema: type[BaseModel] = ImageAnalysisInput

    def _run(self, image_path: str, clinical_prompt: str) -> str:
        # Step 1: Validate the file exists
        if not os.path.exists(image_path):
            return f"ERROR: Image file not found at path: {image_path}"

        # Step 2: Validate it's an image type we support
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            return f"ERROR: Unsupported image format '{ext}'. Use jpg, png, or webp."

        # Step 3: Read and base64-encode the image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Step 4: Build the Ollama API payload
        # The vision model receives both the image and the clinical prompt
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": clinical_prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {
                # 0.2: enough diversity to prevent deterministic repetition loops
                # without making clinical descriptions unpredictable.
                "temperature": 0.2,
                # Hard cap on output length — colour/texture/elevation/border
                # descriptions never need more than ~150 tokens; 400 is generous.
                "num_predict": 400,
                # Primary guard against the "The lesion is not X. The lesion is not Y."
                # infinite repetition pattern that small vision LLMs exhibit when
                # temperature is too low.  1.25 penalises recently seen tokens strongly
                # enough to break the loop while still allowing clinical repetition of
                # normal dermatology vocabulary.
                "repeat_penalty": 1.25,
                # Sliding window of 64 tokens over which repeat_penalty is applied.
                "repeat_last_n": 64,
            },
        }

        # Step 5: Call the Ollama API
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=120.0,   # vision models take longer to load
            )
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]

        except httpx.ConnectError:
            return "ERROR: Cannot connect to Ollama. Ensure 'ollama serve' is running."
        except httpx.HTTPStatusError as e:
            return f"ERROR: Ollama API returned {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"ERROR: Unexpected error during image analysis: {str(e)}"
```

### Step 3 — Test the Tool Directly (No Agent Yet)

Before giving this tool to an agent, test it in isolation. Create `test_image_tool.py`:

```python
# test_image_tool.py
from tools.image_tool import ImageAnalysisTool

tool = ImageAnalysisTool()

# Replace this with the actual path to your test skin image
IMAGE_PATH = r"C:\path\to\your\test_skin_image.jpg"

print("Testing ImageAnalysisTool directly...")
result = tool._run(
    image_path=IMAGE_PATH,
    clinical_prompt=(
        "You are a dermatology specialist. Look at this skin image. "
        "Describe the colour of the lesion compared to the surrounding skin. "
        "Use clinical dermatology colour terminology (erythematous, violaceous, "
        "hyperpigmented, hypopigmented, etc.)."
    )
)

print("\n--- Tool Output ---")
print(result)
```

Run it:
```powershell
python test_image_tool.py
```

**What to verify:**
- You get a clinical colour description (not an error)
- The description uses dermatology terminology
- It specifically compares lesion colour to surrounding skin

**If you get a timeout:** The vision model may be loading from disk. Wait up to 2 minutes and try again.

**If you get an image format error:** Check your image path has no typos. Use raw strings (`r"C:\..."`) to avoid backslash issues on Windows.

---

## Part B — The Four Lesion Agents

All four agents go in one file: **`agents/lesion_agents.py`**

### Step 4 — Set Up the File and Pydantic Outputs

Type the imports and output models at the top of the file:

```python
# agents/lesion_agents.py
# Four specialised agents that each independently examine a skin lesion image.
# Each agent uses the ImageAnalysisTool via direct Python call (not CrewAI tool-calling)
# to run its own specialist examination of the image before the crew executes.

from typing import Literal
from pydantic import Field, field_validator
from crewai import Agent, Task
from config import VISION_LLM
from utils.resilient_base import ResilientBase


# ── Output schemas ────────────────────────────────────────────────────────────
# These Pydantic models define exactly what each agent must return.
# CrewAI will validate the agent's output against these.
# All four inherit ResilientBase so that markdown code-fence wrappers and
# other garbage characters the LLM emits around the JSON are stripped before
# Pydantic parses the output.

class ColourOutput(ResilientBase):
    lesion_colour: str = Field(description="Clinical colour description of the lesion")
    reason: str = Field(default="", description="Clinical reasoning behind the colour assessment")

    @field_validator("lesion_colour", "reason", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""


class SurfaceOutput(ResilientBase):
    surface: Literal["dry", "scaly", "blistered", "crusted", "smooth", "weeping", "mixed"] = Field(
        description="Primary surface characteristic"
    )
    reason: str = Field(default="", description="Clinical reasoning with reference to patient age/sex if relevant")

    @field_validator("reason", mode="before")
    @classmethod
    def coerce_null_str(cls, v):
        return v if v is not None else ""

    @field_validator("surface", mode="before")
    @classmethod
    def normalise_surface(cls, v: str) -> str:
        """Map free-text LLM surface descriptions to the allowed Literal values."""
        if not isinstance(v, str):
            return v
        lower = v.lower()
        if "blister" in lower or "vesicl" in lower or "bulla" in lower:
            return "blistered"
        if "crust" in lower or "impetiginiz" in lower:
            return "crusted"
        if "weep" in lower or "exudat" in lower or "ooze" in lower or "moist" in lower:
            return "weeping"
        if "scaly" in lower or "scale" in lower or "squam" in lower or "flak" in lower:
            return "scaly"
        if "dry" in lower or "xerosis" in lower:
            return "dry"
        if "smooth" in lower or "glossy" in lower or "shiny" in lower:
            return "smooth"
        return "mixed"


class LevellingOutput(ResilientBase):
    levelling: Literal["raised", "flat", "depressed"] = Field(
        description="Elevation of the lesion relative to surrounding skin"
    )
    reason: str = Field(default="", description="Clinical reasoning for the elevation assessment")

    @field_validator("reason", mode="before")
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
        if any(w in lower for w in ("raised", "elevat", "dome", "papule", "nodule", "plaque")):
            return "raised"
        if any(w in lower for w in ("depress", "atrophic", "pitted", "indented", "concave")):
            return "depressed"
        if any(w in lower for w in ("flat", "macular", "macule", "level", "flush")):
            return "flat"
        return "raised"


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
```

> **Why free-text `str` fields for Border and Shape?** Unlike elevation (raised/flat/depressed), border and shape characteristics have too many valid clinical descriptions to constrain to a fixed `Literal`. Forcing the model into a 4-option bucket (e.g. `regular/irregular/well-defined/not well-defined`) discards useful clinical nuance. Instead, the model is free to report its own clinical finding — and downstream agents (Differential, CMO) receive richer, more accurate input.
>
> **Why `ResilientBase`?** All five output classes inherit from `utils.resilient_base.ResilientBase` instead of `BaseModel`. `ResilientBase` overrides `model_validate_json` to extract the outermost `{ … }` block from the raw LLM string before handing it to Pydantic's JSON parser. This silently strips markdown code fences (`` ```json … ``` ``) and any other surrounding text the LLM emits adjacent to the JSON.

### Step 5 — Lesion Colour Agent

Add this to `agents/lesion_agents.py`:

```python
# ── Agent 1: Lesion Colour ────────────────────────────────────────────────────
# No tools on this agent. The agent examines the image directly via ImageAnalysisTool._run()
# before the crew runs — the result is the agent's own clinical observation, not a
# third-party description. The task description frames it as "your observations".

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
            'A JSON object with exactly two keys:\n'
            '"lesion_colour": a clinical colour description using dermatology terminology\n'
            '"reason": 1-2 sentences of clinical reasoning mentioning skin tone context\n'
            'Example: {"lesion_colour": "erythematous with violaceous borders", '
            '"reason": "On medium-dark skin, the redness appears as a darker violaceous hue..."}'
        ),
        agent=agent,
        output_pydantic=ColourOutput,
        context=context,
    )
```

### Step 6 — Test the Colour Agent

**`test_lesion_colour.py`** (project root):

```python
from crewai import Crew, Process
from agents.biodata_agent import create_biodata_agent, create_biodata_task
from agents.lesion_agents import create_colour_agent, create_colour_task
from tools.image_tool import ImageAnalysisTool

IMAGE_PATH = r"C:\path\to\your\test_skin_image.jpg"

# The Colour Analyst independently examines the image.
# The prompt is written from the agent's specialist perspective —
# the model IS the agent looking at the image, forming its own clinical view.
print("Colour Analyst examining image...")
_tool = ImageAnalysisTool()
vision_result = _tool._run(
    image_path=IMAGE_PATH,
    clinical_prompt=(
        "You are a Dermatology Colour Analyst directly examining a skin lesion. "
        "Look at the entire image. Describe the complete colour profile of the lesion: "
        "name all shades present using clinical terminology (erythematous, violaceous, "
        "hyperpigmented, hypopigmented, melanotic, etc.), note any colour variation "
        "or gradient across the lesion, and compare the lesion colour to the surrounding skin. "
        "Form your own independent clinical colour assessment."
    ),
)
print(f"Colour Analyst observations: {vision_result}\n")

biodata_agent = create_biodata_agent()
biodata_task = create_biodata_task(biodata_agent)

colour_agent = create_colour_agent()
colour_task = create_colour_task(colour_agent, IMAGE_PATH, biodata_task, vision_result=vision_result)

crew = Crew(
    agents=[biodata_agent, colour_agent],
    tasks=[biodata_task, colour_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
print("\n=== COLOUR AGENT OUTPUT ===")
print(result)
print("\nPydantic output:", colour_task.output.pydantic if colour_task.output else "None")
```

Run it. Verify the output is valid JSON with `lesion_colour` and `reason` fields.

**Delete `test_lesion_colour.py` when it passes.**

---

### Step 7 — Lesion Surface/Texture Agent

Add to `agents/lesion_agents.py`:

```python
# ── Agent 2: Lesion Surface / Texture ─────────────────────────────────────────

def create_texture_agent() -> Agent:
    return Agent(
        role="Dermatology Texture and Surface Analyst",
        goal=(
            "Characterise the surface texture of the skin lesion with clinical precision. "
            "Account for patient age and sex when interpreting texture findings."
        ),
        backstory=(
            "You are a specialist in dermoscopy and lesion morphology. "
            "You know that scaling is more pronounced in elderly skin due to reduced sebum production. "
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
            'A JSON object with exactly two keys:\n'
            '"surface": one of: dry / scaly / blistered / crusted / smooth / weeping / mixed\n'
            '"reason": 1-2 sentences explaining the finding with age/sex context if relevant\n'
            'Example: {"surface": "scaly", "reason": "Fine silvery scaling noted... '
            'In a 45-year-old male, this pattern is consistent with..."}'
        ),
        agent=agent,
        output_pydantic=SurfaceOutput,
        context=context,
    )
```

**Test it the same way** with a `test_lesion_texture.py`. The texture agent examines the image independently using a specialist prompt:

```python
vision_result = ImageAnalysisTool()._run(
    image_path=IMAGE_PATH,
    clinical_prompt=(
        "You are a Dermatology Texture and Surface Analyst directly examining a skin lesion. "
        "Look at the entire image. Describe the complete surface texture of the lesion: "
        "note scaling, dryness, crusting, blistering, weeping, smoothness, or any other "
        "surface irregularity you observe. Describe the texture in clinical terms and note "
        "any areas where surface characteristics vary. "
        "Form your own independent clinical texture assessment."
    ),
)
texture_task = create_texture_task(texture_agent, IMAGE_PATH, biodata_task, vision_result=vision_result)
```

---

### Step 8 — Lesion Levelling Agent

Add to `agents/lesion_agents.py`:

```python
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
            "provide your final elevation assessment (raised / flat / depressed). "
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
            'A JSON object with exactly two keys:\n'
            '"levelling": exactly one of: raised / flat / depressed (no other values allowed)\n'
            '"reason": 1-2 sentences of clinical reasoning referencing specific visual cues\n'
            'Example: {"levelling": "raised", "reason": "Visible shadow at lesion margins '
            'and central dome shape indicate elevation above surrounding skin level."}'
        ),
        agent=agent,
        output_pydantic=LevellingOutput,
        context=context,
    )
```

---

### Step 9 — Lesion Border Agent

Add to `agents/lesion_agents.py`:

```python
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
            'A JSON object with exactly two keys:\n'
            '"border": a clinical description of the lesion border and edge characteristics\n'
            '"reason": 1-2 sentences of clinical reasoning for the border assessment\n'
            'Example: {"border": "irregular, notched edge with abrupt transition to surrounding skin", '
            '"reason": "The lesion periphery shows multiple sharp notches and asymmetric extension, '
            'raising concern for dysplastic growth."}'
        ),
        agent=agent,
        output_pydantic=BorderOutput,
        context=context,
    )
```

### Step 9b — Lesion Shape Agent

Add to `agents/lesion_agents.py`:

```python
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
            'A JSON object with exactly two keys:\n'
            '"shape": a clinical description of the lesion\'s geometric shape and outline\n'
            '"reason": 1-2 sentences of clinical reasoning for the shape assessment\n'
            'Example: {"shape": "oval with mild asymmetry along the long axis", '
            '"reason": "The lesion outline describes an elongated oval form with slight variation '
            'at the superior pole, consistent with early asymmetric expansion."}'
        ),
        agent=agent,
        output_pydantic=ShapeOutput,
        context=context,
    )
```

---

## Step 10 — Test All Five Lesion Agents Together

Now test all five running sequentially with shared biodata context. The vision model is called **five times** — once per agent, each from that agent's specialist perspective using a comprehensive first-person examination prompt:

```python
# test_all_lesion_agents.py
from crewai import Crew, Process
from agents.biodata_agent import create_biodata_agent, create_biodata_task
from agents.lesion_agents import (
    create_colour_agent, create_colour_task,
    create_texture_agent, create_texture_task,
    create_levelling_agent, create_levelling_task,
    create_border_agent, create_border_task,
    create_shape_agent, create_shape_task,
)
from tools.image_tool import ImageAnalysisTool

IMAGE_PATH = r"C:\path\to\your\test_skin_image.jpg"

# Each agent independently examines the image using the vision model (VISION_LLM/medgemma).
# The prompt is written from each specialist's perspective — the model IS the agent,
# looking at the full image and forming its own complete clinical view.
# (Direct Python call required because medgemma does not support CrewAI tool-calling format.)
print("Running independent vision examination (5 specialist passes)...")
_tool = ImageAnalysisTool()

colour_vision = _tool._run(
    image_path=IMAGE_PATH,
    clinical_prompt=(
        "You are a Dermatology Colour Analyst directly examining a skin lesion. "
        "Look at the entire image. Describe the complete colour profile of the lesion: "
        "name all shades present using clinical terminology (erythematous, violaceous, "
        "hyperpigmented, hypopigmented, melanotic, etc.), note any colour variation "
        "or gradient across the lesion, and compare the lesion colour to the surrounding skin. "
        "Form your own independent clinical colour assessment."
    ),
)
texture_vision = _tool._run(
    image_path=IMAGE_PATH,
    clinical_prompt=(
        "You are a Dermatology Texture and Surface Analyst directly examining a skin lesion. "
        "Look at the entire image. Describe the complete surface texture of the lesion: "
        "note scaling, dryness, crusting, blistering, weeping, smoothness, or any other "
        "surface irregularity you observe. Describe the texture in clinical terms and note "
        "any areas where surface characteristics vary. "
        "Form your own independent clinical texture assessment."
    ),
)
level_vision = _tool._run(
    image_path=IMAGE_PATH,
    clinical_prompt=(
        "You are a Dermatology Morphology and Elevation Analyst directly examining a skin lesion. "
        "Look at the entire image. Determine the complete elevation profile of the lesion: "
        "is it raised, flat, or depressed relative to surrounding skin? "
        "Describe any shadow patterns at the lesion edge, dome or concave shape, "
        "and 3D visual cues visible in the photograph. "
        "Form your own independent clinical elevation assessment."
    ),
)
border_vision = _tool._run(
    image_path=IMAGE_PATH,
    clinical_prompt=(
        "You are a Dermatology Border Analyst directly examining a skin lesion. "
        "Look at the entire image. Assess the border and edge characteristics of the lesion: "
        "describe how the lesion transitions to surrounding skin, the contour of the edge, "
        "any notching, asymmetry, satellite lesions, or ABCDE red flags at the periphery. "
        "Form your own independent clinical border assessment."
    ),
)
shape_vision = _tool._run(
    image_path=IMAGE_PATH,
    clinical_prompt=(
        "You are a Dermatology Shape Analyst directly examining a skin lesion. "
        "Look at the entire image. Describe the geometric shape and overall outline of the lesion: "
        "its form (circular, oval, linear, annular, polycyclic, etc.), overall symmetry, "
        "and any distinctive structural characteristics you observe. "
        "Form your own independent clinical shape assessment."
    ),
)
print("Independent vision examination complete.\n")

# Create agents
biodata_agent   = create_biodata_agent()
colour_agent    = create_colour_agent()
texture_agent   = create_texture_agent()
levelling_agent = create_levelling_agent()
border_agent    = create_border_agent()
shape_agent     = create_shape_agent()

# Create tasks — each receives its own specialist vision observations
biodata_task    = create_biodata_task(biodata_agent)
colour_task     = create_colour_task(colour_agent, IMAGE_PATH, biodata_task, vision_result=colour_vision)
texture_task    = create_texture_task(texture_agent, IMAGE_PATH, biodata_task, vision_result=texture_vision)
levelling_task  = create_levelling_task(levelling_agent, IMAGE_PATH, biodata_task, vision_result=level_vision)
border_task     = create_border_task(border_agent, IMAGE_PATH, biodata_task, vision_result=border_vision)
shape_task      = create_shape_task(shape_agent, IMAGE_PATH, biodata_task, vision_result=shape_vision)

crew = Crew(
    agents=[biodata_agent, colour_agent, texture_agent, levelling_agent, border_agent, shape_agent],
    tasks=[biodata_task, colour_task, texture_task, levelling_task, border_task, shape_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()

print("\n" + "="*60)
print("ALL LESION AGENTS — COMBINED RESULTS")
print("="*60)

for task, label in [
    (colour_task,   "COLOUR"),
    (texture_task,  "TEXTURE"),
    (levelling_task,"LEVELLING"),
    (border_task,   "BORDER"),
    (shape_task,    "SHAPE"),
]:
    print(f"\n[{label}]")
    if task.output and task.output.pydantic:
        print(task.output.pydantic.model_dump_json(indent=2))
    else:
        print(task.output.raw if task.output else "No output")
```

Run it and observe:
- The vision model runs 5 times upfront — each call is a full specialist examination with its own comprehensive prompt
- Each agent's task description says **"Your clinical observations from the image"** — it IS the agent's own examination
- Each agent then validates its observations against biodata context and structures the output as JSON
- The 5 agents produce independent conclusions — this diversity of perspectives enriches the downstream differential

**Delete `test_all_lesion_agents.py` when it passes.**

---

## Known Failure Mode — MedGemma Repetition Loop

Small vision LLMs like MedGemma 1.5b can enter a deterministic **repetition loop** if the Ollama generation parameters are not set carefully. The symptom is the model producing hundreds of identical sentences such as:

```
The lesion is not hyperpigmented. The lesion is not hypopigmented. The lesion is not melanotic.
The lesion is not violaceous. The lesion is not hyperpigmented. The lesion is not hypopigmented...
```

**Why it happens:**
A very low temperature (e.g. 0.1 or 0.0) eliminates token diversity. Once the model produces a "not X, not Y, not Z" pattern, the probability mass collapses onto repeating the same tokens because that is what the model has been predicting so far. Without a `repeat_penalty`, there is nothing to push it out of the loop.

**Failure cascade:**
1. `ImageAnalysisTool._run()` produces the repetitive text as the `vision_result`
2. This garbage string is injected into the lesion task description
3. The lesion agent task fails because the model cannot extract a 2-field JSON from the garbled input
4. The crew fails with: *"Failed to convert text into a Pydantic model"*
5. The recovery crew kicks in and runs the orchestrator in isolation

**The fix applied everywhere MedGemma is called:**
- `repeat_penalty: 1.25` — directly penalises recently seen tokens, breaking the loop before it starts. Applied in both `tools/image_tool.py` (direct httpx call) and `config.py` `VISION_LLM` (all CrewAI-managed MedGemma agents: lesion, clarification, differential, treatment).
- `repeat_last_n: 64` — sliding window of 64 tokens for the penalty check
- `temperature: 0.2` (image tool only) — enough diversity to break out of repetition while keeping descriptions consistent
- `num_predict: 400` (image tool only) — hard cap on output tokens (colour/texture/etc. descriptions need ~100–150 tokens)

> **Do NOT add `num_ctx` to `VISION_LLM` in `config.py`.** MedGemma 1.5b has a small native context window; forcing a larger `num_ctx` wastes VRAM and can destabilise the model. The `num_ctx: 16384` increase was applied only to `ORCHESTRATOR_LLM` (Qwen 2.5), which receives all 9 upstream task outputs as context and legitimately requires it.

---

## Checkpoint ✅

- [ ] `tools/image_tool.py` exists with `ImageAnalysisTool`
- [ ] Calling `tool._run(path, prompt)` directly returns a clinical assessment (test this in Step 3)
- [ ] `agents/lesion_agents.py` has all 5 agents and 5 task factories
- [ ] **No agent has `tools=[...]`** — each agent examines the image via direct `_run()` call before the crew runs
- [ ] Each task factory accepts a `vision_result: str = None` parameter
- [ ] Each task description (when `vision_result` is provided) says **"You directly examined..."** and **"Your clinical observations from the image:"** — not "A vision AI..."
- [ ] The 5 vision prompts in your test script are **comprehensive and first-person** (each written as the specialist agent examining the full image)
- [ ] Each task has an `output_pydantic` schema
- [ ] All five output schemas (`ColourOutput`, `SurfaceOutput`, `LevellingOutput`, `BorderOutput`, `ShapeOutput`) inherit from `ResilientBase`, not `BaseModel`
- [ ] `utils/resilient_base.py` exists and `ResilientBase` is imported at the top of `lesion_agents.py`
- [ ] `BorderOutput` and `ShapeOutput` use free-text `str` fields — no `Literal` constraint — letting the model form its own clinical description
- [ ] Running all 5 agents together produces 5 independent structured JSON outputs
- [ ] Biodata context is visible in the texture and levelling agent reasoning
- [ ] No infinite tool-call loop occurs (agents complete in a single LLM pass each)

---

*Next → `06_DECOMPOSITION_AGENT.md`*
