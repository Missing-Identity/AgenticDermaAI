import base64
import os
import httpx
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"


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
                # 0.2 gives enough token diversity to prevent deterministic repetition
                # loops without making clinical descriptions unpredictable.
                "temperature": 0.2,
                # Hard cap on output length. Colour/texture/elevation/border descriptions
                # never need more than ~150 tokens; 400 is generous.
                "num_predict": 400,
                # Penalise recent tokens. This is the primary guard against the
                # "The lesion is not X. The lesion is not Y." infinite repetition
                # pattern that small vision LLMs (like MedGemma 1.5b) exhibit.
                "repeat_penalty": 1.25,
                # Look back 64 tokens when applying the repeat penalty — enough to
                # catch single-sentence loops without suppressing valid medical terms
                # that naturally reoccur in clinical text (e.g. "the lesion").
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