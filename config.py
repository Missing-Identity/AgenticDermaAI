import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Model handles ─────────────────────────────────────────────────────────────
#
# Model assignment policy:
#   VISION_LLM       → Free-text medical reasoning from image-derived tasks.
#   TEXT_LLM         → Biodata agent only (data formatting, no clinical reasoning).
#   ORCHESTRATOR_LLM → Long-context text reasoning + tool calling.
#   FORMATTER_LLM    → Raw-text to strict schema conversion (schema adapter).

# Medical vision model — all clinical reasoning agents
# Supports image input via base64; strong medical knowledge base
#
# extra_body / options:
#   repeat_penalty=1.25 — penalises recently seen tokens to prevent MedGemma 1.5b
#     from entering deterministic repetition loops (e.g. "The lesion is not X.
#     The lesion is not Y." repeated hundreds of times). Applied to every
#     CrewAI-managed MedGemma task: lesion agents, clarification, differential,
#     and treatment.
#   repeat_last_n=64 — sliding window of 64 tokens over which the penalty is
#     calculated; wide enough to catch sentence loops, narrow enough not to
#     suppress repeated-but-valid medical terms.
#
# NOTE: do NOT add num_ctx here. MedGemma 1.5b has a small native context
# window; forcing a large num_ctx wastes VRAM and can destabilise the model.
VISION_LLM = LLM(
    model="ollama/hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M",
    base_url=OLLAMA_BASE_URL,
    max_tokens=512,
    extra_body={"options": {"repeat_penalty": 1.25, "repeat_last_n": 64}},
)

# Text model — Biodata agent only (data formatting, not clinical reasoning)
# Minimal context needed: profile JSON is small, no upstream outputs in context.
TEXT_LLM = LLM(
    model="ollama/qwen2.5:7b-instruct",
    base_url=OLLAMA_BASE_URL,
    timeout=120,
)

# Tool-calling / orchestration model — Research agent + Orchestrator agent
# qwen2.5:7b: best-in-class for strict JSON/Pydantic schema compliance among
# local models; reliably separates input context from output schema.
# Supports OpenAI-style function calls required by PubMedSearchTool.
# Preferred over qwen3:8b — qwen3's thinking-mode tokens interfere with
# CrewAI's tool-call parsing, causing placeholder outputs.
#
# num_ctx: The Orchestrator receives all 9 upstream task outputs as context.
# The default Ollama context window (2048 tokens) is too small and causes the
# model to silently hang or produce no output. 16384 comfortably fits the full
# multi-agent context. qwen2.5 supports up to 32768 tokens.
#
# timeout: Hard timeout (seconds) on the Ollama HTTP call. Without this, a
# stuck generation hangs the entire process indefinitely. 360 s (6 min) is
# generous for a local 7B model; CrewAI will raise an exception and the
# recovery crew in derma_crew.py will attempt to salvage the run.
ORCHESTRATOR_LLM = LLM(
    model="ollama/qwen2.5:7b-instruct",
    base_url=OLLAMA_BASE_URL,
    num_ctx=16384,
    timeout=360,
)

# Dedicated formatter model for raw-text → strict schema conversion.
# This runs in the new schema adapter layer and keeps MedGemma free to
# reason in natural clinical prose.
FORMATTER_MODEL = os.getenv("FORMATTER_MODEL", "qwen2.5:7b-instruct")
FORMATTER_LLM = LLM(
    model=f"ollama/{FORMATTER_MODEL}",
    base_url=OLLAMA_BASE_URL,
    num_ctx=8192,
    timeout=180,
)