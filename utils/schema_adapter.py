"""
Convert free-text clinical outputs into validated Pydantic objects.

MedGemma is allowed to answer in unconstrained prose. This adapter then
uses a formatter model to map the prose to each target schema.
"""

from __future__ import annotations

import json
from typing import Any, get_args, get_origin

import httpx
from pydantic.fields import PydanticUndefined

from config import OLLAMA_BASE_URL, FORMATTER_MODEL


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def _unwrap_if_nested(json_str: str) -> str:
    """
    If the JSON is a single-key wrapper dict (e.g. {"FinalDiagnosis": {...}}),
    return the inner dict as a JSON string so it can be validated against a
    flat Pydantic schema.  Only unwraps one level; leaves everything else alone.
    """
    try:
        obj = json.loads(json_str)
        if isinstance(obj, dict) and len(obj) == 1:
            inner = next(iter(obj.values()))
            if isinstance(inner, dict):
                return json.dumps(inner)
    except Exception:
        pass
    return json_str


def _repair_truncated_json(text: str) -> str:
    """Attempt to close unclosed braces/brackets in a truncated JSON string."""
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    stack = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in ("{", "["):
                stack.append("}" if ch == "{" else "]")
            elif ch in ("}", "]"):
                if stack and stack[-1] == ch:
                    stack.pop()
    if in_string:
        text += '"'
    text += "".join(reversed(stack))
    return text


def _default_for_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        if annotation is bool:
            return False
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if annotation is str:
            return ""
        return None

    if origin in (list, tuple, set):
        return []
    if origin is dict:
        return {}
    if origin is Any:
        return ""
    if origin is type(None):
        return None
    if origin is list:
        return []
    if origin is tuple:
        return []
    if origin is set:
        return []
    if origin is dict:
        return {}
    if origin is None:
        return ""
    if origin is not None and str(origin).endswith("Literal"):
        return args[0] if args else ""
    if origin is not None and str(origin).endswith("Union"):
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if not non_none:
            return None
        return _default_for_annotation(non_none[0])

    return ""


def _build_fallback_dict(model_cls: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        if field.default is not PydanticUndefined:
            data[name] = field.default
            continue
        if field.default_factory is not None:
            data[name] = field.default_factory()
            continue
        data[name] = _default_for_annotation(field.annotation)
    return data


def _call_formatter(prompt: str) -> str:
    payload = {
        "model": FORMATTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 3000,
            "num_ctx": 8192,
            "repeat_penalty": 1.1,
        },
    }
    res = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120.0)
    res.raise_for_status()
    content = res.json().get("message", {}).get("content", "")
    return content if isinstance(content, str) else str(content)


def adapt_to_model(raw_text: str, model_cls: Any, schema_name: str) -> tuple[Any, dict[str, Any]]:
    """
    Convert raw free-text output into `model_cls`.
    Returns: (parsed_model, metadata)
    """
    # Fast path: if the raw text already contains valid JSON that satisfies the schema,
    # use it directly without paying the cost of a formatter LLM call.
    if raw_text:
        try:
            raw_block = _extract_json_block(raw_text)
            if not raw_block:
                raise ValueError("No JSON block found in raw text")
            unwrapped = _unwrap_if_nested(raw_block)
            repaired = _repair_truncated_json(unwrapped)
            parsed = model_cls.model_validate_json(repaired)
            print(f"[SchemaAdapter] {schema_name}: direct parse succeeded — skipping formatter")
            return parsed, {"status": "direct", "error": "", "attempts": 0}
        except Exception as direct_err:
            print(f"[SchemaAdapter] {schema_name}: direct parse failed ({direct_err}), falling back to formatter")

    schema_json = json.dumps(model_cls.model_json_schema(), ensure_ascii=True)
    last_error = ""

    for attempt in (1, 2):
        strict_extra = (
            ""
            if attempt == 1
            else (
                "Your previous response was invalid JSON for this schema.\n"
                f"Validation error: {last_error}\n"
                "Return ONLY one valid JSON object, no prose, no markdown.\n"
            )
        )
        prompt = (
            f"You are a strict JSON formatter.\n"
            f"Target schema name: {schema_name}\n"
            f"Target JSON schema: {schema_json}\n\n"
            f"Raw clinical text to convert:\n{raw_text}\n\n"
            f"{strict_extra}"
            "Output requirements:\n"
            "- Return exactly one JSON object\n"
            "- Do not include markdown fences\n"
            "- Do not include comments\n"
            "- Use empty strings/lists when uncertain\n"
        )

        try:
            formatted = _call_formatter(prompt)
            raw_block = _extract_json_block(formatted)
            unwrapped = _unwrap_if_nested(raw_block)
            repaired = _repair_truncated_json(unwrapped)
            parsed = model_cls.model_validate_json(repaired)
            return parsed, {"status": "ok" if attempt == 1 else "recovered", "error": "", "attempts": attempt}
        except Exception as e:
            last_error = str(e)

    try:
        fallback = _build_fallback_dict(model_cls)
        parsed = model_cls.model_validate(fallback)
        return parsed, {"status": "defaulted", "error": last_error, "attempts": 2}
    except Exception:
        # Absolute last-resort fallback
        parsed = model_cls.model_construct(**_build_fallback_dict(model_cls))
        return parsed, {"status": "defaulted", "error": last_error, "attempts": 2}

