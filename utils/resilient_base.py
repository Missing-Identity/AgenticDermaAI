"""
Shared Pydantic base class for all agent output models.

LLMs frequently produce JSON that is structurally correct but has one of
several common formatting faults:

  1. Wrapped in markdown code fences  (```json ... ```)
  2. Preceded/followed by narrative text
  3. Unquoted ellipsis placeholder values  (: ...)
  4. String values that start with an ellipsis  ("...actual text")
  5. Invalid JSON escape sequences  (\s, \p, etc.)

_sanitize_json() fixes all of these before handing the string to
Pydantic's JSON parser, so field validators can run normally.
"""

import re
import json
from pydantic import BaseModel


def _sanitize_json(text: str) -> str:
    """Fix the most common LLM JSON formatting faults."""

    # 1. Unwrap markdown code fences if present
    fence = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 2. Extract the outermost { … } block, dropping surrounding prose
    brace = re.search(r'\{.*\}', text, re.DOTALL)
    if brace:
        text = brace.group()

    # 3. Fix unquoted ellipsis placeholder values
    #    e.g.  "surface": ...  →  "surface": ""
    text = re.sub(r'(:\s*)\.\.\.(\s*[,\}])', r'\1""\2', text)

    # 4. Fix string values that open with an ellipsis
    #    e.g.  "reason": "...actual text"  →  "reason": "actual text"
    text = re.sub(r'(:\s*)"\.\.\.([^"]*)"', r'\1"\2"', text)

    # 5. Fix invalid JSON escape sequences
    #    Only attempt if the JSON is currently broken due to an escape error.
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        if 'escape' in str(e).lower() or 'invalid' in str(e).lower():
            # Replace any backslash not followed by a valid JSON escape character.
            # Valid: \" \\ \/ \b \f \n \r \t \uXXXX
            text = re.sub(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', text)

    return text


class ResilientBase(BaseModel):
    """BaseModel subclass that sanitises raw LLM output before JSON parsing."""

    @classmethod
    def model_validate_json(cls, json_data, *, strict=None, context=None):  # type: ignore[override]
        if isinstance(json_data, (bytes, bytearray)):
            json_data = json_data.decode()
        if isinstance(json_data, str):
            json_data = _sanitize_json(json_data)
        return super().model_validate_json(json_data, strict=strict, context=context)
