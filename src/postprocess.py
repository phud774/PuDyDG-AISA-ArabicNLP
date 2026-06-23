from __future__ import annotations

import ast
import json
import re
from typing import Any

from src.constants import ARABIC_DIGIT_TRANSLATION, END_TURN, ID_LIKE_FIELDS
from src.dataset import candidate_function_names, allowed_parameter_properties


def extract_first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    quote = ""
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            continue

        if ch in {'"', "'"}:
            in_string = True
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_object(text: str) -> tuple[dict[str, Any], bool]:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    object_text = extract_first_balanced_object(text)
    if object_text is None:
        return {}, False

    try:
        value = json.loads(object_text)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(object_text)
        except (ValueError, SyntaxError):
            return {}, False

    return (value, True) if isinstance(value, dict) else ({}, False)


def parse_function_name(text: str, candidates: list[str]) -> tuple[str, bool]:
    lower = text.strip().lower()

    tag_match = re.search(
        r"<function_name>\s*([A-Za-z_][A-Za-z0-9_]*)\s*</function_name>",
        text,
    )
    if tag_match:
        name = tag_match.group(1)
        if name in candidates or name == "none":
            return name, True

    for name in candidates:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
            return name, True

    first_line = text.split(END_TURN, 1)[0].strip().splitlines()[0].strip() if text.strip() else ""
    first_token_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", first_line)
    if first_token_match:
        token = first_token_match.group(0)
        if token in candidates or token == "none":
            return token, True

    no_call_markers = [
        "none",
        "false",
        "لا يحتاج",
        "لا تتطلب",
        "لا يتطلب",
        "بدون أداة",
        "لا توجد دالة",
    ]
    if any(marker in lower for marker in no_call_markers):
        return "none", True

    return "none", False


def parse_scalar_from_custom_call(key: str, value: str) -> Any:
    value = value.strip()
    if key in ID_LIKE_FIELDS:
        return value

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    ascii_value = value.translate(ARABIC_DIGIT_TRANSLATION)
    try:
        if re.fullmatch(r"[+-]?\d+", ascii_value):
            return int(ascii_value)
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)", ascii_value):
            return float(ascii_value)
    except ValueError:
        pass
    return value


def parse_direct_output(text: str) -> dict[str, Any]:
    output = {
        "requires_function": False,
        "function_name": "none",
        "arguments": {},
        "think": "",
    }

    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
    if think_match:
        output["think"] = think_match.group(1).strip()

    call_match = re.search(
        r"<start_function_call>\s*call:([A-Za-z_][A-Za-z0-9_]*)\{(.*?)\}"\
        r"\s*<end_function_call>",
        text,
        re.DOTALL,
    )
    if not call_match:
        return output

    output["requires_function"] = True
    output["function_name"] = call_match.group(1)
    args_block = call_match.group(2)

    matches = re.findall(
        r"([A-Za-z_][A-Za-z0-9_]*):(?:<escape>(.*?)<escape>|([^,}]+))",
        args_block,
        re.DOTALL,
    )
    for key, escaped_value, raw_value in matches:
        value = escaped_value if escaped_value != "" else raw_value
        output["arguments"][key] = parse_scalar_from_custom_call(key, value)

    return output


def to_ascii_number_string(value: Any) -> str:
    return str(value).translate(ARABIC_DIGIT_TRANSLATION).replace(",", "").strip()


def cast_by_schema(key: str, value: Any, schema: dict[str, Any]) -> Any:
    if value is None:
        return None
    if key in ID_LIKE_FIELDS:
        return str(value).strip()

    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        expected_type = next((t for t in expected_type if t != "null"), None)

    try:
        if expected_type == "integer":
            if isinstance(value, bool):
                return int(value)
            return int(float(to_ascii_number_string(value)))
        if expected_type == "number":
            number = float(to_ascii_number_string(value))
            return int(number) if number.is_integer() else number
        if expected_type == "boolean":
            if isinstance(value, bool):
                return value
            lowered = str(value).strip().lower()
            if lowered in {"true", "1", "yes", "نعم"}:
                return True
            if lowered in {"false", "0", "no", "لا"}:
                return False
        if expected_type == "string":
            return str(value).strip()
        if expected_type == "array":
            return value if isinstance(value, list) else str(value).strip()
    except (TypeError, ValueError):
        pass
    return value


def sanitize_arguments(
    raw_args: Any,
    row: dict[str, Any],
    function_name: str,
) -> dict[str, Any]:
    properties = allowed_parameter_properties(row, function_name)
    if not isinstance(raw_args, dict):
        return {}

    result: dict[str, Any] = {}
    for key, value in raw_args.items():
        key = str(key)
        if properties and key not in properties:
            continue
        if value is None or value == "":
            continue
        result[key] = cast_by_schema(key, value, properties.get(key, {}))
    return result
