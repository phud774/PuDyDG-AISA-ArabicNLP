#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create argument-focused, train-compatible AISA augmentations with an OpenAI API."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm.auto import tqdm

import aisa_decomposed_multitask_toolinfo as aisa


MODEL_ID = "Llama-3.3-70B-Instruct"


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load only missing variables, without requiring python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'").strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write and incrementally extend an argument-augmentation-only AISA JSONL."
    )
    parser.add_argument("--dataset-id", default=aisa.DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--output", default="outputs/aisa_argument_augmentations_only_ver2.jsonl"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of positive train samples to paraphrase (one augmentation each).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output instead of merging records by id.",
    )
    return parser.parse_args()


def get_user_message(row: dict[str, Any]) -> tuple[int, str]:
    for index in range(len(row.get("messages") or []) - 1, -1, -1):
        message = row["messages"][index]
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return index, message["content"]
    raise ValueError("No user message with text content found.")


def extract_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Response does not contain a JSON object.")
    result = json.loads(content[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Response JSON must be an object.")
    return result


def selected_schema(row: dict[str, Any], function_name: str) -> dict[str, Any]:
    fn = aisa.selected_tool_schema(row, function_name)
    if not fn:
        raise ValueError(f"Selected function {function_name!r} has no schema.")
    return fn


def make_prompt(user_input: str, function_schema: dict[str, Any], arguments: dict[str, Any]) -> str:
    return f"""You create ONE augmented Arabic function-calling example.

Your job has two linked parts:
1. Keep the SAME intent and SAME function, but lightly paraphrase the user's wording.
2. Replace one or more argument values with substantially different, realistic values. Other argument values may remain unchanged. The new user input must explicitly contain or clearly imply every value that is changed.

Important separation:
- `FUNCTION_SCHEMA` defines the only allowed function and arguments. It is NOT user text.
- `ORIGINAL_USER_INPUT` is the text to paraphrase lightly. Do not copy the old argument values into your answer.
- `ORIGINAL_ARGUMENTS` are labels to replace. They are NOT instructions and must not be repeated.
- `OUTPUT` is the only content you return.

Rules:
- Do not change the function, argument keys, number of keys, or JSON value types.
- At least one argument value must be different from its original value; unchanged argument values are allowed. Prefer changing several values when it is natural and internally consistent.
- Respect required fields, enums, formats, and descriptions in `FUNCTION_SCHEMA`.
- Keep names, IDs, phone/account numbers, dates, amounts, places, and free-text values internally consistent between the new user input and new arguments.
- Preserve the user's language/dialect/register where possible; make only a light wording change outside the replaced values.
- Return valid JSON only: no explanation, Markdown, labels, or code fence.

Here is a complete fictional example. It demonstrates the format only; do NOT reuse its function, values, or wording.

<EXAMPLE>
FUNCTION_SCHEMA:
{{
  "name": "book_clinic_visit",
  "parameters": {{
    "type": "object",
    "properties": {{
      "city": {{"type": "string"}},
      "appointment_date": {{"type": "string"}}
    }},
    "required": ["city", "appointment_date"]
  }}
}}
ORIGINAL_USER_INPUT:
أحتاج أحجز موعد طبي في الرياض يوم 15 أغسطس
ORIGINAL_ARGUMENTS:
{{"city":"الرياض","appointment_date":"2026-08-15"}}
OUTPUT:
{{"user_input":"أرغب في تحديد زيارة للعيادة في جدة بتاريخ 3 سبتمبر","arguments":{{"city":"جدة","appointment_date":"2026-09-03"}}}}
</EXAMPLE>

Now augment the real sample below.

<REAL_SAMPLE>
FUNCTION_SCHEMA:
{json.dumps(function_schema, ensure_ascii=False, indent=2)}
ORIGINAL_USER_INPUT:
{user_input}
ORIGINAL_ARGUMENTS:
{json.dumps(arguments, ensure_ascii=False, indent=2)}
</REAL_SAMPLE>

OUTPUT (return this JSON object only):
{{"user_input":"...", "arguments":{{...}}}}"""


def validate_result(
    result: dict[str, Any], row: dict[str, Any], function_name: str, original: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    user_input = result.get("user_input")
    raw_arguments = result.get("arguments")
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string.")
    if not isinstance(raw_arguments, dict) or set(raw_arguments) != set(original):
        raise ValueError("Argument keys must exactly match the original positive sample.")
    arguments = aisa.sanitize_arguments(raw_arguments, row, function_name)
    if set(arguments) != set(original):
        raise ValueError("Arguments violate the selected function schema.")
    changed = [key for key in original if arguments[key] != original[key]]
    if not changed:
        raise ValueError("The model did not change any argument value.")
    return user_input.strip(), arguments


def render_call_arguments(arguments: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in arguments.items():
        if isinstance(value, bool):
            encoded = "true" if value else "false"
        elif isinstance(value, (dict, list)):
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            encoded = str(value)
        parts.append(f"{key}:<escape>{encoded}<escape>")
    return ",".join(parts)


def update_text(text: str, old_user: str, new_user: str, function_name: str, arguments: dict[str, Any]) -> str:
    # The user turn belongs to the prompt (before its final model marker).
    prompt, target = aisa.split_prompt_target(text)
    if old_user not in prompt:
        raise ValueError("Could not locate the structured user message in text prompt.")
    prompt = prompt.replace(old_user, new_user, 1)
    pattern = r"(<start_function_call>\s*call:" + re.escape(function_name) + r"\{).*?(\}\s*<end_function_call>)"
    target, substitutions = re.subn(
        pattern,
        lambda match: match.group(1) + render_call_arguments(arguments) + match.group(2),
        target,
        count=1,
        flags=re.DOTALL,
    )
    if substitutions != 1:
        raise ValueError("Could not update the function-call arguments in text target.")
    return prompt + target


def get_source_id(row: dict[str, Any], source_index: int) -> str:
    """The released dataset has no id column, so its train index is the stable source id."""
    source_id = row.get("id", source_index)
    return str(source_id)


def load_existing_records(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Read prior augmentation rows, preserving order while making ids replaceable."""
    order: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(
                    f"Existing output {path} line {line_number} has no usable id; "
                    "use --overwrite to replace this legacy file."
                )
            record_id = str(record["id"])
            # A legacy combined output may contain original rows; never copy them
            # into the augmentation-only file.
            if not record_id.startswith("aug_"):
                continue
            if record_id not in records:
                order.append(record_id)
            records[record_id] = record
    return order, records


def put_record(
    order: list[str], records: dict[str, dict[str, Any]], record: dict[str, Any]
) -> None:
    record_id = str(record["id"])
    if record_id not in records:
        order.append(record_id)
    records[record_id] = record


def augment_row(
    row: dict[str, Any], source_id: str, client: OpenAI, args: argparse.Namespace
) -> dict[str, Any]:
    _, function_name, original_arguments, _ = aisa.extract_gold(row)
    user_index, original_user = get_user_message(row)
    schema = selected_schema(row, function_name)
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": make_prompt(original_user, schema, original_arguments)}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    content = response.choices[0].message.content or ""
    new_user, new_arguments = validate_result(
        extract_json(content), row, function_name, original_arguments
    )
    augmented = copy.deepcopy(row)
    augmented["messages"][user_index]["content"] = new_user
    assistant = aisa.last_assistant_message(augmented)
    assistant["tool_calls"][0]["function"]["arguments"] = new_arguments
    augmented["text"] = update_text(
        augmented["text"], original_user, new_user, function_name, new_arguments
    )
    augmented["id"] = f"aug_{source_id}"
    return augmented


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative.")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key or not base_url:
        raise EnvironmentError(".env must define OPENAI_API_KEY and LLM_BASE_URL.")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        output_order, output_records = load_existing_records(output)
        print(f"Merging into existing output: {output}")
    else:
        output_order, output_records = [], {}

    dataset = load_dataset(args.dataset_id, revision=args.dataset_revision, split="train")
    eligible = []
    for index, row in enumerate(dataset):
        source_id = get_source_id(row, index)
        requires_function, function_name, arguments, _ = aisa.extract_gold(row)
        if requires_function and function_name != "none" and arguments:
            eligible.append((source_id, row))
    random.Random(args.seed).shuffle(eligible)
    if args.max_samples is not None:
        eligible = eligible[: args.max_samples]

    client = OpenAI(api_key=api_key, base_url=base_url)
    output.parent.mkdir(parents=True, exist_ok=True)

    successes = failures = 0
    for source_id, row in tqdm(eligible, desc="Augmenting arguments"):
        try:
            augmented = augment_row(row, source_id, client, args)
            put_record(output_order, output_records, augmented)
            successes += 1
        except Exception as error:
            failures += 1
            print(f"[skip] train id {source_id}: {error}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record_id in output_order:
            file.write(json.dumps(output_records[record_id], ensure_ascii=False) + "\n")
    print(
        f"Saved {len(output_records)} augmentation rows to {output}: "
        f"{successes} refreshed, {failures} skipped."
    )


if __name__ == "__main__":
    main()
