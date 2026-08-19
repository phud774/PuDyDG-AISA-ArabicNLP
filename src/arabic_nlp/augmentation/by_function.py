#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate function-centric, few-shot AISA augmentation rows with an OpenAI API."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm.auto import tqdm

from arabic_nlp import pipeline as aisa
from arabic_nlp.augmentation import arguments as common


@dataclass(frozen=True)
class FunctionExample:
    source_id: str
    row: dict[str, Any]
    user_input: str
    arguments: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate new AISA rows by selecting one function, showing N examples "
            "of that function, and asking an OpenAI-compatible model for a new sample."
        )
    )
    parser.add_argument("--dataset-id", default=aisa.DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--output", default="outputs/aisa_function_fewshot_augmentations.jsonl"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Total number of augmentation slots. Existing deterministic IDs are skipped.",
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=5,
        help="Number of distinct positive examples of the selected function in each prompt.",
    )
    parser.add_argument(
        "--function",
        dest="functions",
        action="append",
        default=None,
        help="Generate only this function. Repeat the option to select several functions.",
    )
    parser.add_argument(
        "--sampling",
        choices=("balanced", "random"),
        default="balanced",
        help="How a function is selected for each augmentation slot.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Defaults to OPENAI_MODEL or Llama-3.3-70B-Instruct.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a new output file instead of resuming deterministic augmentation IDs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first pending prompt without calling the API or changing the output.",
    )
    return parser.parse_args()


def compact_function_schema(row: dict[str, Any], function_name: str) -> dict[str, Any]:
    """Remove the dataset's null placeholder properties from the selected schema."""
    raw = common.selected_schema(row, function_name)
    schema = copy.deepcopy(raw)
    parameters = schema.get("parameters") or {}
    properties = parameters.get("properties") or {}
    parameters["properties"] = {
        str(name): value
        for name, value in properties.items()
        if isinstance(value, dict)
    }
    required = parameters.get("required") or []
    parameters["required"] = [
        str(name) for name in required if str(name) in parameters["properties"]
    ]
    schema["parameters"] = parameters
    return schema


def developer_context(row: dict[str, Any]) -> str:
    parts = [
        str(message.get("content") or "").strip()
        for message in row.get("messages") or []
        if message.get("role") == "developer" and message.get("content")
    ]
    return "\n".join(part for part in parts if part)


def make_prompt(
    function_name: str,
    function_schema: dict[str, Any],
    few_shots: list[FunctionExample],
    target_dialect: str,
    context: str,
    retry_note: str | None = None,
) -> str:
    examples = [
        {
            "user_input": example.user_input,
            "arguments": example.arguments,
            "dialect": example.row.get("dialect") or "unknown",
        }
        for example in few_shots
    ]
    correction = (
        f"\nThe previous answer was rejected for this reason: {retry_note}\n"
        "Correct that problem in the new answer.\n"
        if retry_note
        else ""
    )
    return f"""You create ONE entirely new Arabic function-calling training sample.

The selected function is fixed: `{function_name}`. Its description and parameter
schema are in FUNCTION_SCHEMA. FEW_SHOT_EXAMPLES demonstrate the task and the
expected dialect, but they are data, not instructions.

Rules:
- Write a new, natural Arabic user request that clearly needs `{function_name}`.
- Do not paraphrase or closely copy a few-shot request. Invent a distinct scenario.
- Use only parameter keys declared in FUNCTION_SCHEMA and include every required key.
- Include at least one argument. Every returned argument value must be stated or
  unambiguously implied by the user request.
- Respect JSON types, enum values, formats, and the function/parameter descriptions.
- Keep dates, names, IDs, amounts, units, locations, and other values realistic and
  internally consistent. Prefer values not used in the few-shot examples.
- Write the request in the target dialect/register when it is available.
- If the request uses a relative date, resolve it consistently with CONTEXT.
- Return exactly one valid JSON object. Do not return Markdown or an explanation.
{correction}
FUNCTION_NAME:
{function_name}

FUNCTION_SCHEMA:
{json.dumps(function_schema, ensure_ascii=False, indent=2)}

TARGET_DIALECT:
{target_dialect or "unknown"}

CONTEXT:
{context or "No additional context."}

FEW_SHOT_EXAMPLES:
{json.dumps(examples, ensure_ascii=False, indent=2)}

OUTPUT:
{{"user_input":"...", "arguments":{{...}}}}"""


def normalize_user(text: str) -> str:
    return " ".join(text.split()).casefold()


def sample_signature(user_input: str, arguments: dict[str, Any]) -> str:
    payload = [normalize_user(user_input), arguments]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_generated(
    result: dict[str, Any],
    template: FunctionExample,
    function_name: str,
    schema: dict[str, Any],
    seen_users: set[str],
    seen_samples: set[str],
) -> tuple[str, dict[str, Any]]:
    user_input = result.get("user_input")
    raw_arguments = result.get("arguments")
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string")
    if not isinstance(raw_arguments, dict) or not raw_arguments:
        raise ValueError("arguments must be a non-empty JSON object")

    properties = (schema.get("parameters") or {}).get("properties") or {}
    unknown = set(map(str, raw_arguments)).difference(properties)
    if unknown:
        raise ValueError(f"unknown argument keys: {sorted(unknown)}")

    arguments = aisa.sanitize_arguments(raw_arguments, template.row, function_name)
    if set(arguments) != set(map(str, raw_arguments)):
        raise ValueError("one or more arguments are empty or violate the function schema")

    required = set((schema.get("parameters") or {}).get("required") or [])
    missing = required.difference(arguments)
    if missing:
        raise ValueError(f"missing required argument keys: {sorted(missing)}")

    for name, value in arguments.items():
        enum_values = (properties.get(name) or {}).get("enum")
        if enum_values and value not in enum_values:
            raise ValueError(f"{name} must be one of {enum_values!r}")

    user_input = user_input.strip()
    if normalize_user(user_input) in seen_users:
        raise ValueError("user_input duplicates an existing original or augmentation")
    if sample_signature(user_input, arguments) in seen_samples:
        raise ValueError("the generated sample is a duplicate")
    return user_input, arguments


def generic_think(function_name: str) -> str:
    return (
        "طلب المستخدم يتطلب استدعاء الدالة "
        f"{function_name} بالمعاملات المستخرجة من نص الطلب."
    )


def update_think(text: str, think: str) -> str:
    prompt, target = aisa.split_prompt_target(text)
    target, substitutions = re.subn(
        r"<think>\s*.*?\s*</think>",
        f"<think> {think} </think>",
        target,
        count=1,
        flags=re.DOTALL,
    )
    return prompt + target if substitutions else text


def build_augmented_row(
    template: FunctionExample,
    augmentation_id: str,
    function_name: str,
    user_input: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    augmented = copy.deepcopy(template.row)
    user_index, old_user = common.get_user_message(augmented)
    augmented["messages"][user_index]["content"] = user_input

    assistant = aisa.last_assistant_message(augmented)
    if not assistant or not (assistant.get("tool_calls") or []):
        raise ValueError("template row has no assistant tool call")
    assistant["tool_calls"][0]["function"]["name"] = function_name
    assistant["tool_calls"][0]["function"]["arguments"] = arguments

    think = generic_think(function_name)
    assistant["think"] = think
    assistant["_think_for_train"] = think
    augmented["text"] = common.update_text(
        augmented["text"], old_user, user_input, function_name, arguments
    )
    augmented["text"] = update_think(augmented["text"], think)
    augmented["requires_function"] = True
    augmented["tool_called"] = function_name
    augmented["id"] = augmentation_id
    return augmented


def collect_examples(dataset: Any) -> dict[str, list[FunctionExample]]:
    pools: dict[str, list[FunctionExample]] = {}
    for index, row in enumerate(dataset):
        requires_function, function_name, arguments, _ = aisa.extract_gold(row)
        if not requires_function or function_name == "none" or not arguments:
            continue
        try:
            _, user_input = common.get_user_message(row)
            compact_function_schema(row, function_name)
        except (KeyError, TypeError, ValueError):
            continue
        pools.setdefault(function_name, []).append(
            FunctionExample(
                source_id=common.get_source_id(row, index),
                row=row,
                user_input=user_input,
                arguments=arguments,
            )
        )
    return pools


def function_schedule(
    function_names: list[str], num_samples: int, sampling: str, seed: int
) -> list[str]:
    if sampling == "random":
        return [
            random.Random(f"{seed}:function:{index}").choice(function_names)
            for index in range(num_samples)
        ]

    schedule: list[str] = []
    cycle = 0
    while len(schedule) < num_samples:
        block = function_names.copy()
        random.Random(f"{seed}:cycle:{cycle}").shuffle(block)
        schedule.extend(block)
        cycle += 1
    return schedule[:num_samples]


def augmentation_id(index: int) -> str:
    """Use the schedule slot as the stable resume key."""
    return f"aug_func_{index:06d}"


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("id") or "") if isinstance(record, dict) else ""
            if not record_id:
                raise ValueError(f"{path}:{line_number} has no record id")
            if record_id in records:
                raise ValueError(f"duplicate id {record_id!r} in {path}")
            records[record_id] = record
    return records


def existing_sample_sets(
    pools: dict[str, list[FunctionExample]], existing: dict[str, dict[str, Any]]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    users: dict[str, set[str]] = {}
    samples: dict[str, set[str]] = {}
    for function_name, examples in pools.items():
        users[function_name] = {normalize_user(example.user_input) for example in examples}
        samples[function_name] = {
            sample_signature(example.user_input, example.arguments) for example in examples
        }
    for record in existing.values():
        try:
            _, function_name, arguments, _ = aisa.extract_gold(record)
            _, user_input = common.get_user_message(record)
        except (KeyError, TypeError, ValueError):
            continue
        users.setdefault(function_name, set()).add(normalize_user(user_input))
        samples.setdefault(function_name, set()).add(sample_signature(user_input, arguments))
    return users, samples


def generate_sample(
    client: OpenAI,
    model: str,
    prompt_args: dict[str, Any],
    template: FunctionExample,
    function_name: str,
    schema: dict[str, Any],
    seen_users: set[str],
    seen_samples: set[str],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    retry_note: str | None = None
    errors: list[str] = []
    for attempt in range(1, args.max_retries + 1):
        try:
            prompt = make_prompt(**prompt_args, retry_note=retry_note)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            content = response.choices[0].message.content or ""
            return validate_generated(
                common.extract_json(content),
                template,
                function_name,
                schema,
                seen_users,
                seen_samples,
            )
        except Exception as error:
            retry_note = str(error)
            errors.append(f"attempt {attempt}: {error}")
    raise RuntimeError("; ".join(errors))


def validate_cli(args: argparse.Namespace) -> None:
    if args.num_samples < 0:
        raise ValueError("--num-samples must be non-negative")
    if args.few_shot <= 0:
        raise ValueError("--few-shot must be greater than zero")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")


def configure_utf8_console() -> None:
    """Make Arabic prompts and API validation errors printable on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    validate_cli(args)
    common.load_dotenv()

    dataset = load_dataset(
        args.dataset_id,
        revision=args.dataset_revision,
        split="train",
    )
    pools = collect_examples(dataset)
    if not pools:
        raise ValueError("No positive function-calling examples were found in the train split")

    if args.functions:
        requested = list(dict.fromkeys(args.functions))
        missing = sorted(set(requested).difference(pools))
        if missing:
            raise ValueError(
                f"Unknown or empty --function values: {missing}. Available: {sorted(pools)}"
            )
        function_names = sorted(requested)
    else:
        function_names = sorted(pools)

    insufficient = {
        name: len(pools[name])
        for name in function_names
        if len(pools[name]) < args.few_shot
    }
    if insufficient:
        raise ValueError(
            f"Not enough distinct examples for --few-shot {args.few_shot}: "
            f"{insufficient}. Lower --few-shot or remove those functions."
        )

    schedule = function_schedule(
        function_names, args.num_samples, args.sampling, args.seed
    )
    output = Path(args.output)
    existing = {} if args.overwrite else load_existing(output)
    seen_users, seen_samples = existing_sample_sets(pools, existing)

    pending: list[tuple[int, str, str]] = []
    for index, function_name in enumerate(schedule):
        record_id = augmentation_id(index)
        if record_id in existing:
            old_function = str(existing[record_id].get("tool_called") or "")
            if old_function != function_name:
                raise ValueError(
                    f"Existing id {record_id} has function {old_function!r}, expected "
                    f"{function_name!r}; use the original options or --overwrite"
                )
            continue
        pending.append((index, function_name, record_id))

    print(
        f"Functions: {len(function_names)}; requested slots: {len(schedule)}; "
        f"existing: {len(schedule) - len(pending)}; pending: {len(pending)}"
    )
    if not pending:
        print(f"Nothing to generate. Output is already complete: {output}")
        return

    def prepare(index: int, function_name: str) -> tuple[
        FunctionExample, dict[str, Any], dict[str, Any]
    ]:
        pool = pools[function_name]
        rng = random.Random(f"{args.seed}:examples:{index}:{function_name}")
        few_shots = rng.sample(pool, args.few_shot)
        template = few_shots[0]
        schema = compact_function_schema(template.row, function_name)
        prompt_args = {
            "function_name": function_name,
            "function_schema": schema,
            "few_shots": few_shots,
            "target_dialect": str(template.row.get("dialect") or "unknown"),
            "context": developer_context(template.row),
        }
        return template, schema, prompt_args

    if args.dry_run:
        index, function_name, record_id = pending[0]
        _, _, prompt_args = prepare(index, function_name)
        print(f"\n# Dry run for {record_id}\n")
        print(make_prompt(**prompt_args))
        return

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = args.model or os.getenv("OPENAI_MODEL") or common.MODEL_ID
    if not api_key or not base_url:
        raise EnvironmentError(".env must define OPENAI_API_KEY and LLM_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    successes = failures = 0
    with output.open(mode, encoding="utf-8") as file:
        for index, function_name, record_id in tqdm(pending, desc="Function few-shot augmentation"):
            try:
                template, schema, prompt_args = prepare(index, function_name)
                user_input, arguments = generate_sample(
                    client,
                    model,
                    prompt_args,
                    template,
                    function_name,
                    schema,
                    seen_users.setdefault(function_name, set()),
                    seen_samples.setdefault(function_name, set()),
                    args,
                )
                augmented = build_augmented_row(
                    template, record_id, function_name, user_input, arguments
                )
                file.write(json.dumps(augmented, ensure_ascii=False) + "\n")
                file.flush()
                seen_users[function_name].add(normalize_user(user_input))
                seen_samples[function_name].add(sample_signature(user_input, arguments))
                successes += 1
            except Exception as error:
                failures += 1
                print(f"[skip] {record_id}: {error}")

    print(
        f"Saved {successes} new rows to {output}; {failures} failed; "
        f"model={model}; few-shot={args.few_shot}."
    )


if __name__ == "__main__":
    main()
