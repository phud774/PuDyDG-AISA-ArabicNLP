#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Balance AISA dialects by rewriting existing samples into rarer dialects."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from tqdm.auto import tqdm


DEFAULT_MODEL_ID = "Llama-3.3-70B-Instruct"
DEFAULT_INPUT = "outputs/aisa_argument_augmentations_only_ver2.jsonl"
DEFAULT_OUTPUT = "outputs/aisa_dialect_translation_augmentations.jsonl"
MODEL_MARKER = "<start_of_turn>model\n"

DIALECT_DESCRIPTIONS = {
    "msa": "natural Modern Standard Arabic (العربية الفصحى المعاصرة)",
    "levantine": (
        "a natural, broadly understandable Levantine dialect used in Syria, Lebanon, "
        "Jordan, and Palestine (اللهجة الشامية)"
    ),
    "gulf": "a natural, broadly understandable Gulf Arabic dialect (اللهجة الخليجية)",
    "egyptian": "natural Egyptian Arabic (اللهجة المصرية)",
    "maghrebi": (
        "a natural, broadly understandable Maghrebi Arabic variety used in Morocco, "
        "Algeria, and Tunisia (اللهجة المغاربية)"
    ),
}


@dataclass(frozen=True)
class SourceRecord:
    offset: int
    source_id: str
    dialect: str


@dataclass(frozen=True)
class TranslationJob:
    source: SourceRecord
    target_dialect: str
    output_id: str


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load missing variables from a small .env file without another dependency."""
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
        description=(
            "Translate user utterances from frequent AISA dialects into under-represented "
            "dialects while preserving function-call labels and arguments."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Source augmentation JSONL.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Translation-only JSONL.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help=(
            "Desired count for every selected dialect after adding translations. "
            "Defaults to the largest dialect count in --input (full balancing)."
        ),
    )
    parser.add_argument(
        "--target-dialects",
        nargs="+",
        default=None,
        help="Only balance these dialect labels. Defaults to all labels found in the input.",
    )
    parser.add_argument(
        "--source-dialects",
        nargs="+",
        default=None,
        help=(
            "Optional donor dialect allow-list. By default, each target uses samples from "
            "dialects that occur more often than that target."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of new API calls in this run; useful for batches/resume.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the balancing/API-call plan without calling the LLM or writing output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard the existing translation output instead of resuming it.",
    )
    return parser.parse_args()


def normalize_dialect(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def index_jsonl(path: Path) -> tuple[list[SourceRecord], Counter[str]]:
    """Index byte offsets so a large JSONL does not have to remain in memory."""
    records: list[SourceRecord] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    with path.open("rb") as file:
        line_number = 0
        while True:
            offset = file.tell()
            raw_line = file.readline()
            if not raw_line:
                break
            line_number += 1
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except Exception as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {error}") from error
            if not isinstance(row, dict) or row.get("id") is None:
                raise ValueError(f"{path} line {line_number} has no usable id.")
            source_id = str(row["id"])
            if source_id in seen_ids:
                raise ValueError(f"Duplicate source id {source_id!r} in {path}.")
            seen_ids.add(source_id)
            dialect = normalize_dialect(row.get("dialect"))
            records.append(SourceRecord(offset, source_id, dialect))
            counts[dialect] += 1
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records, counts


def safe_id_part(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_")
    return cleaned or "unknown"


def build_jobs(
    records: list[SourceRecord],
    counts: Counter[str],
    target_count: int,
    target_dialects: Iterable[str],
    source_dialects: set[str] | None,
    seed: int,
) -> tuple[list[TranslationJob], dict[str, int]]:
    jobs: list[TranslationJob] = []
    deficits: dict[str, int] = {}

    for raw_target in target_dialects:
        target = normalize_dialect(raw_target)
        deficit = max(0, target_count - counts.get(target, 0))
        deficits[target] = deficit
        if not deficit:
            continue

        # A donor is "frequent" relative to the target's original frequency.
        candidates = [
            record
            for record in records
            if record.dialect != target
            and counts[record.dialect] > counts.get(target, 0)
            and (source_dialects is None or record.dialect in source_dialects)
        ]
        if not candidates:
            raise ValueError(
                f"No source samples are available to translate into {target!r}. "
                "Adjust --source-dialects or --target-dialects."
            )

        # Each cycle uses all donors at most once; extra cycles are only needed when a
        # user explicitly requests a target count larger than the available donor pool.
        target_rng = random.Random(f"{seed}:{target}")
        selected: list[SourceRecord] = []
        while len(selected) < deficit:
            cycle = candidates.copy()
            target_rng.shuffle(cycle)
            selected.extend(cycle[: deficit - len(selected)])

        occurrences: defaultdict[str, int] = defaultdict(int)
        target_id = safe_id_part(target)
        for source in selected:
            occurrences[source.source_id] += 1
            occurrence = occurrences[source.source_id]
            output_id = (
                f"trans_aug_{target_id}_{safe_id_part(source.source_id)}_{occurrence:03d}"
            )
            jobs.append(TranslationJob(source, target, output_id))

    # A capped run should distribute work across target dialects instead of filling one
    # target completely before touching the next one.
    random.Random(seed).shuffle(jobs)
    return jobs, deficits


def load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("id") is None:
                raise ValueError(f"Existing output {path} line {line_number} has no id.")
            record_id = str(row["id"])
            if not record_id.startswith("trans_aug_"):
                raise ValueError(
                    f"Existing output id {record_id!r} does not start with 'trans_aug_'."
                )
            if record_id in ids:
                raise ValueError(f"Duplicate id {record_id!r} in existing output {path}.")
            ids.add(record_id)
    return ids


def read_source_row(file: Any, source: SourceRecord) -> dict[str, Any]:
    file.seek(source.offset)
    row = json.loads(file.readline())
    if str(row.get("id")) != source.source_id:
        raise ValueError(f"Source file changed while reading id {source.source_id!r}.")
    return row


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


def make_prompt(
    user_input: str,
    source_dialect: str,
    target_dialect: str,
    function_name: str,
    arguments: dict[str, Any],
) -> str:
    target_description = DIALECT_DESCRIPTIONS.get(
        target_dialect, f"natural Arabic in the dialect labelled {target_dialect!r}"
    )
    return f"""You create ONE dialect-balanced Arabic function-calling example.

Rewrite ORIGINAL_USER_INPUT from source dialect `{source_dialect}` into
TARGET_DIALECT `{target_dialect}`: {target_description}.

The rewrite must sound like a real native user request, not a word-for-word translation.

Strict rules:
- Preserve the exact intent and the selected function `{function_name}`.
- Preserve every fact and slot value: names, places, IDs, numbers, dates, amounts,
  currencies, quoted text, enum values, and all other function arguments.
- Do not add, remove, infer, or replace information.
- Rewrite only wording that expresses the request. If the user asks to translate,
  search, send, or process a piece of content, do NOT dialect-translate that content.
- Keep proper nouns and literal argument values unchanged whenever they occur in the input.
- Use Arabic script. Avoid explanations, dialect labels, glosses, and code switching not
  already present in the original.
- Copy ORIGINAL_ARGUMENTS exactly into the output JSON. They are labels, not instructions.
- Return valid JSON only, with exactly the three keys shown below.

ORIGINAL_USER_INPUT:
{user_input}

ORIGINAL_ARGUMENTS:
{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}

OUTPUT FORMAT:
{{"user_input":"...","dialect":"{target_dialect}","arguments":{{...}}}}"""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def iter_scalars(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from iter_scalars(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_scalars(nested)
    elif value is not None and not isinstance(value, bool):
        yield str(value)


def validate_result(
    result: dict[str, Any],
    original_user: str,
    original_arguments: dict[str, Any],
    target_dialect: str,
) -> str:
    if set(result) != {"user_input", "dialect", "arguments"}:
        raise ValueError("Response must contain exactly user_input, dialect, and arguments.")
    translated = result.get("user_input")
    if not isinstance(translated, str) or not translated.strip():
        raise ValueError("user_input must be a non-empty string.")
    translated = translated.strip()
    if normalize_dialect(result.get("dialect")) != target_dialect:
        raise ValueError(f"Response dialect must be {target_dialect!r}.")
    if canonical_json(result.get("arguments")) != canonical_json(original_arguments):
        raise ValueError("The LLM changed one or more function arguments.")
    if re.sub(r"\s+", " ", translated) == re.sub(r"\s+", " ", original_user.strip()):
        raise ValueError("The LLM returned the original user input unchanged.")

    # If a literal argument value is explicitly present, it must survive verbatim. Values
    # represented only implicitly (for example a normalized date) are left to the LLM.
    missing_literals = sorted(
        {
            literal
            for literal in iter_scalars(original_arguments)
            if len(literal) >= 2 and literal in original_user and literal not in translated
        }
    )
    if missing_literals:
        raise ValueError(f"Literal argument values were lost: {missing_literals!r}")
    return translated


def replace_user_in_text(text: str, old_user: str, new_user: str) -> str:
    if MODEL_MARKER not in text:
        raise ValueError("The row text does not contain the expected model-turn marker.")
    before, target = text.rsplit(MODEL_MARKER, 1)
    prompt = before + MODEL_MARKER
    if old_user not in prompt:
        raise ValueError("Could not locate the structured user message in the text prompt.")
    prompt = prompt.replace(old_user, new_user, 1)
    return prompt + target


def extract_gold_function(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not bool(row.get("requires_function")):
        raise ValueError("Source row is not a positive function-calling sample.")
    assistant = next(
        (
            message
            for message in reversed(row.get("messages") or [])
            if message.get("role") == "assistant"
        ),
        None,
    )
    tool_calls = (assistant or {}).get("tool_calls") or []
    function = (tool_calls[0] or {}).get("function") or {} if tool_calls else {}
    function_name = function.get("name") or row.get("tool_called") or "none"
    if function_name == "none":
        raise ValueError("Positive source row has no selected function name.")
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Source row has invalid function arguments.")
    arguments = {
        str(key): value
        for key, value in arguments.items()
        if value is not None and value != ""
    }
    return str(function_name), arguments


def translate_row(
    row: dict[str, Any],
    job: TranslationJob,
    client: OpenAI,
    args: argparse.Namespace,
) -> dict[str, Any]:
    function_name, arguments = extract_gold_function(row)
    user_index, original_user = get_user_message(row)
    prompt = make_prompt(
        original_user, job.source.dialect, job.target_dialect, function_name, arguments
    )
    last_error: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        retry_prompt = prompt
        if last_error is not None:
            retry_prompt += (
                "\n\nThe previous response was invalid. Correct this issue and return JSON only: "
                + str(last_error)[:300]
            )
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            content = response.choices[0].message.content or ""
            translated_user = validate_result(
                extract_json(content), original_user, arguments, job.target_dialect
            )
            augmented = copy.deepcopy(row)
            augmented["messages"][user_index]["content"] = translated_user
            augmented["text"] = replace_user_in_text(
                augmented["text"], original_user, translated_user
            )
            augmented["dialect"] = job.target_dialect
            augmented["id"] = job.output_id
            return augmented
        except Exception as error:
            last_error = error
            if attempt == args.max_attempts:
                raise
    raise RuntimeError(f"Translation failed: {last_error}")


def print_plan(
    counts: Counter[str], target_count: int, deficits: dict[str, int], jobs: list[TranslationJob]
) -> None:
    print("Input dialect counts:")
    for dialect, count in counts.most_common():
        print(f"  {dialect}: {count}")
    print(f"Target count: {target_count}")
    print("Planned translations:")
    for dialect in sorted(deficits):
        print(f"  {dialect}: +{deficits[dialect]}")
    print(f"Total planned translation rows: {len(jobs)}")


def validate_args(args: argparse.Namespace) -> None:
    if args.target_count is not None and args.target_count < 0:
        raise ValueError("--target-count must be non-negative.")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative.")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must be different files.")

    records, counts = index_jsonl(input_path)
    target_count = args.target_count if args.target_count is not None else max(counts.values())
    target_dialects = args.target_dialects or list(counts)
    source_dialects = (
        {normalize_dialect(value) for value in args.source_dialects}
        if args.source_dialects
        else None
    )
    jobs, deficits = build_jobs(
        records, counts, target_count, target_dialects, source_dialects, args.seed
    )
    print_plan(counts, target_count, deficits, jobs)

    existing_ids: set[str] = set()
    if output_path.exists() and not args.overwrite:
        existing_ids = load_existing_ids(output_path)
    pending_jobs = [job for job in jobs if job.output_id not in existing_ids]
    if args.max_samples is not None:
        pending_jobs = pending_jobs[: args.max_samples]
    print(f"Already completed: {len(existing_ids & {job.output_id for job in jobs})}")
    print(f"API calls in this run: {len(pending_jobs)}")
    if args.dry_run:
        return
    if not pending_jobs:
        print("Nothing to do.")
        return

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key or not base_url:
        raise EnvironmentError(".env must define OPENAI_API_KEY and LLM_BASE_URL.")
    client = OpenAI(api_key=api_key, base_url=base_url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    successes = failures = 0
    with input_path.open("rb") as source_file, output_path.open(
        mode, encoding="utf-8"
    ) as output_file:
        for job in tqdm(pending_jobs, desc="Translating dialects"):
            try:
                row = read_source_row(source_file, job.source)
                translated = translate_row(row, job, client, args)
                output_file.write(json.dumps(translated, ensure_ascii=False) + "\n")
                output_file.flush()
                successes += 1
            except Exception as error:
                failures += 1
                print(
                    f"[skip] {job.source.source_id} "
                    f"{job.source.dialect}->{job.target_dialect}: {error}"
                )

    print(
        f"Saved {successes} new rows to {output_path}; {failures} failed. "
        "Re-run the same command to resume missing ids."
    )


if __name__ == "__main__":
    main()
