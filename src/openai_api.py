from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def _load_env_file(env_path: Path | None = None) -> dict[str, str]:
    env_path = env_path or Path(__file__).resolve().parents[1] / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _load_kaggle_secret(secret_name: str) -> str:
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception:
        return ""

    try:
        return UserSecretsClient().get_secret(secret_name) or ""
    except Exception:
        return ""


def _pick_config_value(env_values: dict[str, str], *candidate_names: str) -> str:
    for name in candidate_names:
        value = env_values.get(name, "")
        if value:
            return value

    for name in candidate_names:
        value = _load_kaggle_secret(name)
        if value:
            return value

    return ""


def get_openai_config() -> dict[str, str]:
    env_values = {**os.environ, **_load_env_file()}
    api_key = _pick_config_value(env_values, "OPENAI_API_KEY", "OPENAI_KEY", "KAGGLE_OPENAI_API_KEY", "KAGGLE_OPENAI_KEY")
    base_url = _pick_config_value(env_values, "LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "KAGGLE_OPENAI_BASE_URL")
    model_name = _pick_config_value(env_values, "OPENAI_MODEL", "KAGGLE_OPENAI_MODEL")
    return {"api_key": api_key, "base_url": base_url, "model_name": model_name}


def build_arguments_api_prompt(
    current_prompt: str,
    function_name: str,
    fewshot_examples: list[dict[str, Any]],
) -> str:
    parts = [
        "You are an Arabic function-call argument extractor.",
        "Return only a valid JSON object matching the requested function schema.",
        "Do not include markdown, explanation, or extra text.",
    ]

    if fewshot_examples:
        parts.append("Examples:")
        for idx, example in enumerate(fewshot_examples, start=1):
            parts.append(f"Example {idx}:")
            parts.append(f"Function: {example['function_name']}")
            parts.append(f"Input: {example['prompt']}")
            parts.append(f"Output: {json.dumps(example['arguments'], ensure_ascii=False, separators=(',', ':'))}")

    parts.append("Now complete the following:")
    parts.append(f"Function: {function_name}")
    parts.append(f"Input: {current_prompt}")
    parts.append("Output:")
    return "\n".join(parts)


def generate_arguments_with_openai_api(
    prompt: str,
    function_name: str,
    fewshot_examples: list[dict[str, Any]],
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - runtime fallback
        raise RuntimeError("openai package is not installed") from exc

    config = get_openai_config()
    api_key = config.get("api_key", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured. Set it in the environment or Kaggle Secrets.")

    client = OpenAI(api_key=api_key, base_url=config.get("base_url") or None)
    request_model = model_name or config.get("model_name") or "gpt-4o-mini"

    response = client.chat.completions.create(
        model=request_model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract arguments for function calls from Arabic requests. "
                    "Return strictly a JSON object."
                ),
            },
            {
                "role": "user",
                "content": build_arguments_api_prompt(prompt, function_name, fewshot_examples),
            },
        ],
    )

    content = response.choices[0].message.content or ""
    return content.strip()


def normalize_text(text: str) -> list[str]:
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text.lower())
    return [token for token in tokens if token]


def score_text_similarity(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left))
    right_tokens = set(normalize_text(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return len(overlap) / max(1, len(left_tokens | right_tokens))


def retrieve_fewshot_examples(
    row: dict[str, Any],
    retrieval_rows: list[dict[str, Any]],
    function_name: str,
    k: int = 3,
) -> list[dict[str, Any]]:
    if not retrieval_rows:
        return []

    current_text = str(row.get("text") or "")
    current_prompt, _ = "", ""
    try:
        from src.utils import split_prompt_target

        current_prompt, _ = split_prompt_target(current_text)
    except Exception:
        current_prompt = current_text

    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in retrieval_rows:
        if candidate is row:
            continue

        requires_function = bool(candidate.get("requires_function", False))
        candidate_name = str(candidate.get("function_name") or candidate.get("tool_called") or "none")
        arguments = candidate.get("arguments") or {}

        if not requires_function or candidate_name != function_name:
            continue
        if not arguments:
            continue

        candidate_text = str(candidate.get("text") or "")
        candidate_prompt = str(candidate.get("prompt") or candidate_text)
        try:
            candidate_prompt, _ = split_prompt_target(candidate_text)
        except Exception:
            candidate_prompt = str(candidate.get("prompt") or candidate_text)

        score = score_text_similarity(current_prompt, candidate_prompt)
        scored.append((score, {"prompt": candidate_prompt, "function_name": candidate_name, "arguments": arguments}))

    scored.sort(key=lambda item: item[0], reverse=True)
    examples = [example for _, example in scored[:k]]
    if len(examples) < k:
        for candidate in retrieval_rows:
            if candidate is row:
                continue
            requires_function = bool(candidate.get("requires_function", False))
            candidate_name = str(candidate.get("function_name") or candidate.get("tool_called") or "none")
            arguments = candidate.get("arguments") or {}
            if not requires_function or candidate_name != function_name:
                continue
            if not arguments:
                continue
            candidate_prompt = str(candidate.get("prompt") or candidate.get("text") or "")
            if any(example["prompt"] == candidate_prompt for example in examples):
                continue
            examples.append({"prompt": candidate_prompt, "function_name": candidate_name, "arguments": arguments})
            if len(examples) >= k:
                break
    return examples
