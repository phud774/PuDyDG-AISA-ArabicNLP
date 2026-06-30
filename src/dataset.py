from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from src.constants import DEFAULT_DATASET_ID
from src.utils import MODEL_MARKER, split_prompt_target


def load_aisa_dataset(args: Any) -> DatasetDict:
    download_mode = "force_redownload" if args.force_redownload else "reuse_dataset_if_exists"
    ds = load_dataset(
        args.dataset_id or DEFAULT_DATASET_ID,
        revision=args.dataset_revision,
        download_mode=download_mode,
    )
    required_splits = {"train", "dev"}
    missing = required_splits.difference(ds.keys())
    if missing:
        raise ValueError(
            f"Dataset must contain train and dev splits; missing: {sorted(missing)}. "
            f"Available: {list(ds.keys())}"
        )

    expected_columns = {
        "text",
        "requires_function",
        "tool_called",
        "messages",
        "tools",
        "tools_sampled",
        "negative_category",
        "dialect",
    }
    missing_columns = expected_columns.difference(ds["train"].column_names)
    if missing_columns:
        raise ValueError(
            f"Unexpected dataset schema. Missing columns: {sorted(missing_columns)}. "
            f"Found: {ds['train'].column_names}"
        )

    print(ds)
    print("Train columns:", ds["train"].column_names)
    print(
        "Train positives/negatives:",
        sum(bool(x) for x in ds["train"]["requires_function"]),
        "/",
        sum(not bool(x) for x in ds["train"]["requires_function"]),
    )
    print(
        "Dev positives/negatives:",
        sum(bool(x) for x in ds["dev"]["requires_function"]),
        "/",
        sum(not bool(x) for x in ds["dev"]["requires_function"]),
    )
    return ds


def last_assistant_message(row: dict[str, Any]) -> dict[str, Any] | None:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return message
    return None


def extract_gold(row: dict[str, Any]) -> tuple[bool, str, dict[str, Any], str]:
    requires_function = bool(row.get("requires_function"))
    assistant = last_assistant_message(row)
    function_name = "none"
    arguments: dict[str, Any] = {}
    think = ""

    if assistant:
        tool_calls = assistant.get("tool_calls") or []
        if tool_calls:
            function = (tool_calls[0] or {}).get("function") or {}
            function_name = function.get("name") or row.get("tool_called") or "none"
            raw_args = function.get("arguments") or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            if isinstance(raw_args, dict):
                arguments = {
                    str(k): v
                    for k, v in raw_args.items()
                    if v is not None and v != ""
                }

        think = (
            assistant.get("think")
            or assistant.get("_think_for_train")
            or ""
        )

    if requires_function and function_name == "none":
        function_name = row.get("tool_called") or "none"
    if not requires_function:
        function_name = "none"
        arguments = {}

    _, full_target = split_prompt_target(row["text"])
    if not think:
        match = re.search(r"<think>\s*(.*?)\s*</think>", full_target, re.DOTALL)
        if match:
            think = match.group(1).strip()

    return requires_function, function_name, arguments, str(think or "").strip()


def extract_call_block(full_target: str) -> str | None:
    match = re.search(
        r"<start_function_call>\s*call:.*?<end_function_call>",
        full_target,
        re.DOTALL,
    )
    return match.group(0).strip() if match else None


def tool_function(tool: dict[str, Any]) -> dict[str, Any]:
    return tool.get("function") or tool


def candidate_function_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool in row.get("tools_sampled") or []:
        fn = tool_function(tool)
        name = fn.get("name")
        if name:
            names.append(str(name))
    return names


def selected_tool_schema(row: dict[str, Any], function_name: str) -> dict[str, Any] | None:
    for tool in row.get("tools_sampled") or []:
        fn = tool_function(tool)
        if fn.get("name") == function_name:
            return fn
    for tool in row.get("tools") or []:
        fn = tool_function(tool)
        if fn.get("name") == function_name:
            return fn
    return None


def allowed_parameter_properties(
    row: dict[str, Any],
    function_name: str,
) -> dict[str, dict[str, Any]]:
    fn = selected_tool_schema(row, function_name)
    if not fn:
        return {}
    parameters = fn.get("parameters") or {}
    properties = parameters.get("properties") or {}
    return {
        str(k): (v if isinstance(v, dict) else {})
        for k, v in properties.items()
        if v is not None
    }
