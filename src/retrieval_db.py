from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.dataset import extract_gold, split_prompt_target


def build_retrieval_database(
    train_ds: Any,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    records: list[dict[str, Any]] = []
    for row in train_ds:
        requires_function, function_name, arguments, _ = extract_gold(row)
        if not requires_function or function_name == "none":
            continue

        base_prompt, _ = split_prompt_target(row["text"])
        records.append(
            {
                "text": row.get("text", ""),
                "prompt": base_prompt,
                "requires_function": True,
                "tool_called": function_name,
                "function_name": function_name,
                "arguments": arguments,
                "source": "train",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def load_retrieval_database(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
