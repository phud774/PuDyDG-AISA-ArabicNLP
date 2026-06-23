from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download

from src.constants import OFFICIAL_EVAL_REPO
from src.dataset import extract_gold
from src.utils import save_jsonl


def build_official_gold(dev_ds: Any) -> list[dict[str, Any]]:
    gold: list[dict[str, Any]] = []
    for index, row in enumerate(dev_ds):
        requires_function, function_name, arguments, _ = extract_gold(row)
        gold.append(
            {
                "id": index,
                "tool_called": function_name if requires_function else "none",
                "arguments": arguments if requires_function else {},
                "dialect": (row.get("dialect") or "unknown").lower(),
                "requires_function": requires_function,
            }
        )
    return gold


def load_official_evaluator() -> Any:
    normalize_path = Path(
        hf_hub_download(
            repo_id=OFFICIAL_EVAL_REPO,
            repo_type="space",
            filename="normalize.py",
        )
    )
    eval_path = Path(
        hf_hub_download(
            repo_id=OFFICIAL_EVAL_REPO,
            repo_type="space",
            filename="eval_lib.py",
        )
    )

    sys.path.insert(0, str(normalize_path.parent))
    spec = importlib.util.spec_from_file_location("aisa_official_eval_lib", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official evaluator from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def evaluate_and_save(
    predictions: list[dict[str, Any]],
    dev_ds: Any,
    output_dir: Path,
) -> dict[str, Any]:
    gold = build_official_gold(dev_ds)
    evaluate = load_official_evaluator()
    metrics = evaluate(predictions, gold)

    metrics_path = output_dir / "aisa_dev_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics saved to {metrics_path}")
    return metrics
