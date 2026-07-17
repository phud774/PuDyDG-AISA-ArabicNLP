#!/usr/bin/env python3
"""Extract only augmentation rows from a combined AISA JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/aisa_train_with_argument_augmentations.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/aisa_argument_augmentations_only.jsonl",
    )
    args = parser.parse_args()

    input_path, output_path = Path(args.input), Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("id", "")).startswith("aug_"):
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(f"Saved {count} augmentation rows to {output_path}")


if __name__ == "__main__":
    main()
