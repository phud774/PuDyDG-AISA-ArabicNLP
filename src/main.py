from __future__ import annotations

import random
from pathlib import Path

from transformers import set_seed

from src.dataset import load_aisa_dataset
from src.eval import evaluate_and_save
from src.inference import run_decomposed_inference
from src.modeling import load_trained_model_for_inference, train_model
from src.utils import parse_args, save_jsonl


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_aisa_dataset(args)

    if args.mode in {"train", "all"}:
        model, tokenizer = train_model(args, ds["train"])
    else:
        model, tokenizer = load_trained_model_for_inference(args)

    if args.mode in {"infer", "all"}:
        predictions, debug_rows = run_decomposed_inference(
            args=args,
            model=model,
            tokenizer=tokenizer,
            dev_ds=ds["dev"],
            train_ds=ds.get("train"),
        )

        submission_path = output_dir / "aisa_dev_submission.jsonl"
        debug_path = output_dir / "aisa_dev_debug.jsonl"
        save_jsonl(predictions, submission_path)
        save_jsonl(debug_rows, debug_path)

        print(f"Submission saved to {submission_path}")
        print(f"Debug generations saved to {debug_path}")
        evaluate_and_save(predictions, ds["dev"], output_dir)


if __name__ == "__main__":
    main()
