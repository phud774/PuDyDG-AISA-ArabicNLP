from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.constants import DEVELOPER_MARKER, END_TURN, MODEL_MARKER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "infer", "all"], default="all")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--dataset_id", default=None)
    parser.add_argument("--dataset_revision", default="main")
    parser.add_argument("--output_dir", default="outputs/aisa_decomposed")
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="PEFT adapter or merged model directory used by --mode infer. "
             "Defaults to --output_dir.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument(
        "--negative_repeat",
        type=int,
        default=20,
        help="Extra repetition factor for the very small no-call subset.",
    )
    parser.add_argument(
        "--force_redownload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-download the dataset so an old pre-v1.2 cache is not reused.",
    )

    parser.add_argument("--inference_batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens_direct", type=int, default=256)
    parser.add_argument("--max_new_tokens_name", type=int, default=24)
    parser.add_argument("--max_new_tokens_args", type=int, default=160)
    parser.add_argument(
        "--fill_empty_think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a short Arabic fallback when direct generation emits no <think>.",
    )
    parser.add_argument(
        "--merge_adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save a merged full model after training.",
    )
    parser.add_argument(
        "--use_openai_args",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use OpenAI API for argument generation instead of the local model.",
    )
    parser.add_argument(
        "--openai_args_model",
        default=None,
        help="Model name for OpenAI argument extraction.",
    )
    parser.add_argument(
        "--openai_args_max_tokens",
        type=int,
        default=256,
        help="Max output tokens for OpenAI argument generation.",
    )
    parser.add_argument(
        "--openai_args_fewshot_k",
        type=int,
        default=3,
        help="Number of few-shot examples to retrieve for OpenAI argument generation.",
    )
    parser.add_argument(
        "--retrieval_db_path",
        default="outputs/aisa_decomposed/retrieval_db.jsonl",
        help="Path to the persisted retrieval database built from the train split.",
    )
    parser.add_argument(
        "--build_retrieval_db",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build or rebuild the retrieval database from the train split before inference.",
    )
    return parser.parse_args()


def choose_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def split_prompt_target(text: str) -> tuple[str, str]:
    if MODEL_MARKER not in text:
        raise ValueError("The row text does not contain the expected Gemma model-turn marker.")
    before, target = text.rsplit(MODEL_MARKER, 1)
    return before + MODEL_MARKER, target


def inject_task_instruction(base_prompt: str, instruction: str) -> str:
    if DEVELOPER_MARKER not in base_prompt:
        raise ValueError("Prompt has no developer turn.")
    return base_prompt.replace(
        DEVELOPER_MARKER,
        DEVELOPER_MARKER + instruction.strip() + "\n",
        1,
    )


def ensure_end_turn(target: str) -> str:
    target = target.strip()
    if not target.endswith(END_TURN):
        target += END_TURN
    return target


def load_tokenizer(model_or_checkpoint: str, fallback_model_id: str | None = None) -> AutoTokenizer:
    candidates = [model_or_checkpoint]
    if fallback_model_id and fallback_model_id != model_or_checkpoint:
        candidates.append(fallback_model_id)

    last_error: Exception | None = None
    for candidate in candidates:
        for use_fast in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=use_fast)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
                return tokenizer
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_error = exc

    raise RuntimeError(
        "Failed to load tokenizer. Install sentencepiece and ensure the tokenizer/model files are available."
    ) from last_error


def granular_instructions(
    function_name: str,
) -> dict[str, str]:
    return {
        "tool_decision": (
            "مهمة فرعية للتدريب متعدد المهام: حدّد فقط هل يحتاج طلب المستخدم "
            "إلى استدعاء إحدى الأدوات المرشحة. أخرج true أو false فقط، من دون شرح."
        ),
        "function_name": (
            "مهمة فرعية للتدريب متعدد المهام: اختر اسم الدالة الصحيحة فقط من "
            "الدوال المرشحة. أخرج اسم الدالة وحده، أو none إذا لم يحتج الطلب إلى أداة."
        ),
        "arguments": (
            "مهمة فرعية للتدريب متعدد المهام: الدالة المستهدفة هي "
            f"{function_name}. استخرج فقط أزواج المعاملات والقيم الموجودة في طلب "
            "المستخدم والمتوافقة مع مخطط هذه الدالة. أخرج كائن JSON صالحاً فقط. "
            "إذا لم توجد دالة فأخرج {}."
        ),
        "call_only": (
            "مهمة فرعية للتدريب متعدد المهام: أنشئ كتلة استدعاء الدالة فقط "
            "بصيغة AISA، بدءاً من <start_function_call> وانتهاءً بـ "
            "<end_function_call>. لا تكتب <think> ولا أي نص إضافي."
        ),
    }


def build_function_name_prompt(row: dict[str, Any]) -> str:
    base_prompt, _ = split_prompt_target(row["text"])
    instruction = granular_instructions("none")["function_name"]
    return inject_task_instruction(base_prompt, instruction)


def build_arguments_prompt(row: dict[str, Any], function_name: str) -> str:
    base_prompt, _ = split_prompt_target(row["text"])
    instruction = granular_instructions(function_name)["arguments"]
    return inject_task_instruction(base_prompt, instruction)


def fallback_think(function_name: str) -> str:
    if function_name == "none":
        return "طلب المستخدم لا يحتاج إلى استدعاء أداة من الأدوات المتاحة."
    return f"طلب المستخدم يتطلب استخدام الدالة {function_name} مع المعاملات المستخرجة."


def batched_generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    max_prompt_length: int,
    description: str,
) -> list[str]:
    if not prompts:
        return []

    tokenizer.padding_side = "left"
    outputs: list[str] = []
    model.eval()

    device = next(model.parameters()).device

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        ).to(device)

        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        prompt_width = encoded["input_ids"].shape[1]
        new_tokens = generated[:, prompt_width:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
        outputs.extend(decoded)

    return outputs


def save_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
