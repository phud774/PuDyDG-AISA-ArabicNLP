#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decomposed multi-task fine-tuning + dev inference for AISA-ArabicFC.

Main idea:
    Reuse every original training row in several explicitly instructed tasks:
      1) tool_decision       -> true / false
      2) function_name       -> one candidate function name or none
      3) arguments           -> JSON arguments, conditioned on the target function
      4) call_only           -> exact AISA <start_function_call> block
      5) full_track_b        -> original <think> + structured call target

The full_track_b examples retain the original prompt unchanged. Granular tasks receive
an additional Arabic instruction in the first developer turn. Therefore, ordinary
AISA prompts at inference still request the original full behavior.

Default inference is hybrid/decomposed:
    - predict function name with the function-name subtask;
    - predict arguments with the argument subtask;
    - run one direct full generation to obtain <think> and as a fallback;
    - deterministically write the official JSONL schema.

Example:
    pip install -U transformers datasets peft accelerate huggingface_hub sentencepiece

    python aisa_decomposed_multitask.py \
        --mode all \
        --output_dir outputs/aisa_decomposed \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8

Inference only from a saved PEFT adapter:
    python aisa_decomposed_multitask.py \
        --mode infer \
        --checkpoint_dir outputs/aisa_decomposed \
        --output_dir outputs/aisa_decomposed
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_MARKER = "<start_of_turn>model\n"
DEVELOPER_MARKER = "<start_of_turn>developer\n"
END_TURN = "<end_of_turn>"

DEFAULT_MODEL_ID = "TuwaiqAcademy/AISA-AR-FunctionCall-Think"
DEFAULT_DATASET_ID = "TuwaiqAcademy/AISA-ArabicFC"
OFFICIAL_EVAL_REPO = "TuwaiqAcademy/AISA-ArabicFC-SharedTask-Leaderboard"

# The full task is repeated because it is the behavior required at final inference.
# Arguments are repeated because ArgEM is the main bottleneck/most heavily weighted metric.
DEFAULT_TASK_REPEATS = {
    "full_track_b": 2,
    "arguments": 2,
    "function_name": 1,
    "tool_decision": 1,
    "call_only": 1,
}

ID_LIKE_FIELDS = {
    "id_number",
    "iqama_number",
    "visa_number",
    "recipient_iban",
    "iban",
    "insurance_number",
    "passport_number",
    "phone",
    "phone_number",
    "national_id",
    "account_number",
    "reference_number",
    "border_number",
}

ARABIC_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "infer", "all"], default="all")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID)
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
    return parser.parse_args()


def choose_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_aisa_dataset(args: argparse.Namespace) -> DatasetDict:
    download_mode = "force_redownload" if args.force_redownload else "reuse_dataset_if_exists"
    ds = load_dataset(
        args.dataset_id,
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


def split_prompt_target(text: str) -> tuple[str, str]:
    if MODEL_MARKER not in text:
        raise ValueError("The row text does not contain the expected Gemma model-turn marker.")
    before, target = text.rsplit(MODEL_MARKER, 1)
    return before + MODEL_MARKER, target


def inject_task_instruction(base_prompt: str, instruction: str) -> str:
    """
    Insert a granular-task instruction in the first developer turn while preserving
    all original candidate declarations, time context, and the user utterance.
    """
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


# ---------------------------------------------------------------------------
# Dataset structure readers
# ---------------------------------------------------------------------------

def last_assistant_message(row: dict[str, Any]) -> dict[str, Any] | None:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return message
    return None


def extract_gold(row: dict[str, Any]) -> tuple[bool, str, dict[str, Any], str]:
    """
    Read gold labels from the structured `messages` field, matching the official
    leaderboard loader. The raw text target is only used to recover `think` if needed.
    """
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
    # Defensive fallback: the complete 27-tool registry is also available.
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


# ---------------------------------------------------------------------------
# Multi-task example generation
# ---------------------------------------------------------------------------

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


def iter_multitask_examples(
    dataset: Dataset,
    task_repeats: dict[str, int],
    negative_repeat: int,
) -> Iterator[dict[str, str]]:
    for row_index, row in enumerate(dataset):
        base_prompt, full_target_raw = split_prompt_target(row["text"])
        full_target = ensure_end_turn(full_target_raw)

        requires_function, function_name, arguments, _ = extract_gold(row)
        instructions = granular_instructions(function_name)
        negative_multiplier = negative_repeat if not requires_function else 1

        examples: dict[str, tuple[str, str] | None] = {
            # Keep original prompt unchanged for final behavior.
            "full_track_b": (base_prompt, full_target),
            "tool_decision": (
                inject_task_instruction(base_prompt, instructions["tool_decision"]),
                ensure_end_turn("true" if requires_function else "false"),
            ),
            "function_name": (
                inject_task_instruction(base_prompt, instructions["function_name"]),
                ensure_end_turn(function_name),
            ),
            "arguments": (
                inject_task_instruction(base_prompt, instructions["arguments"]),
                ensure_end_turn(
                    json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ),
            ),
            "call_only": None,
        }

        if requires_function:
            call_block = extract_call_block(full_target_raw)
            if call_block:
                examples["call_only"] = (
                    inject_task_instruction(base_prompt, instructions["call_only"]),
                    ensure_end_turn(call_block),
                )

        for task_name, pair in examples.items():
            if pair is None:
                continue
            repeat = int(task_repeats.get(task_name, 0))
            if repeat <= 0:
                continue
            if not requires_function:
                repeat *= negative_multiplier

            prompt, target = pair
            for repeat_index in range(repeat):
                yield {
                    "prompt": prompt,
                    "target": target,
                    "task": task_name,
                    "source_index": str(row_index),
                    "repeat_index": str(repeat_index),
                }


def build_multitask_dataset(
    train_ds: Dataset,
    negative_repeat: int,
    seed: int,
) -> Dataset:
    mt_ds = Dataset.from_generator(
        iter_multitask_examples,
        gen_kwargs={
            "dataset": train_ds,
            "task_repeats": DEFAULT_TASK_REPEATS,
            "negative_repeat": negative_repeat,
        },
    )
    mt_ds = mt_ds.shuffle(seed=seed)

    task_counts = Counter(mt_ds["task"])
    print("Multi-task rows:", len(mt_ds))
    print("Task distribution:", dict(task_counts))
    return mt_ds


def tokenize_supervised_pair(
    example: dict[str, str],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    target_ids = tokenizer(
        example["target"],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
        target_ids.append(eos_id)

    if len(target_ids) >= max_length:
        target_ids = target_ids[: max_length - 1]
        if eos_id is not None:
            target_ids.append(eos_id)
        prompt_ids = []
    else:
        prompt_budget = max_length - len(target_ids)
        if len(prompt_ids) > prompt_budget:
            # Defensive head+tail truncation: preserve early tool declarations and
            # the final time/user/model turns. Current AISA prompts normally fit.
            head = max(1, int(prompt_budget * 0.72))
            tail = max(0, prompt_budget - head)
            prompt_ids = prompt_ids[:head] + (prompt_ids[-tail:] if tail else [])

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Model loading and training
# ---------------------------------------------------------------------------

def load_tokenizer(model_or_checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_or_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def find_lora_target_modules(model: torch.nn.Module) -> list[str]:
    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = [name for name in candidates if name in present]
    if not targets:
        raise RuntimeError(
            "Could not find standard Gemma attention/MLP projection modules for LoRA."
        )
    return targets


def load_base_model(model_id: str, dtype: torch.dtype) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )


def train_model(
    args: argparse.Namespace,
    train_ds: Dataset,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning requires a CUDA GPU for this script.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = choose_dtype()
    tokenizer = load_tokenizer(args.model_id)
    base_model = load_base_model(args.model_id, dtype=dtype)
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    target_modules = find_lora_target_modules(base_model)
    print("LoRA target modules:", target_modules)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    multitask_ds = build_multitask_dataset(
        train_ds=train_ds,
        negative_repeat=args.negative_repeat,
        seed=args.seed,
    )

    tokenized_ds = multitask_ds.map(
        tokenize_supervised_pair,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length,
        },
        remove_columns=multitask_ds.column_names,
        desc="Tokenizing prompt/target pairs",
    )

    lengths = [len(x) for x in tokenized_ds["input_ids"]]
    print(
        "Token lengths:",
        {
            "min": min(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
            "max": max(lengths),
            "at_limit": sum(length >= args.max_length for length in lengths),
        },
    )

    bf16 = dtype == torch.bfloat16
    fp16 = dtype == torch.float16

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        data_seed=args.seed,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=collator,
    )
    trainer.train()

    trainer.model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": args.model_id,
        "dataset": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "task_repeats": DEFAULT_TASK_REPEATS,
        "negative_repeat": args.negative_repeat,
        "max_length": args.max_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "seed": args.seed,
    }
    with open(output_dir / "decomposed_training_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if args.merge_adapter:
        merged_dir = output_dir / "merged"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")

    # Restore cache for generation.
    trainer.model.config.use_cache = True
    return trainer.model, tokenizer


def load_trained_model_for_inference(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    checkpoint_dir = Path(args.checkpoint_dir or args.output_dir)
    dtype = choose_dtype()

    if (checkpoint_dir / "adapter_config.json").exists():
        tokenizer_source = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else args.model_id
        tokenizer = load_tokenizer(str(tokenizer_source))
        base_model = load_base_model(args.model_id, dtype=dtype)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        tokenizer = load_tokenizer(str(checkpoint_dir))
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


# ---------------------------------------------------------------------------
# Parsing and post-processing
# ---------------------------------------------------------------------------

def extract_first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    quote = ""
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            continue

        if ch in {'"', "'"}:
            in_string = True
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_object(text: str) -> tuple[dict[str, Any], bool]:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    object_text = extract_first_balanced_object(text)
    if object_text is None:
        return {}, False

    try:
        value = json.loads(object_text)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(object_text)
        except (ValueError, SyntaxError):
            return {}, False

    return (value, True) if isinstance(value, dict) else ({}, False)


def parse_function_name(text: str, candidates: list[str]) -> tuple[str, bool]:
    lower = text.strip().lower()

    tag_match = re.search(
        r"<function_name>\s*([A-Za-z_][A-Za-z0-9_]*)\s*</function_name>",
        text,
    )
    if tag_match:
        name = tag_match.group(1)
        if name in candidates or name == "none":
            return name, True

    # Exact candidate occurrence has priority over generic text cleanup.
    for name in candidates:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
            return name, True

    first_line = text.split(END_TURN, 1)[0].strip().splitlines()[0].strip() if text.strip() else ""
    first_token_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", first_line)
    if first_token_match:
        token = first_token_match.group(0)
        if token in candidates or token == "none":
            return token, True

    no_call_markers = [
        "none",
        "false",
        "لا يحتاج",
        "لا تتطلب",
        "لا يتطلب",
        "بدون أداة",
        "لا توجد دالة",
    ]
    if any(marker in lower for marker in no_call_markers):
        return "none", True

    return "none", False


def parse_scalar_from_custom_call(key: str, value: str) -> Any:
    value = value.strip()
    if key in ID_LIKE_FIELDS:
        return value

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    ascii_value = value.translate(ARABIC_DIGIT_TRANSLATION)
    try:
        if re.fullmatch(r"[+-]?\d+", ascii_value):
            return int(ascii_value)
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)", ascii_value):
            return float(ascii_value)
    except ValueError:
        pass
    return value


def parse_direct_output(text: str) -> dict[str, Any]:
    output = {
        "requires_function": False,
        "function_name": "none",
        "arguments": {},
        "think": "",
    }

    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
    if think_match:
        output["think"] = think_match.group(1).strip()

    call_match = re.search(
        r"<start_function_call>\s*call:([A-Za-z_][A-Za-z0-9_]*)\{(.*?)\}"
        r"\s*<end_function_call>",
        text,
        re.DOTALL,
    )
    if not call_match:
        return output

    output["requires_function"] = True
    output["function_name"] = call_match.group(1)
    args_block = call_match.group(2)

    matches = re.findall(
        r"([A-Za-z_][A-Za-z0-9_]*):(?:<escape>(.*?)<escape>|([^,}]+))",
        args_block,
        re.DOTALL,
    )
    for key, escaped_value, raw_value in matches:
        value = escaped_value if escaped_value != "" else raw_value
        output["arguments"][key] = parse_scalar_from_custom_call(key, value)

    return output


def to_ascii_number_string(value: Any) -> str:
    return str(value).translate(ARABIC_DIGIT_TRANSLATION).replace(",", "").strip()


def cast_by_schema(key: str, value: Any, schema: dict[str, Any]) -> Any:
    if value is None:
        return None
    if key in ID_LIKE_FIELDS:
        return str(value).strip()

    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        expected_type = next((t for t in expected_type if t != "null"), None)

    try:
        if expected_type == "integer":
            if isinstance(value, bool):
                return int(value)
            return int(float(to_ascii_number_string(value)))
        if expected_type == "number":
            number = float(to_ascii_number_string(value))
            return int(number) if number.is_integer() else number
        if expected_type == "boolean":
            if isinstance(value, bool):
                return value
            lowered = str(value).strip().lower()
            if lowered in {"true", "1", "yes", "نعم"}:
                return True
            if lowered in {"false", "0", "no", "لا"}:
                return False
        if expected_type == "string":
            return str(value).strip()
        if expected_type == "array":
            # Keep genuine arrays. For a string, the official evaluator handles
            # list-like fields using its published separators/normalization.
            return value if isinstance(value, list) else str(value).strip()
    except (TypeError, ValueError):
        pass
    return value


def sanitize_arguments(
    raw_args: dict[str, Any],
    row: dict[str, Any],
    function_name: str,
) -> dict[str, Any]:
    properties = allowed_parameter_properties(row, function_name)
    if not isinstance(raw_args, dict):
        return {}

    result: dict[str, Any] = {}
    for key, value in raw_args.items():
        key = str(key)
        if properties and key not in properties:
            continue
        if value is None or value == "":
            continue
        result[key] = cast_by_schema(key, value, properties.get(key, {}))
    return result


# ---------------------------------------------------------------------------
# Batched generation and decomposed inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
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

    for start in tqdm(range(0, len(prompts), batch_size), desc=description):
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


def run_decomposed_inference(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dev_ds: Dataset,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [dev_ds[i] for i in range(len(dev_ds))]

    # Direct generation supplies the Arabic think trace and fallback full call.
    direct_prompts = [split_prompt_target(row["text"])[0] for row in rows]
    direct_raw = batched_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=direct_prompts,
        batch_size=args.inference_batch_size,
        max_new_tokens=args.max_new_tokens_direct,
        max_prompt_length=args.max_length,
        description="Direct Track-B generation",
    )
    direct_parsed = [parse_direct_output(text) for text in direct_raw]

    # Granular stage 1: choose a function name or none.
    name_prompts = [build_function_name_prompt(row) for row in rows]
    name_raw = batched_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=name_prompts,
        batch_size=args.inference_batch_size,
        max_new_tokens=args.max_new_tokens_name,
        max_prompt_length=args.max_length,
        description="Function-name stage",
    )

    predicted_names: list[str] = []
    name_validity: list[bool] = []
    for row, raw, direct in zip(rows, name_raw, direct_parsed):
        name, valid = parse_function_name(raw, candidate_function_names(row))
        if not valid:
            direct_name = direct.get("function_name") or "none"
            if direct_name in candidate_function_names(row) or direct_name == "none":
                name = direct_name
        predicted_names.append(name)
        name_validity.append(valid)

    # Granular stage 2: extract JSON arguments conditioned on the predicted function.
    positive_indices = [
        index for index, function_name in enumerate(predicted_names)
        if function_name != "none"
    ]
    argument_prompts = [
        build_arguments_prompt(rows[index], predicted_names[index])
        for index in positive_indices
    ]
    argument_raw_positive = batched_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=argument_prompts,
        batch_size=args.inference_batch_size,
        max_new_tokens=args.max_new_tokens_args,
        max_prompt_length=args.max_length,
        description="Argument stage",
    )

    argument_raw_by_index: dict[int, str] = {
        index: raw
        for index, raw in zip(positive_indices, argument_raw_positive)
    }

    predictions: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        function_name = predicted_names[index]
        direct = direct_parsed[index]

        if function_name == "none":
            arguments = {}
            args_valid = True
            args_raw = ""
        else:
            args_raw = argument_raw_by_index.get(index, "")
            parsed_args, args_valid = parse_json_object(args_raw)
            arguments = sanitize_arguments(parsed_args, row, function_name)

            # Use direct full-call arguments only when the granular JSON stage failed.
            if not args_valid and direct.get("function_name") == function_name:
                arguments = sanitize_arguments(
                    direct.get("arguments") or {},
                    row,
                    function_name,
                )

        think = str(direct.get("think") or "").strip()
        if not think and args.fill_empty_think:
            think = fallback_think(function_name)

        prediction = {
            "id": index,
            "tool_called": function_name,
            "arguments": arguments,
            "think": think,
        }
        predictions.append(prediction)

        debug_rows.append(
            {
                "id": index,
                "candidate_functions": candidate_function_names(row),
                "function_name_raw": name_raw[index],
                "function_name_valid": name_validity[index],
                "arguments_raw": args_raw,
                "arguments_json_valid": args_valid,
                "direct_raw": direct_raw[index],
                "final_prediction": prediction,
            }
        )

    return predictions, debug_rows


# ---------------------------------------------------------------------------
# Saving and official evaluation
# ---------------------------------------------------------------------------

def save_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_official_gold(dev_ds: Dataset) -> list[dict[str, Any]]:
    """
    Equivalent to the organizer's published data_loader.load_gold().
    """
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
    """
    Download and import the organizer's current eval_lib.py + normalize.py from
    the leaderboard Space, avoiding an approximate local ArgEM implementation.
    """
    from huggingface_hub import hf_hub_download

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
    dev_ds: Dataset,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_aisa_dataset(args)

    model: torch.nn.Module
    tokenizer: AutoTokenizer

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
