from __future__ import annotations

import json
from collections import Counter
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from src.constants import DEFAULT_TASK_REPEATS
from src.dataset import extract_gold
from src.utils import inject_task_instruction, ensure_end_turn, split_prompt_target


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
) -> Any:
    for row_index, row in enumerate(dataset):
        base_prompt, full_target_raw = split_prompt_target(row["text"])
        full_target = ensure_end_turn(full_target_raw)

        requires_function, function_name, arguments, _ = extract_gold(row)
        instructions = granular_instructions(function_name)
        negative_multiplier = negative_repeat if not requires_function else 1

        examples: dict[str, tuple[str, str] | None] = {
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
            from src.dataset import extract_call_block

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
    from datasets import Dataset as HFDataset

    mt_ds = HFDataset.from_generator(
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
