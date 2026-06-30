from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from src.constants import DEFAULT_TASK_REPEATS
from src.multitask import build_multitask_dataset, tokenize_supervised_pair
from src.utils import choose_dtype, load_tokenizer


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
    args: Any,
    train_ds: Any,
) -> tuple[torch.nn.Module, object]:
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

    trainer.model.config.use_cache = True
    return trainer.model, tokenizer


def load_trained_model_for_inference(
    args: Any,
) -> tuple[torch.nn.Module, object]:
    checkpoint_dir = Path(args.checkpoint_dir or args.output_dir)
    dtype = choose_dtype()

    if (checkpoint_dir / "adapter_config.json").exists():
        tokenizer_source = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else args.model_id
        tokenizer = load_tokenizer(str(tokenizer_source), fallback_model_id=args.model_id)
        base_model = load_base_model(args.model_id, dtype=dtype)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        tokenizer = load_tokenizer(str(checkpoint_dir), fallback_model_id=args.model_id)
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
