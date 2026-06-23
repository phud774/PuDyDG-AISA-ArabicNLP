from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer

from src.dataset import candidate_function_names, split_prompt_target
from src.postprocess import parse_direct_output, parse_function_name, parse_json_object, sanitize_arguments
from src.utils import batched_generate, build_arguments_prompt, build_function_name_prompt, fallback_think


def run_decomposed_inference(
    args: Any,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dev_ds: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [dev_ds[i] for i in range(len(dev_ds))]

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
