# AISA Arabic Function-Calling Reproduction Package

Important note on reported results. At submission time, we reran training and inference once to prepare this reproduction package. The metrics and prediction artifacts in outputs/ therefore differ from the experimental results reported in the paper. The paper reports results obtained during our original experimental runs, whereas the packaged outputs come from the later rerun and are affected by nondeterministic training and differences in the execution environment.

This repository contains the model artifacts, source code, and outputs needed to reproduce the submitted AISA Arabic function-calling system.
Base model: https://huggingface.co/TuwaiqAcademy/AISA-AR-FunctionCall-Think

## Repository contents

- `aisa_decomposed_multitask_toolinfo.py`: training and inference entry point.
- `augment_aisa_by_function.py` and `augment_aisa_arguments.py`: optional augmentation generator and helper.
- `aisa_function_fewshot_augmentations.jsonl`: exact augmentation data used in the reported training run.
- `outputs/aisa_decomposed`: final adapter, tokenizer, configuration, original predictions, debug files, and metrics from the initial run.
- `reproduced`: verification outputs produced on RTX 4090, RTX 3090, and RTX PRO 5000 GPUs.

Intermediate optimizer checkpoints are intentionally omitted because they are not needed for inference or reproduction of the submitted predictions.

## Inference

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_aisa.txt
```

For exact reproduction of the submitted files, we recommend an NVIDIA RTX PRO 5000 GPU:

```bash
python3 aisa_decomposed_multitask_toolinfo.py \
  --mode infer \
  --checkpoint_dir outputs/aisa_decomposed \
  --output_dir reproduced/aisa_decomposed
```

The final test prediction is written to `reproduced/aisa_decomposed/aisa_test_submission.jsonl`. The same directory also contains development predictions, debug-generation files, and available evaluation metrics.

## Training and inference from scratch

```bash
python3 aisa_decomposed_multitask_toolinfo.py \
  --mode all \
  --output_dir outputs/aisa_decomposed \
  --num_train_epochs 1 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 8
```

The supplied `aisa_function_fewshot_augmentations.jsonl` file is used by default during training.

## Optional: regenerate the augmentation file

Use a separate environment:

```bash
python3 -m venv .venv-augmentation
source .venv-augmentation/bin/activate
pip install -r requirements_augmentation.txt
```

Create a local `.env` file (never commit it) containing:

```dotenv
OPENAI_API_KEY=YOUR_API_KEY
LLM_BASE_URL=YOUR_OPENAI_COMPATIBLE_BASE_URL
OPENAI_MODEL=Llama-3.3-70B-Instruct
```

Then run:

```bash
python3 augment_aisa_by_function.py \
  --output generated/aisa_function_fewshot_augmentations.jsonl \
  --num-samples 1500
```

Regeneration is stochastic and is not expected to reproduce the supplied augmentation file exactly. The current sanitizer may also reject more candidates than the earlier version used to generate the supplied file.

## Data declaration

The model was trained on the official AISA-ArabicFC training split together with 500 synthetic argument-extraction examples generated only from official training examples and released tool schemas. Test examples, test gold labels, and external datasets containing test items were not used for training.

## Reproducibility notes

The training seed was fixed at 42, but deterministic PyTorch/CUDA algorithms and deterministic GPU kernel settings were not enforced. Retraining is therefore not expected to reproduce the original weights or predictions exactly.

The original training and inference run was performed on a temporary Vast.ai GPU instance, with predictions generated immediately after training in the same `--mode all` process. The instance was later destroyed, so its exact container image, CUDA driver, and package state are unavailable.

Inference on an NVIDIA RTX PRO 5000 reproduced every original prediction, metric, and debug file byte-for-byte. The original and reproduced test submissions have SHA-256:

```text
be09bdcad2db5932435127c3f837f1982e9661dbd59b24fd8d5982ba4d0c8562
```

The RTX 4090 runs are mutually byte-identical but differ slightly from the original output, illustrating that greedy BF16 generation can vary across GPU architectures or CUDA kernel paths.
