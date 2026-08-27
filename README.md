# BLIP Image Captioning Fine-Tuning

A compact VLM fine-tuning project for adapting **BLIP** to an image-caption dataset. It covers the full path from dataset preparation to training, generation-based evaluation, and inference on local images using PyTorch and Hugging Face Transformers.

The default experiment uses the public `lambdalabs/pokemon-blip-captions` dataset and `Salesforce/blip-image-captioning-base`. The sample limits keep the first run manageable while leaving the dataset, model, and training settings configurable from the command line.

## Pipeline

1. Load an image-caption dataset and create a reproducible train/validation split.
2. Convert images to RGB, normalize caption text, and tokenize captions with `BlipProcessor`.
3. Mask padding tokens in the language-model labels so they do not contribute to the loss.
4. Fine-tune `BlipForConditionalGeneration` with `Seq2SeqTrainer`, warmup, weight decay, and gradient accumulation.
5. Generate captions for the validation images and report exact match, token-level F1, length, diversity, repetition, and empty-output diagnostics.
6. Save per-example references and predictions for qualitative error analysis.
7. Save the trained model and processor for local image captioning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

The training command needs access to the Hugging Face Hub the first time it downloads the dataset and base model. A CUDA-enabled PyTorch installation is recommended, but the code also runs on CPU for small experiments.

## Train

Run the default small experiment:

```bash
python -m vlm_finetune.train
```

For a quick local smoke run, reduce both sample limits and the number of epochs:

```bash
python -m vlm_finetune.train \
  --max-train-samples 32 \
  --max-validation-samples 8 \
  --epochs 1 \
  --output-dir artifacts/smoke-run \
  --report-dir reports/smoke-run
```

Training can continue from a saved Trainer checkpoint:

```bash
python -m vlm_finetune.train \
  --resume-from-checkpoint artifacts/blip-captioner/checkpoint-100
```

The default configuration uses a maximum caption length of 64 tokens, batch size 4, learning rate `5e-5`, and one epoch. These are starting points for a small experiment, not fixed assumptions about every dataset.

## Inference

After training, generate captions for local images:

```bash
python -m vlm_finetune.infer \
  --model-dir artifacts/blip-captioner \
  --image examples/photo-one.jpg examples/photo-two.jpg \
  --num-beams 3 \
  --repetition-penalty 1.1 \
  --output reports/inference.json \
  --csv-output reports/inference.csv
```

The command prints one JSON record per image with its path and generated caption. When `--output` is provided, it also writes the same records to a JSON file; `--csv-output` writes a simple two-column review file. `--num-beams` controls deterministic beam-search width during generation, and `--repetition-penalty` can discourage repeated phrases. Use `--device cuda`, `--device mps`, or `--device cpu` to select a device explicitly; `auto` selects the first available accelerator.

An optional prompt can be applied to every image when a particular caption style or prefix is useful:

```bash
python -m vlm_finetune.infer \
  --model-dir artifacts/blip-captioner \
  --image examples/photo-one.jpg \
  --prompt "a watercolor illustration"
```

If `--prompt` is omitted, BLIP generates an unconstrained caption.

For a folder of images, use `--image-dir`; add `--recursive` to include nested folders:

```bash
python -m vlm_finetune.infer \
  --model-dir artifacts/blip-captioner \
  --image-dir examples \
  --recursive \
  --limit 25
```

## Outputs

- `artifacts/blip-captioner/` contains the fine-tuned model, processor, and Trainer checkpoints.
- `reports/run_config.json` records the experiment settings.
- `reports/metrics.json` contains training metrics, validation caption metrics, and generation diagnostics such as output length, distinct n-grams, repetition, and empty-output rate.
- `reports/caption_predictions.json` stores validation references and generated captions for review.
- `reports/inference.json` and `reports/inference.csv` are optional exports from local-image inference.
- `artifacts/blip-captioner/run_results.json` is the raw metrics file written by the Trainer.

## Project layout

```text
src/vlm_finetune/
├── config.py       experiment settings and validation
├── data.py         dataset loading and processor preparation
├── model.py        BLIP model and processor loading
├── training.py     Hugging Face Trainer configuration
├── train.py        end-to-end fine-tuning command
├── generation.py   batched caption generation
├── metrics.py      lightweight caption metrics
├── evaluate.py     validation-set evaluation
└── infer.py        local-image inference command
```
