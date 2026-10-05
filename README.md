# BLIP Image Captioning Fine-Tuning

A compact PyTorch pipeline for adapting `Salesforce/blip-image-captioning-base` to an image-caption dataset. It covers reproducible data preparation, masked-label fine-tuning, generation-based evaluation, and inference on local images.

The default experiment uses `lambdalabs/pokemon-blip-captions`. Sample limits and training settings are configurable, so the same code can be used for a quick smoke run or a larger fine-tuning job.

## Pipeline

1. Load the dataset and create a reproducible train/validation split.
2. Normalize captions and prepare BLIP processor inputs with padded labels masked from the loss.
3. Fine-tune `BlipForConditionalGeneration` with the Hugging Face Trainer.
4. Evaluate generated captions with token F1, ROUGE-L, BLEU-2, bootstrap intervals, length slices, diversity, repetition, and vocabulary coverage.
5. Save per-example predictions for qualitative review and use the trained processor/model for local inference.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

The first run downloads the dataset and base model from the Hugging Face Hub. CUDA is recommended for training; CPU is sufficient for small tests.

## Train

```bash
python -m vlm_finetune.train
```

For a small local run:

```bash
python -m vlm_finetune.train \
  --max-train-samples 32 \
  --max-validation-samples 8 \
  --epochs 1 \
  --output-dir artifacts/smoke-run \
  --report-dir reports/smoke-run
```

Training can resume from a Trainer checkpoint with `--resume-from-checkpoint`. Generation settings used during validation, including beam width, minimum length, repetition penalty, and no-repeat n-gram size, are available as command-line options.

## Inference

Caption individual files:

```bash
python -m vlm_finetune.infer \
  --model-dir artifacts/blip-captioner \
  --image examples/photo-one.jpg examples/photo-two.jpg \
  --num-beams 3 \
  --output reports/inference.json \
  --csv-output reports/inference.csv
```

Use `--image-dir` for a folder, `--recursive` for nested folders, and `--device cuda`, `mps`, or `cpu` to select a device. An optional `--prompt` can provide a caption prefix or style.

## Outputs

- `artifacts/` contains the fine-tuned model, processor, and Trainer checkpoints.
- `reports/metrics.json` stores validation and generation diagnostics with uncertainty intervals.
- `reports/data_profile.json` records split statistics and normalized-caption overlap.
- `reports/caption_predictions.json` stores references, predictions, token F1, and length deltas.
- `reports/inference.json` and `reports/inference.csv` provide local-image predictions.
- `reports/inference_manifest.json` records the selected images and decoding settings.

## Project layout

```text
src/vlm_finetune/
├── config.py       experiment settings and validation
├── data.py         dataset loading, profiling, and preprocessing
├── model.py        BLIP model and processor loading
├── training.py     Trainer configuration
├── train.py        fine-tuning entry point
├── generation.py   batched caption generation
├── metrics.py      caption metrics and diagnostics
├── evaluate.py     validation evaluation and exports
└── infer.py        local-image inference
tests/
```
