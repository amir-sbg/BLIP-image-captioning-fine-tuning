from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from datasets import DatasetDict

from .config import FineTuneConfig, prepare_output_directories
from .data import load_caption_dataset, prepare_dataset
from .evaluate import evaluate_captions
from .model import load_blip
from .training import run_training


def _config_payload(config: FineTuneConfig) -> dict:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["report_dir"] = str(config.report_dir)
    return payload


def _write_json(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def run(config: FineTuneConfig) -> dict:
    prepare_output_directories(config)
    _write_json(_config_payload(config), config.report_dir / "run_config.json")

    raw_dataset = load_caption_dataset(
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        validation_size=config.validation_size,
        seed=config.seed,
        image_column=config.image_column,
        caption_column=config.caption_column,
        max_train_samples=config.max_train_samples,
        max_validation_samples=config.max_validation_samples,
    )
    model, processor = load_blip(config.model_name)
    tokenized_dataset = DatasetDict(
        {
            split: prepare_dataset(
                dataset,
                processor=processor,
                max_length=config.max_length,
                image_column=config.image_column,
                caption_column=config.caption_column,
            )
            for split, dataset in raw_dataset.items()
        }
    )
    metrics = run_training(
        model=model,
        processor=processor,
        train_dataset=tokenized_dataset["train"],
        validation_dataset=tokenized_dataset["validation"],
        config=config,
    )
    metrics["generation"] = evaluate_captions(
        model=model,
        processor=processor,
        dataset=raw_dataset["validation"],
        device=next(model.parameters()).device,
        image_column=config.image_column,
        caption_column=config.caption_column,
        batch_size=config.eval_batch_size,
        max_new_tokens=config.max_new_tokens,
        num_beams=config.eval_num_beams,
        repetition_penalty=config.eval_repetition_penalty,
        predictions_path=config.report_dir / "caption_predictions.json",
    )
    _write_json(metrics, config.report_dir / "metrics.json")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune BLIP on an image-caption dataset."
    )
    parser.add_argument("--dataset-name", default="lambdalabs/pokemon-blip-captions")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--model-name", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--caption-column")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/blip-captioner"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-validation-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-num-beams", type=int, default=3)
    parser.add_argument("--eval-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def config_from_args(args: argparse.Namespace) -> FineTuneConfig:
    return FineTuneConfig(**vars(args))


if __name__ == "__main__":
    run(config_from_args(build_parser().parse_args()))
