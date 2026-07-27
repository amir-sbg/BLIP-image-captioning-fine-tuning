from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FineTuneConfig:
    dataset_name: str = "lambdalabs/pokemon-blip-captions"
    dataset_split: str = "train"
    model_name: str = "Salesforce/blip-image-captioning-base"
    image_column: str = "image"
    caption_column: str | None = None
    output_dir: Path = Path("artifacts/blip-captioner")
    report_dir: Path = Path("reports")
    max_length: int = 64
    validation_size: float = 0.10
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    epochs: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    max_train_samples: int | None = 512
    max_validation_samples: int | None = 64
    max_new_tokens: int = 32
    resume_from_checkpoint: Path | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.dataset_name.strip() or not self.dataset_split.strip():
            raise ValueError("dataset name and split must not be empty")
        if not self.model_name.strip() or not self.image_column.strip():
            raise ValueError("model name and image column must not be empty")
        if self.caption_column is not None and not self.caption_column.strip():
            raise ValueError("caption_column must not be empty")
        if self.max_length < 4:
            raise ValueError("max_length must be at least 4")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if self.train_batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch sizes must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        if self.logging_steps < 1 or self.save_steps < 1:
            raise ValueError("logging_steps and save_steps must be at least 1")
        for name, value in (
            ("max_train_samples", self.max_train_samples),
            ("max_validation_samples", self.max_validation_samples),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")


def prepare_output_directories(config: FineTuneConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
