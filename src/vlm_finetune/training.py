from __future__ import annotations

from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)

from .config import FineTuneConfig


def build_training_args(config: FineTuneConfig) -> Seq2SeqTrainingArguments:
    values = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "num_train_epochs": config.epochs,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": config.seed,
    }
    try:
        return Seq2SeqTrainingArguments(eval_strategy="steps", **values)
    except TypeError:
        return Seq2SeqTrainingArguments(evaluation_strategy="steps", **values)


def build_trainer(
    model,
    processor,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: FineTuneConfig,
) -> Seq2SeqTrainer:
    values = {
        "model": model,
        "args": build_training_args(config),
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": default_data_collator,
    }
    try:
        return Seq2SeqTrainer(processing_class=processor, **values)
    except TypeError:
        return Seq2SeqTrainer(tokenizer=processor, **values)


def run_training(
    model,
    processor,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: FineTuneConfig,
) -> dict:
    set_seed(config.seed)
    trainer = build_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=config,
    )
    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    metrics.update(trainer.evaluate())
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    trainer.save_metrics("run", metrics)
    return metrics
