from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from .data import resolve_caption_column
from .generation import generate_captions
from .metrics import caption_metrics


def save_caption_predictions(
    references: list[str],
    predictions: list[str],
    path: Path,
) -> None:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "index": index,
            "reference": reference,
            "prediction": prediction,
        }
        for index, (reference, prediction) in enumerate(
            zip(references, predictions)
        )
    ]
    path.write_text(json.dumps(records, indent=2) + "\n")


def evaluate_captions(
    model,
    processor,
    dataset: Dataset,
    device,
    image_column: str = "image",
    caption_column: str | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 32,
    num_beams: int = 3,
    repetition_penalty: float = 1.0,
    predictions_path: Path | None = None,
) -> dict[str, float | int]:
    if image_column not in dataset.column_names:
        raise ValueError(f"image column not found: {image_column}")
    resolved_caption_column = resolve_caption_column(dataset, caption_column)
    references = [str(caption) for caption in dataset[resolved_caption_column]]
    predictions = generate_captions(
        model=model,
        processor=processor,
        images=dataset[image_column],
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )
    if predictions_path is not None:
        save_caption_predictions(references, predictions, predictions_path)
    return caption_metrics(references, predictions)
