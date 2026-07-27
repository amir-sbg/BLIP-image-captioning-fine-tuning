from __future__ import annotations

from datasets import Dataset

from .data import resolve_caption_column
from .generation import generate_captions
from .metrics import caption_metrics


def evaluate_captions(
    model,
    processor,
    dataset: Dataset,
    device,
    image_column: str = "image",
    caption_column: str | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 32,
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
    )
    return caption_metrics(references, predictions)
