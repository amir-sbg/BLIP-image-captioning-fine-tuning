from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_CAPTION_COLUMNS = ("caption", "text", "sentence")


def resolve_caption_column(
    dataset: Dataset,
    requested: str | None = None,
) -> str:
    if requested is not None:
        if requested not in dataset.column_names:
            raise ValueError(f"caption column not found: {requested}")
        return requested

    for column in DEFAULT_CAPTION_COLUMNS:
        if column in dataset.column_names:
            return column
    raise ValueError(
        "could not find a caption column; pass caption_column explicitly"
    )


def validate_caption_dataset(
    dataset: Dataset,
    image_column: str = "image",
    caption_column: str | None = None,
) -> str:
    if image_column not in dataset.column_names:
        raise ValueError(f"image column not found: {image_column}")
    return resolve_caption_column(dataset, caption_column)


def _limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def load_caption_dataset(
    dataset_name: str,
    split: str = "train",
    validation_size: float = 0.10,
    seed: int = 42,
    image_column: str = "image",
    caption_column: str | None = None,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
) -> DatasetDict:
    dataset = load_dataset(dataset_name, split=split)
    validate_caption_dataset(dataset, image_column, caption_column)
    dataset = _limit_dataset(dataset, max_train_samples)
    if len(dataset) < 2:
        raise ValueError("at least two image-caption examples are required")

    split_dataset = dataset.train_test_split(
        test_size=validation_size,
        seed=seed,
    )
    return DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": _limit_dataset(
                split_dataset["test"], max_validation_samples
            ),
        }
    )


def prepare_dataset(
    dataset: Dataset,
    processor,
    max_length: int,
    image_column: str = "image",
    caption_column: str | None = None,
) -> Dataset:
    resolved_caption_column = validate_caption_dataset(
        dataset,
        image_column=image_column,
        caption_column=caption_column,
    )
    pad_token_id = processor.tokenizer.pad_token_id

    def encode_batch(batch: dict) -> dict:
        images = [
            image.convert("RGB") if hasattr(image, "convert") else image
            for image in batch[image_column]
        ]
        captions = [str(caption) for caption in batch[resolved_caption_column]]
        encoded = processor(
            images=images,
            text=captions,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = [
            [token if token != pad_token_id else -100 for token in tokens]
            for tokens in encoded["input_ids"]
        ]
        return encoded

    return dataset.map(
        encode_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preparing image-caption pairs",
    )
