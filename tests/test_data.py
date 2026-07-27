from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

from vlm_finetune.config import FineTuneConfig
from vlm_finetune.data import (
    prepare_dataset,
    resolve_caption_column,
    validate_caption_dataset,
)


class TinyTokenizer:
    pad_token_id = 0


class TinyProcessor:
    tokenizer = TinyTokenizer()

    def __call__(self, images, text, **kwargs):
        return {
            "pixel_values": np.zeros((len(images), 3, 2, 2), dtype=np.float32),
            "input_ids": [[5, 0, 0] if caption == "one" else [6, 7, 0] for caption in text],
        }


def tiny_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "image": ["image-1", "image-2"],
            "text": ["one", "two words"],
        }
    )


def test_caption_column_is_inferred() -> None:
    assert resolve_caption_column(tiny_dataset()) == "text"
    assert validate_caption_dataset(tiny_dataset()) == "text"


def test_missing_image_column_is_explicit() -> None:
    dataset = Dataset.from_dict({"text": ["caption"]})
    with pytest.raises(ValueError, match="image column"):
        validate_caption_dataset(dataset)


def test_prepare_dataset_masks_padding_tokens() -> None:
    prepared = prepare_dataset(tiny_dataset(), TinyProcessor(), max_length=3)
    assert set(prepared.column_names) == {"pixel_values", "input_ids", "labels"}
    assert prepared[0]["labels"] == [5, -100, -100]
    assert prepared[1]["labels"] == [6, 7, -100]


def test_fine_tune_config_rejects_empty_dataset_name() -> None:
    with pytest.raises(ValueError, match="dataset name"):
        FineTuneConfig(dataset_name=" ")
