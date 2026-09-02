from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

from vlm_finetune.config import FineTuneConfig
from vlm_finetune.data import (
    caption_dataset_profile,
    normalize_caption,
    prepare_dataset,
    resolve_caption_column,
    split_caption_dataset,
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


def test_caption_dataset_profile_summarizes_split() -> None:
    profile = caption_dataset_profile(tiny_dataset())

    assert profile["rows"] == 2
    assert profile["caption_column"] == "text"
    assert profile["min_caption_tokens"] == 1
    assert profile["mean_caption_tokens"] == 1.5
    assert profile["max_caption_tokens"] == 2


def test_missing_image_column_is_explicit() -> None:
    dataset = Dataset.from_dict({"text": ["caption"]})
    with pytest.raises(ValueError, match="image column"):
        validate_caption_dataset(dataset)


def test_prepare_dataset_masks_padding_tokens() -> None:
    prepared = prepare_dataset(tiny_dataset(), TinyProcessor(), max_length=3)
    assert set(prepared.column_names) == {"pixel_values", "input_ids", "labels"}
    assert prepared[0]["labels"] == [5, -100, -100]
    assert prepared[1]["labels"] == [6, 7, -100]


def test_sample_limits_are_applied_after_the_split() -> None:
    dataset = Dataset.from_dict(
        {
            "image": [f"image-{index}" for index in range(20)],
            "text": [f"caption {index}" for index in range(20)],
        }
    )

    split = split_caption_dataset(
        dataset,
        validation_size=0.25,
        seed=7,
        max_train_samples=4,
        max_validation_samples=3,
    )

    assert len(split["train"]) == 4
    assert len(split["validation"]) == 3


def test_caption_text_is_normalized_before_encoding() -> None:
    assert normalize_caption("  a   small\nobject  ") == "a small object"


def test_empty_caption_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="caption text"):
        normalize_caption("   ")


def test_fine_tune_config_rejects_empty_dataset_name() -> None:
    with pytest.raises(ValueError, match="dataset name"):
        FineTuneConfig(dataset_name=" ")


def test_fine_tune_config_validates_eval_decoding() -> None:
    with pytest.raises(ValueError, match="eval_num_beams"):
        FineTuneConfig(eval_num_beams=0)
    with pytest.raises(ValueError, match="eval_repetition_penalty"):
        FineTuneConfig(eval_repetition_penalty=0.8)
