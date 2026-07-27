from __future__ import annotations

import pytest
import torch

from vlm_finetune.generation import generate_captions
from vlm_finetune.metrics import caption_metrics


class TinyProcessor:
    def __call__(self, images, return_tensors):
        return {
            "pixel_values": torch.ones((len(images), 3, 2, 2)),
        }

    def batch_decode(self, generated_ids, skip_special_tokens):
        return ["a small object"] * len(generated_ids)


class TinyModel:
    def eval(self):
        return self

    def generate(self, pixel_values, max_new_tokens, num_beams):
        return torch.zeros((len(pixel_values), 3), dtype=torch.long)


def test_generation_batches_images() -> None:
    captions = generate_captions(
        TinyModel(),
        TinyProcessor(),
        ["one", "two", "three"],
        torch.device("cpu"),
        batch_size=2,
    )
    assert captions == ["a small object"] * 3


def test_caption_metrics_include_token_overlap() -> None:
    metrics = caption_metrics(
        ["a red car", "a blue bike"],
        ["red car", "a green bike"],
    )
    assert metrics["n_examples"] == 2
    assert 0 < metrics["token_f1"] < 1
    assert metrics["exact_match"] == 0


def test_caption_metrics_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        caption_metrics(["one"], [])
