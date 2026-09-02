from __future__ import annotations

import pytest
import torch

from vlm_finetune.evaluate import save_caption_predictions
from vlm_finetune.generation import generate_captions
from vlm_finetune.metrics import (
    caption_diagnostics,
    caption_metrics,
    novel_prediction_token_rate,
    reference_token_coverage,
)


class TinyProcessor:
    def __init__(self) -> None:
        self.prompts = []

    def __call__(self, images, return_tensors, text=None):
        if text is not None:
            self.prompts.extend(text)
        return {
            "pixel_values": torch.ones((len(images), 3, 2, 2)),
        }

    def batch_decode(self, generated_ids, skip_special_tokens):
        return ["a small object"] * len(generated_ids)


class TinyModel:
    def __init__(self) -> None:
        self.repetition_penalty = None
        self.num_beams = None

    def eval(self):
        return self

    def generate(self, pixel_values, max_new_tokens, num_beams, repetition_penalty):
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
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


def test_generation_rejects_invalid_beam_count() -> None:
    with pytest.raises(ValueError, match="num_beams"):
        generate_captions(
            TinyModel(),
            TinyProcessor(),
            ["one"],
            torch.device("cpu"),
            num_beams=0,
        )


def test_generation_rejects_invalid_repetition_penalty() -> None:
    with pytest.raises(ValueError, match="repetition_penalty"):
        generate_captions(
            TinyModel(),
            TinyProcessor(),
            ["one"],
            torch.device("cpu"),
            repetition_penalty=0.9,
        )


def test_generation_passes_repetition_penalty_to_model() -> None:
    model = TinyModel()
    generate_captions(
        model,
        TinyProcessor(),
        ["one"],
        torch.device("cpu"),
        repetition_penalty=1.2,
    )

    assert model.repetition_penalty == 1.2


def test_generation_passes_beam_count_to_model() -> None:
    model = TinyModel()
    generate_captions(
        model,
        TinyProcessor(),
        ["one"],
        torch.device("cpu"),
        num_beams=4,
    )

    assert model.num_beams == 4


def test_generation_passes_prompt_to_processor() -> None:
    processor = TinyProcessor()
    generate_captions(
        TinyModel(),
        processor,
        ["one", "two", "three"],
        torch.device("cpu"),
        batch_size=2,
        prompt="a watercolor illustration",
    )
    assert processor.prompts == ["a watercolor illustration"] * 3


def test_caption_metrics_include_token_overlap() -> None:
    metrics = caption_metrics(
        ["a red car", "a blue bike"],
        ["red car", "a green bike"],
    )
    assert metrics["n_examples"] == 2
    assert 0 < metrics["token_f1"] < 1
    assert metrics["exact_match"] == 0
    assert metrics["reference_mean_tokens"] == 3
    assert metrics["prediction_mean_tokens"] == 2.5


def test_caption_metrics_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        caption_metrics(["one"], [])


def test_caption_diagnostics_tracks_empty_predictions() -> None:
    diagnostics = caption_diagnostics(
        ["a small red bird", "a beach photo"],
        ["", "beach photo"],
    )

    assert diagnostics["empty_predictions"] == 1
    assert diagnostics["empty_prediction_rate"] == 0.5
    assert diagnostics["prediction_to_reference_length"] == pytest.approx(2 / 7)


def test_caption_diagnostics_reports_diversity_and_repetition() -> None:
    diagnostics = caption_diagnostics(
        ["red bird on branch", "blue fish in water"],
        ["red red red bird", "red red red bird"],
    )

    assert diagnostics["prediction_distinct_unigrams"] < 1.0
    assert diagnostics["prediction_distinct_bigrams"] < diagnostics["reference_distinct_bigrams"]
    assert diagnostics["prediction_repeated_bigram_rate"] > 0.0


def test_caption_metrics_report_reference_coverage_and_novel_tokens() -> None:
    references = ["red bird on branch", "blue fish in water"]
    predictions = ["red bird on branch", "purple robot"]

    assert reference_token_coverage(references, predictions) == pytest.approx(4 / 8)
    assert novel_prediction_token_rate(references, predictions) == pytest.approx(2 / 6)

    diagnostics = caption_diagnostics(references, predictions)
    assert diagnostics["reference_token_coverage"] == pytest.approx(4 / 8)
    assert diagnostics["novel_prediction_token_rate"] == pytest.approx(2 / 6)


def test_caption_predictions_are_saved_for_review(tmp_path) -> None:
    output_path = tmp_path / "reports" / "caption_predictions.json"
    save_caption_predictions(
        ["a red car", "a blue bike"],
        ["red car", "a green bike"],
        output_path,
    )
    records = output_path.read_text()
    assert '"reference": "a red car"' in records
    assert '"prediction": "a green bike"' in records
