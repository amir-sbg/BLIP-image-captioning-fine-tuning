from __future__ import annotations

import re
from collections import Counter


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_f1(reference: str, prediction: str) -> float:
    reference_tokens = Counter(_tokens(reference))
    prediction_tokens = Counter(_tokens(prediction))
    overlap = sum((reference_tokens & prediction_tokens).values())
    if not reference_tokens or not prediction_tokens:
        return float(reference_tokens == prediction_tokens)
    precision = overlap / sum(prediction_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def caption_diagnostics(
    references: list[str],
    predictions: list[str],
) -> dict[str, float | int]:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")

    reference_lengths = [len(_tokens(caption)) for caption in references]
    prediction_lengths = [len(_tokens(caption)) for caption in predictions]
    reference_mean = _mean(reference_lengths)
    prediction_mean = _mean(prediction_lengths)
    empty_predictions = sum(length == 0 for length in prediction_lengths)

    return {
        "reference_mean_tokens": reference_mean,
        "prediction_mean_tokens": prediction_mean,
        "prediction_to_reference_length": (
            prediction_mean / reference_mean if reference_mean else 0.0
        ),
        "empty_prediction_rate": empty_predictions / len(predictions) if predictions else 0.0,
        "empty_predictions": empty_predictions,
    }


def caption_metrics(
    references: list[str],
    predictions: list[str],
) -> dict[str, float | int]:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    if not references:
        raise ValueError("at least one caption is required")

    exact_matches = sum(
        reference.strip().lower() == prediction.strip().lower()
        for reference, prediction in zip(references, predictions)
    )
    token_f1 = sum(
        _token_f1(reference, prediction)
        for reference, prediction in zip(references, predictions)
    ) / len(references)
    return {
        "n_examples": len(references),
        "exact_match": exact_matches / len(references),
        "token_f1": token_f1,
        **caption_diagnostics(references, predictions),
    }
