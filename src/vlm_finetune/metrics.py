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


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be at least 1")
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _distinct_ngram_ratio(captions: list[str], n: int) -> float:
    all_ngrams = [
        ngram
        for caption in captions
        for ngram in _ngrams(_tokens(caption), n)
    ]
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def _repeated_ngram_rate(captions: list[str], n: int) -> float:
    repeated = 0
    total = 0
    for caption in captions:
        grams = _ngrams(_tokens(caption), n)
        total += len(grams)
        repeated += len(grams) - len(set(grams))
    return repeated / total if total else 0.0


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
        "prediction_distinct_unigrams": _distinct_ngram_ratio(predictions, 1),
        "prediction_distinct_bigrams": _distinct_ngram_ratio(predictions, 2),
        "prediction_repeated_bigram_rate": _repeated_ngram_rate(predictions, 2),
        "reference_distinct_bigrams": _distinct_ngram_ratio(references, 2),
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
