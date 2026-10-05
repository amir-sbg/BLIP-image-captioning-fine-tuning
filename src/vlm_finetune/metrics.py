from __future__ import annotations

import re
from collections import Counter


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


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


def token_f1_score(reference: str, prediction: str) -> float:
    return _token_f1(reference, prediction)


def rouge_l_score(reference: str, prediction: str) -> float:
    reference_tokens = _tokens(reference)
    prediction_tokens = _tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return float(reference_tokens == prediction_tokens)

    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current

    overlap = previous[-1]
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def caption_pair_diagnostics(reference: str, prediction: str) -> dict[str, float | int]:
    reference_tokens = _tokens(reference)
    prediction_tokens = _tokens(prediction)
    return {
        "reference_tokens": len(reference_tokens),
        "prediction_tokens": len(prediction_tokens),
        "token_f1": round(token_f1_score(reference, prediction), 4),
        "rouge_l": round(rouge_l_score(reference, prediction), 4),
        "length_delta": len(prediction_tokens) - len(reference_tokens),
    }


def caption_pair_flags(reference: str, prediction: str) -> list[str]:
    diagnostics = caption_pair_diagnostics(reference, prediction)
    flags = []
    if diagnostics["prediction_tokens"] == 0:
        flags.append("empty_prediction")
    if float(diagnostics["token_f1"]) < 0.25:
        flags.append("low_token_overlap")
    if int(diagnostics["length_delta"]) <= -3:
        flags.append("much_shorter_than_reference")
    if int(diagnostics["length_delta"]) >= 4:
        flags.append("much_longer_than_reference")
    return flags


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


def _vocabulary(captions: list[str]) -> set[str]:
    return {token for caption in captions for token in _tokens(caption)}


def reference_token_coverage(references: list[str], predictions: list[str]) -> float:
    reference_vocab = _vocabulary(references)
    if not reference_vocab:
        return 0.0
    prediction_vocab = _vocabulary(predictions)
    return len(reference_vocab & prediction_vocab) / len(reference_vocab)


def novel_prediction_token_rate(references: list[str], predictions: list[str]) -> float:
    reference_vocab = _vocabulary(references)
    prediction_tokens = [token for caption in predictions for token in _tokens(caption)]
    if not prediction_tokens:
        return 0.0
    novel = sum(1 for token in prediction_tokens if token not in reference_vocab)
    return novel / len(prediction_tokens)


def caption_length_buckets(
    references: list[str],
    predictions: list[str],
    tolerance: int = 2,
) -> dict[str, int]:
    if tolerance < 0:
        raise ValueError("tolerance must not be negative")
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    buckets = {"too_short": 0, "close": 0, "too_long": 0}
    for reference, prediction in zip(references, predictions):
        delta = len(_tokens(prediction)) - len(_tokens(reference))
        if delta < -tolerance:
            buckets["too_short"] += 1
        elif delta > tolerance:
            buckets["too_long"] += 1
        else:
            buckets["close"] += 1
    return buckets


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
    length_buckets = caption_length_buckets(references, predictions)

    return {
        "reference_mean_tokens": reference_mean,
        "prediction_mean_tokens": prediction_mean,
        "mean_length_delta": prediction_mean - reference_mean,
        "prediction_to_reference_length": (
            prediction_mean / reference_mean if reference_mean else 0.0
        ),
        "prediction_distinct_unigrams": _distinct_ngram_ratio(predictions, 1),
        "prediction_distinct_bigrams": _distinct_ngram_ratio(predictions, 2),
        "prediction_repeated_bigram_rate": _repeated_ngram_rate(predictions, 2),
        "reference_token_coverage": reference_token_coverage(references, predictions),
        "novel_prediction_token_rate": novel_prediction_token_rate(
            references,
            predictions,
        ),
        "reference_distinct_bigrams": _distinct_ngram_ratio(references, 2),
        "empty_prediction_rate": empty_predictions / len(predictions) if predictions else 0.0,
        "empty_predictions": empty_predictions,
        "length_bucket_too_short": length_buckets["too_short"],
        "length_bucket_close": length_buckets["close"],
        "length_bucket_too_long": length_buckets["too_long"],
    }


def caption_quality_flags(metrics: dict[str, float | int]) -> list[str]:
    flags = []
    if float(metrics.get("empty_prediction_rate", 0.0)) > 0.0:
        flags.append("empty_predictions")
    if float(metrics.get("prediction_to_reference_length", 1.0)) < 0.65:
        flags.append("short_caption_bias")
    if float(metrics.get("prediction_to_reference_length", 1.0)) > 1.50:
        flags.append("long_caption_bias")
    if float(metrics.get("prediction_repeated_bigram_rate", 0.0)) > 0.20:
        flags.append("repetitive_bigrams")
    if float(metrics.get("reference_token_coverage", 1.0)) < 0.40:
        flags.append("low_reference_vocabulary_coverage")
    if float(metrics.get("novel_prediction_token_rate", 0.0)) > 0.50:
        flags.append("many_novel_tokens")
    return flags


def caption_metrics(
    references: list[str],
    predictions: list[str],
) -> dict[str, float | int | list[str]]:
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
    rouge_l = sum(
        rouge_l_score(reference, prediction)
        for reference, prediction in zip(references, predictions)
    ) / len(references)
    metrics = {
        "n_examples": len(references),
        "exact_match": exact_matches / len(references),
        "token_f1": token_f1,
        "rouge_l": rouge_l,
        **caption_diagnostics(references, predictions),
    }
    metrics["quality_flags"] = caption_quality_flags(metrics)
    return metrics
