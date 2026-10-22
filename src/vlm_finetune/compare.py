from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import paired_bootstrap_difference


def _load_predictions(path: Path) -> tuple[list[str], list[str]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"prediction file must contain a non-empty list: {path}")
    try:
        references = [str(record["reference"]) for record in records]
        predictions = [str(record["prediction"]) for record in records]
    except (KeyError, TypeError) as error:
        raise ValueError(f"prediction records are malformed: {path}") from error
    return references, predictions


def compare_prediction_files(
    baseline_path: Path,
    candidate_path: Path,
    metric: str = "token_f1",
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    references, baseline_predictions = _load_predictions(baseline_path)
    candidate_references, candidate_predictions = _load_predictions(candidate_path)
    if references != candidate_references:
        raise ValueError("baseline and candidate files do not contain aligned references")
    return paired_bootstrap_difference(
        references,
        baseline_predictions,
        candidate_predictions,
        metric=metric,
        samples=samples,
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two aligned BLIP caption runs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=("token_f1", "rouge_l", "bleu_2", "chrf"),
        default="token_f1",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    report = compare_prediction_files(
        args.baseline,
        args.candidate,
        metric=args.metric,
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
