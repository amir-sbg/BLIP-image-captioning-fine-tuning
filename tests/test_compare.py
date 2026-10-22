from __future__ import annotations

import json

from vlm_finetune.compare import compare_prediction_files


def test_compare_prediction_files_uses_aligned_examples(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    references = ["a red bird", "a blue fish"]
    baseline_path.write_text(
        json.dumps(
            [
                {"reference": reference, "prediction": "an object"}
                for reference in references
            ]
        )
    )
    candidate_path.write_text(
        json.dumps(
            [
                {"reference": reference, "prediction": reference}
                for reference in references
            ]
        )
    )

    report = compare_prediction_files(
        baseline_path,
        candidate_path,
        samples=100,
    )

    assert report["n_examples"] == 2
    assert report["candidate_minus_baseline"] > 0
