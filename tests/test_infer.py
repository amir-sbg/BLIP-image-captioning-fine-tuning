from __future__ import annotations

import json

import pytest

from vlm_finetune.infer import (
    build_inference_manifest,
    build_parser,
    discover_image_paths,
    limit_image_paths,
    save_inference_csv,
    save_inference_results,
)


def test_inference_parser_reads_beam_count_as_an_integer(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--image",
            str(tmp_path / "image.jpg"),
            "--num-beams",
            "5",
        ]
    )
    assert args.num_beams == 5


def test_inference_parser_reads_repetition_penalty(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--image",
            str(tmp_path / "image.jpg"),
            "--repetition-penalty",
            "1.15",
        ]
    )

    assert args.repetition_penalty == 1.15


def test_inference_parser_accepts_a_caption_prompt(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--image",
            str(tmp_path / "image.jpg"),
            "--prompt",
            "a watercolor illustration",
        ]
    )

    assert args.prompt == "a watercolor illustration"


def test_inference_parser_accepts_manifest_output(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--image",
            str(tmp_path / "image.jpg"),
            "--manifest-output",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert args.manifest_output == tmp_path / "manifest.json"


def test_discover_image_paths_filters_and_sorts_files(tmp_path) -> None:
    (tmp_path / "zebra.JPG").write_text("")
    (tmp_path / "notes.txt").write_text("")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "apple.png").write_text("")

    assert discover_image_paths(tmp_path) == [tmp_path / "zebra.JPG"]
    assert discover_image_paths(tmp_path, recursive=True) == [
        nested / "apple.png",
        tmp_path / "zebra.JPG",
    ]


def test_discover_image_paths_rejects_missing_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="not found"):
        discover_image_paths(tmp_path / "missing")


def test_limit_image_paths_keeps_original_order(tmp_path) -> None:
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"]

    assert limit_image_paths(paths, 2) == paths[:2]
    assert limit_image_paths(paths, None) == paths


def test_limit_image_paths_rejects_non_positive_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="limit"):
        limit_image_paths([tmp_path / "a.jpg"], 0)


def test_inference_results_can_be_saved_as_json(tmp_path) -> None:
    output_path = tmp_path / "reports" / "captions.json"
    save_inference_results(
        [{"image": "photo.jpg", "caption": "a small house"}],
        output_path,
    )

    assert json.loads(output_path.read_text()) == [
        {"image": "photo.jpg", "caption": "a small house"}
    ]


def test_inference_manifest_records_decoding_settings(tmp_path) -> None:
    image_paths = [tmp_path / "b.PNG", tmp_path / "a.jpg"]
    manifest = build_inference_manifest(
        model_dir=tmp_path / "model",
        image_paths=image_paths,
        device_name="cpu",
        batch_size=2,
        max_new_tokens=24,
        num_beams=4,
        repetition_penalty=1.1,
        prompt="a close-up photo",
    )

    assert manifest["image_count"] == 2
    assert manifest["image_extensions"] == [".jpg", ".png"]
    assert manifest["decoding"]["num_beams"] == 4
    assert manifest["decoding"]["prompt"] == "a close-up photo"


def test_inference_results_can_be_saved_as_csv(tmp_path) -> None:
    output_path = tmp_path / "reports" / "captions.csv"
    save_inference_csv(
        [{"image": "photo.jpg", "caption": "a small house"}],
        output_path,
    )

    assert output_path.read_text().splitlines() == [
        "image,caption",
        "photo.jpg,a small house",
    ]
