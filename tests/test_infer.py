from __future__ import annotations

import pytest

from vlm_finetune.infer import build_parser, discover_image_paths


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
