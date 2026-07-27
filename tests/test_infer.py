from __future__ import annotations

from vlm_finetune.infer import build_parser


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
