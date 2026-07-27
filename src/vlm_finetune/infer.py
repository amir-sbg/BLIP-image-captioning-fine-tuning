from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .generation import generate_captions
from .model import load_blip

IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".webp", ".bmp"})


def discover_image_paths(image_dir: Path, recursive: bool = False) -> list[Path]:
    if not image_dir.is_dir():
        raise ValueError(f"image directory not found: {image_dir}")
    candidates = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def _select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def caption_files(
    model_dir: Path,
    image_paths: list[Path],
    device_name: str = "auto",
    batch_size: int = 4,
    max_new_tokens: int = 32,
    num_beams: int = 3,
) -> list[dict[str, str]]:
    device = _select_device(device_name)
    model, processor = load_blip(str(model_dir))
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    captions = generate_captions(
        model=model,
        processor=processor,
        images=images,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return [
        {"image": str(path), "caption": caption}
        for path, caption in zip(image_paths, captions)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate captions with a fine-tuned BLIP model.")
    parser.add_argument("--model-dir", type=Path, required=True)
    image_inputs = parser.add_mutually_exclusive_group(required=True)
    image_inputs.add_argument("--image", type=Path, nargs="+")
    image_inputs.add_argument("--image-dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=3)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    image_paths = args.image
    if args.image_dir is not None:
        try:
            image_paths = discover_image_paths(args.image_dir, args.recursive)
        except ValueError as error:
            parser.error(str(error))
        if not image_paths:
            parser.error("no supported image files were found")
    print(
        json.dumps(
            caption_files(
                model_dir=args.model_dir,
                image_paths=image_paths,
                device_name=args.device,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            ),
            indent=2,
        )
    )
