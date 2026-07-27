from __future__ import annotations

from collections.abc import Sequence

import torch


def _move_to_device(values, device: torch.device):
    if hasattr(values, "to"):
        return values.to(device)
    return {key: value.to(device) for key, value in values.items()}


def generate_captions(
    model,
    processor,
    images: Sequence,
    device: torch.device,
    batch_size: int = 4,
    max_new_tokens: int = 32,
    num_beams: int = 3,
) -> list[str]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if num_beams < 1:
        raise ValueError("num_beams must be at least 1")
    if not images:
        return []

    model.eval()
    captions: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            inputs = processor(
                images=list(images[start : start + batch_size]),
                return_tensors="pt",
            )
            inputs = _move_to_device(inputs, device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            captions.extend(
                caption.strip()
                for caption in processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            )
    return captions
