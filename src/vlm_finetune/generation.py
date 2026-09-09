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
    min_new_tokens: int | None = None,
    num_beams: int = 3,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    prompt: str | None = None,
) -> list[str]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if min_new_tokens is not None:
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens must not be negative")
        if min_new_tokens > max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens")
    if num_beams < 1:
        raise ValueError("num_beams must be at least 1")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must not be negative")
    if prompt is not None and not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not images:
        return []

    model.eval()
    captions: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_images = list(images[start : start + batch_size])
            processor_kwargs = {
                "images": batch_images,
                "return_tensors": "pt",
            }
            if prompt is not None:
                processor_kwargs["text"] = [prompt] * len(batch_images)
            inputs = processor(**processor_kwargs)
            inputs = _move_to_device(inputs, device)
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
            }
            if min_new_tokens is not None:
                generate_kwargs["min_new_tokens"] = min_new_tokens
            generated_ids = model.generate(**inputs, **generate_kwargs)
            captions.extend(
                caption.strip()
                for caption in processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            )
    return captions
