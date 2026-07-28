from __future__ import annotations

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


def load_blip(
    model_name: str,
    device: torch.device | str | None = None,
) -> tuple[BlipForConditionalGeneration, BlipProcessor]:
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    if device is not None:
        model.to(device)
    return model, processor
