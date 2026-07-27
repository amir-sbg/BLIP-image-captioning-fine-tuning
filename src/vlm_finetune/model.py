from __future__ import annotations

from transformers import BlipForConditionalGeneration, BlipProcessor


def load_blip(model_name: str) -> tuple[
    BlipForConditionalGeneration,
    BlipProcessor,
]:
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return model, processor
