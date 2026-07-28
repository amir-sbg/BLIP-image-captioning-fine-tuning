from __future__ import annotations

import torch

from vlm_finetune import model as model_module


class FakeModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self


def test_load_blip_places_inference_model_on_requested_device(monkeypatch) -> None:
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module.BlipProcessor,
        "from_pretrained",
        lambda cls, name: object(),
    )
    monkeypatch.setattr(
        model_module.BlipForConditionalGeneration,
        "from_pretrained",
        lambda cls, name: fake_model,
    )

    loaded_model, _ = model_module.load_blip("local-model", device="cpu")

    assert loaded_model is fake_model
    assert loaded_model.device == torch.device("cpu")
