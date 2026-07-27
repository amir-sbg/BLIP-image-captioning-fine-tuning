from __future__ import annotations

from vlm_finetune import training
from vlm_finetune.config import FineTuneConfig


class FakeResult:
    metrics = {"train_loss": 0.4}


class FakeTrainer:
    def __init__(self) -> None:
        self.train_kwargs = None

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return FakeResult()

    def evaluate(self):
        return {"eval_loss": 0.3}

    def save_model(self, output_dir):
        pass

    def save_metrics(self, split, metrics):
        pass


class FakeProcessor:
    def save_pretrained(self, output_dir):
        pass


def test_run_training_passes_checkpoint_to_trainer(monkeypatch, tmp_path) -> None:
    fake_trainer = FakeTrainer()
    monkeypatch.setattr(training, "build_trainer", lambda **kwargs: fake_trainer)
    config = FineTuneConfig(
        output_dir=tmp_path / "output",
        resume_from_checkpoint=tmp_path / "checkpoint-100",
    )

    metrics = training.run_training(
        model=None,
        processor=FakeProcessor(),
        train_dataset=None,
        validation_dataset=None,
        config=config,
    )

    assert fake_trainer.train_kwargs == {
        "resume_from_checkpoint": str(tmp_path / "checkpoint-100")
    }
    assert metrics == {"train_loss": 0.4, "eval_loss": 0.3}
