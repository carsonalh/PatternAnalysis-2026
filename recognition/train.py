"""Direct-recurrence TimeGAN: reconstruction, transition, then joint training.

Objectives follow TimeGAN equations (7)-(11), using non-saturating BCE for G:
https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf
There is no static branch, supervisor network, or moment-matching objective.
"""

from dataclasses import asdict, dataclass
from itertools import chain
import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import LOBSTERLevel10Dataset, WindowDataset
from modules import TimeGAN
from predict import generate_wiener_paths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    sequence_length: int = 64
    batch_size: int = 64
    learning_rate: float = 3e-4
    gradient_clip: float = 1.0
    lam: float = 1.0
    eta: float = 10.0
    autoencoder_steps: int = 1_000
    transition_steps: int = 1_000
    joint_steps: int = 5_000

    def __post_init__(self):
        if self.sequence_length < 2 or self.batch_size < 1:
            raise ValueError("Training needs sequence_length >= 2 and batch_size >= 1")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or min(self.lam, self.eta) < 0:
            raise ValueError("Learning rate/clipping must be positive and loss weights nonnegative")
        if min(self.autoencoder_steps, self.transition_steps, self.joint_steps) < 0:
            raise ValueError("Training step budgets must be nonnegative")


def supervised_loss(generator: nn.Module, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(generator.teacher_forced(latents, noise), latents[:, 1:])


class TimeGANTrainer:
    def __init__(self, model: TimeGAN, config: TrainingConfig = TrainingConfig()):
        self.model = model
        self.config = config
        self.opt_ER = torch.optim.Adam(chain(model.embedder.parameters(), model.decoder.parameters()),
                                       lr=config.learning_rate)
        self.opt_G = torch.optim.Adam(model.generator.parameters(), lr=config.learning_rate)
        self.opt_D = torch.optim.Adam(model.discriminator.parameters(), lr=config.learning_rate)

    def train_only(self, *active: nn.Module) -> None:
        """Freeze parameters, not autograd through a frozen network's inputs."""
        # cuDNN RNNs need training-mode buffers to differentiate through inputs,
        # including a frozen discriminator during the generator update.
        self.model.train()
        for network in (self.model.embedder, self.model.decoder,
                        self.model.generator, self.model.discriminator):
            network.requires_grad_(network in active)
            network.zero_grad(set_to_none=True)

    def _step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite TimeGAN loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip, error_if_nonfinite=True)
        optimizer.step()

    def _noise(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return generate_wiener_paths(
            x.shape[0], x.shape[1], self.model.generator.noise_dims,
            device=x.device, dtype=x.dtype)

    def autoencoder_step(self, x: torch.Tensor) -> dict[str, float]:
        self.train_only(self.model.embedder, self.model.decoder)
        loss = F.mse_loss(self.model(x), x)
        self._step(self.opt_ER, loss)
        return {"reconstruction": loss.item()}

    def transition_step(self, x: torch.Tensor) -> dict[str, float]:
        self.train_only(self.model.generator)
        noise, _ = self._noise(x)
        with torch.no_grad():
            latents = self.model.embedder(x)
        loss = supervised_loss(self.model.generator, latents, noise)
        self._step(self.opt_G, loss)
        return {"g_supervised": loss.item()}

    def discriminator_step(self, x: torch.Tensor) -> dict[str, float]:
        self.train_only(self.model.discriminator)
        noise, seed = self._noise(x)
        with torch.no_grad():
            real = self.model.embedder(x)
            fake = self.model.generator.sample(noise, seed)
        real_logits = self.model.discriminator(real)
        fake_logits = self.model.discriminator(fake)
        loss = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
        self._step(self.opt_D, loss)
        return {"discriminator": loss.item()}

    def generator_step(self, x: torch.Tensor) -> dict[str, float]:
        self.train_only(self.model.generator)
        noise, seed = self._noise(x)
        with torch.no_grad():
            real = self.model.embedder(x)
        fake = self.model.generator.sample(noise, seed)
        # D's parameters are frozen, but D(fake) must remain differentiable.
        logits = self.model.discriminator(fake)
        adversarial = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        supervised = supervised_loss(self.model.generator, real, noise)
        loss = adversarial + self.config.eta * supervised
        self._step(self.opt_G, loss)
        return {"generator": loss.item(), "adversarial": adversarial.item(),
                "g_supervised": supervised.item()}

    def embedding_step(self, x: torch.Tensor) -> dict[str, float]:
        self.train_only(self.model.embedder, self.model.decoder)
        noise, _ = self._noise(x)
        latents = self.model.embedder(x)
        reconstruction = F.mse_loss(self.model.decoder(latents), x)
        # Both the target and previous latent stay attached to E through frozen G.
        supervised = supervised_loss(self.model.generator, latents, noise)
        loss = reconstruction + self.config.lam * supervised
        self._step(self.opt_ER, loss)
        return {"embedding": loss.item(), "reconstruction": reconstruction.item(),
                "e_supervised": supervised.item()}

    def joint_step(self, x: torch.Tensor) -> dict[str, float]:
        # Each update recomputes its graph and draws fresh noise.
        return (self.discriminator_step(x) | self.generator_step(x) | self.embedding_step(x))

    def fit(self, loader: DataLoader, log_every: int = 100) -> list[dict]:
        if len(loader) == 0:
            raise ValueError("The training split contains no complete windows")
        if log_every < 1:
            raise ValueError("log_every must be positive")
        device = next(self.model.parameters()).device
        batches = iter(loader)
        history = []
        stages = (
            ("autoencoder", self.config.autoencoder_steps, self.autoencoder_step),
            ("transition", self.config.transition_steps, self.transition_step),
            ("joint", self.config.joint_steps, self.joint_step),
        )
        for stage, steps, update in stages:
            for step in range(1, steps + 1):
                try:
                    x = next(batches)
                except StopIteration:
                    batches = iter(loader)
                    x = next(batches)
                if x.shape[1] != self.config.sequence_length:
                    raise ValueError("Loader windows must match the configured sequence length")
                metrics = update(x.to(device))
                history.append({"stage": stage, "step": step, **metrics})
                if step == 1 or step == steps or step % log_every == 0:
                    logger.info("%s %d/%d: %s", stage, step, steps,
                                ", ".join(f"{key}={value:.5f}" for key, value in metrics.items()))
        self.model.requires_grad_(True)
        self.model.eval()
        return history


def save_checkpoint(path: Path, model: TimeGAN, config: TrainingConfig,
                    data: LOBSTERLevel10Dataset) -> None:
    """Save sampling weights and the training-only feature normalization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": model.model_config,
        "training_config": asdict(config),
        "feature_mean": torch.as_tensor(data.feature_mean),
        "feature_std": torch.as_tensor(data.feature_std),
        "feature_names": list(data.FEATURE_NAMES),
    }, path)


def main():
    import argparse
    import json
    import numpy as np
    import pandas as pd

    from evaluation import evaluate_timegan

    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train a recurrent event-time TimeGAN on AMZN")
    parser.add_argument("--output", type=Path, default=Path("runs/timegan"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    for name in ("sequence_length", "batch_size", "autoencoder_steps", "transition_steps", "joint_steps"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "gradient_clip", "lam", "eta"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--latent-dims", type=int, default=128)
    parser.add_argument("--noise-dims", type=int, default=128)
    parser.add_argument("--evaluation-samples", type=int, default=256)
    parser.add_argument("--predictor-steps", type=int, default=200)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if min(args.latent_dims, args.noise_dims, args.evaluation_samples) < 1 or args.predictor_steps < 1:
        parser.error("Model dimensions, evaluation samples and predictor steps must be positive")
    try:
        config = TrainingConfig(**{name: getattr(args, name) for name in asdict(defaults)})
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_data = LOBSTERLevel10Dataset(train=True)
    loader = DataLoader(WindowDataset(train_data.features, config.sequence_length),
                        batch_size=config.batch_size, shuffle=True)
    model = TimeGAN(latent_dims=args.latent_dims, noise_dims=args.noise_dims).to(args.device)
    history = TimeGANTrainer(model, config).fit(loader)
    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output / "model.pt", model, config, train_data)
    pd.DataFrame(history).to_csv(args.output / "losses.csv", index=False)
    if not args.skip_evaluation:
        validation_data = LOBSTERLevel10Dataset(train=False)
        report, generated = evaluate_timegan(model, train_data, validation_data,
                                             config.sequence_length, args.evaluation_samples,
                                             args.predictor_steps, args.seed)
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        np.save(args.output / "generated_features.npy", generated.numpy())
        train_data.feature_sequence_to_df(generated[0]).to_csv(
            args.output / "sample_order_book.csv", index=False)
        scores = report["predictive_mae_standardized"]
        logger.info("Held-out reconstruction MSE=%.5f; predictive MAE: real=%.5f, synthetic=%.5f",
                    report["validation_losses"]["reconstruction_mse"],
                    scores["train_real_test_real"], scores["train_synthetic_test_real"])
    logger.info("Saved model and results to %s", args.output)


if __name__ == "__main__":
    main()
