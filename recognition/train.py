"""Direct-recurrence TimeGAN: reconstruction, transition, then joint training.

Objectives follow TimeGAN equations (7)-(11), using non-saturating BCE for G:
https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf
There is no static branch or separate supervisor network. Optional midpoint
log-return context and moment losses extend the joint generator objective.
"""

from dataclasses import asdict, dataclass
from itertools import chain
import logging
import math
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
    discriminator_learning_rate: float | None = None
    generator_updates: int = 2
    discriminator_loss_threshold: float | None = 0.15
    gradient_diagnostic_interval: int = 500
    context_loss_weight: float = 0.0
    context_horizon: int = 16
    noise_kind: str = "wiener_increments"
    price_moment_weight: float = 0.0
    conditional_drift_weight: float = 0.0

    def __post_init__(self):
        if self.sequence_length < 2 or self.batch_size < 1:
            raise ValueError("Training needs sequence_length >= 2 and batch_size >= 1")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or min(self.lam, self.eta) < 0:
            raise ValueError("Learning rate/clipping must be positive and loss weights nonnegative")
        if min(self.autoencoder_steps, self.transition_steps, self.joint_steps) < 0:
            raise ValueError("Training step budgets must be nonnegative")
        if (self.discriminator_learning_rate is not None
                and (not math.isfinite(self.discriminator_learning_rate)
                     or self.discriminator_learning_rate <= 0)):
            raise ValueError("Discriminator learning rate must be finite and positive")
        if not isinstance(self.generator_updates, int) or self.generator_updates < 1:
            raise ValueError("Generator updates must be a positive integer")
        if (self.discriminator_loss_threshold is not None
                and (not math.isfinite(self.discriminator_loss_threshold)
                     or self.discriminator_loss_threshold < 0)):
            raise ValueError("Discriminator threshold must be finite and nonnegative, or None")
        if (not isinstance(self.gradient_diagnostic_interval, int)
                or self.gradient_diagnostic_interval < 0):
            raise ValueError("Gradient diagnostic interval must be a nonnegative integer")
        if not math.isfinite(self.context_loss_weight) or self.context_loss_weight < 0:
            raise ValueError("Context loss weight must be finite and nonnegative")
        if not isinstance(self.context_horizon, int) or self.context_horizon < 1:
            raise ValueError("Context horizon must be a positive integer")
        if ((self.context_loss_weight > 0 or self.conditional_drift_weight > 0)
                and self.context_horizon >= self.sequence_length):
            raise ValueError("Context horizon must leave at least one real context event")
        if self.noise_kind not in ("wiener_paths", "wiener_increments"):
            raise ValueError("Noise kind must be wiener_paths or wiener_increments")
        if not math.isfinite(self.price_moment_weight) or self.price_moment_weight < 0:
            raise ValueError("Price moment weight must be finite and nonnegative")
        if not math.isfinite(self.conditional_drift_weight) or self.conditional_drift_weight < 0:
            raise ValueError("Conditional drift weight must be finite and nonnegative")


def supervised_loss(generator: nn.Module, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(generator.teacher_forced(latents, noise), latents[:, 1:])


def context_price_loss(model: TimeGAN, x: torch.Tensor, noise: torch.Tensor,
                       horizon: int) -> torch.Tensor:
    """Decoded price-path error after a real prefix, using generated feedback.

    Feature zero is standardized midpoint log return. Encoding only the prefix
    prevents future targets from entering the conditioning state. Reusing the
    start of the Wiener path resets its clock without changing increment variance.
    """
    if not 1 <= horizon < x.shape[1]:
        raise ValueError("The continuation must leave a nonempty real prefix")
    split = x.shape[1] - horizon
    with torch.no_grad():
        context = model.embedder(x[:, :split])[:, -1]
    continuation = model.generator.continue_from(context, noise[:, 1:horizon + 1])
    # R's parameters stay frozen in a G update, but gradients pass through R to G.
    predicted_returns = model.decoder(continuation)[..., 0]
    errors = predicted_returns - x[:, split:, 0]
    # Cumulative log-return errors supervise the price path. Divide by sqrt(t)
    # so later horizons do not dominate solely through accumulated variance.
    time = torch.arange(1, horizon + 1, device=x.device, dtype=x.dtype)
    errors = errors.cumsum(dim=1) / time.sqrt()
    return errors.square().mean()


def conditional_drift_loss(model: TimeGAN, x: torch.Tensor, noise: torch.Tensor,
                           horizon: int, zero_return: float) -> torch.Tensor:
    """Regularize expected cumulative log returns toward a zero-drift prior.

    Paired Gaussian innovations (+z, -z) reduce Monte Carlo error in the mean.
    The zero target is expressed in training-standardized units. This prior
    makes a lognormal forecast's median stay near its observed starting price.
    """
    if not 1 <= horizon < x.shape[1]:
        raise ValueError("The continuation must leave a nonempty real prefix")
    with torch.no_grad():
        context = model.embedder(x[:, :-horizon])[:, -1]
    innovations = noise[:, 1:horizon + 1]
    paired = torch.stack((innovations, -innovations), dim=1).flatten(0, 1)
    future = model.generator.continue_from(context.repeat_interleave(2, dim=0), paired)
    returns = model.decoder(future)[..., 0].reshape(len(x), 2, horizon)
    expected = returns.mean(dim=1) - zero_return
    time = torch.arange(1, horizon + 1, device=x.device, dtype=x.dtype)
    return (expected.cumsum(dim=1) / time.sqrt()).square().mean()


class TimeGANTrainer:
    def __init__(self, model: TimeGAN, config: TrainingConfig = TrainingConfig(),
                 *, return_mean: float = 0.0, return_std: float = 1.0):
        if not math.isfinite(return_mean) or not math.isfinite(return_std) or return_std <= 0:
            raise ValueError("Return normalization must have finite mean and positive SD")
        self.model = model
        self.config = config
        self.zero_return = -return_mean / return_std
        self.joint_cycles = 0
        self.update_counts = dict(generator=0, embedding=0, discriminator=0, discriminator_skipped=0)
        model.noise_kind = config.noise_kind
        model.context_horizon = config.context_horizon
        model.sequence_length = config.sequence_length
        self.opt_ER = torch.optim.Adam(chain(model.embedder.parameters(), model.decoder.parameters()),
                                       lr=config.learning_rate)
        self.opt_G = torch.optim.Adam(model.generator.parameters(), lr=config.learning_rate)
        self.opt_D = torch.optim.Adam(model.discriminator.parameters(),
                                     lr=(config.discriminator_learning_rate
                                         if config.discriminator_learning_rate is not None
                                         else config.learning_rate))

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
        paths, seed = generate_wiener_paths(
            x.shape[0], x.shape[1], self.model.generator.noise_dims,
            device=x.device, dtype=x.dtype)
        if self.config.noise_kind == "wiener_increments":
            paths = torch.cat((paths[:, :1], paths[:, 1:] - paths[:, :-1]), dim=1)
        return paths, seed

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
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        loss = real_loss + fake_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite discriminator loss")
        threshold = self.config.discriminator_loss_threshold
        update = threshold is None or loss.item() > threshold
        # Reuse this graph and noise for the gate and update. A skip must not
        # advance Adam's moments or step counter. Our BCE has two terms; the
        # authors' separate-supervisor implementation has a third fake term.
        if update:
            self._step(self.opt_D, loss)
        return {"discriminator": loss.item(), "d_real_bce": real_loss.item(),
                "d_fake_bce": fake_loss.item(),
                "d_real_accuracy": (real_logits >= 0).float().mean().item(),
                "d_fake_accuracy": (fake_logits < 0).float().mean().item(),
                "d_updates": int(update), "d_skipped": int(not update)}

    def _gradient_diagnostics(self, components: dict[str, torch.Tensor]) -> dict[str, float]:
        """Measure weighted gradients on the update graph, before clipping.

        autograd.grad leaves parameter .grad and the RNG untouched. Zero-fill
        unused initializer parameters so component vectors share coordinates.
        """
        parameters = tuple(self.model.generator.parameters())
        vectors = {}
        for name, component in components.items():
            gradients = (torch.autograd.grad(component, parameters, retain_graph=True,
                                            allow_unused=True) if component.requires_grad
                         else [None] * len(parameters))
            vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                       for p, g in zip(parameters, gradients)])
        reference = vectors["adversarial"]
        reference_norm = reference.norm()
        metrics = {"g_total_gradient_norm": sum(vectors.values()).norm().item()}
        for name, vector in vectors.items():
            norm = vector.norm()
            metrics[f"gradient_norm_{name}"] = norm.item()
            # A zero component has no direction; report cosine zero by convention.
            metrics[f"gradient_cosine_{name}_adversarial"] = (
                (vector @ reference) / (norm * reference_norm).clamp_min(1e-12)).item()
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Non-finite generator gradient diagnostic")
        return metrics

    def generator_step(self, x: torch.Tensor, *, diagnose: bool = False) -> dict[str, float]:
        self.train_only(self.model.generator)
        noise, seed = self._noise(x)
        with torch.no_grad():
            real = self.model.embedder(x)
        fake = self.model.generator.sample(noise, seed)
        # D's parameters are frozen, but D(fake) must remain differentiable.
        logits = self.model.discriminator(fake)
        adversarial = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        supervised = supervised_loss(self.model.generator, real, noise)
        components = {"adversarial": adversarial, "g_supervised": supervised}
        if self.config.context_loss_weight > 0:
            context = context_price_loss(self.model, x, noise, self.config.context_horizon)
            components["g_context_price"] = context
        if self.config.price_moment_weight > 0:
            returns = self.model.decoder(fake)[..., 0]
            target = x[..., 0]
            moments = ((returns.mean() - target.mean()).square()
                       + (returns.std(correction=0) - target.std(correction=0)).square())
            components["g_price_moments"] = moments
        if self.config.conditional_drift_weight > 0:
            drift = conditional_drift_loss(self.model, x, noise, self.config.context_horizon,
                                           self.zero_return)
            components["g_conditional_drift"] = drift
        weights = {"adversarial": 1.0, "g_supervised": self.config.eta,
                   "g_context_price": self.config.context_loss_weight,
                   "g_price_moments": self.config.price_moment_weight,
                   "g_conditional_drift": self.config.conditional_drift_weight}
        weighted = {name: components.get(name, x.new_zeros(())) * weight
                    for name, weight in weights.items()}
        loss = sum(weighted.values())
        metrics = {name: components.get(name, x.new_zeros(())).item() for name in weights}
        metrics.update({f"weighted_{name}": value.item() for name, value in weighted.items()})
        if diagnose:
            metrics.update(self._gradient_diagnostics(weighted))
        self._step(self.opt_G, loss)
        return {"generator": loss.item(), **metrics}

    def embedding_step(self, x: torch.Tensor) -> dict[str, float]:
        self.train_only(self.model.embedder, self.model.decoder)
        noise, _ = self._noise(x)
        latents = self.model.embedder(x)
        decoded = self.model.decoder(latents)
        reconstruction = F.mse_loss(decoded, x)
        # Both the target and previous latent stay attached to E through frozen G.
        supervised = supervised_loss(self.model.generator, latents, noise)
        loss = reconstruction + self.config.lam * supervised
        self._step(self.opt_ER, loss)
        return {"embedding": loss.item(), "reconstruction": reconstruction.item(),
                "return_reconstruction": F.mse_loss(decoded[..., 0], x[..., 0]).item(),
                "e_supervised": supervised.item(),
                "weighted_e_supervised": (self.config.lam * supervised).item()}

    def joint_step(self, x: torch.Tensor) -> dict[str, float]:
        # Authors' joint schedule: repeat (G, E/R), then conditionally update D.
        self.joint_cycles += 1
        interval = self.config.gradient_diagnostic_interval
        diagnose = interval > 0 and self.joint_cycles % interval == 0
        rows = []
        for update in range(self.config.generator_updates):
            rows.append(self.generator_step(x, diagnose=diagnose and update == 0)
                        | self.embedding_step(x))
        metrics = {name: sum(row[name] for row in rows if name in row)
                         / sum(name in row for row in rows)
                   for name in dict.fromkeys(name for row in rows for name in row)}
        metrics.update(self.discriminator_step(x))
        metrics.update(g_updates=len(rows), e_updates=len(rows))
        self.update_counts["generator"] += len(rows)
        self.update_counts["embedding"] += len(rows)
        self.update_counts["discriminator"] += metrics["d_updates"]
        self.update_counts["discriminator_skipped"] += metrics["d_skipped"]
        return metrics

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
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.model_config,
        "training_config": asdict(config),
        "feature_mean": torch.as_tensor(data.feature_mean),
        "feature_std": torch.as_tensor(data.feature_std),
        "feature_names": list(data.feature_names),
        "price_representation": "log_return",  # Checkpoint format marker, not a training option.
        "reference_midpoint": data.reference_midpoint,
        "data_config": data.data_config,
        "split_time_seconds": data.split_time_seconds,
    }
    torch.save(checkpoint, path)


def main():
    import argparse
    import json
    import numpy as np
    import pandas as pd

    from evaluation import evaluate_timegan

    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train a recurrent event-time TimeGAN on AMZN")
    parser.add_argument("--output", type=Path, default=Path("runs/log_return/timegan"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-interval-seconds", type=float, default=None,
                        help="Use previous-tick clock snapshots; omit for original event rows")
    for name in ("sequence_length", "batch_size", "autoencoder_steps", "transition_steps", "joint_steps",
                 "context_horizon", "gradient_diagnostic_interval"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "gradient_clip", "lam", "eta", "context_loss_weight",
                 "price_moment_weight", "conditional_drift_weight"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--latent-dims", type=int, default=128)
    parser.add_argument("--noise-dims", type=int, default=128)
    parser.add_argument("--discriminator-learning-rate", type=float, default=None)
    parser.add_argument("--generator-updates", type=int, default=defaults.generator_updates)
    parser.add_argument("--discriminator-loss-threshold", type=float,
                        default=defaults.discriminator_loss_threshold,
                        help="Update D only above this summed BCE; zero effectively disables the gate")
    parser.add_argument("--noise-kind", choices=("wiener_paths", "wiener_increments"),
                        default=defaults.noise_kind)
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
    train_data = LOBSTERLevel10Dataset(train=True, sampling_interval_seconds=args.sampling_interval_seconds)
    loader = DataLoader(WindowDataset(train_data.features, config.sequence_length),
                        batch_size=config.batch_size, shuffle=True)
    model = TimeGAN(latent_dims=args.latent_dims, noise_dims=args.noise_dims).to(args.device)
    history = TimeGANTrainer(model, config, return_mean=train_data.feature_mean[0],
                             return_std=train_data.feature_std[0]).fit(loader)
    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output / "model.pt", model, config, train_data)
    pd.DataFrame(history).to_csv(args.output / "losses.csv", index=False)
    if not args.skip_evaluation:
        validation_data = LOBSTERLevel10Dataset(train=False, sampling_interval_seconds=args.sampling_interval_seconds)
        report, generated = evaluate_timegan(model, train_data, validation_data,
                                             sequence_length=config.sequence_length,
                                             samples=args.evaluation_samples,
                                             predictor_steps=args.predictor_steps, seed=args.seed)
        (args.output / "evaluation.json").write_text(
            json.dumps(asdict(report), default=torch.Tensor.tolist, indent=2, allow_nan=False) + "\n")
        np.save(args.output / "generated_features.npy", generated.numpy())
        train_data.feature_sequence_to_df(generated[0]).to_csv(
            args.output / "sample_order_book.csv", index=False)
        scores = report.predictive_mae_standardized
        logger.info("Held-out reconstruction MSE=%.5f; predictive MAE: real=%.5f, synthetic=%.5f",
                    report.validation_losses.reconstruction_mse,
                    scores.train_real_test_real, scores.train_synthetic_test_real)
    logger.info("Saved model and results to %s", args.output)


if __name__ == "__main__":
    main()
