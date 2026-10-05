"""Controlled TimeGAN training experiments for the initial midpoint jump.

Each run starts from seed 0, uses a separately seeded data sampler, and keeps
the original architecture, Wiener noise, losses, and stage budgets fixed.
"""

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import LOBSTERLevel10Dataset, WindowDataset
from modules import TimeGAN
from predict import generate_wiener_paths
from train import TimeGANTrainer, TrainingConfig, save_checkpoint


EXPERIMENTS = {
    "baseline": (3e-4, 1),
    "discriminator_1e-4": (1e-4, 1),
    "discriminator_3e-5": (3e-5, 1),
    "generator_twice": (3e-4, 2),
}
EVALUATION_SEED = 0


def midpoint(books):
    return ((books["ASKp1"] + books["BIDp1"]) / 20_000).to_numpy()


def noise_paths(batch, horizon, model, sequence_length):
    """Keep training increment variance, including in long exploratory rollouts."""
    parameter = next(model.parameters())
    rng = torch.Generator(device=parameter.device).manual_seed(EVALUATION_SEED)
    paths, _ = generate_wiener_paths(
        batch, horizon + 1, model.generator.noise_dims,
        device=parameter.device, dtype=parameter.dtype, rng=rng,
    )
    paths = paths * (horizon / (sequence_length - 1)) ** 0.5
    if getattr(model, "noise_kind", "wiener_paths") == "wiener_increments":
        paths = torch.cat((paths[:, :1], paths[:, 1:] - paths[:, :-1]), dim=1)
    return paths


@torch.inference_mode()
def complete(model, contexts, paths, chunk_size=1024):
    """Continue an observed prefix using only generated history thereafter."""
    if hasattr(model, "price_volatility"):
        from price_dynamics import innovation_completion
        return innovation_completion(model, model.price_volatility, contexts, paths,
                                      model.price_calibration)
    if getattr(model, "price_representation", "level") == "return":
        # Repeat the trained conditional task using only generated history.
        # No validation observations or price offsets are inserted mid-forecast.
        block = model.context_horizon
        context_length = contexts.shape[1] - block
        if context_length < 1:
            raise ValueError("The continuation needs a nonempty context")
        history = contexts[:, -context_length:]
        outputs = []
        for start in range(0, paths.shape[1] - 1, block):
            noise = paths[:, start + 1:start + block + 1]
            latent = model.embedder(history)[:, -1]
            future = model.decoder(model.generator.continue_from(latent, noise))
            outputs.append(future.cpu())
            history = torch.cat((history, future), dim=1)[:, -context_length:]
        return torch.cat(outputs, dim=1)
    h = model.embedder(contexts)[:, -1]
    horizon = paths.shape[1] - 1
    features = torch.empty(len(contexts), horizon, model.model_config["feature_dims"])
    buffer = h.new_empty(len(h), min(chunk_size, horizon), h.shape[-1])
    for step in range(1, horizon + 1):
        h = model.generator.cell(paths[:, step], h)
        offset = (step - 1) % chunk_size
        buffer[:, offset] = h
        if step % chunk_size == 0 or step == horizon:
            width = offset + 1
            features[:, step - width:step] = model.decoder(buffer[:, :width]).cpu()
    return features


@torch.inference_mode()
def diagnostics(model, training, validation, config):
    """Separate noise-driven spread from deterministic drift at eight contexts.

Later validation contexts use only observations preceding their own forecast.
No validation events are used for training, and normalization stays fixed.
"""
    device = next(model.parameters()).device
    length, horizon, samples = config.sequence_length, 128, 32
    features = torch.cat((training.features, validation.features))
    prices = np.r_[midpoint(training.books), midpoint(validation.books)]
    ends = len(training) + np.linspace(0, len(validation) - horizon, 8).round().astype(int)
    contexts = torch.stack([features[end - length:end] for end in ends]).to(device)
    initial = prices[ends - 1]
    observed = np.stack([prices[end:end + horizon] for end in ends])
    paired_contexts = contexts.repeat_interleave(samples, dim=0)
    paths = noise_paths(len(paired_contexts), horizon, model, length)
    fake = complete(model, paired_contexts, paths)

    def project(sequences, anchors):
        return np.stack([midpoint(training.feature_sequence_to_df(
            sequence, initial_midpoint=anchor * 10_000))
            for sequence, anchor in zip(sequences, anchors)])

    generated = project(fake, np.repeat(initial, samples)).reshape(len(contexts), samples, horizon)
    zero = project(complete(model, contexts, paths[:len(contexts)] * 0), initial)
    rises = generated[:, :, 63] - initial[:, None]
    endpoint_errors = generated[:, :, 63] - observed[:, 63, None]
    context_rows = pd.DataFrame({
        "context_event": ends,
        "starting_price_dollars": initial,
        "observed_64_event_change_dollars": observed[:, 63] - initial,
        "generated_64_event_change_mean_dollars": rises.mean(axis=1),
        "generated_64_event_change_std_dollars": rises.std(axis=1),
        "zero_noise_64_event_change_dollars": zero[:, 63] - initial,
    })
    reconstruction = F.mse_loss(model(contexts), contexts).item()
    # The first row is the exact training/validation boundary from the notebook.
    metrics = {
        "boundary_first_event_change_dollars": float(generated[0, :, 0].mean() - initial[0]),
        "boundary_16_event_change_dollars": float(generated[0, :, 15].mean() - initial[0]),
        "boundary_16_event_std_dollars": float(generated[0, :, 15].std()),
        "context_mean_abs_16_event_error_dollars": float(np.abs(
            generated[:, :, 15].mean(axis=1) - observed[:, 15]).mean()),
        "boundary_64_event_change_dollars": float(rises[0].mean()),
        "boundary_64_event_std_dollars": float(rises[0].std()),
        "boundary_zero_noise_64_event_change_dollars": float(zero[0, 63] - initial[0]),
        "boundary_observed_64_event_change_dollars": float(observed[0, 63] - initial[0]),
        "context_mean_abs_64_event_error_dollars": float(np.abs(endpoint_errors.mean(axis=1)).mean()),
        "context_mean_64_event_error_dollars": float(endpoint_errors.mean()),
        "context_mean_abs_zero_noise_64_event_change_dollars": float(np.abs(zero[:, 63] - initial).mean()),
        "context_reconstruction_mse": reconstruction,
        "boundary_change_std_dollars": float(np.diff(
            np.column_stack((np.full(samples, initial[0]), generated[0, :, :64])), axis=1).std()),
        "boundary_observed_change_std_dollars": float(np.diff(np.r_[initial[0], observed[0, :64]]).std()),
    }
    curves = {
        "starting_price_dollars": float(initial[0]),
        "observed": observed[0].tolist(),
        "generated_mean": generated[0].mean(axis=0).tolist(),
        "generated_std": generated[0].std(axis=0).tolist(),
        "generated_examples": generated[0, :3].tolist(),
        "zero_noise": zero[0].tolist(),
    }
    return metrics, curves, context_rows


class ExperimentTrainer(TimeGANTrainer):
    """Evaluate snapshots without advancing the training random-number stream."""

    def __init__(self, model, config, training, validation, directory):
        super().__init__(model, config)
        self.training = training
        self.validation = validation
        self.directory = directory
        self.joint_iteration = 0
        self.snapshots = []
        self.snapshot_steps = {max(1, config.joint_steps // 5),
                               max(1, config.joint_steps // 2), config.joint_steps}

    def joint_step(self, x):
        metrics = super().joint_step(x)
        self.joint_iteration += 1
        if self.joint_iteration in self.snapshot_steps:
            step = self.joint_iteration
            was_training = self.model.training
            self.model.eval()
            with torch.random.fork_rng(devices=[]):
                measured, curves, contexts = diagnostics(
                    self.model, self.training, self.validation, self.config)
            self.model.train(was_training)
            self.snapshots.append({"joint_step": step, **measured})
            (self.directory / f"diagnostics_step_{step}.json").write_text(
                json.dumps({"metrics": measured, "curves": curves}, indent=2, allow_nan=False) + "\n")
            contexts.to_csv(self.directory / f"contexts_step_{step}.csv", index=False)
            save_checkpoint(self.directory / f"model_step_{step}.pt", self.model, self.config, self.training)
            pd.DataFrame(self.snapshots).to_csv(self.directory / "snapshots.csv", index=False)
            logging.info("Boundary jump at joint step %d: noise=%.4f dollars; zero-noise=%.4f dollars",
                         step, measured["boundary_64_event_change_dollars"],
                         measured["boundary_zero_noise_64_event_change_dollars"])
        return metrics


def run_experiment(experiment, config, output, seed=0, threads=4):
    """Shared fresh-training protocol for loss and optimizer comparisons."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logging.basicConfig(level=logging.INFO, format=f"{experiment}: %(message)s")
    directory = output / f"{experiment}_seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "model.pt").exists():
        raise FileExistsError(f"A completed run already exists at {directory}; choose another output.")
    training = LOBSTERLevel10Dataset(True, price_representation=config.price_representation)
    validation = LOBSTERLevel10Dataset(False, price_representation=config.price_representation)
    # Independent sampler RNG keeps real batches identical even with extra G updates.
    loader = DataLoader(WindowDataset(training.features, config.sequence_length),
                        batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = TimeGAN()
    trainer = ExperimentTrainer(model, config, training, validation, directory)
    started = time.monotonic()
    history = trainer.fit(loader, log_every=500)
    elapsed = time.monotonic() - started
    save_checkpoint(directory / "model.pt", model, config, training)
    pd.DataFrame(history).to_csv(directory / "losses.csv", index=False)
    measured, curves, contexts = diagnostics(model, training, validation, config)
    (directory / "diagnostics_final.json").write_text(
        json.dumps({"metrics": measured, "curves": curves}, indent=2, allow_nan=False) + "\n")
    contexts.to_csv(directory / "contexts_final.csv", index=False)
    metadata = {
        "experiment": experiment, "seed": seed, "threads": threads,
        "training_config": asdict(config), "model_config": model.model_config,
        "training_seconds": elapsed,
        "joint_generator_updates": config.joint_steps * config.generator_updates,
        "joint_discriminator_updates": config.joint_steps,
        "joint_embedding_updates": config.joint_steps,
        "evaluation_contexts": 8, "evaluation_samples_per_context": 32,
        "evaluation_seed": EVALUATION_SEED,
    }
    (directory / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Keep the original full-validation chart and the deterministic comparison.
    torch.set_num_threads(1)
    horizon = len(validation)
    paths = noise_paths(3, horizon, model, config.sequence_length)
    context = training.features[-config.sequence_length:][None]
    noisy = complete(model, context.repeat(3, 1, 1), paths)
    zero = complete(model, context, paths[:1] * 0)
    anchor = midpoint(training.books)[-1] * 10_000
    full_curves = np.stack([midpoint(training.feature_sequence_to_df(
        sequence, initial_midpoint=anchor)) for sequence in noisy])
    zero_curve = midpoint(training.feature_sequence_to_df(zero[0], initial_midpoint=anchor))
    np.savez_compressed(directory / "completion_midpoints.npz", noisy=full_curves, zero=zero_curve)
    logging.info("Completed in %.1f minutes; results saved to %s", elapsed / 60, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--output", type=Path, default=Path("runs/training_balance"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--autoencoder-steps", type=int, default=1000)
    parser.add_argument("--transition-steps", type=int, default=1000)
    parser.add_argument("--joint-steps", type=int, default=5000)
    args = parser.parse_args()
    d_rate, updates = EXPERIMENTS[args.experiment]
    config = TrainingConfig(
        discriminator_learning_rate=d_rate, generator_updates=updates,
        autoencoder_steps=args.autoencoder_steps,
        transition_steps=args.transition_steps, joint_steps=args.joint_steps,
    )
    run_experiment(args.experiment, config, args.output, args.seed, args.threads)


if __name__ == "__main__":
    main()
