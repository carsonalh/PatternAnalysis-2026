"""Refine G's loss balance with a completed TimeGAN representation held fixed.

Joint updates can move the decoder after G's drift penalty is evaluated. This
small-learning-rate pass lets G adapt to the final embedder and decoder while
retaining adversarial, latent, moment, and conditional drift objectives.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import LOBSTERLevel10Dataset, WindowDataset
from loss_weight_report import endpoint_distribution
from predict import load_checkpoint
from train import TimeGANTrainer, TrainingConfig, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--moment-weight", type=float, default=100)
    parser.add_argument("--drift-weight", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("Refinement needs a positive step count")
    if (args.output / "model.pt").exists():
        raise FileExistsError("Choose a new refinement directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    config = replace(TrainingConfig(**checkpoint["training_config"]),
                     learning_rate=args.learning_rate, price_moment_weight=args.moment_weight,
                     conditional_drift_weight=args.drift_weight,
                     autoencoder_steps=0, transition_steps=0, joint_steps=0)
    interval = checkpoint.get("data_config", {}).get("sampling_interval_seconds")
    training = LOBSTERLevel10Dataset(True, sampling_interval_seconds=interval)
    validation = LOBSTERLevel10Dataset(False, sampling_interval_seconds=interval)
    np.testing.assert_allclose(checkpoint["feature_mean"].cpu(), training.feature_mean)
    np.testing.assert_allclose(checkpoint["feature_std"].cpu(), training.feature_std)
    trainer = TimeGANTrainer(model, config, return_mean=training.feature_mean[0],
                             return_std=training.feature_std[0])
    loader = DataLoader(WindowDataset(training.features, config.sequence_length),
                        batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    batches, history = iter(loader), []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        metrics = trainer.generator_step(batch.to(args.device))
        history.append({"step": step, **metrics})
        if step % 500 == 0 or step == args.steps:
            model.eval()
            measured, _, _ = endpoint_distribution(model, training, validation, samples=128, seed=0)
            print(json.dumps({"step": step, **measured}), flush=True)
            save_checkpoint(args.output / f"model_step_{step}.pt", model, config, training)
    model.eval()
    model.requires_grad_(True)
    for name, value in model.state_dict().items():
        if not name.startswith("generator.") and not torch.equal(value, checkpoint["model_state"][name]):
            raise AssertionError(f"Refinement changed a fixed parameter: {name}")
    save_checkpoint(args.output / "model.pt", model, config, training)
    pd.DataFrame(history).to_csv(args.output / "losses.csv", index=False)
    metadata = {"parent_checkpoint": str(args.checkpoint.resolve()),
                "parent_training_config": checkpoint["training_config"],
                "training_config": asdict(config), "refinement_steps": args.steps,
                "refinement_batch_size": 128, "refinement_seed": args.seed,
                "fixed_networks": ["embedder", "decoder", "discriminator"],
                "fixed_networks_verified_unchanged": True,
                "training_seconds": time.monotonic() - started,
                "data_config": training.data_config,
                "validation_used_for_model_comparison": True}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
