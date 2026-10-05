"""Build paired price-level, noise-reset, and decoder-return diagnostics.

Only the noise clock changes in the reset control; the original model weights
and real starting context are retained. Return-model forecasts use generated
history after the initial real prefix. No validation future is supplied.
"""

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from dataset import LOBSTERLevel10Dataset
from predict import load_checkpoint
from train import TrainingConfig
import training_balance_experiment as experiment


def main():
    project_dir = Path(__file__).resolve().parent
    root = project_dir / "runs/return_price"
    original = project_dir / "runs/context_loss/context_weight_100_horizon_16_seed_0"
    for name in ("previous_levels", "reset_noise"):
        directory = root / f"{name}_seed_0"
        directory.mkdir(exist_ok=True)
        for filename in ("model.pt", "run.json", "losses.csv"):
            shutil.copy2(original / filename, directory / filename)
        if name == "previous_levels":
            for filename in ("completion_midpoints.npz", "snapshots.csv"):
                shutil.copy2(original / filename, directory / filename)

    def reset_clock(paths, block=16):
        steps = torch.arange(paths.shape[1], device=paths.device)
        starts = ((steps - 1).clamp_min(0) // block) * block
        return paths - paths[:, starts]

    original_complete = experiment.complete

    def reset_complete(model, contexts, paths):
        return original_complete(model, contexts, reset_clock(paths))

    names = ["previous_levels", "reset_noise",
             "returns_context_1_moments_10", "returns_context_10_moments_10"]
    torch.set_num_threads(1)
    for name in names:
        directory = root / f"{name}_seed_0"
        model, checkpoint = load_checkpoint(directory / "model.pt")
        config = TrainingConfig(**checkpoint["training_config"])
        training = LOBSTERLevel10Dataset(True, price_representation=config.price_representation)
        validation = LOBSTERLevel10Dataset(False, price_representation=config.price_representation)
        method = reset_complete if name == "reset_noise" else original_complete
        experiment.complete = method
        metrics, curves, contexts = experiment.diagnostics(model, training, validation, config)
        anchor = experiment.midpoint(training.books)[-1]
        context = training.features[-config.sequence_length:][None]

        def project(features):
            return np.stack([experiment.midpoint(training.feature_sequence_to_df(
                sequence, initial_midpoint=anchor * 10000)) for sequence in features])

        paths = experiment.noise_paths(32, 1024, model, config.sequence_length)
        prices = project(method(model, context.repeat(32, 1, 1), paths))
        zero = project(method(model, context, paths[:1] * 0))[0]
        observed = experiment.midpoint(validation.books)[:1024]
        metrics.update(
            boundary_1024_event_change_dollars=float(prices[:, -1].mean() - anchor),
            boundary_1024_event_std_dollars=float(prices[:, -1].std()),
            boundary_zero_noise_1024_event_change_dollars=float(zero[-1] - anchor),
            boundary_peak_mean_rise_1024_dollars=float(max(0, prices.mean(0).max() - anchor)),
            boundary_max_single_event_change_1024_dollars=float(np.abs(np.diff(
                np.column_stack((np.full(32, anchor), prices)), axis=1)).max()),
        )
        (directory / "long_boundary.json").write_text(json.dumps({
            "starting_price_dollars": anchor, "observed": observed.tolist(),
            "generated_mean": prices.mean(0).tolist(), "generated_std": prices.std(0).tolist(),
            "zero_noise": zero.tolist(),
        }, indent=2, allow_nan=False))
        (directory / "diagnostics_final.json").write_text(json.dumps(
            {"metrics": metrics, "curves": curves}, indent=2, allow_nan=False))
        contexts.to_csv(directory / "contexts_final.csv", index=False)
        contexts.to_csv(directory / f"contexts_step_{config.joint_steps}.csv", index=False)
        if name == "reset_noise":
            pd.DataFrame([{"joint_step": config.joint_steps, **metrics}]).to_csv(
                directory / "snapshots.csv", index=False)
            paths = experiment.noise_paths(3, len(validation), model, config.sequence_length)
            full = project(method(model, context.repeat(3, 1, 1), paths))
            full_zero = project(method(model, context, paths[:1] * 0))[0]
            np.savez_compressed(directory / "completion_midpoints.npz", noisy=full, zero=full_zero)
        print(name, json.dumps({key: metrics[key] for key in (
            "boundary_first_event_change_dollars", "boundary_64_event_change_dollars",
            "boundary_1024_event_change_dollars", "boundary_zero_noise_1024_event_change_dollars",
            "boundary_64_event_std_dollars", "context_mean_abs_64_event_error_dollars",
        )}), flush=True)
    experiment.complete = original_complete


if __name__ == "__main__":
    main()
