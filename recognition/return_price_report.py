"""Report decoder log-return forecasts and their integrated price paths.

Both context-weight experiments use the same fixed dataset. Forecasts start
from the observed training prefix and use only generated history thereafter.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import LOBSTERLevel10Dataset
from predict import load_checkpoint
from train import TrainingConfig
from training_balance_experiment import complete, diagnostics, midpoint, noise_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/log_return"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    training, validation = LOBSTERLevel10Dataset(True), LOBSTERLevel10Dataset(False)
    anchor = midpoint(training.books)[-1]
    for weight in (1, 10):
        directory = args.root / f"returns_context_{weight}_moments_10_seed_0"
        model, checkpoint = load_checkpoint(directory / "model.pt")
        config = TrainingConfig(**checkpoint["training_config"])
        np.testing.assert_allclose(checkpoint["feature_mean"].cpu(), training.feature_mean)
        np.testing.assert_allclose(checkpoint["feature_std"].cpu(), training.feature_std)
        metrics, curves, contexts = diagnostics(model, training, validation, config)
        context = training.features[-config.sequence_length:][None]

        def project(features):
            return np.stack([midpoint(training.feature_sequence_to_df(
                sequence, initial_midpoint=anchor * 10_000)) for sequence in features])

        def log_returns(features):
            return features[..., 0].numpy() * training.feature_std[0] + training.feature_mean[0]

        paths = noise_paths(32, 1024, model, config.sequence_length)
        features = complete(model, context.repeat(32, 1, 1), paths)
        zero_features = complete(model, context, paths[:1] * 0)
        prices, zero = project(features), project(zero_features)[0]
        predicted_returns = log_returns(features)
        metrics.update(
            boundary_1024_first_log_return_mean=float(predicted_returns[:, 0].mean()),
            boundary_1024_first_log_return_std=float(predicted_returns[:, 0].std()),
            boundary_1024_event_change_dollars=float(prices[:, -1].mean() - anchor),
            boundary_1024_event_std_dollars=float(prices[:, -1].std()),
            boundary_zero_noise_1024_event_change_dollars=float(zero[-1] - anchor),
        )
        (directory / "long_boundary.json").write_text(json.dumps({
            "starting_price_dollars": anchor,
            "observed": midpoint(validation.books)[:1024].tolist(),
            "generated_mean": prices.mean(0).tolist(), "generated_std": prices.std(0).tolist(),
            "zero_noise": zero.tolist(),
            "generated_log_return_mean": predicted_returns.mean(0).tolist(),
            "generated_log_return_std": predicted_returns.std(0).tolist(),
        }, indent=2, allow_nan=False) + "\n")
        (directory / "diagnostics_final.json").write_text(json.dumps(
            {"metrics": metrics, "curves": curves}, indent=2, allow_nan=False) + "\n")
        contexts.to_csv(directory / "contexts_final.csv", index=False)
        paths = noise_paths(3, len(validation), model, config.sequence_length)
        full = complete(model, context.repeat(3, 1, 1), paths)
        full_zero = complete(model, context, paths[:1] * 0)
        np.savez_compressed(directory / "completion_log_returns.npz",
                            noisy=log_returns(full), zero=log_returns(full_zero)[0])
        np.savez_compressed(directory / "completion_midpoints.npz",
                            noisy=project(full), zero=project(full_zero)[0])
        print(f"Context weight {weight}: first predicted log return "
              f"{metrics['boundary_first_log_return_mean']:+.3e}; "
              f"mean 1,024-event price change ${metrics['boundary_1024_event_change_dollars']:+.4f}",
              flush=True)


if __name__ == "__main__":
    main()
