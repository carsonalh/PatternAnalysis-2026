"""Compare paired conditional rollouts and draw the notebook's three-path plots.

Volatility is measured before tick projection. Long-path bias and multi-context
dispersion are reported separately so a volatile but drifting model is visible.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import LOBSTERLevel10Dataset
from predict import load_checkpoint
from training_balance_experiment import complete, midpoint, noise_paths


@torch.inference_mode()
def endpoint_distribution(model, training, validation, samples=256, seed=0):
    """Forecast after the first validation snapshot, which is the known anchor.

    Later validation prices score the forecast only. Endpoints are integrated
    decoder returns with no recentering, scaling, or substituted price process.
    """
    if getattr(model, "noise_kind", None) != "wiener_increments":
        raise ValueError("Terminal comparison requires a model trained with Wiener increments")
    length = model.sequence_length
    context = torch.cat((training.features[-length + 1:], validation.features[:1]))[None]
    device = next(model.parameters()).device
    horizon = len(validation) - 1
    rng = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(samples, horizon, model.generator.noise_dims,
                        generator=rng, device=device) / math.sqrt(length - 1)
    noise = torch.cat((noise.new_zeros(samples, 1, noise.shape[-1]), noise), dim=1)
    features = complete(model, context.to(device).repeat(samples, 1, 1), noise)
    returns = features[..., 0].numpy() * training.feature_std[0] + training.feature_mean[0]
    anchor = midpoint(validation.books)[0]
    cumulative = np.cumsum(returns, axis=1)
    prices = anchor * np.exp(cumulative)
    return summarize_endpoints(training, validation, returns, prices, seed), returns, prices


def summarize_endpoints(training, validation, returns, prices, seed=None):
    """Score one or several pooled noise draws without modifying their outputs."""
    samples, horizon = returns.shape
    if samples < 32 or horizon != len(validation) - 1 or prices.shape != returns.shape:
        raise ValueError("Endpoint scoring needs >=32 complete futures after the first validation snapshot")
    if not np.isfinite(returns).all() or not np.isfinite(prices).all() or (prices <= 0).any():
        raise FloatingPointError("Generated futures must have finite returns and positive finite prices")
    anchor = midpoint(validation.books)[0]
    np.testing.assert_allclose(prices, anchor * np.exp(np.cumsum(returns, axis=1)), rtol=1e-6)
    terminal = prices[:, -1]
    log_endpoints = returns.sum(axis=1)
    sd = log_endpoints.std()
    standardized = (log_endpoints - log_endpoints.mean()) / max(sd, 1e-12)
    # Fit in log space: normal cumulative returns imply lognormal final prices.
    normal_cdf = np.array([NormalDist().cdf(value) for value in np.sort(standardized)])
    upper = np.arange(1, samples + 1) / samples
    lower = np.arange(samples) / samples
    ks = max(np.max(upper - normal_cdf), np.max(normal_cdf - lower))
    observed = validation.features[1:, 0].numpy() * training.feature_std[0] + training.feature_mean[0]
    training_prices = midpoint(training.books)
    training_changes = np.log(training_prices[horizon:] / training_prices[:-horizon])
    ordered = np.sort(terminal)
    median_margin = 1.96 * math.sqrt(samples) / 2
    median_low = max(0, int(samples / 2 - median_margin))
    median_high = min(samples - 1, int(samples / 2 + median_margin))
    metrics = {
        "anchor_dollars": float(anchor), "horizon_seconds": horizon,
        "samples": samples, "evaluation_seed": seed,
        "return_sd_ratio": float(returns.std(axis=1).mean() / observed.std()),
        "terminal_median_dollars": float(np.median(terminal)),
        "terminal_median_bias_dollars": float(np.median(terminal) - anchor),
        "median_ci95_low_dollars": float(ordered[median_low]),
        "median_ci95_high_dollars": float(ordered[median_high]),
        "terminal_sd_dollars": float(terminal.std()),
        "terminal_log_sd": float(sd),
        "terminal_sd_ratio_to_training_horizon": float(sd / training_changes.std()),
        "terminal_upward_share": float((terminal > anchor).mean()),
        "terminal_p05_dollars": float(np.quantile(terminal, 0.05)),
        "terminal_p95_dollars": float(np.quantile(terminal, 0.95)),
        "terminal_log_skew": float(np.mean(standardized ** 3)),
        "terminal_log_excess_kurtosis": float(np.mean(standardized ** 4) - 3),
        "terminal_log_normal_ks": float(ks),
        "return_lag1_correlation": float(np.corrcoef(returns[:, :-1].ravel(), returns[:, 1:].ravel())[0, 1]),
        "validation_lag1_correlation": float(np.corrcoef(observed[:-1], observed[1:])[0, 1]),
    }
    return metrics


def linear_r_squared(paths):
    time = np.arange(paths.shape[-1], dtype=float)
    time -= time.mean()
    centered = paths - paths.mean(axis=-1, keepdims=True)
    slopes = (centered * time).sum(axis=-1, keepdims=True) / (time @ time)
    residual = centered - slopes * time
    return 1 - np.sum(residual ** 2, axis=-1) / np.maximum(
        np.sum(centered ** 2, axis=-1), 1e-30)


@torch.inference_mode()
def context_metrics(model, training, validation, directory):
    """32 futures at each of eight fixed contexts; observed futures are scoring only."""
    horizon, samples, length = 1024, 32, model.sequence_length
    features = torch.cat((training.features, validation.features))
    prices = np.r_[midpoint(training.books), midpoint(validation.books)]
    ends = len(training) + np.linspace(0, len(validation) - horizon, 8).round().astype(int)
    contexts = torch.stack([features[end - length:end] for end in ends])
    device = next(model.parameters()).device
    paired = contexts.repeat_interleave(samples, dim=0).to(device)
    noise = noise_paths(len(paired), horizon, model, length)
    generated = complete(model, paired, noise)
    returns = (generated[..., 0].numpy() * training.feature_std[0]
               + training.feature_mean[0]).reshape(8, samples, horizon)
    observed = np.stack([features[end:end + horizon, 0].numpy() for end in ends])
    observed = observed * training.feature_std[0] + training.feature_mean[0]
    # Unrounded prices isolate learned return dynamics from book projection effects.
    anchors = prices[ends - 1]
    paths = anchors[:, None, None] * np.exp(returns.cumsum(axis=-1))
    changes = paths[..., -1] - anchors[:, None]
    context_rows = pd.DataFrame({
        "context_event": ends,
        "return_sd": returns.std(axis=-1).mean(axis=1),
        "observed_return_sd": observed.std(axis=-1),
        "endpoint_mean_change_dollars": changes.mean(axis=1),
        "endpoint_sd_dollars": changes.std(axis=1),
        "upward_path_share": (changes > 0).mean(axis=1),
        "drift_to_dispersion": np.abs(changes.mean(axis=1)) / np.maximum(changes.std(axis=1), 1e-12),
    })
    context_rows.to_csv(directory / "contexts_1024.csv", index=False)
    np.savez_compressed(directory / "contexts_1024.npz", context_events=ends,
                        returns=returns, unrounded_midpoints=paths, observed_returns=observed)
    return {
        "context_return_sd": float(returns.std(axis=-1).mean()),
        "context_return_sd_ratio": float(returns.std(axis=-1).mean() / observed.std()),
        "context_endpoint_sd_dollars": float(changes.std(axis=1).mean()),
        "context_drift_to_dispersion": float(context_rows.drift_to_dispersion.median()),
        "context_upward_path_share": float((changes > 0).mean()),
    }


def prediction_figure(training, validation, returns, prices, title, output):
    """Same full-day two-panel view and three fixed noise draws as the notebook."""
    boundary = len(training)
    real = np.r_[training.features[:, 0].numpy(), validation.features[:, 0].numpy()]
    real = real * training.feature_std[0] + training.feature_mean[0]
    train_prices, observed_prices = midpoint(training.books), midpoint(validation.books)
    all_x = np.arange(boundary + len(validation))
    xlabel = "Event index"
    if training.sampling_interval_seconds is not None:
        times = np.r_[training.timestamps, validation.timestamps]
        all_x = (times - times[0]) / 60
        xlabel = "Minutes since first snapshot"
        title = f"{training.sampling_interval_seconds:g}s snapshots | {title}"
    future_x = all_x[boundary:]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    axes[0].plot(all_x[:boundary], real[:boundary], color="tab:blue", linewidth=0.5, label="Training")
    axes[0].plot(future_x, real[boundary:], color="tab:orange", linewidth=0.5, label="Validation")
    axes[1].plot(all_x[:boundary], train_prices, color="tab:blue", label="Training")
    axes[1].plot(future_x, observed_prices, color="tab:orange", label="Validation")
    for index, (r, p) in enumerate(zip(returns, prices)):
        axes[0].plot(future_x, r, color="tab:green", alpha=0.4, linewidth=0.5,
                     label="Three generated return paths" if index == 0 else None)
        axes[1].plot(np.r_[all_x[boundary - 1], future_x], np.r_[train_prices[-1], p],
                     color="tab:green", alpha=0.6,
                     label="Three integrated price paths" if index == 0 else None)
    for ax in axes:
        ax.axvline((all_x[boundary - 1] + all_x[boundary]) / 2, color="gray", linestyle="--")
        ax.set(xlabel=xlabel)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.legend(loc="best", fontsize=9)
    axes[0].set(title="Decoder midpoint log returns", ylabel="Midpoint log return")
    axes[1].set(title="Prices accumulated from the last training midpoint", ylabel="Midpoint ($)")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def prediction_comparison_figure(training, validation, panels, output):
    """Compare the saved three-path futures with identical time and price axes."""
    anchor = midpoint(training.books)[-1]
    observed = np.r_[anchor, midpoint(validation.books)]
    x = np.arange(len(observed))
    xlabel = "Events after last training book"
    if training.sampling_interval_seconds is not None:
        x = np.r_[training.timestamps[-1], validation.timestamps]
        x = (x - x[0]) / 60
        xlabel = "Minutes after last training snapshot"
    fig, axes = plt.subplots((len(panels) + 1) // 2, 2, figsize=(14, 10),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    for ax, (row, prices) in zip(axes.flat, panels):
        ax.plot(x, observed, color="tab:orange", linewidth=1.2, label="Observed validation")
        for index, path in enumerate(prices):
            ax.plot(x, np.r_[anchor, path], color="tab:green", alpha=0.65,
                    linewidth=1, label="Three generated paths" if index == 0 else None)
        ax.set_title(f"Context {row['context_weight']:g} | Latent {row['latent_weight']:g} | Seed {row['seed']}\n"
                     f"Return SD: {row['full_return_sd_ratio']:.0%} of validation")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="best", fontsize=9)
    fig.supxlabel(xlabel)
    fig.supylabel("Midpoint ($)")
    interval = training.sampling_interval_seconds
    prefix = f"{interval:g}s snapshots | " if interval is not None else ""
    fig.suptitle(prefix + "Loss-weight comparison: same three noise draws", fontsize=14)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", type=Path, help="An existing notebook run, read without modifying it")
    args = parser.parse_args()
    torch.set_num_threads(1)
    datasets, dataset_protocols = {}, []
    figures = args.root / "figures"
    figures.mkdir(exist_ok=True)
    rows, comparison_panels = [], []
    checkpoints = sorted(args.root.glob("context_*_seed_*/model.pt"))
    if args.reference:
        checkpoints.insert(0, args.reference / "model.pt")
    for checkpoint_path in checkpoints:
        directory = checkpoint_path.parent
        if not (directory / "completion_midpoints.npz").exists():
            print(f"Skipping incomplete rollout: {directory.name}", flush=True)
            continue
        model, checkpoint = load_checkpoint(checkpoint_path, device=args.device)
        interval = checkpoint.get("data_config", {}).get("sampling_interval_seconds")
        if interval not in datasets:
            training = LOBSTERLevel10Dataset(True, sampling_interval_seconds=interval)
            validation = LOBSTERLevel10Dataset(False, sampling_interval_seconds=interval)
            datasets[interval] = training, validation
            dataset_protocols.append({"sampling_interval_seconds": interval,
                                      "training_rows": len(training), "validation_rows": len(validation),
                                      "split_time_seconds": training.split_time_seconds})
        training, validation = datasets[interval]
        real = validation.features[:, 0].numpy() * training.feature_std[0] + training.feature_mean[0]
        anchor = midpoint(training.books)[-1]
        np.testing.assert_allclose(checkpoint["feature_mean"].cpu(), training.feature_mean)
        np.testing.assert_allclose(checkpoint["feature_std"].cpu(), training.feature_std)
        with np.load(directory / "completion_log_returns.npz") as saved:
            returns, zero_returns = saved["noisy"], saved["zero"]
        with np.load(directory / "completion_midpoints.npz") as saved:
            prices = saved["noisy"]
        config = checkpoint["training_config"]
        metadata = json.loads((directory / "run.json").read_text())
        evaluation_directory = args.root / "evaluations" / directory.name
        evaluation_directory.mkdir(parents=True, exist_ok=True)
        blocks = returns[:, :returns.shape[1] // 1024 * 1024].reshape(3, -1, 1024).sum(-1)
        row = {
            "run": directory.name, "context_weight": config["context_loss_weight"],
            "latent_weight": config["eta"], "seed": metadata["seed"],
            "training_device": metadata.get("device", "cpu"),
            "sampling_interval_seconds": interval,
            "training_rows": len(training), "validation_rows": len(validation),
            "validation_return_sd": float(real.std()),
            "full_return_sd": float(returns.std(axis=1).mean()),
            "full_return_sd_ratio": float(returns.std(axis=1).mean() / real.std()),
            "full_return_mean": float(returns.mean()),
            "full_linear_r_squared": float(linear_r_squared(prices).mean()),
            "full_upward_block_share": float((blocks > 0).mean()),
            "full_endpoint_min_dollars": float((prices[:, -1] - anchor).min()),
            "full_endpoint_max_dollars": float((prices[:, -1] - anchor).max()),
            "zero_noise_endpoint_change_dollars": float(anchor * np.expm1(zero_returns.sum())),
            **context_metrics(model, training, validation, evaluation_directory),
        }
        rows.append(row)
        comparison_panels.append((interval, row, prices))
        title = (f"Context weight {config['context_loss_weight']:g} | "
                 f"Latent weight {config['eta']:g} | Seed {metadata['seed']}")
        prediction_figure(training, validation, returns, prices, title, figures / f"{directory.name}.png")
        pd.DataFrame(rows).to_csv(args.root / "comparison.csv", index=False)
        print(json.dumps(row), flush=True)
        del model
    # Keep event-based references separate from clock-based forecasts.
    for interval, (training, validation) in datasets.items():
        panels = [(row, prices) for cadence, row, prices in comparison_panels if cadence == interval]
        suffix = "events" if interval is None else f"{interval:g}s"
        prediction_comparison_figure(training, validation, panels,
                                     figures / f"prediction_comparison_{suffix}.png")
    (args.root / "evaluation_protocol.json").write_text(json.dumps({
        "contexts": 8, "samples_per_context": 32, "horizon": 1024, "evaluation_seed": 0,
        "evaluation_device": args.device, "noise_paired_across_models": True,
        "full_prediction_paths": 3, "datasets": dataset_protocols,
        "validation_used_for_model_comparison": True,
        "selection_note": "Inspect volatility, drift, and dispersion together; no noise-path selection.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
