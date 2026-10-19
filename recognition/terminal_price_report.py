"""Inspect learned return volatility and the final-price distribution.

The first validation snapshot is observed conditioning data. Every later return
comes from TimeGAN's decoder; no post-generation centering or scaling is used.
"""

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import LOBSTERLevel10Dataset
from loss_weight_report import endpoint_distribution
from predict import load_checkpoint
from training_balance_experiment import midpoint


def return_price_figure(validation, returns, prices, metrics, output, *, limits=None):
    """Show raw decoder returns and the same three fixed price continuations."""
    observed = validation.features[1:, 0].numpy() * validation.feature_std[0] + validation.feature_mean[0]
    anchor = metrics["anchor_dollars"]
    x = np.arange(prices.shape[1] + 1) / 60
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout="constrained")
    axes[0].plot(x[1:], observed, color="tab:orange", linewidth=0.6, label="Observed validation returns")
    axes[1].plot(x, midpoint(validation.books), color="tab:orange", label="Observed validation price")
    for index in range(3):
        axes[0].plot(x[1:], returns[index], color="tab:green", alpha=0.4, linewidth=0.5,
                     label="Three generated return paths" if index == 0 else None)
        axes[1].plot(x, np.r_[anchor, prices[index]], color="tab:green", alpha=0.65,
                     label="Three integrated price paths" if index == 0 else None)
    axes[0].set(title=f"Decoder log returns: {metrics['return_sd_ratio']:.1%} of validation SD", ylabel="Midpoint log return")
    axes[1].set(title=f"Prices integrated from the first validation midpoint: USD {anchor:.3f}",
                ylabel="Midpoint ($)", xlabel="Minutes after first validation snapshot")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.15)
    if limits:
        axes[0].set_ylim(*limits["returns"])
        axes[1].set_ylim(*limits["price"])
    fig.savefig(output, dpi=140)
    plt.close(fig)


def distribution_figure(validation, metrics, prices, config, output, *, limits=None):
    anchor = metrics["anchor_dollars"]
    x = np.arange(prices.shape[1] + 1) / 60
    paths = np.column_stack((np.full(len(prices), anchor), prices))
    observed = midpoint(validation.books)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax = axes[0, 0]
    ax.plot(x, observed, color="tab:orange", label="Observed validation")
    for index, path in enumerate(paths[:3]):
        ax.plot(x, path, color="tab:green", alpha=0.65,
                label="First three generated paths" if index == 0 else None)
    ax.set(title="Three fixed noise draws", xlabel="Minutes after first validation snapshot", ylabel="Midpoint ($)")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    quantiles = np.quantile(paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ax.fill_between(x, quantiles[0], quantiles[4], color="tab:green", alpha=0.15, label="90% ensemble interval")
    ax.fill_between(x, quantiles[1], quantiles[3], color="tab:green", alpha=0.25, label="50% ensemble interval")
    ax.plot(x, quantiles[2], color="tab:green", label="Ensemble median")
    ax.plot(x, observed, color="tab:orange", linewidth=1, label="Observed validation")
    ax.axhline(anchor, color="gray", linestyle="--", label=f"Starting price: ${anchor:.3f}")
    ax.set(title=f"{len(prices):,} generated futures", xlabel="Minutes after first validation snapshot", ylabel="Midpoint ($)")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    terminal = prices[:, -1]
    log_endpoints = np.log(terminal)
    mu, sigma = log_endpoints.mean(), log_endpoints.std()
    grid = np.linspace(terminal.min(), terminal.max(), 400)
    ax.hist(terminal, bins=35, density=True, color="tab:green", alpha=0.45, label="Generated final prices")
    if sigma > 0:
        density = np.exp(-0.5 * ((np.log(grid) - mu) / sigma) ** 2) / (grid * sigma * np.sqrt(2 * np.pi))
        ax.plot(grid, density, color="black", linewidth=1.5, label="Fitted lognormal density")
    ax.axvline(anchor, color="tab:orange", linestyle="--", label="Requested median")
    ax.axvline(np.median(terminal), color="tab:green", label="Generated median")
    ax.set(title=f"Final median: USD {np.median(terminal):.3f}; SD: USD {terminal.std():.3f}",
           xlabel="Final midpoint ($)", ylabel="Density")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    normal = NormalDist()
    expected = np.array([normal.inv_cdf((index + 0.5) / len(terminal)) for index in range(len(terminal))])
    actual = np.sort((log_endpoints - mu) / max(sigma, 1e-12))
    ax.plot(expected, actual, ".", color="tab:green", markersize=3)
    ax.plot(expected, expected, color="gray", linestyle="--")
    ax.set(title=f"Log-price normal Q–Q; CDF distance {metrics['terminal_log_normal_ks']:.3f}",
           xlabel="Normal quantile", ylabel="Standardized log-price quantile")
    for ax in axes.flat:
        ax.grid(alpha=0.15)
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    if limits:
        axes[0, 0].set_ylim(*limits["price"])
        axes[0, 1].set_ylim(*limits["price"])
        axes[1, 0].set_xlim(*limits["terminal"])
    fig.suptitle(f"1s snapshots | Context {config['context_loss_weight']:g} | Latent {config['eta']:g} | "
                 f"Moments {config['price_moment_weight']:g} | Drift {config.get('conditional_drift_weight', 0):g}\n"
                 f"Return volatility: {metrics['return_sd_ratio']:.1%} of validation", fontsize=14)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.samples < 32:
        parser.error("Use at least 32 futures for distribution diagnostics")
    directory = args.root / f"terminal_report_seed_{args.seed}_n_{args.samples}"
    directory.mkdir(parents=True, exist_ok=True)
    training = LOBSTERLevel10Dataset(True, sampling_interval_seconds=1)
    validation = LOBSTERLevel10Dataset(False, sampling_interval_seconds=1)
    checkpoints = [args.checkpoint] if args.checkpoint else sorted(args.root.glob("context*/model.pt"))
    for path in checkpoints:
        model, checkpoint = load_checkpoint(path, device=args.device)
        if checkpoint.get("data_config", {}).get("sampling_interval_seconds") != 1:
            raise ValueError("This report requires one-second checkpoints")
        np.testing.assert_allclose(checkpoint["feature_mean"].cpu(), training.feature_mean)
        np.testing.assert_allclose(checkpoint["feature_std"].cpu(), training.feature_std)
        metrics, returns, prices = endpoint_distribution(model, training, validation, args.samples, args.seed)
        config = checkpoint["training_config"]
        name = path.parent.name if path.stem == "model" else f"{path.parent.name}_{path.stem}"
        row = {"run": name, "checkpoint": str(path), "context_weight": config["context_loss_weight"],
               "latent_weight": config["eta"], "moment_weight": config["price_moment_weight"],
               "drift_weight": config.get("conditional_drift_weight", 0), **metrics}
        np.savez_compressed(directory / f"{name}.npz", returns=returns, prices=prices)
        (directory / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")
        distribution_figure(validation, metrics, prices, config, directory / f"{name}.png")
        return_price_figure(validation, returns, prices, metrics, directory / f"{name}_returns.png")
        # Preserve results when individual checkpoints are evaluated separately.
        rows = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
        pd.DataFrame(rows).to_csv(directory / "comparison.csv", index=False)
        print(json.dumps(row), flush=True)
        del model


if __name__ == "__main__":
    main()
