"""Run the fixed-recipe, three-seed schedule experiment and its guardrail report.

This is model-selection evaluation on one repeatedly inspected validation day,
not a test of generalization to new days. A failed schedule is reported without
replacing the selected model or changing the prescribed loss weights.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import LOBSTERLevel10Dataset
from loss_weight_report import endpoint_distribution, summarize_endpoints
from modules import TimeGAN
from optimization_metrics import assess_guardrails, validation_loss_metrics, verify_checkpoint_data
from predict import load_checkpoint
from terminal_price_report import distribution_figure, return_price_figure
from train import TrainingConfig
from training_balance_experiment import run_experiment


DEFAULT_BASELINE = Path("runs/log_return/one_second_balance_20261010/selected/model.pt")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def baseline_recipe(baseline):
    """Use full-training provenance; the selected checkpoint itself stores joint_steps=0."""
    selection = json.loads((baseline.parent / "selection.json").read_text())
    source = Path(selection["source_checkpoint"])
    recipe = json.loads((source.parent / "run.json").read_text())
    return recipe, selection["refinement_steps"]


def evaluate_checkpoint(path, directory, training, validation, baseline_losses, args):
    directory.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint(path, args.device)
    verify_checkpoint_data(checkpoint, training)
    losses = validation_loss_metrics(model, checkpoint["training_config"], training, validation)
    write_json(directory / "validation_losses.json", losses)
    all_returns, all_prices, draws = [], [], []
    for seed in args.noise_seeds:
        metrics, returns, prices = endpoint_distribution(model, training, validation, args.samples, seed)
        draws.append(metrics)
        all_returns.append(returns)
        all_prices.append(prices)
        write_json(directory / f"noise_seed_{seed}.json", metrics)
        print(json.dumps({"checkpoint": str(path), "noise_seed": seed,
                          "volatility_ratio": metrics["return_sd_ratio"],
                          "median_bias_dollars": metrics["terminal_median_bias_dollars"]}), flush=True)
    returns, prices = np.concatenate(all_returns), np.concatenate(all_prices)
    pooled = summarize_endpoints(training, validation, returns, prices)
    # Order is deliberate: the first three paths always come from noise seed 7.
    np.savez_compressed(directory / "futures.npz", returns=returns, prices=prices)
    reference = baseline_losses if baseline_losses is not None else losses
    report = {"checkpoint": str(path.resolve()), "training_config": checkpoint["training_config"],
              "distribution": pooled, "per_noise_seed": draws, "losses": losses,
              "guardrails": assess_guardrails(pooled, losses, reference),
              "validation_is_model_selection_data": True,
              "first_validation_snapshot_is_observed_context": True,
              "post_generation_calibration": False}
    write_json(directory / "metrics.json", report)
    return report


def loss_figure(joint, refinement, baseline_joint, output, title):
    """Raw loss components; stage boundaries and changing weights stay explicit."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    metrics = [("reconstruction", "Feature reconstruction MSE"),
               ("g_supervised", "Latent supervision MSE"),
               ("discriminator", "Discriminator BCE (real + fake)"),
               ("adversarial", "Generator adversarial BCE")]
    for ax, (key, label) in zip(axes.flat, metrics):
        for history, color, name in [(baseline_joint, "gray", "Selected model's joint training"),
                                     (joint, "tab:blue", "New joint schedule")]:
            ax.plot(history.step, history[key].rolling(100, min_periods=1).mean(),
                    color=color, label=name)
        if key in refinement:
            ax.plot(joint.step.max() + refinement.step,
                    refinement[key].rolling(100, min_periods=1).mean(),
                    color="tab:green", label="Fixed generator-only refinement")
        ax.axvline(joint.step.max(), color="gray", linestyle="--")
        ax.set(title=label, xlabel="Joint cycles, then refinement updates",
               xlim=(0, joint.step.max() + refinement.step.max()))
        if key in ("reconstruction", "g_supervised"):
            ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"{title} | 100-update moving averages\n"
                 "D is not updated during refinement; generator loss weights differ between stages")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def gradient_figure(histories, output):
    fig, axes = plt.subplots(len(histories), 2, figsize=(13, 3.5 * len(histories)),
                             squeeze=False, layout="constrained")
    components = ("adversarial", "g_supervised", "g_context_price", "g_price_moments", "g_conditional_drift")
    for row, (seed, history) in enumerate(histories.items()):
        if "g_total_gradient_norm" not in history:
            for ax in axes[row]:
                ax.text(0.5, 0.5, "No diagnostic cycle reached", ha="center", transform=ax.transAxes)
            continue
        selected = history.dropna(subset=["g_total_gradient_norm"])
        for index, name in enumerate(components):
            label = name.removeprefix("g_").replace("_", " ")
            axes[row, 0].plot(selected.step, selected[f"gradient_norm_{name}"].clip(lower=1e-12),
                              marker=".", label=label, color=f"C{index}")
            if name != "adversarial":
                axes[row, 1].plot(selected.step, selected[f"gradient_cosine_{name}_adversarial"],
                                  marker=".", label=label, color=f"C{index}")
        axes[row, 0].set(yscale="log", title=f"Training seed {seed}: weighted gradient norms",
                         xlabel="Joint cycle", ylabel="L2 norm before clipping")
        axes[row, 1].set(title=f"Training seed {seed}: alignment with adversarial gradient",
                         xlabel="Joint cycle", ylabel="Cosine", ylim=(-1.05, 1.05))
        axes[row, 1].axhline(0, color="gray", linewidth=0.8)
        for ax in axes[row]:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    fig.suptitle("First generator update in every diagnostic cycle; negative cosine indicates competing directions")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def render_report(root, reports, seeds, recipe, args, protocol):
    """Use identical axes and the saved numerical arrays for every price figure."""
    arrays = {}
    for name in reports:
        with np.load(root / "reports" / name / "futures.npz") as data:
            arrays[name] = {"returns": data["returns"], "prices": data["prices"]}
    price_low = min(float(data["prices"].min()) for data in arrays.values())
    price_high = max(float(data["prices"].max()) for data in arrays.values())
    observed_prices = (args.validation.books.ASKp1 + args.validation.books.BIDp1).to_numpy() / 20_000
    price_low, price_high = min(price_low, observed_prices.min()), max(price_high, observed_prices.max())
    observed = args.validation.features[1:, 0].numpy() * args.training.feature_std[0] + args.training.feature_mean[0]
    return_low = min(float(observed.min()), *(float(data["returns"][:3].min()) for data in arrays.values()))
    return_high = max(float(observed.max()), *(float(data["returns"][:3].max()) for data in arrays.values()))
    limits = {"price": (price_low, price_high), "returns": (return_low, return_high),
              "terminal": (min(float(data["prices"][:, -1].min()) for data in arrays.values()),
                           max(float(data["prices"][:, -1].max()) for data in arrays.values()))}
    for name, report in reports.items():
        directory = root / "reports" / name
        data = arrays[name]
        distribution_figure(args.validation, report["distribution"], data["prices"], report["training_config"],
                            directory / "forecast_distribution.png", limits=limits)
        return_price_figure(args.validation, data["returns"], data["prices"], report["distribution"],
                            directory / "return_price_paths.png", limits=limits)
        write_json(directory / "plot_metadata.json", {"common_axes": limits, "fixed_paths": [0, 1, 2],
                   "path_noise_seed": args.noise_seeds[0], "anchor_dollars": report["distribution"]["anchor_dollars"],
                   "horizon_seconds": report["distribution"]["horizon_seconds"],
                   "rendered": True, "human_visual_review_completed": False})
    histories, rows = {}, []
    baseline_joint = pd.read_csv(Path(recipe["parent_checkpoint"]).parent / "losses.csv").query("stage == 'joint'")
    for seed in seeds:
        joint = pd.read_csv(root / "joint" / f"schedule_seed_{seed}" / "losses.csv").query("stage == 'joint'")
        refinement = pd.read_csv(root / f"seed_{seed}" / "losses.csv")
        histories[seed] = joint
        loss_figure(joint, refinement, baseline_joint, root / "reports" / f"seed_{seed}" / "loss_curves.png",
                    f"Training seed {seed}")
    gradient_figure(histories, root / "weighted_gradients.png")
    for name, report in reports.items():
        rows.append({"model": name, "passed": report["guardrails"]["passed"], **report["distribution"],
                     **{key: value for key, value in report["losses"].items() if isinstance(value, (float, int))},
                     **{f"generated_book_{key}": value for key, value in
                        report["losses"]["books"]["generated_unconditional"].items()},
                     **{f"validation_book_{key}": value for key, value in
                        report["losses"]["books"]["validation"].items()}})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "comparison.csv", index=False)
    complete_protocol = (tuple(seeds) == (0, 1, 2) and tuple(args.noise_seeds) == (7, 17, 27)
                         and args.samples == 1024 and protocol["refinement_steps"] == 1000
                         and protocol["joint_training_config"]["generator_updates"] == 2
                         and protocol["joint_training_config"]["discriminator_loss_threshold"] == 0.15
                         and protocol["joint_training_config"]["gradient_diagnostic_interval"] == 500
                         and all(protocol["joint_training_config"][key] == value for key, value in
                                 dict(autoencoder_steps=1000, transition_steps=1000, joint_steps=5000).items()))
    accepted = complete_protocol and all(reports[f"seed_{seed}"]["guardrails"]["passed"] for seed in seeds)
    acceptance = {"accepted_across_all_training_seeds": accepted, "complete_protocol": complete_protocol,
                  "training_seed_results": {str(seed): reports[f"seed_{seed}"]["guardrails"] for seed in seeds},
                  "automatic_checkpoint_promotion": False, "validation_is_model_selection_data": True,
                  "visual_review_completed": False}
    write_json(root / "acceptance.json", acceptance)
    table = ["| Model | Return SD / validation | Median bias ($) | Endpoint SD ($) | Log CDF distance | All gates |",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        table.append(f"| {row['model']} | {row['return_sd_ratio']:.1%} | {row['terminal_median_bias_dollars']:+.4f} | "
                     f"{row['terminal_sd_dollars']:.4f} | {row['terminal_log_normal_ks']:.4f} | "
                     f"{'PASS' if row['passed'] else 'FAIL'} |")
    loss_table = ["| Model | Latent supervised MSE | Reconstruction MSE | Return reconstruction MSE | G adversarial BCE | D BCE |",
                  "|---|---:|---:|---:|---:|---:|"]
    book_table = ["| Data | Mean spread ($) | Spread SD ($) | Mean ask depth (shares) | Mean bid depth (shares) |",
                  "|---|---:|---:|---:|---:|"]
    observed_books = reports["baseline"]["losses"]["books"]["validation"]
    book_data = {"Validation": observed_books} | {
        name: report["losses"]["books"]["generated_unconditional"] for name, report in reports.items()}
    for name, values in book_data.items():
        book_table.append(f"| {name} | {values['spread_mean_dollars']:.4f} | {values['spread_std_dollars']:.4f} | "
                          f"{values['ask_depth_mean_shares']:.1f} | {values['bid_depth_mean_shares']:.1f} |")
    for row in rows:
        loss_table.append(f"| {row['model']} | {row['latent_supervision_mse']:.6g} | {row['reconstruction_mse']:.6g} | "
                          f"{row['return_reconstruction_mse']:.6g} | {row['adversarial']:.3f} | {row['discriminator_bce']:.3f} |")
    text = ("# One-second TimeGAN schedule and guardrails\n\n"
            f"Accepted across all three training seeds: **{accepted}**. Full prescribed protocol: **{complete_protocol}**.\n\n"
            "The saved selected checkpoint is preserved. Architecture, stage budgets, loss weights and final refinement "
            "are fixed; joint scheduling changes to two (generator, embedding) pairs followed by a discriminator "
            "update only when its pre-update summed BCE exceeds 0.15. The reference implementation has an additional "
            "fake BCE term, so its threshold is adapted to our two-term loss.\n\n" + "\n".join(table) + "\n\n"
            "Acceptance uses each training seed's pooled 3,072 futures. Individual noise-seed scores, median 95% "
            "simulation confidence intervals, skew, excess kurtosis, upward share, return autocorrelation, raw/weighted "
            "validation losses and tick-projected spread/depth statistics are in each `metrics.json`. Loss comparisons "
            "use the same 35 non-overlapping validation windows and ten shared noise draws.\n\n"
            "The gates are return SD ratio 0.9–1.1, median bias within ±$0.05, fitted-normal log-price CDF distance "
            "≤0.03, endpoint SD $0.6496–$0.9744, and each raw supervised/overall/return reconstruction MSE ≤1.1 times "
            "the freshly evaluated selected baseline. See `acceptance.json` for every value and limit.\n\n"
            + "\n".join(loss_table) + "\n\nBook statistics below use unconditional, tick-projected 64-second windows; "
            "they are reported separately from the unrounded conditional futures.\n\n" + "\n".join(book_table) + "\n\n"
            "GAN loss and discriminator accuracy describe training balance; lowering generator BCE by weakening D "
            "does not independently establish more realistic generation. Different runs also learn different latent "
            "coordinates. This repeatedly inspected validation day is model-selection data, not an untouched test day. "
            "Price outputs come directly from decoder returns, without scaling or recentering.\n\n"
            "All price plots share axes, retain fixed first-three noise draws and begin at the first validation snapshot. "
            "Loss curves are raw 100-update averages; weighted gradient norms and cosine alignment are measured before "
            "clipping on the first generator update every 500 joint cycles. Actual update/skip counts are saved in joint "
            "run metadata and training logs.\n\n")
    for name, report in reports.items():
        directory = (root / "reports" / name).resolve()
        text += f"## {name}\n\n![Decoded return and price paths]({directory}/return_price_paths.png)\n\n"
        text += f"![Forecast ensemble and lognormal checks]({directory}/forecast_distribution.png)\n\n"
        if name != "baseline":
            text += f"![Raw training losses]({directory}/loss_curves.png)\n\n"
    text += f"![Weighted gradient diagnostics]({root.resolve()}/weighted_gradients.png)\n"
    (root / "README.md").write_text(text)
    print(json.dumps(acceptance), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=Path("runs/log_return/schedule_guardrails_20261010"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--discriminator-loss-threshold", type=float, default=0.15)
    parser.add_argument("--gradient-diagnostic-interval", type=int, default=500)
    parser.add_argument("--autoencoder-steps", type=int)
    parser.add_argument("--transition-steps", type=int)
    parser.add_argument("--joint-steps", type=int)
    parser.add_argument("--refinement-steps", type=int)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--render-only", action="store_true",
                        help="Regenerate plots/tables from existing evaluated arrays; no training or sampling")
    args = parser.parse_args()
    if args.samples < 32 or len(set(args.training_seeds)) != len(args.training_seeds) or len(set(args.noise_seeds)) != len(args.noise_seeds):
        parser.error("Use >=32 futures and distinct training/noise seeds")
    torch.set_num_threads(2)
    args.baseline = args.baseline.resolve()
    recipe, refinement_steps = baseline_recipe(args.baseline)
    base_config = TrainingConfig(**recipe["parent_training_config"])
    changes = {key: getattr(args, key) for key in ("autoencoder_steps", "transition_steps", "joint_steps")
               if getattr(args, key) is not None}
    config = replace(base_config, generator_updates=2,
                     discriminator_loss_threshold=args.discriminator_loss_threshold,
                     gradient_diagnostic_interval=args.gradient_diagnostic_interval, **changes)
    refinement_steps = args.refinement_steps if args.refinement_steps is not None else refinement_steps
    if refinement_steps < 1:
        parser.error("Refinement needs at least one update")
    _, baseline_checkpoint = load_checkpoint(args.baseline)
    if baseline_checkpoint["model_config"] != TimeGAN().model_config:
        parser.error("This schedule comparison preserves the existing 128-dimensional architecture")
    baseline_hash = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    protocol = {"baseline_checkpoint": str(args.baseline), "baseline_sha256": baseline_hash,
                "joint_training_config": asdict(config), "refinement_training_config": recipe["training_config"],
                "refinement_steps": refinement_steps, "training_seeds": args.training_seeds,
                "noise_seeds": args.noise_seeds, "futures_per_noise_seed": args.samples,
                "validation_is_model_selection_data": True, "post_generation_calibration": False}
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists() and json.loads((root / "protocol.json").read_text()) != protocol:
        raise ValueError("Existing output has a different protocol; choose a new output directory")
    write_json(root / "protocol.json", protocol)
    for seed in args.training_seeds:
        joint = root / "joint" / f"schedule_seed_{seed}"
        refined = root / f"seed_{seed}"
        if not args.report_only and not args.render_only and not (joint / "model.pt").exists():
            run_experiment("schedule", config, root / "joint", seed, 2, args.device,
                           sampling_interval_seconds=1)
        if not args.report_only and not args.render_only and not (refined / "model.pt").exists():
            refinement_config = recipe["training_config"]
            subprocess.run([sys.executable, str(Path(__file__).with_name("refine_loss_weights.py")),
                            str(joint / "model.pt"), "--output", str(refined), "--steps", str(refinement_steps),
                            "--learning-rate", str(refinement_config["learning_rate"]),
                            "--moment-weight", str(refinement_config["price_moment_weight"]),
                            "--drift-weight", str(refinement_config["conditional_drift_weight"]),
                            "--seed", str(seed), "--device", args.device], check=True)
        if not (refined / "model.pt").exists():
            raise FileNotFoundError(f"Missing completed refinement for training seed {seed}")
    args.training = LOBSTERLevel10Dataset(True, sampling_interval_seconds=1)
    args.validation = LOBSTERLevel10Dataset(False, sampling_interval_seconds=1)
    if args.render_only:
        names = ["baseline"] + [f"seed_{seed}" for seed in args.training_seeds]
        reports = {name: json.loads((root / "reports" / name / "metrics.json").read_text()) for name in names}
        render_report(root, reports, args.training_seeds, recipe, args, protocol)
        return
    reports = {"baseline": evaluate_checkpoint(args.baseline, root / "reports" / "baseline",
                                               args.training, args.validation, None, args)}
    for seed in args.training_seeds:
        reports[f"seed_{seed}"] = evaluate_checkpoint(root / f"seed_{seed}" / "model.pt",
                        root / "reports" / f"seed_{seed}", args.training, args.validation,
                        reports["baseline"]["losses"], args)
    render_report(root, reports, args.training_seeds, recipe, args, protocol)
    if hashlib.sha256(args.baseline.read_bytes()).hexdigest() != baseline_hash:
        raise AssertionError("The selected baseline checkpoint changed")


if __name__ == "__main__":
    main()
