"""Fit conditional volatility using training prefixes and their real returns."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import LOBSTERLevel10Dataset, WindowDataset
from predict import load_checkpoint
from price_dynamics import PriceVolatility, innovation_completion
from train import TrainingConfig
import training_balance_experiment as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/return_price/innovation_head_seed_0"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("Calibration steps and threads must be positive")
    if (args.output / "price_head.pt").exists():
        raise FileExistsError("Use another output directory for a new calibration")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, checkpoint = load_checkpoint(args.checkpoint)
    if model.price_representation != "return" or model.noise_kind != "wiener_increments":
        raise ValueError("The price head needs a return model with increment noise")
    model.requires_grad_(False)
    config = TrainingConfig(**checkpoint["training_config"])
    training = LOBSTERLevel10Dataset(True, price_representation="return")
    validation = LOBSTERLevel10Dataset(False, price_representation="return")
    raw_returns = training.features[:, 0].double().numpy() * training.feature_std[0] + training.feature_mean[0]
    calibration = {
        "return_mean": float(training.feature_mean[0]),
        "return_std": float(training.feature_std[0]),
        "log_expected_price_factor": float(np.log1p(np.expm1(raw_returns).mean())),
        "sequence_length": config.sequence_length,
    }
    head = PriceVolatility(model.model_config["latent_dims"])
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-4)
    loader = DataLoader(WindowDataset(training.features, config.sequence_length),
                        batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    batches = iter(loader)
    split = config.sequence_length - config.context_horizon
    history = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        try:
            x = next(batches)
        except StopIteration:
            batches = iter(loader)
            x = next(batches)
        with torch.no_grad():
            context = model.embedder(x[:, :split])[:, -1]
        sigma = head(context)
        mu = ((calibration["log_expected_price_factor"] - 0.5 * (sigma * calibration["return_std"]).square()
               - calibration["return_mean"]) / calibration["return_std"])
        loss = (sigma.log() + 0.5 * ((x[:, split:, 0] - mu) / sigma).square()).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "gaussian_nll": loss.item()})
        if step == 1 or step % 250 == 0:
            print(f"Volatility calibration {step}/{args.steps}: NLL={loss.item():.5f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    head.eval()
    torch.save({"state_dict": head.state_dict(), "latent_dims": model.model_config["latent_dims"],
                "calibration": calibration}, args.output / "price_head.pt")
    # The unified checkpoint selects the innovation head automatically on load.
    model.price_volatility = head
    model.price_calibration = calibration
    model.model_config["price_dynamics"] = "innovation_head"
    checkpoint.update(model_state=model.state_dict(), model_config=model.model_config,
                      price_calibration=calibration)
    torch.save(checkpoint, args.output / "model.pt")
    metadata = json.loads((args.checkpoint.parent / "run.json").read_text())
    metadata.update(price_model="conditional_volatility_with_empirical_drift", calibration=calibration,
                    calibration_steps=args.steps, calibration_seconds=time.monotonic() - started)
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pd.DataFrame(history).to_csv(args.output / "volatility_losses.csv", index=False)
    metrics, curves, contexts = experiment.diagnostics(model, training, validation, config)
    (args.output / "diagnostics_final.json").write_text(json.dumps(
        {"metrics": metrics, "curves": curves}, indent=2, allow_nan=False) + "\n")
    contexts.to_csv(args.output / "contexts_final.csv", index=False)
    contexts.to_csv(args.output / f"contexts_step_{config.joint_steps}.csv", index=False)
    pd.DataFrame([{"joint_step": config.joint_steps, **metrics}]).to_csv(args.output / "snapshots.csv", index=False)
    shutil.copy2(args.checkpoint.parent / "losses.csv", args.output / "losses.csv")
    torch.set_num_threads(1)
    context = training.features[-config.sequence_length:][None]
    paths = experiment.noise_paths(3, len(validation), model, config.sequence_length)
    noisy = innovation_completion(model, head, context.repeat(3, 1, 1), paths, calibration)
    zero = innovation_completion(model, head, context, paths[:1] * 0, calibration)
    anchor = experiment.midpoint(training.books)[-1] * 10_000
    project = lambda x: experiment.midpoint(training.feature_sequence_to_df(x, initial_midpoint=anchor))
    np.savez_compressed(args.output / "completion_midpoints.npz",
                        noisy=np.stack([project(x) for x in noisy]), zero=project(zero[0]))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
