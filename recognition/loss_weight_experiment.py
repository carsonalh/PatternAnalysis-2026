"""Retrain context/latent-loss comparisons with the notebook's fixed features.

The CLI exposes context, latent, return-moment, and conditional drift penalties.
Warm-ups, data sampling, features, architecture, and noise stay fixed by default.
"""

import argparse
from pathlib import Path

from train import TrainingConfig
from training_balance_experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-weight", type=float, required=True)
    parser.add_argument("--latent-weight", type=float, required=True)
    parser.add_argument("--moment-weight", type=float, default=10)
    parser.add_argument("--embedder-weight", type=float, default=1)
    parser.add_argument("--drift-weight", type=float, default=0)
    parser.add_argument("--generator-updates", type=int, default=2)
    parser.add_argument("--discriminator-loss-threshold", type=float, default=0.15)
    parser.add_argument("--gradient-diagnostic-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampling-interval-seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/log_return/loss_weights"))
    parser.add_argument("--autoencoder-steps", type=int, default=1000)
    parser.add_argument("--transition-steps", type=int, default=1000)
    parser.add_argument("--joint-steps", type=int, default=5000)
    args = parser.parse_args()
    config = TrainingConfig(
        context_loss_weight=args.context_weight, eta=args.latent_weight,
        discriminator_learning_rate=1e-4, price_moment_weight=args.moment_weight,
        lam=args.embedder_weight,
        conditional_drift_weight=args.drift_weight,
        generator_updates=args.generator_updates,
        discriminator_loss_threshold=args.discriminator_loss_threshold,
        gradient_diagnostic_interval=args.gradient_diagnostic_interval,
        noise_kind="wiener_increments", context_horizon=16,
        autoencoder_steps=args.autoencoder_steps, transition_steps=args.transition_steps,
        joint_steps=args.joint_steps,
    )
    name = f"context_{args.context_weight:g}_latent_{args.latent_weight:g}"
    if args.moment_weight != 10:
        name += f"_moments_{args.moment_weight:g}"
    if args.embedder_weight != 1:
        name += f"_embedder_{args.embedder_weight:g}"
    if args.drift_weight != 0:
        name += f"_drift_{args.drift_weight:g}"
    name += f"_g{args.generator_updates}_dgate{args.discriminator_loss_threshold:g}"
    run_experiment(name, config, args.output, args.seed, args.threads, args.device,
                   sampling_interval_seconds=args.sampling_interval_seconds)


if __name__ == "__main__":
    main()
