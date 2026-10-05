"""Remove absolute-price reset by learning returns with stationary innovations.

Prices are integrated from the last observed midpoint. Forecast blocks use
only past generated features after the initial context. Return mean/std
matching discourages biased drift and collapsed or excessive volatility.
"""

import argparse
from pathlib import Path

from train import TrainingConfig
from training_balance_experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, default=1)
    parser.add_argument("--moment-weight", type=float, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("runs/return_price"))
    parser.add_argument("--autoencoder-steps", type=int, default=1000)
    parser.add_argument("--transition-steps", type=int, default=1000)
    parser.add_argument("--joint-steps", type=int, default=5000)
    args = parser.parse_args()
    config = TrainingConfig(
        discriminator_learning_rate=1e-4, price_representation="return",
        noise_kind="wiener_increments", price_moment_weight=args.moment_weight,
        context_loss_weight=args.weight, context_horizon=16,
        autoencoder_steps=args.autoencoder_steps, transition_steps=args.transition_steps,
        joint_steps=args.joint_steps,
    )
    name = f"returns_context_{args.weight:g}_moments_{args.moment_weight:g}"
    run_experiment(name, config, args.output, args.seed, args.threads)


if __name__ == "__main__":
    main()
