"""Test a decoded midpoint loss on 16-event context-conditioned continuations.

Compare against discriminator_1e-4_seed_0 from the training-balance experiment.
Architecture, data order, noise, warm-ups, and training budgets are identical.
Only the generator's joint-training loss changes; validation stays held out.
"""

import argparse
from pathlib import Path

from train import TrainingConfig
from training_balance_experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("runs/context_loss"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--autoencoder-steps", type=int, default=1000)
    parser.add_argument("--transition-steps", type=int, default=1000)
    parser.add_argument("--joint-steps", type=int, default=5000)
    args = parser.parse_args()
    config = TrainingConfig(
        discriminator_learning_rate=1e-4, context_loss_weight=args.weight,
        context_horizon=args.horizon, autoencoder_steps=args.autoencoder_steps,
        transition_steps=args.transition_steps, joint_steps=args.joint_steps,
    )
    experiment = f"context_weight_{args.weight:g}_horizon_{args.horizon}"
    run_experiment(experiment, config, args.output, args.seed, args.threads)


if __name__ == "__main__":
    main()
