import numpy as np
import torch

from dataset import features_to_order_book
from modules import LATENT_DIMS, TimeGAN


def generate_wiener_paths(
    batch_size: int, sequence_length: int = 64, noise_dims: int = LATENT_DIMS,
    *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate multidimensional Wiener paths (B, T, Z) and Gaussian seeds (B, Z).

    The grid spans [0, 1] in event order; it does not model elapsed clock time.
    """
    if batch_size < 1 or sequence_length < 2 or noise_dims < 1:
        raise ValueError("Noise needs a positive batch/dimension and at least two events")
    increments = torch.randn(batch_size, sequence_length - 1, noise_dims,
                             device=device, dtype=dtype, generator=rng)
    increments *= (1 / (sequence_length - 1)) ** 0.5
    zero = torch.zeros(batch_size, 1, noise_dims, device=device, dtype=dtype)
    paths = torch.cat((zero, increments.cumsum(dim=1)), dim=1)
    seed = torch.randn(batch_size, noise_dims, device=device, dtype=dtype, generator=rng)
    return paths, seed


@torch.no_grad()
def generate_features(model: TimeGAN, batch_size: int, sequence_length: int = 64) -> torch.Tensor:
    """Generate standardized features without real books or the embedder."""
    parameter = next(model.generator.parameters())
    paths, seed = generate_wiener_paths(
        batch_size, sequence_length, model.generator.noise_dims,
        device=parameter.device, dtype=parameter.dtype)
    if getattr(model, "noise_kind", "wiener_paths") == "wiener_increments":
        paths = torch.cat((paths[:, :1], paths[:, 1:] - paths[:, :-1]), dim=1)
        paths *= ((sequence_length - 1) / (getattr(model, "sequence_length", sequence_length) - 1)) ** 0.5
    latents = model.generator.sample(paths, seed)
    # Every generated feature, including midpoint returns, comes from recovery.
    return model.decoder(latents)


def load_checkpoint(path, device: torch.device | str = "cpu") -> tuple[TimeGAN, dict]:
    """Load weights and training normalization for data-independent sampling."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_config = dict(checkpoint["model_config"])
    if (model_config.pop("price_dynamics", "decoder") != "decoder"
            or "price_calibration" in checkpoint
            or any(name.startswith("price_volatility.") for name in checkpoint["model_state"])):
        raise ValueError(
            "This checkpoint replaces the TimeGAN decoder's midpoint output. "
            "Use a TimeGAN checkpoint such as "
            "runs/return_price/returns_context_1_moments_10_seed_0/model.pt."
        )
    model = TimeGAN(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.noise_kind = checkpoint["training_config"].get("noise_kind", "wiener_paths")
    model.price_representation = checkpoint.get("price_representation", "level")
    model.context_horizon = checkpoint["training_config"].get("context_horizon", 16)
    model.sequence_length = checkpoint["training_config"]["sequence_length"]
    model.eval()
    return model, checkpoint


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Sample event-time AMZN windows from TimeGAN")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/samples"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model, checkpoint = load_checkpoint(args.checkpoint)
    features = generate_features(model, args.samples, checkpoint["training_config"]["sequence_length"])
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "features.npy", features.cpu().numpy())
    mean = checkpoint["feature_mean"].cpu().numpy()
    std = checkpoint["feature_std"].cpu().numpy()
    for index, sequence in enumerate(features):
        features_to_order_book(
            sequence, mean, std,
            price_representation=checkpoint.get("price_representation", "level"),
            initial_midpoint=checkpoint.get("reference_midpoint"),
        ).to_csv(
            args.output / f"order_book_{index:04d}.csv", index=False)
    print(f"Saved {args.samples} windows to {args.output}")


if __name__ == "__main__":
    main()
