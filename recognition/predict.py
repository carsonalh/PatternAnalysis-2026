import numpy as np
import numpy.typing as npt
import torch

from dataset import features_to_order_book
from modules import LATENT_DIMS, TimeGAN

def wiener_process_noise(x: npt.NDArray) -> npt.NDArray:
    """
    Based off an ndarray of time samples x, compute the wiener process based on
    the gaps of the last dimension of x.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] == 0 or not np.isfinite(x).all():
        raise ValueError("Time samples must be finite and nonempty")
    z0 = np.zeros(shape=(*x.shape[:-1], 1))
    diffs = x[..., 1:] - x[..., :x.shape[-1] - 1]
    if (diffs < 0).any():
        raise ValueError("Time samples must be nondecreasing")
    z = np.random.normal(size=(*x.shape[:-1], x.shape[-1] - 1))
    # A Wiener increment has variance dt, so its standard deviation is sqrt(dt).
    z = np.concatenate((z0, np.sqrt(diffs) * z), axis=-1)
    z = np.cumsum(z, axis=-1)
    return z


def draw_noise(
    batch_size: int, sequence_length: int = 64, noise_dims: int = LATENT_DIMS,
    *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent Wiener paths (B, T, Z) and Gaussian initial seeds (B, Z).

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
    paths, seed = draw_noise(batch_size, sequence_length, model.generator.noise_dims,
                             device=parameter.device, dtype=parameter.dtype)
    return model.decoder(model.generator.sample(paths, seed))


def load_checkpoint(path, device: torch.device | str = "cpu") -> tuple[TimeGAN, dict]:
    """Load weights and training normalization for data-independent sampling."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TimeGAN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
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
        features_to_order_book(sequence, mean, std).to_csv(
            args.output / f"order_book_{index:04d}.csv", index=False)
    print(f"Saved {args.samples} windows to {args.output}")


if __name__ == "__main__":
    main()
