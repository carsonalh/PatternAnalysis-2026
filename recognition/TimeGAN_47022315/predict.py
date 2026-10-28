import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import LOBSTERLevel10Dataset, load_dfs, positive_inverse

WINDOW_SIZE = 1_000
WARMUP_SIZE = 512


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.manual_seed(0)
    books, _ = load_dfs()
    log_midpoint = np.log((books["ASKp1"].to_numpy() + books["BIDp1"].to_numpy()) / 2)
    split = round((1 - LOBSTERLevel10Dataset._VALIDATION_RATIO) * len(books))
    training_returns = torch.from_numpy(np.diff(log_midpoint[:split]))
    mean, std = training_returns.mean(), training_returns.std().clamp_min(1e-8)
    model = torch.load("timegan.pth", map_location="cpu", weights_only=False).eval()
    with torch.inference_mode():
        training = LOBSTERLevel10Dataset(train=True)
        context = training.data[-WARMUP_SIZE:].to(next(model.parameters()))
        context = context.unsqueeze(0).expand(3, -1, -1)
        warmup = context.size(1)
        # Burn in the noise-only generator; real training history warms the supervisor and decoder.
        noise = torch.randn(3, warmup + WINDOW_SIZE, model._latent_dims, dtype=context.dtype)
        synthetic = model.generator(noise)[:, warmup:]
        latent = model.supervisor(torch.cat((model.encode(context), synthetic), dim=1))
        decoded = model.decode(latent)[:, warmup:].double()
        physical = decoded * training.std + training.mean
        physical[..., 1:] = positive_inverse(physical[..., 1:])
        assert torch.isfinite(physical).all() and (physical[..., 1:] > 0).all()
        normalized = decoded[..., 0]
        returns = normalized * std + mean
        # Each sample starts from the last training midpoint; these are independent paths.
        generated = (log_midpoint[split - 1] + returns.cumsum(1)).exp().numpy() / 10000
    actual = np.exp(log_midpoint[split:split + WINDOW_SIZE]) / 10000
    plt.figure(figsize=(10, 5))
    plt.plot(actual, color="black", linewidth=2, label="Validation")
    for i, prices in enumerate(generated):
        plt.plot(prices, alpha=0.75, label=f"TimeGAN sample {i + 1}")
    plt.xlabel("Validation event")
    plt.ylabel("Midpoint (USD)")
    plt.ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.title(f"Log/exp: AMZN midpoint: first {WINDOW_SIZE} validation events")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predictions_closeup.png", dpi=150)
