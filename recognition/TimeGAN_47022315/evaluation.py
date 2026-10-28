import numpy as np
import torch
from dataset import LOBSTERLevel10Dataset


def kl_divergence(validation_returns, predicted_returns, bins: int = 50) -> float:
    """Estimate KL(validation || predicted) in nats; returns must share a scale."""
    actual, predicted = (np.asarray(x, dtype=float).ravel() for x in (validation_returns, predicted_returns))
    if any(x.size == 0 or not np.isfinite(x).all() for x in (actual, predicted)):
        raise ValueError("Returns must be nonempty and finite.")
    edges = np.histogram_bin_edges(np.concatenate((actual, predicted)), bins=bins)
    # Shared bins retain both tails; a small pseudocount avoids zero probabilities.
    p, q = (np.histogram(x, bins=edges)[0].astype(float) + 1e-8 for x in (actual, predicted))
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.manual_seed(0)
    # Column 0 contains midpoint log returns; both use the same training scaler.
    validation = LOBSTERLevel10Dataset(train=False).data[:, 0].numpy()
    model = torch.load("timegan.pth", map_location="cpu", weights_only=False).eval()
    with torch.inference_mode():
        predicted = torch.cat([
            model.decode(model.generate(batch_size=16, window_size=512))[:, :, 0].flatten()
            for _ in range(0, len(validation), 16 * 512)
        ])[:len(validation)].double().numpy()
    print(f"KL(validation || predicted): {kl_divergence(validation, predicted):.6f} nats (50 bins)")
