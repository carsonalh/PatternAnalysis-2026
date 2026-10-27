"""Run the LOB experiment: python main.py (outputs in runs/timegan_lob)."""
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import LOBSTERLevel10Dataset, download_zip
from evaluation import evaluate
from predict import predict
from training import fit


def plot(train, validation, predictions, output):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), layout="constrained")
    for ax, (name, values) in zip(axes[:2], predictions.items()):
        ax.plot(np.exp(validation[:, 0].numpy()) / 10000, color="black", label="Validation")
        ax.plot(np.exp(values[:, 0]) / 10000, label=name)
        ax.set(ylabel="Midpoint ($)", xlabel="Events after training"); ax.legend()
    indices = np.r_[np.arange(39, 20, -2), np.arange(20, 40, 2)]
    for ax, values, title in zip(axes[2:], (validation.numpy(), predictions["timegan"]), ("Validation", "TimeGAN")):
        image = ax.imshow(values[:, indices].T, aspect="auto", vmin=float(train[:, 20:].min()), vmax=float(train[:, 20:].max()))
        ax.set(title=title, ylabel="BID10 … BID1 | ASK1 … ASK10", xlabel="Events after training"); fig.colorbar(image, ax=ax, label="log1p size")
    fig.savefig(output / "validation.png", dpi=150)


def main():
    train, validation = (LOBSTERLevel10Dataset(train=s).data for s in (True, False))
    model, normalized, center, scale, history = fit(train)
    output = Path(__file__).resolve().parent / "runs/timegan_lob"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "center": center, "scale": scale,
                "history": history, "seed": 20261025, "steps": (1000, 1000, 5000),
                "data_sha256": hashlib.sha256(download_zip().read_bytes()).hexdigest()}, output / "model.pt")
    predictions = predict(model, normalized, len(validation), center, scale, train)
    metrics = {name: evaluate(train, validation, values) for name, values in predictions.items()}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(output / "predictions.npz", validation=validation.numpy(), **predictions)
    plot(train, validation, predictions, output)
    print(json.dumps({name: {k: v for k, v in result.items() if k != "feature_kl"} for name, result in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
