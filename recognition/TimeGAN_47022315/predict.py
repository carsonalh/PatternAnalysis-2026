import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import LOBSTERLevel10Dataset, load_dfs, positive_inverse
from evaluation import book_arrays, reconstruct_book

WINDOW_SIZE = 1_500
WARMUP_SIZE = 512
NUM_SAMPLES = 3


def plot_order_book_comparison(actual_book, predicted_book, filename: str, sample_number: int) -> None:
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable

    actual_asks, actual_bids, actual_ask_sizes, actual_bid_sizes = actual_book
    predicted_asks, predicted_bids, predicted_ask_sizes, predicted_bid_sizes = predicted_book
    panels = (
        (actual_asks, actual_bids, 10_000, "Validation prices", "Price (USD)"),
        (predicted_asks, predicted_bids, 10_000, f"Generated sample {sample_number} prices", "Price (USD)"),
        (actual_ask_sizes, actual_bid_sizes, 1, "Validation displayed sizes", "Shares"),
        (predicted_ask_sizes, predicted_bid_sizes, 1, f"Generated sample {sample_number} displayed sizes", "Shares"),
    )
    figure = plt.figure(figsize=(15, 9), layout="constrained")
    grid = figure.add_gridspec(
        3, 3, height_ratios=(0.18, 1, 1), width_ratios=(1, 1, 0.045), wspace=0.18, hspace=0.22,
    )
    header_axis = figure.add_subplot(grid[0, :2])
    header_axis.axis("off")
    header_axis.text(0.5, 0.78, "Reconstructed ten-level order books", ha="center", va="center", fontsize=16)
    header_axis.legend(
        handles=(Line2D([], [], color="black", label="Ask"),
                 Line2D([], [], color="black", linestyle="--", label="Bid")),
        loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False,
    )
    axes = np.array([
        [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
        [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])],
    ])
    color_axis = figure.add_subplot(grid[1:, 2])
    colors, levels = plt.get_cmap("viridis"), Normalize(1, 10)
    events = np.arange(len(actual_asks))
    for axis, (asks, bids, scale, title, ylabel) in zip(axes.flat, panels):
        for level in range(10):
            color = colors(levels(level + 1))
            axis.plot(events, asks[:, level] / scale, color=color, linewidth=1)
            axis.plot(events, bids[:, level] / scale, color=color, linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in axes[1]:
        axis.set_xlabel("Validation event")
    figure.colorbar(
        ScalarMappable(norm=levels, cmap=colors), cax=color_axis,
        ticks=np.arange(1, 11), label="Book level (1 = best, 10 = deepest)",
    )
    figure.savefig(filename, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.manual_seed(0)
    books, _ = load_dfs()
    log_midpoint = np.log((books["ASKp1"].to_numpy() + books["BIDp1"].to_numpy()) / 2)
    split = round((1 - LOBSTERLevel10Dataset._VALIDATION_RATIO - LOBSTERLevel10Dataset._TEST_RATIO) * (len(books) - 1))
    window_size = min(WINDOW_SIZE, len(books) - split - 1)
    training = LOBSTERLevel10Dataset(split="train")
    mean, std = training.mean[0], training.std[0]
    model = torch.load("timegan.pth", map_location="cpu", weights_only=False).eval()
    with torch.inference_mode():
        context = training.data[-WARMUP_SIZE:].to(next(model.parameters()))
        context = context.unsqueeze(0).expand(NUM_SAMPLES, -1, -1)
        warmup = context.size(1)
        # Burn in the noise-only generator; real training history warms the supervisor and decoder.
        noise = torch.randn(NUM_SAMPLES, warmup + window_size, model._latent_dims, dtype=context.dtype)
        synthetic = model.generator(noise)[:, warmup:]
        latent = model.supervisor(torch.cat((model.encode(context), synthetic), dim=1))
        decoded = model.decode(latent)[:, warmup:].double()
        physical = decoded * training.std + training.mean
        physical[..., 1:] = positive_inverse(physical[..., 1:])
        assert torch.isfinite(physical).all() and (physical[..., 1:] > 0).all()
        normalized = decoded[..., 0]
        returns = normalized * std + mean
        # Each sample starts from the final training book; paths are independent.
        generated = (log_midpoint[split] + returns.cumsum(1)).exp().numpy() / 10000
    actual = np.exp(log_midpoint[split + 1:split + 1 + window_size]) / 10000
    warmup_history = np.exp(log_midpoint[split - WARMUP_SIZE + 1:split + 1]) / 10000
    warmup_x = np.arange(-len(warmup_history) + 1, 1)
    validation_x = np.arange(1, len(actual) + 1)
    plt.figure(figsize=(10, 5))
    plt.axvspan(warmup_x[0], 0, color="tab:blue", alpha=0.08, label="Warm-up period")
    plt.plot(warmup_x, warmup_history, color="tab:blue", alpha=0.7, label="Warm-up history")
    plt.plot(validation_x, actual, color="black", linewidth=2, label="Validation")
    for i, prices in enumerate(generated):
        plt.plot(validation_x, prices, alpha=0.75, label=f"TimeGAN sample {i + 1}")
    plt.xlabel("Event relative to the final training book")
    plt.ylabel("Midpoint (USD)")
    plt.ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.title(f"Log/exp: AMZN midpoint: first {len(actual)} validation events")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predictions_closeup.png", dpi=150)
    actual_books = books.iloc[split + 1:split + 1 + len(actual)]
    actual_book = book_arrays(actual_books)
    initial_midpoint = 0.5 * (books["ASKp1"].iloc[split] + books["BIDp1"].iloc[split])
    for sample_number, features in enumerate(decoded, start=1):
        predicted_book = reconstruct_book(
            features.cpu().numpy(), training.mean.numpy(), training.std.numpy(), initial_midpoint,
        )
        filename = f"predicted_order_book_sample_{sample_number}.png"
        plot_order_book_comparison(actual_book, predicted_book, filename, sample_number)
        print(f"Saved reconstructed order-book visualization to {filename}")
