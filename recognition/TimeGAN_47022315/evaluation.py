import argparse
import time
import numpy as np
import torch
from dataset import LOBSTERLevel10Dataset, load_dfs

WARMUP_SIZE = 512
NUM_SAMPLES = 3


def kl_divergence(validation_values, predicted_values, bins: int = 50) -> float:
    """Estimate KL(validation || predicted) in nats; samples must share a scale."""
    actual, predicted = (np.asarray(x, dtype=float).ravel() for x in (validation_values, predicted_values))
    if any(x.size == 0 or not np.isfinite(x).all() for x in (actual, predicted)):
        raise ValueError("Samples must be nonempty and finite.")
    edges = np.histogram_bin_edges(actual, bins=bins)
    edges[[0, -1]] = (-np.inf, np.inf)
    # Reference-only bins are stable across models; overflow bins retain generated tails.
    p, q = (np.histogram(x, bins=edges)[0].astype(float) + 1e-8 for x in (actual, predicted))
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def absolute_return_autocorrelation(returns) -> float:
    """Pearson lag-1 autocorrelation of absolute midpoint log returns."""
    values = np.abs(np.asarray(returns, dtype=float).ravel())
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Returns must contain at least two finite values.")
    if values[:-1].std() == 0 or values[1:].std() == 0:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def depth_ssim(actual_depth, predicted_depth) -> float:
    """Global SSIM for mean bid/ask volume profiles across the ten levels."""
    actual, predicted = (np.asarray(x, dtype=float) for x in (actual_depth, predicted_depth))
    data_range = max(actual.max(), predicted.max()) - min(actual.min(), predicted.min())
    if data_range == 0:
        return 1.0
    actual, predicted = actual / data_range, predicted / data_range
    c1, c2 = 0.01**2, 0.03**2
    mean_a, mean_p = actual.mean(), predicted.mean()
    var_a, var_p = actual.var(), predicted.var()
    covariance = np.mean((actual - mean_a) * (predicted - mean_p))
    return float(((2 * mean_a * mean_p + c1) * (2 * covariance + c2)) /
                 ((mean_a**2 + mean_p**2 + c1) * (var_a + var_p + c2)))


def reconstruct_book(features, mean, std, initial_midpoint):
    """Decode one continuous generated path into prices and displayed sizes."""
    physical = np.asarray(features) * np.asarray(std) + np.asarray(mean)
    with np.errstate(over="ignore", invalid="ignore"):
        midpoint = initial_midpoint * np.exp(np.cumsum(physical[:, 0]))
        lower, upper = np.log(1e-8), np.log(np.finfo(float).max) - 1
        def positive(values):
            return np.exp(np.clip(values, lower, upper))

        spread, ask_gaps, bid_gaps = positive(physical[:, 1]), positive(physical[:, 2:11]), positive(physical[:, 11:20])
        ask1, bid1 = midpoint + spread / 2, midpoint - spread / 2
        asks = np.column_stack((ask1, ask1[:, None] + np.cumsum(ask_gaps, axis=1)))
        bids = np.column_stack((bid1, bid1[:, None] - np.cumsum(bid_gaps, axis=1)))
        sizes = positive(physical[:, 20:40])
    # Dataset size columns alternate ask and bid at each price level.
    return asks, bids, sizes[:, 0::2], sizes[:, 1::2]


def structural_violation_rates(asks, bids, ask_sizes, bid_sizes):
    """Return per-book rates for invalid prices, ordering, spread, and depth."""
    arrays = tuple(np.asarray(x, dtype=float) for x in (asks, bids, ask_sizes, bid_sizes))
    nonfinite = ~np.logical_and.reduce([np.isfinite(x).all(axis=1) for x in arrays])
    crossed = (arrays[0][:, 0] <= arrays[1][:, 0]) & ~nonfinite
    nonmonotonic = (np.any(np.diff(arrays[0], axis=1) <= 0, axis=1) |
                    np.any(np.diff(arrays[1], axis=1) >= 0, axis=1)) & ~nonfinite
    nonpositive_price = (np.any(arrays[0] <= 0, axis=1) | np.any(arrays[1] <= 0, axis=1)) & ~nonfinite
    nonpositive_depth = (np.any(arrays[2] <= 0, axis=1) | np.any(arrays[3] <= 0, axis=1)) & ~nonfinite
    invalid = nonfinite | crossed | nonmonotonic | nonpositive_price | nonpositive_depth
    return {name: float(rate.mean()) for name, rate in {
        "any": invalid, "crossed/locked": crossed, "non-monotonic": nonmonotonic,
        "nonpositive price": nonpositive_price, "nonpositive depth": nonpositive_depth,
        "non-finite": nonfinite,
    }.items()}


def book_arrays(books):
    """Extract price and size ladders from a LOBSTER order-book frame."""
    def values(field):
        return np.column_stack([books[f"{side}{field}{level}"].to_numpy(dtype=float)
                                for level in range(1, 11) for side in ("ASK", "BID")])

    prices, sizes = values("p"), values("s")
    return prices[:, 0::2], prices[:, 1::2], sizes[:, 0::2], sizes[:, 1::2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TimeGAN on validation or test data.")
    parser.add_argument("--test", action="store_true", help="Evaluate the test split, warmed up on validation data.")
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    split_name = "test" if args.test else "validation"
    training_data = LOBSTERLevel10Dataset(split="train")
    validation_data = LOBSTERLevel10Dataset(split="validation")
    test_data = LOBSTERLevel10Dataset(split="test")
    split_data = {"validation": validation_data, "test": test_data}
    model = torch.load("timegan.pth", map_location="cpu", weights_only=False).eval()

    books, _ = load_dfs()
    transition_count = len(books) - 1
    train_end = round(training_data._train_ratio * transition_count)
    validation_end = round((training_data._train_ratio + training_data._VALIDATION_RATIO) * transition_count)
    predictions, generation_times, warmups = {}, {}, {}
    for name, evaluation_data in split_data.items():
        warmup_data = training_data if name == "validation" else validation_data
        actual = evaluation_data.data[:, 0].numpy()
        context = warmup_data.data[-WARMUP_SIZE:].to(next(model.parameters())).unsqueeze(0).expand(NUM_SAMPLES, -1, -1)
        warmup = context.size(1)

        torch.manual_seed(0)
        generation_start = time.perf_counter()
        with torch.inference_mode():
            noise = torch.randn(NUM_SAMPLES, warmup + len(actual), model._latent_dims,
                                dtype=context.dtype, device=context.device)
            synthetic = model.generator(noise)[:, warmup:]
            latent = model.supervisor(torch.cat((model.encode(context), synthetic), dim=1))
            predictions[name] = model.decode(latent)[:, warmup:].double().cpu().numpy()
        generation_times[name] = time.perf_counter() - generation_start
        warmups[name] = warmup

    evaluation_data = split_data[split_name]
    predicted = predictions[split_name]
    generation_seconds, warmup = generation_times[split_name], warmups[split_name]
    actual = evaluation_data.data[:, 0].numpy() * evaluation_data.std[0].item() + evaluation_data.mean[0].item()
    begin, end = (validation_end, transition_count) if args.test else (train_end, validation_end)
    actual_books = books.iloc[begin + 1:end + 1]

    actual_asks, actual_bids, actual_ask_sizes, actual_bid_sizes = book_arrays(actual_books)
    initial_midpoint = 0.5 * (books["ASKp1"].iloc[begin] + books["BIDp1"].iloc[begin])
    predicted_books = [reconstruct_book(sample, evaluation_data.mean.numpy(), evaluation_data.std.numpy(), initial_midpoint)
                       for sample in predicted]
    predicted_book = tuple(np.concatenate([book[index] for book in predicted_books], axis=0) for index in range(4))

    actual_spread = actual_books["ASKp1"].to_numpy() - actual_books["BIDp1"].to_numpy()
    physical_predicted = predicted * evaluation_data.std.numpy() + evaluation_data.mean.numpy()
    predicted_spread = np.exp(np.clip(physical_predicted[:, :, 1], np.log(1e-8), np.log(np.finfo(float).max) - 1))
    actual_depth = np.stack((actual_ask_sizes.mean(axis=0), actual_bid_sizes.mean(axis=0)))
    predicted_depth = np.stack((predicted_book[2].mean(axis=0), predicted_book[3].mean(axis=0)))

    print(f"Evaluating {split_name}: {len(actual)} events; warm-up: {'validation' if args.test else 'training'} ({warmup} events)")
    generated_events = predicted.shape[0] * predicted.shape[1]
    print(f"CPU generation latency for {predicted.shape[0]} paths (including warm-up): {generation_seconds:.3f}s; "
          f"{1_000 * generation_seconds / generated_events:.3f} ms/event; "
          f"throughput {generated_events / generation_seconds:.1f} events/s")
    print(f"Actual {split_name} |return| lag-1 autocorrelation: {absolute_return_autocorrelation(actual):.6f}")
    generated_autocorrelations = [
        absolute_return_autocorrelation(sample[:, 0] * evaluation_data.std[0].item() + evaluation_data.mean[0].item())
        for sample in predicted
    ]
    print(f"Generated {split_name} |return| lag-1 autocorrelation (mean of {len(predicted)} paths): "
          f"{np.mean(generated_autocorrelations):.6f}")
    print(f"Mid-price return KL({split_name} || predicted): {kl_divergence(actual, physical_predicted[:, :, 0]):.6f} nats (50 reference bins)")
    print(f"Spread KL({split_name} || predicted): {kl_divergence(actual_spread, predicted_spread):.6f} nats (50 reference bins)")
    print(f"Mean depth-profile SSIM: {depth_ssim(actual_depth, predicted_depth):.6f}")
    for name, sample in ((split_name.title(), (actual_asks, actual_bids, actual_ask_sizes, actual_bid_sizes)),
                         ("Generated", predicted_book)):
        rates = structural_violation_rates(*sample)
        print(f"{name} structural violations (% of books): " + ", ".join(
            f"{key} {100 * value:.4f}%" for key, value in rates.items()))
