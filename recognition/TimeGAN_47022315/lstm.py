import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianLSTM(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 80)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.lstm(x)
        mu, sigma = self.linear(x).chunk(2, dim=-1)
        return mu, F.softplus(sigma).clamp_min(1e-6)


def sample_features(
    model: GaussianLSTM,
    warmup: torch.Tensor,
    num_events: int,
    num_samples: int,
) -> torch.Tensor:
    context = warmup.unsqueeze(0).expand(num_samples, -1, -1)
    state, sampled = None, []
    with torch.inference_mode():
        for _ in range(num_events):
            hidden, state = model.lstm(context, state)
            mean, raw_std = model.linear(hidden[:, -1:]).chunk(2, dim=-1)
            context = torch.distributions.Normal(mean, F.softplus(raw_std).clamp_min(1e-6)).sample()
            sampled.append(context[:, 0, :])
    return torch.stack(sampled, dim=1)


if __name__ == "__main__":
    from dataset import LOBSTERLevel10Dataset, WindowDataset, load_dfs
    from evaluation import (absolute_return_autocorrelation, book_arrays, depth_ssim,
                            kl_divergence, reconstruct_book, structural_violation_rates)
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    training_data = LOBSTERLevel10Dataset(split="train")
    validation_data = LOBSTERLevel10Dataset(split="validation")
    test_data = LOBSTERLevel10Dataset(split="test")
    dataset = WindowDataset(training_data.data, window_size=512, stride=4)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = GaussianLSTM().to(device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(5):
        losses = []

        for window, next in data_loader:
            window, next = window.to(device, dtype), next.to(device, dtype)
            shifted = torch.cat((window[:, 1:, :], next[:, None, :]), dim=1)

            mean, std = model(window)

            loss = F.gaussian_nll_loss(mean, shifted, std.square())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch + 1}; NLL Loss: {np.mean(losses):.3f}")

    window_size, num_samples = dataset.window_size, 3
    books, _ = load_dfs()
    log_midpoint = np.log((books["ASKp1"].to_numpy() + books["BIDp1"].to_numpy()) / 2)
    feature_mean = training_data.mean.numpy()
    feature_std = training_data.std.numpy()
    train_end = len(training_data)
    model.eval().cpu()
    predictions = {}
    for name, evaluation_data, warmup_data in (
        ("validation", validation_data, training_data.data),
        ("test", test_data, validation_data.data),
    ):
        warmup = warmup_data[-window_size:].cpu().to(dtype)
        torch.manual_seed(0)
        generation_start = time.perf_counter()
        predicted = sample_features(model, warmup, len(evaluation_data), num_samples)
        generation_seconds = time.perf_counter() - generation_start
        predictions[name] = predicted.double().numpy()
        raw_predictions = predictions[name] * feature_std + feature_mean
        actual_features = evaluation_data.data[:, :2].numpy() * feature_std[:2] + feature_mean[:2]
        split_begin = train_end if name == "validation" else train_end + len(validation_data)
        actual_book = book_arrays(books.iloc[split_begin + 1:split_begin + 1 + len(evaluation_data)])
        initial_midpoint = (books["ASKp1"].iloc[split_begin] + books["BIDp1"].iloc[split_begin]) / 2
        generated_books = [reconstruct_book(sample, feature_mean, feature_std, initial_midpoint)
                           for sample in predictions[name]]
        generated_book = tuple(np.concatenate([book[index] for book in generated_books], axis=0)
                               for index in range(4))
        actual_depth = np.stack((actual_book[2].mean(axis=0), actual_book[3].mean(axis=0)))
        generated_depth = np.stack((generated_book[2].mean(axis=0), generated_book[3].mean(axis=0)))
        generated_events = num_samples * len(evaluation_data)
        print(f"CPU generation latency for {num_samples} paths (including warm-up): {generation_seconds:.3f}s; "
              f"{1_000 * generation_seconds / generated_events:.3f} ms/event; "
              f"throughput {generated_events / generation_seconds:.1f} events/s")
        print(f"Mid-price return KL({name} || predicted): "
              f"{kl_divergence(actual_features[:, 0], raw_predictions[:, :, 0]):.6f} nats "
              f"(50 reference bins; {num_samples} paths)")
        actual_spread = np.exp(np.clip(actual_features[:, 1], np.log(1e-8), np.log(np.finfo(float).max) - 1))
        predicted_spread = np.exp(np.clip(raw_predictions[:, :, 1], np.log(1e-8), np.log(np.finfo(float).max) - 1))
        print(f"Spread KL({name} || predicted): "
              f"{kl_divergence(actual_spread, predicted_spread):.6f} nats "
              f"(50 reference bins; {num_samples} paths)")
        actual_returns = evaluation_data.data[:, 0].numpy() * feature_std[0] + feature_mean[0]
        generated_correlations = [absolute_return_autocorrelation(sample[:, 0]) for sample in raw_predictions]
        print(f"|return| lag-1 autocorrelation: actual {absolute_return_autocorrelation(actual_returns):.6f}; "
              f"generated mean {np.mean(generated_correlations):.6f}")
        print(f"Mean depth-profile SSIM: {depth_ssim(actual_depth, generated_depth):.6f}")
        for label, sample in (("Actual", actual_book), ("Generated", generated_book)):
            rates = structural_violation_rates(*sample)
            print(f"{label} structural violations (% of books): " + ", ".join(
                f"{key} {100 * value:.4f}%" for key, value in rates.items()))

    num_plot_events = min(1_500, len(validation_data))
    generated = np.exp(
        log_midpoint[train_end] + (predictions["validation"][:, :num_plot_events, 0] * feature_std[0]
                                   + feature_mean[0]).cumsum(1)
    ) / 10000
    actual = np.exp(log_midpoint[train_end + 1:train_end + 1 + num_plot_events]) / 10000
    warmup_history = np.exp(log_midpoint[train_end - window_size + 1:train_end + 1]) / 10000
    warmup_x = np.arange(-len(warmup_history), 0)
    validation_x = np.arange(len(actual))
    plt.figure(figsize=(10, 5))
    plt.axvspan(-window_size, 0, color="tab:blue", alpha=0.08, label="Warm-up period")
    plt.plot(warmup_x, warmup_history, color="tab:blue", alpha=0.7, label="Warm-up history")
    plt.plot(validation_x, actual, color="black", linewidth=2, label="Validation")
    for i, prices in enumerate(generated):
        plt.plot(validation_x, prices, alpha=0.75, label=f"GaussianLSTM sample {i + 1}")
    for boundary in range(window_size, generated.shape[1], window_size):
        plt.axvline(boundary - 0.5, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Event (warm-up history before validation)")
    plt.ylabel("Midpoint (USD)")
    plt.ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lstm_predictions.png", dpi=150)
