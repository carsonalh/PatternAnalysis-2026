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


if __name__ == "__main__":
    from dataset import LOBSTERLevel10Dataset, WindowDataset, load_dfs
    from evaluation import kl_divergence
    from torch.utils.data import DataLoader
    import numpy as np
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    dataset = WindowDataset(LOBSTERLevel10Dataset(train=True).data, window_size=256)
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

    window_size, num_windows, num_samples = dataset.window_size, 3, 3
    books, _ = load_dfs()
    log_midpoint = np.log((books["ASKp1"].to_numpy() + books["BIDp1"].to_numpy()) / 2)
    split = round((1 - LOBSTERLevel10Dataset._VALIDATION_RATIO) * len(books))
    training_returns = torch.from_numpy(np.diff(log_midpoint[:split]))
    return_mean, return_std = training_returns.mean(), training_returns.std().clamp_min(1e-8)
    context = dataset.data[-window_size:].unsqueeze(0).expand(num_samples, -1, -1).to(device, dtype)
    model.eval()
    with torch.inference_mode():
        # Warm up on training history, then feed sampled features back into the LSTM.
        state, sampled = None, []
        for _ in range(num_windows * window_size):
            hidden, state = model.lstm(context, state)
            mean, raw_std = model.linear(hidden[:, -1:]).chunk(2, dim=-1)
            context = torch.distributions.Normal(mean, F.softplus(raw_std).clamp_min(1e-6)).sample()
            sampled.append(context[:, 0, 0])
        returns = torch.stack(sampled, dim=1).cpu().double() * return_std + return_mean
        generated = (log_midpoint[split - 1] + returns.cumsum(1)).exp().numpy() / 10000
    actual_returns = np.diff(log_midpoint[split - 1:split + returns.size(1)])
    print(f"KL(validation || predicted): {kl_divergence(actual_returns, returns.numpy()):.6f} nats "
          f"(50 bins; first {returns.size(1)} validation events)")
    actual = np.exp(log_midpoint[split:split + generated.shape[1]]) / 10000
    plt.figure(figsize=(10, 5))
    plt.plot(actual, color="black", linewidth=2, label="Validation")
    for i, prices in enumerate(generated):
        plt.plot(prices, alpha=0.75, label=f"GaussianLSTM sample {i + 1}")
    for boundary in range(window_size, generated.shape[1], window_size):
        plt.axvline(boundary - 0.5, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Validation event")
    plt.ylabel("Midpoint (USD)")
    plt.ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.legend()
    plt.tight_layout()
    plt.savefig("lstm_predictions.png", dpi=150)
