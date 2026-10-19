"""Small next-event book predictor; run with `uv run python simple_gru.py`."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import LOBSTERLevel10Dataset, WindowDataset

class SimpleGRU(nn.Module):
    def __init__(self, features=39, hidden=64):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(features, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, features)  # Standardized targets can be negative.
    def forward(self, x, state=None):
        x, state = self.gru(self.input(x), state)
        return self.output(x), state

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, validation = LOBSTERLevel10Dataset(), LOBSTERLevel10Dataset(train=False)
    model = SimpleGRU(features=train.FEATURE_DIMS - 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(6):  # Epoch zero measures performance before any updates.
        for split, data in (("train", train), ("validation", validation)):
            model.train(epoch > 0 and split == "train")
            loader = DataLoader(WindowDataset(data.features[:, 1:], 33, stride=32),
                                batch_size=128, shuffle=model.training)
            total, count = 0.0, 0
            with torch.set_grad_enabled(model.training):
                for batch in loader:
                    x = batch.to(device)
                    loss = (model(x[:, :-1])[0] - x[:, 1:]).square().mean()
                    if model.training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total, count = total + loss.item() * x[:, 1:].numel(), count + x[:, 1:].numel()
            print(f"Epoch {epoch}: {split} standardized MSE = {total / count:.6f}", flush=True)
    with torch.no_grad():
        generated, state = [train.features[-32:, 1:].unsqueeze(0).to(device)], None
        for _ in range(64):
            prediction, state = model(generated[-1], state)
            generated.append(prediction[:, -1:])
        # Gaussian log returns, fitted via training normalization, give log-normal midpoints.
        features = torch.cat((torch.randn(64, 1), torch.cat(generated[1:], dim=1)[0].cpu()), dim=1)
    train.feature_sequence_to_df(features).to_csv("data/simple_gru_sample.csv", index=False)

if __name__ == "__main__":
    main()
