import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineLSTM(nn.Module):
    def __init__(self, *, hidden_size: int = 1024, num_layers: int = 3) -> None:
        super().__init__()
        self._lstm = nn.LSTM(
            batch_first=True,
            input_size=20,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self._linear = nn.Linear(hidden_size, 40)

    def forward(self, x: torch.Tensor, hx: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not hx:
            # Hidden and cell states use (layers, batch, hidden), even with batch_first=True.
            state_shape = (self._lstm.num_layers, x.shape[0], self._lstm.hidden_size)
            h = x.new_zeros(state_shape)
            c = x.new_zeros(state_shape)
            hx = (h, c)
        x, (h, c) = self._lstm(x, hx=hx)
        x = self._linear(x)
        mu = x[..., :20]
        sigma = 1e-6 + F.softplus(x[..., 20:])
        x = torch.distributions.Normal(mu, sigma).rsample()
        return x, (mu, sigma), (h, c)

class SplitBaselineLSTM(BaselineLSTM):
    """Gaussian log returns with fixed training volatility; freeze the mean head for zero drift."""
    def __init__(self, midpoint_scale: float, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("midpoint_scale", torch.tensor(midpoint_scale))
        self._midpoint = nn.Linear(self._lstm.hidden_size, 1, bias=False)
        nn.init.zeros_(self._midpoint.weight)

    def forward(self, x, hx=None):
        hidden, state = self._lstm(x, hx)
        mu = self._midpoint(hidden)
        other = self._linear(hidden)[..., 1:20]
        midpoint = x[..., :1] + self.midpoint_scale * (mu + torch.randn_like(mu))
        return torch.cat((midpoint, other), dim=-1), (mu, other), state

    def loss(self, x, target):
        _, (mu, other), _ = self(x)
        returns = (target[..., :1] - x[..., :1]) / self.midpoint_scale
        return F.gaussian_nll_loss(mu, returns, torch.ones_like(mu)) + F.mse_loss(other, target[..., 1:])

class GaussianReturnLSTM(nn.Module):
    """Predict a normal distribution for the next standardized log return."""
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x, state=None):
        hidden, state = self.lstm(x, state)
        mu, raw_sigma = self.head(hidden[:, -1]).chunk(2, dim=-1)
        return mu, F.softplus(raw_sigma) + 1e-3, state
