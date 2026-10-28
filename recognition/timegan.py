import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, Decoder, Generator, Discriminator


def standardize_returns(data):
    """Convert log midpoint levels to returns, then fit training-only scales."""
    features = data[1:].clone()
    features[:, 0] -= data[:-1, 0]
    mean, std = features.mean(0), features.std(0).clamp_min(1e-8)
    return (features - mean) / std, mean, std


def midpoint_prices(normalized, mean, std, initial_price):
    """Undo return scaling and anchor a price window in USD, outside the model."""
    returns = normalized[:, 1:, 0].double() * std[0] + mean[0]
    initial = torch.as_tensor(initial_price, dtype=returns.dtype, device=returns.device).reshape(-1, 1)
    initial = initial.expand(len(returns), 1)
    return torch.cat((initial, initial * returns.cumsum(1).exp()), dim=1)


class TimeGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.generator = Generator()
        self.supervisor = Generator(in_features=128, layers=2)
        self.discriminator = Discriminator()
        self._latent_dims = 128

    def forward(self, batch_size: int = 1) -> None:
        raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return h

    def generate(self, batch_size: int, window_size: int) -> torch.Tensor:
        parameter = next(self.parameters())
        noise = torch.randn(batch_size, window_size, self._latent_dims, dtype=parameter.dtype, device=parameter.device)
        e = self.generator(noise)
        h = self.supervisor(e)
        return h

    def discriminate(self, h: torch.Tensor) -> torch.Tensor:
        x = self.discriminator(h)
        return x

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        x = self.decoder(h)
        return x
