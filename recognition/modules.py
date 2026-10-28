import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import FEATURE_DIMS


LATENT_DIMS = 128


class LSTMBlock(nn.Module):
    """Map (batch, time, features) sequences, retaining an output at every step."""
    def __init__(self, *, in_features: int, out_features: int, hidden_features: int, layers: int):
        super().__init__()
        self._lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_features,
            num_layers=layers,
            batch_first=True,
        )
        self._linear = nn.Linear(in_features=hidden_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._lstm(x)
        x = self._linear(x)
        return x


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._lstm = LSTMBlock(
            in_features=FEATURE_DIMS,
            hidden_features=LATENT_DIMS,
            out_features=LATENT_DIMS,
            layers=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._lstm(x)
        x = F.sigmoid(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._lstm = LSTMBlock(
            in_features=LATENT_DIMS,
            hidden_features=LATENT_DIMS,
            out_features=FEATURE_DIMS,
            layers=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._lstm(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_features: int = LATENT_DIMS, layers: int = 3) -> None:
        super().__init__()
        self._lstm = LSTMBlock(
            in_features=in_features,
            hidden_features=LATENT_DIMS,
            out_features=LATENT_DIMS,
            layers=layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._lstm(x)
        x = F.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._lstm = LSTMBlock(
            in_features=LATENT_DIMS,
            hidden_features=LATENT_DIMS,
            out_features=1,
            layers=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._lstm(x)
        x = F.sigmoid(x)
        return x


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
