import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, Decoder, Generator, Discriminator
from dataset import FEATURE_DIMS


class TimeGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._encoder = Encoder()
        self._decoder = Decoder()
        self._generator = Generator()
        self._supervisor = Generator(in_features=128, layers=2)
        self._discriminator = Discriminator()
        self._latent_dims = 128

    def forward(self, batch_size: int = 1) -> None:
        raise NotImplementedError()

    # OPEN LOOP METHODS

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        return h

    def generate(self, batch_size: int, window_size: int) -> torch.Tensor:
        parameter = next(self.parameters())
        noise = torch.randn(batch_size, window_size, self._latent_dims, dtype=parameter.dtype, device=parameter.device)
        e = self._generator(noise)
        h = self._supervisor(e)
        return h

    def discriminate(self, h: torch.Tensor) -> torch.Tensor:
        x = self._discriminator(h)
        return x

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        x = self._decoder(h)
        return x

    """
    def auto_encode(self, x: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z, state = self._encoder(x, state)
        x = self._decoder(z)
        return x, state

    def discriminate_real(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z, _ = self._encoder(x)
            z = z[:, -1, :]
        x = self._discriminator(z)
        return x

    def discriminate_fake(self, batch_size: int, window_size: int) -> torch.Tensor:
        with torch.no_grad():
            z = self.generate(batch_size, window_size)
        x = self.discriminator(z)
        return x

    def generate(self, batch_size: int, window_size: int) -> torch.Tensor:
        noise = torch.randn(batch_size, window_size, 128)
        z = self._generator(noise)
        return z

    # TODO supervised loss (KL loss)
    """
