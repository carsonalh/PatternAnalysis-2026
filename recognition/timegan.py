import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, Decoder, Generator, Discriminator


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
