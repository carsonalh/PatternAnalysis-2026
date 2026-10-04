import torch
from torch import nn

class Embedder(nn.Module):
    def __init__(self, feature_dims: int = 2, latent_dims: int = 128, hidden_dims: int = 256):
        super().__init__()
        self.input_layer = nn.Linear(feature_dims, latent_dims)
        self.hidden_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, latent_dims),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x += h
        x = self.hidden_layers(x)
        return x


FEATURE_DIMS = 2
LATENT_DIMS = 128
_DECODER_HIDDEN_DIMS = 256


Decoder = nn.Sequential(
    nn.Linear(LATENT_DIMS, _DECODER_HIDDEN_DIMS),
    nn.ReLU(),
    nn.Linear(_DECODER_HIDDEN_DIMS, _DECODER_HIDDEN_DIMS),
    nn.ReLU(),
    nn.Linear(_DECODER_HIDDEN_DIMS, FEATURE_DIMS),
)
