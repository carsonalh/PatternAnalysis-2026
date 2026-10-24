from typing import Callable, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, *, in_features: int, out_features: int, hidden_features: int, layers: int, create_activation: Callable[[], nn.Module]):
        module_layers = []
        for i in range(layers):
            feat_in = hidden_features
            feat_out = hidden_features
            if i == 0:
                feat_in = in_features
            if i == layers - 1:
                feat_out = out_features

            module_layers.append(nn.Linear(in_features=feat_in, out_features=feat_out))
            if i < layers - 1:
                module_layers.append(create_activation())

        self._mlp = nn.Sequential(*module_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mlp(x)
        return x


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO encoder really should be an RNN
        self._mlp = MLP(
            in_features=20,
            hidden_features=128,
            out_features=128,
            layers=3,
            create_activation=lambda: nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mlp(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._mlp = MLP(
            in_features=128,
            hidden_features=128,
            out_features=20,
            layers=3,
            create_activation=lambda: nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mlp(x)
        return x


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._mlp = MLP(
            in_features=128,
            hidden_features=128,
            out_features=128,
            layers=3,
            create_activation=lambda: nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mlp(x)
        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._mlp = MLP(
            in_features=128,
            hidden_features=128,
            out_features=1,
            layers=3,
            create_activation=lambda: nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mlp(x)
        return x


class TimeGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._encoder = Encoder()
