import torch
import torch.nn as nn


class ECGLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
        )
        self._linear = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._lstm(x)
        x = self._linear(x)
        return x
