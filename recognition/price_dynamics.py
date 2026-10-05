"""Conditional price innovations with drift calibrated on training data.

This is a hybrid forecast model: TimeGAN generates book structure, while a
volatility head generates midpoint returns. Its conditional expected price
change is fixed to empirical training drift, so decoder-level attraction
cannot cause a price reset. It does not learn directional price forecasts.
"""

import torch
from torch import nn


class PriceVolatility(nn.Sequential):
    """Predict positive return SD, in training-standardized return units."""

    def __init__(self, latent_dims=128):
        super().__init__(nn.Linear(latent_dims, 64), nn.SiLU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, latent):
        return super().forward(latent) + 1e-4


def price_innovations(volatility, past_latents, inputs, calibration):
    """Standardized log returns; volatility is measurable from the past."""
    mean, std = calibration["return_mean"], calibration["return_std"]
    sigma = volatility(past_latents).squeeze(-1) * std
    shock = inputs * (calibration["sequence_length"] - 1) ** 0.5
    returns = calibration["log_expected_price_factor"] - 0.5 * sigma.square() + sigma * shock
    return (returns - mean) / std


@torch.inference_mode()
def innovation_completion(model, volatility, contexts, inputs, calibration):
    """Generate from the real starting prefix, then solely from generated past.

    The volatility for each block is estimated before its noise is seen.
    The lognormal correction fixes conditional arithmetic-price drift; return
    normalization and LOBSTER quote rounding are handled separately.
    """
    block = model.context_horizon
    length = contexts.shape[1] - block
    if (length < 1 or inputs.shape[0] != contexts.shape[0]
            or getattr(model, "noise_kind", None) != "wiener_increments"):
        raise ValueError("Need a real prefix and matching noise batch")
    history = contexts[:, -length:]
    outputs = []
    for start in range(0, inputs.shape[1] - 1, block):
        context = model.embedder(history)[:, -1]
        noise = inputs[:, start + 1:start + block + 1]
        future = model.decoder(model.generator.continue_from(context, noise))
        future[..., 0] = price_innovations(volatility, context[:, None], noise[..., 0], calibration)
        outputs.append(future.cpu())
        history = torch.cat((history, future), dim=1)[:, -length:]
    return torch.cat(outputs, dim=1)


def load_price_volatility(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    head = PriceVolatility(checkpoint["latent_dims"])
    head.load_state_dict(checkpoint["state_dict"])
    return head.eval(), checkpoint["calibration"]
