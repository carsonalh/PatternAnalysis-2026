"""Full-history warm-state continuation; the lognormal midpoint is a labeled baseline."""
import numpy as np
import torch
from modules import Generator


@torch.inference_mode()
def predict(model, normalized_train, horizon, center, scale, raw_train, seed=20261025):
    """Condition recurrent states on training only; TimeGAN is not a fitted forecaster."""
    # CPU accepts the complete day as one window; cuDNN rejects this sequence length.
    parameter = next(model.cpu().parameters())
    history = torch.as_tensor(normalized_train, device=parameter.device, dtype=parameter.dtype)[None]
    center, scale = (torch.as_tensor(x, device=parameter.device, dtype=parameter.dtype) for x in (center, scale))
    def step(module, values, state=None):
        block = module._lstm
        hidden, state = block._lstm(values, state)
        output = block._linear(hidden)
        return (output.sigmoid() if isinstance(module, Generator) else output), state
    model.eval()
    encoded = model.encode(history)  # The entire training split is one inference window.
    _, supervisor_state = step(model._supervisor, encoded)
    _, decoder_state = step(model._decoder, encoded)
    rng = torch.Generator(device=history.device).manual_seed(seed)
    noise = torch.randn(1, horizon, model._latent_dims, device=history.device, dtype=history.dtype, generator=rng)
    latent, _ = step(model._supervisor, model._generator(noise), supervisor_state)
    output, _ = step(model._decoder, latent, decoder_state)
    output = (output[0] * scale + center).cpu().numpy().astype(np.float64)
    train_midpoint = torch.as_tensor(raw_train).cpu().numpy()[:, 0]
    output[:, 0] = train_midpoint[-1] + output[:, 0].cumsum()
    hybrid, returns = output.copy(), np.diff(train_midpoint)
    hybrid[:, 0] = train_midpoint[-1] + np.random.default_rng(seed).normal(returns.mean(), returns.std(), horizon).cumsum()
    return {"timegan": output, "lognormal_midpoint": hybrid}
