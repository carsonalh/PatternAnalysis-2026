import torch
from torch import nn
from dataset import FEATURE_DIMS

LATENT_DIMS = 128


class Embedder(nn.Module):
    """A causal history summary at every event in a (B, T, F) window."""

    def __init__(self, feature_dims: int = FEATURE_DIMS, latent_dims: int = LATENT_DIMS):
        super().__init__()
        self.recurrent = nn.GRU(feature_dims, latent_dims, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("The embedder expects (batch, time, features) windows")
        # Omitting the initial state resets history for every independent window.
        latents, _ = self.recurrent(x)
        return latents


class Decoder(nn.Sequential):
    """Recover standardized book features independently at each event."""

    def __init__(self, feature_dims: int = FEATURE_DIMS, latent_dims: int = LATENT_DIMS):
        super().__init__(
            nn.Linear(latent_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dims),
        )


class Generator(nn.Module):
    """The recurrent state is the latent, shared by both generation modes."""

    def __init__(self, noise_dims: int = LATENT_DIMS, latent_dims: int = LATENT_DIMS):
        super().__init__()
        self.noise_dims = noise_dims
        self.initial = nn.Sequential(nn.Linear(noise_dims, latent_dims), nn.Tanh())
        self.cell = nn.GRUCell(noise_dims, latent_dims)

    def sample(self, noise: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        """Generate (B, T, H) latents using only Wiener paths and initial noise."""
        if (noise.ndim != 3 or noise.shape[1] < 1 or noise.shape[2] != self.noise_dims
                or seed.shape != (noise.shape[0], self.noise_dims)):
            raise ValueError("Expected (B, T, noise_dims) paths and (B, noise_dims) seeds")
        h = self.initial(seed)
        outputs = [h]
        for t in range(1, noise.shape[1]):
            # Keep the feedback attached: adversarial gradients span the window.
            h = self.cell(noise[:, t], h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)

    def teacher_forced(self, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict h[t] from the real h[t-1], returning (B, T-1, H)."""
        if (latents.ndim != 3 or noise.ndim != 3 or latents.shape[:2] != noise.shape[:2]
                or latents.shape[1] < 2 or noise.shape[2] != self.noise_dims):
            raise ValueError("Latents and noise need matching windows of at least two events")
        batch, time, dims = latents.shape
        # With no separate memory state, the one-step predictions can be batched.
        predictions = self.cell(
            noise[:, 1:].reshape(-1, self.noise_dims),
            latents[:, :-1].reshape(-1, dims),
        )
        return predictions.reshape(batch, time - 1, dims)


class Discriminator(nn.Module):
    """Score latent windows with one real/fake logit per event."""

    def __init__(self, latent_dims: int = LATENT_DIMS):
        super().__init__()
        self.recurrent = nn.GRU(latent_dims, 64, batch_first=True, bidirectional=True)
        self.output = nn.Linear(128, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        states, _ = self.recurrent(latents)
        return self.output(states)


class TimeGAN(nn.Module):
    """Four temporal networks; no static covariates or separate supervisor."""

    def __init__(self, feature_dims: int = FEATURE_DIMS, latent_dims: int = LATENT_DIMS,
                 noise_dims: int = LATENT_DIMS):
        super().__init__()
        self.model_config = dict(feature_dims=feature_dims, latent_dims=latent_dims,
                                 noise_dims=noise_dims)
        self.embedder = Embedder(feature_dims, latent_dims)
        self.decoder = Decoder(feature_dims, latent_dims)
        self.generator = Generator(noise_dims, latent_dims)
        self.discriminator = Discriminator(latent_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.embedder(x))
