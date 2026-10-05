"""Held-out feasibility checks in feature space and after book projection."""

import torch
from torch import nn
from torch.nn import functional as F

from dataset import LOBSTERLevel10Dataset, WindowDataset
from modules import TimeGAN
from predict import generate_wiener_paths, generate_features


def _correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    scale = (x.square().mean(dim=0) * y.square().mean(dim=0)).sqrt().clamp_min(1e-12)
    return (x * y).mean(dim=0) / scale


def sequence_statistics(windows: torch.Tensor) -> dict[str, torch.Tensor]:
    """Never join the last event of one window to the first of another."""
    windows = windows.detach().cpu().double()
    flat = windows.flatten(0, 1)
    centered = flat - flat.mean(dim=0)
    std = flat.std(dim=0, correction=0)
    correlations = (centered.T @ centered / len(flat)) / torch.outer(std, std).clamp_min(1e-12)
    changes = windows[:, 1:] - windows[:, :-1]
    return {
        "mean": flat.mean(dim=0),
        "std": std,
        "change_mean": changes.mean(dim=(0, 1)),
        "change_std": changes.std(dim=(0, 1), correction=0),
        "lag1_correlation": _correlation(windows[:, :-1].flatten(0, 1), windows[:, 1:].flatten(0, 1)),
        "feature_correlation": correlations,
        "window_mean_std": windows.mean(dim=1).std(dim=0, correction=0),
    }


def compare_sequences(real: torch.Tensor, fake: torch.Tensor) -> dict:
    real_stats, fake_stats = sequence_statistics(real), sequence_statistics(fake)
    return {
        "absolute_errors": {key: (real_stats[key] - fake_stats[key]).abs().mean().item()
                            for key in real_stats},
        "real": {key: value.tolist() for key, value in real_stats.items()},
        "generated": {key: value.tolist() for key, value in fake_stats.items()},
    }


def book_statistics(books: torch.Tensor) -> dict[str, float]:
    """Summarize projected (B, T, 40) LOBSTER books; prices are in dollars."""
    levels = books.double().reshape(*books.shape[:2], 10, 4)
    midpoint = (levels[..., 0, 0] + levels[..., 0, 2]) / 20_000
    spread = (levels[..., 0, 0] - levels[..., 0, 2]) / 10_000
    ask_depth = levels[..., 1].sum(dim=-1)
    bid_depth = levels[..., 3].sum(dim=-1)
    changes = midpoint[:, 1:] - midpoint[:, :-1]
    return {
        "spread_mean_dollars": spread.mean().item(),
        "spread_std_dollars": spread.std(correction=0).item(),
        "ask_depth_mean_shares": ask_depth.mean().item(),
        "ask_depth_std_shares": ask_depth.std(correction=0).item(),
        "bid_depth_mean_shares": bid_depth.mean().item(),
        "bid_depth_std_shares": bid_depth.std(correction=0).item(),
        "midpoint_change_mean_dollars": changes.mean().item(),
        "midpoint_change_std_dollars": changes.std(correction=0).item(),
        "midpoint_lag1_correlation": _correlation(midpoint[:, :-1].reshape(-1, 1),
                                                  midpoint[:, 1:].reshape(-1, 1)).item(),
    }


class _NextStepPredictor(nn.Module):
    def __init__(self, feature_dims: int):
        super().__init__()
        self.recurrent = nn.GRU(feature_dims, 32, batch_first=True)
        self.output = nn.Linear(32, feature_dims)

    def forward(self, x):
        states, _ = self.recurrent(x)
        return self.output(states)


def next_step_mae(training: torch.Tensor, validation: torch.Tensor,
                  steps: int = 200, seed: int = 0) -> float:
    """Train a small predictor, then score standardized features on real windows."""
    training, validation = training.detach().cpu(), validation.detach().cpu()
    # Real and synthetic baselines get the same initialization and update budget.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        predictor = _NextStepPredictor(training.shape[-1])
        optimizer = torch.optim.Adam(predictor.parameters(), lr=3e-4)
        for _ in range(steps):
            batch = training[torch.randint(len(training), (min(64, len(training)),))]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(predictor(batch[:, :-1]), batch[:, 1:])
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            errors = [F.l1_loss(predictor(batch[:, :-1]), batch[:, 1:], reduction="sum")
                      for batch in validation.split(64)]
        return torch.stack(errors).sum().item() / validation[:, 1:].numel()


def _sample_windows(features: torch.Tensor, sequence_length: int, count: int) -> torch.Tensor:
    windows = WindowDataset(features, sequence_length, stride=sequence_length)
    if len(windows) == 0:
        raise ValueError("Evaluation splits need at least one complete window")
    # Cover each split without overlapping evaluation windows or random leakage.
    indices = torch.linspace(0, len(windows) - 1, min(count, len(windows))).round().long()
    return torch.stack([windows[index.item()] for index in indices])


@torch.no_grad()
def _validation_losses(model: TimeGAN, real: torch.Tensor) -> dict[str, float]:
    device = next(model.parameters()).device
    x = real.to(device)
    latents = model.embedder(x)
    noise, _ = generate_wiener_paths(len(x), x.shape[1], model.generator.noise_dims, device=device)
    return {
        "reconstruction_mse": F.mse_loss(model.decoder(latents), x).item(),
        "transition_mse": F.mse_loss(model.generator.teacher_forced(latents, noise), latents[:, 1:]).item(),
    }


def evaluate_timegan(model: TimeGAN, training: LOBSTERLevel10Dataset,
                     validation: LOBSTERLevel10Dataset, sequence_length: int = 64,
                     samples: int = 256, predictor_steps: int = 200,
                     seed: int = 0) -> tuple[dict, torch.Tensor]:
    real_training = _sample_windows(training.features, sequence_length, samples)
    real_validation = _sample_windows(validation.features, sequence_length, samples)
    model.eval()
    generated = generate_features(model, len(real_training), sequence_length).cpu()

    def project(windows):
        return torch.stack([torch.from_numpy(training.feature_sequence_to_df(window).to_numpy(copy=True))
                            for window in windows])

    report = {
        "feature_names": list(training.feature_names),
        "training_windows": len(real_training),
        "validation_windows": len(real_validation),
        "validation_losses": _validation_losses(model, real_validation),
        "unprojected_feature_statistics": compare_sequences(real_validation, generated),
        "projected_book_statistics": {
            "real": book_statistics(project(real_validation)),
            "generated": book_statistics(project(generated)),
        },
        "predictive_mae_standardized": {
            "train_real_test_real": next_step_mae(real_training, real_validation, predictor_steps, seed),
            "train_synthetic_test_real": next_step_mae(generated, real_validation, predictor_steps, seed),
        },
    }
    return report, generated
