"""Compare real and generated features, reconstructed books, and predictions."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from dataset import LOBSTERLevel10Dataset, WindowDataset
from modules import TimeGAN
from predict import generate_wiener_paths, generate_features


@dataclass(frozen=True)
class SequenceStatistics:
    """Per-feature tensors and a correlation matrix; comparison errors are scalars."""

    mean: torch.Tensor
    std: torch.Tensor
    change_mean: torch.Tensor
    change_std: torch.Tensor
    lag1_correlation: torch.Tensor
    feature_correlation: torch.Tensor
    window_mean_std: torch.Tensor  # Variation between window averages.


@dataclass(frozen=True)
class SequenceComparison:
    real: SequenceStatistics
    generated: SequenceStatistics
    absolute_errors: SequenceStatistics


@dataclass(frozen=True)
class BookStatistics:
    spread_mean_dollars: float
    spread_std_dollars: float
    ask_depth_mean_shares: float
    ask_depth_std_shares: float
    bid_depth_mean_shares: float
    bid_depth_std_shares: float
    midpoint_change_mean_dollars: float
    midpoint_change_std_dollars: float
    midpoint_lag1_correlation: float


@dataclass(frozen=True)
class BookComparison:
    real: BookStatistics
    generated: BookStatistics


@dataclass(frozen=True)
class ValidationLosses:
    reconstruction_mse: float
    transition_mse: float


@dataclass(frozen=True)
class PredictionScores:
    """Both predictors are tested on held-out real data, in standardized units."""

    train_real_test_real: float
    train_synthetic_test_real: float


@dataclass(frozen=True)
class EvaluationReport:
    feature_names: list[str]
    training_windows: int
    validation_windows: int
    validation_losses: ValidationLosses
    unprojected_feature_statistics: SequenceComparison
    projected_book_statistics: BookComparison
    predictive_mae_standardized: PredictionScores


LOBSTER_PRICE_UNITS_PER_DOLLAR = 10_000


def _correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Column-wise correlation; constant columns have correlation zero."""
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    scale = (x.square().mean(dim=0) * y.square().mean(dim=0)).sqrt().clamp_min(1e-12)
    return (x * y).mean(dim=0) / scale


def sequence_statistics(windows: torch.Tensor) -> SequenceStatistics:
    """Summarize (windows, events, features) without crossing window boundaries."""
    windows = windows.detach().cpu().double()
    flat = windows.flatten(0, 1)
    centered = flat - flat.mean(dim=0)
    std = flat.std(dim=0, correction=0)
    covariance = centered.T @ centered / len(flat)
    correlations = covariance / torch.outer(std, std).clamp_min(1e-12)
    previous, following = windows[:, :-1], windows[:, 1:]
    changes = following - previous
    return SequenceStatistics(
        mean=flat.mean(dim=0),
        std=std,
        change_mean=changes.mean(dim=(0, 1)),
        change_std=changes.std(dim=(0, 1), correction=0),
        lag1_correlation=_correlation(previous.flatten(0, 1), following.flatten(0, 1)),
        feature_correlation=correlations,
        window_mean_std=windows.mean(dim=1).std(dim=0, correction=0),
    )


def compare_sequences(real: torch.Tensor, generated: torch.Tensor) -> SequenceComparison:
    real_stats = sequence_statistics(real)
    generated_stats = sequence_statistics(generated)
    errors = SequenceStatistics(
        mean=(real_stats.mean - generated_stats.mean).abs().mean(),
        std=(real_stats.std - generated_stats.std).abs().mean(),
        change_mean=(real_stats.change_mean - generated_stats.change_mean).abs().mean(),
        change_std=(real_stats.change_std - generated_stats.change_std).abs().mean(),
        lag1_correlation=(real_stats.lag1_correlation - generated_stats.lag1_correlation).abs().mean(),
        feature_correlation=(real_stats.feature_correlation - generated_stats.feature_correlation).abs().mean(),
        window_mean_std=(real_stats.window_mean_std - generated_stats.window_mean_std).abs().mean(),
    )
    return SequenceComparison(real=real_stats, generated=generated_stats, absolute_errors=errors)


def book_statistics(books: torch.Tensor) -> BookStatistics:
    """Summarize (windows, events, 40) books in dollars and shares."""
    # Each level stores ask price, ask size, bid price, bid size, in that order.
    ask_prices, ask_sizes, bid_prices, bid_sizes = books.double().reshape(*books.shape[:2], 10, 4).unbind(-1)
    best_ask, best_bid = ask_prices[..., 0], bid_prices[..., 0]
    midpoint = (best_ask + best_bid) / (2 * LOBSTER_PRICE_UNITS_PER_DOLLAR)
    spread = (best_ask - best_bid) / LOBSTER_PRICE_UNITS_PER_DOLLAR
    ask_depth, bid_depth = ask_sizes.sum(dim=-1), bid_sizes.sum(dim=-1)
    previous, following = midpoint[:, :-1], midpoint[:, 1:]
    changes = following - previous
    return BookStatistics(
        spread_mean_dollars=spread.mean().item(),
        spread_std_dollars=spread.std(correction=0).item(),
        ask_depth_mean_shares=ask_depth.mean().item(),
        ask_depth_std_shares=ask_depth.std(correction=0).item(),
        bid_depth_mean_shares=bid_depth.mean().item(),
        bid_depth_std_shares=bid_depth.std(correction=0).item(),
        midpoint_change_mean_dollars=changes.mean().item(),
        midpoint_change_std_dollars=changes.std(correction=0).item(),
        midpoint_lag1_correlation=_correlation(previous.reshape(-1, 1), following.reshape(-1, 1)).item(),
    )


class _NextStepPredictor(nn.Module):
    def __init__(self, feature_dims: int):
        super().__init__()
        self.recurrent = nn.GRU(feature_dims, 32, batch_first=True)
        self.output = nn.Linear(32, feature_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.recurrent(x)
        return self.output(states)


def _train_predictor(training: torch.Tensor, steps: int) -> _NextStepPredictor:
    predictor = _NextStepPredictor(training.shape[-1])
    optimizer = torch.optim.Adam(predictor.parameters(), lr=3e-4)
    for _ in range(steps):
        batch = training[torch.randint(len(training), (min(64, len(training)),))]
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(predictor(batch[:, :-1]), batch[:, 1:])
        loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
    return predictor


@torch.no_grad()
def _score_predictor(predictor: _NextStepPredictor, validation: torch.Tensor) -> float:
    errors = []
    for batch in validation.split(64):
        predictions = predictor(batch[:, :-1])
        errors.append(F.l1_loss(predictions, batch[:, 1:], reduction="sum"))
    return torch.stack(errors).sum().item() / validation[:, 1:].numel()


def next_step_mae(training: torch.Tensor, validation: torch.Tensor,
                  steps: int = 200, seed: int = 0) -> float:
    """Train on (windows, events, features), then measure next-event error."""
    # Match predictor initialization and sampling without advancing the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        predictor = _train_predictor(training.detach().cpu(), steps)
        return _score_predictor(predictor, validation.detach().cpu())


def _select_nonoverlapping_windows(features: torch.Tensor, sequence_length: int, count: int) -> torch.Tensor:
    windows = WindowDataset(features, sequence_length, stride=sequence_length)
    if len(windows) == 0:
        raise ValueError("Evaluation splits need at least one complete window")
    # Cover each split without overlapping evaluation windows or random leakage.
    indices = torch.linspace(0, len(windows) - 1, min(count, len(windows))).round().long()
    return torch.stack([windows[index.item()] for index in indices])


def _reconstruct_books(data: LOBSTERLevel10Dataset, windows: torch.Tensor) -> torch.Tensor:
    """Undo training normalization and enforce book constraints for each window."""
    books = []
    for window in windows:
        frame = data.feature_sequence_to_df(window)
        books.append(torch.from_numpy(frame.to_numpy(copy=True)))
    return torch.stack(books)


@torch.no_grad()
def _validation_losses(model: TimeGAN, real: torch.Tensor) -> ValidationLosses:
    device = next(model.parameters()).device
    x = real.to(device)
    latents = model.embedder(x)
    noise, _ = generate_wiener_paths(len(x), x.shape[1], model.generator.noise_dims, device=device)
    if getattr(model, "noise_kind", "wiener_paths") == "wiener_increments":
        noise = torch.cat((noise[:, :1], noise[:, 1:] - noise[:, :-1]), dim=1)
    reconstruction = model.decoder(latents)
    next_latents = model.generator.teacher_forced(latents, noise)
    return ValidationLosses(
        reconstruction_mse=F.mse_loss(reconstruction, x).item(),
        transition_mse=F.mse_loss(next_latents, latents[:, 1:]).item(),
    )


def evaluate_timegan(model: TimeGAN, training: LOBSTERLevel10Dataset,
                     validation: LOBSTERLevel10Dataset, sequence_length: int = 64,
                     samples: int = 256, predictor_steps: int = 200,
                     seed: int = 0) -> tuple[EvaluationReport, torch.Tensor]:
    """Evaluate held-out realism and usefulness; seed controls only the predictors."""
    real_training = _select_nonoverlapping_windows(training.features, sequence_length, samples)
    real_validation = _select_nonoverlapping_windows(validation.features, sequence_length, samples)
    model.eval()
    generated = generate_features(model, len(real_training), sequence_length).cpu()

    losses = _validation_losses(model, real_validation)
    feature_comparison = compare_sequences(real_validation, generated)
    real_books = _reconstruct_books(training, real_validation)
    generated_books = _reconstruct_books(training, generated)
    book_comparison = BookComparison(
        real=book_statistics(real_books), generated=book_statistics(generated_books))
    real_score = next_step_mae(real_training, real_validation, predictor_steps, seed)
    generated_score = next_step_mae(generated, real_validation, predictor_steps, seed)

    report = EvaluationReport(
        feature_names=list(training.feature_names),
        training_windows=len(real_training),
        validation_windows=len(real_validation),
        validation_losses=losses,
        unprojected_feature_statistics=feature_comparison,
        projected_book_statistics=book_comparison,
        predictive_mae_standardized=PredictionScores(
            train_real_test_real=real_score, train_synthetic_test_real=generated_score),
    )
    return report, generated
