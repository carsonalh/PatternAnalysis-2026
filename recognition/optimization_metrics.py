"""Paired validation losses and explicit gates for the one-second study.

GAN BCE is a training-balance diagnostic. Price-distribution gates and decoded
book statistics assess behavior independently of each run's latent coordinates.
"""

from dataclasses import asdict
import math

import numpy as np
import torch
from torch.nn import functional as F

from evaluation import _select_nonoverlapping_windows, book_statistics
from train import conditional_drift_loss, context_price_loss


MSE_KEYS = ("latent_supervision_mse", "reconstruction_mse", "return_reconstruction_mse")


def verify_checkpoint_data(checkpoint, training):
    if checkpoint.get("data_config", {}).get("sampling_interval_seconds") != 1:
        raise ValueError("Optimization comparisons require one-second checkpoints")
    if checkpoint["training_config"]["sequence_length"] != 64:
        raise ValueError("Optimization comparisons use the fixed 64-second windows")
    if checkpoint["training_config"]["noise_kind"] != "wiener_increments":
        raise ValueError("Optimization comparisons require Wiener increments")
    if checkpoint.get("split_time_seconds") != training.split_time_seconds:
        raise ValueError("Checkpoint split differs from the original event cutoff")
    np.testing.assert_allclose(checkpoint["feature_mean"].cpu(), training.feature_mean)
    np.testing.assert_allclose(checkpoint["feature_std"].cpu(), training.feature_std)


@torch.inference_mode()
def validation_loss_metrics(model, config, training, validation, noise_seeds=range(1000, 1010)):
    """Identical 35 non-overlapping windows and ten shared draws per model.

    Reconstruction averages all standardized features; return reconstruction
    isolates feature zero. Latent supervision is raw teacher-forced MSE.
    Book statistics use the ordinary tick-projected export, explicitly separate
    from the unrounded price-distribution scores.
    """
    model.eval()
    device = next(model.parameters()).device
    real = _select_nonoverlapping_windows(validation.features, 64, 35).to(device)
    latents = model.embedder(real)
    decoded = model.decoder(latents)
    reconstruction = F.mse_loss(decoded, real).item()
    return_reconstruction = F.mse_loss(decoded[..., 0], real[..., 0]).item()
    real_logits = model.discriminator(latents)
    real_bce = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)).item()
    seeds = tuple(noise_seeds)
    if not seeds:
        raise ValueError("Validation losses need at least one shared noise draw")
    rows, generated = [], []
    zero_return = -training.feature_mean[0] / training.feature_std[0]
    for seed in seeds:
        rng = torch.Generator(device=device).manual_seed(seed)
        innovations = torch.randn(len(real), 63, model.generator.noise_dims,
                                  generator=rng, device=device) / math.sqrt(63)
        noise = torch.cat((innovations.new_zeros(len(real), 1, innovations.shape[-1]), innovations), dim=1)
        initial = torch.randn(len(real), model.generator.noise_dims, generator=rng, device=device)
        fake = model.generator.sample(noise, initial)
        logits = model.discriminator(fake)
        fake_features = model.decoder(fake)
        generated.append(fake_features.cpu())
        supervised = F.mse_loss(model.generator.teacher_forced(latents, noise), latents[:, 1:]).item()
        fake_bce = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).item()
        adversarial = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).item()
        predicted = fake_features[..., 0]
        moments = ((predicted.mean() - real[..., 0].mean()).square()
                   + (predicted.std(correction=0) - real[..., 0].std(correction=0)).square()).item()
        context = context_price_loss(model, real, noise, config["context_horizon"]).item()
        drift = conditional_drift_loss(model, real, noise, config["context_horizon"], zero_return).item()
        row = dict(reconstruction_mse=reconstruction, return_reconstruction_mse=return_reconstruction,
                   latent_supervision_mse=supervised,
                   embedding_loss=reconstruction + config["lam"] * supervised,
                   d_real_bce=real_bce, d_fake_bce=fake_bce, discriminator_bce=real_bce + fake_bce,
                   d_real_accuracy=(real_logits >= 0).float().mean().item(),
                   d_fake_accuracy=(logits < 0).float().mean().item(),
                   adversarial=adversarial, g_supervised=supervised, g_context_price=context,
                   g_price_moments=moments, g_conditional_drift=drift)
        weights = dict(adversarial=1, g_supervised=config["eta"],
                       g_context_price=config["context_loss_weight"],
                       g_price_moments=config["price_moment_weight"],
                       g_conditional_drift=config.get("conditional_drift_weight", 0))
        row.update({f"weighted_{name}": row[name] * weight for name, weight in weights.items()})
        row["generator_loss"] = sum(row[f"weighted_{name}"] for name in weights)
        rows.append(row)
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Non-finite validation loss")
    fake_books = torch.stack([torch.from_numpy(training.feature_sequence_to_df(window).to_numpy(copy=True))
                              for window in torch.cat(generated)])
    real_books = _select_nonoverlapping_windows(torch.from_numpy(validation.books.to_numpy(copy=True)), 64, 35)
    result["books"] = {"validation": asdict(book_statistics(real_books)),
                       "generated_unconditional": asdict(book_statistics(fake_books)),
                       "projection": "Tick rounding and positive counts; price-distribution metrics are unrounded"}
    result["window_count"] = len(real)
    result["noise_seeds"] = list(seeds)
    return result


def assess_guardrails(distribution, losses, baseline_losses):
    """Fail closed for missing/non-finite values; include limits in the report."""
    checks = {}

    def check(name, value, lower, upper):
        valid = isinstance(value, (int, float, np.number)) and math.isfinite(value)
        checks[name] = {"value": float(value) if valid else None,
                        "lower": lower, "upper": upper,
                        "passed": bool(valid and lower <= value <= upper)}

    check("return_volatility", distribution.get("return_sd_ratio"), 0.9, 1.1)
    check("terminal_median_bias_dollars", distribution.get("terminal_median_bias_dollars"), -0.05, 0.05)
    check("terminal_log_normal_cdf_distance", distribution.get("terminal_log_normal_ks"), 0, 0.03)
    check("terminal_sd_dollars", distribution.get("terminal_sd_dollars"), 0.812 * 0.8, 0.812 * 1.2)
    for key in MSE_KEYS:
        baseline = baseline_losses.get(key)
        if not isinstance(baseline, (int, float)) or not math.isfinite(baseline) or baseline < 0:
            raise ValueError(f"Missing or invalid baseline loss: {key}")
        check(key, losses.get(key), 0, baseline * 1.1)
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}
