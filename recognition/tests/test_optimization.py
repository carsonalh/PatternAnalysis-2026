import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from modules import TimeGAN
from dataset import LOBSTERLevel10Dataset
from optimization_metrics import MSE_KEYS, assess_guardrails, validation_loss_metrics
from test_timegan import make_books
from train import TimeGANTrainer, TrainingConfig


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TimeGAN(latent_dims=4, noise_dims=4)
        self.x = torch.randn(3, 6, 40)
        self.trainer = TimeGANTrainer(self.model, TrainingConfig(sequence_length=6, context_horizon=2))

    def test_generator_embedding_pairs_precede_discriminator_and_counts_are_actual(self):
        order = []
        for name in ("generator_step", "embedding_step", "discriminator_step"):
            original = getattr(self.trainer, name)
            def record(x, _name=name, _original=original, **kwargs):
                order.append(_name)
                return _original(x, **kwargs)
            setattr(self.trainer, name, record)
        result = self.trainer.joint_step(self.x)
        self.assertEqual(order, ["generator_step", "embedding_step"] * 2 + ["discriminator_step"])
        self.assertEqual((result["g_updates"], result["e_updates"], result["d_updates"]), (2, 2, 1))
        self.assertEqual(self.trainer.update_counts, dict(generator=2, embedding=2,
                                                         discriminator=1, discriminator_skipped=0))

    def test_discriminator_gate_below_at_and_above_threshold(self):
        for loss, expected in [(0.149, 0), (0.15, 0), (0.151, 1)]:
            def controlled_bce(logits, targets):
                # Keep a genuine gradient while fixing the pre-update BCE value.
                value = logits.double().mean()
                return value - value.detach() + loss / 2
            with self.subTest(loss=loss), patch("train.F.binary_cross_entropy_with_logits",
                                                side_effect=controlled_bce):
                result = self.trainer.discriminator_step(self.x)
            self.assertEqual(result["d_updates"], expected)
            self.assertEqual(result["d_skipped"], 1 - expected)

    def test_skip_leaves_parameters_and_initialized_adam_state_unchanged(self):
        self.trainer.config = replace(self.trainer.config, discriminator_loss_threshold=None)
        self.trainer.discriminator_step(self.x)
        weights = copy.deepcopy(self.model.discriminator.state_dict())
        state = copy.deepcopy(self.trainer.opt_D.state_dict())
        self.trainer.config = replace(self.trainer.config, discriminator_loss_threshold=100)
        self.trainer.discriminator_step(self.x)
        for name, value in weights.items():
            self.assertTrue(torch.equal(value, self.model.discriminator.state_dict()[name]))
        current = self.trainer.opt_D.state_dict()
        self.assertEqual(state["param_groups"], current["param_groups"])
        for parameter, values in state["state"].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(value, current["state"][parameter][name]))
        self.assertTrue(all(p.grad is None for p in self.model.discriminator.parameters()))

    def test_gradient_logging_preserves_update_and_noise_stream(self):
        config = replace(self.trainer.config, context_loss_weight=0.01,
                         price_moment_weight=100, conditional_drift_weight=100)
        plain = TimeGANTrainer(copy.deepcopy(self.model), config)
        measured = TimeGANTrainer(copy.deepcopy(self.model), config)
        rng = torch.get_rng_state()
        plain.generator_step(self.x)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        result = measured.generator_step(self.x, diagnose=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), after))
        for name, value in plain.model.state_dict().items():
            torch.testing.assert_close(value, measured.model.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))
        self.assertGreater(result["gradient_norm_adversarial"], 0)
        self.assertAlmostEqual(result["gradient_cosine_adversarial_adversarial"], 1, places=5)
        self.assertAlmostEqual(result["generator"], sum(value for name, value in result.items()
                                                       if name.startswith("weighted_")), places=5)

    def test_gradient_interval_and_disabled_components(self):
        self.trainer.config = replace(self.trainer.config, gradient_diagnostic_interval=2)
        first = self.trainer.joint_step(self.x)
        second = self.trainer.joint_step(self.x)
        self.assertNotIn("g_total_gradient_norm", first)
        self.assertIn("g_total_gradient_norm", second)
        self.assertEqual(second["gradient_norm_g_conditional_drift"], 0)
        self.assertEqual(second["gradient_cosine_g_conditional_drift_adversarial"], 0)

    def test_nonfinite_discriminator_loss_is_never_silently_skipped(self):
        with patch("train.F.binary_cross_entropy_with_logits", return_value=torch.tensor(float("nan"))):
            with self.assertRaises(FloatingPointError):
                self.trainer.discriminator_step(self.x)

    def test_new_configuration_rejects_invalid_values(self):
        for threshold in (-1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                TrainingConfig(discriminator_loss_threshold=threshold)
        for interval in (-1, 1.5):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                TrainingConfig(gradient_diagnostic_interval=interval)


class GuardrailTests(unittest.TestCase):
    def setUp(self):
        self.distribution = dict(return_sd_ratio=0.94, terminal_median_bias_dollars=0.01,
                                 terminal_log_normal_ks=0.02, terminal_sd_dollars=0.812)
        self.baseline = {key: 0.001 for key in MSE_KEYS}
        self.losses = self.baseline.copy()

    def test_all_required_metrics_and_exact_boundaries(self):
        self.distribution.update(return_sd_ratio=0.9, terminal_median_bias_dollars=0.05,
                                 terminal_log_normal_ks=0.03, terminal_sd_dollars=0.812 * 0.8)
        self.losses = {key: self.baseline[key] * 1.1 for key in MSE_KEYS}
        report = assess_guardrails(self.distribution, self.losses, self.baseline)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["checks"]), 7)

    def test_each_missing_nonfinite_or_outside_metric_fails(self):
        for key, value in [("return_sd_ratio", 1.11), ("terminal_median_bias_dollars", -0.051),
                           ("terminal_log_normal_ks", 0.031), ("terminal_sd_dollars", 0),
                           ("return_sd_ratio", float("nan")), ("return_sd_ratio", None)]:
            with self.subTest(key=key, value=value):
                distribution = self.distribution | {key: value}
                self.assertFalse(assess_guardrails(distribution, self.losses, self.baseline)["passed"])
        for key in MSE_KEYS:
            with self.subTest(key=key):
                self.assertFalse(assess_guardrails(self.distribution, self.losses | {key: 0.00111},
                                                  self.baseline)["passed"])

    def test_validation_audit_is_repeatable_without_changing_weights_or_global_rng(self):
        import pandas as pd
        with patch("dataset.load_dfs", return_value=(make_books(512), pd.DataFrame())):
            training = LOBSTERLevel10Dataset(True)
            validation = LOBSTERLevel10Dataset(False)
        model = TimeGAN(latent_dims=4, noise_dims=4)
        config = TrainingConfig(context_loss_weight=0.01, price_moment_weight=100,
                                conditional_drift_weight=100)
        trainer = TimeGANTrainer(model, config)
        weights = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state()
        first = validation_loss_metrics(model, trainer.config.__dict__, training, validation, noise_seeds=[1000])
        second = validation_loss_metrics(model, trainer.config.__dict__, training, validation, noise_seeds=[1000])
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in weights.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))
        self.assertAlmostEqual(first["generator_loss"], sum(value for name, value in first.items()
                                                           if name.startswith("weighted_")), places=5)


if __name__ == "__main__":
    unittest.main()
