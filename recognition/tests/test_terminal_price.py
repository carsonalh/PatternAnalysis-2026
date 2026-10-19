import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset import LOBSTERLevel10Dataset
from loss_weight_report import endpoint_distribution, summarize_endpoints
from modules import TimeGAN
from test_timegan import make_books
from train import TimeGANTrainer, TrainingConfig


class TerminalPriceTests(unittest.TestCase):
    def test_forecast_observes_first_validation_book_and_preserves_decoder_returns(self):
        import pandas as pd
        with patch("dataset.load_dfs", return_value=(make_books(), pd.DataFrame())):
            training = LOBSTERLevel10Dataset(True)
            validation = LOBSTERLevel10Dataset(False)
        model = TimeGAN(latent_dims=4, noise_dims=4)
        TimeGANTrainer(model, TrainingConfig(sequence_length=6, context_horizon=2))
        physical = np.linspace(-0.0001, 0.0001, 32)[:, None] * np.ones((1, len(validation) - 1))
        features = torch.zeros(32, len(validation) - 1, 40)
        features[..., 0] = torch.from_numpy((physical - training.feature_mean[0]) / training.feature_std[0])
        with patch("loss_weight_report.complete", return_value=features) as rollout:
            metrics, returns, prices = endpoint_distribution(model, training, validation, samples=32)
        context = rollout.call_args.args[1]
        torch.testing.assert_close(context[0], torch.cat((training.features[-5:], validation.features[:1])))
        self.assertEqual(metrics["horizon_seconds"], len(validation) - 1)
        anchor = (validation.books.ASKp1.iloc[0] + validation.books.BIDp1.iloc[0]) / 20_000
        self.assertEqual(metrics["anchor_dollars"], anchor)
        np.testing.assert_allclose(returns, physical, atol=1e-11)
        np.testing.assert_allclose(prices, anchor * np.exp(np.cumsum(returns, axis=1)))
        self.assertAlmostEqual(metrics["terminal_median_dollars"], np.median(prices[:, -1]))
        self.assertAlmostEqual(metrics["terminal_upward_share"], 0.5)

    def test_changing_future_validation_data_cannot_change_forecast(self):
        import pandas as pd
        with patch("dataset.load_dfs", return_value=(make_books(), pd.DataFrame())):
            training = LOBSTERLevel10Dataset(True)
            validation = LOBSTERLevel10Dataset(False)
        model = TimeGAN(latent_dims=4, noise_dims=4)
        TimeGANTrainer(model, TrainingConfig(sequence_length=6, context_horizon=2))
        _, first_returns, first_prices = endpoint_distribution(model, training, validation, samples=32, seed=7)
        validation.features[1:, 0] += torch.linspace(-0.5, 0.5, len(validation) - 1)
        validation.books.iloc[1:, 0::2] += 1000
        _, second_returns, second_prices = endpoint_distribution(model, training, validation, samples=32, seed=7)
        np.testing.assert_array_equal(first_returns, second_returns)
        np.testing.assert_array_equal(first_prices, second_prices)

    def test_pooled_metrics_preserve_anchor_and_reject_recentered_prices(self):
        import pandas as pd
        with patch("dataset.load_dfs", return_value=(make_books(), pd.DataFrame())):
            training = LOBSTERLevel10Dataset(True)
            validation = LOBSTERLevel10Dataset(False)
        anchor = (validation.books.ASKp1.iloc[0] + validation.books.BIDp1.iloc[0]) / 20_000
        returns = np.repeat(np.linspace(-1e-5, 1e-5, 64)[:, None], len(validation) - 1, axis=1)
        prices = anchor * np.exp(returns.cumsum(axis=1))
        metrics = summarize_endpoints(training, validation, returns, prices)
        self.assertEqual(metrics["samples"], 64)
        self.assertLessEqual(metrics["median_ci95_low_dollars"], anchor)
        self.assertGreaterEqual(metrics["median_ci95_high_dollars"], anchor)
        with self.assertRaises(AssertionError):
            summarize_endpoints(training, validation, returns, prices + 1)


if __name__ == "__main__":
    unittest.main()
