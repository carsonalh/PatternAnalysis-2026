import json
from dataclasses import asdict
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import LOBSTERLevel10Dataset, ORDER_BOOK_COLUMNS, WindowDataset, features_to_order_book
from evaluation import book_statistics, compare_sequences, evaluate_timegan, next_step_mae, sequence_statistics
from modules import TimeGAN
from predict import generate_wiener_paths, generate_features, load_checkpoint
from train import TimeGANTrainer, TrainingConfig, context_price_loss, save_checkpoint, supervised_loss

# Small recurrent tests are substantially faster without many CPU worker threads.
torch.set_num_threads(1)


def make_books(events=100):
    rows = []
    for t in range(events):
        row = []
        for level in range(10):
            row.extend((2_200_100 + (t + level) * 100, 10 + t + level,
                        2_200_000 + (t - level) * 100, 20 + 2 * t + level))
        rows.append(row)
    return pd.DataFrame(rows, columns=ORDER_BOOK_COLUMNS)


class DatasetTests(unittest.TestCase):
    def test_windows_preserve_split_and_training_normalization(self):
        books = make_books()
        with patch("dataset.load_dfs", return_value=(books, pd.DataFrame())):
            training = LOBSTERLevel10Dataset(train=True)
            validation = LOBSTERLevel10Dataset(train=False)
        np.testing.assert_array_equal(training.feature_mean, validation.feature_mean)
        np.testing.assert_array_equal(training.feature_std, validation.feature_std)
        self.assertAlmostEqual(training.features[:, 0].mean().item(), 0, places=6)
        self.assertGreater(validation.features[:, 0].mean().item(), 1)
        train_windows = WindowDataset(training.features, 4)
        val_windows = WindowDataset(validation.features, 4)
        self.assertEqual(len(train_windows), 82)
        self.assertEqual(len(val_windows), 12)
        torch.testing.assert_close(train_windows[-1], training.features[-4:])
        torch.testing.assert_close(val_windows[0], validation.features[:4])
        self.assertEqual(training.books.index[-1], 84)
        self.assertEqual(validation.books.index[0], 85)
        reconstructed = validation.feature_sequence_to_df(val_windows[0])
        np.testing.assert_array_equal(reconstructed.to_numpy(), books.iloc[85:89].to_numpy())

    def test_stride_short_splits_and_bounds(self):
        features = torch.arange(20).reshape(10, 2)
        windows = WindowDataset(features, 4, stride=3)
        self.assertEqual(len(windows), 3)
        torch.testing.assert_close(windows[1], features[3:7])
        self.assertEqual(len(WindowDataset(features[:3], 4)), 0)
        with self.assertRaises(IndexError):
            windows[3]
        with self.assertRaises(ValueError):
            WindowDataset(features, 1)

    def test_projection_enforces_book_constraints(self):
        # Negative gap/size logs are clamped only at the export boundary.
        features = np.full((6, 40), -2.0)
        features[:, 0] = np.log(2_200_000)
        levels = features_to_order_book(features).to_numpy().reshape(6, 10, 4)
        self.assertTrue((levels[:, :, 0] > levels[:, :, 2]).all())
        self.assertTrue((np.diff(levels[:, :, 0], axis=1) > 0).all())
        self.assertTrue((np.diff(levels[:, :, 2], axis=1) < 0).all())
        self.assertTrue((levels[:, :, (0, 2)] % 100 == 0).all())
        self.assertTrue((levels[:, :, (1, 3)] >= 1).all())


class NoiseTests(unittest.TestCase):
    def test_wiener_variance_and_independent_channels(self):
        rng = torch.Generator().manual_seed(7)
        paths, seed = generate_wiener_paths(12_000, 9, 4, rng=rng)
        self.assertEqual(paths.shape, (12_000, 9, 4))
        self.assertEqual(seed.shape, (12_000, 4))
        self.assertTrue(torch.equal(paths[:, 0], torch.zeros_like(paths[:, 0])))
        increments = paths[:, 1:] - paths[:, :-1]
        torch.testing.assert_close(increments.var(dim=(0, 1)), torch.full((4,), 1 / 8),
                                   rtol=0.03, atol=0)
        torch.testing.assert_close(paths[:, -1].var(dim=0), torch.ones(4), rtol=0.05, atol=0)
        correlations = torch.corrcoef(increments.flatten(0, 1).T)
        self.assertLess((correlations - torch.eye(4)).abs().max().item(), 0.025)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = TimeGAN(feature_dims=3, latent_dims=8, noise_dims=7)
        self.x = torch.randn(4, 6, 3)
        self.noise, self.seed = generate_wiener_paths(4, 6, 7)

    def test_shapes_teacher_alignment_and_random_first_books(self):
        h = self.model.embedder(self.x)
        fake = self.model.generator.sample(self.noise, self.seed)
        self.assertEqual(h.shape, (4, 6, 8))
        self.assertEqual(fake.shape, h.shape)
        self.assertEqual(self.model.decoder(fake).shape, self.x.shape)
        self.assertEqual(self.model.discriminator(fake).shape, (4, 6, 1))
        predictions = self.model.generator.teacher_forced(h, self.noise)
        manual = torch.stack([self.model.generator.cell(self.noise[:, t], h[:, t - 1])
                              for t in range(1, 6)], dim=1)
        torch.testing.assert_close(predictions, manual)
        self.assertFalse(torch.equal(fake[0, 0], fake[1, 0]))
        self.assertLessEqual(h.abs().max().item(), 1)
        self.assertLessEqual(fake.abs().max().item(), 1)

    def test_causality_and_state_resets(self):
        original_embedding = self.model.embedder(self.x)
        changed_x = self.x.clone()
        changed_x[:, 3:] += 100
        torch.testing.assert_close(self.model.embedder(self.x)[:, :3],
                                   self.model.embedder(changed_x)[:, :3])
        changed_noise = self.noise.clone()
        changed_noise[:, 3:] += 100
        original = self.model.generator.sample(self.noise, self.seed)
        torch.testing.assert_close(original[:, :3],
                                   self.model.generator.sample(changed_noise, self.seed)[:, :3])
        self.model.embedder(changed_x)
        torch.testing.assert_close(original_embedding, self.model.embedder(self.x))
        torch.testing.assert_close(original, self.model.generator.sample(self.noise, self.seed))

    def test_feedback_preserves_gradients_to_initializer(self):
        fake = self.model.generator.sample(self.noise, self.seed)
        fake[:, -1].square().sum().backward()
        self.assertGreater(self.model.generator.initial[0].weight.grad.abs().sum().item(), 0)

    def test_context_rollout_alignment_and_feedback_gradients(self):
        initial = self.model.embedder(self.x[:, :3])[:, -1].detach().requires_grad_()
        noise = self.noise[:, 1:4]
        predicted = self.model.generator.continue_from(initial, noise)
        h = initial
        manual = []
        for step in range(3):
            h = self.model.generator.cell(noise[:, step], h)
            manual.append(h)
        torch.testing.assert_close(predicted, torch.stack(manual, dim=1))
        predicted[:, -1].square().sum().backward()
        self.assertGreater(initial.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.generator.cell.weight_hh.grad.abs().sum().item(), 0)

    def test_sampling_does_not_use_embedder_or_discriminator(self):
        with patch.object(self.model.embedder, "forward", side_effect=AssertionError), \
             patch.object(self.model.discriminator, "forward", side_effect=AssertionError):
            self.assertEqual(generate_features(self.model, 2, 6).shape, (2, 6, 3))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = TimeGAN(feature_dims=3, latent_dims=8, noise_dims=7)
        self.config = TrainingConfig(sequence_length=6, batch_size=4,
                                     autoencoder_steps=2, transition_steps=2, joint_steps=2)
        self.trainer = TimeGANTrainer(self.model, self.config)
        self.x = torch.randn(4, 6, 3)

    def test_optimizer_ownership_for_every_update(self):
        names = ("embedder", "decoder", "generator", "discriminator")
        cases = (
            ("autoencoder_step", {"embedder", "decoder"}),
            ("transition_step", {"generator"}),
            ("discriminator_step", {"discriminator"}),
            ("generator_step", {"generator"}),
            ("embedding_step", {"embedder", "decoder"}),
        )
        for method, active in cases:
            with self.subTest(update=method):
                before = {name: [p.detach().clone() for p in getattr(self.model, name).parameters()]
                          for name in names}
                metrics = getattr(self.trainer, method)(self.x)
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
                for name in names:
                    parameters = list(getattr(self.model, name).parameters())
                    changed = any(not torch.equal(old, new) for old, new in zip(before[name], parameters))
                    self.assertEqual(changed, name in active, name)
                    if name not in active:
                        self.assertTrue(all(p.grad is None for p in parameters))

    def test_transition_warmup_leaves_initializer_untouched(self):
        before = self.model.generator.initial[0].weight.detach().clone()
        self.trainer.transition_step(self.x)
        torch.testing.assert_close(before, self.model.generator.initial[0].weight)
        self.trainer.generator_step(self.x)
        self.assertFalse(torch.equal(before, self.model.generator.initial[0].weight))
        self.assertTrue(self.model.discriminator.training)

    def test_separate_discriminator_rate_and_multiple_generator_updates(self):
        config = TrainingConfig(sequence_length=6, discriminator_learning_rate=1e-4,
                                generator_updates=2)
        trainer = TimeGANTrainer(self.model, config)
        self.assertEqual(trainer.opt_ER.param_groups[0]["lr"], config.learning_rate)
        self.assertEqual(trainer.opt_G.param_groups[0]["lr"], config.learning_rate)
        self.assertEqual(trainer.opt_D.param_groups[0]["lr"], 1e-4)
        with patch.object(trainer, "discriminator_step", wraps=trainer.discriminator_step) as d, \
             patch.object(trainer, "generator_step", wraps=trainer.generator_step) as g, \
             patch.object(trainer, "embedding_step", wraps=trainer.embedding_step) as e:
            metrics = trainer.joint_step(self.x)
        self.assertEqual((d.call_count, g.call_count, e.call_count), (1, 2, 1))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(next(iter(trainer.opt_G.state.values()))["step"].item(), 2)
        self.assertEqual(next(iter(trainer.opt_D.state.values()))["step"].item(), 1)

    def test_training_balance_configuration_validation(self):
        for rate in (0, -1e-4, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                TrainingConfig(discriminator_learning_rate=rate)
        for updates in (0, -1, 1.5):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                TrainingConfig(generator_updates=updates)

    def test_context_loss_configuration_validation(self):
        for weight in (-1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                TrainingConfig(context_loss_weight=weight)
        for horizon in (0, -1, 1.5, 64):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                TrainingConfig(context_loss_weight=1, context_horizon=horizon)

    def test_context_loss_has_no_future_conditioning_and_updates_only_generator(self):
        config = TrainingConfig(sequence_length=6, context_horizon=2, context_loss_weight=10)
        trainer = TimeGANTrainer(self.model, config)
        noise, _ = generate_wiener_paths(4, 6, 7)
        trainer.train_only(self.model.generator)
        # Changing future labels changes the loss, but cannot change the rollout.
        changed = self.x.clone()
        changed[:, -2:, 0] += 100
        with patch.object(self.model.embedder, "forward", wraps=self.model.embedder.forward) as encode, \
             patch.object(self.model.generator, "continue_from", wraps=self.model.generator.continue_from) as roll:
            loss = context_price_loss(self.model, self.x, noise, 2)
            first_context = roll.call_args.args[0].clone()
            changed_loss = context_price_loss(self.model, changed, noise, 2)
            torch.testing.assert_close(first_context, roll.call_args.args[0])
            self.assertTrue(all(call.args[0].shape[1] == 4 for call in encode.call_args_list))
        self.assertGreater(changed_loss.item(), loss.item())
        loss.backward()
        self.assertGreater(self.model.generator.cell.weight_hh.grad.abs().sum().item(), 0)
        for network in (self.model.embedder, self.model.decoder, self.model.discriminator):
            self.assertTrue(all(p.grad is None for p in network.parameters()))
        self.assertIn("g_context_price", trainer.generator_step(self.x))

    def test_zero_context_weight_preserves_loss_and_rng(self):
        disabled = TimeGANTrainer(self.model, TrainingConfig(sequence_length=6, context_loss_weight=0))
        rng_before = torch.get_rng_state()
        with patch("train.context_price_loss", side_effect=AssertionError):
            metrics = disabled.generator_step(self.x)
        rng_after = torch.get_rng_state()
        self.assertNotIn("g_context_price", metrics)
        self.assertAlmostEqual(metrics["generator"], metrics["adversarial"] + 10 * metrics["g_supervised"], places=6)
        torch.set_rng_state(rng_before)
        generate_wiener_paths(4, 6, 7)
        torch.testing.assert_close(torch.get_rng_state(), rng_after)

    def test_supervision_differentiates_through_frozen_generator(self):
        self.trainer.train_only(self.model.embedder, self.model.decoder)
        noise, _ = generate_wiener_paths(4, 6, 7)
        h = self.model.embedder(self.x)
        supervised_loss(self.model.generator, h, noise).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in self.model.embedder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.generator.parameters()))

    def test_fit_restarts_loader_and_handles_partial_batches(self):
        loader = DataLoader(WindowDataset(torch.randn(8, 3), 6), batch_size=2, shuffle=True)
        history = self.trainer.fit(loader)
        self.assertEqual([row["stage"] for row in history], ["autoencoder"] * 2 + ["transition"] * 2 + ["joint"] * 2)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        with self.assertRaises(ValueError):
            self.trainer.fit(DataLoader(WindowDataset(torch.randn(3, 3), 6)))

    def test_checkpoint_round_trip_and_sampling_without_data(self):
        with patch("dataset.load_dfs", return_value=(make_books(), pd.DataFrame())):
            data = LOBSTERLevel10Dataset()
        model = TimeGAN()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, self.config, data)
            with patch("dataset.load_dfs", side_effect=AssertionError):
                loaded, checkpoint = load_checkpoint(path)
                torch.manual_seed(6)
                expected = generate_features(model, 2, 6)
                torch.manual_seed(6)
                actual = generate_features(loaded, 2, 6)
                torch.testing.assert_close(actual, expected)
                book = features_to_order_book(actual[0], checkpoint["feature_mean"].numpy(),
                                              checkpoint["feature_std"].numpy())
                self.assertEqual(book.shape, (6, 40))

    def test_temporal_learning_on_autoregressive_sequences(self):
        torch.manual_seed(13)
        windows = torch.empty(160, 10, 2)
        windows[:, 0] = torch.randn(160, 2)
        for t in range(1, 10):
            windows[:, t] = 0.85 * windows[:, t - 1] + 0.15 * torch.randn(160, 2)
        training, validation = windows[:128], windows[128:]
        model = TimeGAN(feature_dims=2, latent_dims=8, noise_dims=4)
        trainer = TimeGANTrainer(model, TrainingConfig(sequence_length=10, learning_rate=3e-3))
        with torch.no_grad():
            reconstruction_before = F.mse_loss(model(validation), validation).item()
        for _ in range(100):
            trainer.autoencoder_step(training[torch.randint(128, (32,))])
        with torch.no_grad():
            h = model.embedder(validation)
            noise, _ = generate_wiener_paths(32, 10, 4)
            transition_before = supervised_loss(model.generator, h, noise).item()
            reconstruction_after = F.mse_loss(model(validation), validation).item()
        for _ in range(150):
            trainer.transition_step(training[torch.randint(128, (32,))])
        with torch.no_grad():
            transition_after = supervised_loss(model.generator, h, noise).item()
            mean_baseline = F.mse_loss(h[:, 1:].mean(dim=(0, 1)).expand_as(h[:, 1:]), h[:, 1:]).item()
        self.assertLess(reconstruction_after, reconstruction_before * 0.3)
        self.assertLess(transition_after, transition_before * 0.6)
        self.assertLess(transition_after, mean_baseline)


class EvaluationTests(unittest.TestCase):
    def test_full_report_on_separate_mock_book_splits(self):
        with patch("dataset.load_dfs", return_value=(make_books(), pd.DataFrame())):
            training = LOBSTERLevel10Dataset(train=True)
            validation = LOBSTERLevel10Dataset(train=False)
        model = TimeGAN(latent_dims=8, noise_dims=7)
        report, generated = evaluate_timegan(model, training, validation,
                                             sequence_length=4, samples=4, predictor_steps=3)
        self.assertEqual(generated.shape, (4, 4, 40))
        self.assertEqual(report.validation_windows, 3)
        self.assertEqual(report.unprojected_feature_statistics.real.mean.shape, (40,))
        self.assertGreater(report.projected_book_statistics.generated.spread_mean_dollars, 0)
        saved = json.loads(json.dumps(asdict(report), default=torch.Tensor.tolist, allow_nan=False))
        self.assertEqual(len(saved["unprojected_feature_statistics"]["real"]["mean"]), 40)
        self.assertIsInstance(saved["unprojected_feature_statistics"]["absolute_errors"]["mean"], float)

    def test_statistics_do_not_cross_window_boundaries(self):
        windows = torch.tensor([[[0.0], [1.0]], [[100.0], [101.0]]])
        stats = sequence_statistics(windows)
        self.assertEqual(stats.change_mean.item(), 1)
        self.assertEqual(stats.change_std.item(), 0)
        comparison = compare_sequences(windows, windows)
        for value in asdict(comparison.absolute_errors).values():
            self.assertEqual(value.item(), 0)

    def test_book_statistics_use_dollars_and_total_depth(self):
        books = torch.from_numpy(make_books(events=4).to_numpy(copy=True)).unsqueeze(0)
        stats = book_statistics(books)
        self.assertAlmostEqual(stats.spread_mean_dollars, 0.01)
        self.assertEqual(stats.ask_depth_mean_shares, 160)
        self.assertEqual(stats.bid_depth_mean_shares, 275)
        self.assertAlmostEqual(stats.midpoint_change_mean_dollars, 0.01)

    def test_predictive_evaluation_is_reproducible_and_finite(self):
        torch.manual_seed(15)
        windows = torch.randn(8, 4, 2)
        rng_before = torch.get_rng_state()
        first = next_step_mae(windows, windows, steps=3, seed=2)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        second = next_step_mae(windows, windows, steps=3, seed=2)
        self.assertEqual(first, second)
        self.assertTrue(np.isfinite(first))


if __name__ == "__main__":
    unittest.main()
