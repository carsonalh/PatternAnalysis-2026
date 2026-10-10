# Recurrent TimeGAN for AMZN limit order books

A university feasibility study using the LOBSTER AMZN level-10 sample for
21 June 2012. The model follows the four-network formulation of the
[TimeGAN paper](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf),
with direct latent recurrence. It models event order, rather than elapsed time.

## Fixed feature representation

Every training, reconstruction, generation, and prediction path uses the same
40 features:

- Midpoint **log return**: `r_t = log(m_t / m_{t-1})`.
- Spread: `log1p(spread / tick_size)`.
- Eighteen adjacent ask/bid gaps: `log1p(gap / tick_size)`.
- Twenty ask/bid order sizes: `log1p(shares)`.

There is no price-representation argument in the dataset or training configuration.
Feature zero is always `log_midpoint_return`. The first event of the day has
return zero; the first validation return uses the last training midpoint.
The first 85% of events are training data and the remaining 15% are validation.
Both splits use normalization fitted only on training events. Windows are formed
within each split, and shuffling preserves event order within a window.

TimeGAN's decoder predicts all 40 standardized features. It never predicts an
absolute midpoint price. Book export and dollar-price plots undo normalization
and compute `m_t = m_anchor * exp(cumsum(r)_t)`. Conditional forecasts use the
last observed midpoint as their anchor. Independent generated windows use the
saved training median and restart accumulation for each window. Reconstruction
uses the actual midpoint preceding each observed window.

Prices remain in LOBSTER's 1/10,000-dollar units until plotting or reporting.
Tick rounding and positive spread, gap, and size constraints are applied during
book conversion, outside the differentiable model. No external price process,
fixed drift, or validation future modifies decoder predictions.

## Model and training

- `Embedder`: causal GRU, `40 → 128`, one history summary per event.
- `Decoder`: MLP, `128 → 256 → 256 → 40`, with linear outputs.
- `Generator`: `GRUCell(128, 128)` with generated latents fed back as state.
- `Discriminator`: bidirectional GRU with 64 units per direction and a real/fake logit.

The default noise inputs are stationary Wiener increments with variance
`1 / (T - 1)` for a training window of length `T`. A separate Gaussian vector
initializes an unconditional sequence. The accumulated-path noise option is
retained for noise experiments; it does not change any dataset feature.

```sh
uv run python main.py
```

Defaults are 64-event windows, batch size 64, Adam at `3e-4`, and gradient clipping
at norm 1. Training fits the embedder/decoder for 1,000 reconstruction updates,
the generator for 1,000 teacher-forced transition updates, then all networks for
5,000 joint iterations. The default output is `runs/log_return/timegan/`.
The sample ZIP is downloaded into `data/` if it is not already cached.

Reconstruction uses `MSE(R(E(X)), X)`. Joint generator training combines
adversarial feedback with latent transition supervision. Embedder/decoder
training combines reconstruction and latent supervision. Optional context
supervision matches cumulative log-return errors after a real prefix;
return-moment matching penalizes generated log-return mean and SD errors.
The objectives use the four-network direct-recurrence formulation, rather than
a separate supervisor network.

The notebooks use context weight 1, return-moment weight 10, discriminator rate
`1e-4`, and a 16-event continuation after 48 real context events. A context-weight-10
run provides a comparison with the same feature space, architecture, noise,
seed, data sampler, and stage budgets. See the [notebook guide](notebooks/README.md)
for their training and report commands.

For a short procedure check:

```sh
uv run python main.py --autoencoder-steps 10 --transition-steps 10 --joint-steps 10 --evaluation-samples 16 --predictor-steps 20 --output runs/log_return/smoke
```

Check `python main.py --help` for training settings. Checkpoints save network
weights, the fixed feature names, and training normalization. They support
sampling, but do not store optimizer state for resumed training.

## Predictions and evaluation

```sh
uv run python predict.py runs/log_return/returns_context_1_moments_10_seed_0/model.pt --samples 16 --output runs/log_return/samples
```

Sampling needs no real books. It writes standardized log-return feature windows
and one anchored book CSV per window. Loading a checkpoint checks that its
features match the fixed log-return dataset; old price-level weights are rejected.
Historical log-return checkpoints remain readable because their feature names
and normalization already describe the same fixed features.

Conditional continuations repeatedly encode generated feature history, then
predict another block of log-return features. Validation observations never
enter an ongoing rollout. Long continuations can accumulate learned return bias;
log-return prediction removes the absolute-price reset mechanism but does not
guarantee realistic drift, volatility, or book structure.

Evaluation compares all 40 features before book conversion, including midpoint
log returns. It reports feature moments, temporal and cross-feature dependence,
and variation across windows. After book conversion it measures spreads, depth,
and midpoint changes. Real book statistics retain observed prices and anchors.
Next-event predictors train and score in the same standardized log-return feature
space. Valid projected books do not establish realistic temporal generation.

## Notebooks

Read the [four chapters](notebooks/README.md) in order. Executed plots and tables
are saved for reading without retraining:

1. Data, fixed log-return features, normalization, and conversion checks.
2. TimeGAN training, log-return reconstruction, and boundary predictions.
3. Log-return noise ablation, context-loss comparison, and long continuations.
4. Generation distributions, joint/temporal structure, and next-event prediction.

All current notebook predictions use models under `runs/log_return/`.
Earlier price-level experiment artifacts remain under `runs/` for historical
traceability and are excluded from current predictions. `data/` and `runs/`
are ignored by Git; preserve the checkpoints, normalization, and reports separately.

## Checks and research limits

```sh
uv run python -m unittest discover -s tests -v
```

Tests cover the fixed log-return target, split boundaries, normalization, anchored
reconstruction, rejection of price-level checkpoints, window order, noise variance,
causality, gradient flow, optimizer ownership, and data-independent sampling.

This is a one-day, one-training-seed-per-setting development study. Validation
has been inspected repeatedly. Further days, repeated seeds, downstream tasks,
and conditional uncertainty checks are needed for broader feasibility conclusions.
