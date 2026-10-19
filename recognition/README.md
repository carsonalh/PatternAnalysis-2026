# Recurrent TimeGAN for AMZN limit order books

A university feasibility study using the LOBSTER AMZN level-10 sample for
21 June 2012. The model follows the four-network formulation of the
[TimeGAN paper](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf),
with direct latent recurrence. It supports event sequences and fixed clock
snapshots of the order book.

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

Pass `--sampling-interval-seconds 1` to train on one-second snapshots. At each
whole second, the pipeline selects the latest book available at or before that
time, carrying it forward during quiet seconds. It then recomputes log returns
and fits normalization on training snapshots. The original event-based split
time stays fixed, giving 21,141 training and 2,258 validation snapshots from
269,748 events (91.33% fewer rows). Without the flag, training uses every event.
Saved checkpoints record the sampling interval and split time.

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

Joint training now follows the authors' update schedule: two generator and
embedder/decoder pairs, then a discriminator update only when its pre-update
loss exceeds 0.15. `--generator-updates` and `--discriminator-loss-threshold`
configure this behavior. A zero threshold effectively enables every D update;
the Python configuration also accepts `None` to disable gating. Our loss sums
real and fake BCE; the [authors' reference implementation](https://github.com/jsyoon0823/TimeGAN/blob/master/timegan.py)
adds another fake BCE term through its separate supervisor, so the threshold
is an explicit adaptation. Training logs record actual updates and skips.

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

The [one-second comparison](runs/log_return/one_second_20261010/README.md)
contains six completed loss-weight runs, their three-path prediction plots,
and the recovered T3 Code session requirements. See the
[notebook guide](notebooks/README.md#one-second-order-book-sampling) to reproduce it.

The subsequent [loss-balance experiment](runs/log_return/one_second_balance_20261010/README.md)
restores approximately 94% of validation return volatility and a centered,
approximately lognormal final-price distribution. It uses context weight 0.01,
latent weight 10, return-moment weight 100, and a zero-log-drift prior weighted
at 100 during joint training. A final 1,000 generator-only updates use drift
weight 1,000 and learning rate `1e-5`, with the representation held fixed.
The selected checkpoint is `runs/log_return/one_second_balance_20261010/selected/model.pt`.
This neutral-drift prior expresses the requested forecast median; it is a
modeling assumption rather than an estimate of the day's price trend.

### Optimization guardrails and schedule comparison

```sh
uv run python optimization_experiment.py --device cuda
```

This preserves the selected checkpoint and retrains its full recipe with the
new joint schedule for training seeds 0, 1, and 2. Each keeps the original loss
weights, architecture, 1,000/1,000/5,000 stage budgets and 1,000-step generator
refinement. It then evaluates 1,024 futures per noise seed 7, 17, and 27,
anchored at the first validation snapshot. Decoder returns are never rescaled
or recentered. The output defaults to `runs/log_return/schedule_guardrails_20261010/`;
rerunning resumes completed stages, and `--report-only` reevaluates saved models.
`--render-only` redraws figures from already evaluated arrays without resampling.
Smoke-test overrides are marked as an incomplete protocol and cannot pass the
full study's acceptance check.

Each training seed must satisfy return SD within 10% of validation, median
endpoint bias within $0.05, fitted-normal CDF distance of log endpoints ≤0.03,
and endpoint SD within 20% of $0.812. Raw latent supervised, overall reconstruction,
and return-feature reconstruction MSE must each be no more than 10% above a
fresh evaluation of the saved selected model. MSE comparisons use identical
35 non-overlapping validation windows and ten shared noise draws. Median
confidence intervals express simulation uncertainty, not uncertainty over days.

Reports include raw/weighted loss components, real/fake discriminator BCE and
accuracy, return autocorrelation, log-endpoint shape, and tick-projected book
spread/depth statistics. Every 500 joint cycles, weighted component gradient
norms and cosine alignment with the adversarial gradient are measured before
clipping. These diagnostics reuse the update graph without changing the RNG or
optimizer update. Negative cosine indicates opposing local gradient directions.

The report renders fixed three-path plots, ensemble fans, endpoint histograms,
log-price Q–Q plots and loss curves with shared price axes. Plot metadata records
the anchor, axes, path indices, and pending visual review. Inspect rendered plots
before interpreting results; numerical gates do not replace that check. Failed
experiments remain visible and never automatically replace the selected model.

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

This remains a one-day development study with repeatedly inspected validation
data. Earlier settings mostly use one training seed; the schedule comparison
uses three. Further days, downstream tasks, and conditional uncertainty checks
are needed for broader feasibility conclusions. Discriminator BCE is a balance
diagnostic, and different runs learn different latent coordinates; neither a
lower GAN loss nor tick-valid books independently proves realistic generation.
