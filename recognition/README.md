# Recurrent TimeGAN for AMZN limit order books

A university feasibility study using one day of LOBSTER AMZN level-10 data
(2012-06-21). This baseline follows the four-network formulation of the
[TimeGAN paper](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf),
with direct latent recurrence and no static features. It models event order,
not the elapsed time between orders.

## Model and data

The 40 inputs are standardized log midpoint, spread, price gaps, and sizes.
The first 85% of events are training data; the last 15% are held out. Both
splits use normalization fitted only on training events. Windows are formed
after splitting, and shuffling changes window order without changing event
order inside a window. Training windows overlap with stride 1; evaluation
uses nonoverlapping windows distributed across each split.

- `Embedder`: a causal GRU, `40 → 128`, producing one latent per event.
- `Decoder`: an MLP, `128 → 256 → 256 → 40`, with linear feature outputs.
- `Generator`: `GRUCell(128, 128)` whose recurrent state is the latent.
- `Discriminator`: a bidirectional GRU with 64 units per direction and one
  linear real/fake logit per event.

Each latent summarizes its history, but the model retains the entire latent
sequence. The 128-dimensional representation expands each event's 40 inputs;
it is not a smaller per-event bottleneck.

`Generator.sample(paths, seed)` feeds generated latents back. The first latent
is `tanh(Linear(seed))`, using independent Gaussian noise, not static covariates.
`Generator.teacher_forced(latents, paths)` uses the real previous latent with
the same cell weights. Wiener paths have 128 independent channels, start at
zero, and use increments with standard deviation `sqrt(1 / (T - 1))` on a
normalized event grid spanning `[0, 1]`.

## Training

```sh
uv run python main.py
```

Defaults are 64 events per window, batch size 64, Adam at `3e-4`, and gradient
clipping at norm 1. The data ZIP is downloaded into `data/` if it is not cached.
CUDA is selected when available; use `--device cpu` to select CPU explicitly.

There are three stages:

1. 1,000 reconstruction updates of the embedder and decoder.
2. 1,000 supervised transition updates of the generator with the embedder frozen.
3. 5,000 joint iterations: one discriminator update, one generator update, and
   one embedder/decoder update, with fresh noise and graphs for each update.

The losses are `L_R = MSE(R(E(X)), X)` and
`L_S = MSE(G.teacher_forced(H, W), H[:, 1:])`. During joint training, the
generator minimizes real-label BCE plus `10 * L_S`; the embedder/decoder
minimize `L_R + L_S`. Freezing a network's parameters does not disable gradients
through its inputs. The random initializer is trained by the adversarial loss,
not transition pretraining.

The objectives follow [Supplementary Algorithm 1](https://www.vanderschaar-lab.com/papers/NIPS2019_TGAN_Supplementary.pdf).
The warm-up stages and non-saturating generator loss are practical training
choices. This baseline does not reproduce the separate supervisor or
moment-matching losses in the authors' released implementation.

For a quick pipeline check:

```sh
uv run python main.py --autoencoder-steps 10 --transition-steps 10 --joint-steps 10 --evaluation-samples 16 --predictor-steps 20 --output runs/smoke
```

Step budgets, loss weights, learning rate, batch size, sequence length, noise
dimension, and latent dimension are configurable; see `python main.py --help`.
`--discriminator-learning-rate` overrides only the discriminator's rate;
otherwise all networks use `--learning-rate`. `--generator-updates` controls
the number of generator updates per joint iteration (default 1). Each takes
fresh noise; logged generator losses are averaged across those updates.
The default output directory is `runs/timegan/`:

- `model.pt`: all network weights, model configuration, and training normalization.
- `losses.csv`: every update's losses, labeled by training stage.
- `evaluation.json`: held-out reconstruction/transition losses, feature statistics,
  projected book statistics, and predictive scores.
- `generated_features.npy`: generated standardized feature windows.
- `sample_order_book.csv`: one generated window in original LOBSTER units.

`--skip-evaluation` saves just weights and loss history. Checkpoints support
sampling; they do not store optimizer state for training resumption.

## Sampling and evaluation

```sh
uv run python predict.py runs/timegan/model.pt --samples 16 --output runs/samples
```

Sampling needs no real data or network access. It saves standardized features
and one decoded CSV per window. The saved normalization is reused, and the
checkpoint's window length determines the event grid. Tick rounding and
positive spread/gap/size constraints are applied only during CSV conversion,
outside the differentiable model.

The evaluation report compares feature means, standard deviations, changes,
lag-one correlations, cross-feature correlations, and variation across window
means before book conversion. After conversion, it reports spread, depth,
and midpoint-change statistics. Correlations involving constant features are
reported as zero. Book projection enforces valid snapshots; it does not prove
that generated temporal dynamics are realistic.

Predictive evaluation trains two small GRU next-step predictors with identical
initialization and update budgets: one on real training windows and one on
generated windows. Both are scored by MAE on held-out real windows, in
standardized feature units. Defaults use up to 256 windows per split and 200
predictor updates. These are small feasibility checks, not evidence of
out-of-day generalization or guaranteed generation quality.

## Checks

```sh
uv run python -m unittest discover -s tests -v
```

Tests cover split boundaries, normalization, window order, Wiener variance and
channel independence, causality, state resets, teacher-forcing alignment,
gradient flow and optimizer ownership, checkpoint sampling, and learning
transitions on synthetic autoregressive sequences.

## Notebook reading order

The four notebooks form one feasibility review. They have executed plots and
tables and can also be run independently from the project or notebook directory.
The [notebook guide](notebooks/README.md) lists required artifacts and reproduction
commands. Training is opt-in; existing experimental outputs remain in `runs/`.

1. [AMZN data and features](notebooks/order_book_analysis.ipynb): the trading day,
   spread and gap structure, opening outliers, training-only normalization, and
   conversion checks for price-level and return features.
2. [Baseline TimeGAN](notebooks/lobster_timegan.ipynb): architecture, stage losses,
   current-model reconstruction checks, and an exploratory completion overview.
3. [Price dynamics experiments](notebooks/lobster_timegan_return_dynamics.ipynb):
   paired inference-noise ablation, training balance, context loss, and
   decoder-generated returns, with successful and unsuccessful controls
   in shared comparisons.
4. [Generation evaluation](notebooks/lobster_timegan_return_distributions.ipynb):
   price-level versus return TimeGAN midpoint and gap returns, histogram KL, spread/depth,
   temporal and joint structure, and a common-feature prediction check.

The earlier statistics, preprocessing, autoencoder, and Wiener demonstrations
are incorporated into these chapters. Separate noise, balance, and context-loss
notebooks are consolidated into chapter 3. The unrelated MNIST and volatility
learning exercises are removed; their prior versions remain in Git history.

## Price dynamics variants

The optional `--context-loss-weight` adds decoded midpoint supervision to the
joint generator objective. Its default of 0 preserves the baseline. With
`--context-horizon 16`, a 48-event real prefix conditions 16 generated events.
Only the generator is updated by this term; gradients pass through the frozen
decoder. This improves the handoff in the measured runs but does not establish
realistic longer continuations or calibrated uncertainty.

The return variant replaces only log midpoint with its event-to-event log
return and uses stationary Wiener increments. The other 39 features remain
unchanged. Prices integrate from the starting midpoint. The return experiments
also change context supervision and add return moment matching, so their
comparison with the baseline does not isolate representation alone.

All generated features, including midpoint returns, come from the TimeGAN
**decoder**. The return checkpoint emits a standardized log return as feature
zero. Reverse training normalization and accumulate these returns from a
starting midpoint to recover prices. No separate price model replaces this
output and no expected drift is imposed at inference. A learned return bias
can still accumulate and must be reported as a generation error.

The current return example uses
`runs/return_price/returns_context_1_moments_10_seed_0/model.pt`:

```sh
uv run python predict.py runs/return_price/returns_context_1_moments_10_seed_0/model.pt --samples 16 --output runs/return_samples
```

Standalone return windows each use the saved training median as their price
anchor; conditional continuations use the last observed midpoint. Return
accumulation resets between independent windows. The price-level baseline
remains a control. Checkpoints from the removed external-price experiment
are rejected rather than silently replacing the decoder midpoint output.

These are one-day, one-training-seed development comparisons. Validation has
been repeatedly inspected while selecting interventions, so it is not an
untouched final test set. A corrected price reset alone is not evidence of
realistic full-book generation or out-of-day generalization.
