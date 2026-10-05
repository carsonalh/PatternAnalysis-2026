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

## Checks and notebooks

```sh
uv run python -m unittest discover -s tests -v
```

Tests cover split boundaries, normalization, window order, Wiener variance and
channel independence, causality, state resets, teacher-forcing alignment,
gradient flow and optimizer ownership, checkpoint sampling, and learning
transitions on synthetic autoregressive sequences.

`notebooks/lobster_autoencoder.ipynb` now trains on nonoverlapping recurrent
windows and compares reconstructed books. Instantiate recovery with
`Decoder()`; the embedder now requires `(batch, time, features)` inputs rather
than independent snapshots. `notebooks/wiener_process.ipynb` demonstrates the
correct square-root scaling of Wiener increments.

`notebooks/lobster_timegan_noise_ablation.ipynb` compares three paired midpoint
continuations at Wiener amplitudes 1, 0.5, 0.25, and 0, with a separate chart
for each case. It reuses one saved checkpoint to isolate inference-time noise
sensitivity; variance scales with the square of amplitude. Full-validation
rollouts retain the training variance per event and extend beyond the trained
window length. The notebook also compares price-change statistics and the
spread between completion endpoints.

`training_balance_experiment.py` runs controlled tests of the initial midpoint
jump with a fresh baseline, discriminator rates `1e-4` and `3e-5`, or two
generator updates per discriminator update. Run each named setting with:

```sh
uv run python training_balance_experiment.py baseline
uv run python training_balance_experiment.py discriminator_1e-4
uv run python training_balance_experiment.py discriminator_3e-5
uv run python training_balance_experiment.py generator_twice
```

All settings use seed 0, identical model sizes and stage budgets, and a separate
sampler RNG to preserve real-batch order. The two-update setting takes twice as
many joint generator updates. Results under `runs/training_balance/` include
checkpoints, losses, diagnostics at 1,000/2,500/5,000 joint iterations, eight
forecast contexts with 32 completions each, and full-validation continuations.
The report notebook `notebooks/lobster_timegan_training_balance.ipynb` shows
the initial 128 events alongside each full-day completion chart.

The optional `--context-loss-weight` adds decoded midpoint supervision to the
generator's joint loss; its default of 0 preserves the original objective.
With `--context-horizon 16`, the first 48 events of each training window form
a real context. The generator then predicts the last 16 events with its own
latent feedback and a Wiener path restarted at zero, preserving training
increment variance. Only the generator is updated by this term; the frozen
decoder transmits gradients to it. The loss is MSE on standardized log midpoint
(feature zero), before tick rounding. It specifically addresses price drift,
rather than supervising every book feature.

Run the controlled comparison at loss weights 10 and 100 with:

```sh
uv run python context_loss_experiment.py --weight 10
uv run python context_loss_experiment.py --weight 100
```

The reference is the previous `discriminator_1e-4` run. Both new runs retain
its architecture, learning rates, noise, real-batch order, and 1,000/1,000/5,000
stage budgets. They reuse the generator update's Wiener path, so enabling the
loss does not consume additional random draws. Each fits only training data;
validation is used for diagnostics. `notebooks/lobster_timegan_context_loss.ipynb`
compares midpoint completions, early drift, noise sensitivity, and ordinary
generated-window statistics. A squared error against one observed future may
favor smoother prices; this is an experiment in correcting the hand-off, not
evidence of calibrated conditional uncertainty. Long rollouts still exceed
the supervised 16-event horizon.

`notebooks/lobster_timegan_return_dynamics.ipynb` investigates the remaining
price reset with price-level, noise-reset, return, and calibrated-innovation
forecasts. Return features replace log midpoint with its event-to-event log
change; all normalization still uses training events. Their prices are
integrated from the last observed midpoint. Return models use Wiener increments
as stationary inputs and repeat the trained 48-context/16-future task, using
only generated history after the starting prefix.

Learning returns can leave a small bias that accumulates. The recommended
**hybrid price model** uses a separate neural volatility head, fitted by Gaussian
negative log likelihood on real training prefixes and their 16-event futures.
Expected arithmetic-price drift is fixed to the empirical training drift.
The lognormal correction preserves that expectation when volatility varies.
TimeGAN generates the book structure; this head generates midpoint innovations.
It therefore removes decoder attraction to an absolute price level, with an
explicit restriction on directional forecasts. Volatility and joint book
dynamics still need validation.

To reproduce the comparison (the previous context-weight-100 run is the
price-level reference):

```sh
uv run python return_price_experiment.py --weight 1
uv run python return_price_experiment.py --weight 10
uv run python calibrate_price_dynamics.py runs/return_price/returns_context_1_moments_10_seed_0/model.pt
uv run python return_price_report.py
```

The unified `runs/return_price/innovation_head_seed_0/model.pt` checkpoint saves
the volatility head and drift calibration. `load_checkpoint` selects that head
automatically for `generate_features` and the experiment's `complete` function.
Standalone return windows use the saved training median as their price anchor;
conditional continuations use the actual last observed midpoint. Existing
price-level checkpoints retain their original sampling behavior.
