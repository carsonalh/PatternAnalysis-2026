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
