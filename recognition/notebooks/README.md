# Notebook guide

Start with [the compact TimeGAN/GRU comparison](timegan_summary.ipynb): it trains
both models from scratch, plots KL for all 40 features against training data,
and reports paired continuation SSIM on training windows.
In WSL/Debian, run `uv sync --locked` then `uv run --locked jupyter lab` from `recognition/`,
choose **Python 3 (ipykernel)**, and **Restart Kernel and Run All**.
Its short one-second experiment is independent of
the historical chapters below; exact replay requires the same CPU/environment.

All four chapters use one fixed feature definition: midpoint log return plus
39 spread, gap, and order-size features. The model's midpoint input and output
are always `log(m_t / m_{t-1})`, standardized using training events only.
There is no selectable price format. Prices are recovered only by integrating
predicted log returns from an explicit preceding midpoint.

| Chapter | Notebook | Main question |
|---|---|---|
| 1 | [Data and features](order_book_analysis.ipynb) | Are the fixed log-return features and split boundary correct? |
| 2 | [TimeGAN training and predictions](lobster_timegan.ipynb) | Can TimeGAN reconstruct log returns and continue the training history? |
| 3 | [Log-return dynamics](lobster_timegan_return_dynamics.ipynb) | How do noise and context supervision affect return predictions? |
| 4 | [Generation evaluation](lobster_timegan_return_distributions.ipynb) | Do generated log returns and book structure match real sequences? |

Each chapter runs independently from `recognition/` or `recognition/notebooks/`.
Executed outputs are included for reading without rerunning training.

## Running the notebooks

```sh
uv sync
make run-jupyter
```

Use a kernel with the project environment; `make install-kernel` registers one
if needed. Chapter 1 loads the cached LOBSTER sample or downloads it into `data/`.
Chapters 2–4 load the following log-return models; missing experiments raise an
error instead of loading old price-level weights.

| Chapter | Required artifacts |
|---|---|
| 2 | `runs/log_return/returns_context_1_moments_10_seed_0/model.pt` and `losses.csv` |
| 3 | Both context weights 1/10 under `runs/log_return/`, with diagnostics, snapshots, `long_boundary.json`, `completion_log_returns.npz`, and `completion_midpoints.npz` |
| 4 | Both context-weight checkpoints under `runs/log_return/` |

Chapter 2 allows a short in-memory training run with `LOAD_EXISTING = False`.
It uses the same fixed log-return features and loss/noise settings and does not
replace saved experiments. Its reconstruction examples use each real window's
preceding midpoint. Its prediction chart shows decoder log returns directly,
then their integrated prices starting from the last training midpoint.

Chapter 3 compares context-loss weights within the same log-return space and
recomputes a paired 32-path inference-noise ablation. It records the first
predicted log return and integrated price change at the training boundary.
It does not insert validation prices or remove residual drift.

Chapter 4 samples fresh unconditional windows and runs small next-event
predictors. Every predictor includes all 40 fixed features, so its midpoint
target is always log return. Histograms of projected book changes stay within
each independent window. Prices restart from the training median between windows.

## Reproducing training and reports

Run from the project directory, in this order:

```sh
uv run python return_price_experiment.py --weight 1 --threads 4
uv run python return_price_experiment.py --weight 10 --threads 2
uv run python return_price_report.py
```

Each model uses 1,000 reconstruction, 1,000 transition, and 5,000 joint updates,
seed 0, 64-event windows, stationary Wiener increments, discriminator rate
`1e-4`, and log-return moment weight 10. Context supervision predicts 16 events
after 48 real prefix events. The CPU thread count is recorded in each run's
`run.json`; different thread counts can affect floating-point training results.

Completed directories are protected against accidental retraining. To reproduce
into new locations, pass a different `--output` to training and the corresponding
`--root` to the report command, then update the notebook paths consistently.
The report computes short forecasts, 1,024-event diagnostics, and full-validation
predicted log-return and integrated price paths. Every forecast uses generated
history after the initial real prefix.

`data/` and `runs/` are ignored by Git. Preserve them separately: saved notebook
figures cannot replace checkpoints, feature normalization, and diagnostics.

## Context and latent-loss weight experiments

`loss_weight_experiment.py` retrains the same model while changing the context
penalty and the generator's latent-supervision penalty (`eta`). The embedder's
penalty (`lam=1`), return-moment weight 10, features, noise, and training budgets
stay fixed. For example:

```sh
uv run python loss_weight_experiment.py --context-weight 0.1 --latent-weight 1 --output runs/log_return/loss_weights
uv run python loss_weight_report.py --root runs/log_return/loss_weights --reference runs/log_return/returns_context_1_moments_10_seed_0
```

These commands default to CUDA; pass `--device cpu` for CPU execution. Compare
new settings with a baseline retrained on the same device: CPU and CUDA random
streams and numerical results differ. Each run retains three full-validation
paths with fixed noise draws. The report saves the notebook's two-panel plots,
`comparison.csv`, and 32 conditional futures at each of eight starting contexts.
Return volatility is measured before tick rounding; drift and variation between
paths are reported separately. Validation-based selection is exploratory.

## One-second order-book sampling

The one-second experiment uses the latest book at each whole second, including
quiet seconds, and recomputes returns after sampling. It preserves the original
event-based split time and fits normalization only on training snapshots.
There are 23,399 snapshots: 21,141 training and 2,258 validation. A 64-row window
therefore has one-second steps; the conditional horizon is 16 seconds.

```sh
uv run python loss_weight_experiment.py --context-weight 0.1 --latent-weight 10 --sampling-interval-seconds 1 --output runs/log_return/one_second_reproduction
uv run python loss_weight_report.py --root runs/log_return/one_second_reproduction --device cpu
```

Repeat training with `(context, latent)` weights `(0.1, 10)`, `(0.3, 10)`,
`(1, 10)`, `(0.03, 10)`, `(0.1, 1)`, and `(0.3, 1)` to reproduce the six-run
comparison. Training defaults to CUDA; the report above uses CPU, matching the
saved three-path rollouts. The report writes individual notebook-style figures
and a comparison of all futures with shared axes. Completed runs are protected;
choose a new output directory when retraining.

The completed results and session recovery details are in
[`runs/log_return/one_second_20261010/README.md`](../runs/log_return/one_second_20261010/README.md).
The sampled book export includes each snapshot's original event index and time.
The existing four notebooks retain their original event-based experiments.

## Balanced volatility and final-price spread

To reproduce the improved one-second forecast in a fresh directory:

```sh
uv run python loss_weight_experiment.py --context-weight 0.01 --latent-weight 10 --moment-weight 100 --drift-weight 100 --sampling-interval-seconds 1 --output runs/log_return/balanced_reproduction
uv run python refine_loss_weights.py runs/log_return/balanced_reproduction/context_0.01_latent_10_moments_100_drift_100_seed_0/model.pt --output runs/log_return/balanced_reproduction/refined --steps 1000 --moment-weight 100 --drift-weight 1000 --learning-rate 1e-5
uv run python terminal_price_report.py --root runs/log_return/balanced_reproduction --checkpoint runs/log_return/balanced_reproduction/refined/model.pt --samples 1024 --seed 17
```

These commands default to CUDA. The final pass updates only the generator,
using a smaller learning rate after the decoder has stopped changing. Its
conditional drift penalty matches a zero physical log return, expressed using
training normalization, and averages paired Gaussian innovations to reduce
noise in the estimated expected return. All price paths remain integrated
decoder outputs. The penalty is a neutral-drift modeling prior.

The terminal report observes the first validation snapshot as the starting
context and forecasts the remaining 2,257 seconds. It reports return SD,
median error and its Monte Carlo confidence interval, endpoint dispersion,
upward-path share, and the fitted log-price CDF distance. It draws three paths,
ensemble intervals, the terminal histogram, and a log-price normal Q–Q plot.
Only the histogram overlay is fitted; generated prices are not transformed.

Selected results, the search history, and independent-noise checks are in
[`runs/log_return/one_second_balance_20261010/README.md`](../runs/log_return/one_second_balance_20261010/README.md).

## Historical evidence and limits

The previous price-level checkpoints, optimizer-balance sweeps, and noise-reset
controls remain under their original `runs/` paths as historical evidence. They
are excluded from every current notebook prediction. The checkpoint loader
rejects price-level feature names rather than interpreting old weights as
log-return weights. Historical notebooks can be recovered from Git history.

All generated features come from TimeGAN's decoder. No external price process
or fixed drift is applied. A learned log-return bias can accumulate in long
rollouts and remains a generation error. This one-day, one-training-seed-per-setting
study uses validation during development. Inspect book structure and temporal
checks before extending its feasibility conclusions.
