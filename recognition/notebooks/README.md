# Notebook guide

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
