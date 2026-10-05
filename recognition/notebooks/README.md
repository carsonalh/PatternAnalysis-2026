# Notebook guide

Read these four chapters in order. Their filenames are retained so existing
links to the main notebooks still work. Each chapter also runs independently
from `recognition/` or `recognition/notebooks/` with the project environment.
Executed outputs are included for reading without retraining.

| Chapter | Notebook | Main question |
|---|---|---|
| 1 | [Data and features](order_book_analysis.ipynb) | What book structure and data transformations must generation preserve? |
| 2 | [Baseline TimeGAN](lobster_timegan.ipynb) | Can the model reconstruct real books and generate plausible sequences? |
| 3 | [Price dynamics experiments](lobster_timegan_return_dynamics.ipynb) | What causes the initial drift, and which corrections address it? |
| 4 | [Generation evaluation](lobster_timegan_return_distributions.ipynb) | How do the price-level and return TimeGAN compare with real data beyond the price reset? |

## Running the notebooks

Install the dependencies and start Jupyter from the project directory:

```sh
uv sync
make run-jupyter
```

Select a kernel using the project environment; `make install-kernel` registers
one if needed. Chapter 1 loads the LOBSTER sample, downloading it into `data/`
when it is not already cached. Other chapters need the saved models and reports
listed below. They do not silently retrain missing experiments.

| Chapter | Required local artifacts |
|---|---|
| 1 | LOBSTER sample in `data/`, or access to download it |
| 2 | `runs/timegan/model.pt` and `losses.csv`; set `LOAD_EXISTING = False` for an in-memory exploratory run |
| 3 | Original baseline; four `runs/training_balance/` settings; context weights 10/100 under `runs/context_loss/`; return weights 1/10, reset control, and previous-level control under `runs/return_price/` |
| 4 | Original baseline and `runs/return_price/returns_context_1_moments_10_seed_0/model.pt` |

Chapter 3 reads final diagnostics (or the final numbered snapshot), intermediate
`snapshots.csv`, the return controls' `long_boundary.json`, and the return TimeGAN's
`completion_midpoints.npz`. Missing 16-event metrics in older original/balance
reports are recomputed from their matching checkpoints without overwriting
those reports. The short noise ablation is recomputed with 32 paired paths.

Chapter 4 samples fresh unconditional windows from both checkpoints, decodes
each independently, and runs three small next-event predictors. It fits no new
TimeGAN and writes no run artifacts. Its predictors compare
only the common 39 structure features, keeping log prices and log returns out
of the same score.

## Reproducing training and reports

The existing experiment scripts preserve the original settings and saved
evidence. Run from the project directory, in this dependency order:

```sh
uv run python main.py

uv run python training_balance_experiment.py baseline
uv run python training_balance_experiment.py discriminator_1e-4
uv run python training_balance_experiment.py discriminator_3e-5
uv run python training_balance_experiment.py generator_twice

uv run python context_loss_experiment.py --weight 10
uv run python context_loss_experiment.py --weight 100

uv run python return_price_experiment.py --weight 1
uv run python return_price_experiment.py --weight 10
uv run python return_price_report.py
```

The experiment scripts reject completed output directories;
reuse their existing outputs for this local study. New training may differ
from historical runs with another environment or device. To compare new runs,
choose separate output locations and update the notebook paths and report
script references consistently. `return_price_report.py` uses the default
locations, copies the context-weight-100 reference into its control folders,
and regenerates their reports.

`data/` and `runs/` are ignored by Git. Preserve them separately: committed
notebook figures are readable evidence, but cannot replace the checkpoints,
normalization, and diagnostics needed for reproduction.

## How earlier notebooks were consolidated

| Earlier notebook | Destination |
|---|---|
| `lobster_statistics`, `lobster_preprocessing` | Chapter 1: statistics, feature explanation, and conversion checks |
| `lobster_autoencoder` | Chapter 2: reconstruction checks using the current TimeGAN |
| `wiener_process` | Chapter 2: path-versus-increment explanation |
| `lobster_timegan_noise_ablation`, `lobster_timegan_training_balance`, `lobster_timegan_context_loss` | Chapter 3: controlled comparisons, intermediate training evidence, and limitations |
| `embedding_experiment`, `volatility` | Removed learning exercises; prior versions remain in Git history |

The failed controls remain part of the research story. TimeGAN's decoder
generates midpoint returns together with the other 39 features; prices follow
by accumulating those returns. Any remaining drift is a generation error.
No external price process or fixed drift is used. This one-day,
one-training-seed study uses validation throughout development. Interpret
chapter 4's structure and temporal checks before drawing feasibility conclusions.

The removed external-price experiment's saved artifacts remain under `runs/`
for historical traceability, but are excluded from these notebooks and cannot
be sampled through `predict.py`. Both displayed generation variants use only
TimeGAN's generator and decoder.
