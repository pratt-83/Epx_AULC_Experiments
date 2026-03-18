# Epiplexity Heuristic AULC Experiments

This repository is an experimental workspace for measuring epiplexity-style learnability from validation-loss trajectories on genomic sequence. The pipeline builds matched real and control datasets, trains a shared observer across conditions, logs per-condition validation curves, and summarizes those curves with AULC-style metrics.

The codebase currently supports two observer backends:

- `transformer`: a small causal Transformer baseline
- `hyena`: a small causal Hyena language model for longer-range sequence modeling

## Current experiment design

The core question is whether real genomic windows are easier to learn under a fixed optimization budget than matched controls.

Each experiment uses three sequence conditions:

- `real`: sampled genomic windows from `.fna` inputs
- `shuf`: dinucleotide-preserving shuffled controls derived from the real windows
- `rand`: GC-matched i.i.d. random controls derived from the real windows

The training loop uses one shared observer. At each optimization step, the script samples one of the three conditions uniformly, draws a batch from that condition, and trains under a next-token prediction objective. Validation is logged separately for `real`, `shuf`, and `rand`.

## Repository layout

```text
README.md
run_example.sh
run_microbial_20k_2048bp_v1.sh	# not tracked; [experiment_id] = microbial_20k_2048bp_v1


data/
  README.md
  raw/                   # not tracked
  processed/             # not tracked

scripts/
  preprocess_data.py
  train_observer.py
  calc_aulc.py

src/
  data_utils.py
  models.py
  hyena_layers.py
  hyena_observer.py
  epiplexity_metrics.py

outputs/                 # not tracked
  [experiment_id]/
    logs/
    metrics/
    figures/
    ckpt/
```

Raw data, processed datasets, and outputs are excluded from Git. See `data/README.md` for the expected data layout.

## Dependencies

The current code assumes a Python environment with:

- `torch`
- `numpy`
- `matplotlib`
- `ushuffle`

## Pipeline overview

### 1. Preprocess data

`scripts/preprocess_data.py`:

- finds `.fna` files under `data/raw/`
- samples fixed-length real genomic windows
- generates matched shuffled and random controls
- applies a shared train/validation split
- writes processed text files and `meta.tsv`

Expected processed dataset:

```text
data/processed/[experiment_id]/
  real_train.txt
  real_val.txt
  shuf_train.txt
  shuf_val.txt
  rand_train.txt
  rand_val.txt
  meta.tsv
```

### 2. Train shared observer

`scripts/train_observer.py`:

- loads one processed dataset
- selects either the Transformer or Hyena observer
- trains with next-token cross-entropy loss
- logs a true pre-training validation evaluation at step `0`
- evaluates `real`, `shuf`, and `rand` separately during training
- optionally saves a final checkpoint

Main outputs:

```text
outputs/[experiment_id]/
  logs/loss_curves.csv
  ckpt/[model]_observer.pt   # if --save_ckpt
```

Loss CSV schema:

```text
step,loss_real,loss_shuf,loss_rand
```

### 3. Compute AULC metrics and plots

`scripts/calc_aulc.py` reads `logs/loss_curves.csv`, computes summary metrics, and writes:

```text
outputs/[experiment_id]/
  metrics/aulc_summary.json
  metrics/aulc_summary.txt
  figures/loss_curves.png
  figures/aulc_bars.png
```

## Metrics currently reported

For each condition, the current implementation reports:

- `L0_pre`: validation loss before training begins
- `L_start_obs`: first post-update validation loss
- `LT_obs`: final observed validation loss
- `delta_L_from_pre`: total loss reduction from pre-training to the final observed point
- `delta_L_obs`: total loss reduction from the first observed post-update point to the final observed point
- `t50_step_from_pre`: first logged step reaching half of the total drop from the pre-training baseline
- `t50_step_obs`: first logged step reaching half of the total drop from the first post-update baseline
- `AULC_raw`: trapezoidal area under the full logged validation-loss curve
- `AULC_above_final_observed`: area between the loss curve and the final observed validation loss
