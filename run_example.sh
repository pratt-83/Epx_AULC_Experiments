#!/bin/bash
set -euo pipefail

EXPERIMENT_ID="${1:-example_experiment}"
PROCESSED_ROOT="${2:-data/processed}"
OUTPUTS_ROOT="${3:-outputs}"

# run preprocess_data.py separate

python3 scripts/train_observer.py \
	--experiment_id "$EXPERIMENT_ID" \
	--processed_root "$PROCESSED_ROOT" \
	--outputs_root "$OUTPUTS_ROOT" \
	--model transformer \
	--seq_len 2048 \
	--steps 200 \
	--batch_size 4 \
	--eval_every 50 \
	--eval_batches 4 \
	--d_model 128 \
	--n_heads 4 \
	--n_layers 2 \
	--dropout 0.1 \
	--lr 3e-4 \
	--seed 1

python3 scripts/calc_aulc.py \
	--experiment_id "$EXPERIMENT_ID" \
	--outputs_root "$OUTPUTS_ROOT"
