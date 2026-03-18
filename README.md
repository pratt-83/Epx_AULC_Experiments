# Genomic Epiplexity Pilot

A medium-refactor experimental codebase for preliminary epiplexity measurements on genomic sequence using shared-observer loss curves and AULC-style metrics.

## Project purpose

This project asks a learnability question:

Given the same observer architecture and matched optimization budget, do real genomic windows show more favorable predictive learning dynamics than matched control sequences?

The current pilot uses three sequence conditions:

- `real`: sampled genomic DNA windows
- `shuf`: dinucleotide-preserving shuffled controls
- `rand`: GC-matched i.i.d. random controls

A small causal sequence model is trained as a shared observer across these conditions, and validation loss curves are logged separately for each condition. These curves are then summarized using AULC-style metrics.

## High-level workflow

1. **Preprocess genomic windows**
	- Parse `.fna` FASTA files from `data/raw/`
	- Sample fixed-length windows from real genomes
	- Generate matched control windows
	- Split into train and validation sets
	- Write processed datasets to `data/processed/[experiment_id]/`

2. **Train shared observer**
	- Load the six processed files:
		- `real_train.txt`
		- `real_val.txt`
		- `shuf_train.txt`
		- `shuf_val.txt`
		- `rand_train.txt`
		- `rand_val.txt`
	- Train one small causal observer on a balanced mixture of conditions
	- Log validation loss curves to `outputs/[experiment_id]/logs/`

3. **Compute epiplexity-style metrics**
	- Read logged loss curves
	- Compute:
		- `AULC_raw`
		- `AULC_above_final`
		- `delta_L`
		- `t50_step`
	- Write JSON metrics and figures to `outputs/[experiment_id]/`

## Directory layout

```text
README.md

src/
	data_utils.py
	models.py
	epiplexity_metrics.py

scripts/
	preprocess_data.py
	train_observer.py
	calc_aulc.py

data/
	raw/
	processed/

outputs/
	[experiment_id]/
		logs/
		metrics/
		figures/
		ckpt/

slurm/
	run_shared_observer.sh