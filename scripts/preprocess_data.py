#!/usr/bin/env python3

import argparse
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from src.data_utils import (
	build_contig_sampling_weights,
	generate_fixed_controls,
	load_eligible_contigs,
	sample_real_windows,
	split_indices,
	write_meta_tsv,
	write_windows,
)


# ------------------------------------------------------------
# main preprocessing entrypoint
# ------------------------------------------------------------

def main() -> None:
	parser = argparse.ArgumentParser(
		description="Build real/shuf/rand datasets for shared-observer epiplexity experiments."
	)
	parser.add_argument(
		"--experiment_id",
		required=True,
		help="Experiment tag used for data/processed/[experiment_id]/",
	)
	parser.add_argument(
		"--raw_dir",
		default="data/raw",
		help="Directory containing raw .fna files",
	)
	parser.add_argument(
		"--processed_root",
		default="data/processed",
		help="Root directory for processed datasets",
	)
	parser.add_argument(
		"--window_bp",
		type=int,
		default=2048,
		help="Window length in base pairs",
	)
	parser.add_argument(
		"--n_windows",
		type=int,
		default=15000,
		help="Number of real windows to sample",
	)
	parser.add_argument(
		"--val_frac",
		type=float,
		default=0.1,
		help="Validation fraction",
	)
	parser.add_argument(
		"--max_n_frac",
		type=float,
		default=0.05,
		help="Reject windows with N fraction above this threshold",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=1,
		help="Random seed",
	)
	args = parser.parse_args()

	rng = random.Random(args.seed)

	raw_dir = Path(args.raw_dir)
	processed_dir = Path(args.processed_root) / args.experiment_id
	processed_dir.mkdir(parents=True, exist_ok=True)

	fasta_files, contigs = load_eligible_contigs(raw_dir, args.window_bp)
	contig_weights = build_contig_sampling_weights(contigs, args.window_bp)

	# Sample the accepted real windows first. The controls are derived only
	# after this filtering step so they remain matched to the final real set.
	real_windows, meta_rows, total_attempts, rejected_for_n = sample_real_windows(
		contigs=contigs,
		weights=contig_weights,
		window_bp=args.window_bp,
		n_windows=args.n_windows,
		max_n_frac=args.max_n_frac,
		rng=rng,
	)

	shuf_windows, rand_windows = generate_fixed_controls(real_windows, rng)

	# Reuse one shared split across all three conditions so window i in each
	# condition ends up in the same partition.
	n_windows_total = len(real_windows)
	train_indices, val_indices = split_indices(n_windows_total, args.val_frac, rng)

	real_train = [real_windows[i] for i in range(n_windows_total) if i in train_indices]
	real_val = [real_windows[i] for i in range(n_windows_total) if i in val_indices]

	shuf_train = [shuf_windows[i] for i in range(n_windows_total) if i in train_indices]
	shuf_val = [shuf_windows[i] for i in range(n_windows_total) if i in val_indices]

	rand_train = [rand_windows[i] for i in range(n_windows_total) if i in train_indices]
	rand_val = [rand_windows[i] for i in range(n_windows_total) if i in val_indices]

	write_windows(processed_dir / "real_train.txt", real_train)
	write_windows(processed_dir / "real_val.txt", real_val)
	write_windows(processed_dir / "shuf_train.txt", shuf_train)
	write_windows(processed_dir / "shuf_val.txt", shuf_val)
	write_windows(processed_dir / "rand_train.txt", rand_train)
	write_windows(processed_dir / "rand_val.txt", rand_val)
	write_meta_tsv(processed_dir / "meta.tsv", meta_rows)

	print(f"Experiment ID: {args.experiment_id}")
	print(f"Raw FASTA directory: {raw_dir}")
	print(f"Found {len(fasta_files)} .fna files")
	print(f"Eligible contigs: {len(contigs)}")
	print(f"Accepted real windows: {len(real_windows)}")
	print(f"Rejected for N content: {rejected_for_n}")
	print(f"Total sampling attempts: {total_attempts}")
	print(f"Processed data written to: {processed_dir}")


if __name__ == "__main__":
	main()

