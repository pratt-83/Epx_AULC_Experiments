#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from src.epiplexity_metrics import load_loss_curve_csv, summarize_all_curves


# ------------------------------------------------------------
# main metrics entrypoint
# ------------------------------------------------------------

def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compute AULC-style metrics and figures from logged loss curves."
	)
	parser.add_argument(
		"--experiment_id",
		required=True,
		help="Experiment tag used for outputs/[experiment_id]/",
	)
	parser.add_argument(
		"--outputs_root",
		default="outputs",
		help="Root directory containing outputs/[experiment_id]/",
	)
	args = parser.parse_args()

	experiment_dir = Path(args.outputs_root) / args.experiment_id
	log_dir = experiment_dir / "logs"
	metrics_dir = experiment_dir / "metrics"
	figures_dir = experiment_dir / "figures"

	loss_csv_path = log_dir / "loss_curves.csv"
	summary_json_path = metrics_dir / "aulc_summary.json"

	metrics_dir.mkdir(parents=True, exist_ok=True)
	figures_dir.mkdir(parents=True, exist_ok=True)

	steps, curves = load_loss_curve_csv(loss_csv_path)
	summary = summarize_all_curves(steps, curves)

	summary_json_path.write_text(json.dumps(summary, indent=2))
	print(f"Metrics written to: {summary_json_path}")

	plt.figure()
	for condition_name, loss_curve in curves.items():
		plt.plot(steps, loss_curve, label=condition_name)
	plt.xlabel("Training step")
	plt.ylabel("Validation loss")
	plt.legend()
	plt.tight_layout()
	plt.savefig(figures_dir / "loss_curves.png", dpi=200)

	plt.figure()
	labels = list(curves.keys())
	values = [summary[label]["AULC_above_final"] for label in labels]
	plt.bar(labels, values)
	plt.ylabel("AULC above final loss")
	plt.xticks(rotation=30, ha="right")
	plt.tight_layout()
	plt.savefig(figures_dir / "aulc_bars.png", dpi=200)

	print(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
	main()
	