#!/usr/bin/env python3
# calc_aulc.py
#
# PURPOSE
#	Compute AULC-style learnability metrics from the validation loss curves logged by train.py
#	and generate basic figures for the abstract/poster.
#
# CONTEXT IN PROJECT PLAN
#	This is Phase 1 / Q1 measurement extraction:
#	- train.py logs validation loss vs training step for each condition (real/shuf/rand).
#	- This script integrates those curves to compute AULC metrics.
#
# INPUT
#	--csv: CSV produced by train.py with header:
#		step,loss_real,loss_shuf,loss_rand
#
# OUTPUTS
#	1) JSON metrics summary (AULC, AULC-above-final, delta_L, t50)
#	2) Loss curve plot (loss vs step)
#	3) Bar plot of AULC_above_final (often more aligned with "area above final loss" proxy)
#
# METRIC DEFINITIONS
#	Given validation loss curve L(step) evaluated at discrete steps:
#
#	- AULC_raw = ∫ L(step) d(step)
#		This measures overall loss-time exposure.
#
#	- AULC_above_final = ∫ (L(step) - L_T) d(step)
#		This corresponds to the "area above the final loss" heuristic used in prequential-style proxies.
#
#	- delta_L = L0 - LT
#		Magnitude of accessible structure (how much loss drops within budget).
#
#	- t50_step
#		The first step where loss has achieved half of its eventual drop:
#		L(t50) <= L0 - 0.5*(L0-LT)

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
	"""
	Numerical integration by trapezoidal rule.

	Args:
		x: monotonically increasing array of step values
		y: corresponding y-values (loss or loss-offset)

	Returns:
		approximate integral of y over x
	"""
	return float(np.trapz(y, x))


def compute_t50(steps: np.ndarray, loss: np.ndarray) -> float:
	"""
	Compute t50 (first step at which loss has reached half of its eventual drop).

	If the loss does not decrease (delta <= 0), returns NaN.
	If it never reaches the target threshold, returns NaN.
	"""
	L0 = loss[0]
	LT = loss[-1]
	delta = L0 - LT
	if delta <= 0:
		return float("nan")

	target = L0 - 0.5 * delta
	idx = np.where(loss <= target)[0]
	if len(idx) == 0:
		return float("nan")

	return float(steps[idx[0]])


def main():
	# ----------------------------
	# Parse arguments
	# ----------------------------
	ap = argparse.ArgumentParser(description="Compute AULC metrics from loss curve CSV.")
	ap.add_argument("--csv", required=True, help="Path to loss_curves.csv from train.py")
	ap.add_argument("--out_json", default="metrics/aulc_summary.json", help="Where to write JSON summary")
	ap.add_argument("--out_dir", default="figures", help="Directory for figures")
	args = ap.parse_args()

	csv_path = Path(args.csv)
	if not csv_path.exists():
		raise SystemExit(f"CSV not found: {csv_path}")

	# ----------------------------
	# Load CSV (expects header names)
	# ----------------------------
	data = np.genfromtxt(str(csv_path), delimiter=",", names=True)
	steps = data["step"].astype(float)

	curves = {
		"real": data["loss_real"].astype(float),
		"shuf": data["loss_shuf"].astype(float),
		"rand": data["loss_rand"].astype(float),
	}

	# ----------------------------
	# Compute metrics per condition
	# ----------------------------
	out = {}
	for name, loss in curves.items():
		L0 = float(loss[0])
		LT = float(loss[-1])

		out[name] = {
			"L0": L0,
			"LT": LT,
			"delta_L": float(L0 - LT),
			"t50_step": compute_t50(steps, loss),
			"AULC_raw": trapz_auc(steps, loss),
			"AULC_above_final": trapz_auc(steps, loss - LT),
		}

	# ----------------------------
	# Write JSON summary
	# ----------------------------
	out_json = Path(args.out_json)
	out_json.parent.mkdir(parents=True, exist_ok=True)
	out_json.write_text(json.dumps(out, indent=2))
	print("Wrote metrics to:", out_json)

	# ----------------------------
	# Make figures directory
	# ----------------------------
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# ----------------------------
	# Figure 1: loss curves
	# ----------------------------
	plt.figure()
	for name, loss in curves.items():
		plt.plot(steps, loss, label=name)
	plt.xlabel("Training step")
	plt.ylabel("Validation loss")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_dir / "loss_curves.png", dpi=200)

	# ----------------------------
	# Figure 2: AULC above final loss
	#	(closer to “area above final loss” proxy)
	# ----------------------------
	plt.figure()
	labels = ["real", "shuf", "rand"]
	vals = [out[k]["AULC_above_final"] for k in labels]
	plt.bar(labels, vals)
	plt.ylabel("AULC above final loss")
	plt.tight_layout()
	plt.savefig(out_dir / "aulc_bars.png", dpi=200)

	print("Wrote figures to:", out_dir)


if __name__ == "__main__":
	main()

