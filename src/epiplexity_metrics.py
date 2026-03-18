#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ------------------------------------------------------------
# curve summaries
# ------------------------------------------------------------

def trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
	"""
	Compute area under a curve with the trapezoidal rule.
	"""
	return float(np.trapz(y, x))


def compute_t50(steps: np.ndarray, loss: np.ndarray) -> float:
	"""
	Return the first step where the loss has achieved half of its eventual drop.

	If the loss does not decrease, or never reaches the halfway point, return NaN.
	"""
	loss_start = loss[0]
	loss_final = loss[-1]
	loss_drop = loss_start - loss_final

	if loss_drop <= 0:
		return float("nan")

	halfway_loss = loss_start - 0.5 * loss_drop
	indices = np.where(loss <= halfway_loss)[0]

	if len(indices) == 0:
		return float("nan")

	return float(steps[indices[0]])


def summarize_curve(steps: np.ndarray, loss: np.ndarray) -> Dict[str, float]:
	"""
	Summarize one loss curve into the core pilot metrics.
	"""
	loss_start = float(loss[0])
	loss_final = float(loss[-1])

	return {
		"L0": loss_start,
		"LT": loss_final,
		"delta_L": float(loss_start - loss_final),
		"t50_step": compute_t50(steps, loss),
		"AULC_raw": trapz_auc(steps, loss),
		"AULC_above_final": trapz_auc(steps, loss - loss_final),
	}


# ------------------------------------------------------------
# CSV loading
# ------------------------------------------------------------

def load_loss_curve_csv(csv_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
	"""
	Load the standard loss-curve CSV written by train_observer.py.

	Expected columns:
		step,loss_real,loss_shuf,loss_rand
	"""
	if not csv_path.exists():
		raise FileNotFoundError(f"Loss curve CSV not found: {csv_path}")

	data = np.genfromtxt(str(csv_path), delimiter=",", names=True)
	steps = data["step"].astype(float)

	curves = {
		"real": data["loss_real"].astype(float),
		"shuf": data["loss_shuf"].astype(float),
		"rand": data["loss_rand"].astype(float),
	}

	return steps, curves


# ------------------------------------------------------------
# multi-curve summaries
# ------------------------------------------------------------

def summarize_all_curves(steps: np.ndarray, curves: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
	"""
	Summarize all named curves in one dictionary.
	"""
	summary: Dict[str, Dict[str, float]] = {}

	for condition_name, loss_curve in curves.items():
		summary[condition_name] = summarize_curve(steps, loss_curve)

	return summary

