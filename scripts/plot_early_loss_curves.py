#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.epiplexity_metrics import load_loss_curve_csv


def load_and_trim(
    outputs_root: Path,
    experiment_id: str,
    max_step: int,
):
    loss_csv_path = (
        outputs_root
        / experiment_id
        / "logs"
        / "loss_curves.csv"
    )

    steps, curves = load_loss_curve_csv(loss_csv_path)

    keep_indices = [
        index
        for index, step in enumerate(steps)
        if step <= max_step
    ]

    trimmed_steps = [steps[index] for index in keep_indices]

    trimmed_curves = {
        condition_name: [
            loss_curve[index]
            for index in keep_indices
        ]
        for condition_name, loss_curve in curves.items()
    }

    return trimmed_steps, trimmed_curves


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Transformer and Hyena validation-loss curves "
            "up to a selected training step."
        )
    )
    parser.add_argument(
        "--transformer_root",
        default="results_transformer",
        help="Root directory containing Transformer results.",
    )
    parser.add_argument(
        "--hyena_root",
        default="results_hyena",
        help="Root directory containing Hyena results.",
    )
    parser.add_argument(
        "--experiment_id",
        default="microbial_annotation_512bp_v1",
        help="Experiment directory name.",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=5000,
        help="Maximum training step to include.",
    )
    parser.add_argument(
        "--output",
        default="annotation_loss_curves_first_5000.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    transformer_steps, transformer_curves = load_and_trim(
        Path(args.transformer_root),
        args.experiment_id,
        args.max_step,
    )

    hyena_steps, hyena_curves = load_and_trim(
        Path(args.hyena_root),
        args.experiment_id,
        args.max_step,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        sharey=False,
    )

    for condition_name, loss_curve in transformer_curves.items():
        axes[0].plot(
            transformer_steps,
            loss_curve,
            label=condition_name,
        )

    axes[0].set_title(
        "TRANSFORMER",
        fontweight="bold",
        fontsize=16,
    )
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Validation loss")
    axes[0].legend()

    for condition_name, loss_curve in hyena_curves.items():
        axes[1].plot(
            hyena_steps,
            loss_curve,
            label=condition_name,
        )

    axes[1].set_title(
        "HYENA",
        fontweight="bold",
        fontsize=16,
    )
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Validation loss")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(args.output, dpi=200)
    print(f"Figure written to: {args.output}")


if __name__ == "__main__":
    main()
