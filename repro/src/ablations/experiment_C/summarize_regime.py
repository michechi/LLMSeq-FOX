"""
Experiment C summary: aggregate the 12 parity-streaming JSONs into a 2x2
(n x regime) table with mean +/- std AUC across seeds, and a small bar chart.

Usage:

    python3 analysis/experiment_C/scripts/summarize_regime.py \
        --results_dir results/parity_streaming \
        --output_dir  analysis/experiment_C/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

REGIME_COLOR = {"streaming": "#1f77b4", "fixed": "#d62728"}


def collect(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(results_dir.glob("expC_*.json")):
        with p.open() as f:
            d = json.load(f)
        rows.append(
            {
                "n": int(d["n"]),
                "regime": d["regime"],
                "n_train": d.get("n_train"),
                "seed": int(d["seed"]),
                "auc": float(d["test_metrics"]["auc"]),
                "f1": float(d["test_metrics"]["f1"]),
                "acc": float(d["test_metrics"]["accuracy"]),
                "final_loss": d.get("final_train_loss"),
                "elapsed_s": d.get("training_time_s"),
                "file": p.name,
            }
        )
    if not rows:
        raise RuntimeError(f"no expC_*.json under {results_dir}")
    return pd.DataFrame(rows)


SUCCESS_THRESHOLD = 0.9


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["regime", "n"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            f1_mean=("f1", "mean"),
            acc_mean=("acc", "mean"),
            n_seeds=("seed", "count"),
            n_success=("auc", lambda s: int((s > SUCCESS_THRESHOLD).sum())),
            success_rate=("auc", lambda s: float((s > SUCCESS_THRESHOLD).mean())),
        )
        .sort_values(["regime", "n"])
    )
    agg["auc_std"] = agg["auc_std"].fillna(0.0)
    return agg


def plot(agg: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    ns = sorted(agg["n"].unique())
    regimes = ["streaming", "fixed"]
    x_positions = np.arange(len(ns))
    width = 0.38

    for i, regime in enumerate(regimes):
        sub = agg[agg["regime"] == regime].set_index("n").reindex(ns)
        means = sub["auc_mean"].to_numpy()
        stds = sub["auc_std"].to_numpy()
        ax.bar(
            x_positions + (i - 0.5) * width,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            color=REGIME_COLOR[regime],
            label=regime,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.40, 1.05)
    ax.set_title(r"Binary parity ($\ell=2$), regime x sequence length")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = []
    for ext in ("pdf", "png"):
        p = out_dir / f"experiment_C_regime_sweep.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment C summary: parity regime sweep")
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/parity_streaming"),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/experiment_C/figures"),
    )
    p.add_argument(
        "--csv_out",
        type=Path,
        default=Path("analysis/experiment_C/figures/experiment_C_aggregate.csv"),
    )
    args = p.parse_args()

    df = collect(args.results_dir)
    logger.info("collected %d runs", len(df))
    agg = aggregate(df)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.csv_out, index=False)
    logger.info("wrote %s", args.csv_out)
    logger.info("\n%s", agg.to_string(index=False))

    per_seed_path = args.csv_out.with_name("experiment_C_per_seed.csv")
    df_sorted = df.sort_values(["regime", "n", "seed"])
    df_sorted.to_csv(per_seed_path, index=False)
    logger.info("wrote %s (%d rows)", per_seed_path, len(df_sorted))

    for path in plot(agg, args.output_dir):
        logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
