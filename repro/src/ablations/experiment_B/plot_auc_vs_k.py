"""
Experiment B figure: test AUC vs subset size |K| at fixed alphabet ell=26.

Scans the same results directory as Experiment A, filters to anchor runs at
ell=26, and produces one figure (AUC vs k, three curves: Transformer, LSTM,
BERT).

Reuses Experiment A's (ell=26, k=13) runs so we do not regenerate them.

Usage:

    python3 analysis/experiment_B/scripts/plot_auc_vs_k.py \
        --results_dir results/parity_decomp \
        --output_dir  analysis/experiment_B/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

FIXED_ELL = 26
MODELS_ORDER = ("Transformer", "LSTM", "BERT")
MODEL_COLORS = {
    "Transformer": "#1f77b4",
    "LSTM": "#2ca02c",
    "BERT": "#d62728",
}
MODEL_MARKERS = {
    "Transformer": "o",
    "LSTM": "s",
    "BERT": "D",
}

ANCHOR_RE = re.compile(r"^anchor_l(\d+)_k(\d+)_([A-Za-z0-9\-]+)_(\d+)\.json$")


def _normalize_model(name: str) -> str:
    n = name.lower()
    if n.startswith("transformer"):
        return "Transformer"
    if n.startswith("lstm"):
        return "LSTM"
    if n.startswith("bert"):
        return "BERT"
    if n.startswith("llama"):
        return "Llama"
    return name


def collect_runs(results_dir: Path, ell: int) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(results_dir.glob(f"anchor_l{ell}_k*_*.json")):
        m = ANCHOR_RE.match(p.name)
        if m is None:
            logger.warning("skipping unparseable filename: %s", p.name)
            continue
        ell_f, k, model_tok, seed = m.group(1), m.group(2), m.group(3), m.group(4)
        if int(ell_f) != ell:
            continue
        with p.open() as f:
            data = json.load(f)
        test = data.get("test_metrics", {})
        auc = test.get("auc")
        if auc is None:
            logger.warning("skipping %s: no test_metrics.auc", p.name)
            continue
        rows.append(
            {
                "ell": int(ell_f),
                "k": int(k),
                "p": int(k) / int(ell_f),
                "model_raw": model_tok,
                "model": _normalize_model(model_tok),
                "seed": int(seed),
                "auc": float(auc),
                "f1": float(test.get("f1", np.nan)),
                "file": p.name,
            }
        )
    if not rows:
        raise RuntimeError(f"no anchor runs at ell={ell} found under {results_dir}")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["model", "k"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            f1_mean=("f1", "mean"),
            n_seeds=("seed", "count"),
        )
        .sort_values(["model", "k"])
    )
    agg["auc_std"] = agg["auc_std"].fillna(0.0)
    return agg


def plot(agg: pd.DataFrame, ell: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ks_all = sorted(agg["k"].unique())

    for model in MODELS_ORDER:
        sub = agg[agg["model"] == model].sort_values("k")
        if sub.empty:
            continue
        ks = sub["k"].to_numpy()
        means = sub["auc_mean"].to_numpy()
        stds = sub["auc_std"].to_numpy()
        color = MODEL_COLORS[model]
        ax.plot(
            ks,
            means,
            marker=MODEL_MARKERS[model],
            color=color,
            linewidth=2,
            label=model,
        )
        ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.20)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.axvline(ell / 2, color="lightgray", linestyle=":", linewidth=1, label=rf"balanced $|K|=\ell/2$")
    ax.set_xticks(ks_all)
    ax.set_xticklabels([str(k) for k in ks_all])
    ax.set_xlabel(r"Subset size $|K|$")
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.40, 1.02)
    ax.set_title(rf"Parity, $\ell={ell}$, subset-size sweep  ($n=20$, 100K train)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()

    paths: list[Path] = []
    for ext in ("pdf", "png"):
        out_path = out_dir / f"experiment_B_auc_vs_k.{ext}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Plot AUC vs |K| for Experiment B (parity subset-size sweep)")
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/parity_decomp"),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/experiment_B/figures"),
    )
    p.add_argument(
        "--csv_out",
        type=Path,
        default=Path("analysis/experiment_B/figures/experiment_B_aggregate.csv"),
    )
    p.add_argument("--ell", type=int, default=FIXED_ELL)
    args = p.parse_args()

    df = collect_runs(args.results_dir, args.ell)
    logger.info("collected %d runs at ell=%d", len(df), args.ell)
    logger.info("models found: %s", sorted(df["model"].unique()))
    logger.info("ks found:     %s", sorted(df["k"].unique()))

    agg = aggregate(df)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.csv_out, index=False)
    logger.info("wrote aggregate CSV: %s", args.csv_out)

    paths = plot(agg, args.ell, args.output_dir)
    for p_ in paths:
        logger.info("wrote figure: %s", p_)


if __name__ == "__main__":
    main()
