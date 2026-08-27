"""
Experiment A figure: test AUC vs alphabet size ell for parity at |K|=ell/2.

Scans a results directory for anchor run JSONs emitted by
parity_decomposition_dl.py and parity_decomposition_bert.py, aggregates by
(model, ell, seed), and writes both an aggregate CSV and a PDF/PNG figure with
three curves (Transformer, LSTM, BERT) and shaded std bands across seeds.

Usage (CPU-only, runs on login or a small CPU SLURM allocation):

    python3 analysis/experiment_A/scripts/plot_auc_vs_ell.py \
        --results_dir results/parity_decomp \
        --output_dir  analysis/experiment_A/figures
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

# Filename pattern: anchor_l{ell}_k{k}_{ModelName}_{seed}.json
ANCHOR_RE = re.compile(r"^anchor_l(\d+)_k(\d+)_([A-Za-z0-9\-]+)_(\d+)\.json$")


def _normalize_model(name: str) -> str:
    """Map raw model tokens in filenames to canonical labels."""
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


def collect_runs(results_dir: Path) -> pd.DataFrame:
    """Read anchor runs in the balanced regime k = ell // 2 (Exp A's trajectory).

    Skips anchor files with k != ell // 2, so Exp B runs at ell=26 (k in
    {2,4,6,8,10}) do not contaminate the Exp A curve.
    """
    rows: list[dict] = []
    for p in sorted(results_dir.glob("anchor_*.json")):
        m = ANCHOR_RE.match(p.name)
        if m is None:
            logger.warning("skipping unparseable filename: %s", p.name)
            continue
        ell, k, model_tok, seed = m.group(1), m.group(2), m.group(3), m.group(4)
        ell_i, k_i = int(ell), int(k)
        if k_i != max(1, ell_i // 2):
            # Not on the balanced ell/2 trajectory; skip for Exp A figure.
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
                "ell": ell_i,
                "k": k_i,
                "model_raw": model_tok,
                "model": _normalize_model(model_tok),
                "seed": int(seed),
                "auc": float(auc),
                "f1": float(test.get("f1", np.nan)),
                "file": p.name,
            }
        )
    if not rows:
        raise RuntimeError(f"no balanced anchor runs (k=ell/2) found under {results_dir}")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["model", "ell"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            f1_mean=("f1", "mean"),
            n_seeds=("seed", "count"),
        )
        .sort_values(["model", "ell"])
    )
    # std is NaN when n=1; fill with 0 for plotting.
    agg["auc_std"] = agg["auc_std"].fillna(0.0)
    return agg


def plot(agg: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ells_all = sorted(agg["ell"].unique())

    for model in MODELS_ORDER:
        sub = agg[agg["model"] == model].sort_values("ell")
        if sub.empty:
            continue
        ells = sub["ell"].to_numpy()
        means = sub["auc_mean"].to_numpy()
        stds = sub["auc_std"].to_numpy()
        color = MODEL_COLORS[model]
        ax.plot(
            ells,
            means,
            marker=MODEL_MARKERS[model],
            color=color,
            linewidth=2,
            label=model,
        )
        ax.fill_between(ells, means - stds, means + stds, color=color, alpha=0.20)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ells_all)
    ax.set_xticklabels([str(e) for e in ells_all])
    ax.set_xlabel(r"Alphabet size $\ell$")
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.40, 1.02)
    ax.set_title(r"Parity, balanced regime $|K|=\ell/2$  ($n=20$, 100K train)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()

    paths: list[Path] = []
    for ext in ("pdf", "png"):
        out_path = out_dir / f"experiment_A_auc_vs_ell.{ext}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Plot AUC vs ell for Experiment A (parity vocab sweep)")
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/parity_decomp"),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/experiment_A/figures"),
    )
    p.add_argument(
        "--csv_out",
        type=Path,
        default=Path("analysis/experiment_A/figures/experiment_A_aggregate.csv"),
    )
    args = p.parse_args()

    df = collect_runs(args.results_dir)
    logger.info("collected %d runs across %d files", len(df), df["file"].nunique())
    logger.info("models found: %s", sorted(df["model"].unique()))
    logger.info("ells found:   %s", sorted(df["ell"].unique()))

    agg = aggregate(df)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.csv_out, index=False)
    logger.info("wrote aggregate CSV: %s", args.csv_out)

    paths = plot(agg, args.output_dir)
    for p_ in paths:
        logger.info("wrote figure: %s", p_)


if __name__ == "__main__":
    main()
