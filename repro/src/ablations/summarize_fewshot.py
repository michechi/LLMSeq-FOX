"""
Aggregate fewshot_llm_eval JSONs into a (task, model, n_shots) table.

Usage:

    python3 analysis/fewshot_llm/scripts/summarize_fewshot.py \
        --results_dir results/fewshot_llm \
        --output_dir  analysis/fewshot_llm/figures

Produces:
    experiment_D_fewshot_aggregate.csv  (one row per (task, model, n_shots))
    experiment_D_fewshot_per_seed.csv   (one row per run)
    experiment_D_fewshot.pdf/.png       (AUC vs n_shots, faceted by task)
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

MODEL_SHORT = {
    "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
    "meta-llama/Llama-3.1-8B": "Llama-3.1-8B",
    "Qwen/Qwen2.5-14B":        "Qwen2.5-14B",
}
MODEL_ORDER = ["Llama-3.2-1B", "Llama-3.1-8B", "Qwen2.5-14B"]
MODEL_COLOR = {
    "Llama-3.2-1B": "#4c78a8",
    "Llama-3.1-8B": "#54a24b",
    "Qwen2.5-14B":  "#e45756",
}
TASK_ORDER = ["tricky_det", "tricky_rnd", "parity"]
TASK_LABEL = {
    "tricky_det": "Tricky Deterministic",
    "tricky_rnd": "Tricky Random",
    "parity":     "Parity",
}


def collect(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(results_dir.glob("fewshot_*.json")):
        with p.open() as f:
            d = json.load(f)
        m = d["test_metrics"]
        rows.append({
            "task":       d["task"],
            "model":      d["model_name"],
            "model_short": MODEL_SHORT.get(d["model_name"], d["model_name"]),
            "n_shots":    int(d["n_shots"]),
            "seed":       int(d["seed"]),
            "n_test":     int(d["n_test"]),
            "auc":        float(m["auc"]),
            "f1":         float(m["f1"]),
            "acc":        float(m["accuracy"]),
            "precision":  float(m["precision"]),
            "recall":     float(m["recall"]),
            "pos_rate_preds":  float(m.get("positive_rate_preds", float("nan"))),
            "pos_rate_labels": float(m.get("positive_rate_labels", float("nan"))),
            "elapsed_s":  float(d.get("elapsed_s", float("nan"))),
            "file":       p.name,
        })
    if not rows:
        raise RuntimeError(f"no fewshot_*.json under {results_dir}")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["task", "model_short", "n_shots"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            f1_mean=("f1", "mean"),
            acc_mean=("acc", "mean"),
            n_seeds=("seed", "count"),
        )
    )
    agg["auc_std"] = agg["auc_std"].fillna(0.0)
    # Stable sort for readability.
    agg["task"] = pd.Categorical(agg["task"], TASK_ORDER, ordered=True)
    agg["model_short"] = pd.Categorical(agg["model_short"], MODEL_ORDER, ordered=True)
    return agg.sort_values(["task", "model_short", "n_shots"]).reset_index(drop=True)


def plot(agg: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    shot_positions = sorted(agg["n_shots"].unique())

    for ax, task in zip(axes, TASK_ORDER):
        sub = agg[agg["task"] == task]
        for model in MODEL_ORDER:
            s = sub[sub["model_short"] == model].sort_values("n_shots")
            if len(s) == 0:
                continue
            ax.errorbar(
                s["n_shots"], s["auc_mean"],
                yerr=s["auc_std"],
                marker="o", capsize=3, label=model,
                color=MODEL_COLOR[model], linewidth=1.5,
            )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
        ax.set_xticks(shot_positions)
        ax.set_xlabel("# in-context examples")
        ax.set_title(TASK_LABEL[task])
        ax.set_ylim(0.40, 1.02)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Test AUC")
    axes[-1].legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()

    out = []
    for ext in ("pdf", "png"):
        p = out_dir / f"experiment_D_fewshot.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def emit_latex_table(agg: pd.DataFrame, out_path: Path) -> Path:
    """
    LaTeX snippet matching the tab:results_full format: 'mean (std)' per cell
    with leading zero dropped. Rows: (task, model); columns: n_shots.
    """
    def fmt(m, s, n):
        def _d(x):
            if not np.isfinite(x):
                return "---"
            return f".{int(round(x * 1000)):03d}"
        # Single-seed runs (0-shot) have no meaningful std -> "---"
        # matching the Qwen2.5-14B rows in tab:results_full.
        std_str = "---" if n <= 1 else _d(s)
        return f"{_d(m)} ({std_str})"

    lines: list[str] = []
    lines.append(r"\begin{tabular}{@{}llccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Task & Model & 0-shot & 5-shot & 20-shot \\")
    lines.append(r"\midrule")
    for task in TASK_ORDER:
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{TASK_LABEL[task]}}}}}\\")
        for model in MODEL_ORDER:
            sub = agg[(agg["task"] == task) & (agg["model_short"] == model)]
            cells = []
            for k in (0, 5, 20):
                row = sub[sub["n_shots"] == k]
                if len(row) == 0:
                    cells.append("---")
                else:
                    cells.append(fmt(
                        row["auc_mean"].iloc[0],
                        row["auc_std"].iloc[0],
                        int(row["n_seeds"].iloc[0]),
                    ))
            lines.append(f"  & {model} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path,
                   default=Path("results/fewshot_llm"))
    p.add_argument("--output_dir", type=Path,
                   default=Path("analysis/fewshot_llm/figures"))
    args = p.parse_args()

    df = collect(args.results_dir)
    logger.info("collected %d runs", len(df))
    agg = aggregate(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = args.output_dir / "experiment_D_fewshot_per_seed.csv"
    df.sort_values(["task", "model_short", "n_shots", "seed"]).to_csv(per_seed, index=False)
    agg_path = args.output_dir / "experiment_D_fewshot_aggregate.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("wrote %s and %s", agg_path, per_seed)
    logger.info("\n%s", agg.to_string(index=False))

    for path in plot(agg, args.output_dir):
        logger.info("wrote %s", path)

    tex = emit_latex_table(agg, args.output_dir / "experiment_D_fewshot_table.tex")
    logger.info("wrote %s", tex)


if __name__ == "__main__":
    main()
