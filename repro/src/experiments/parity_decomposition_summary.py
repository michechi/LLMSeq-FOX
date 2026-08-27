"""
Parity decomposition — summary aggregator.

Reads every *.json under --input_dir (default results/parity_decomp/) and
produces:

  * summary_table.csv  — wide table, rows = (variant, model),
    cols = <size>_<metric>_{mean,std}
  * summary_table.tex  — LaTeX version of the AUC subtable.
  * summary_barchart.png — grouped bar chart at size=400K.
  * binary_anchor_table.csv — ell × model summary for the anchor runs.

Usage::

    python -m src.experiments.parity_decomposition_summary \\
        --input_dir results/parity_decomp \\
        --output_dir results/parity_decomp
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


def _load_runs(input_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for p in sorted(input_dir.glob("*.json")):
        with p.open() as f:
            runs.append(json.load(f))
    logger.info("loaded %d runs from %s", len(runs), input_dir)
    return runs


def _is_anchor(run: dict[str, Any]) -> bool:
    return run.get("anchor") is not None


def _model_short(run: dict[str, Any]) -> str:
    if "n_params" in run:  # DL run
        return run.get("config", {}).get("_model", run.get("model", "?")) or run.get("model", "?")
    return "Llama-3.2-1B"


def _summarize(
    runs: list[dict[str, Any]], sizes: tuple[str, ...] = ("40K", "400K")
) -> pd.DataFrame:
    rows: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in runs:
        if _is_anchor(r):
            continue
        variant = r["variant"]
        # DL JSONs nest _model inside config; LLM JSONs store model at top.
        model = r.get("config", {}).get("_model") if "config" in r else None
        if not model:
            model = (
                "Llama-3.2-1B"
                if "Llama" in str(r.get("model", ""))
                else r.get("model", "?")
            )
        size = r["size"]
        if size not in sizes:
            continue
        m = r["test_metrics"]
        rows[(variant, model)][f"{size}_auc"].append(float(m["auc"]))
        rows[(variant, model)][f"{size}_f1"].append(float(m["f1"]))
        rows[(variant, model)][f"{size}_precision"].append(float(m["precision"]))
        rows[(variant, model)][f"{size}_recall"].append(float(m["recall"]))

    out_rows: list[dict[str, Any]] = []
    for (variant, model), stats in sorted(rows.items()):
        entry: dict[str, Any] = {"variant": variant, "model": model}
        for key, vals in stats.items():
            arr = np.asarray(vals, dtype=np.float64)
            entry[f"{key}_mean"] = float(arr.mean())
            entry[f"{key}_std"] = float(arr.std(ddof=0))
            entry[f"{key}_n"] = int(arr.size)
        out_rows.append(entry)
    return pd.DataFrame(out_rows)


def _summarize_anchor(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for r in runs:
        if not _is_anchor(r):
            continue
        ell = r["anchor"]["ell"]
        k = r["anchor"]["k"]
        model = _model_short(r)
        rows[(ell, k, model)].append(float(r["test_metrics"]["auc"]))

    out: list[dict[str, Any]] = []
    for (ell, k, model), aucs in sorted(rows.items()):
        arr = np.asarray(aucs, dtype=np.float64)
        out.append(
            {
                "ell": ell,
                "k": k,
                "model": model,
                "auc_mean": float(arr.mean()),
                "auc_std": float(arr.std(ddof=0)),
                "n_seeds": int(arr.size),
            }
        )
    return pd.DataFrame(out)


def _write_latex(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("% no runs to summarize\n")
        return

    # Pivot to (variant, model) rows; AUC mean±std for 40K and 400K.
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Parity decomposition — test AUC mean$\pm$std across 3 seeds.}")
    lines.append(r"\label{tab:parity_decomposition}")
    lines.append(r"\begin{tabular}{l l c c}")
    lines.append(r"\toprule")
    lines.append(r"Variant & Model & AUC @ 40K & AUC @ 400K \\")
    lines.append(r"\midrule")
    for _, row in df.sort_values(["variant", "model"]).iterrows():
        auc40 = f"{row.get('40K_auc_mean', float('nan')):.3f}" if not np.isnan(row.get('40K_auc_mean', float('nan'))) else "--"
        std40 = f"{row.get('40K_auc_std', float('nan')):.3f}" if not np.isnan(row.get('40K_auc_std', float('nan'))) else "--"
        auc400 = f"{row.get('400K_auc_mean', float('nan')):.3f}" if not np.isnan(row.get('400K_auc_mean', float('nan'))) else "--"
        std400 = f"{row.get('400K_auc_std', float('nan')):.3f}" if not np.isnan(row.get('400K_auc_std', float('nan'))) else "--"
        cell40 = f"{auc40} $\\pm$ {std40}" if auc40 != "--" else "--"
        cell400 = f"{auc400} $\\pm$ {std400}" if auc400 != "--" else "--"
        lines.append(f"{row['variant']} & {row['model']} & {cell40} & {cell400} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


def _plot_barchart(df: pd.DataFrame, path: Path, size_tag: str = "400K") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        logger.info("no rows to plot")
        return

    variants = sorted(df["variant"].unique())
    models = sorted(df["model"].unique())
    x = np.arange(len(models))
    width = 0.8 / max(len(variants), 1)

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(models)), 4))
    for i, v in enumerate(variants):
        means = []
        errs = []
        for m in models:
            sub = df[(df["variant"] == v) & (df["model"] == m)]
            if sub.empty:
                means.append(float("nan"))
                errs.append(0.0)
            else:
                means.append(float(sub[f"{size_tag}_auc_mean"].iloc[0]))
                errs.append(float(sub[f"{size_tag}_auc_std"].iloc[0]))
        ax.bar(x + i * width, means, width=width, yerr=errs, capsize=3, label=v)

    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Test AUC")
    ax.set_title(f"Parity decomposition @ {size_tag}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)


def main() -> None:
    p = argparse.ArgumentParser(description="Parity decomposition — summary")
    p.add_argument("--input_dir", type=Path, default=Path("results/parity_decomp"))
    p.add_argument("--output_dir", type=Path, default=Path("results/parity_decomp"))
    p.add_argument("--plot_size", default="400K",
                   help="Size tag for the summary bar chart (default 400K)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(args.input_dir)

    df = _summarize(runs)
    df_anchor = _summarize_anchor(runs)

    df.to_csv(args.output_dir / "summary_table.csv", index=False)
    logger.info("wrote %s", args.output_dir / "summary_table.csv")

    _write_latex(df, args.output_dir / "summary_table.tex")
    logger.info("wrote %s", args.output_dir / "summary_table.tex")

    if not df_anchor.empty:
        df_anchor.to_csv(args.output_dir / "binary_anchor_table.csv", index=False)
        logger.info("wrote %s", args.output_dir / "binary_anchor_table.csv")

    _plot_barchart(df, args.output_dir / f"summary_barchart_{args.plot_size}.png", args.plot_size)

    if not df.empty:
        print(df.to_string(index=False))
    if not df_anchor.empty:
        print("\nBinary anchor:")
        print(df_anchor.to_string(index=False))


if __name__ == "__main__":
    main()
