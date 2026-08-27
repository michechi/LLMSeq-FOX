"""
Experiment F summary: aggregate expF_stage{1,2}_*.json into (arch, regime)
rows, emit a CSV + 3-panel bar chart + drop-in LaTeX table.

Usage:

    python3 analysis/experiment_F/scripts/summarize_curriculum.py \
        --results_dir results/parity_decomp \
        --output_dir  analysis/experiment_F/figures
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


ARCH_ORDER = ["Transformer", "LSTM", "BERT"]
ARCH_COLOR = {"Transformer": "#4c78a8", "LSTM": "#54a24b", "BERT": "#e45756"}
SUCCESS_THRESHOLD = 0.9

# Main-paper parity cold n=20 baseline (Table 17). Same value for all archs.
PARITY_COLD_N20_BASELINE = 0.500


def collect(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(results_dir.glob("expF_stage*.json")):
        with p.open() as f:
            d = json.load(f)
        # run_key is expF_{stage}_{model}_n{N}_seed{S}
        rk = d["run_key"]
        parts = rk.split("_")
        # parts == ["expF", "stage1"/"stage2", <model>, "n{N}", "seed{S}"]
        stage = parts[1]  # stage1 or stage2
        arch = parts[2]
        n_str = parts[3]
        assert n_str.startswith("n"), rk
        n_stage = int(n_str[1:])
        seed_str = parts[4]
        assert seed_str.startswith("seed"), rk
        seed = int(seed_str[len("seed"):])
        m = d["test_metrics"]
        rows.append({
            "arch":     arch,
            "stage":    stage,
            "n":        n_stage,
            "seed":     seed,
            "auc":      float(m["auc"]),
            "f1":       float(m["f1"]),
            "acc":      float(m["accuracy"]),
            "elapsed_s": float(d.get("training_time_s", float("nan"))),
            "file":     p.name,
        })
    if not rows:
        raise RuntimeError(f"no expF_stage*_*.json under {results_dir}")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # Regime = (stage, n). Stage 1 is always n=10; Stage 2 is always n=20.
    df = df.copy()
    df["regime"] = df["stage"] + "_n" + df["n"].astype(str)
    agg = (
        df.groupby(["arch", "regime", "n"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            f1_mean=("f1", "mean"),
            acc_mean=("acc", "mean"),
            n_seeds=("seed", "count"),
            n_success=("auc", lambda s: int((s > SUCCESS_THRESHOLD).sum())),
            success_rate=("auc", lambda s: float((s > SUCCESS_THRESHOLD).mean())),
        )
    )
    agg["auc_std"] = agg["auc_std"].fillna(0.0)
    agg["arch"] = pd.Categorical(agg["arch"], ARCH_ORDER, ordered=True)
    return agg.sort_values(["arch", "regime"]).reset_index(drop=True)


def plot(agg: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(11.5, 3.8), sharey=True)

    for ax, arch in zip(axes, ARCH_ORDER):
        sub = agg[agg["arch"] == arch]
        stage1 = sub[sub["regime"] == "stage1_n10"]
        stage2 = sub[sub["regime"] == "stage2_n20"]
        labels = ["Stage 1\n(cold, n=10)", "cold n=20\n(Table 17)", "Stage 2\n(warm, n=20)"]
        means = [
            stage1["auc_mean"].iloc[0] if len(stage1) else np.nan,
            PARITY_COLD_N20_BASELINE,
            stage2["auc_mean"].iloc[0] if len(stage2) else np.nan,
        ]
        stds = [
            stage1["auc_std"].iloc[0] if len(stage1) else 0.0,
            0.0,
            stage2["auc_std"].iloc[0] if len(stage2) else 0.0,
        ]
        color = ARCH_COLOR[arch]
        ax.bar([0, 1, 2], means, yerr=stds, capsize=4,
               color=[color, "#999999", color], edgecolor="black", linewidth=0.8)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0.40, 1.05)
        ax.set_title(arch)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Test AUC")
    fig.suptitle(
        r"Experiment F: Length curriculum on main-paper parity "
        r"($K=\{W,D,Q,J,X,N\}$, $\ell=26$)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = []
    for ext in ("pdf", "png"):
        p = out_dir / f"experiment_F_curriculum.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def emit_latex_table(agg: pd.DataFrame, out_path: Path) -> Path:
    def fmt(m, s, n):
        def _d(x):
            if not np.isfinite(x):
                return "---"
            r = int(round(x * 1000))
            if r >= 1000:
                return "1.000"
            return f".{r:03d}"
        std_str = "---" if n <= 1 else _d(s)
        return f"{_d(m)} ({std_str})"

    lines: list[str] = []
    lines.append(r"\begin{tabular}{@{}lccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Arch & Stage 1 (cold, $n{=}10$) & Cold $n{=}20$ (Table~\ref{tab:results_full}) & Stage 2 (warm, $n{=}20$) \\")
    lines.append(r"\midrule")
    for arch in ARCH_ORDER:
        sub = agg[agg["arch"] == arch]
        s1 = sub[sub["regime"] == "stage1_n10"]
        s2 = sub[sub["regime"] == "stage2_n20"]
        c1 = (
            fmt(s1["auc_mean"].iloc[0], s1["auc_std"].iloc[0], int(s1["n_seeds"].iloc[0]))
            if len(s1) else "---"
        )
        c2 = f".{int(round(PARITY_COLD_N20_BASELINE * 1000)):03d} (---)"
        c3 = (
            fmt(s2["auc_mean"].iloc[0], s2["auc_std"].iloc[0], int(s2["n_seeds"].iloc[0]))
            if len(s2) else "---"
        )
        lines.append(f"{arch} & {c1} & {c2} & {c3} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path,
                   default=Path("results/parity_decomp"))
    p.add_argument("--output_dir", type=Path,
                   default=Path("analysis/experiment_F/figures"))
    args = p.parse_args()

    df = collect(args.results_dir)
    logger.info("collected %d runs", len(df))
    agg = aggregate(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_seed = args.output_dir / "experiment_F_per_seed.csv"
    df.sort_values(["arch", "stage", "seed"]).to_csv(per_seed, index=False)
    agg_path = args.output_dir / "experiment_F_aggregate.csv"
    agg.to_csv(agg_path, index=False)
    logger.info("wrote %s and %s", agg_path, per_seed)
    logger.info("\n%s", agg.to_string(index=False))

    for path in plot(agg, args.output_dir):
        logger.info("wrote %s", path)

    tex = emit_latex_table(agg, args.output_dir / "experiment_F_table.tex")
    logger.info("wrote %s", tex)


if __name__ == "__main__":
    main()
