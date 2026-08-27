"""
Experiment G summary: subset-size and length sweeps on parity at l=26.

Two views:
  - Subset-size view: aggregate anchor_l26_k{2,4,6,8,10,13}_*.json (n=20) into
    a (|K|, arch) table + 3-panel bar chart + drop-in LaTeX table.
  - Length view (only if expG_*_n*_seed*.json files exist): aggregate the
    |K|=13 length sweep (n in {10,15,20,30}) into a (n, arch) table + figure
    + LaTeX. The n=20 row is reused from anchor_l26_k13 to avoid retraining.

Usage:

    python3 analysis/experiment_G/scripts/summarize_subset_length.py \
        --results_dir results/parity_decomp \
        --output_dir  analysis/experiment_G/figures
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


ARCH_ORDER = ["Transformer", "LSTM", "BERT"]
ARCH_COLOR = {"Transformer": "#4c78a8", "LSTM": "#54a24b", "BERT": "#e45756"}
SUCCESS_THRESHOLD = 0.9
ELL_FIXED = 26
K_BALANCED = 13

# Subset-size sweep: which k values were run (anchor_l26_k*_*.json).
K_VALUES_SUBSET = [2, 4, 6, 8, 10, 13]

# Length-sweep grid at |K|=13. n=20 is reused from anchor_l26_k13_*.json;
# the others need expG_*_n{N}_seed{S}.json from Phase 2 of the plan.
N_VALUES_LENGTH = [10, 15, 20, 30]


_ANCHOR_RE = re.compile(r"^anchor_l(\d+)_k(\d+)_([A-Za-z]+)_(\d+)$")
_EXPG_RE = re.compile(r"^expG_([A-Za-z]+)_n(\d+)_seed(\d+)$")


def _read_metrics(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def collect_anchor(results_dir: Path) -> pd.DataFrame:
    """Existing 54 runs at l=26, n=20, k in {2,4,6,8,10,13}."""
    rows: list[dict] = []
    for p in sorted(results_dir.glob("anchor_l*_k*_*.json")):
        m = _ANCHOR_RE.match(p.stem)
        if not m:
            continue
        ell, k, arch, seed = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        if ell != ELL_FIXED:
            continue
        d = _read_metrics(p)
        tm = d["test_metrics"]
        rows.append({
            "arch":     arch,
            "k":        k,
            "n":        20,
            "seed":     seed,
            "auc":      float(tm["auc"]),
            "f1":       float(tm["f1"]),
            "acc":      float(tm["accuracy"]),
            "elapsed_s": float(d.get("training_time_s", float("nan"))),
            "file":     p.name,
            "source":   "anchor",
        })
    return pd.DataFrame(rows)


def collect_expg(results_dir: Path) -> pd.DataFrame:
    """New length-sweep runs (Phase 2), if present."""
    rows: list[dict] = []
    for p in sorted(results_dir.glob("expG_*_n*_seed*.json")):
        m = _EXPG_RE.match(p.stem)
        if not m:
            continue
        arch, n, seed = m.group(1), int(m.group(2)), int(m.group(3))
        d = _read_metrics(p)
        tm = d["test_metrics"]
        rows.append({
            "arch":     arch,
            "k":        K_BALANCED,
            "n":        n,
            "seed":     seed,
            "auc":      float(tm["auc"]),
            "f1":       float(tm["f1"]),
            "acc":      float(tm["accuracy"]),
            "elapsed_s": float(d.get("training_time_s", float("nan"))),
            "file":     p.name,
            "source":   "expG",
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    agg = (
        df.groupby(group_cols, as_index=False)
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
    return agg.sort_values(["arch"] + [c for c in group_cols if c != "arch"]).reset_index(drop=True)


def _fmt_meanstd(m: float, s: float, n: int) -> str:
    def _d(x: float) -> str:
        if not np.isfinite(x):
            return "---"
        r = int(round(x * 1000))
        if r >= 1000:
            return "1.000"
        return f".{r:03d}"
    std_str = "---" if n <= 1 else _d(s)
    return f"{_d(m)} ({std_str})"


def plot_subset(agg_subset: pd.DataFrame, out_dir: Path) -> list[Path]:
    """3-panel bar chart, one per arch, x = |K|, y = AUC."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(11.5, 3.8), sharey=True)

    for ax, arch in zip(axes, ARCH_ORDER):
        sub = agg_subset[agg_subset["arch"] == arch].sort_values("k")
        ks = sub["k"].tolist()
        means = sub["auc_mean"].tolist()
        stds = sub["auc_std"].tolist()
        x = np.arange(len(ks))
        color = ARCH_COLOR[arch]
        # Highlight the balanced point |K|=13 in a slightly darker shade.
        colors = [color if k != K_BALANCED else "#222222" for k in ks]
        ax.bar(x, means, yerr=stds, capsize=4,
               color=colors, edgecolor="black", linewidth=0.8)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"|K|={k}" for k in ks], fontsize=9, rotation=20)
        ax.set_ylim(0.40, 1.05)
        ax.set_title(arch)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Test AUC")
    fig.suptitle(
        r"Experiment G: Subset-size sweep on parity ($\ell=26,\ n=20$). "
        r"Black bar marks the balanced case $|K|/|\mathcal{A}|=1/2$.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out: list[Path] = []
    for ext in ("pdf", "png"):
        p = out_dir / f"experiment_G_subset.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def plot_length(agg_length: pd.DataFrame, out_dir: Path) -> list[Path]:
    """3-panel bar chart, one per arch, x = n, y = AUC at |K|=13."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(11.5, 3.8), sharey=True)

    for ax, arch in zip(axes, ARCH_ORDER):
        sub = agg_length[agg_length["arch"] == arch].sort_values("n")
        ns = sub["n"].tolist()
        means = sub["auc_mean"].tolist()
        stds = sub["auc_std"].tolist()
        x = np.arange(len(ns))
        color = ARCH_COLOR[arch]
        ax.bar(x, means, yerr=stds, capsize=4,
               color=color, edgecolor="black", linewidth=0.8)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"n={n}" for n in ns], fontsize=9)
        ax.set_ylim(0.40, 1.05)
        ax.set_title(arch)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Test AUC")
    fig.suptitle(
        rf"Experiment G: Length-difficulty curve at the balanced case "
        rf"($\ell=26,\ |K|={K_BALANCED}$).",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out: list[Path] = []
    for ext in ("pdf", "png"):
        p = out_dir / f"experiment_G_length.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def emit_subset_table(agg_subset: pd.DataFrame, out_path: Path) -> Path:
    ks = sorted(agg_subset["k"].unique().tolist())
    lines: list[str] = []
    col_spec = "@{}l" + "c" * len(ks) + "@{}"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Arch & " + " & ".join(rf"$|K|{{=}}{k}$" for k in ks) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for arch in ARCH_ORDER:
        sub = agg_subset[agg_subset["arch"] == arch].set_index("k")
        cells: list[str] = []
        for k in ks:
            if k in sub.index:
                row = sub.loc[k]
                cells.append(_fmt_meanstd(row["auc_mean"], row["auc_std"], int(row["n_seeds"])))
            else:
                cells.append("---")
        lines.append(f"{arch} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def emit_length_table(agg_length: pd.DataFrame, out_path: Path) -> Path:
    ns = sorted(agg_length["n"].unique().tolist())
    lines: list[str] = []
    col_spec = "@{}l" + "c" * len(ns) + "@{}"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Arch & " + " & ".join(rf"$n{{=}}{n}$" for n in ns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for arch in ARCH_ORDER:
        sub = agg_length[agg_length["arch"] == arch].set_index("n")
        cells: list[str] = []
        for n in ns:
            if n in sub.index:
                row = sub.loc[n]
                cells.append(_fmt_meanstd(row["auc_mean"], row["auc_std"], int(row["n_seeds"])))
            else:
                cells.append("---")
        lines.append(f"{arch} & " + " & ".join(cells) + r" \\")
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
                   default=Path("analysis/experiment_G/figures"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Subset-size view: anchor_l26_k* runs at n=20.
    anchor_df = collect_anchor(args.results_dir)
    if anchor_df.empty:
        raise RuntimeError(f"no anchor_l{ELL_FIXED}_k*_*.json under {args.results_dir}")
    logger.info("collected %d anchor runs", len(anchor_df))

    subset_df = anchor_df[anchor_df["k"].isin(K_VALUES_SUBSET)].copy()
    agg_subset = aggregate(subset_df, ["arch", "k"])
    logger.info("subset-size aggregate:\n%s", agg_subset.to_string(index=False))

    subset_df.sort_values(["arch", "k", "seed"]).to_csv(
        args.output_dir / "experiment_G_subset_per_seed.csv", index=False)
    agg_subset.to_csv(args.output_dir / "experiment_G_subset_aggregate.csv", index=False)
    for path in plot_subset(agg_subset, args.output_dir):
        logger.info("wrote %s", path)
    tex = emit_subset_table(agg_subset, args.output_dir / "experiment_G_subset.tex")
    logger.info("wrote %s", tex)

    # Length view: pull n=20 row from anchor_l26_k13, the rest from expG_*_n*.
    expg_df = collect_expg(args.results_dir)
    n20_at_k13 = anchor_df[anchor_df["k"] == K_BALANCED].copy()
    n20_at_k13["source"] = "anchor_l26_k13"
    length_df = pd.concat([n20_at_k13, expg_df], ignore_index=True)
    if length_df.empty:
        logger.info("no length-sweep data found; skipping length view")
        return

    agg_length = aggregate(length_df, ["arch", "n"])
    logger.info("length aggregate (|K|=%d):\n%s", K_BALANCED, agg_length.to_string(index=False))

    length_df.sort_values(["arch", "n", "seed"]).to_csv(
        args.output_dir / "experiment_G_length_per_seed.csv", index=False)
    agg_length.to_csv(args.output_dir / "experiment_G_length_aggregate.csv", index=False)

    if expg_df.empty:
        logger.info("only n=20 cell available; skipping length figure/table for now")
        return

    for path in plot_length(agg_length, args.output_dir):
        logger.info("wrote %s", path)
    tex_l = emit_length_table(agg_length, args.output_dir / "experiment_G_length.tex")
    logger.info("wrote %s", tex_l)


if __name__ == "__main__":
    main()
