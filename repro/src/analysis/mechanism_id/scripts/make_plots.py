"""Produce the small set of figures referenced in report.md.

Reads CSVs from analysis/mechanism_id/results/, writes PNGs to plots/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import PLOTS_DIR, RESULTS_DIR


def fig_ladder():
    df = pd.read_csv(RESULTS_DIR / "phase2_ladder_standard.csv")
    piv = df.pivot_table(index="family", columns=["tag", "model"],
                         values="AUC", aggfunc="max")
    order = ["A1_count26", "A2_count_key", "B_residue", "C1k_lagpair_key",
             "C1_lagpair", "D1k_lagtrigram_key"]
    order = [r for r in order if r in piv.index]
    piv = piv.loc[order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(piv.index))
    width = 0.18
    cols = [("6", "logreg_L2"), ("6", "xgboost"),
            ("9", "logreg_L2"), ("9", "xgboost")]
    labels = ["_6 LogReg", "_6 XGB", "_9 LogReg", "_9 XGB"]
    for i, (c, lab) in enumerate(zip(cols, labels)):
        if c in piv.columns:
            ax.bar(x + (i - 1.5) * width, piv[c].values, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=25, ha="right")
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="grey", ls=":", lw=0.7)
    ax.axhline(0.67, color="C3", ls="--", lw=0.9, label="_9 AUC* = 0.67")
    ax.axhline(1.00, color="C2", ls="--", lw=0.9, label="_6 AUC* = 1.00")
    ax.set_title("Baseline ladder AUC on standard test split")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    p = PLOTS_DIR / "fig_ladder_standard.png"
    fig.savefig(p, dpi=150)
    print(f"wrote {p}")


def fig_heldout():
    df = pd.read_csv(RESULTS_DIR / "phase4_heldout_rule.csv")
    mean_on = df.groupby("family")["AUC_onrule"].mean()
    mean_off = df.groupby("family")["AUC_offrule"].mean()
    fams = list(mean_on.index)
    x = np.arange(len(fams))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.36
    ax.bar(x - w / 2, mean_on.values, w, label="on-rule (same rule)")
    ax.bar(x + w / 2, mean_off.values, w, label="off-rule (held-out rule)")
    ax.axhline(0.5, color="grey", ls=":", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=20, ha="right")
    ax.set_ylabel("Mean AUC across 6 held-out rules")
    ax.set_title("Held-out-rule generalization (Phase 4)")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    plt.tight_layout()
    p = PLOTS_DIR / "fig_heldout_rule.png"
    fig.savefig(p, dpi=150)
    print(f"wrote {p}")


def fig_parity():
    df = pd.read_csv(RESULTS_DIR / "phase5_parity.csv")
    real = df[df["regime"] == "paper_m6_l26_n20"].copy()
    real["label"] = real["variant"] + " / " + real["model"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    order = [
        "A_hidden_letters_onehot / logreg",
        "A_hidden_letters_onehot / mlp_64",
        "A_hidden_count26 / logreg",
        "A_hidden_count26 / mlp_64",
        "B_known_membership_bits / logreg",
        "B_known_membership_bits / mlp_64",
        "C_known_key_count / logreg",
        "C_known_key_count / mlp_64",
        "C_known_key_count_parity / mlp_64",
        "D_total_key_count / mlp_64",
    ]
    real = real.set_index("label").reindex(order).reset_index()
    x = np.arange(len(real))
    ax.bar(x, real["AUC"].values,
           color=["C0" if v.startswith("A_") else
                  "C2" if v.startswith("B_") else
                  "C3" if v.startswith("C_") else "C4"
                  for v in real["label"]])
    ax.axhline(0.5, color="grey", ls=":", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(real["label"], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("Parity task AUC")
    ax.set_title("Parity decomposition: hidden-K (A) vs. known-K (B,C,D)")
    plt.tight_layout()
    p = PLOTS_DIR / "fig_parity_decomp.png"
    fig.savefig(p, dpi=150)
    print(f"wrote {p}")


def main():
    fig_ladder()
    fig_heldout()
    fig_parity()


if __name__ == "__main__":
    main()
