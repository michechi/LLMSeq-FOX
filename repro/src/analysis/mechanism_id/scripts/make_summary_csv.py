"""Collapse per-phase CSVs into one summary table for report.md."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR


def _safe_read(name: str) -> pd.DataFrame | None:
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def main():
    rows = []
    # Phase 1 — rule match accuracy
    p1 = _safe_read("phase1_label_audit.csv")
    if p1 is not None:
        for _, r in p1.iterrows():
            rows.append({
                "section": "phase1_label_audit",
                "task_variant": {
                    "6": "Tricky Det (stored tag=6)",
                    "9": "Tricky Rnd (stored tag=9)",
                    "test_just_pair": "Parity (stored)",
                }.get(r["tag"], r["tag"]),
                "baseline_family": r["rule"],
                "model": "exact_rule",
                "AUC": None,
                "F1": None,
                "accuracy_vs_stored": float(r["acc_vs_stored_int"]),
                "rho_rule": float(r["rho_rule"]),
                "notes": f"{int(r['n_rows'])} rows",
            })

    # Phase 2 — baseline ladder standard
    p2 = _safe_read("phase2_ladder_standard.csv")
    if p2 is not None:
        for _, r in p2.iterrows():
            rows.append({
                "section": "phase2_ladder_standard",
                "task_variant": {"6": "Tricky Det (tag=6)", "9": "Tricky Rnd (tag=9)"}
                .get(r["tag"], r["tag"]),
                "baseline_family": r["family"],
                "model": r["model"],
                "AUC": float(r["AUC"]),
                "F1": float(r["F1"]) if pd.notna(r["F1"]) else None,
                "accuracy_vs_stored": None,
                "rho_rule": None,
                "notes": f"dim={int(r['feat_dim'])}, n_train={int(r['n_train'])}",
            })

    # Phase 3 — matched histogram
    p3 = _safe_read("phase3_matched_histogram.csv")
    if p3 is not None:
        for _, r in p3.iterrows():
            rows.append({
                "section": "phase3_matched_histogram",
                "task_variant": {"6": "Tricky Det (tag=6)", "9": "Tricky Rnd (tag=9)"}
                .get(r["tag"], r["tag"]),
                "baseline_family": f"{r['family']}@{r['eval_regime']}",
                "model": r["model"],
                "AUC": float(r["AUC"]) if pd.notna(r["AUC"]) else None,
                "F1": float(r["F1"]) if pd.notna(r["F1"]) else None,
                "accuracy_vs_stored": None,
                "rho_rule": None,
                "notes": f"n_eval={int(r['n_eval'])}",
            })

    # Phase 4 — held-out-rule
    p4 = _safe_read("phase4_heldout_rule.csv")
    if p4 is not None:
        for _, r in p4.iterrows():
            for k_name, k_col in [("offrule", "AUC_offrule"), ("onrule", "AUC_onrule")]:
                rows.append({
                    "section": "phase4_heldout_rule",
                    "task_variant": f"heldout_rule_{int(r['heldout_rule_idx'])}_{k_name}",
                    "baseline_family": r["family"], "model": r["model"],
                    "AUC": float(r[k_col]) if pd.notna(r[k_col]) else None,
                    "F1": float(r["F1_offrule"]) if k_name == "offrule"
                           and pd.notna(r["F1_offrule"]) else None,
                    "accuracy_vs_stored": None, "rho_rule": None,
                    "notes": f"lag={int(r['heldout_lag'])} S={r['heldout_S']}",
                })

    # Phase 5 — parity
    p5 = _safe_read("phase5_parity.csv")
    if p5 is not None:
        for _, r in p5.iterrows():
            rows.append({
                "section": "phase5_parity",
                "task_variant": r["regime"], "baseline_family": r["variant"],
                "model": r["model"],
                "AUC": float(r["AUC"]) if pd.notna(r["AUC"]) else None,
                "F1": float(r["F1"]) if "F1" in r and pd.notna(r.get("F1")) else None,
                "accuracy_vs_stored": None, "rho_rule": None,
                "notes": f"dim={int(r['feat_dim'])}"
                          if "feat_dim" in r else "",
            })

    # Phase 6 — oracle audit
    p6 = _safe_read("phase6_oracle.csv")
    if p6 is not None:
        for _, r in p6.iterrows():
            rows.append({
                "section": "phase6_oracle", "task_variant": f"{r['tag']}/{r['split']}",
                "baseline_family": r["rule"], "model": "exact_rule",
                "AUC": float(r["AUC_oracle"]), "F1": float(r["F1_oracle"]),
                "accuracy_vs_stored": float(r["noise_acc"]),
                "rho_rule": float(r["rho_pred"]),
                "notes": f"expected_pi={float(r['expected_pi'])}",
            })

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary_all_phases.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
