"""Phase 1 — Label audit (Q1): verify which rule matches stored labels.

For each of the three stored datasets (Tricky Det. = '6', Tricky Rnd. = '9',
Parity = 'test_just_pair'), apply every candidate rule and report agreement
with the stored `Outcome` column.

Output: analysis/mechanism_id/results/phase1_label_audit.csv
        analysis/mechanism_id/results/phase1_counterexamples.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DATA_DIR, KAPPA_IMPL, KAPPA_PAPER, KEY_LETTERS_IMPL, KEY_LETTERS_PAPER,
    KEY_SET_IMPL, RESULTS_DIR, load_split, rule_greedy_monotone_impl,
    rule_paper_exists_monotone_chain, rule_pair_any_lag, rule_pair_strict,
    rule_parity_paper, rule_parity_total_count, tokens,
)

KEY_SET_PAPER = frozenset(KEY_LETTERS_PAPER)


def audit_tricky(tag: str, lag: int = 7, rows: int = 50_000) -> pd.DataFrame:
    print(f"[phase1] auditing Tricky tag={tag}, lag={lag}, rows={rows}")
    t0 = time.time()
    X_tr, y_tr = load_split(tag, "train", rows=rows)
    X_val, y_val = load_split(tag, "val", rows=rows // 5)
    X_te, y_te = load_split(tag, "test", rows=rows // 5)
    df = pd.concat([X_tr, X_val, X_te], ignore_index=True)
    y = pd.concat([y_tr, y_val, y_te], ignore_index=True)["Outcome"].values.astype(float)
    y_int = (y >= 0.5).astype(int)

    tok_lists = [tokens(s) for s in df["Sequences"]]
    n_rows = len(tok_lists)
    lengths = np.array([len(t) for t in tok_lists])

    print(f"  rows={n_rows}  seq_len mean={lengths.mean():.1f} min={lengths.min()} max={lengths.max()}")
    print(f"  raw label mean={y.mean():.4f}  (0/1 decoded mean={y_int.mean():.4f})")

    # Candidate rules (both key conventions)
    def apply(rule, *args, **kwargs):
        return np.fromiter(
            (rule(t, *args, **kwargs) for t in tok_lists), dtype=int, count=n_rows,
        )

    # Extra probe: pair-in-S with no κ constraint (checks whether labels
    # depend on ordering or only presence at lag λ).
    def rule_pair_any_key(toks, lag, key_set, kappa):
        n = len(toks)
        for t in range(n - lag):
            if toks[t] in key_set and toks[t + lag] in key_set:
                return 1
        return 0

    rules = {}
    for sname, sset, skap in [
        ("S_impl", KEY_SET_IMPL, KAPPA_IMPL),
        ("S_paper", KEY_SET_PAPER, KAPPA_PAPER),
    ]:
        rules[f"pair_any_key_{sname}"] = apply(rule_pair_any_key, lag, sset, skap)
        rules[f"pair_any_lag7_{sname}"] = apply(rule_pair_any_lag, lag, sset, skap)
        rules[f"pair_strict_lag7_{sname}"] = apply(rule_pair_strict, lag, sset, skap)
        rules[f"paper_def1_min2_{sname}"] = apply(
            rule_paper_exists_monotone_chain, lag, sset, skap, min_len=2)
        rules[f"paper_def1_min3_{sname}"] = apply(
            rule_paper_exists_monotone_chain, lag, sset, skap, min_len=3)
        rules[f"paper_def1_min4_{sname}"] = apply(
            rule_paper_exists_monotone_chain, lag, sset, skap, min_len=4)
        for tol in (False, True):
            for mcl in (2, 3, 4):
                rules[f"greedy_tol{int(tol)}_min{mcl}_{sname}"] = apply(
                    rule_greedy_monotone_impl, lag, sset, skap,
                    tolerance=tol, min_chain_length=mcl)
    rules["parity_total_count_S_impl"] = apply(rule_parity_total_count, KEY_SET_IMPL)
    rules["parity_paper_def4_S_impl"] = apply(rule_parity_paper, KEY_SET_IMPL)
    rules["parity_total_count_S_paper"] = apply(rule_parity_total_count, KEY_SET_PAPER)
    rules["parity_paper_def4_S_paper"] = apply(rule_parity_paper, KEY_SET_PAPER)

    # Conjunction rules: Def.1 AND second_key_even (probes
    # assign_outcome_positional_2_key logic from do_assign_outcome_pos_4.py).
    CANDIDATE_SECOND_KEYS = [
        ("ATLM", frozenset("ATLM")),
        ("ABST", frozenset("ABST")),  # letters_4_odds in test_simulation_det.py
        ("BSTU", frozenset("BSTU")),
    ]
    for pn, pset, pkap in [("S_paper", KEY_SET_PAPER, KAPPA_PAPER)]:
        base = apply(rule_paper_exists_monotone_chain, lag, pset, pkap, min_len=2)
        for sk_name, sk_set in CANDIDATE_SECOND_KEYS:
            def _sk_even(tk, n=sk_set):
                return int(sum(1 for t in tk if t in n) % 2 == 0)
            sk_mask = np.fromiter(
                (_sk_even(t) for t in tok_lists), dtype=int, count=n_rows,
            )
            conj = (base & sk_mask)
            rules[f"def1AND2nd{sk_name}_{pn}"] = conj

    out = []
    for rule_name, preds in rules.items():
        # Agreement with stored integer label and (for noisy case) with
        # probability-style label.
        acc = float((preds == y_int).mean())
        rho_rule = float(preds.mean())
        # Correlation between rule output and noisy stored label — useful
        # for detecting noise patterns.
        corr = float(np.corrcoef(preds.astype(float), y)[0, 1]) if y.std() > 0 else np.nan
        out.append({
            "tag": tag,
            "rule": rule_name,
            "n_rows": n_rows,
            "rho_rule": rho_rule,
            "rho_stored": float(y.mean()),
            "rho_stored_int": float(y_int.mean()),
            "acc_vs_stored_int": acc,
            "corr_vs_stored_noisy": corr,
        })
    t1 = time.time()
    print(f"  done in {t1 - t0:.1f}s")

    df_out = pd.DataFrame(out)

    # Counterexamples: save up to 50 sequences where the top candidate
    # disagrees with stored label, to inspect whether differences are
    # due to label noise or rule mismatch.
    best_rule = df_out.sort_values("acc_vs_stored_int", ascending=False).iloc[0]["rule"]
    # Also: rank-summary for easy scanning
    print(df_out.sort_values("acc_vs_stored_int", ascending=False)
              [["rule", "rho_rule", "acc_vs_stored_int"]].to_string(index=False))
    preds_best = rules[best_rule]
    disagree_idx = np.where(preds_best != y_int)[0]
    ce = []
    for i in disagree_idx[:50]:
        ce.append({
            "tag": tag,
            "best_rule": best_rule,
            "idx": int(i),
            "seq": df["Sequences"].iloc[i],
            "stored_label": float(y[i]),
            "stored_label_int": int(y_int[i]),
            "pred_by_best_rule": int(preds_best[i]),
        })
    ce_df = pd.DataFrame(ce)
    return df_out, ce_df


def audit_parity(tag: str = "test_just_pair", rows: int = 20_000) -> pd.DataFrame:
    print(f"[phase1] auditing Parity tag={tag}, rows={rows}")
    t0 = time.time()
    X_tr, y_tr = load_split(tag, "train", rows=rows)
    X_val, y_val = load_split(tag, "val", rows=rows // 5)
    X_te, y_te = load_split(tag, "test", rows=rows // 5)
    df = pd.concat([X_tr, X_val, X_te], ignore_index=True)
    y = pd.concat([y_tr, y_val, y_te], ignore_index=True)["Outcome"].values.astype(float)
    y_int = (y >= 0.5).astype(int)

    tok_lists = [tokens(s) for s in df["Sequences"]]
    n_rows = len(tok_lists)

    def apply(rule, *args):
        return np.fromiter(
            (rule(t, *args) for t in tok_lists), dtype=int, count=n_rows,
        )

    rules = {
        "parity_total_count_S_impl": apply(rule_parity_total_count, KEY_SET_IMPL),
        "parity_paper_def4_S_impl":  apply(rule_parity_paper, KEY_SET_IMPL),
        "parity_total_count_S_paper": apply(rule_parity_total_count, KEY_SET_PAPER),
        "parity_paper_def4_S_paper": apply(rule_parity_paper, KEY_SET_PAPER),
    }

    out = []
    for rule_name, preds in rules.items():
        acc = float((preds == y_int).mean())
        out.append({
            "tag": tag,
            "rule": rule_name,
            "n_rows": n_rows,
            "rho_rule": float(preds.mean()),
            "rho_stored_int": float(y_int.mean()),
            "acc_vs_stored_int": acc,
        })
    t1 = time.time()
    print(f"  done in {t1 - t0:.1f}s")

    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=50_000,
                   help="rows to audit per split (subsample for speed)")
    p.add_argument("--lag", type=int, default=7)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "phase1_label_audit.csv")
    p.add_argument("--counterexamples", type=Path,
                   default=RESULTS_DIR / "phase1_counterexamples.csv")
    args = p.parse_args()

    frames = []
    ce_frames = []
    for tag in ("6", "9"):
        df, ce = audit_tricky(tag, lag=args.lag, rows=args.rows)
        frames.append(df)
        ce_frames.append(ce)

    pdf = audit_parity("test_just_pair", rows=args.rows // 2)
    frames.append(pdf)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"[phase1] wrote {args.out}  ({len(out)} rows)")

    if ce_frames:
        ce_all = pd.concat(ce_frames, ignore_index=True)
        ce_all.to_csv(args.counterexamples, index=False)
        print(f"[phase1] wrote {args.counterexamples}  ({len(ce_all)} counterexamples)")

    # Print a summary table to stdout
    print("\n=== SUMMARY (acc vs stored integer label) ===")
    piv = out.pivot_table(
        index="rule", columns="tag", values="acc_vs_stored_int", aggfunc="mean",
    )
    print(piv.round(4))


if __name__ == "__main__":
    main()
