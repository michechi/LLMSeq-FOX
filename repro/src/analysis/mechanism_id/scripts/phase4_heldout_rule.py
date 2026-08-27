"""Phase 4 — Held-out-rule generalisation.

Generate several synthetic Tricky datasets with different random (S, κ, λ).
Train baseline-ladder models on ALL rules except one, and evaluate on the
held-out rule.

Label rule used (matches the paper Def. 1 with min_len=2, exact-lag,
tokens in S, κ nondecreasing): exists a chain of length >=2 such that
consecutive positions are at spacing λ, tokens ∈ S, κ non-decreasing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    ALPHABET, RESULTS_DIR, feat_count26, feat_lag_pair, feat_lag_pair_key_only,
    feat_lag_pair_position, feat_residue, rule_paper_exists_monotone_chain,
    rule_pair_any_lag,
)

SEED_BASE = 20260401


def sample_rule(seed: int, alphabet=ALPHABET, m: int = 6, lag_choices=(5, 6, 7, 8, 9)):
    rng = np.random.default_rng(seed)
    S = tuple(sorted(rng.choice(alphabet, size=m, replace=False).tolist()))
    kappa_perm = rng.permutation(m)
    kappa = {s: int(kappa_perm[i]) for i, s in enumerate(S)}
    lag = int(rng.choice(lag_choices))
    return {"S": S, "kappa": kappa, "lag": lag, "seed": seed}


def generate_rule_dataset(rule: dict, n_sequences: int, n: int = 20, seed: int = 0):
    """Generate sequences i.i.d. uniform over ALPHABET and label them with
    the pair-any-lag rule (equivalent to Def.1 with min_len=2)."""
    rng = np.random.default_rng(seed)
    alph = np.array(ALPHABET)
    key_set = frozenset(rule["S"])
    kappa = rule["kappa"]
    lag = rule["lag"]

    idx = rng.integers(0, 26, size=(n_sequences, n))
    seqs_arr = alph[idx]

    labels = np.zeros(n_sequences, dtype=int)
    # Fast path: pair-any-lag rule.
    for i in range(n_sequences):
        labels[i] = rule_pair_any_lag(seqs_arr[i].tolist(), lag, key_set, kappa)

    seq_strings = ["\x1f".join(row) for row in seqs_arr.tolist()]
    return pd.DataFrame({"Sequences": seq_strings, "Outcome": labels})


def featurise(df: pd.DataFrame, fam: str, lag: int, n: int) -> np.ndarray:
    from common import tokens
    seqs = df["Sequences"].tolist()
    if fam == "A1_count26":
        ex = feat_count26
    elif fam == "B_residue":
        ex = lambda t: feat_residue(t, lag)  # noqa
    elif fam == "C1_lagpair":
        ex = lambda t: feat_lag_pair(t, lag)  # noqa
    elif fam == "C2_lagpair_pos":
        ex = lambda t: feat_lag_pair_position(t, lag, n)  # noqa
    else:
        raise ValueError(fam)
    return np.stack([ex(tokens(s)) for s in seqs], 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_rules", type=int, default=6)
    p.add_argument("--n_per_rule", type=int, default=20_000)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--m", type=int, default=6)
    p.add_argument("--families", nargs="+",
                   default=["A1_count26", "B_residue", "C1_lagpair", "C2_lagpair_pos"])
    p.add_argument("--out", type=Path,
                   default=RESULTS_DIR / "phase4_heldout_rule.csv")
    p.add_argument("--fixed_lag", type=int, default=0,
                   help="if >0, all rules use this lag (pure S/κ variation)")
    args = p.parse_args()

    rules = []
    if args.fixed_lag > 0:
        for i in range(args.n_rules):
            r = sample_rule(seed=SEED_BASE + i, m=args.m, lag_choices=(args.fixed_lag,))
            rules.append(r)
    else:
        for i in range(args.n_rules):
            rules.append(sample_rule(seed=SEED_BASE + i, m=args.m))

    datasets = []
    for i, r in enumerate(rules):
        print(f"[phase4] generating rule {i}: S={r['S']} κ={r['kappa']} λ={r['lag']}",
              flush=True)
        ds = generate_rule_dataset(r, args.n_per_rule, n=args.n, seed=SEED_BASE + 1000 + i)
        print(f"  rho = {ds['Outcome'].mean():.4f}")
        datasets.append(ds)

    out_rows = []
    # Leave-one-out cross-validation over rules.
    for heldout in range(len(rules)):
        r_heldout = rules[heldout]
        lag_h = r_heldout["lag"]
        other = [i for i in range(len(rules)) if i != heldout]

        # Each family is trained with the λ of the HELD-OUT rule (worst-case
        # generalisation: we must use knowledge of the target lag).
        print(f"\n[phase4] holdout = rule {heldout} (λ={lag_h})", flush=True)

        for fam in args.families:
            t0 = time.time()
            Xtr_list = [featurise(datasets[i], fam, lag_h, args.n) for i in other]
            ytr = np.concatenate([datasets[i]["Outcome"].values for i in other])
            Xtr = np.concatenate(Xtr_list, axis=0)
            Xte = featurise(datasets[heldout], fam, lag_h, args.n)
            yte = datasets[heldout]["Outcome"].values
            # In-rule (on-rule) generalisation: also report performance on a
            # held-out split of the same rule, as an upper bound.
            split = int(0.8 * len(datasets[heldout]))
            X_ir_tr = featurise(datasets[heldout].iloc[:split], fam, lag_h, args.n)
            y_ir_tr = datasets[heldout]["Outcome"].values[:split]
            X_ir_te = featurise(datasets[heldout].iloc[split:], fam, lag_h, args.n)
            y_ir_te = datasets[heldout]["Outcome"].values[split:]
            build_t = time.time() - t0

            # Off-rule (held out): train only on other rules.
            clf = LogisticRegression(max_iter=2000, solver="lbfgs",
                                     random_state=SEED_BASE, n_jobs=-1)
            clf.fit(Xtr, ytr)
            if len(np.unique(yte)) >= 2:
                pte = clf.predict_proba(Xte)[:, 1]
                auc_off = float(roc_auc_score(yte, pte))
                thrs = np.linspace(0, 1, 201)
                bt = thrs[np.argmax([f1_score(yte, pte >= t) for t in thrs])]
                f1_off = float(f1_score(yte, pte >= bt))
            else:
                auc_off = float("nan")
                f1_off = float("nan")

            # On-rule baseline.
            clf2 = LogisticRegression(max_iter=2000, solver="lbfgs",
                                      random_state=SEED_BASE, n_jobs=-1)
            clf2.fit(X_ir_tr, y_ir_tr)
            if len(np.unique(y_ir_te)) >= 2:
                pir = clf2.predict_proba(X_ir_te)[:, 1]
                auc_on = float(roc_auc_score(y_ir_te, pir))
            else:
                auc_on = float("nan")

            out_rows.append({
                "heldout_rule_idx": heldout,
                "heldout_S": "".join(r_heldout["S"]),
                "heldout_lag": lag_h,
                "family": fam,
                "model": "logreg_L2",
                "AUC_offrule": auc_off,
                "F1_offrule": f1_off,
                "AUC_onrule": auc_on,
                "n_train_offrule": len(Xtr),
                "n_test_offrule": len(yte),
                "rho_test_offrule": float(np.mean(yte)),
                "build_sec": build_t,
            })
            print(f"  fam={fam}: AUC_off={auc_off} AUC_on={auc_on} "
                  f"dim={Xtr.shape[1]} built in {build_t:.1f}s", flush=True)
            pd.DataFrame(out_rows).to_csv(args.out, index=False)

    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"[phase4] wrote {args.out}")


if __name__ == "__main__":
    main()
