"""Evaluate a trained artifact on the matched completion pairs.

Every model is scored on EXACTLY the same generated pair CSVs through the
normal binary-classification inference path (materialized candidate
sequences; no mask tokens). Writes:

* per-candidate predictions (parquet, falls back to csv.gz without pyarrow)
* per-dataset aggregate rows appended (fcntl-locked) to
  results/matched_completion/pair_results.csv
* per-dataset breakdown rows (orientation / kappa gap / chain combination)
  appended to pair_results_breakdown.csv

Model kinds:
  dl        - checkpoints from src.oc_completion.train_dl (best.pt)
  baseline  - joblib artifacts from src.oc_completion.train_baselines
  oracle    - the canonical OC oracle itself (acceptance check: accuracy 1.0)
  hf        - checkpoints from src.oc_completion.train_hf (BERT / Llama)

For `baseline` families `count26` and `lag_pair` the two candidates of every
two-hole pair MUST tie exactly; any margin beyond 1e-9 raises.

Usage (from repro/ root):
    DATA_DIR=... python -m src.oc_completion.eval_pairs \
        --model_kind dl --checkpoint checkpoints/.../best.pt \
        --task ocdet [--smoke] [--splits test]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.oc_completion.scoring import (
    MAIN_TOL,
    TOLERANCES,
    grouped_metrics,
    pair_metrics,
    score_pairs,
)
from src.oc_completion.train_dl import DATA_ROOT, RESULTS_DIR

EXACT_TIE_FAMILIES = {"count26", "lag_pair"}
EXACT_TIE_TOL = 1e-9

PAIR_RESULT_COLUMNS = [
    "timestamp", "model", "training_mode", "task", "seed", "family",
    "background", "split", "n_pairs", "win_rate", "tie_rate", "loss_rate",
    "pair_accuracy", "pair_accuracy_ci_lo", "pair_accuracy_ci_hi",
    "mean_margin", "median_margin", "flattened_auc",
    "pair_accuracy_tol1e-06", "tie_rate_tol1e-06",
    "pair_accuracy_tol1e-10", "tie_rate_tol1e-10",
    "exact_tie_check", "checkpoint",
]


def append_rows(csv_path: Path, rows: list[dict], columns: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if os.fstat(f.fileno()).st_size == 0:
                f.write(",".join(columns) + "\n")
            for row in rows:
                cells = []
                for c in columns:
                    s = str(row.get(c, ""))
                    if "," in s or '"' in s:
                        s = '"' + s.replace('"', '""') + '"'
                    cells.append(s)
                f.write(",".join(cells) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --------------------------------------------------------------------------
# score_fn builders per model kind
# --------------------------------------------------------------------------
def score_fn_dl(checkpoint: Path, threads: int):
    import torch

    from src.experiments.DL_TR_baselines_experiment import create_model
    from src.oc_completion.train_dl import model_logits

    torch.set_num_threads(threads)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = create_model(ck["model_name"], ck["config"])
    model.load_state_dict(ck["model"])
    model.eval()
    meta = {"model": ck["model_name"], "seed": ck.get("seed", ""),
            "training_mode": "scratch"}
    return (lambda seqs: model_logits(model, seqs, "cpu")), meta


def score_fn_baseline(checkpoint: Path):
    import joblib

    from src.oc_completion.train_baselines import FAMILIES

    art = joblib.load(checkpoint)
    fam, kind, model = art["family"], art["feature_kind"], art["model"]

    if kind == "kgram":
        def fn(seqs):
            p = np.clip(model.score(seqs), 1e-12, 1 - 1e-12)
            return np.stack([np.zeros(len(p)), np.log(p / (1 - p))], axis=1)
    else:
        _, builder, model_kind = FAMILIES[fam]

        def fn(seqs):
            X = builder(seqs)
            if model_kind == "logreg":
                # single-threaded float64 matmul: multi-threaded BLAS gives
                # row-position-dependent rounding (~1e-7), which would break
                # the exact-tie invariant on matched candidates
                from threadpoolctl import threadpool_limits
                if hasattr(X, "toarray"):
                    Xd = X.astype(np.float64)
                    with threadpool_limits(limits=1):
                        d = np.asarray(Xd @ model.coef_[0]) .ravel() + model.intercept_[0]
                else:
                    with threadpool_limits(limits=1):
                        d = X.astype(np.float64) @ model.coef_[0] + model.intercept_[0]
            else:
                p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
                d = np.log(p / (1 - p))
            return np.stack([np.zeros(len(d)), d], axis=1)

    meta = {"model": fam, "seed": 9550, "training_mode": "baseline"}
    return fn, meta


def score_fn_oracle():
    from src.oc_completion.oracle import oc_label

    def fn(seqs):
        z = np.array([oc_label(s) for s in seqs], dtype=float)
        return np.stack([np.zeros(len(z)), z], axis=1)

    return fn, {"model": "oracle", "seed": 0, "training_mode": "oracle"}


def score_fn_hf(checkpoint: Path, threads: int, batch_size: int):
    from src.oc_completion.train_hf import load_checkpoint_for_eval
    return load_checkpoint_for_eval(checkpoint, threads=threads,
                                    batch_size=batch_size)


# --------------------------------------------------------------------------
def evaluate_artifact(score_fn, meta: dict, task: str, smoke: bool,
                      splits: list[str], checkpoint: str,
                      out_dir: Path | None = None) -> list[dict]:
    suffix = "_smoke" if smoke else ""
    pairs_dir = DATA_ROOT / f"pairs{suffix}"
    out_dir = out_dir or (RESULTS_DIR / "pair_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_rows, brk_frames, pred_frames = [], [], []
    for family in ("two_hole", "one_hole"):
        for background in ("clean", "heldout"):
            for split in splits:
                path = pairs_dir / f"pairs_{family}_{background}_{split}{suffix}.csv"
                if not path.exists():
                    continue
                pairs = pd.read_csv(path)
                scored = score_pairs(score_fn, pairs)

                tie_check = ""
                if (meta["training_mode"] == "baseline"
                        and meta["model"] in EXACT_TIE_FAMILIES
                        and family == "two_hole"):
                    max_abs = float(np.abs(scored["margin"]).max())
                    if max_abs > EXACT_TIE_TOL:
                        raise AssertionError(
                            f"{meta['model']} must tie on matched pairs but "
                            f"max |margin| = {max_abs:g} on {path.name}")
                    tie_check = f"pass(max|m|={max_abs:.2e})"

                met = pair_metrics(scored)
                agg_rows.append({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model": meta["model"],
                    "training_mode": meta["training_mode"],
                    "task": task, "seed": meta["seed"], "family": family,
                    "background": background, "split": split,
                    **{k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in met.items()
                       if k in PAIR_RESULT_COLUMNS or k in (
                           "n_pairs", "win_rate", "tie_rate", "loss_rate",
                           "pair_accuracy", "pair_accuracy_ci_lo",
                           "pair_accuracy_ci_hi", "mean_margin",
                           "median_margin", "flattened_auc")},
                    "pair_accuracy_tol1e-06": round(met["pair_accuracy_tol1e-06"], 6),
                    "tie_rate_tol1e-06": round(met["tie_rate_tol1e-06"], 6),
                    "pair_accuracy_tol1e-10": round(met["pair_accuracy_tol1e-10"], 6),
                    "tie_rate_tol1e-10": round(met["tie_rate_tol1e-10"], 6),
                    "exact_tie_check": tie_check, "checkpoint": checkpoint,
                })

                # breakdowns (two-hole only: orientation, kappa gap, chains)
                if family == "two_hole":
                    for by in (["orientation"], ["kappa_gap"],
                               ["chain_r", "chain_s"]):
                        g = grouped_metrics(scored, by)
                        g.insert(0, "group_keys", "+".join(by))
                        g.insert(0, "split", split)
                        g.insert(0, "background", background)
                        g.insert(0, "family", family)
                        g.insert(0, "task", task)
                        g.insert(0, "seed", meta["seed"])
                        g.insert(0, "training_mode", meta["training_mode"])
                        g.insert(0, "model", meta["model"])
                        brk_frames.append(g)

                keep = scored[[
                    "pair_id", "orientation", "kappa_gap", "chain_r",
                    "chain_s", "positive_index", "cand0_logit0",
                    "cand0_logit1", "cand0_logodds", "cand0_prob1",
                    "cand1_logit0", "cand1_logit1", "cand1_logodds",
                    "cand1_prob1", "margin"]].copy()
                keep.insert(0, "split", split)
                keep.insert(0, "background", background)
                keep.insert(0, "family", family)
                keep.insert(0, "seed", meta["seed"])
                keep.insert(0, "task", task)
                keep.insert(0, "training_mode", meta["training_mode"])
                keep.insert(0, "model", meta["model"])
                pred_frames.append(keep)

                print(f"[eval_pairs] {meta['model']}/{task} "
                      f"{family}/{background}/{split}: "
                      f"pair_acc={met['pair_accuracy']:.4f} "
                      f"tie={met['tie_rate']:.4f} "
                      f"mean_margin={met['mean_margin']:.4f}", flush=True)

    append_rows(RESULTS_DIR / f"pair_results{'_smoke' if smoke else ''}.csv",
                agg_rows, PAIR_RESULT_COLUMNS)
    if brk_frames:
        brk = pd.concat(brk_frames, ignore_index=True)
        brk_path = RESULTS_DIR / f"pair_results_breakdown{'_smoke' if smoke else ''}.csv"
        header = not brk_path.exists()
        brk.to_csv(brk_path, mode="a", header=header, index=False)
    preds = pd.concat(pred_frames, ignore_index=True)
    stem = (f"{meta['model']}_{task}_seed{meta['seed']}"
            f"{'_smoke' if smoke else ''}")
    try:
        preds.to_parquet(out_dir / f"{stem}.parquet", index=False)
    except (ImportError, ValueError):
        preds.to_csv(out_dir / f"{stem}.csv.gz", index=False)
    return agg_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_kind",
                    choices=["dl", "baseline", "oracle", "hf"], required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--task", choices=["ocdet", "ocnoisy"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--hf_batch_size", type=int, default=64)
    args = ap.parse_args()

    if args.model_kind == "dl":
        fn, meta = score_fn_dl(args.checkpoint, args.threads)
    elif args.model_kind == "baseline":
        fn, meta = score_fn_baseline(args.checkpoint)
    elif args.model_kind == "hf":
        fn, meta = score_fn_hf(args.checkpoint, args.threads,
                               args.hf_batch_size)
    else:
        fn, meta = score_fn_oracle()

    evaluate_artifact(fn, meta, args.task, args.smoke,
                      args.splits.split(","),
                      str(args.checkpoint or "oracle"))


if __name__ == "__main__":
    main()
