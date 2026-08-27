"""
Build visit-level dataset from CKD cohort for the visit-shuffle robustness test.

Groups diagnosis codes by admission (hadm_id), represents each visit as a
multi-hot vector of CCS codes, sorts visits by admittime (preserved by the
upstream cohort builder), and writes a 60/20/20 stratified split.

Output (under <MIMIC_TRAINING>):
  visit_level_{train,val,test}.pkl   (list of dicts with visit_sequence array)
  visit_level_meta.pkl               (vocab, code_to_idx, vocab_size, ...)
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import MIMIC_PROCESSED, MIMIC_TRAINING

DEFAULT_SEED = 42
DEFAULT_MIN_VISITS = 1  # keep all patients to match code-level cohort exactly


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--cohort", type=Path,
                   default=Path(MIMIC_PROCESSED) / "ckd_cohort_ccs.csv")
    p.add_argument("--output_dir", type=Path, default=Path(MIMIC_TRAINING))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--min_visits", type=int, default=DEFAULT_MIN_VISITS)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.cohort)
    print(f"Loaded cohort: {len(df)} patients")

    all_codes = set()
    for c in df["codes"]:
        all_codes.update(json.loads(c))
    vocab = sorted(all_codes)
    code_to_idx = {c: i for i, c in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"Vocabulary: {vocab_size} CCS codes")

    records = []
    for _, row in df.iterrows():
        codes = json.loads(row["codes"])
        hadms = json.loads(row["hadm_ids"])

        visits = OrderedDict()
        for c, h in zip(codes, hadms):
            visits.setdefault(h, set()).add(c)

        if len(visits) < args.min_visits:
            continue

        visit_vectors = []
        for _, code_set in visits.items():
            v = np.zeros(vocab_size, dtype=np.float32)
            for c in code_set:
                if c in code_to_idx:
                    v[code_to_idx[c]] = 1.0
            visit_vectors.append(v)

        records.append({
            "subject_id": row["subject_id"],
            "label": row["label"],
            "num_visits": len(visit_vectors),
            "visit_sequence": np.stack(visit_vectors),
        })

    print(f"\nFiltered to >= {args.min_visits} visits: {len(records)} patients")

    labels = [r["label"] for r in records]
    n1 = sum(labels)
    n0 = len(labels) - n1
    print(f"  Y=1: {n1} ({100*n1/len(records):.1f}%)")
    print(f"  Y=0: {n0}")

    num_visits = [r["num_visits"] for r in records]
    print(f"  Visits: median={np.median(num_visits):.0f}, mean={np.mean(num_visits):.1f}, "
          f"max={np.max(num_visits)}")

    indices = list(range(len(records)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, stratify=labels, random_state=args.seed
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=args.seed
    )
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for name, idx in splits.items():
        subset = [records[i] for i in idx]
        pos = sum(r["label"] for r in subset)
        print(f"  {name}: {len(subset)} patients, {pos} positive ({100*pos/len(subset):.1f}%)")

    metadata = {
        "vocab": vocab,
        "code_to_idx": code_to_idx,
        "vocab_size": vocab_size,
        "min_visits": args.min_visits,
        "seed": args.seed,
    }

    for name, idx in splits.items():
        subset = [records[i] for i in idx]
        path = args.output_dir / f"visit_level_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(subset, f)
        print(f"Saved {path} ({len(subset)} patients)")

    meta_path = args.output_dir / "visit_level_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
