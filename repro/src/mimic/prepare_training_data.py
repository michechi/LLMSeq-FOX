"""
Prepare MIMIC CKD cohort data for ordered vs shuffled training experiments.

Reads <MIMIC_PROCESSED>/ckd_cohort_ccs.csv, builds vocabulary, creates a
stratified 60/20/20 split, and saves ordered + shuffled versions with
identical patient assignments. Sequences are joined with the unit-separator
delimiter '\\x1f' (matches the synthetic-data convention in src.common.SEP).

Outputs (under <MIMIC_TRAINING>):
  vocab.json
  X_{train,val,test}_{ordered,shuffled}.csv
  y_{train,val,test}_{ordered,shuffled}.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import MIMIC_PROCESSED, MIMIC_TRAINING, SEP

DEFAULT_SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--cohort", type=Path,
                   default=Path(MIMIC_PROCESSED) / "ckd_cohort_ccs.csv")
    p.add_argument("--output_dir", type=Path, default=Path(MIMIC_TRAINING))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.cohort)
    print(f"Loaded {len(df)} patients")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    df["codes_list"] = df["codes"].apply(json.loads)

    all_codes = sorted({code for codes in df["codes_list"] for code in codes})
    vocab = {code: idx + 1 for idx, code in enumerate(all_codes)}
    vocab_size = len(vocab) + 1  # +1 for padding at index 0
    print(f"Vocabulary size: {vocab_size} ({len(vocab)} unique codes + padding)")

    vocab_path = args.output_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({"code_to_idx": vocab, "vocab_size": vocab_size}, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")

    df["seq_ordered"] = df["codes_list"].apply(lambda codes: SEP.join(codes))

    rng = random.Random(args.seed)

    def shuffle_codes(codes):
        shuffled = list(codes)
        rng.shuffle(shuffled)
        return SEP.join(shuffled)

    df["seq_shuffled"] = df["codes_list"].apply(shuffle_codes)

    # 60/20/20 stratified
    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.4, stratify=df["label"], random_state=args.seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=df.loc[temp_idx, "label"], random_state=args.seed
    )
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for name, idx in splits.items():
        n = len(idx)
        pos = df.loc[idx, "label"].sum()
        print(f"  {name}: {n} patients, {pos} positive ({100*pos/n:.1f}%)")

    for data_type in ("ordered", "shuffled"):
        seq_col = f"seq_{data_type}"
        for split_name, idx in splits.items():
            subset = df.loc[idx]
            x_path = args.output_dir / f"X_{split_name}_{data_type}.csv"
            y_path = args.output_dir / f"y_{split_name}_{data_type}.csv"
            pd.DataFrame({"Sequences": subset[seq_col].values}).to_csv(x_path, index=False)
            pd.DataFrame({"Outcome": subset["label"].values}).to_csv(y_path, index=False)

    print(f"\nSaved 12 CSV files to {args.output_dir}")

    print("\n--- Verification ---")
    x_train_ord = pd.read_csv(args.output_dir / "X_train_ordered.csv")
    x_train_shuf = pd.read_csv(args.output_dir / "X_train_shuffled.csv")
    print(f"Train ordered rows: {len(x_train_ord)}, shuffled rows: {len(x_train_shuf)}")

    ord_codes = set(x_train_ord.iloc[0]["Sequences"].split(SEP))
    shuf_codes = set(x_train_shuf.iloc[0]["Sequences"].split(SEP))
    print(f"First patient same codes: {ord_codes == shuf_codes}")


if __name__ == "__main__":
    main()
