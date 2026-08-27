"""
Main-paper parity generator, configurable sequence length.

Produces a length-parameterised version of the main-paper parity dataset
(test_just_pair), with the exact same key subset K = {W, D, Q, J, X, N} and
label function Y = 1 iff count_{t: X_t in K}(1) is even.

Used by Experiment F (length curriculum on main-paper parity): we generate
matched n=10 and n=20 splits so a model can be trained on the short variant
and warm-started onto the long one.

Output naming (mirrors `X_{train,val,test}_test_just_pair.csv`):
    X_{train,val,test}_main_parity_n{N}.csv
    y_{train,val,test}_main_parity_n{N}.csv

CLI::

    python -m src.data.main_parity_variants --n 10
    python -m src.data.main_parity_variants --n 10 --n 20 --output_dir data/simulation/curriculum
"""

from __future__ import annotations

import argparse
import logging
import string
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


SEP = "\x1f"  # Must match src/data/parity_variants.py:SEP and _parse_tokens.
KEY_LETTERS: tuple[str, ...] = ("W", "D", "Q", "J", "X", "N")  # main-paper K
KEY_SET: frozenset[str] = frozenset(KEY_LETTERS)
ALPHABET: list[str] = list(string.ascii_uppercase)  # A..Z

DEFAULT_OUTPUT_DIR = Path("data/simulation/curriculum")


def _gen_split(n_rows: int, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ALPHABET), size=(n_rows, n))
    seqs = np.take(np.array(ALPHABET), idx)
    counts = np.zeros(n_rows, dtype=np.int64)
    for k in KEY_LETTERS:
        counts += (seqs == k).sum(axis=1)
    labels = (counts % 2 == 0).astype(np.int64)
    X = pd.DataFrame({"Sequences": [SEP.join(row) for row in seqs]})
    y = pd.DataFrame({"Outcome": labels})
    return X, y


def build(
    output_dir: Path,
    n: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    splits = [
        ("train", n_train, int(rng.integers(0, 2**31))),
        ("val",   n_val,   int(rng.integers(0, 2**31))),
        ("test",  n_test,  int(rng.integers(0, 2**31))),
    ]
    tag = f"main_parity_n{n}"
    for split, rows, split_seed in splits:
        X, y = _gen_split(rows, n, split_seed)
        x_path = output_dir / f"X_{split}_{tag}.csv"
        y_path = output_dir / f"y_{split}_{tag}.csv"
        X.to_csv(x_path, index=False)
        y.to_csv(y_path, index=False)
        # Cheap sanity: key-letter frequency per position, label prior.
        key_freq = 0.0
        for k in KEY_LETTERS:
            key_freq += X["Sequences"].str.contains(k).mean()
        label_rate = y["Outcome"].mean()
        logger.info(
            "wrote %s/%s  rows=%d  key-letter-per-seq=%.3f (~%.2f expected)  P(Y=1)=%.3f",
            tag, split, rows, key_freq, 6 * n / 26.0, label_rate,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Main-paper parity generator at configurable n.")
    p.add_argument("--n", type=int, action="append", required=True,
                   help="Sequence length. Repeatable: --n 10 --n 20 generates both.")
    p.add_argument("--n_train", type=int, default=100_000)
    p.add_argument("--n_val",   type=int, default=20_000)
    p.add_argument("--n_test",  type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()
    for n in args.n:
        build(
            output_dir=args.output_dir,
            n=n,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed + 1_000_000 * n,  # independent RNG per n
        )


if __name__ == "__main__":
    main()
