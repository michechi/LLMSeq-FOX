"""
Parity decomposition — input-variant generator.

Reads the existing parity dataset (test_just_pair, n=20, S={W,D,Q,J,X,N},
label Y* = 1 iff count(S) is even) and writes three parallel variants of the
training data plus (optionally) a binary anchor control:

    raw      — letter sequence X, unchanged.
    masked   — letters X + parallel MaskBits column b_t = 1{X_t in S}.
    bitonly  — binary sequence b_t only; encoded as {'B','C'} so the existing
               DL dataset class (ord(c)-ord('A')) maps to {1, 2}, avoiding
               the padding_idx=0 slot used by TransformerClassifier / LSTM /
               GRU in src/experiments/DL_TR_baselines_experiment.py.

CLI::

    python -m src.data.parity_variants --verify
    python -m src.data.parity_variants --dry_run           # 1K only
    python -m src.data.parity_variants --build_full        # 1K + 40K + 400K
    python -m src.data.parity_variants --build_anchor --anchor_ell 2
"""

from __future__ import annotations

import argparse
import logging
import os
import string
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
SEP = "\x1f"
KEY_LETTERS: tuple[str, ...] = ("W", "D", "Q", "J", "X", "N")
KEY_SET: frozenset[str] = frozenset(KEY_LETTERS)

# bitonly: 0 -> 'B' (id 1), 1 -> 'C' (id 2). Never 'A' (id 0 == padding_idx).
BIT_TO_LETTER: dict[int, str] = {0: "B", 1: "C"}

DEFAULT_DATA_DIR = Path("data/simulation/tested")
DEFAULT_OUTPUT_DIR = Path("data/simulation/parity_decomp")
BASE_ID = "test_just_pair"
SIZES_FULL: tuple[int, ...] = (1_000, 40_000, 400_000)
SIZES_DRY: tuple[int, ...] = (1_000,)

ANCHOR_DEFAULT_N = 20
ANCHOR_SIZE = 400_000
ANCHOR_VAL_SIZE = 100_000
ANCHOR_TEST_SIZE = 100_000


# --------------------------------------------------------------------------- #
# Core encoders                                                               #
# --------------------------------------------------------------------------- #
def _tokens(raw_sequence: str) -> list[str]:
    """Split a stored sequence into per-position letters."""
    if SEP in raw_sequence:
        return [t for t in raw_sequence.split(SEP) if t]
    return list(raw_sequence)


def membership_bits(tokens: Sequence[str], key_set: frozenset[str] = KEY_SET) -> list[int]:
    return [1 if t in key_set else 0 for t in tokens]


def encode_masked(raw_sequence: str) -> tuple[list[int], list[int]]:
    """Return (letter_ids in {0..25}, mask_ids in {0,1})."""
    toks = _tokens(raw_sequence)
    letter_ids = [ord(t) - ord("A") for t in toks]
    mask_ids = membership_bits(toks)
    return letter_ids, mask_ids


def encode_bit_only(raw_sequence: str) -> list[int]:
    """Return DL-compatible ids in {1,2} (never 0)."""
    toks = _tokens(raw_sequence)
    bits = membership_bits(toks)
    return [b + 1 for b in bits]


# --------------------------------------------------------------------------- #
# Text renderers (for CSV and for LLM prompts)                                #
# --------------------------------------------------------------------------- #
def _render_bit_only_sequence(tokens: Sequence[str]) -> str:
    bits = membership_bits(tokens)
    return SEP.join(BIT_TO_LETTER[b] for b in bits)


def _render_mask_bits(tokens: Sequence[str]) -> str:
    return SEP.join(str(b) for b in membership_bits(tokens))


def make_llm_prompt(raw_sequence: str, variant: str) -> str:
    """
    Variant-aware prompt for CausalLM fine-tuning.

    raw     : "Sequential events: D W T P ... Outcome (0 or 1):"
    masked  : "Sequential events: D/1 W/1 T/0 P/0 ... Outcome (0 or 1):"
    bitonly : "Sequential bits: 0 0 1 0 0 1 ... Outcome (0 or 1):"
    """
    toks = _tokens(raw_sequence)
    if variant == "raw":
        body = " ".join(toks)
        return f"Sequential events: {body}\nOutcome (0 or 1):"
    if variant == "masked":
        pairs = [f"{t}/{b}" for t, b in zip(toks, membership_bits(toks))]
        body = " ".join(pairs)
        return f"Sequential events: {body}\nOutcome (0 or 1):"
    if variant == "bitonly":
        body = " ".join(str(b) for b in membership_bits(toks))
        return f"Sequential bits: {body}\nOutcome (0 or 1):"
    raise ValueError(f"unknown variant: {variant}")


# --------------------------------------------------------------------------- #
# CSV builders                                                                #
# --------------------------------------------------------------------------- #
def _stratified_subsample(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n: int,
    seed: int,
    label_col: str = "Outcome",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n >= len(X):
        return X.reset_index(drop=True), y.reset_index(drop=True)
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y[label_col], random_state=seed
    )
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


def _write_variant_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    split: str,
    variant: str,
    size_tag: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _variant_id(variant, size_tag)
    x_out = output_dir / f"X_{split}_{run_id}.csv"
    y_out = output_dir / f"y_{split}_{run_id}.csv"

    if variant == "raw":
        out_df = X[["Sequences"]].copy()
    elif variant == "masked":
        out_df = X[["Sequences"]].copy()
        out_df["MaskBits"] = X["Sequences"].map(lambda s: _render_mask_bits(_tokens(s)))
    elif variant == "bitonly":
        out_df = pd.DataFrame(
            {"Sequences": X["Sequences"].map(lambda s: _render_bit_only_sequence(_tokens(s)))}
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    out_df.to_csv(x_out, index=False)
    y.to_csv(y_out, index=False)
    logger.info("wrote %s (%d rows)", x_out, len(out_df))


def _variant_id(variant: str, size_tag: str) -> str:
    return f"parity_{variant}_{size_tag}"


def _size_tag(n: int) -> str:
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}K"
    return str(n)


def build_variant_csvs(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sizes: Iterable[int] = SIZES_FULL,
    seed: int = 0,
    base_id: str = BASE_ID,
) -> None:
    """
    For each size in `sizes`, stratified-subsample the parity training split
    to that size, then emit raw / masked / bitonly CSV pairs. Val and test
    remain full-size and are emitted once per variant with size_tag='full'.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading base %s from %s", base_id, data_dir)
    X_train = pd.read_csv(data_dir / f"X_train_{base_id}.csv").fillna("")
    y_train = pd.read_csv(data_dir / f"y_train_{base_id}.csv").fillna(0)
    X_val = pd.read_csv(data_dir / f"X_val_{base_id}.csv").fillna("")
    y_val = pd.read_csv(data_dir / f"y_val_{base_id}.csv").fillna(0)
    X_test = pd.read_csv(data_dir / f"X_test_{base_id}.csv").fillna("")
    y_test = pd.read_csv(data_dir / f"y_test_{base_id}.csv").fillna(0)

    logger.info(
        "base sizes  train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test)
    )

    for variant in ("raw", "masked", "bitonly"):
        _write_variant_split(X_val, y_val, "val", variant, "full", output_dir)
        _write_variant_split(X_test, y_test, "test", variant, "full", output_dir)

    for n in sizes:
        X_sub, y_sub = _stratified_subsample(X_train, y_train, n, seed=seed)
        tag = _size_tag(n)
        logger.info("subsampled train -> size=%s (rows=%d, seed=%d)", tag, len(X_sub), seed)
        for variant in ("raw", "masked", "bitonly"):
            _write_variant_split(X_sub, y_sub, "train", variant, tag, output_dir)


# --------------------------------------------------------------------------- #
# Binary anchor generator                                                     #
# --------------------------------------------------------------------------- #
def build_binary_anchor(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    alphabet_size: int = 2,
    n_keys: int = 1,
    n: int = ANCHOR_DEFAULT_N,
    n_train: int = ANCHOR_SIZE,
    n_val: int = ANCHOR_VAL_SIZE,
    n_test: int = ANCHOR_TEST_SIZE,
    seed: int = 0,
) -> None:
    """
    Create a parity-at-smaller-alphabet dataset.

    Alphabet = first `alphabet_size` letters of {B,C,D,E,...} (starting at
    'B' to avoid padding_idx=0 in the DL pipeline). Key set = first `n_keys`
    letters. Label = 1 iff count(key_set) is even (matches rule of the main
    parity dataset).
    """
    assert 1 <= n_keys <= alphabet_size
    assert alphabet_size <= 26

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if alphabet_size == 26:
        # Full alphabet. 'A' maps to letter-id 1 downstream (pad_idx=0 stays clear).
        alphabet = list(string.ascii_uppercase)
    else:
        # Skip 'A' to stay clear of padding_idx=0.
        alphabet = list(string.ascii_uppercase[1 : 1 + alphabet_size])
    key_set = set(alphabet[:n_keys])
    # Backward-compatible tag: omit `_n{N}` when n equals the legacy default
    # so existing `anchor_l{ell}_k{k}.csv` files keep their names.
    tag = f"anchor_l{alphabet_size}_k{n_keys}"
    if n != ANCHOR_DEFAULT_N:
        tag = f"{tag}_n{n}"

    rng = np.random.default_rng(seed)

    def _gen_split(n_rows: int, split_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        r = np.random.default_rng(split_seed)
        indices = r.integers(0, alphabet_size, size=(n_rows, n))
        seqs = np.take(np.array(alphabet), indices)
        counts = np.array([[ch in key_set for ch in row] for row in seqs]).sum(axis=1)
        labels = (counts % 2 == 0).astype(int)

        X = pd.DataFrame({"Sequences": [SEP.join(row) for row in seqs]})
        y = pd.DataFrame({"Outcome": labels})
        return X, y

    splits = [
        ("train", n_train, int(rng.integers(0, 2**31))),
        ("val", n_val, int(rng.integers(0, 2**31))),
        ("test", n_test, int(rng.integers(0, 2**31))),
    ]
    for split, rows, split_seed in splits:
        X, y = _gen_split(rows, split_seed)
        X.to_csv(output_dir / f"X_{split}_{tag}.csv", index=False)
        y.to_csv(output_dir / f"y_{split}_{tag}.csv", index=False)
        bit_rate = (X["Sequences"].str.contains(next(iter(key_set)))).mean()  # cheap sanity
        label_rate = y["Outcome"].mean()
        logger.info(
            "anchor %s/%s rows=%d  contains_any_key_rate=%.3f  label_rate=%.3f",
            tag,
            split,
            rows,
            bit_rate,
            label_rate,
        )


# --------------------------------------------------------------------------- #
# Verification                                                                #
# --------------------------------------------------------------------------- #
def verify(data_dir: Path = DEFAULT_DATA_DIR, n_check: int = 5000) -> None:
    """Sanity-check the parity rule and bit-rate on the existing CSV."""
    data_dir = Path(data_dir)
    X = pd.read_csv(data_dir / f"X_train_{BASE_ID}.csv").fillna("")
    y = pd.read_csv(data_dir / f"y_train_{BASE_ID}.csv").fillna(0)
    n = min(n_check, len(X))

    mismatches = 0
    bit_rates: list[float] = []
    for i in range(n):
        toks = _tokens(X["Sequences"].iloc[i])
        bits = membership_bits(toks)
        pred = 1 if (sum(bits) % 2 == 0) else 0
        true = int(y["Outcome"].iloc[i])
        if pred != true:
            mismatches += 1
        bit_rates.append(sum(bits) / len(bits))

    mean_rate = float(np.mean(bit_rates))
    logger.info("verify: checked %d rows", n)
    logger.info("verify: parity(count(S)%%2==0) matches Y*: mismatches=%d/%d", mismatches, n)
    logger.info("verify: mean bit rate (fraction of positions in S) = %.3f (expected ~0.23)", mean_rate)
    logger.info("verify: label prior P(Y=1) = %.3f", float(y["Outcome"].iloc[:n].mean()))

    assert mismatches == 0, (
        f"parity rule mismatch: {mismatches}/{n}. Expected exact parity of "
        f"S={sorted(KEY_SET)} count (even->1, odd->0)."
    )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parity decomposition — data variant builder")
    p.add_argument("--verify", action="store_true", help="Sanity-check the parity rule")
    p.add_argument("--dry_run", action="store_true", help="Build only the 1K-row variants")
    p.add_argument("--build_full", action="store_true", help="Build 1K + 40K + 400K variants")
    p.add_argument("--build_anchor", action="store_true", help="Build binary anchor control")
    p.add_argument("--anchor_ell", type=int, default=2, help="Anchor alphabet size (single value)")
    p.add_argument(
        "--anchor_ells",
        type=str,
        default=None,
        help="Comma-separated list of alphabet sizes; overrides --anchor_ell when set (e.g. '2,4,8,16,26').",
    )
    p.add_argument("--anchor_k", type=int, default=None, help="# keys (default: ell // 2)")
    p.add_argument(
        "--anchor_n",
        type=int,
        default=ANCHOR_DEFAULT_N,
        help=(
            "Sequence length per anchor sample (default: %(default)s). "
            "Overridden by --anchor_lens when set."
        ),
    )
    p.add_argument(
        "--anchor_lens",
        type=str,
        default=None,
        help=(
            "Comma-separated list of sequence lengths; overrides --anchor_n "
            "(e.g. '10,15,30' to build three CSVs side-by-side)."
        ),
    )
    p.add_argument(
        "--anchor_n_train",
        type=int,
        default=ANCHOR_SIZE,
        help="Train rows per anchor dataset.",
    )
    p.add_argument(
        "--anchor_n_val",
        type=int,
        default=ANCHOR_VAL_SIZE,
        help="Val rows per anchor dataset.",
    )
    p.add_argument(
        "--anchor_n_test",
        type=int,
        default=ANCHOR_TEST_SIZE,
        help="Test rows per anchor dataset.",
    )
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verify:
        verify(args.data_dir)

    sizes: tuple[int, ...] | None = None
    if args.dry_run:
        sizes = SIZES_DRY
    elif args.build_full:
        sizes = SIZES_FULL

    if sizes is not None:
        build_variant_csvs(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            sizes=sizes,
            seed=args.seed,
        )

    if args.build_anchor:
        if args.anchor_ells is not None:
            ells = [int(s) for s in args.anchor_ells.split(",") if s.strip()]
        else:
            ells = [args.anchor_ell]
        if args.anchor_lens is not None:
            ns = [int(s) for s in args.anchor_lens.split(",") if s.strip()]
        else:
            ns = [args.anchor_n]
        for ell in ells:
            n_keys = args.anchor_k if args.anchor_k is not None else max(1, ell // 2)
            for n in ns:
                build_binary_anchor(
                    output_dir=args.output_dir,
                    alphabet_size=ell,
                    n_keys=n_keys,
                    n=n,
                    n_train=args.anchor_n_train,
                    n_val=args.anchor_n_val,
                    n_test=args.anchor_n_test,
                    seed=args.seed,
                )


if __name__ == "__main__":
    main()
