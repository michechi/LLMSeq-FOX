"""Regenerate the OC classification datasets for the matched-completion study.

Follows the repository's established procedure (paper Section "Setup and
notation" + `src/generators/properties_9.py` / `test_simulation_det.py`):

1. draw an oversized pool of iid Uniform(A..Z) sequences of length n=20
   (with replacement within a row), de-duplicated across rows;
2. compute the latent label Y* with the canonical OC oracle
   (`src.oc_completion.oracle.oc_label_tokens` — single implementation);
3. stratify on Y* to build train/val/test splits, strictly preserving the
   latent prevalence;
4. OC-Deterministic: Y = Y*.  OC-Noisy: flip Y* independently with
   probability pi = 0.3 AFTER splitting.
Both Y (Outcome) and Y* (Latent) are stored for every split.

The stored legacy tag `_6` does NOT implement the paper OC rule (its labels
ignore kappa; see src/analysis/mechanism_id/report.md sections 1-2), so both
tasks are regenerated here rather than reused.

Usage (from repro/ root):
    DATA_DIR=/root/LLMSeq/data python -m src.oc_completion.gen_datasets \
        [--smoke] [--data_seed 9550] [--n_workers 90]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import string
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.oc_completion.oracle import (
    ALPHABET,
    LAG,
    MECHANISM,
    N_EVENTS,
    SEP,
    oc_label_tokens,
)

REPO_ROOT = Path(os.environ.get("LLMSEQ_ROOT", "/root/LLMSeq"))
DEFAULT_OUT = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")) / "simulation" / "oc_completion"

NOISE_PI = 0.3

FULL_SIZES = {"train": 400_000, "val": 50_000, "test": 50_000}
SMOKE_SIZES = {"train": 2_000, "val": 500, "test": 500}


def generate_pool(n_pool: int, seed: int, n_events: int = N_EVENTS) -> list[str]:
    """Unique uniform iid sequences (letters may repeat within a row)."""
    rng = np.random.default_rng(seed)
    letters = np.asarray(ALPHABET, dtype="<U1")
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n_pool:
        k = max(100_000, n_pool - len(out))
        idx = rng.integers(0, len(letters), size=(k, n_events))
        rows = letters[idx]
        for row in rows.tolist():
            key = SEP.join(row)
            if key not in seen:
                seen.add(key)
                out.append(key)
                if len(out) >= n_pool:
                    break
    return out


def _label_chunk(seqs: list[str]) -> list[int]:
    return [oc_label_tokens(s.split(SEP)) for s in seqs]


def label_pool(seqs: list[str], n_workers: int) -> np.ndarray:
    chunks = [seqs[i::n_workers] for i in range(n_workers)]
    with Pool(processes=n_workers) as pool:
        results = pool.map(_label_chunk, chunks)
    y = np.empty(len(seqs), dtype=np.int64)
    for w, res in enumerate(results):
        y[w::n_workers] = res
    return y


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(sizes: dict, tag_suffix: str, data_seed: int, n_workers: int,
          out_dir: Path, pool_factor: float = 2.2) -> dict:
    t0 = time.time()
    n_total = sum(sizes.values())
    n_pool = int(n_total * pool_factor)
    print(f"[gen] pool={n_pool} sequences (seed {data_seed}) ...", flush=True)
    pool_seqs = generate_pool(n_pool, data_seed)
    print(f"[gen] labeling with canonical oracle ({n_workers} workers) ...", flush=True)
    y_star = label_pool(pool_seqs, n_workers)
    rho_pool = float(y_star.mean())
    print(f"[gen] latent prevalence rho={rho_pool:.4f}", flush=True)

    # Stratified subsample to the exact total, then stratified train/val/test.
    X_all = pd.Series(pool_seqs, name="Sequences")
    keep, _ = train_test_split(
        X_all.index, train_size=n_total, stratify=y_star, random_state=data_seed
    )
    X_kept = X_all.loc[keep].reset_index(drop=True)
    y_kept = pd.Series(y_star[keep], name="Latent").reset_index(drop=True)

    idx_train, idx_rest = train_test_split(
        X_kept.index, train_size=sizes["train"], stratify=y_kept,
        random_state=data_seed + 1,
    )
    rest_y = y_kept.loc[idx_rest]
    idx_val, idx_test = train_test_split(
        idx_rest, train_size=sizes["val"], stratify=rest_y,
        random_state=data_seed + 2,
    )
    split_idx = {"train": idx_train, "val": idx_val, "test": idx_test}

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_splits = {}
    for task, pi in (("ocdet", 0.0), ("ocnoisy", NOISE_PI)):
        tag = f"{task}{tag_suffix}"
        for split, idx in split_idx.items():
            X = X_kept.loc[idx].reset_index(drop=True)
            lat = y_kept.loc[idx].reset_index(drop=True).astype(int)
            if pi > 0:
                noise_rng = np.random.default_rng(
                    [data_seed, 7, hash_split(split)]
                )
                flip = noise_rng.random(len(lat)) < pi
                obs = (lat.values ^ flip.astype(int)).astype(int)
            else:
                obs = lat.values.copy()
            X.to_frame().to_csv(out_dir / f"X_{split}_{tag}.csv", index=False)
            pd.DataFrame({"Outcome": obs, "Latent": lat.values}).to_csv(
                out_dir / f"y_{split}_{tag}.csv", index=False
            )
            manifest_splits[f"{tag}/{split}"] = {
                "n": int(len(X)),
                "latent_prevalence": float(lat.mean()),
                "observed_prevalence": float(obs.mean()),
                "noise_pi": pi,
                "X_sha256": sha256_file(out_dir / f"X_{split}_{tag}.csv"),
                "y_sha256": sha256_file(out_dir / f"y_{split}_{tag}.csv"),
            }
            print(f"[gen] wrote {tag}/{split}: n={len(X)} "
                  f"rho*={lat.mean():.4f} rho_obs={obs.mean():.4f}", flush=True)

    manifest = {
        "mechanism": MECHANISM,
        "proposal": "iid Uniform(A..Z), length 20, with replacement, deduplicated pool",
        "pool_size": n_pool,
        "pool_latent_prevalence": rho_pool,
        "data_seed": data_seed,
        "noise_pi": NOISE_PI,
        "noise_applied": "after stratified splitting, independent per split",
        "split_procedure": "stratified on Y* (sklearn train_test_split), "
                           "random_state = data_seed, data_seed+1, data_seed+2",
        "sizes": sizes,
        "splits": manifest_splits,
        "generation_time_s": round(time.time() - t0, 1),
        "command": " ".join(sys.argv),
    }
    with open(out_dir / f"dataset_manifest{tag_suffix or ''}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[gen] done in {manifest['generation_time_s']}s -> {out_dir}", flush=True)
    return manifest


def hash_split(split: str) -> int:
    return {"train": 1, "val": 2, "test": 3}[split]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny smoke-test datasets")
    ap.add_argument("--data_seed", type=int, default=9550)
    ap.add_argument("--n_workers", type=int, default=min(90, os.cpu_count() or 1))
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    sizes = SMOKE_SIZES if args.smoke else FULL_SIZES
    suffix = "_smoke" if args.smoke else ""
    build(sizes, suffix, args.data_seed, args.n_workers, args.out_dir)


if __name__ == "__main__":
    main()
