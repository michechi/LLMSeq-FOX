"""Unified exhaustive-hole and strict-pair evaluation for OC audit artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.oc_completion.ordered_baselines import load_baseline_scorer
from src.oc_completion.ordered_io import RESULT_ROOT, atomic_json_dump, pi_slug
from src.oc_completion.ordered_metrics import (
    calibrate_tie_tolerance,
    position_localization_metrics,
    stability_metrics,
    strict_pair_metrics,
    valid_filling_metrics,
)
from src.oc_completion.oracle import ALPHABET, N_EVENTS, SEP

ScoreFn = Callable[[list[str]], np.ndarray]


def _tokens(sequence: str) -> list[str]:
    toks = sequence.split(SEP) if SEP in sequence else list(sequence)
    if len(toks) != N_EVENTS:
        raise ValueError(f"expected {N_EVENTS} events, got {len(toks)}")
    return toks


def enumerate_position_candidates(sequence: str, position: int) -> list[str]:
    """All 26 complete replacements, in canonical alphabet order."""
    base = _tokens(sequence)
    if not 0 <= position < N_EVENTS:
        raise IndexError(position)
    candidates = []
    for letter in ALPHABET:
        completed = list(base)
        completed[position] = letter
        candidates.append(SEP.join(completed))
    return candidates


def _probability(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f"scorer returned {z.shape}, expected [N,2]")
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    return (exp[:, 1] / exp.sum(axis=1)).astype(np.float32)


def _load_scorer(args) -> tuple[ScoreFn, dict[str, Any]]:
    if args.model_kind == "baseline":
        return load_baseline_scorer(args.checkpoint)
    if args.model_kind == "oracle":
        from src.oc_completion.oracle import oc_label

        def score(sequences):
            y = np.asarray([oc_label(s) for s in sequences], dtype=np.float32)
            return np.stack([1.0 - y, y], axis=1)

        return score, {"model": "oracle", "seed": 0,
                       "validation_threshold": 0.5,
                       "training_mode": "oracle"}
    if args.model_kind in ("scratch", "random_scratch"):
        from src.oc_completion.ordered_train_dl import load_scratch_scorer
        if args.model_kind == "random_scratch":
            from src.oc_completion.ordered_train_dl import make_random_scorer
            scorer, meta = make_random_scorer(
                args.model, seed=args.seed, device=args.device,
                batch_size=args.batch_size)
        else:
            scorer, meta = load_scratch_scorer(
                args.checkpoint, device=args.device, batch_size=args.batch_size)
        return scorer, _normalise_metadata(meta)
    if args.model_kind == "hf":
        from src.oc_completion.ordered_train_hf import load_hf_scorer
        scorer, meta = load_hf_scorer(args.checkpoint, device=args.device,
                                      batch_size=args.batch_size)
        return scorer, _normalise_metadata(meta)
    raise KeyError(args.model_kind)


def _normalise_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    meta = dict(meta)
    meta.setdefault("seed", meta.get("model_seed", 0))
    meta.setdefault("validation_threshold",
                    meta.get("validation_selected_threshold", 0.5))
    meta.setdefault("checkpoint_epoch", meta.get("epoch", -1))
    return meta


def _artifact_id(meta: dict, pi: float, checkpoint: Path | None) -> str:
    raw = "|".join([
        str(meta.get("model", "unknown")), pi_slug(pi), str(meta.get("seed", "")),
        str(meta.get("checkpoint_epoch", meta.get("epoch", ""))),
        str(checkpoint or "none"),
    ])
    digest = hashlib.sha256(raw.encode()).hexdigest()[:10]
    safe_model = str(meta.get("model", "model")).replace("/", "_")
    return f"{safe_model}_pi{pi_slug(pi)}_s{meta.get('seed', 0)}_{digest}"


def _manifest_sequence_column(manifest: pd.DataFrame) -> str:
    for name in ("X", "base_sequence", "sequence"):
        if name in manifest:
            return name
    raise ValueError("hole manifest has no X/base_sequence/sequence column")


def _select_holes(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    if "split" not in manifest:
        return manifest.copy()
    if split in ("diagnostic", "epoch_diagnostic") and "is_epoch_diagnostic" in manifest:
        selected = manifest[manifest["is_epoch_diagnostic"].astype(bool)].copy()
    else:
        aliases = {split, f"hole_{split}"}
        if split in ("val", "validation"):
            aliases.update({"val", "validation", "hole_val", "hole_validation"})
        if split == "test":
            aliases.update({"test", "hole_test"})
        selected = manifest[manifest["split"].astype(str).isin(aliases)].copy()
    if selected.empty:
        raise ValueError(f"no hole rows for split={split}; found {manifest['split'].unique()}")
    return selected


def _load_oracle_array(path: Path) -> np.ndarray:
    archive = np.load(path, allow_pickle=False)
    for key in ("oracle_labels", "candidate_Y_star", "labels"):
        if key in archive:
            return np.asarray(archive[key], dtype=np.int8)
    raise ValueError(f"{path} lacks oracle_labels/candidate_Y_star/labels")


def _oracle_rows_for_manifest(all_manifest: pd.DataFrame, selected: pd.DataFrame,
                              labels: np.ndarray) -> np.ndarray:
    if len(labels) != len(all_manifest):
        raise ValueError("oracle array first dimension != hole manifest rows")
    if "manifest_index" in all_manifest:
        indices = selected["manifest_index"].to_numpy(dtype=int)
    else:
        indices = selected.index.to_numpy(dtype=int)
    return labels[indices]


def _score_hole_chunk(score_fn: ScoreFn, rows: pd.DataFrame,
                      oracle: np.ndarray, seq_col: str,
                      meta: dict, pi: float, *,
                      include_prediction_rows: bool = True
                      ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
    logits_out = np.empty((len(rows), N_EVENTS, len(ALPHABET)), dtype=np.float32)
    probs_out = np.empty_like(logits_out)
    frames = []
    for local_s, row in enumerate(rows.itertuples(index=False)):
        sequence = str(getattr(row, seq_col))
        original = _tokens(sequence)
        flat_candidates = []
        for position in range(N_EVENTS):
            flat_candidates.extend(enumerate_position_candidates(sequence, position))
        raw = np.asarray(score_fn(flat_candidates), dtype=np.float32)
        if raw.shape != (N_EVENTS * len(ALPHABET), 2):
            raise ValueError(f"scorer returned {raw.shape} for 520 candidates")
        pos_logits = raw[:, 1].reshape(N_EVENTS, len(ALPHABET))
        probs = _probability(raw).reshape(N_EVENTS, len(ALPHABET))
        logits_out[local_s] = pos_logits
        probs_out[local_s] = probs
        base_id = getattr(row, "base_sequence_id")
        base_y = int(getattr(row, "base_Y_star"))
        if include_prediction_rows:
            frames.append(pd.DataFrame({
            "base_sequence_id": np.repeat(base_id, N_EVENTS * len(ALPHABET)),
            "base_Y_star": np.repeat(base_y, N_EVENTS * len(ALPHABET)),
            "position": np.repeat(np.arange(N_EVENTS, dtype=np.int8), len(ALPHABET)),
            "original_letter": np.repeat(np.asarray(original), len(ALPHABET)),
            "candidate_letter": np.tile(np.asarray(ALPHABET), N_EVENTS),
            "candidate_Y_star": oracle[local_s].ravel().astype(np.int8),
            "positive_logit": pos_logits.ravel().astype(np.float32),
            "positive_probability": probs.ravel().astype(np.float32),
            "model": str(meta.get("model", "unknown")),
            "noise_level": float(pi),
            "model_seed": int(meta.get("seed", 0)),
            "checkpoint_epoch": int(meta.get("checkpoint_epoch", meta.get("epoch", -1))),
            }))
    predictions = pd.concat(frames, ignore_index=True) if frames else None
    return logits_out, probs_out, predictions


def evaluate_holes(
    score_fn: ScoreFn,
    meta: dict,
    *,
    pi: float,
    manifest_path: Path,
    oracle_path: Path,
    split: str,
    out_dir: Path,
    chunk_bases: int = 100,
    bootstrap_iters: int = 1000,
) -> dict[str, Path]:
    """Score a fixed hole manifest, resuming at base-sequence chunks."""
    all_manifest = pd.read_parquet(manifest_path).reset_index(drop=True)
    if "manifest_index" not in all_manifest:
        all_manifest["manifest_index"] = np.arange(len(all_manifest))
    selected = _select_holes(all_manifest, split).reset_index(drop=True)
    all_labels = _load_oracle_array(oracle_path)
    oracle = _oracle_rows_for_manifest(all_manifest, selected, all_labels)
    seq_col = _manifest_sequence_column(selected)
    pred_dir = out_dir / "hole_prediction_shards"
    pred_dir.mkdir(parents=True, exist_ok=True)
    logits = np.empty_like(oracle, dtype=np.float32)
    probabilities = np.empty_like(oracle, dtype=np.float32)
    for start in range(0, len(selected), chunk_bases):
        stop = min(start + chunk_bases, len(selected))
        shard = pred_dir / f"bases_{start:06d}_{stop:06d}.parquet"
        chunk = selected.iloc[start:stop]
        if shard.exists():
            old = pd.read_parquet(shard)
            expected = (stop - start) * N_EVENTS * len(ALPHABET)
            if len(old) != expected:
                raise RuntimeError(f"incomplete existing shard {shard}")
            logits[start:stop] = old.positive_logit.to_numpy(dtype=np.float32).reshape(
                stop - start, N_EVENTS, len(ALPHABET))
            probabilities[start:stop] = old.positive_probability.to_numpy(
                dtype=np.float32).reshape(stop - start, N_EVENTS, len(ALPHABET))
            continue
        z, p, prediction_rows = _score_hole_chunk(
            score_fn, chunk, oracle[start:stop], seq_col, meta, pi)
        logits[start:stop], probabilities[start:stop] = z, p
        tmp = shard.with_suffix(".tmp.parquet")
        assert prediction_rows is not None
        prediction_rows.to_parquet(tmp, index=False)
        os.replace(tmp, shard)

    np.savez_compressed(out_dir / f"hole_scores_{split}.npz",
                        positive_logits=logits, probabilities=probabilities,
                        base_sequence_ids=selected.base_sequence_id.to_numpy())
    position = position_localization_metrics(
        selected, oracle, logits, probabilities,
        bootstrap_iters=bootstrap_iters,
    )
    repair = valid_filling_metrics(
        selected, oracle, logits,
        bootstrap_iters=bootstrap_iters,
    )
    threshold = float(meta.get("validation_threshold", 0.5))
    stability = stability_metrics(selected, oracle, logits, probabilities,
                                  {"fixed_0p5": 0.5,
                                   "validation_selected": threshold},
                                  bootstrap_iters=bootstrap_iters)
    common = {
        "model": meta.get("model", "unknown"), "noise_level": pi,
        "model_seed": meta.get("seed", 0),
        "checkpoint_epoch": meta.get("checkpoint_epoch", meta.get("epoch", -1)),
        "split": split,
    }
    for frame in (position, repair, stability):
        for key, value in reversed(list(common.items())):
            frame.insert(0, key, value)
    paths = {}
    for name, frame in (("hole_position_metrics", position),
                        ("hole_repair_metrics", repair),
                        ("hole_stability_metrics", stability)):
        path = out_dir / f"{name}_{split}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def _select_pairs(manifest: pd.DataFrame, split: str, background: str) -> pd.DataFrame:
    out = manifest.copy()
    if "family" in out:
        out = out[out.family.eq("strict_four_chain")]
    if "split" in out:
        out = out[out.split.astype(str).isin({split, f"strict_{split}"})]
    if "background" in out:
        out = out[out.background.eq(background)]
    if out.empty:
        raise ValueError(f"no strict pairs for {background}/{split}")
    return out.reset_index(drop=True)


def _score_pairs_same_batch(score_fn: ScoreFn, pairs: pd.DataFrame,
                            pair_chunk: int = 512) -> pd.DataFrame:
    frames = []
    for start in range(0, len(pairs), pair_chunk):
        chunk = pairs.iloc[start:start + pair_chunk].copy()
        # Interleaving guarantees the two candidates enter the same inference
        # call and adjacent locations in any sufficiently sized model batch.
        interleaved = np.column_stack([chunk.cand0, chunk.cand1]).ravel().tolist()
        raw = np.asarray(score_fn(interleaved), dtype=np.float32)
        if raw.shape != (2 * len(chunk), 2):
            raise ValueError(f"strict scorer returned {raw.shape}")
        z = raw[:, 1].reshape(len(chunk), 2)
        p = _probability(raw).reshape(len(chunk), 2)
        pos_idx = chunk.positive_index.to_numpy(dtype=np.int8)
        neg_idx = 1 - pos_idx
        row = np.arange(len(chunk))
        out = chunk.copy()
        out["cand0_positive_logit"] = z[:, 0]
        out["cand1_positive_logit"] = z[:, 1]
        out["cand0_positive_probability"] = p[:, 0]
        out["cand1_positive_probability"] = p[:, 1]
        out["positive_logit"] = z[row, pos_idx]
        out["negative_logit"] = z[row, neg_idx]
        out["margin"] = out.positive_logit - out.negative_logit
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def _measure_jitter(score_fn: ScoreFn, pairs: pd.DataFrame,
                    n_candidates: int = 128, repeats: int = 4) -> dict[str, float]:
    candidates = np.column_stack([pairs.cand0, pairs.cand1]).ravel()[:n_candidates].tolist()
    repeated = []
    for _ in range(repeats):
        raw = np.asarray(score_fn(candidates), dtype=np.float32)
        repeated.append(raw[:, 1])
    return calibrate_tie_tolerance(np.stack(repeated, axis=1))


def evaluate_strict_pairs(
    score_fn: ScoreFn,
    meta: dict,
    *,
    pi: float,
    manifest_path: Path,
    split: str,
    background: str,
    out_dir: Path,
    pair_chunk: int = 512,
    bootstrap_iters: int = 1000,
) -> tuple[Path, Path]:
    pairs = _select_pairs(pd.read_parquet(manifest_path), split, background)
    jitter = _measure_jitter(score_fn, pairs)
    scored = _score_pairs_same_batch(score_fn, pairs, pair_chunk=pair_chunk)
    tolerance = jitter["tie_tolerance"]
    scored["strict_win"] = scored.margin > tolerance
    scored["tie"] = np.abs(scored.margin) <= tolerance
    scored["loss"] = scored.margin < -tolerance
    cluster_col = ("base_sequence_id" if background == "heldout"
                   and "base_sequence_id" in scored else "pair_id")
    metrics = strict_pair_metrics(scored, tolerance=tolerance,
                                  cluster_col=cluster_col,
                                  n_boot=bootstrap_iters)
    metrics.update(jitter)
    metrics.update({
        "model": meta.get("model", "unknown"), "noise_level": pi,
        "model_seed": meta.get("seed", 0),
        "checkpoint_epoch": meta.get("checkpoint_epoch", meta.get("epoch", -1)),
        "split": split, "background": background,
    })
    pred_path = out_dir / f"strict_pair_predictions_{background}_{split}.parquet"
    scored.to_parquet(pred_path, index=False)
    metric_path = out_dir / f"strict_pair_metrics_{background}_{split}.csv"
    pd.DataFrame([metrics]).to_csv(metric_path, index=False)
    return pred_path, metric_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=("scratch", "hf", "baseline",
                                                  "oracle", "random_scratch"),
                        required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--pi", type=float, required=True)
    parser.add_argument("--seed", type=int, default=9550)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=520)
    parser.add_argument("--hole-split", default="test")
    parser.add_argument("--pair-split", default="test")
    parser.add_argument("--backgrounds", default="clean,heldout")
    parser.add_argument("--hole-manifest", type=Path,
                        default=RESULT_ROOT / "hole_manifest.parquet")
    parser.add_argument("--hole-oracle", type=Path,
                        default=RESULT_ROOT / "hole_oracle_labels.npz")
    parser.add_argument("--strict-manifest", type=Path,
                        default=RESULT_ROOT / "strict_pair_manifest.parquet")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--skip-holes", action="store_true")
    parser.add_argument("--skip-pairs", action="store_true")
    parser.add_argument("--chunk-bases", type=int, default=100)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    args = parser.parse_args()
    if args.model_kind not in ("oracle", "random_scratch") and not args.checkpoint:
        parser.error("--checkpoint is required for this model kind")
    score_fn, meta = _load_scorer(args)
    meta.setdefault("model", args.model or args.model_kind)
    meta.setdefault("seed", args.seed)
    artifact_id = _artifact_id(meta, args.pi, args.checkpoint)
    out_dir = args.result_root / "runs" / "evaluation" / artifact_id
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    if not args.skip_holes:
        outputs.update({k: str(v) for k, v in evaluate_holes(
            score_fn, meta, pi=args.pi, manifest_path=args.hole_manifest,
            oracle_path=args.hole_oracle, split=args.hole_split,
            out_dir=out_dir, chunk_bases=args.chunk_bases,
            bootstrap_iters=args.bootstrap_iters).items()})
    if not args.skip_pairs:
        for background in args.backgrounds.split(","):
            pred, metric = evaluate_strict_pairs(
                score_fn, meta, pi=args.pi, manifest_path=args.strict_manifest,
                split=args.pair_split, background=background, out_dir=out_dir,
                bootstrap_iters=args.bootstrap_iters)
            outputs[f"strict_predictions_{background}"] = str(pred)
            outputs[f"strict_metrics_{background}"] = str(metric)
    atomic_json_dump({"status": "complete", "artifact_id": artifact_id,
                      "metadata": meta, "outputs": outputs}, out_dir / "done.json")
    print(f"[ordered_eval] complete: {artifact_id} -> {out_dir}")


if __name__ == "__main__":
    main()
