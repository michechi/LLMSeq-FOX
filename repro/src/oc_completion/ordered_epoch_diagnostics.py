"""Post-training diagnostic trajectories for saved OC epoch snapshots.

This command never participates in training, early stopping, threshold
selection, or checkpoint selection.  It evaluates the fixed diagnostic holes
and strict validation pairs only after epoch snapshots already exist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.oc_completion.ordered_eval import (
    _load_oracle_array,
    _manifest_sequence_column,
    _measure_jitter,
    _normalise_metadata,
    _oracle_rows_for_manifest,
    _score_hole_chunk,
    _score_pairs_same_batch,
    _select_holes,
    _select_pairs,
)
from src.oc_completion.ordered_io import RESULT_ROOT
from src.oc_completion.ordered_metrics import (
    position_localization_metrics,
    stability_metrics,
    strict_pair_metrics,
    valid_filling_metrics,
)


def _snapshot_scorer(kind: str, path: Path, device: str, batch_size: int):
    if kind == "scratch":
        from src.oc_completion.ordered_train_dl import load_scratch_scorer
        scorer, meta = load_scratch_scorer(path, device=device,
                                            batch_size=batch_size)
    else:
        from src.oc_completion.ordered_train_hf import load_hf_scorer
        scorer, meta = load_hf_scorer(path, device=device,
                                      batch_size=batch_size)
    return scorer, _normalise_metadata(meta)


def _standard_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "history.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "epoch" not in frame:
        raise ValueError(f"{path} has no epoch column")
    return frame.set_index("epoch", drop=False)


def _standard_value(row: pd.Series, *names: str):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return float("nan")


def evaluate_trajectory(
    *,
    model_kind: str,
    run_dir: Path,
    hole_manifest_path: Path,
    hole_oracle_path: Path,
    strict_manifest_path: Path,
    device: str = "auto",
    batch_size: int = 520,
    diagnostic_bases: int = 500,
    strict_pairs: int = 2000,
    bootstrap_iters: int = 200,
) -> pd.DataFrame:
    snapshots = sorted((run_dir / "epochs").glob("epoch_*_eval.pt"))
    if not snapshots:
        raise FileNotFoundError(f"no epoch snapshots in {run_dir / 'epochs'}")
    history = _standard_history(run_dir)
    full_holes = pd.read_parquet(hole_manifest_path).reset_index(drop=True)
    if "manifest_index" not in full_holes:
        full_holes["manifest_index"] = np.arange(len(full_holes))
    holes = _select_holes(full_holes, "diagnostic").head(diagnostic_bases).reset_index(drop=True)
    all_oracle = _load_oracle_array(hole_oracle_path)
    oracle = _oracle_rows_for_manifest(full_holes, holes, all_oracle)
    seq_col = _manifest_sequence_column(holes)
    strict = _select_pairs(pd.read_parquet(strict_manifest_path), "val", "heldout").head(
        strict_pairs).reset_index(drop=True)

    rows = []
    for snapshot in snapshots:
        scorer, meta = _snapshot_scorer(model_kind, snapshot, device, batch_size)
        epoch = int(meta.get("checkpoint_epoch", snapshot.stem.split("_")[1]))
        logits_chunks, prob_chunks = [], []
        for start in range(0, len(holes), 50):
            stop = min(start + 50, len(holes))
            z, p, _ = _score_hole_chunk(
                scorer, holes.iloc[start:stop], oracle[start:stop], seq_col,
                meta, float(meta.get("noise_pi", 0.0)),
                include_prediction_rows=False)
            logits_chunks.append(z)
            prob_chunks.append(p)
        logits = np.concatenate(logits_chunks)
        probabilities = np.concatenate(prob_chunks)
        position = position_localization_metrics(holes, oracle, logits, probabilities)
        repair = valid_filling_metrics(holes, oracle, logits)
        standard = history.loc[epoch]
        threshold = float(_standard_value(
            standard, "validation_selected_threshold", "selected_threshold"))
        if not np.isfinite(threshold):
            threshold = float(meta.get("validation_threshold", 0.5))
        stability = stability_metrics(
            holes, oracle, logits, probabilities,
            {"validation_selected": threshold})
        jitter = _measure_jitter(scorer, strict)
        strict_scored = _score_pairs_same_batch(scorer, strict)
        strict_metrics = strict_pair_metrics(
            strict_scored, tolerance=jitter["tie_tolerance"],
            cluster_col="base_sequence_id", n_boot=bootstrap_iters)
        pos_all = position[position.segment.eq("all")].iloc[0]
        if "all_targets" in set(repair.segment):
            repair_all = repair[repair.segment.eq("all_targets")].iloc[0]
        else:
            target_rows = repair[repair.segment.isin(("target_0", "target_1"))]
            repair_all = target_rows.mean(numeric_only=True)
        fixed = stability.iloc[0:0]
        if len(stability):
            # Weight the category rates by their number of fixed positions.
            weights = stability.n_positions.to_numpy(dtype=float)
            fixed_flip = float(np.average(stability.prediction_flip_rate, weights=weights))
        else:
            fixed_flip = float("nan")
        rows.append({
            "model": meta.get("model", meta.get("model_name", "unknown")),
            "noise_level": float(meta.get("noise_pi", 0.0)),
            "model_seed": int(meta.get("seed", meta.get("model_seed", 0))),
            "epoch": epoch,
            "training_loss": _standard_value(standard, "train_loss", "training_loss"),
            "standard_validation_loss": _standard_value(
                standard, "validation_loss", "val_loss"),
            "standard_observed_validation_auc": _standard_value(
                standard, "validation_observed_auc", "observed_validation_auc"),
            "standard_latent_validation_auc": _standard_value(
                standard, "validation_latent_auc", "latent_validation_auc"),
            "standard_validation_f1": _standard_value(
                standard, "validation_observed_f1", "observed_validation_f1"),
            "hole_position_localization_auroc": pos_all.position_auroc,
            "hole_repair_macro_auc": repair_all.macro_candidate_auc,
            "hole_top1_valid_filling_accuracy": repair_all.top1_valid_filling_accuracy,
            "fixed_position_prediction_flip_rate": fixed_flip,
            "strict_matched_pair_accuracy": strict_metrics["pair_accuracy"],
            "strict_matched_pair_mean_margin": strict_metrics["mean_margin"],
            "strict_tie_tolerance": jitter["tie_tolerance"],
            "snapshot": str(snapshot),
        })
    result = pd.DataFrame(rows).sort_values("epoch")
    result.to_csv(run_dir / "epoch_diagnostics.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=("scratch", "hf"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hole-manifest", type=Path,
                        default=RESULT_ROOT / "hole_manifest.parquet")
    parser.add_argument("--hole-oracle", type=Path,
                        default=RESULT_ROOT / "hole_oracle_labels.npz")
    parser.add_argument("--strict-manifest", type=Path,
                        default=RESULT_ROOT / "strict_pair_manifest.parquet")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=520)
    parser.add_argument("--diagnostic-bases", type=int, default=500)
    parser.add_argument("--strict-pairs", type=int, default=2000)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    args = parser.parse_args()
    result = evaluate_trajectory(
        model_kind=args.model_kind, run_dir=args.run_dir,
        hole_manifest_path=args.hole_manifest,
        hole_oracle_path=args.hole_oracle,
        strict_manifest_path=args.strict_manifest, device=args.device,
        batch_size=args.batch_size, diagnostic_bases=args.diagnostic_bases,
        strict_pairs=args.strict_pairs, bootstrap_iters=args.bootstrap_iters)
    print(f"[ordered_epoch_diagnostics] wrote {len(result)} epochs to "
          f"{args.run_dir / 'epoch_diagnostics.csv'}")


if __name__ == "__main__":
    main()
