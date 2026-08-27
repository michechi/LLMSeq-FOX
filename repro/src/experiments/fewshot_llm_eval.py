"""
Few-shot / in-context classification evaluation for CausalLM models on
Tricky Deterministic, Tricky Random, and Parity.

Method: for each test sequence, build a prompt with k labeled train examples
followed by the query, do one forward pass, and compare the logits at the
final position for the " 0" and " 1" token ids. Softmax over those two
logits gives P(Y=1 | prompt). Metrics (AUC, F1, accuracy) are computed from
that probability. No fine-tuning.

Task -> CSV mapping (data/simulation/tested/):
    tricky_det -> X_{train,test}_6.csv
    tricky_rnd -> X_{train,test}_9.csv
    parity     -> X_{train,test}_test_just_pair.csv

Output: one JSON at --output_dir/{run_key}.json with per-cell metrics.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


TASK_FILES = {
    "tricky_det": (
        "X_train_6.csv", "y_train_6.csv",
        "X_test_6.csv",  "y_test_6.csv",
    ),
    "tricky_rnd": (
        "X_train_9.csv", "y_train_9.csv",
        "X_test_9.csv",  "y_test_9.csv",
    ),
    "parity": (
        "X_train_test_just_pair.csv", "y_train_test_just_pair.csv",
        "X_test_test_just_pair.csv",  "y_test_test_just_pair.csv",
    ),
}


def _seq_body(seq: str) -> str:
    # 20-char uppercase sequence → space-separated letters, matching the
    # `raw` variant in src/data/parity_variants.py:make_llm_prompt.
    return " ".join(list(seq))


def _render_example(seq: str, label: int | None) -> str:
    head = f"Sequential events: {_seq_body(seq)}\nOutcome (0 or 1):"
    if label is None:
        return head
    return f"{head} {int(label)}"


def build_prompt(shots: list[tuple[str, int]], query: str) -> str:
    parts = [_render_example(s, y) for s, y in shots]
    parts.append(_render_example(query, None))
    return "\n\n".join(parts)


def sample_shots(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    k: int,
    seed: int,
) -> list[tuple[str, int]]:
    if k == 0:
        return []
    rng = np.random.default_rng(seed)
    labels = y_train["Outcome"].to_numpy().astype(int)
    idx_pos = np.flatnonzero(labels == 1)
    idx_neg = np.flatnonzero(labels == 0)
    n_pos = k // 2
    n_neg = k - n_pos
    chosen_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
    chosen_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
    chosen = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(chosen)
    seqs = X_train["Sequences"].astype(str).tolist()
    return [(seqs[i], int(labels[i])) for i in chosen]


def load_model(model_name: str, cache_dir: Path | None) -> tuple:
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def _digit_token_id(tokenizer, ch: str) -> int:
    # Prompt ends with "Outcome (0 or 1):", so the continuation naturally
    # begins with a space + digit. Encode " 0" / " 1" and take the token id
    # that decodes to the digit (BPE may split into [" ", "0"] or [" 0"]).
    ids = tokenizer.encode(f" {ch}", add_special_tokens=False)
    for tid in ids:
        if tokenizer.decode([tid]).strip() == ch:
            return tid
    # Fallback to the last id if none match cleanly.
    return ids[-1]


@torch.no_grad()
def score_batch(
    tokenizer,
    model,
    prompts: list[str],
    tok0: int,
    tok1: int,
    max_length: int,
) -> np.ndarray:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)
    last_logits = out.logits[:, -1, :]
    diff = last_logits[:, tok1] - last_logits[:, tok0]
    return torch.sigmoid(diff).float().cpu().numpy()


def _sanitize(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def run_key(model_name: str, task: str, n_shots: int, seed: int) -> str:
    return f"fewshot_{task}_{_sanitize(model_name)}_k{n_shots}_seed{seed}"


def main() -> None:
    p = argparse.ArgumentParser(description="Few-shot LLM eval.")
    p.add_argument("--model_name", required=True)
    p.add_argument("--task", required=True, choices=list(TASK_FILES))
    p.add_argument("--n_shots", type=int, required=True)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=Path,
                   default=Path("data/simulation/tested"))
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=3072)
    args = p.parse_args()

    rk = run_key(args.model_name, args.task, args.n_shots, args.seed)
    out_path = args.output_dir / f"{rk}.json"
    if out_path.exists():
        logger.info("output already exists, skipping: %s", out_path)
        return

    ftr = TASK_FILES[args.task]
    X_train = pd.read_csv(args.data_dir / ftr[0]).fillna("")
    y_train = pd.read_csv(args.data_dir / ftr[1]).fillna(0)
    X_test = pd.read_csv(args.data_dir / ftr[2]).fillna("")
    y_test = pd.read_csv(args.data_dir / ftr[3]).fillna(0)

    rng = np.random.default_rng(args.seed)
    n_test = min(args.n_test, len(X_test))
    sel = np.sort(rng.choice(len(X_test), size=n_test, replace=False))
    X_test = X_test.iloc[sel].reset_index(drop=True)
    y_test = y_test.iloc[sel].reset_index(drop=True)

    shots = sample_shots(X_train, y_train, args.n_shots, args.seed)
    logger.info("task=%s model=%s shots=%d seed=%d n_test=%d",
                args.task, args.model_name, len(shots), args.seed, n_test)

    prompts = [build_prompt(shots, str(s)) for s in X_test["Sequences"]]

    tokenizer, model = load_model(args.model_name, args.cache_dir)
    tok0 = _digit_token_id(tokenizer, "0")
    tok1 = _digit_token_id(tokenizer, "1")
    logger.info("label tokens: 0 -> %d (%r), 1 -> %d (%r)",
                tok0, tokenizer.decode([tok0]),
                tok1, tokenizer.decode([tok1]))

    sample_len = len(tokenizer(prompts[0], return_tensors="pt")["input_ids"][0])
    logger.info("prompt_token_len[0]=%d (max_length=%d)", sample_len, args.max_length)

    start = datetime.datetime.now()
    probs: list[float] = []
    n_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        lo = bi * args.batch_size
        hi = lo + args.batch_size
        p1 = score_batch(tokenizer, model, prompts[lo:hi], tok0, tok1, args.max_length)
        probs.extend(p1.tolist())
        if bi % 25 == 0 or bi == n_batches - 1:
            logger.info("batch %d/%d", bi + 1, n_batches)
    elapsed = (datetime.datetime.now() - start).total_seconds()

    probs_arr = np.asarray(probs, dtype=np.float32)
    labels_arr = y_test["Outcome"].to_numpy().astype(np.int32)
    preds_arr = (probs_arr >= 0.5).astype(np.int32)
    try:
        auc = float(roc_auc_score(labels_arr, probs_arr))
    except ValueError:
        auc = 0.5
    metrics = {
        "auc": auc,
        "f1": float(f1_score(labels_arr, preds_arr, zero_division=0)),
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds_arr, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels_arr, preds_arr).tolist(),
        "positive_rate_labels": float(labels_arr.mean()),
        "positive_rate_preds": float(preds_arr.mean()),
    }

    summary = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "run_key": rk,
        "model_name": args.model_name,
        "task": args.task,
        "n_shots": args.n_shots,
        "n_test": int(n_test),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "elapsed_s": elapsed,
        "test_metrics": metrics,
        "label_token_0": int(tok0),
        "label_token_1": int(tok1),
        "prompt_token_len_example": int(sample_len),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s", out_path)
    logger.info("AUC=%.4f F1=%.4f Acc=%.4f (t=%.1fs)",
                metrics["auc"], metrics["f1"], metrics["accuracy"], elapsed)


if __name__ == "__main__":
    main()
