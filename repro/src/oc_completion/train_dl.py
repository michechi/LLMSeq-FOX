"""Train the from-scratch sequence models (LSTM / Transformer / RNNTransformer)
on the regenerated OC classification data, with proper checkpoints, resumable
state and per-epoch matched-completion diagnostics.

Model classes, dataset/collate and encoding are IMPORTED from the repository's
canonical experiment script `src.experiments.DL_TR_baselines_experiment`
(single source for architectures). Training recipe follows the paper's
documented optimal configurations:

  LSTM:        emb 32, hidden 128, 2 layers, dropout 0.2, lr 2e-3
  Transformer: emb 64, ff 128, 3 layers, 4 heads, dropout 0.3, lr 5e-4
  RNNTransformer: repo OPTIMAL_CONFIGS (emb 64, rnn 32, 1 layer, ff 256,
                  dropout 0.1, lr 1e-3) - the paper appendix and code differ;
                  the code values are used and the deviation recorded.

  batch 64, Adam, cross-entropy, max 30 epochs, early stopping patience 5 on
  ORIGINAL validation F1 (argmax threshold, as in the repo code).

The per-epoch completion diagnostic (pair accuracy / mean margin on a fixed
completion-validation subset) is recorded but NEVER used for early stopping or
checkpoint selection.

Usage (from repro/ root):
    DATA_DIR=/root/LLMSeq/data python -m src.oc_completion.train_dl \
        --task ocdet --model LSTM --seed 9550 [--smoke] [--resume]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader

from src.experiments.DL_TR_baselines_experiment import (
    LetterSequenceDataset,
    OPTIMAL_CONFIGS,
    OPTIMAL_LR,
    collate_fn,
    create_model,
    preprocess_sequence,
)
from src.oc_completion.oracle import MECHANISM, N_EVENTS
from src.oc_completion.scoring import quick_pair_diag

REPO_ROOT = Path(os.environ.get("LLMSEQ_ROOT", "/root/LLMSeq"))
DATA_ROOT = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")) / "simulation" / "oc_completion"
RESULTS_DIR = REPO_ROOT / "results" / "matched_completion"
CKPT_ROOT = REPO_ROOT / "checkpoints" / "oc_completion"

TASK_DIRNAME = {"ocdet": "oc_deterministic", "ocnoisy": "oc_noisy"}

# Paper-documented optimal recipes (NLDL appendix, tab:dl_optimal); these are
# the task-spec expected defaults. RNNTransformer uses the repo's as-run
# OPTIMAL_CONFIGS because paper and code disagree (recorded deviation).
RECIPES = {
    "LSTM": {
        "config": {"vocab_size": 26, "embedding_dim": 32, "hidden_dim": 128,
                   "num_layers": 2, "num_classes": 2, "dropout": 0.2},
        "lr": 2e-3,
        "provenance": "paper NLDL tab:dl_optimal (code OPTIMAL_CONFIGS differs:"
                      " emb64/drop0.5/lr5e-4 - recorded deviation)",
    },
    "Transformer": {
        "config": {"vocab_size": 26, "embedding_dim": 64, "num_heads": 4,
                   "num_layers": 3, "dim_feedforward": 128, "num_classes": 2,
                   "dropout": 0.3, "max_seq_length": N_EVENTS},
        "lr": 5e-4,
        "provenance": "paper NLDL tab:dl_optimal (code OPTIMAL_CONFIGS differs:"
                      " 8heads/2layers/ff512/lr1e-3 - recorded deviation)",
    },
    "RNNTransformer": {
        "config": {**OPTIMAL_CONFIGS["RNNTransformer"], "max_seq_length": N_EVENTS},
        "lr": OPTIMAL_LR["RNNTransformer"],
        "provenance": "repo OPTIMAL_CONFIGS (paper appendix differs:"
                      " emb128/1+3layers/drop0.3/lr1e-4 - recorded deviation)",
    },
}

BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5

RESULT_COLUMNS = [
    "timestamp", "model", "training_mode", "task", "seed", "train_rows",
    "epochs_done", "best_epoch", "val_f1", "val_auc", "val_loss",
    "test_auc_obs", "test_auc_latent", "test_f1", "test_precision",
    "test_recall", "threshold", "n_params", "recipe", "wallclock_s",
    "checkpoint", "smoke", "host",
]


def append_result(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if os.fstat(f.fileno()).st_size == 0:
                f.write(",".join(RESULT_COLUMNS) + "\n")
            f.write(",".join(_csv_cell(row.get(c, "")) for c in RESULT_COLUMNS) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _csv_cell(v) -> str:
    s = str(v)
    if "," in s or '"' in s:
        s = '"' + s.replace('"', '""') + '"'
    return s


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(task: str, split: str, smoke: bool):
    tag = f"{task}_smoke" if smoke else task
    X = pd.read_csv(DATA_ROOT / f"X_{split}_{tag}.csv")["Sequences"].apply(
        preprocess_sequence)
    y = pd.read_csv(DATA_ROOT / f"y_{split}_{tag}.csv")
    return X.tolist(), y["Outcome"].astype(int).tolist(), y["Latent"].astype(int).tolist()


def make_loader(X, y, shuffle, batch_size=BATCH_SIZE, generator=None):
    ds = LetterSequenceDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0, generator=generator)


@torch.no_grad()
def model_logits(model, seqs, device, batch_size=1024) -> np.ndarray:
    """score_fn interface for scoring: sequence strings -> (N, 2) logits."""
    model.eval()
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = [preprocess_sequence(s) for s in seqs[i:i + batch_size]]
        enc = torch.stack([
            torch.tensor([ord(c) - ord("A") for c in s], dtype=torch.long)
            for s in chunk]).to(device)
        out.append(model(enc).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, probs, preds, labels = [], [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        losses.append(criterion(logits, yb).item() * len(yb))
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.extend(p.cpu().numpy())
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(yb.cpu().numpy())
    probs, preds, labels = map(np.asarray, (probs, preds, labels))
    return {
        "loss": float(np.sum(losses) / len(labels)),
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.5,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "probs": probs,
        "preds": preds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ocdet", "ocnoisy"], required=True)
    ap.add_argument("--model", choices=list(RECIPES), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--diag_pairs", type=int, default=500,
                    help="completion-validation pairs scored per epoch")
    ap.add_argument("--pairs_dir", type=Path, default=None)
    ap.add_argument("--results_csv", type=Path,
                    default=RESULTS_DIR / "training_results.csv")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "_smoke" if args.smoke else ""

    run_dir = (CKPT_ROOT / TASK_DIRNAME[args.task] / args.model.lower()
               / f"seed_{args.seed}{suffix}")
    run_dir.mkdir(parents=True, exist_ok=True)
    done_marker = run_dir / "done.json"
    if done_marker.exists() and args.resume:
        print(f"[train_dl] {run_dir} already complete - skipping")
        return

    recipe = RECIPES[args.model]
    set_seed(args.seed)

    X_train, y_train, _ = load_split(args.task, "train", args.smoke)
    X_val, y_val, _ = load_split(args.task, "val", args.smoke)
    X_test, y_test, ystar_test = load_split(args.task, "test", args.smoke)

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = make_loader(X_train, y_train, True, generator=gen)
    val_loader = make_loader(X_val, y_val, False)
    test_loader = make_loader(X_test, y_test, False)

    # fixed completion-validation diagnostic set (never used for selection)
    pairs_dir = args.pairs_dir or (DATA_ROOT / f"pairs{suffix}")
    diag_path = pairs_dir / f"pairs_two_hole_heldout_val{suffix}.csv"
    diag_pairs = pd.read_csv(diag_path).head(args.diag_pairs)

    model = create_model(args.model, recipe["config"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=recipe["lr"])
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_f1, best_epoch, no_improve = 0, -1.0, -1, 0
    history = []
    t_prev = 0.0
    last_path, best_path, final_path = (run_dir / "last.pt",
                                        run_dir / "best.pt",
                                        run_dir / "final.pt")
    if args.resume and last_path.exists():
        # always load to CPU: rng states must stay ByteTensors on CPU, and
        # model/optimizer load_state_dict move tensors to the live device
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_f1, best_epoch = ck["best_f1"], ck["best_epoch"]
        no_improve, history = ck["no_improve"], ck["history"]
        t_prev = ck.get("wallclock_s", 0.0)
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["np_rng"])
        random.setstate(ck["py_rng"])
        print(f"[train_dl] resumed at epoch {start_epoch}")

    config_payload = {
        "model": args.model, "task": args.task, "seed": args.seed,
        "recipe": recipe, "batch_size": BATCH_SIZE, "optimizer": "Adam",
        "loss": "CrossEntropyLoss", "max_epochs": args.max_epochs,
        "patience": args.patience, "early_stopping_metric": "val_f1(argmax)",
        "threshold": "argmax", "mechanism": MECHANISM,
        "data_dir": str(DATA_ROOT), "smoke": args.smoke,
        "diag_pairs_file": str(diag_path), "n_diag_pairs": len(diag_pairs),
        "note": "completion diagnostics never influence early stopping or "
                "checkpoint selection",
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_payload, f, indent=2, default=str)

    t0 = time.time()
    stopped = False
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        ep_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(yb)
            seen += len(yb)

        val = evaluate(model, val_loader, device, criterion)
        diag = quick_pair_diag(
            lambda seqs: model_logits(model, seqs, device), diag_pairs)
        rec = {
            "epoch": epoch + 1,
            "train_loss": ep_loss / seen,
            "val_loss": val["loss"], "val_auc": val["auc"],
            "val_f1": val["f1"], "val_precision": val["precision"],
            "val_recall": val["recall"], **diag,
            "seconds": round(t_prev + time.time() - t0, 1),
        }
        history.append(rec)
        print(f"[train_dl] {args.model}/{args.task}/s{args.seed} "
              f"ep{epoch+1} loss={rec['train_loss']:.4f} "
              f"val_f1={val['f1']:.4f} val_auc={val['auc']:.4f} "
              f"pair_acc={diag['completion_pair_acc']:.4f}", flush=True)

        if val["f1"] > best_f1:
            best_f1, best_epoch, no_improve = val["f1"], epoch + 1, 0
            torch.save({"model": model.state_dict(),
                        "config": recipe["config"], "model_name": args.model,
                        "epoch": epoch + 1, "val": {k: v for k, v in val.items()
                                                    if k not in ("probs", "preds")},
                        "seed": args.seed, "task": args.task},
                       best_path)
        else:
            no_improve += 1

        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_f1": best_f1,
                    "best_epoch": best_epoch, "no_improve": no_improve,
                    "history": history,
                    "wallclock_s": t_prev + time.time() - t0,
                    "torch_rng": torch.get_rng_state(),
                    "np_rng": np.random.get_state(),
                    "py_rng": random.getstate()}, last_path)
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_improve >= args.patience:
            stopped = True
            break

    torch.save({"model": model.state_dict(), "config": recipe["config"],
                "model_name": args.model, "epoch": len(history),
                "seed": args.seed, "task": args.task}, final_path)

    # restore best checkpoint for test evaluation
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test = evaluate(model, test_loader, device, criterion)
    auc_latent = (float(roc_auc_score(ystar_test, test["probs"]))
                  if len(set(ystar_test)) > 1 else 0.5)
    wall = round(t_prev + time.time() - t0, 1)

    val_best = best["val"]
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model, "training_mode": "scratch", "task": args.task,
        "seed": args.seed, "train_rows": len(X_train),
        "epochs_done": len(history), "best_epoch": best_epoch,
        "val_f1": round(val_best["f1"], 6), "val_auc": round(val_best["auc"], 6),
        "val_loss": round(val_best["loss"], 6),
        "test_auc_obs": round(test["auc"], 6),
        "test_auc_latent": round(auc_latent, 6),
        "test_f1": round(test["f1"], 6),
        "test_precision": round(test["precision"], 6),
        "test_recall": round(test["recall"], 6),
        "threshold": "argmax", "n_params": n_params,
        "recipe": f"{args.model} {recipe['config']} lr={recipe['lr']} "
                  f"Adam bs={BATCH_SIZE} patience={args.patience} "
                  f"early=val_f1 [{recipe['provenance']}]",
        "wallclock_s": wall, "checkpoint": str(run_dir),
        "smoke": args.smoke, "host": os.uname().nodename,
    }
    append_result(args.results_csv, row)
    with open(done_marker, "w") as f:
        json.dump(row, f, indent=2, default=str)
    print(f"[train_dl] DONE {args.model}/{args.task}/s{args.seed}: "
          f"test_auc_obs={test['auc']:.4f} auc_latent={auc_latent:.4f} "
          f"best_epoch={best_epoch} wall={wall}s", flush=True)


if __name__ == "__main__":
    main()
