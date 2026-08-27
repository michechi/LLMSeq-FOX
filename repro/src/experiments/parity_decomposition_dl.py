"""
Parity decomposition — DL driver (Custom Transformer, LSTM).

Imports the model architectures and optimal configs from
`src.experiments.DL_TR_baselines_experiment`. Adds:

  * a VariantAwareDataset that yields either (letter_ids, label) or
    (letter_ids, mask_ids, label), depending on the input variant.
  * MaskedTransformer / MaskedLSTM wrappers that sum a per-position mask-bit
    embedding into the letter embedding before the existing encoder.
  * a variant-aware training loop, test-time prediction dump, and per-run
    JSON log under `results/parity_decomp/`.

The existing DL_TR_baselines_experiment.py is never mutated — we deep-copy
its OPTIMAL_CONFIGS to sidestep the module-level `config['max_seq_length']`
mutation in its own runner.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from src.experiments.DL_TR_baselines_experiment import (  # noqa: E402
    LSTMClassifier,
    OPTIMAL_CONFIGS,
    OPTIMAL_LR,
    TransformerClassifier,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

SEP = "\x1f"
VARIANTS = ("raw", "masked", "bitonly")
MODELS = ("Transformer", "LSTM")

# Shift all letter ids by +1 so 'A' -> 1, 'Z' -> 26. This keeps id 0 as a true
# padding slot (never used in our data), working around the base
# TransformerClassifier / LSTM / GRU behavior in
# src/experiments/DL_TR_baselines_experiment.py, which treats `x == 0` as
# padding (padding_idx=0 on the embedding + `(x == 0)` src_key_padding_mask
# in TransformerClassifier.forward). With the shift, real tokens never equal
# 0, so the mask is inactive and 'A' is no longer silently dropped from
# attention / pooling.
LETTER_OFFSET = 1
ALPHABET_SIZE = 26  # A..Z
MAIN_VOCAB_SIZE = ALPHABET_SIZE + LETTER_OFFSET  # = 27


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def set_seed_dl(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_tokens(seq: str) -> list[str]:
    if SEP in seq:
        return [t for t in seq.split(SEP) if t]
    return list(seq)


def _tokens_to_letter_ids(tokens: list[str]) -> list[int]:
    return [ord(t[0]) - ord("A") + LETTER_OFFSET for t in tokens]


def _parse_mask_bits(raw: str) -> list[int]:
    if SEP in raw:
        return [int(t) for t in raw.split(SEP) if t != ""]
    return [int(c) for c in raw if c in ("0", "1")]


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
class VariantAwareDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series | np.ndarray, variant: str):
        self.variant = variant
        self.sequences = X["Sequences"].astype(str).tolist()
        self.labels = np.asarray(y, dtype=np.int64)
        self.mask_bits = (
            X["MaskBits"].astype(str).tolist() if variant == "masked" else None
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        toks = _parse_tokens(self.sequences[idx])
        letter_ids = torch.tensor(_tokens_to_letter_ids(toks), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.variant == "masked":
            mask_ids = torch.tensor(_parse_mask_bits(self.mask_bits[idx]), dtype=torch.long)
            if mask_ids.numel() != letter_ids.numel():
                raise ValueError(
                    f"mask/letter length mismatch at idx={idx}: "
                    f"{mask_ids.numel()} vs {letter_ids.numel()}"
                )
            return letter_ids, mask_ids, y
        return letter_ids, y


def _collate_2(batch):
    letters, labels = zip(*batch)
    return torch.stack(letters), torch.stack(labels)


def _collate_3(batch):
    letters, masks, labels = zip(*batch)
    return torch.stack(letters), torch.stack(masks), torch.stack(labels)


# --------------------------------------------------------------------------- #
# Masked wrappers (per-position mask-bit channel summed into letter embed)    #
# --------------------------------------------------------------------------- #
class MaskedTransformer(nn.Module):
    def __init__(self, base_cfg: dict):
        super().__init__()
        self.base = TransformerClassifier(**base_cfg)
        self.mask_embedding = nn.Embedding(2, base_cfg["embedding_dim"])

    def forward(self, x_letter: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x_letter.size()
        embedded = self.base.embedding(x_letter)
        pos_enc = self.base.pos_encoding[:, :seq_len, :]
        embedded = embedded + pos_enc

        padding_mask = x_letter == 0  # from letter tensor, pre-sum
        mask_embed = self.mask_embedding(x_mask)
        mask_embed = mask_embed * (~padding_mask).unsqueeze(-1).float()
        embedded = embedded + mask_embed

        encoded = self.base.transformer(embedded, src_key_padding_mask=padding_mask)
        expanded = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * expanded).sum(dim=1) / (expanded.sum(dim=1) + 1e-9)
        pooled = self.base.layer_norm(pooled)
        pooled = self.base.dropout(pooled)
        return self.base.fc(pooled)


class MaskedLSTM(nn.Module):
    def __init__(self, base_cfg: dict):
        super().__init__()
        self.base = LSTMClassifier(**base_cfg)
        self.mask_embedding = nn.Embedding(2, base_cfg["embedding_dim"])

    def forward(self, x_letter: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        batch_size = x_letter.size(0)
        letter_embed = self.base.embedding(x_letter)
        mask_embed = self.mask_embedding(x_mask)
        embedded = letter_embed + mask_embed

        device = x_letter.device
        h0 = torch.zeros(self.base.num_layers, batch_size, self.base.hidden_dim, device=device)
        c0 = torch.zeros(self.base.num_layers, batch_size, self.base.hidden_dim, device=device)
        _, (h_n, _) = self.base.lstm(embedded, (h0, c0))
        last_hidden = h_n[-1]
        out = self.base.dropout(last_hidden)
        return self.base.fc(out)


# --------------------------------------------------------------------------- #
# Model factory                                                               #
# --------------------------------------------------------------------------- #
def _frozen_config(
    model_name: str,
    variant: str,
    max_seq_length: int,
    vocab_override: int | None = None,
) -> dict:
    cfg = copy.deepcopy(OPTIMAL_CONFIGS[model_name])
    if model_name == "Transformer":
        cfg["max_seq_length"] = max_seq_length
    # All letter ids are shifted by LETTER_OFFSET (=1). The main 26-letter
    # alphabet therefore occupies ids 1..26, leaving id 0 as true padding.
    # Bit-only stores its letters as {B, C}, which under the shift become
    # ids {2, 3}, still comfortably inside MAIN_VOCAB_SIZE. So we can use a
    # single vocab size for raw / masked / bitonly and still keep the
    # padding_idx=0 slot unused in real data.
    if vocab_override is not None:
        cfg["vocab_size"] = vocab_override
    else:
        cfg["vocab_size"] = MAIN_VOCAB_SIZE
    return cfg


def build_model(
    model_name: str,
    variant: str,
    max_seq_length: int,
    vocab_override: int | None = None,
) -> tuple[nn.Module, float, dict]:
    cfg = _frozen_config(model_name, variant, max_seq_length, vocab_override=vocab_override)
    lr = OPTIMAL_LR[model_name]
    if variant == "masked":
        if model_name == "Transformer":
            model = MaskedTransformer(cfg)
        elif model_name == "LSTM":
            model = MaskedLSTM(cfg)
        else:
            raise ValueError(f"unsupported model for masked variant: {model_name}")
    else:
        if model_name == "Transformer":
            model = TransformerClassifier(**cfg)
        elif model_name == "LSTM":
            model = LSTMClassifier(**cfg)
        else:
            raise ValueError(f"unsupported model: {model_name}")
    return model, lr, cfg


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def _forward(model: nn.Module, batch, device: torch.device, variant: str):
    if variant == "masked":
        x_l, x_m, y = (t.to(device, non_blocking=True) for t in batch)
        return model(x_l, x_m), y
    x_l, y = (t.to(device, non_blocking=True) for t in batch)
    return model(x_l), y


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    variant: str,
    epochs: int,
    lr: float,
    patience: int,
) -> dict:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = -1.0
    best = {}
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            logits, y = _forward(model, batch, device, variant)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        v_probs, v_preds, v_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                logits, y = _forward(model, batch, device, variant)
                p = torch.softmax(logits, dim=1)[:, 1]
                v_probs.extend(p.cpu().numpy())
                v_preds.extend(logits.argmax(dim=1).cpu().numpy())
                v_labels.extend(y.cpu().numpy())
        try:
            val_auc = float(roc_auc_score(v_labels, v_probs))
        except ValueError:
            val_auc = 0.5
        val_f1 = float(f1_score(v_labels, v_preds, zero_division=0))
        val_p = float(precision_score(v_labels, v_preds, zero_division=0))
        val_r = float(recall_score(v_labels, v_preds, zero_division=0))

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_precision": val_p,
                "val_recall": val_r,
            }
        )

        logger.info(
            "epoch %2d/%d  train_loss=%.4f  val_auc=%.4f  val_f1=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            val_auc,
            val_f1,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best = {
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_precision": val_p,
                "val_recall": val_r,
                "epoch": epoch + 1,
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best": best, "history": history}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    variant: str,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    for batch in loader:
        logits, y = _forward(model, batch, device, variant)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.extend(p.cpu().numpy())
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    probs_arr = np.asarray(probs, dtype=np.float32)
    preds_arr = np.asarray(preds, dtype=np.int32)
    labels_arr = np.asarray(labels, dtype=np.int32)
    try:
        auc = float(roc_auc_score(labels_arr, probs_arr))
    except ValueError:
        auc = 0.5
    cm = confusion_matrix(labels_arr, preds_arr).tolist()
    metrics = {
        "auc": auc,
        "f1": float(f1_score(labels_arr, preds_arr, zero_division=0)),
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds_arr, zero_division=0)),
        "confusion_matrix": cm,
    }
    return metrics, labels_arr, probs_arr, preds_arr


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def _variant_paths(data_dir: Path, variant: str, size_tag: str) -> dict:
    run_id = f"parity_{variant}_{size_tag}"
    val_run = f"parity_{variant}_full"
    test_run = f"parity_{variant}_full"
    return {
        "X_train": data_dir / f"X_train_{run_id}.csv",
        "y_train": data_dir / f"y_train_{run_id}.csv",
        "X_val": data_dir / f"X_val_{val_run}.csv",
        "y_val": data_dir / f"y_val_{val_run}.csv",
        "X_test": data_dir / f"X_test_{test_run}.csv",
        "y_test": data_dir / f"y_test_{test_run}.csv",
    }


def _anchor_paths(data_dir: Path, ell: int, k: int, n: int | None = None) -> dict:
    # Backward-compatible: n=20 (or n=None) keeps the legacy anchor_l{ell}_k{k}
    # tag so existing CSVs are reused; n != 20 produces anchor_l{ell}_k{k}_n{N}.
    tag = f"anchor_l{ell}_k{k}"
    if n is not None and n != 20:
        tag = f"{tag}_n{n}"
    return {
        "X_train": data_dir / f"X_train_{tag}.csv",
        "y_train": data_dir / f"y_train_{tag}.csv",
        "X_val": data_dir / f"X_val_{tag}.csv",
        "y_val": data_dir / f"y_val_{tag}.csv",
        "X_test": data_dir / f"X_test_{tag}.csv",
        "y_test": data_dir / f"y_test_{tag}.csv",
    }


def _main_curriculum_paths(data_dir: Path, n_stage: int) -> dict:
    tag = f"main_parity_n{n_stage}"
    return {
        "X_train": data_dir / f"X_train_{tag}.csv",
        "y_train": data_dir / f"y_train_{tag}.csv",
        "X_val": data_dir / f"X_val_{tag}.csv",
        "y_val": data_dir / f"y_val_{tag}.csv",
        "X_test": data_dir / f"X_test_{tag}.csv",
        "y_test": data_dir / f"y_test_{tag}.csv",
    }


def load_variant_data(
    variant: str,
    size_tag: str,
    data_dir: Path,
    anchor: tuple[int, int] | None = None,
    curriculum_n: int | None = None,
    anchor_n: int | None = None,
) -> tuple[dict[str, pd.DataFrame], str]:
    if curriculum_n is not None:
        paths = _main_curriculum_paths(data_dir, curriculum_n)
        logger.info("loading main-parity curriculum n=%d from %s",
                    curriculum_n, data_dir)
    elif anchor is not None:
        ell, k = anchor
        paths = _anchor_paths(data_dir, ell, k, n=anchor_n)
        logger.info("loading binary anchor ell=%d k=%d n=%s from %s",
                    ell, k, anchor_n if anchor_n is not None else "default", data_dir)
    else:
        paths = _variant_paths(data_dir, variant, size_tag)
        logger.info("loading %s / %s from %s", variant, size_tag, data_dir)

    out: dict[str, pd.DataFrame] = {}
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(p)
        out[name] = pd.read_csv(p).fillna("")

    if curriculum_n is not None:
        return out, "curriculum"
    return out, ("anchor" if anchor is not None else variant)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict:
    set_seed_dl(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s  variant=%s  model=%s  size=%s  seed=%d",
                device, args.variant, args.model, args.size, args.seed)

    anchor: tuple[int, int] | None = None
    anchor_n: int | None = None
    curriculum_n: int | None = None
    if args.size.startswith("main_n"):
        # main_n{N}  -> main-paper parity (K = {W,D,Q,J,X,N}, ell=26) at length N.
        # Used by Experiment F (length curriculum on main-paper parity).
        curriculum_n = int(args.size[len("main_n"):])
        variant_for_data = "raw"
    elif args.size.startswith("binary"):
        # binary{ell}            -> k = ell // 2 (balanced), n = ANCHOR_DEFAULT_N
        # binary{ell}k{k}        -> explicit k (e.g., binary26k4 for Exp B)
        # binary{ell}k{k}_n{N}   -> explicit (k, n); used by Experiment G length sweep
        rest = args.size[len("binary"):]
        if "_n" in rest:
            rest, n_str = rest.split("_n", 1)
            anchor_n = int(n_str)
        if "k" in rest:
            ell_str, k_str = rest.split("k", 1)
            ell = int(ell_str)
            k = int(k_str)
        else:
            ell = int(rest)
            k = max(1, ell // 2)
        anchor = (ell, k)
        variant_for_data = "bitonly"  # anchors are stored in bitonly-like B/C form
    else:
        variant_for_data = args.variant

    data, _ = load_variant_data(
        variant_for_data,
        args.size,
        Path(args.data_dir),
        anchor=anchor,
        curriculum_n=curriculum_n,
        anchor_n=anchor_n,
    )

    label_col = "Outcome"

    # Optional eval subsampling for dry-runs. Stratified, deterministic (seed).
    def _maybe_subsample_eval(X: pd.DataFrame, y_df: pd.DataFrame, cap: int | None) -> tuple[pd.DataFrame, np.ndarray]:
        if cap is None or cap <= 0 or cap >= len(X):
            return X, y_df[label_col].to_numpy()
        from sklearn.model_selection import train_test_split as _tts
        X_sub, _, y_sub, _ = _tts(
            X, y_df, train_size=cap, stratify=y_df[label_col], random_state=args.seed
        )
        return X_sub.reset_index(drop=True), y_sub[label_col].reset_index(drop=True).to_numpy()

    data["X_val"], y_val = _maybe_subsample_eval(data["X_val"], data["y_val"], args.max_eval_samples)
    data["X_test"], y_test = _maybe_subsample_eval(data["X_test"], data["y_test"], args.max_eval_samples)
    data["X_train"], y_train_arr = _maybe_subsample_eval(
        data["X_train"], data["y_train"], args.max_train_samples
    )
    y_train = y_train_arr

    variant_for_dataset = args.variant if anchor is None else "raw"
    train_ds = VariantAwareDataset(data["X_train"], y_train, variant_for_dataset)
    val_ds = VariantAwareDataset(data["X_val"], y_val, variant_for_dataset)
    test_ds = VariantAwareDataset(data["X_test"], y_test, variant_for_dataset)

    all_sequences = (
        data["X_train"]["Sequences"].tolist()
        + data["X_val"]["Sequences"].tolist()
        + data["X_test"]["Sequences"].tolist()
    )
    max_seq_length = max(len(_parse_tokens(s)) for s in all_sequences)
    logger.info("max_seq_length=%d  train=%d  val=%d  test=%d",
                max_seq_length, len(train_ds), len(val_ds), len(test_ds))

    collate = _collate_3 if variant_for_dataset == "masked" else _collate_2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=0)

    effective_variant = args.variant if anchor is None else "bitonly"

    # For anchors, pick vocab_size from the actual alphabet. Letters are
    # drawn from {B, C, D, ...} with ids starting at 1 + LETTER_OFFSET = 2.
    # max_id = alphabet_size + LETTER_OFFSET, so vocab = max_id + 1.
    vocab_override: int | None = None
    if anchor is not None:
        vocab_override = anchor[0] + LETTER_OFFSET + 1

    model, lr, cfg = build_model(args.model, effective_variant, max_seq_length, vocab_override=vocab_override)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model=%s  effective_variant=%s  n_params=%d  lr=%g",
                args.model, effective_variant, n_params, lr)

    if args.init_checkpoint is not None:
        ckpt_path = Path(args.init_checkpoint)
        logger.info("loading init checkpoint from %s", ckpt_path)
        state = torch.load(ckpt_path, map_location=device)
        # For curriculum warm-start, Stage 1 was trained at a shorter n, so
        # the pos_encoding param has shape [1, n1, d] vs [1, n2, d] in the
        # n=n2 model. Copy the overlapping slice, keep random init for the
        # rest. Works for any number of shape-mismatched parameters.
        current_state = model.state_dict()
        for key in list(state.keys()):
            if key in current_state and state[key].shape != current_state[key].shape:
                ckpt_shape = tuple(state[key].shape)
                cur_shape = tuple(current_state[key].shape)
                logger.info("shape-mismatch on %r: ckpt=%s model=%s; slice+merge",
                            key, ckpt_shape, cur_shape)
                merged = current_state[key].clone()
                slices = tuple(slice(0, min(a, b)) for a, b in zip(ckpt_shape, cur_shape))
                merged[slices] = state[key][slices].to(merged.dtype).to(merged.device)
                state[key] = merged
        model.load_state_dict(state)

    start = datetime.datetime.now()
    train_result = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        variant=variant_for_dataset,
        epochs=args.epochs,
        lr=lr,
        patience=args.patience,
    )
    elapsed = (datetime.datetime.now() - start).total_seconds()

    if args.save_checkpoint is not None:
        ckpt_path = Path(args.save_checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_path)
        logger.info("saved best-state checkpoint to %s", ckpt_path)

    test_metrics, test_labels, test_probs, test_preds = evaluate(
        model=model, loader=test_loader, device=device, variant=variant_for_dataset
    )

    # Persist
    output_dir = Path(args.output_dir)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    run_key = _run_key(args.variant, args.model, args.size, args.seed,
                       anchor=anchor, curriculum_n=curriculum_n,
                       anchor_n=anchor_n,
                       init_checkpoint=args.init_checkpoint)
    pred_path = output_dir / "predictions" / f"{run_key}.csv"
    pd.DataFrame(
        {"y_true": test_labels, "y_prob": test_probs, "y_pred": test_preds}
    ).to_csv(pred_path, index=False)

    summary = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "run_key": run_key,
        "variant": args.variant,
        "effective_variant": effective_variant,
        "model": args.model,
        "size": args.size,
        "seed": args.seed,
        "device": str(device),
        "n_params": n_params,
        "config": cfg,
        "optimizer": {"type": "Adam", "lr": lr},
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "data_dir": str(args.data_dir),
        "max_seq_length": max_seq_length,
        "training_time_s": elapsed,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "val_best": train_result["best"],
        "test_metrics": test_metrics,
        "history": train_result["history"],
        "predictions_csv": str(pred_path),
        "anchor": {"ell": anchor[0], "k": anchor[1]} if anchor else None,
        "curriculum_n": curriculum_n,
        "init_checkpoint": (str(args.init_checkpoint) if args.init_checkpoint else None),
        "save_checkpoint": (str(args.save_checkpoint) if args.save_checkpoint else None),
    }

    json_path = output_dir / f"{run_key}.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "[%s] test  AUC=%.4f  F1=%.4f  P=%.4f  R=%.4f  t=%.1fs",
        run_key,
        test_metrics["auc"],
        test_metrics["f1"],
        test_metrics["precision"],
        test_metrics["recall"],
        elapsed,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def _run_key(
    variant: str,
    model_name: str,
    size_tag: str,
    seed: int,
    anchor: tuple[int, int] | None = None,
    curriculum_n: int | None = None,
    anchor_n: int | None = None,
    init_checkpoint: str | None = None,
) -> str:
    if curriculum_n is not None:
        stage = "stage2" if init_checkpoint else "stage1"
        return f"expF_{stage}_{model_name}_n{curriculum_n}_seed{seed}"
    if anchor is not None:
        ell, k = anchor
        # Experiment G: balanced-case length sweep at l=26, |K|=13, n != 20.
        if anchor_n is not None and ell == 26 and k == 13:
            return f"expG_{model_name}_n{anchor_n}_seed{seed}"
        return f"anchor_l{ell}_k{k}_{model_name}_{seed}"
    return f"{variant}_{model_name}_{size_tag}_{seed}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parity decomposition — DL driver")
    p.add_argument("--variant", choices=VARIANTS, default="raw",
                   help="Input variant. Ignored when --size is binary2/binary4 "
                        "(anchors use the bitonly representation).")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--size", required=True,
                   help="Training-size tag: 1K, 40K, 400K, binary2, binary4")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data_dir", type=Path,
                   default=Path("data/simulation/parity_decomp"))
    p.add_argument("--output_dir", type=Path,
                   default=Path("results/parity_decomp"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Cap val/test at this many rows (stratified). Useful "
                        "for dry runs; leave unset for the full eval sets.")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Cap training size at this many rows (stratified). "
                        "Primarily for dry-running binary anchors without "
                        "regenerating the CSV. Leave unset to use the whole "
                        "training CSV.")
    p.add_argument("--init_checkpoint", type=Path, default=None,
                   help="Optional: load model state_dict from this .pt file before "
                        "training (fresh optimizer). Used by Experiment F curriculum "
                        "to warm-start Stage 2 (n=20) from Stage 1 (n=10) weights.")
    p.add_argument("--save_checkpoint", type=Path, default=None,
                   help="Optional: after training, save the best model state_dict "
                        "to this .pt file. Used by Experiment F Stage 1 to persist "
                        "weights for the Stage 2 warm-start.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
