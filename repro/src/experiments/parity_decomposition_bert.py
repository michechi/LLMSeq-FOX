"""
Parity decomposition, BERT driver (AutoModelForSequenceClassification).

Sibling to src/experiments/parity_decomposition_llm.py. Reuses _variant_paths,
_anchor_paths, _load_csvs, _render_prompts from the LLM driver, and
find_optimal_threshold from LLM_fraction_experiment. Training loop is a plain
AdamW + cross-entropy loop with val-metric early stopping, following the pattern
in src/experiments/score_counterfactual_pairs.py:run_llm_encoder.

Anchor datasets (size=binary{ell}) are consumed directly: stored letters from a
smaller alphabet (e.g. {B,C} for ell=2, {A..Z} for ell=26) are rendered as raw
"Sequential events: ..." prompts regardless of --variant.
"""

from __future__ import annotations

import argparse
import datetime
import gc
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class BertMeanPoolClassifier(nn.Module):
    """BertModel encoder + mask-aware mean-pool + dropout + Linear head.
    Drop-in replacement for BertForSequenceClassification that classifies on the
    mean of last_hidden_state over attended positions instead of [CLS]+pooler.
    """

    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1,
                 from_config: bool = False, cache_dir: str | None = None):
        super().__init__()
        from transformers import AutoConfig
        if from_config:
            cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            self.bert = AutoModel.from_config(cfg)
        else:
            self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, T, D]
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        loss = F.cross_entropy(logits, labels) if labels is not None else None

        class _Out:
            pass
        o = _Out()
        o.loss = loss
        o.logits = logits
        return o

from src.experiments.LLM_fraction_experiment import find_optimal_threshold
from src.experiments.parity_decomposition_llm import (
    VARIANTS,
    _anchor_paths,
    _load_csvs,
    _render_prompts,
    _variant_paths,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BertSeqClsDataset(Dataset):
    """Eagerly-tokenized dataset for BERT SequenceClassification."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def train_and_evaluate_bert(model, train_loader, val_loader, args, device):
    """AdamW + cross-entropy training with val-based early stopping.

    Returns best_auc, best_f1, best_val_loss, epochs_done, val_probs, val_labels
    taken at the best-validation epoch.
    """
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_auc = 0.0
    best_f1 = 0.0
    best_state: dict | None = None
    best_val_probs: np.ndarray | None = None
    best_val_labels: np.ndarray | None = None
    no_improve = 0
    epochs_done = 0

    for epoch in range(args.epochs):
        epochs_done = epoch + 1
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach().item()
            n_batches += 1
        avg_train_loss = total_loss / max(1, n_batches)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        probs_accum: list[float] = []
        labels_accum: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                val_loss += float(out.loss.item())
                val_batches += 1
                p = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy()
                probs_accum.extend(p.tolist())
                labels_accum.extend(labels.detach().cpu().numpy().tolist())
        avg_val_loss = val_loss / max(1, val_batches)
        val_probs_arr = np.asarray(probs_accum, dtype=np.float32)
        val_labels_arr = np.asarray(labels_accum, dtype=np.int32)
        try:
            val_auc = float(roc_auc_score(val_labels_arr, val_probs_arr))
        except ValueError:
            val_auc = 0.5
        val_preds_half = (val_probs_arr >= 0.5).astype(np.int32)
        val_f1 = float(f1_score(val_labels_arr, val_preds_half, zero_division=0))

        logger.info(
            "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  val_f1=%.4f",
            epoch + 1,
            args.epochs,
            avg_train_loss,
            avg_val_loss,
            val_auc,
            val_f1,
        )

        if args.early == "loss":
            improved = avg_val_loss < best_val_loss
        elif args.early == "auc":
            improved = val_auc > best_auc
        else:
            improved = val_f1 > best_f1

        if improved:
            best_val_loss = avg_val_loss
            best_auc = val_auc
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val_probs = val_probs_arr
            best_val_labels = val_labels_arr
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return (
        best_auc,
        best_f1,
        best_val_loss,
        epochs_done,
        best_val_probs if best_val_probs is not None else np.zeros(0, dtype=np.float32),
        best_val_labels if best_val_labels is not None else np.zeros(0, dtype=np.int32),
    )


@torch.no_grad()
def evaluate_with_predictions(model, test_loader, device, threshold: float = 0.5):
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        p = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(labels_t.detach().cpu().numpy().tolist())
    probs_arr = np.asarray(probs, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)
    preds_arr = (probs_arr >= threshold).astype(np.int32)
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
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(labels_arr, preds_arr).tolist(),
    }
    return metrics, labels_arr, probs_arr, preds_arr


def _run_key(variant, size_tag, seed, anchor, curriculum_n=None, anchor_n=None,
             init_checkpoint=None):
    if curriculum_n is not None:
        stage = "stage2" if init_checkpoint else "stage1"
        return f"expF_{stage}_BERT_n{curriculum_n}_seed{seed}"
    if anchor is not None:
        ell, k = anchor
        # Experiment G: balanced-case length sweep at l=26, |K|=13, n != 20.
        if anchor_n is not None and ell == 26 and k == 13:
            return f"expG_BERT_n{anchor_n}_seed{seed}"
        return f"anchor_l{ell}_k{k}_BERT_{seed}"
    return f"{variant}_BERT_{size_tag}_{seed}"


def _main_curriculum_paths_bert(data_dir, n_stage):
    tag = f"main_parity_n{n_stage}"
    return {
        "X_train": data_dir / f"X_train_{tag}.csv",
        "y_train": data_dir / f"y_train_{tag}.csv",
        "X_val":   data_dir / f"X_val_{tag}.csv",
        "y_val":   data_dir / f"y_val_{tag}.csv",
        "X_test":  data_dir / f"X_test_{tag}.csv",
        "y_test":  data_dir / f"y_test_{tag}.csv",
    }


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "device=%s  variant=%s  size=%s  seed=%d  model=%s",
        device,
        args.variant,
        args.size,
        args.seed,
        args.model_name,
    )

    anchor: tuple[int, int] | None = None
    anchor_n: int | None = None
    curriculum_n: int | None = None
    if args.size.startswith("main_n"):
        # main_n{N}  -> main-paper parity (K={W,D,Q,J,X,N}, ell=26) at length N.
        # Used by Experiment F (length curriculum on main-paper parity).
        curriculum_n = int(args.size[len("main_n"):])
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

    if curriculum_n is not None:
        paths = _main_curriculum_paths_bert(Path(args.data_dir), curriculum_n)
        variant_for_prompt = "raw"
    elif anchor is not None:
        paths = _anchor_paths(Path(args.data_dir), anchor[0], anchor[1], n=anchor_n)
        variant_for_prompt = "raw"
    else:
        paths = _variant_paths(Path(args.data_dir), args.variant, args.size)
        variant_for_prompt = args.variant

    data = _load_csvs(paths)
    y_train = data["y_train"]["Outcome"].tolist()
    y_val = data["y_val"]["Outcome"].tolist()
    y_test = data["y_test"]["Outcome"].tolist()

    logger.info(
        "data sizes  train=%d  val=%d  test=%d",
        len(data["X_train"]),
        len(data["X_val"]),
        len(data["X_test"]),
    )

    train_texts = _render_prompts(data["X_train"], variant_for_prompt)
    val_texts = _render_prompts(data["X_val"], variant_for_prompt)
    test_texts = _render_prompts(data["X_test"], variant_for_prompt)
    if args.no_prompt:
        SEP = "\x1f"

        def _strip(seqs):
            return [s.replace(SEP, " ") for s in seqs]

        train_texts = _strip(data["X_train"]["Sequences"].tolist())
        val_texts = _strip(data["X_val"]["Sequences"].tolist())
        test_texts = _strip(data["X_test"]["Sequences"].tolist())
    if train_texts:
        logger.info("sample prompt (%s, no_prompt=%s): %r",
                    variant_for_prompt, args.no_prompt, train_texts[0])

    cache_dir = str(args.cache_dir) if args.cache_dir else None
    tok_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name, cache_dir=cache_dir)

    train_ds = BertSeqClsDataset(train_texts, y_train, tokenizer, max_length=args.max_length)
    val_ds = BertSeqClsDataset(val_texts, y_val, tokenizer, max_length=args.max_length)
    test_ds = BertSeqClsDataset(test_texts, y_test, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    if args.mean_pool:
        model = BertMeanPoolClassifier(
            args.model_name, num_labels=2, dropout=0.1,
            from_config=args.from_config, cache_dir=cache_dir,
        ).to(device)
        logger.info("model: BertMeanPoolClassifier(model_name=%s, from_config=%s)",
                    args.model_name, args.from_config)
    elif args.from_config:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_name, num_labels=2,
                                         cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_config(cfg).to(device)
        logger.info("model: random-init BERT from config %s (no pretrained weights)",
                    args.model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2, cache_dir=cache_dir
        ).to(device)

    if args.init_checkpoint is not None:
        ckpt_path = Path(args.init_checkpoint)
        logger.info("loading init checkpoint from %s", ckpt_path)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

    start = datetime.datetime.now()
    best_auc, best_f1, best_val_loss, epochs_done, val_probs, val_labels = train_and_evaluate_bert(
        model, train_loader, val_loader, args, device
    )
    optimal_threshold, val_f1_opt = find_optimal_threshold(val_labels, val_probs)
    logger.info("optimal threshold=%.3f (val_f1=%.4f)", optimal_threshold, val_f1_opt)

    test_metrics, test_labels, test_probs, test_preds = evaluate_with_predictions(
        model, test_loader, device, threshold=optimal_threshold
    )
    elapsed = (datetime.datetime.now() - start).total_seconds()

    output_dir = Path(args.output_dir)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    run_key = _run_key(
        args.variant, args.size, args.seed, anchor,
        curriculum_n=curriculum_n,
        anchor_n=anchor_n,
        init_checkpoint=args.init_checkpoint,
    )
    pred_path = output_dir / "predictions" / f"{run_key}.csv"

    if args.save_checkpoint is not None:
        ckpt_path = Path(args.save_checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items()},
            ckpt_path,
        )
        logger.info("saved best-state checkpoint to %s", ckpt_path)
    pd.DataFrame(
        {"y_true": test_labels, "y_prob": test_probs, "y_pred": test_preds}
    ).to_csv(pred_path, index=False)

    summary = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "run_key": run_key,
        "variant": args.variant,
        "variant_for_prompt": variant_for_prompt,
        "model": "BERT",
        "model_name": args.model_name,
        "size": args.size,
        "seed": args.seed,
        "device": str(device),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "optimizer": "AdamW",
        "early": args.early,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "training_time_s": elapsed,
        "val_best": {
            "auc": float(best_auc),
            "f1": float(best_f1),
            "loss": float(best_val_loss),
            "epochs_done": int(epochs_done),
        },
        "optimal_threshold": float(optimal_threshold),
        "val_f1_at_optimal_threshold": float(val_f1_opt),
        "test_metrics": test_metrics,
        "anchor": {"ell": anchor[0], "k": anchor[1]} if anchor else None,
        "curriculum_n": curriculum_n,
        "init_checkpoint": (str(args.init_checkpoint) if args.init_checkpoint else None),
        "save_checkpoint": (str(args.save_checkpoint) if args.save_checkpoint else None),
        "predictions_csv": str(pred_path),
        "sample_prompt": train_texts[0] if train_texts else "",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parity decomposition, BERT driver (AutoModelForSequenceClassification)"
    )
    p.add_argument("--variant", choices=VARIANTS, default="raw")
    p.add_argument(
        "--size",
        required=True,
        help="Training-size tag: 1K, 40K, 400K, binary2, binary4, binary8, binary16, binary26",
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model_name", default="google-bert/bert-base-uncased")
    p.add_argument("--data_dir", type=Path, default=Path("data/simulation/parity_decomp"))
    p.add_argument("--output_dir", type=Path, default=Path("results/parity_decomp"))
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=Path(os.environ.get("SCRATCH", "/tmp")) / "cache",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--early", choices=["auc", "loss", "f1"], default="loss")
    p.add_argument("--init_checkpoint", type=Path, default=None,
                   help="Optional: load BERT state_dict from this .pt file before "
                        "training (fresh optimizer). Used by Experiment F curriculum "
                        "to warm-start Stage 2 (n=20) from Stage 1 (n=10).")
    p.add_argument("--save_checkpoint", type=Path, default=None,
                   help="Optional: after training, save the fine-tuned state_dict "
                        "to this .pt file for Experiment F Stage 1.")
    p.add_argument("--no_prompt", action="store_true",
                   help="Bypass the 'Sequential events: ... Outcome (0 or 1):' "
                        "wrapper from make_llm_prompt and feed BERT only the "
                        "space-separated letter sequence.")
    p.add_argument("--from_config", action="store_true",
                   help="Random-init BERT (architecture only) instead of "
                        "from_pretrained. Diagnostic for whether the failure "
                        "mode is the pretrained basin or the architecture.")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="AdamW weight decay (default 0.01).")
    p.add_argument("--tokenizer_name", type=str, default=None,
                   help="Override tokenizer (defaults to --model_name). "
                        "Useful when model_name doesn't ship its own tokenizer "
                        "(e.g. prajjwal1/bert-tiny which reuses bert-base-uncased's "
                        "WordPiece vocab).")
    p.add_argument("--mean_pool", action="store_true",
                   help="Use mask-aware mean-pool over last_hidden_state instead "
                        "of [CLS]+BertPooler. Diagnostic for whether the [CLS]-style "
                        "pooling is the bottleneck on parity.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
