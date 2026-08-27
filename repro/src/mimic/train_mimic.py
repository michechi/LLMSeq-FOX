"""
Unified training script for MIMIC CKD->ESRD ordered vs shuffled experiments
(paper Table tab:mimic_ckd_audit, neural rows).

Supports:
  - DL models (integer encoding):  transformer, lstm, bilstm
  - LLM-style fine-tuning:         bert, llama
  - Bag-of-codes baselines:        xgboost, logreg

Usage:
    python -m src.mimic.train_mimic --model transformer --data ordered
    python -m src.mimic.train_mimic --model lstm --data shuffled
    python -m src.mimic.train_mimic --model bert --data ordered --peft
    python -m src.mimic.train_mimic --model llama --data ordered --tiny --tiny_type 1M
    python -m src.mimic.train_mimic --model llama --data ordered --peft

Inputs:  <MIMIC_TRAINING>/{X,y}_{train,val,test}_{ordered,shuffled}.csv,
         <MIMIC_TRAINING>/vocab.json
Outputs: <MIMIC_RESULTS>/model_results.csv (one row per run, appended)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common import CACHE_DIR, MIMIC_RESULTS, MIMIC_TRAINING, SEP

try:
    import torch
    import torch.nn as nn
    from torch.amp import autocast
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LlamaConfig,
        get_linear_schedule_with_warmup,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class _Dummy:
        pass

    Dataset = _Dummy

    class nn:
        Module = _Dummy

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

DELIMITER = SEP
DATA_DIR = str(MIMIC_TRAINING)
RESULTS_CSV = str(Path(MIMIC_RESULTS) / "model_results.csv")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MIMIC ordered vs shuffled training")
    p.add_argument("--model", type=str, required=True,
                   choices=["transformer", "lstm", "bilstm", "bert", "llama",
                            "xgboost", "logreg"])
    p.add_argument("--data", type=str, required=True, choices=["ordered", "shuffled"])

    # DL hyper-parameters
    p.add_argument("--max_seq", type=int, default=1024,
                   help="Max sequence length for DL models")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)

    # LLM / BERT specific
    p.add_argument("--max_length", type=int, default=None,
                   help="Tokenizer max_length. Default: 512 (bert), 2048 (llama)")
    p.add_argument("--llm_lr", type=float, default=2e-5)
    p.add_argument("--llm_batch_size", type=int, default=8)
    p.add_argument("--llm_patience", type=int, default=3)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # LLM model options
    p.add_argument("--model_name", type=str, default=None,
                   help="HuggingFace model name "
                        "(default: bert-base-uncased / meta-llama/Llama-3.1-8B)")
    p.add_argument("--tiny", action="store_true", help="Use tiny model (llama only)")
    p.add_argument("--tiny_type", type=str, default="1M")
    p.add_argument("--peft", action="store_true", help="Use LoRA")
    p.add_argument("--use_quantization", action="store_true", help="4-bit quantization (llama)")
    p.add_argument("--cold_start", action="store_true",
                   help="Random-init: same architecture, no pretrained weights")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# DL Dataset: integer-encoded sequences
# ---------------------------------------------------------------------------
class IntegerSeqDataset(Dataset):
    def __init__(self, sequences, labels, code_to_idx, max_seq):
        self.labels = labels
        self.max_seq = max_seq
        self.encoded = []
        for seq in sequences:
            tokens = seq.split(DELIMITER)
            ids = [code_to_idx.get(t, 0) for t in tokens][:max_seq]
            ids += [0] * (max_seq - len(ids))
            self.encoded.append(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# DL Model architectures
# ---------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_heads=4,
                 num_layers=2, dim_feedforward=256, num_classes=2,
                 dropout=0.3, max_seq_length=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        _, seq_len = x.size()
        embedded = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        padding_mask = (x == 0)
        encoded = self.transformer(embedded, src_key_padding_mask=padding_mask)
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)
        pooled = self.dropout(self.layer_norm(pooled))
        return self.fc(pooled)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64,
                 num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        _, (h_n, _) = self.lstm(embedded, (h0, c0))
        return self.fc(self.dropout(h_n[-1]))


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64,
                 num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=x.device)
        _, (h_n, _) = self.lstm(embedded, (h0, c0))
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(last_hidden))


# ---------------------------------------------------------------------------
# LLM Datasets
# ---------------------------------------------------------------------------
def standard_narrative_prompt(seq_str: str) -> str:
    events = seq_str.split(DELIMITER)
    return f"Sequential events: {' '.join(events)}\nOutcome (0 or 1):"


class TemporalCausalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        causal_labels = input_ids.clone()
        causal_labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": causal_labels,
            "outcome_labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SequenceClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class CausalLMWithClassificationHead(nn.Module):
    def __init__(self, backbone_model, num_classes=2):
        super().__init__()
        self.backbone = backbone_model
        self.config = backbone_model.config
        self.classification_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size // 2, num_classes),
        )
        self.classification_head = self.classification_head.to(
            dtype=backbone_model.dtype, device=backbone_model.device
        )

    def forward(self, input_ids, attention_mask=None, labels=None, outcome_labels=None):
        causal_outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, output_hidden_states=True, return_dict=True,
        )
        hidden_states = causal_outputs.hidden_states[-1]
        batch_size = input_ids.shape[0]
        reps = []
        for b in range(batch_size):
            if attention_mask is not None:
                pos = attention_mask[b].sum().item() - 1
            else:
                pos = input_ids.shape[1] - 1
            reps.append(hidden_states[b, pos, :])
        cls_input = torch.stack(reps)
        cls_logits = self.classification_head(cls_input)

        if outcome_labels is not None:
            cls_loss = nn.functional.cross_entropy(cls_logits, outcome_labels)
            total_loss = causal_outputs.loss + cls_loss
            return {"loss": total_loss, "logits": cls_logits}
        return {"loss": causal_outputs.loss, "logits": cls_logits}

    def resize_token_embeddings(self, new_num_tokens):
        return self.backbone.resize_token_embeddings(new_num_tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(data_type: str):
    out = {}
    for split in ("train", "val", "test"):
        x = pd.read_csv(
            os.path.join(DATA_DIR, f"X_{split}_{data_type}.csv"),
            na_values=["", "None", "NaN", "na", "nan"],
        ).fillna("")
        y = pd.read_csv(
            os.path.join(DATA_DIR, f"y_{split}_{data_type}.csv"),
            na_values=["", "None", "NaN", "na", "nan"],
        ).fillna("")
        out[split] = (x, y)
    return out


def compute_class_weights(y_train):
    labels = y_train["Outcome"].values
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    w = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)
    logger.info(f"Class weights: {w.tolist()}")
    return w


def find_optimal_threshold(labels, preds):
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        f = f1_score(labels, (np.array(preds) >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = t
    return best_thresh, best_f1


def evaluate_test(labels, preds, threshold):
    pred_bin = (np.array(preds) >= threshold).astype(int)
    return {
        "auc": roc_auc_score(labels, preds),
        "f1": f1_score(labels, pred_bin, zero_division=0),
        "precision": precision_score(labels, pred_bin, zero_division=0),
        "recall": recall_score(labels, pred_bin, zero_division=0),
    }


def append_results(row):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a") as f:
        if write_header:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


def _hf_cache_dir() -> str:
    cache = os.environ.get("SCRATCH")
    if cache:
        cache = os.path.join(cache, "cache")
    else:
        cache = str(CACHE_DIR / "hf")
    os.makedirs(cache, exist_ok=True)
    return cache


# ---------------------------------------------------------------------------
# DL training loop
# ---------------------------------------------------------------------------
def train_dl(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    with open(os.path.join(DATA_DIR, "vocab.json")) as f:
        vocab_info = json.load(f)
    code_to_idx = vocab_info["code_to_idx"]
    vocab_size = vocab_info["vocab_size"]
    logger.info(f"Vocab size: {vocab_size}")

    data = load_data(args.data)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    class_weights = compute_class_weights(y_train).to(device)

    train_ds = IntegerSeqDataset(X_train["Sequences"].tolist(),
                                 y_train["Outcome"].tolist(), code_to_idx, args.max_seq)
    val_ds = IntegerSeqDataset(X_val["Sequences"].tolist(),
                               y_val["Outcome"].tolist(), code_to_idx, args.max_seq)
    test_ds = IntegerSeqDataset(X_test["Sequences"].tolist(),
                                y_test["Outcome"].tolist(), code_to_idx, args.max_seq)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    if args.model == "transformer":
        model = TransformerClassifier(
            vocab_size=vocab_size, embedding_dim=64, num_heads=4,
            num_layers=2, dim_feedforward=256, dropout=args.dropout,
            max_seq_length=args.max_seq,
        )
    elif args.model == "lstm":
        model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=32,
                               hidden_dim=64, num_layers=2, dropout=args.dropout)
    elif args.model == "bilstm":
        model = BiLSTMClassifier(vocab_size=vocab_size, embedding_dim=32,
                                 hidden_dim=64, num_layers=2, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown DL model: {args.model}")

    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    final_epoch = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        final_epoch = epoch + 1

    elapsed = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs = seqs.to(device)
            probs = torch.softmax(model(seqs), dim=-1)[:, 1].cpu().numpy()
            val_preds.extend(probs)
            val_labels.extend(labels.numpy())
    threshold, val_f1 = find_optimal_threshold(val_labels, val_preds)
    logger.info(f"Optimal threshold: {threshold:.2f} (Val F1: {val_f1:.4f})")

    test_preds, test_labels = [], []
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            probs = torch.softmax(model(seqs), dim=-1)[:, 1].cpu().numpy()
            test_preds.extend(probs)
            test_labels.extend(labels.numpy())
    metrics = evaluate_test(test_labels, test_preds, threshold)
    logger.info(
        f"Test AUC: {metrics['auc']:.4f} | Test F1: {metrics['f1']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
    )

    append_results({
        "model": args.model, "data_type": args.data,
        "auc": f"{metrics['auc']:.4f}", "f1": f"{metrics['f1']:.4f}",
        "precision": f"{metrics['precision']:.4f}", "recall": f"{metrics['recall']:.4f}",
        "threshold": f"{threshold:.2f}",
        "train_samples": len(train_ds), "epochs": final_epoch,
        "time_s": f"{elapsed:.1f}",
    })
    logger.info(f"Results appended to {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# BERT training
# ---------------------------------------------------------------------------
def train_bert(args):
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BertConfig,
        get_linear_schedule_with_warmup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    max_length = args.max_length or 512
    model_name = args.model_name or "bert-base-uncased"
    batch_size = args.llm_batch_size
    patience = args.llm_patience
    lr = args.llm_lr

    data = load_data(args.data)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    cache_dir = _hf_cache_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True,
    )

    def to_texts(x_df):
        return [standard_narrative_prompt(s) for s in x_df["Sequences"].tolist()]

    train_ds = SequenceClassificationDataset(
        to_texts(X_train), y_train["Outcome"].tolist(), tokenizer, max_length
    )
    val_ds = SequenceClassificationDataset(
        to_texts(X_val), y_val["Outcome"].tolist(), tokenizer, max_length
    )
    test_ds = SequenceClassificationDataset(
        to_texts(X_test), y_test["Outcome"].tolist(), tokenizer, max_length
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    if args.tiny:
        dims = {
            "0.2M": [64, 2, 2, 256], "1M": [128, 4, 4, 512],
            "5M": [256, 6, 4, 1024], "10M": [384, 6, 6, 1536],
            "25M": [512, 8, 8, 2048], "50M": [768, 12, 12, 3072],
        }
        hs, nl, nh, inter = dims.get(args.tiny_type, [128, 4, 4, 512])
        cfg = BertConfig(
            vocab_size=len(tokenizer), hidden_size=hs, num_hidden_layers=nl,
            num_attention_heads=nh, intermediate_size=inter,
            max_position_embeddings=512, num_labels=2,
        )
        model = AutoModelForSequenceClassification.from_config(cfg)
    elif args.cold_start:
        cfg = AutoConfig.from_pretrained(model_name, num_labels=2)
        model = AutoModelForSequenceClassification.from_config(cfg)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, cache_dir=cache_dir, trust_remote_code=True,
        )

    if args.peft:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "key", "value"],
            lora_dropout=0.1, bias="none", task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    final_epoch = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.detach().item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                val_loss += outputs.loss.item()
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch["labels"].cpu().numpy())
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        final_epoch = epoch + 1

    elapsed = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            val_preds.extend(probs)
            val_labels.extend(batch["labels"].cpu().numpy())
    threshold, val_f1 = find_optimal_threshold(val_labels, val_preds)
    logger.info(f"Optimal threshold: {threshold:.2f} (Val F1: {val_f1:.4f})")

    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            test_preds.extend(probs)
            test_labels.extend(batch["labels"].cpu().numpy())
    metrics = evaluate_test(test_labels, test_preds, threshold)
    logger.info(
        f"Test AUC: {metrics['auc']:.4f} | Test F1: {metrics['f1']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
    )

    append_results({
        "model": f"bert({model_name})", "data_type": args.data,
        "auc": f"{metrics['auc']:.4f}", "f1": f"{metrics['f1']:.4f}",
        "precision": f"{metrics['precision']:.4f}", "recall": f"{metrics['recall']:.4f}",
        "threshold": f"{threshold:.2f}",
        "train_samples": len(train_ds), "epochs": final_epoch,
        "time_s": f"{elapsed:.1f}",
    })
    logger.info(f"Results appended to {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# Llama training
# ---------------------------------------------------------------------------
def train_llama(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    max_length = args.max_length or 2048
    model_name = args.model_name or "meta-llama/Llama-3.1-8B"
    batch_size = args.llm_batch_size
    patience = args.llm_patience
    lr = args.llm_lr

    cache_dir = _hf_cache_dir()
    hf_token = os.environ.get("HF_TOKEN", None)

    data = load_data(args.data)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, cache_dir=cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def to_texts(x_df):
        return [standard_narrative_prompt(s) for s in x_df["Sequences"].tolist()]

    train_ds = TemporalCausalDataset(
        to_texts(X_train), y_train["Outcome"].tolist(), tokenizer, max_length
    )
    val_ds = TemporalCausalDataset(
        to_texts(X_val), y_val["Outcome"].tolist(), tokenizer, max_length
    )
    test_ds = TemporalCausalDataset(
        to_texts(X_test), y_test["Outcome"].tolist(), tokenizer, max_length
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    if args.tiny:
        dims = {
            "0.1M": [32, 2, 2, 128, 100], "0.2M": [64, 4, 4, 256, 100],
            "1M": [128, 4, 4, 512, 100], "5M": [256, 6, 6, 820, 100],
            "10M": [512, 6, 8, 1024, 2000], "25M": [384, 6, 12, 1024, 4000],
            "50M": [512, 8, 16, 1280, 8000],
        }
        hs, nl, nh, inter, vs = dims.get(args.tiny_type, [128, 4, 4, 512, 100])
        tiny_cfg = LlamaConfig(
            hidden_size=hs, num_hidden_layers=nl, num_attention_heads=nh,
            num_key_value_heads=nh // 2, intermediate_size=inter,
            vocab_size=vs, max_position_embeddings=512,
            rope_theta=10000.0, torch_dtype=torch.bfloat16, tie_word_embeddings=True,
        )
        base_model = AutoModelForCausalLM.from_config(tiny_cfg)
        base_model = base_model.to(torch.bfloat16).to(device)
        base_model.config.pad_token_id = tokenizer.pad_token_id
        model = CausalLMWithClassificationHead(base_model, num_classes=2)
        model.classification_head = model.classification_head.to(torch.bfloat16)
    else:
        bnb_config = None
        if args.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if args.cold_start:
            cfg = AutoConfig.from_pretrained(model_name)
            base_model = AutoModelForCausalLM.from_config(cfg)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
                device_map="auto" if args.use_quantization else None,
                token=hf_token, cache_dir=cache_dir,
                tie_word_embeddings=True, quantization_config=bnb_config,
            )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        model = CausalLMWithClassificationHead(base_model, num_classes=2)

        if args.peft:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
            )
            model.backbone = get_peft_model(model.backbone, lora_cfg)
            model.backbone.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    if not args.use_quantization and not args.tiny:
        model = model.to(device, dtype=torch.bfloat16)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_backbone_state = None
    best_head_state = None
    epochs_no_improve = 0
    final_epoch = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs["loss"] / args.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.detach().item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                val_loss += outputs["loss"].item()
                probs = torch.softmax(outputs["logits"], dim=-1)[:, 1].cpu().float().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch["outcome_labels"].cpu().numpy())
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_backbone_state = {
                k: v.cpu().clone() for k, v in model.backbone.state_dict().items()
            }
            best_head_state = {
                k: v.cpu().clone() for k, v in model.classification_head.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        final_epoch = epoch + 1

    elapsed = time.time() - t0

    if best_backbone_state is not None:
        model.backbone.load_state_dict(best_backbone_state)
        model.classification_head.load_state_dict(best_head_state)

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1].cpu().float().numpy()
            val_preds.extend(probs)
            val_labels.extend(batch["outcome_labels"].cpu().numpy())
    threshold, val_f1 = find_optimal_threshold(val_labels, val_preds)
    logger.info(f"Optimal threshold: {threshold:.2f} (Val F1: {val_f1:.4f})")

    test_preds, test_labels = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1].cpu().float().numpy()
            test_preds.extend(probs)
            test_labels.extend(batch["outcome_labels"].cpu().numpy())
    metrics = evaluate_test(test_labels, test_preds, threshold)
    logger.info(
        f"Test AUC: {metrics['auc']:.4f} | Test F1: {metrics['f1']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
    )

    tag = "tiny" if args.tiny else model_name.replace("/", "_")
    if args.peft:
        tag += "_lora"
    append_results({
        "model": f"llama({tag})", "data_type": args.data,
        "auc": f"{metrics['auc']:.4f}", "f1": f"{metrics['f1']:.4f}",
        "precision": f"{metrics['precision']:.4f}", "recall": f"{metrics['recall']:.4f}",
        "threshold": f"{threshold:.2f}",
        "train_samples": len(train_ds), "epochs": final_epoch,
        "time_s": f"{elapsed:.1f}",
    })
    logger.info(f"Results appended to {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# Bag-of-codes sklearn models (XGBoost / LogReg) -- no GPU required
# ---------------------------------------------------------------------------
def train_sklearn(args):
    import time as _time

    data = load_data(args.data)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    y_tr = y_train["Outcome"].values
    y_va = y_val["Outcome"].values
    y_te = y_test["Outcome"].values

    vocab = sorted({
        tok for seq in X_train["Sequences"].tolist()
        for tok in seq.split(DELIMITER)
    })
    vocab_idx = {v: i for i, v in enumerate(vocab)}
    logger.info(f"BoC vocabulary: {len(vocab)} codes")

    def to_boc(x_df):
        X = np.zeros((len(x_df), len(vocab)), dtype=np.float32)
        for i, seq in enumerate(x_df["Sequences"].tolist()):
            for tok in seq.split(DELIMITER):
                if tok in vocab_idx:
                    X[i, vocab_idx[tok]] = 1
        return X

    X_tr = to_boc(X_train)
    X_va = to_boc(X_val)
    X_te = to_boc(X_test)

    t0 = _time.time()

    if args.model == "xgboost":
        from xgboost import XGBClassifier
        scale_pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pw, random_state=args.seed,
            eval_metric="logloss", verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif args.model == "logreg":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            max_iter=1000, C=1.0, random_state=args.seed, class_weight="balanced",
        )
        model.fit(X_tr, y_tr)
    else:
        raise ValueError(f"Unknown sklearn model: {args.model}")

    elapsed = _time.time() - t0

    preds_va = model.predict_proba(X_va)[:, 1]
    preds_te = model.predict_proba(X_te)[:, 1]

    threshold, val_f1 = find_optimal_threshold(y_va, preds_va)
    logger.info(f"Optimal threshold: {threshold:.2f} (Val F1: {val_f1:.4f})")

    metrics = evaluate_test(y_te, preds_te, threshold)
    logger.info(
        f"Test AUC: {metrics['auc']:.4f} | Test F1: {metrics['f1']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
    )

    append_results({
        "model": f"{args.model}_BoC", "data_type": args.data,
        "auc": f"{metrics['auc']:.4f}", "f1": f"{metrics['f1']:.4f}",
        "precision": f"{metrics['precision']:.4f}", "recall": f"{metrics['recall']:.4f}",
        "threshold": f"{threshold:.2f}",
        "train_samples": len(X_tr), "epochs": "-",
        "time_s": f"{elapsed:.1f}",
    })
    logger.info(f"Results appended to {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    logger.info(f"Model: {args.model}, Data: {args.data}")

    if args.model in ("xgboost", "logreg"):
        train_sklearn(args)
    elif args.model in ("transformer", "lstm", "bilstm"):
        assert HAS_TORCH, "PyTorch is required for DL models"
        train_dl(args)
    elif args.model == "bert":
        assert HAS_TORCH, "PyTorch is required for BERT"
        train_bert(args)
    elif args.model == "llama":
        assert HAS_TORCH, "PyTorch is required for Llama"
        train_llama(args)


if __name__ == "__main__":
    main()
