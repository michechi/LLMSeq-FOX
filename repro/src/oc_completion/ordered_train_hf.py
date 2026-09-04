"""Pretrained LoRA training for the Ordered Compliance hole audit.

The runner consumes the immutable parquet splits written by
``src.oc_completion.ordered_data`` and trains only on complete 20-letter
sequences with ordinary binary labels.  Hole candidates and structural
metadata never enter the training path.

Two arms are intentionally supported:

``bert_lora``
    ``bert-base-uncased`` with LoRA on query/key/value.

``llama_lora``
    ``meta-llama/Llama-3.2-1B`` with LoRA on q/k/v/o projections and the
    repository's two-layer MLP classification head applied to the final
    non-padding hidden state.  Training is classification-only cross entropy;
    no next-token labels, loss, or logits are used.

Checkpoints contain only trainable adapter/classification-head tensors plus
resume state.  The frozen pretrained backbone is always reconstructed from
the recorded model identifier.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.oc_completion.oracle import ALPHABET, N_EVENTS, SEP
from src.oc_completion.ordered_data import (
    DEFAULT_DATA_ROOT,
    NOISE_LEVELS,
    load_split,
    pi_tag,
)


REPO_ROOT = Path(os.environ.get("LLMSEQ_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_CHECKPOINT_ROOT = Path(
    os.environ.get(
        "OC_CHECKPOINT_ROOT",
        REPO_ROOT / "results" / "oc_hole_audit" / "checkpoints",
    )
)
DEFAULT_HF_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or os.environ.get("HF_HOME")
    or (REPO_ROOT / ".cache" / "huggingface")
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "bert_lora": {
        "kind": "bert",
        "model_name": "bert-base-uncased",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "lora_targets": ("query", "key", "value"),
        "lora_task_type": "SEQ_CLS",
    },
    "llama_lora": {
        "kind": "llama",
        "model_name": "meta-llama/Llama-3.2-1B",
        "micro_batch": 8,
        "gradient_accumulation": 2,
        "lora_targets": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "lora_task_type": "FEATURE_EXTRACTION",
    },
}

LEARNING_RATE = 2e-5
EFFECTIVE_BATCH_SIZE = 16
MAX_EPOCHS = 20
PATIENCE = 3
WARMUP_RATIO = 0.06
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TOKEN_LENGTH_SHORT = 64
TOKEN_LENGTH_LONG = 200


def atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_model_seed(seed: int) -> None:
    """Seed only model-side stochasticity; data/noise seeds live in manifests."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sequence_letters(sequence: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(sequence, str):
        if SEP in sequence:
            letters = tuple(sequence.split(SEP))
        elif " " in sequence:
            letters = tuple(sequence.split())
        else:
            letters = tuple(sequence)
    else:
        letters = tuple(sequence)
    if len(letters) != N_EVENTS or any(letter not in ALPHABET for letter in letters):
        raise ValueError("expected one complete 20-letter A-Z sequence")
    return letters


def build_prompt(kind: str, sequence: str | Sequence[str]) -> str:
    """Create the complete-sequence BERT or exact causal prompt."""
    events = " ".join(_sequence_letters(sequence))
    if kind == "bert":
        return f"Sequential events: {events}"
    if kind == "llama":
        return f"Sequential events: {events}\nOutcome (0 or 1):"
    raise ValueError(f"unsupported prompt kind {kind}")


def replacement_probe_sequences() -> Iterable[str]:
    """All 20 positions by all 26 letters in a neutral complete sequence."""
    base = ["A"] * N_EVENTS
    for position in range(N_EVENTS):
        for letter in ALPHABET:
            candidate = list(base)
            candidate[position] = letter
            yield SEP.join(candidate)


def _normalise_batch_ids(encoded: Any) -> list[list[int]]:
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (int, np.integer)):
        ids = [ids]
    return [[int(token) for token in row] for row in ids]


def _audit_prompt_batches(
    tokenizer: Any,
    prompts: Iterable[str],
    batch_size: int,
) -> dict[str, int]:
    maximum = 0
    count = 0
    unknown_tokens = 0
    prompts_with_unknown = 0
    buffer: list[str] = []
    unknown_id = getattr(tokenizer, "unk_token_id", None)

    def consume(texts: list[str]) -> None:
        nonlocal maximum, count, unknown_tokens, prompts_with_unknown
        if not texts:
            return
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        rows = _normalise_batch_ids(encoded)
        if len(rows) != len(texts):
            raise AssertionError("tokenizer returned a different batch size")
        count += len(rows)
        for row in rows:
            maximum = max(maximum, len(row))
            if unknown_id is not None:
                n_unknown = sum(token == int(unknown_id) for token in row)
                unknown_tokens += n_unknown
                prompts_with_unknown += int(n_unknown > 0)

    for prompt in prompts:
        buffer.append(prompt)
        if len(buffer) >= batch_size:
            consume(buffer)
            buffer = []
    consume(buffer)
    return {
        "prompt_count": count,
        "maximum_untruncated_token_length": maximum,
        "unknown_token_count": unknown_tokens,
        "prompts_with_unknown_tokens": prompts_with_unknown,
    }


def audit_tokenization(
    tokenizer: Any,
    kind: str,
    split_sequences: Mapping[str, Sequence[str]],
    model_name: str,
    data_fingerprint: str,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Audit every actual prompt and all 520 position/letter probes untruncated."""
    split_stats: dict[str, dict[str, int]] = {}
    maximum = 0
    unknown_tokens = 0
    prompts_with_unknown = 0
    actual_count = 0
    for split in ("train", "val", "test"):
        stats = _audit_prompt_batches(
            tokenizer,
            (build_prompt(kind, sequence) for sequence in split_sequences[split]),
            batch_size,
        )
        split_stats[split] = stats
        maximum = max(maximum, stats["maximum_untruncated_token_length"])
        unknown_tokens += stats["unknown_token_count"]
        prompts_with_unknown += stats["prompts_with_unknown_tokens"]
        actual_count += stats["prompt_count"]

    probe_stats = _audit_prompt_batches(
        tokenizer,
        (build_prompt(kind, sequence) for sequence in replacement_probe_sequences()),
        batch_size,
    )
    maximum = max(maximum, probe_stats["maximum_untruncated_token_length"])
    unknown_tokens += probe_stats["unknown_token_count"]
    prompts_with_unknown += probe_stats["prompts_with_unknown_tokens"]
    if maximum <= TOKEN_LENGTH_SHORT:
        selected_max_length = TOKEN_LENGTH_SHORT
    elif maximum <= TOKEN_LENGTH_LONG:
        selected_max_length = TOKEN_LENGTH_LONG
    else:
        raise ValueError(
            f"untruncated prompt length {maximum} exceeds allowed maximum "
            f"{TOKEN_LENGTH_LONG}"
        )

    token_ids: dict[str, list[int]] = {}
    for letter in ALPHABET:
        encoded = tokenizer(
            letter,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        token_ids[letter] = _normalise_batch_ids(encoded)[0]

    return {
        "schema_version": 1,
        "arm": f"{kind}_lora",
        "kind": kind,
        "model_name": model_name,
        "data_fingerprint": data_fingerprint,
        "audit_used_truncation": False,
        "actual_prompt_count": actual_count,
        "replacement_probe_count": probe_stats["prompt_count"],
        "total_prompt_count": actual_count + probe_stats["prompt_count"],
        "maximum_untruncated_token_length": maximum,
        "would_truncate_at_64": int(maximum > TOKEN_LENGTH_SHORT),
        "selected_max_length": selected_max_length,
        "number_truncated_examples": 0,
        "unknown_token_count": unknown_tokens,
        "prompts_with_unknown_tokens": prompts_with_unknown,
        "token_ids_A_Z": token_ids,
        "split_stats": split_stats,
        "replacement_probe_stats": probe_stats,
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token": getattr(tokenizer, "unk_token", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
    }


def build_tokenizer(
    arm: str,
    model_name: str,
    cache_dir: Path,
    tokenizer_path: Path | None = None,
) -> Any:
    from transformers import AutoTokenizer

    source = (
        str(tokenizer_path)
        if tokenizer_path is not None and tokenizer_path.exists()
        else model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=str(cache_dir))
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"{model_name} tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class LlamaSequenceClassifier(nn.Module):
    """LoRA backbone plus the current MLP head on final non-padding state."""

    def __init__(self, backbone: nn.Module, num_labels: int = 2):
        super().__init__()
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        last_non_padding = attention_mask.to(torch.long).sum(dim=1) - 1
        last_non_padding = last_non_padding.clamp_min(0)
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        representations = hidden[batch_index, last_non_padding]
        return self.classification_head(representations)


def build_hf_model(
    arm: str,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    cache_dir: Path,
    model_name: str | None = None,
) -> nn.Module:
    """Reconstruct a frozen pretrained backbone with the exact LoRA recipe."""
    from peft import LoraConfig, get_peft_model

    spec = MODEL_SPECS[arm]
    resolved_name = model_name or spec["model_name"]
    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(spec["lora_targets"]),
        bias="none",
        task_type=spec["lora_task_type"],
    )
    if spec["kind"] == "bert":
        from transformers import AutoModelForSequenceClassification

        backbone = AutoModelForSequenceClassification.from_pretrained(
            resolved_name,
            num_labels=2,
            cache_dir=str(cache_dir),
        )
        model = get_peft_model(backbone, lora)
    else:
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(
            resolved_name,
            torch_dtype=dtype,
            cache_dir=str(cache_dir),
        )
        backbone.config.pad_token_id = tokenizer.pad_token_id
        adapted = get_peft_model(backbone, lora)
        model = LlamaSequenceClassifier(adapted)
        model.classification_head.to(dtype=dtype)
    return model.to(device)


def model_logits(
    arm: str,
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if MODEL_SPECS[arm]["kind"] == "bert":
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    return model(input_ids=input_ids, attention_mask=attention_mask)


class PromptDataset(Dataset):
    """Complete prompts and binary classification labels only."""

    def __init__(
        self,
        arm: str,
        sequences: Sequence[str],
        observed_labels: Sequence[int],
        latent_labels: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ):
        self.kind = MODEL_SPECS[arm]["kind"]
        self.sequences = list(sequences)
        self.observed_labels = np.asarray(observed_labels, dtype=np.int64)
        self.latent_labels = np.asarray(latent_labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        if not (
            len(self.sequences)
            == len(self.observed_labels)
            == len(self.latent_labels)
        ):
            raise ValueError("sequences and labels have different lengths")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            build_prompt(self.kind, self.sequences[index]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.observed_labels[index], dtype=torch.long),
            "Y_star": torch.tensor(self.latent_labels[index], dtype=torch.long),
        }


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else 0.5


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_model(
    arm: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    count = 0
    all_logits: list[np.ndarray] = []
    all_observed: list[np.ndarray] = []
    all_latent: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast_context(device):
            logits = model_logits(arm, model, input_ids, attention_mask)
            loss = nn.functional.cross_entropy(logits.float(), labels)
        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        count += batch_size
        all_logits.append(logits.float().cpu().numpy())
        all_observed.append(batch["labels"].numpy())
        all_latent.append(batch["Y_star"].numpy())
    logits_np = np.concatenate(all_logits)
    observed = np.concatenate(all_observed)
    latent = np.concatenate(all_latent)
    probabilities = torch.softmax(torch.from_numpy(logits_np), dim=1)[:, 1].numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "loss": total_loss / count,
        "observed_auc": _safe_auc(observed, probabilities),
        "latent_auc": _safe_auc(latent, probabilities),
        "f1_at_0p5": float(f1_score(observed, predictions, zero_division=0)),
        "logits": logits_np,
        "probabilities": probabilities,
        "observed_labels": observed,
        "latent_labels": latent,
    }


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 181)
    scores = np.asarray(
        [f1_score(labels, probabilities >= threshold, zero_division=0) for threshold in candidates]
    )
    best_score = float(scores.max())
    tied = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1e-12))
    best_index = int(tied[np.argmin(np.abs(candidates[tied] - 0.5))])
    return float(candidates[best_index]), best_score


class EarlyStopping:
    def __init__(self, patience: int = PATIENCE):
        if patience <= 0:
            raise ValueError("patience must be positive")
        self.patience = int(patience)
        self.best_loss = float("inf")
        self.counter = 0

    def update(self, validation_loss: float, epsilon=0.00001) -> bool:
        if not math.isfinite(validation_loss):
            raise ValueError("validation loss must be finite")
        if (validation_loss+epsilon) < self.best_loss:
            self.best_loss = float(validation_loss)
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


def linear_warmup_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Transformers-compatible linear warmup followed by linear decay."""
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("invalid warmup/total step counts")

    def multiplier(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract adapter/head tensors without any frozen backbone tensor."""
    state = model.state_dict()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing = trainable_names - set(state)
    if missing:
        raise KeyError(f"trainable parameters absent from state_dict: {sorted(missing)[:3]}")
    return {name: state[name].detach().cpu().clone() for name in sorted(trainable_names)}


def load_trainable_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    current = model.state_dict()
    unexpected = set(state) - set(current)
    if unexpected:
        raise KeyError(
            f"checkpoint contains unexpected trainable tensors: {sorted(unexpected)[:3]}"
        )
    current.update({name: tensor for name, tensor in state.items()})
    model.load_state_dict(current, strict=True)


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": loader_generator.get_state(),
    }


def restore_rng_state(state: Mapping[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    loader_generator.set_state(state["loader_generator"])


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    loader_generator: torch.Generator,
    arm: str,
    pi: float,
    model_seed: int,
    epoch: int,
    best_validation_loss: float,
    best_epoch: int,
    patience_counter: int,
    max_length: int,
    history: Sequence[Mapping[str, Any]],
    reload_probe: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_format": "lora_adapter_and_classification_head_only",
        "adapter_and_head_state_dict": trainable_state_dict(model),
        "trainable_parameter_names": sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "gradient_scaler": (
            scaler.state_dict() if scaler is not None and scaler.is_enabled() else None
        ),
        "rng_states": capture_rng_state(loader_generator),
        "current_epoch": int(epoch),
        "best_validation_metric": float(best_validation_loss),
        "best_validation_loss": float(best_validation_loss),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
        "history": [dict(record) for record in history],
        "arm": arm,
        "noise_pi": float(pi),
        "model_seed": int(model_seed),
        "max_length": int(max_length),
        "selection_criterion": "standard observed-label validation loss",
        "reload_probe": dict(reload_probe),
    }


class HFScorer:
    """Evaluation-mode complete-sequence scorer returning FP32 logits."""

    def __init__(
        self,
        arm: str,
        model: nn.Module,
        tokenizer: Any,
        max_length: int,
        device: torch.device,
        batch_size: int = 64,
    ):
        self.arm = arm
        self.kind = MODEL_SPECS[arm]["kind"]
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.device = device
        self.batch_size = int(batch_size)

    @torch.no_grad()
    def __call__(self, sequences: Sequence[str]) -> np.ndarray:
        self.model.eval()
        outputs: list[np.ndarray] = []
        for start in range(0, len(sequences), self.batch_size):
            prompts = [
                build_prompt(self.kind, sequence)
                for sequence in sequences[start : start + self.batch_size]
            ]
            encoded = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            with autocast_context(self.device):
                logits = model_logits(self.arm, self.model, input_ids, attention_mask)
            outputs.append(logits.float().cpu().numpy())
        if not outputs:
            return np.empty((0, 2), dtype=np.float32)
        return np.concatenate(outputs).astype(np.float32, copy=False)

    def score_sequences(
        self, sequences: Sequence[str], batch_size: int | None = None
    ) -> dict[str, np.ndarray]:
        previous = self.batch_size
        if batch_size is not None:
            self.batch_size = int(batch_size)
        try:
            logits = self(sequences)
        finally:
            self.batch_size = previous
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
        return {
            "logits": logits,
            "positive_logit": logits[:, 1],
            "positive_probability": probabilities.astype(np.float32, copy=False),
        }


def _reload_probe(scorer: HFScorer, sequence: str) -> dict[str, Any]:
    logits = scorer([sequence])[0]
    return {"sequence": sequence, "classification_logits_fp32": logits.tolist()}


def _verify_reload_probe(scorer: HFScorer, probe: Mapping[str, Any]) -> float:
    expected = np.asarray(probe["classification_logits_fp32"], dtype=np.float32)
    actual = scorer([probe["sequence"]])[0]
    difference = float(np.max(np.abs(actual - expected)))
    tolerance = 5e-3 if scorer.device.type == "cuda" else 1e-5
    if difference > tolerance:
        raise AssertionError(
            f"checkpoint reload changed logits by {difference}, tolerance={tolerance}"
        )
    return difference


def _checkpoint_path(checkpoint_or_run_dir: Path | str) -> tuple[Path, Path]:
    candidate = Path(checkpoint_or_run_dir)
    if candidate.is_dir():
        return candidate / "best_standard.pt", candidate
    run_dir = candidate.parent.parent if candidate.parent.name == "epochs" else candidate.parent
    return candidate, run_dir


def load_hf_scorer(
    checkpoint_or_run_dir: Path | str,
    device: str | torch.device | None = None,
    batch_size: int = 64,
    verify_reload: bool = True,
) -> tuple[HFScorer, dict[str, Any]]:
    """Rebuild the frozen backbone and load an adapter/head-only checkpoint."""
    checkpoint_path, run_dir = _checkpoint_path(checkpoint_or_run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arm = checkpoint["arm"]
    if device is None or str(device) == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    dtype = torch.bfloat16 if resolved_device.type == "cuda" else torch.float32
    cache_dir = Path(config.get("hf_cache", DEFAULT_HF_CACHE))
    tokenizer = build_tokenizer(
        arm,
        config["model_name"],
        cache_dir,
        tokenizer_path=run_dir / "tokenizer",
    )
    model = build_hf_model(
        arm,
        tokenizer,
        resolved_device,
        dtype,
        cache_dir,
        model_name=config["model_name"],
    )
    load_trainable_state_dict(model, checkpoint["adapter_and_head_state_dict"])
    scorer = HFScorer(
        arm,
        model,
        tokenizer,
        checkpoint["max_length"],
        resolved_device,
        batch_size,
    )
    reload_difference = None
    if verify_reload and checkpoint.get("reload_probe"):
        reload_difference = _verify_reload_probe(scorer, checkpoint["reload_probe"])
    done_path = run_dir / "done.json"
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    metadata = {
        "model": arm,
        "arm": arm,
        "model_name": config["model_name"],
        "noise_pi": checkpoint["noise_pi"],
        "model_seed": checkpoint["model_seed"],
        "checkpoint_epoch": checkpoint["current_epoch"],
        "max_length": checkpoint["max_length"],
        "validation_selected_threshold": done.get("validation_selected_threshold"),
        "reload_max_abs_difference": reload_difference,
    }
    return scorer, metadata


def _data_fingerprint(data_root: Path) -> str:
    manifest = data_root / "dataset_manifest.json"
    if manifest.exists():
        return sha256_file(manifest)
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        for family in ("splits", "noise"):
            path = data_root / family / f"{split}.parquet"
            digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _data_manifest_metadata(data_root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for filename in ("dataset_manifest.json", "noise_manifest.json"):
        path = data_root / filename
        if path.exists():
            metadata[filename.removesuffix(".json")] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return metadata


def _valid_done(run_dir: Path) -> bool:
    required = (
        "best_standard.pt",
        "last.pt",
        "config.json",
        "history.csv",
        "done.json",
        "tokenization_audit.json",
    )
    if any(not (run_dir / name).is_file() for name in required):
        return False
    try:
        done = json.loads((run_dir / "done.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        done.get("status") == "complete"
        and done.get("selected_checkpoint") == "best_standard.pt"
    )


def _load_or_create_audit(
    run_dir: Path,
    tokenizer: Any,
    arm: str,
    model_name: str,
    split_frames: Mapping[str, pd.DataFrame],
    data_fingerprint: str,
    audit_batch_size: int,
    resume: bool,
) -> dict[str, Any]:
    path = run_dir / "tokenization_audit.json"
    if resume and path.exists():
        audit = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "arm": arm,
            "model_name": model_name,
            "data_fingerprint": data_fingerprint,
        }
        if all(audit.get(key) == value for key, value in expected.items()):
            if audit.get("number_truncated_examples") != 0:
                raise ValueError("stored tokenization audit contains truncation")
            return audit
    audit = audit_tokenization(
        tokenizer,
        MODEL_SPECS[arm]["kind"],
        {split: split_frames[split]["X"].astype(str).tolist() for split in split_frames},
        model_name,
        data_fingerprint,
        batch_size=audit_batch_size,
    )
    audit["arm"] = arm
    atomic_json_dump(audit, path)
    return audit


def _hardware(device: torch.device) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _history_write(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(history).to_csv(temporary, index=False)
    temporary.replace(path)


def _make_loaders(
    arm: str,
    frames: Mapping[str, pd.DataFrame],
    tokenizer: Any,
    max_length: int,
    micro_batch: int,
    eval_batch: int,
    workers: int,
    model_seed: int,
    device: torch.device,
) -> tuple[dict[str, DataLoader], torch.Generator]:
    datasets = {
        split: PromptDataset(
            arm,
            frame["X"].astype(str).tolist(),
            frame["Y_observed"].to_numpy(),
            frame["Y_star"].to_numpy(),
            tokenizer,
            max_length,
        )
        for split, frame in frames.items()
    }
    generator = torch.Generator().manual_seed(model_seed)
    common = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=micro_batch,
            shuffle=True,
            generator=generator,
            **common,
        ),
        "val": DataLoader(datasets["val"], batch_size=eval_batch, shuffle=False, **common),
        "test": DataLoader(datasets["test"], batch_size=eval_batch, shuffle=False, **common),
    }
    return loaders, generator


def train(args: argparse.Namespace) -> dict[str, Any]:
    if float(args.pi) not in NOISE_LEVELS:
        raise ValueError(f"pi must be one of {NOISE_LEVELS}")
    spec = MODEL_SPECS[args.arm]
    micro_batch = args.micro_batch or int(spec["micro_batch"])
    accumulation = args.gradient_accumulation or int(spec["gradient_accumulation"])
    if micro_batch * accumulation != EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            f"micro_batch * gradient_accumulation must equal {EFFECTIVE_BATCH_SIZE}"
        )
    suffix = "_smoke" if args.smoke else ""
    run_dir = args.run_dir or (
        args.checkpoint_root
        / args.arm
        / f"pi_{pi_tag(args.pi)}"
        / f"seed_{args.model_seed}{suffix}"
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "epochs").mkdir(exist_ok=True)
    if args.resume and _valid_done(run_dir):
        print(f"[ordered_train_hf] valid done.json found; skipping {run_dir}", flush=True)
        return json.loads((run_dir / "done.json").read_text(encoding="utf-8"))

    set_model_seed(args.model_seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    frames = {
        split: load_split(args.data_root, split, args.pi)
        for split in ("train", "val", "test")
    }
    audit_frames = {split: frame for split, frame in frames.items()}
    data_fingerprint = _data_fingerprint(args.data_root)
    data_metadata = _data_manifest_metadata(args.data_root)

    tokenizer = build_tokenizer(args.arm, spec["model_name"], args.hf_cache)
    audit = _load_or_create_audit(
        run_dir,
        tokenizer,
        args.arm,
        spec["model_name"],
        audit_frames,
        data_fingerprint,
        args.audit_batch_size,
        args.resume,
    )
    max_length = int(audit["selected_max_length"])
    if max_length not in (TOKEN_LENGTH_SHORT, TOKEN_LENGTH_LONG):
        raise ValueError(f"invalid audited max length {max_length}")
    tokenizer.save_pretrained(run_dir / "tokenizer")
    if args.max_train_rows:
        frames["train"] = frames["train"].iloc[: args.max_train_rows].reset_index(drop=True)

    model = build_hf_model(
        args.arm,
        tokenizer,
        device,
        dtype,
        args.hf_cache,
        model_name=spec["model_name"],
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise AssertionError("LoRA model has no trainable parameters")
    loaders, loader_generator = _make_loaders(
        args.arm,
        frames,
        tokenizer,
        max_length,
        micro_batch,
        args.eval_batch,
        args.workers,
        args.model_seed,
        device,
    )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=LEARNING_RATE)
    updates_per_epoch = math.ceil(len(loaders["train"]) / accumulation)
    total_updates = updates_per_epoch * args.max_epochs
    warmup_updates = int(WARMUP_RATIO * total_updates)
    scheduler = linear_warmup_decay_scheduler(optimizer, warmup_updates, total_updates)
    # BF16 does not require loss scaling.  Keep the explicit field in every
    # resume checkpoint so FP16 variants cannot silently omit scaler state.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    stopper = EarlyStopping(args.patience)
    history: list[dict[str, Any]] = []
    best_epoch = 0
    start_epoch = 1
    prior_training_seconds = 0.0

    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        expected = {
            "arm": args.arm,
            "noise_pi": float(args.pi),
            "model_seed": int(args.model_seed),
            "max_length": max_length,
            "maximum_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "micro_batch_size": int(micro_batch),
            "gradient_accumulation": int(accumulation),
            "data_fingerprint": data_fingerprint,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(
                    f"resume mismatch for {key}: {checkpoint.get(key)!r} != {value!r}"
                )
        load_trainable_state_dict(model, checkpoint["adapter_and_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("gradient_scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["gradient_scaler"])
        restore_rng_state(checkpoint["rng_states"], loader_generator)
        start_epoch = int(checkpoint["current_epoch"]) + 1
        stopper.best_loss = float(checkpoint["best_validation_loss"])
        stopper.counter = int(checkpoint["patience_counter"])
        best_epoch = int(checkpoint["best_epoch"])
        history = [dict(record) for record in checkpoint.get("history", [])]
        prior_training_seconds = float(checkpoint.get("training_seconds", 0.0))

    configuration = {
        "schema_version": 1,
        "arm": args.arm,
        "kind": spec["kind"],
        "model_name": spec["model_name"],
        "noise_pi": float(args.pi),
        "model_seed": int(args.model_seed),
        "data_root": str(args.data_root),
        "data_fingerprint": data_fingerprint,
        "data_seed": data_metadata.get("dataset_manifest", {}).get("data_seed"),
        "noise_seed": data_metadata.get("noise_manifest", {}).get("noise_seed"),
        "mechanism": data_metadata.get("dataset_manifest", {}).get("mechanism"),
        "hf_cache": str(args.hf_cache),
        "prompt": (
            "Sequential events: x1 ... x20"
            if spec["kind"] == "bert"
            else "Sequential events: x1 ... x20\\nOutcome (0 or 1):"
        ),
        "training_objective": "binary classification cross entropy only",
        "uses_next_token_loss": False,
        "llama_pooling": "final non-padding hidden state",
        "llama_head": "Linear(hidden,hidden/2)-Tanh-Dropout(0.1)-Linear(hidden/2,2)",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "micro_batch_size": micro_batch,
        "gradient_accumulation": accumulation,
        "maximum_epochs": args.max_epochs,
        "early_stopping_patience": args.patience,
        "early_stopping_criterion": "standard observed-label validation loss",
        "warmup_ratio": WARMUP_RATIO,
        "warmup_updates": warmup_updates,
        "total_updates": total_updates,
        "scheduler": "linear decay",
        "lora": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "targets": list(spec["lora_targets"]),
            "bias": "none",
        },
        "max_length": max_length,
        "tokenization_audit": "tokenization_audit.json",
        "dtype": str(dtype),
        "hardware": _hardware(device),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "checkpoint_state": "LoRA adapter and classification head only",
        "smoke": bool(args.smoke),
    }
    atomic_json_dump(configuration, run_dir / "config.json")

    scorer = HFScorer(
        args.arm,
        model,
        tokenizer,
        max_length,
        device,
        batch_size=args.eval_batch,
    )
    probe_sequence = frames["val"].iloc[0]["X"]
    started = time.time()
    if stopper.should_stop:
        start_epoch = args.max_epochs + 1

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        examples = 0
        for batch_index, batch in enumerate(loaders["train"], start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast_context(device):
                logits = model_logits(args.arm, model, input_ids, attention_mask)
                loss = nn.functional.cross_entropy(logits.float(), labels)
            (loss / accumulation).backward()
            if batch_index % accumulation == 0 or batch_index == len(loaders["train"]):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            examples += batch_size

        validation = evaluate_model(args.arm, model, loaders["val"], device)
        improved = stopper.update(validation["loss"])
        if improved:
            best_epoch = epoch
        elapsed = prior_training_seconds + time.time() - started
        record = {
            "epoch": epoch,
            "train_loss": total_loss / examples,
            "standard_validation_loss": validation["loss"],
            "standard_observed_validation_auc": validation["observed_auc"],
            "standard_latent_validation_auc": validation["latent_auc"],
            "standard_validation_f1": validation["f1_at_0p5"],
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "best_epoch_so_far": best_epoch,
            "patience_counter": stopper.counter,
            "training_seconds": elapsed,
        }
        history.append(record)
        _history_write(history, run_dir / "history.csv")
        probe = _reload_probe(scorer, probe_sequence)
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            loader_generator,
            args.arm,
            args.pi,
            args.model_seed,
            epoch,
            stopper.best_loss,
            best_epoch,
            stopper.counter,
            max_length,
            history,
            probe,
        )
        payload["training_seconds"] = elapsed
        payload["maximum_epochs"] = int(args.max_epochs)
        payload["patience"] = int(args.patience)
        payload["micro_batch_size"] = int(micro_batch)
        payload["gradient_accumulation"] = int(accumulation)
        payload["model_name"] = spec["model_name"]
        payload["data_fingerprint"] = data_fingerprint
        payload["lora"] = dict(configuration["lora"])
        payload["tokenizer_metadata"] = {
            "path": "tokenizer",
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "padding_side": tokenizer.padding_side,
        }
        # Commit a newly selected standard checkpoint before advancing the
        # resume pointer.  A pre-emption between these writes can only cause
        # the epoch to be repeated; it cannot lose the selected weights.
        if improved:
            atomic_torch_save(payload, run_dir / "best_standard.pt")
        atomic_torch_save(payload, last_path)
        atomic_torch_save(
            {
                "schema_version": 1,
                "state_format": "lora_adapter_and_classification_head_only",
                "adapter_and_head_state_dict": payload["adapter_and_head_state_dict"],
                "trainable_parameter_names": payload["trainable_parameter_names"],
                "arm": args.arm,
                "noise_pi": float(args.pi),
                "model_seed": int(args.model_seed),
                "current_epoch": epoch,
                "max_length": max_length,
                "model_name": spec["model_name"],
                "lora": dict(configuration["lora"]),
                "tokenizer_metadata": payload["tokenizer_metadata"],
                "validation": {
                    key: value
                    for key, value in validation.items()
                    if not isinstance(value, np.ndarray)
                },
                "reload_probe": probe,
            },
            run_dir / "epochs" / f"epoch_{epoch:03d}_eval.pt",
        )
        print(
            f"[ordered_train_hf] {args.arm} pi={args.pi:.1f} seed={args.model_seed} "
            f"epoch={epoch} train={record['train_loss']:.5f} "
            f"val={validation['loss']:.5f} auc={validation['observed_auc']:.4f}",
            flush=True,
        )
        if stopper.should_stop:
            break

    best_path = run_dir / "best_standard.pt"
    if not best_path.exists():
        raise RuntimeError("training produced no best_standard.pt")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    load_trainable_state_dict(model, best["adapter_and_head_state_dict"])
    reload_difference = _verify_reload_probe(scorer, best["reload_probe"])
    validation = evaluate_model(args.arm, model, loaders["val"], device)
    threshold, validation_f1 = select_f1_threshold(
        validation["observed_labels"], validation["probabilities"]
    )
    test = evaluate_model(args.arm, model, loaders["test"], device)
    test_predictions = (test["probabilities"] >= threshold).astype(np.uint8)
    training_seconds = prior_training_seconds + time.time() - started
    done = {
        "status": "complete",
        "arm": args.arm,
        "model_name": spec["model_name"],
        "noise_pi": float(args.pi),
        "model_seed": int(args.model_seed),
        "selected_checkpoint": "best_standard.pt",
        "selection_criterion": "standard observed-label validation loss",
        "best_epoch": int(best["best_epoch"]),
        "epochs_completed": int(history[-1]["epoch"]),
        "validation_selected_threshold": threshold,
        "validation_f1_at_selected_threshold": validation_f1,
        "standard_validation_loss": validation["loss"],
        "standard_observed_validation_auc": validation["observed_auc"],
        "standard_latent_validation_auc": validation["latent_auc"],
        "observed_label_test_auc": test["observed_auc"],
        "latent_label_test_auc": test["latent_auc"],
        "observed_label_f1": float(
            f1_score(test["observed_labels"], test_predictions, zero_division=0)
        ),
        "observed_label_precision": float(
            precision_score(test["observed_labels"], test_predictions, zero_division=0)
        ),
        "observed_label_recall": float(
            recall_score(test["observed_labels"], test_predictions, zero_division=0)
        ),
        "training_seconds": training_seconds,
        "hardware": _hardware(device),
        "max_length": max_length,
        "tokenization_audit": "tokenization_audit.json",
        "checkpoint_reload_max_abs_difference": reload_difference,
        "checkpoint_paths": {
            "best_standard": str(best_path),
            "last": str(last_path),
            "epochs": str(run_dir / "epochs"),
        },
    }
    atomic_json_dump(done, run_dir / "done.json")
    return done


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--pi", type=float, choices=NOISE_LEVELS, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--audit-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    if args.max_epochs <= 0 or args.patience <= 0:
        parser.error("max epochs and patience must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
