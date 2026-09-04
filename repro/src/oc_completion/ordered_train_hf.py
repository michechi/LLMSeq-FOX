"""Generic pretrained-model training for the Ordered Compliance hole audit.

The runner consumes the immutable parquet splits written by
``src.oc_completion.ordered_data`` and trains only on complete 20-letter
sequences with ordinary binary labels.  Hole candidates and structural
metadata never enter the training path.

The historical ``bert_lora`` and ``llama_lora`` arms remain supported with
their original defaults and directory layout.  A model can also be selected
directly with ``--model-name``/``--model-kind`` and trained either with LoRA,
full fine-tuning, or 4/8-bit QLoRA.

The two historical arms are:

``bert_lora``
    ``bert-base-uncased`` with LoRA on query/key/value.

``llama_lora``
    ``meta-llama/Llama-3.2-1B`` with LoRA on q/k/v/o projections and the
    repository's two-layer MLP classification head applied to the final
    non-padding hidden state.  Training is classification-only cross entropy;
    no next-token labels, loss, or logits are used.

Checkpoints contain only trainable tensors plus resume state.  Thus PEFT
checkpoints never materialise or serialize the frozen pretrained backbone;
it is reconstructed from the complete model recipe recorded alongside every
checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
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

DTYPE_NAMES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _model_kind_to_prompt_kind(model_kind: str) -> str:
    aliases = {
        "bert": "bert",
        "encoder": "bert",
        "llama": "llama",
        "decoder": "llama",
    }
    try:
        return aliases[model_kind]
    except KeyError as error:
        raise ValueError(f"unsupported model kind {model_kind!r}") from error


def _arm_model_kind(arm: str) -> str:
    kind = MODEL_SPECS[arm]["kind"]
    return "encoder" if kind == "bert" else "decoder"


def _parse_lora_targets(value: str | Sequence[str] | None, model_kind: str) -> list[str]:
    if value is None:
        return (
            ["query", "key", "value"]
            if model_kind == "encoder"
            else ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
    values = value.split(",") if isinstance(value, str) else list(value)
    targets = [str(target).strip() for target in values if str(target).strip()]
    if not targets:
        raise ValueError("LoRA target list must not be empty")
    if len(set(targets)) != len(targets):
        raise ValueError("LoRA target list contains duplicates")
    return targets


def _safe_model_tag(
    model_name: str,
    peft: bool,
    quantization: str,
    identity: Mapping[str, Any] | None = None,
) -> str:
    base = model_name.rstrip("/").rsplit("/", 1)[-1]
    safe = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in base
    )
    safe = (safe.strip(".-_") or "model")[:48]
    tuning = "lora" if peft else "full"
    encoded = json.dumps(identity or {}, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()[:10]
    return f"{safe}_{tuning}_{quantization}_{fingerprint}"


def _resolve_dtype_name(value: str, device: torch.device) -> str:
    if value != "auto":
        return value
    return "bf16" if device.type == "cuda" else "fp32"


def _arm_recipe_is_overridden(args: argparse.Namespace) -> bool:
    """Whether an arm invocation changes more than its output tag."""
    return any(
        (
            getattr(args, "model_name", None) is not None,
            getattr(args, "model_revision", None) is not None,
            getattr(args, "tokenizer_name", None) is not None,
            getattr(args, "tokenizer_revision", None) is not None,
            getattr(args, "peft", True) is not True,
            getattr(args, "quantization", "none") != "none",
            getattr(args, "dtype", "auto") != "auto",
            getattr(args, "compute_dtype", "auto") != "auto",
            bool(getattr(args, "gradient_checkpointing", False)),
            getattr(args, "lora_r", LORA_RANK) != LORA_RANK,
            getattr(args, "lora_alpha", LORA_ALPHA) != LORA_ALPHA,
            getattr(args, "lora_dropout", LORA_DROPOUT) != LORA_DROPOUT,
            getattr(args, "lora_targets", None) is not None,
            bool(getattr(args, "trust_remote_code", False)),
            bool(getattr(args, "local_files_only", False)),
        )
    )


def _validate_recipe(recipe: Mapping[str, Any], device: torch.device) -> None:
    if recipe["model_kind"] not in ("encoder", "decoder"):
        raise ValueError("model_kind must be encoder or decoder")
    if recipe["quantization"] not in ("none", "8bit", "4bit"):
        raise ValueError("quantization must be none, 8bit, or 4bit")
    if recipe["quantization"] != "none":
        if device.type != "cuda":
            raise ValueError("4-bit and 8-bit training require CUDA")
        if not recipe["peft"]:
            raise ValueError("4-bit and 8-bit training require --peft")
    if recipe["dtype"] not in DTYPE_NAMES or recipe["compute_dtype"] not in DTYPE_NAMES:
        raise ValueError("unsupported dtype in model recipe")
    if int(recipe["lora_r"]) <= 0 or int(recipe["lora_alpha"]) <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= float(recipe["lora_dropout"]) < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")
    tag = str(recipe["model_tag"])
    if not tag or tag in (".", "..") or "/" in tag or "\\" in tag:
        raise ValueError("model_tag must be one non-empty path component")


def resolve_training_recipe(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Resolve legacy or generic CLI arguments into a JSON-safe model recipe."""
    arm = getattr(args, "arm", None)
    if arm is not None:
        spec = MODEL_SPECS[arm]
        model_kind = _arm_model_kind(arm)
        default_name = str(spec["model_name"])
        legacy_defaults = not _arm_recipe_is_overridden(args)
    else:
        model_kind = getattr(args, "model_kind", None)
        default_name = getattr(args, "model_name", None)
        if not default_name or not model_kind:
            raise ValueError("generic training requires --model-name and --model-kind")
        legacy_defaults = False

    model_name = getattr(args, "model_name", None) or default_name
    requested_peft = getattr(args, "peft", None)
    peft = True if requested_peft is None else bool(requested_peft)
    quantization = getattr(args, "quantization", None) or "none"
    dtype_name = _resolve_dtype_name(getattr(args, "dtype", "auto") or "auto", device)
    compute_name = _resolve_dtype_name(
        getattr(args, "compute_dtype", "auto") or "auto", device
    )
    targets = _parse_lora_targets(
        getattr(args, "lora_targets", None), model_kind
    )
    model_tag = getattr(args, "model_tag", None)
    tag_is_auto = model_tag is None and not legacy_defaults
    tag_identity = {
        "model_name": str(model_name),
        "model_kind": str(model_kind),
        "model_revision": getattr(args, "model_revision", None),
        "tokenizer_name": getattr(args, "tokenizer_name", None),
        "tokenizer_revision": getattr(args, "tokenizer_revision", None),
        "peft": peft,
        "quantization": quantization,
        "dtype": dtype_name,
        "compute_dtype": compute_name,
        "gradient_checkpointing": bool(
            getattr(args, "gradient_checkpointing", False)
        ),
        "lora_r": int(getattr(args, "lora_r", LORA_RANK)),
        "lora_alpha": int(getattr(args, "lora_alpha", LORA_ALPHA)),
        "lora_dropout": float(getattr(args, "lora_dropout", LORA_DROPOUT)),
        "lora_targets": targets,
    }
    if model_tag is None:
        model_tag = (
            arm
            if legacy_defaults
            else _safe_model_tag(model_name, peft, quantization, tag_identity)
        )
    recipe = {
        "schema_version": 2,
        "legacy_arm": arm,
        "legacy_defaults": legacy_defaults,
        "model_name": str(model_name),
        "model_kind": str(model_kind),
        "model_tag": str(model_tag),
        "model_tag_is_auto": tag_is_auto,
        "model_revision": getattr(args, "model_revision", None),
        "tokenizer_name": getattr(args, "tokenizer_name", None),
        "tokenizer_revision": getattr(args, "tokenizer_revision", None),
        "peft": peft,
        "quantization": quantization,
        "dtype": dtype_name,
        "compute_dtype": compute_name,
        "gradient_checkpointing": bool(
            getattr(args, "gradient_checkpointing", False)
        ),
        "lora_r": int(getattr(args, "lora_r", LORA_RANK)),
        "lora_alpha": int(getattr(args, "lora_alpha", LORA_ALPHA)),
        "lora_dropout": float(getattr(args, "lora_dropout", LORA_DROPOUT)),
        "lora_targets": targets,
        "lora_bias": "none",
        "lora_task_type": "SEQ_CLS" if model_kind == "encoder" else "FEATURE_EXTRACTION",
        "trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
        "local_files_only": bool(getattr(args, "local_files_only", False)),
    }
    _validate_recipe(recipe, device)
    return recipe


def pin_remote_revisions(
    recipe: Mapping[str, Any], cache_dir: Path
) -> dict[str, Any]:
    """Resolve mutable Hub revisions to immutable commit hashes before training."""
    pinned = dict(recipe)
    if bool(pinned.get("legacy_defaults", False)):
        return pinned

    from transformers import AutoConfig, AutoTokenizer

    common = {
        "cache_dir": str(cache_dir),
        "trust_remote_code": bool(pinned.get("trust_remote_code", False)),
        "local_files_only": bool(pinned.get("local_files_only", False)),
    }
    model_name = str(pinned["model_name"])
    requested_model_revision = pinned.get("model_revision")
    pinned["requested_model_revision"] = requested_model_revision
    if not Path(model_name).expanduser().exists():
        config = AutoConfig.from_pretrained(
            model_name, revision=requested_model_revision, **common
        )
        commit_hash = getattr(config, "_commit_hash", None)
        if not commit_hash:
            raise RuntimeError(f"could not resolve an immutable revision for {model_name}")
        pinned["model_revision"] = str(commit_hash)

    tokenizer_name = pinned.get("tokenizer_name")
    requested_tokenizer_revision = pinned.get("tokenizer_revision")
    if tokenizer_name:
        pinned["requested_tokenizer_revision"] = requested_tokenizer_revision
    if tokenizer_name and str(tokenizer_name) != model_name:
        if not Path(str(tokenizer_name)).expanduser().exists():
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_name), revision=requested_tokenizer_revision, **common
            )
            tokenizer_commit = tokenizer.init_kwargs.get("_commit_hash")
            if not tokenizer_commit:
                raise RuntimeError(
                    f"could not resolve an immutable tokenizer revision for {tokenizer_name}"
                )
            pinned["tokenizer_revision"] = str(tokenizer_commit)
    else:
        pinned["tokenizer_revision"] = pinned.get("model_revision")

    return pinned


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
    kind = _model_kind_to_prompt_kind(kind)
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
    *,
    recipe: Mapping[str, Any] | None = None,
) -> Any:
    from transformers import AutoTokenizer

    source = (
        str(tokenizer_path)
        if tokenizer_path is not None and tokenizer_path.exists()
        else str(recipe.get("tokenizer_name") or model_name) if recipe else model_name
    )
    load_kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)}
    if recipe is not None and not bool(recipe.get("legacy_defaults", False)):
        revision = (
            recipe.get("tokenizer_revision")
            if recipe.get("tokenizer_name")
            else recipe.get("model_revision")
        )
        load_kwargs.update(
            revision=revision,
            trust_remote_code=bool(recipe.get("trust_remote_code", False)),
            local_files_only=bool(recipe.get("local_files_only", False)),
        )
    tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"{model_name} tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class LlamaSequenceClassifier(nn.Module):
    """Decoder backbone plus the current MLP head on final non-padding state."""

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
    *,
    recipe: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Reconstruct a pretrained model from a complete legacy or v2 recipe."""
    if recipe is None:
        if arm not in MODEL_SPECS:
            raise ValueError("a model recipe is required for a generic model")
        spec = MODEL_SPECS[arm]
        resolved_recipe: dict[str, Any] = {
            "model_name": model_name or spec["model_name"],
            "model_kind": _arm_model_kind(arm),
            "model_tag": arm,
            "model_tag_is_auto": False,
            "model_revision": None,
            "tokenizer_name": None,
            "tokenizer_revision": None,
            "peft": True,
            "quantization": "none",
            "dtype": "bf16" if dtype == torch.bfloat16 else "fp32",
            "compute_dtype": "bf16" if dtype == torch.bfloat16 else "fp32",
            "gradient_checkpointing": False,
            "lora_r": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_targets": list(spec["lora_targets"]),
            "lora_bias": "none",
            "lora_task_type": spec["lora_task_type"],
            "trust_remote_code": False,
            "local_files_only": False,
            "legacy_arm": arm,
            "legacy_defaults": True,
        }
    else:
        resolved_recipe = dict(recipe)

    resolved_name = str(resolved_recipe["model_name"])
    model_kind = str(resolved_recipe["model_kind"])
    quantization = str(resolved_recipe.get("quantization", "none"))
    peft_enabled = bool(resolved_recipe.get("peft", True))
    if quantization != "none" and device.type != "cuda":
        raise ValueError("4-bit and 8-bit model loading requires CUDA")
    if quantization != "none" and not peft_enabled:
        raise ValueError("4-bit and 8-bit model loading requires PEFT")
    model_dtype = DTYPE_NAMES[str(resolved_recipe.get("dtype", "fp32"))]
    compute_dtype = DTYPE_NAMES[
        str(resolved_recipe.get("compute_dtype", resolved_recipe.get("dtype", "fp32")))
    ]
    parameter_dtype = (
        torch.float32
        if quantization == "none" and not peft_enabled
        else model_dtype
    )
    load_kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)}
    if not bool(resolved_recipe.get("legacy_defaults", False)):
        load_kwargs.update(
            revision=resolved_recipe.get("model_revision"),
            trust_remote_code=bool(resolved_recipe.get("trust_remote_code", False)),
            local_files_only=bool(resolved_recipe.get("local_files_only", False)),
        )
    # Keep the historical BERT arm's FP32 load exact. PEFT backbones use the
    # requested storage dtype; full fine-tuning keeps FP32 trainable parameters
    # and uses compute_dtype for mixed-precision autocast.
    if not (
        resolved_recipe.get("legacy_arm") == "bert_lora"
        and resolved_recipe.get("legacy_defaults", True)
        and quantization == "none"
    ):
        load_kwargs["torch_dtype"] = parameter_dtype
    if quantization != "none":
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError(
                "quantized training requires bitsandbytes; install it in the training environment"
            )
        if importlib.util.find_spec("accelerate") is None:
            raise RuntimeError(
                "quantized device placement requires the accelerate package"
            )
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError(
                "quantized loading requires a Transformers build with bitsandbytes support"
            ) from error

        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        load_kwargs["device_map"] = {"": device_index}

    if model_kind == "encoder":
        from transformers import AutoModelForSequenceClassification

        backbone = AutoModelForSequenceClassification.from_pretrained(
            resolved_name, num_labels=2, **load_kwargs
        )
    else:
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(resolved_name, **load_kwargs)
        backbone.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = not bool(
                resolved_recipe.get("gradient_checkpointing", False)
            )

    gradient_checkpointing = bool(
        resolved_recipe.get("gradient_checkpointing", False)
    )
    if gradient_checkpointing and not hasattr(backbone, "gradient_checkpointing_enable"):
        raise ValueError(
            f"{resolved_name} does not expose Transformers gradient checkpointing"
        )
    if quantization != "none":
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as error:
            raise RuntimeError("quantized training requires the peft package") from error

        backbone = prepare_model_for_kbit_training(
            backbone,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        if peft_enabled and hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

    if peft_enabled:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as error:
            raise RuntimeError("--peft requires the peft package") from error

        lora = LoraConfig(
            r=int(resolved_recipe["lora_r"]),
            lora_alpha=int(resolved_recipe["lora_alpha"]),
            lora_dropout=float(resolved_recipe["lora_dropout"]),
            target_modules=list(resolved_recipe["lora_targets"]),
            bias=str(resolved_recipe.get("lora_bias", "none")),
            task_type=str(resolved_recipe["lora_task_type"]),
        )
        backbone = get_peft_model(backbone, lora)

    if model_kind == "encoder":
        model = backbone
    else:
        model = LlamaSequenceClassifier(backbone)
        head_dtype = compute_dtype if quantization != "none" else parameter_dtype
        model.classification_head.to(device=device, dtype=head_dtype)

    # Calling .to() on a bitsandbytes-quantized model is unsupported.  Its
    # backbone was placed by device_map and the decoder head above was placed
    # explicitly.  Non-quantized models retain the historical whole-model move.
    if quantization == "none":
        model = model.to(device)
    return model


def model_logits(
    arm: str,
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    kind = MODEL_SPECS[arm]["kind"] if arm in MODEL_SPECS else arm
    if _model_kind_to_prompt_kind(kind) == "bert":
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
        self.kind = (
            MODEL_SPECS[arm]["kind"] if arm in MODEL_SPECS else _model_kind_to_prompt_kind(arm)
        )
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


def autocast_context(device: torch.device, dtype: torch.dtype = torch.bfloat16):
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_model(
    arm: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
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
        with autocast_context(device, compute_dtype):
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
    """Extract trainable tensors without constructing a full model state dict."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in sorted(model.named_parameters())
        if parameter.requires_grad
    }


def load_trainable_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load trainable tensors directly, without materialising frozen tensors."""
    parameters = dict(model.named_parameters())
    unexpected = set(state) - set(parameters)
    if unexpected:
        raise KeyError(
            f"checkpoint contains unexpected trainable tensors: {sorted(unexpected)[:3]}"
        )
    expected = {
        name for name, parameter in parameters.items() if parameter.requires_grad
    }
    missing = expected - set(state)
    if missing:
        raise KeyError(f"checkpoint is missing trainable tensors: {sorted(missing)[:3]}")
    with torch.no_grad():
        for name, tensor in state.items():
            parameter = parameters[name]
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"shape mismatch for {name}: {tuple(tensor.shape)} != "
                    f"{tuple(parameter.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


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
    recipe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    peft_enabled = True if recipe is None else bool(recipe.get("peft", True))
    payload = {
        "schema_version": 1 if recipe is None else 2,
        "state_format": (
            "lora_adapter_and_classification_head_only"
            if peft_enabled
            else "all_trainable_parameters"
        ),
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
    if recipe is not None:
        payload["model_recipe"] = dict(recipe)
    return payload


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
        *,
        model_kind: str | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        self.arm = arm
        resolved_kind = (
            MODEL_SPECS[arm]["kind"]
            if arm in MODEL_SPECS
            else model_kind or arm
        )
        self.kind = _model_kind_to_prompt_kind(resolved_kind)
        self.model_kind = "encoder" if self.kind == "bert" else "decoder"
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.device = device
        self.batch_size = int(batch_size)
        self.compute_dtype = compute_dtype

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
            with autocast_context(self.device, self.compute_dtype):
                logits = model_logits(self.model_kind, self.model, input_ids, attention_mask)
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


def _legacy_recipe_from_config(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Upgrade a schema-v1 arm checkpoint in memory for generic reloading."""
    arm = str(checkpoint["arm"])
    spec = MODEL_SPECS[arm]
    dtype_name = "bf16" if device.type == "cuda" else "fp32"
    lora = checkpoint.get("lora", config.get("lora", {}))
    return {
        "schema_version": 1,
        "legacy_arm": arm,
        "legacy_defaults": True,
        "model_name": str(
            config.get("model_name", checkpoint.get("model_name", spec["model_name"]))
        ),
        "model_kind": _arm_model_kind(arm),
        "model_tag": arm,
        "model_tag_is_auto": False,
        "model_revision": None,
        "tokenizer_name": None,
        "tokenizer_revision": None,
        "peft": True,
        "quantization": "none",
        "dtype": dtype_name,
        "compute_dtype": dtype_name,
        "gradient_checkpointing": False,
        "lora_r": int(lora.get("rank", LORA_RANK)),
        "lora_alpha": int(lora.get("alpha", LORA_ALPHA)),
        "lora_dropout": float(lora.get("dropout", LORA_DROPOUT)),
        "lora_targets": list(lora.get("targets", spec["lora_targets"])),
        "lora_bias": str(lora.get("bias", "none")),
        "lora_task_type": str(spec["lora_task_type"]),
        "trust_remote_code": False,
        "local_files_only": False,
    }


def load_hf_scorer(
    checkpoint_or_run_dir: Path | str,
    device: str | torch.device | None = None,
    batch_size: int = 64,
    verify_reload: bool = True,
    cache_dir: Path | str | None = None,
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
    stored_recipe = checkpoint.get("model_recipe") or config.get("model_recipe")
    if (
        checkpoint.get("model_recipe") is not None
        and config.get("model_recipe") is not None
        and checkpoint["model_recipe"] != config["model_recipe"]
    ):
        raise ValueError("checkpoint model recipe does not match config.json")
    recipe = (
        dict(stored_recipe)
        if stored_recipe is not None
        else _legacy_recipe_from_config(config, checkpoint, resolved_device)
    )
    _validate_recipe(recipe, resolved_device)
    dtype = DTYPE_NAMES[str(recipe["dtype"])]
    compute_dtype = DTYPE_NAMES[str(recipe["compute_dtype"])]
    environment_cache = os.environ.get("HF_HUB_CACHE")
    if environment_cache is None and os.environ.get("HF_HOME"):
        environment_cache = str(Path(os.environ["HF_HOME"]) / "hub")
    resolved_cache_dir = Path(
        cache_dir or environment_cache or config.get("hf_cache", DEFAULT_HF_CACHE)
    )
    tokenizer = build_tokenizer(
        arm,
        recipe["model_name"],
        resolved_cache_dir,
        tokenizer_path=run_dir / "tokenizer",
        recipe=recipe,
    )
    model = build_hf_model(
        arm,
        tokenizer,
        resolved_device,
        dtype,
        resolved_cache_dir,
        model_name=recipe["model_name"],
        recipe=recipe,
    )
    load_trainable_state_dict(model, checkpoint["adapter_and_head_state_dict"])
    scorer = HFScorer(
        arm,
        model,
        tokenizer,
        checkpoint["max_length"],
        resolved_device,
        batch_size,
        model_kind=recipe["model_kind"],
        compute_dtype=compute_dtype,
    )
    reload_difference = None
    if verify_reload and checkpoint.get("reload_probe"):
        reload_difference = _verify_reload_probe(scorer, checkpoint["reload_probe"])
    done_path = run_dir / "done.json"
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    metadata = {
        "model": arm,
        "arm": arm,
        "model_name": recipe["model_name"],
        "model_tag": recipe["model_tag"],
        "model_kind": recipe["model_kind"],
        "peft": recipe["peft"],
        "quantization": recipe["quantization"],
        "model_recipe": recipe,
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


def _requested_recipe_matches_stored(
    requested: Mapping[str, Any], stored: Mapping[str, Any]
) -> bool:
    """Compare a CLI recipe with a stored recipe whose Hub revisions are pinned."""
    comparable = dict(stored)
    comparable["model_revision"] = comparable.pop(
        "requested_model_revision", comparable.get("model_revision")
    )
    if comparable.get("tokenizer_name"):
        comparable["tokenizer_revision"] = comparable.pop(
            "requested_tokenizer_revision", comparable.get("tokenizer_revision")
        )
    else:
        comparable.pop("requested_tokenizer_revision", None)
        comparable["tokenizer_revision"] = requested.get("tokenizer_revision")
    return comparable == dict(requested)


def _validate_resume_configuration(
    config_path: Path,
    stored: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    noise_pi: float,
    model_seed: int,
    data_fingerprint: str,
    effective_batch: int,
    micro_batch: int,
    accumulation: int,
    eval_batch: int,
    max_epochs: int,
    patience: int,
    max_train_rows: int,
    smoke: bool,
) -> None:
    """Refuse to reuse a run directory for a different experiment."""
    expected = {
        "model_recipe": dict(recipe),
        "noise_pi": float(noise_pi),
        "model_seed": int(model_seed),
        "data_fingerprint": data_fingerprint,
        "effective_batch_size": int(effective_batch),
        "micro_batch_size": int(micro_batch),
        "gradient_accumulation": int(accumulation),
        "eval_batch_size": int(eval_batch),
        "maximum_epochs": int(max_epochs),
        "early_stopping_patience": int(patience),
        "max_train_rows": int(max_train_rows),
        "smoke": bool(smoke),
    }
    mismatches = {
        key: {"stored": stored.get(key), "requested": value}
        for key, value in expected.items()
        if stored.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={values['stored']!r} (requested {values['requested']!r})"
            for key, values in mismatches.items()
        )
        raise ValueError(f"resume configuration does not match {config_path}: {details}")


def _load_or_create_audit(
    run_dir: Path,
    tokenizer: Any,
    arm: str,
    model_name: str,
    split_frames: Mapping[str, pd.DataFrame],
    data_fingerprint: str,
    audit_batch_size: int,
    resume: bool,
    *,
    model_kind: str | None = None,
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
        MODEL_SPECS[arm]["kind"] if arm in MODEL_SPECS else model_kind or arm,
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


def _resolve_batch_configuration(
    args: argparse.Namespace,
    legacy_spec: Mapping[str, Any] | None,
    model_kind: str,
) -> tuple[int, int, int, int]:
    """Resolve micro/evaluation batches and gradient accumulation."""
    effective_batch = int(args.effective_batch_size)
    default_micro_batch = (
        int(legacy_spec["micro_batch"])
        if legacy_spec is not None
        else 16 if model_kind == "encoder" else 1
    )
    default_accumulation = (
        int(legacy_spec["gradient_accumulation"])
        if legacy_spec is not None
        else EFFECTIVE_BATCH_SIZE // default_micro_batch
    )

    if args.micro_batch is None and args.gradient_accumulation is None:
        micro_batch = default_micro_batch
        if effective_batch == EFFECTIVE_BATCH_SIZE:
            accumulation = default_accumulation
        else:
            if effective_batch % micro_batch:
                raise ValueError(
                    "effective batch size must be divisible by the default "
                    f"micro-batch {micro_batch}; pass --micro-batch explicitly"
                )
            accumulation = effective_batch // micro_batch
    elif args.micro_batch is None:
        accumulation = int(args.gradient_accumulation)
        if effective_batch % accumulation:
            raise ValueError(
                "effective batch size must be divisible by gradient accumulation"
            )
        micro_batch = effective_batch // accumulation
    elif args.gradient_accumulation is None:
        micro_batch = int(args.micro_batch)
        if effective_batch % micro_batch:
            raise ValueError(
                "effective batch size must be divisible by the micro-batch size"
            )
        accumulation = effective_batch // micro_batch
    else:
        micro_batch = int(args.micro_batch)
        accumulation = int(args.gradient_accumulation)

    if micro_batch * accumulation != effective_batch:
        raise ValueError(
            "micro_batch * gradient_accumulation must equal the selected "
            f"effective batch size {effective_batch}"
        )
    eval_batch = int(args.eval_batch) if args.eval_batch is not None else (
        64 if legacy_spec is not None or model_kind == "encoder" else 2
    )
    return micro_batch, accumulation, eval_batch, effective_batch


def train(args: argparse.Namespace) -> dict[str, Any]:
    if float(args.pi) not in NOISE_LEVELS:
        raise ValueError(f"pi must be one of {NOISE_LEVELS}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    requested_recipe = resolve_training_recipe(args, device)
    legacy_spec = MODEL_SPECS.get(args.arm) if args.arm is not None else None
    micro_batch, accumulation, eval_batch, effective_batch = (
        _resolve_batch_configuration(
            args,
            legacy_spec,
            requested_recipe["model_kind"],
        )
    )
    suffix = "_smoke" if args.smoke else ""
    run_identity = (
        args.arm
        if args.arm is not None
        and requested_recipe["legacy_defaults"]
        and getattr(args, "model_tag", None) is None
        else requested_recipe["model_tag"]
    )
    run_dir = args.run_dir or (
        args.checkpoint_root
        / run_identity
        / f"pi_{pi_tag(args.pi)}"
        / f"seed_{args.model_seed}{suffix}"
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "epochs").mkdir(exist_ok=True)
    config_path = run_dir / "config.json"
    data_fingerprint = _data_fingerprint(args.data_root)
    stored_config: dict[str, Any] | None = None
    if args.resume and config_path.exists():
        stored_config = json.loads(config_path.read_text(encoding="utf-8"))
        stored_recipe = stored_config.get("model_recipe")
        if stored_recipe is not None:
            if not _requested_recipe_matches_stored(requested_recipe, stored_recipe):
                raise ValueError(f"resume model recipe does not match {config_path}")
            recipe = dict(stored_recipe)
        else:
            recipe = requested_recipe
    else:
        if config_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing run {run_dir}; use --resume or a new run"
            )
        recipe = pin_remote_revisions(requested_recipe, args.hf_cache)
    if stored_config is not None and stored_config.get("model_recipe") is not None:
        _validate_resume_configuration(
            config_path,
            stored_config,
            recipe=recipe,
            noise_pi=args.pi,
            model_seed=args.model_seed,
            data_fingerprint=data_fingerprint,
            effective_batch=effective_batch,
            micro_batch=micro_batch,
            accumulation=accumulation,
            eval_batch=eval_batch,
            max_epochs=args.max_epochs,
            patience=args.patience,
            max_train_rows=args.max_train_rows,
            smoke=args.smoke,
        )
    if args.resume and _valid_done(run_dir):
        print(f"[ordered_train_hf] valid done.json found; skipping {run_dir}", flush=True)
        return json.loads((run_dir / "done.json").read_text(encoding="utf-8"))

    set_model_seed(args.model_seed)
    torch.set_num_threads(args.threads)
    dtype = DTYPE_NAMES[recipe["dtype"]]
    compute_dtype = DTYPE_NAMES[recipe["compute_dtype"]]
    runtime_arm = run_identity
    dataset_kind = args.arm or recipe["model_kind"]
    frames = {
        split: load_split(args.data_root, split, args.pi)
        for split in ("train", "val", "test")
    }
    audit_frames = {split: frame for split, frame in frames.items()}
    data_metadata = _data_manifest_metadata(args.data_root)

    tokenizer = build_tokenizer(
        runtime_arm,
        recipe["model_name"],
        args.hf_cache,
        recipe=recipe,
    )
    audit = _load_or_create_audit(
        run_dir,
        tokenizer,
        runtime_arm,
        recipe["model_name"],
        audit_frames,
        data_fingerprint,
        args.audit_batch_size,
        args.resume,
        model_kind=recipe["model_kind"],
    )
    max_length = int(audit["selected_max_length"])
    if max_length not in (TOKEN_LENGTH_SHORT, TOKEN_LENGTH_LONG):
        raise ValueError(f"invalid audited max length {max_length}")
    tokenizer.save_pretrained(run_dir / "tokenizer")
    if args.max_train_rows:
        frames["train"] = frames["train"].iloc[: args.max_train_rows].reset_index(drop=True)

    model = build_hf_model(
        runtime_arm,
        tokenizer,
        device,
        dtype,
        args.hf_cache,
        model_name=recipe["model_name"],
        recipe=recipe,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise AssertionError("LoRA model has no trainable parameters")
    loaders, loader_generator = _make_loaders(
        dataset_kind,
        frames,
        tokenizer,
        max_length,
        micro_batch,
        eval_batch,
        args.workers,
        args.model_seed,
        device,
    )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=LEARNING_RATE)
    updates_per_epoch = math.ceil(len(loaders["train"]) / accumulation)
    total_updates = updates_per_epoch * args.max_epochs
    warmup_updates = int(WARMUP_RATIO * total_updates)
    scheduler = linear_warmup_decay_scheduler(optimizer, warmup_updates, total_updates)
    # BF16 does not require loss scaling; FP16 does.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and compute_dtype == torch.float16
    )
    stopper = EarlyStopping(args.patience)
    history: list[dict[str, Any]] = []
    best_epoch = 0
    start_epoch = 1
    prior_training_seconds = 0.0

    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        expected = {
            "arm": runtime_arm,
            "noise_pi": float(args.pi),
            "model_seed": int(args.model_seed),
            "max_length": max_length,
            "maximum_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "effective_batch_size": int(effective_batch),
            "micro_batch_size": int(micro_batch),
            "gradient_accumulation": int(accumulation),
            "data_fingerprint": data_fingerprint,
        }
        if checkpoint.get("model_recipe") is not None:
            expected["model_recipe"] = recipe
        for key, value in expected.items():
            actual = checkpoint.get(key)
            if key == "effective_batch_size" and actual is None:
                # Checkpoints written before effective batches became configurable
                # already contain both factors, so their value is unambiguous.
                stored_micro = checkpoint.get("micro_batch_size")
                stored_accumulation = checkpoint.get("gradient_accumulation")
                if stored_micro is not None and stored_accumulation is not None:
                    actual = int(stored_micro) * int(stored_accumulation)
            if actual != value:
                raise ValueError(
                    f"resume mismatch for {key}: {actual!r} != {value!r}"
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
        "schema_version": 2,
        "arm": runtime_arm,
        "legacy_arm": args.arm,
        "kind": _model_kind_to_prompt_kind(recipe["model_kind"]),
        "model_kind": recipe["model_kind"],
        "model_tag": recipe["model_tag"],
        "model_name": recipe["model_name"],
        "model_recipe": recipe,
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
            if recipe["model_kind"] == "encoder"
            else "Sequential events: x1 ... x20\\nOutcome (0 or 1):"
        ),
        "training_objective": "binary classification cross entropy only",
        "uses_next_token_loss": False,
        "llama_pooling": "final non-padding hidden state",
        "llama_head": "Linear(hidden,hidden/2)-Tanh-Dropout(0.1)-Linear(hidden/2,2)",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "effective_batch_size": effective_batch,
        "micro_batch_size": micro_batch,
        "gradient_accumulation": accumulation,
        "eval_batch_size": eval_batch,
        "maximum_epochs": args.max_epochs,
        "early_stopping_patience": args.patience,
        "early_stopping_criterion": "standard observed-label validation loss",
        "warmup_ratio": WARMUP_RATIO,
        "warmup_updates": warmup_updates,
        "total_updates": total_updates,
        "scheduler": "linear decay",
        "lora": {
            "enabled": recipe["peft"],
            "rank": recipe["lora_r"],
            "alpha": recipe["lora_alpha"],
            "dropout": recipe["lora_dropout"],
            "targets": list(recipe["lora_targets"]),
            "bias": recipe["lora_bias"],
        },
        "quantization": recipe["quantization"],
        "compute_dtype": recipe["compute_dtype"],
        "gradient_checkpointing": recipe["gradient_checkpointing"],
        "max_length": max_length,
        "tokenization_audit": "tokenization_audit.json",
        "dtype": recipe["dtype"],
        "parameter_dtype": (
            "fp32"
            if recipe["quantization"] == "none" and not recipe["peft"]
            else recipe["dtype"]
        ),
        "hardware": _hardware(device),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "checkpoint_state": (
            "LoRA adapter and classification head only"
            if recipe["peft"]
            else "all trainable parameters"
        ),
        "max_train_rows": int(args.max_train_rows),
        "smoke": bool(args.smoke),
    }
    atomic_json_dump(configuration, run_dir / "config.json")

    scorer = HFScorer(
        runtime_arm,
        model,
        tokenizer,
        max_length,
        device,
        batch_size=eval_batch,
        model_kind=recipe["model_kind"],
        compute_dtype=compute_dtype,
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
            with autocast_context(device, compute_dtype):
                logits = model_logits(recipe["model_kind"], model, input_ids, attention_mask)
                loss = nn.functional.cross_entropy(logits.float(), labels)
            scaled_loss = loss / accumulation
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if batch_index % accumulation == 0 or batch_index == len(loaders["train"]):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            examples += batch_size

        validation = evaluate_model(
            recipe["model_kind"], model, loaders["val"], device, compute_dtype
        )
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
            runtime_arm,
            args.pi,
            args.model_seed,
            epoch,
            stopper.best_loss,
            best_epoch,
            stopper.counter,
            max_length,
            history,
            probe,
            recipe,
        )
        payload["training_seconds"] = elapsed
        payload["maximum_epochs"] = int(args.max_epochs)
        payload["patience"] = int(args.patience)
        payload["effective_batch_size"] = int(effective_batch)
        payload["micro_batch_size"] = int(micro_batch)
        payload["gradient_accumulation"] = int(accumulation)
        payload["model_name"] = recipe["model_name"]
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
                "schema_version": 2,
                "state_format": payload["state_format"],
                "adapter_and_head_state_dict": payload["adapter_and_head_state_dict"],
                "trainable_parameter_names": payload["trainable_parameter_names"],
                "arm": runtime_arm,
                "noise_pi": float(args.pi),
                "model_seed": int(args.model_seed),
                "current_epoch": epoch,
                "max_length": max_length,
                "model_name": recipe["model_name"],
                "model_recipe": recipe,
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
            f"[ordered_train_hf] {runtime_arm} pi={args.pi:.1f} seed={args.model_seed} "
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
    validation = evaluate_model(
        recipe["model_kind"], model, loaders["val"], device, compute_dtype
    )
    threshold, validation_f1 = select_f1_threshold(
        validation["observed_labels"], validation["probabilities"]
    )
    test = evaluate_model(
        recipe["model_kind"], model, loaders["test"], device, compute_dtype
    )
    test_predictions = (test["probabilities"] >= threshold).astype(np.uint8)
    training_seconds = prior_training_seconds + time.time() - started
    done = {
        "status": "complete",
        "arm": runtime_arm,
        "legacy_arm": args.arm,
        "model_name": recipe["model_name"],
        "model_tag": recipe["model_tag"],
        "model_kind": recipe["model_kind"],
        "peft": recipe["peft"],
        "quantization": recipe["quantization"],
        "noise_pi": float(args.pi),
        "model_seed": int(args.model_seed),
        "effective_batch_size": int(effective_batch),
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
    parser.add_argument("--arm", choices=tuple(MODEL_SPECS), default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-kind", choices=("encoder", "decoder"), default=None)
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument(
        "--peft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable LoRA/PEFT (use --no-peft for full fine-tuning)",
    )
    parser.add_argument(
        "--quantization", choices=("none", "8bit", "4bit"), default="none"
    )
    parser.add_argument(
        "--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lora-r", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument(
        "--lora-targets",
        default=None,
        help="comma-separated target-module suffixes; defaults depend on model kind",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
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
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=EFFECTIVE_BATCH_SIZE,
        help=f"examples per optimizer update (default: {EFFECTIVE_BATCH_SIZE})",
    )
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument(
        "--eval-batch",
        type=int,
        default=None,
        help="evaluation batch size (default: 64 for encoders/legacy, 2 for decoders)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--audit-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    if args.max_epochs <= 0 or args.patience <= 0:
        parser.error("max epochs and patience must be positive")
    if args.arm is None and (args.model_name is None or args.model_kind is None):
        parser.error("provide --arm, or both --model-name and --model-kind")
    if args.arm is not None and args.model_kind is not None:
        expected_kind = _arm_model_kind(args.arm)
        if args.model_kind != expected_kind:
            parser.error(
                f"--arm {args.arm} has model kind {expected_kind}, not {args.model_kind}"
            )
    if args.arm is not None and _arm_recipe_is_overridden(args) and not args.model_tag:
        parser.error("arm recipe overrides require an explicit --model-tag")
    if args.peft is False and args.quantization != "none":
        parser.error("quantized training requires --peft")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        parser.error("LoRA rank and alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("LoRA dropout must be in [0, 1)")
    if args.effective_batch_size <= 0 or any(
        value is not None and value <= 0
        for value in (args.micro_batch, args.gradient_accumulation)
    ) or (args.eval_batch is not None and args.eval_batch <= 0):
        parser.error("batch sizes and gradient accumulation must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
