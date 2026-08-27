"""
Parity decomposition — LLM driver (Llama-3.2-1B + LoRA).

Reuses the CausalLM + classification-head wrapper, training loop, threshold
optimization, and tokenizer loader from src.experiments.LLM_fraction_experiment.
Adds:

  * a variant-aware prompt builder:
      raw     — "Sequential events: D W T P ... Outcome (0 or 1):"
      masked  — "Sequential events: D/1 W/1 T/0 ... Outcome (0 or 1):"
      bitonly — "Sequential bits: 0 0 1 0 0 1 ... Outcome (0 or 1):"
  * prediction dump to disk alongside the per-run JSON.

Hyperparameters follow scripts/slurm/sensitivity_LLM.slurm (peft=True,
bs=64, lr=2e-5, epochs=20, patience=3, max_length=128).
"""

from __future__ import annotations

# Workaround: the arm64 PyTorch 25.06 container ships torchao 0.11.0+git, but
# current PEFT requires torchao >= 0.16. PEFT's import_utils.is_torchao_available
# raises ImportError when torchao is found but below threshold, breaking
# get_peft_model even when we are not using torchao. Hide torchao from
# importlib.find_spec so PEFT treats it as absent and skips the dispatcher.
# bf16 LoRA does not need torchao.
import importlib.util as _importlib_util
_orig_find_spec = _importlib_util.find_spec
def _hide_torchao(name, *args, **kwargs):
    if name == "torchao" or name.startswith("torchao."):
        return None
    return _orig_find_spec(name, *args, **kwargs)
_importlib_util.find_spec = _hide_torchao

import argparse
import copy
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
from torch.utils.data import DataLoader

from src.data.parity_variants import make_llm_prompt
from src.experiments.LLM_fraction_experiment import (  # noqa: E402
    TemporalCausalDataset,
    find_optimal_threshold,
    load_model_causal,
    load_tokenizer,
    set_seed,
    train_and_evaluate_causal,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

VARIANTS = ("raw", "masked", "bitonly")


# --------------------------------------------------------------------------- #
# Data preparation                                                            #
# --------------------------------------------------------------------------- #
def _variant_paths(data_dir: Path, variant: str, size_tag: str) -> dict[str, Path]:
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


def _anchor_paths(data_dir: Path, ell: int, k: int, n: int | None = None) -> dict[str, Path]:
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


def _load_csvs(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(p)
        out[name] = pd.read_csv(p).fillna("")
    return out


def _raw_sequences_from_variant(x_row: pd.Series, effective_variant: str) -> str:
    """
    Normalize a variant-specific stored sequence into the raw-letter string that
    make_llm_prompt expects. Masked: use the Sequences column verbatim (letters
    intact). Bit-only: the stored Sequences are B/C encoded — we render as raw
    text "0 0 1 ..." for the LLM via make_llm_prompt(variant="bitonly") but we
    must first reconstruct the bit string. For anchors (effective_variant =
    "anchor_bitonly"), similar treatment.
    """
    raise NotImplementedError  # keep isolated, helper below uses variants directly


def _render_prompts(
    X: pd.DataFrame,
    variant: str,
    X_raw: pd.DataFrame | None = None,
) -> list[str]:
    """
    Build prompts for every row of X.

    variant == "raw"     → prompt letters from X["Sequences"]
    variant == "masked"  → prompt letters + "/0" or "/1" using the MaskBits
                           column in X (if present) or recompute from X_raw.
    variant == "bitonly" → prompt bits "0 1 0 ...". If X["Sequences"] stores
                           the bitonly CSV (B/C encoding), we map back to bits.
                           If X_raw is provided, we recompute bits from the
                           original letters.
    """
    prompts: list[str] = []
    for i, seq in enumerate(X["Sequences"].astype(str).tolist()):
        if variant == "raw":
            prompts.append(make_llm_prompt(seq, "raw"))
        elif variant == "masked":
            if "MaskBits" in X.columns:
                letters = seq.split("\x1f") if "\x1f" in seq else list(seq)
                bits = X["MaskBits"].iloc[i]
                bit_toks = bits.split("\x1f") if "\x1f" in bits else list(bits)
                letters = [l for l in letters if l]
                bit_toks = [b for b in bit_toks if b != ""]
                pairs = " ".join(f"{l}/{b}" for l, b in zip(letters, bit_toks))
                prompts.append(f"Sequential events: {pairs}\nOutcome (0 or 1):")
            else:
                prompts.append(make_llm_prompt(seq, "masked"))
        elif variant == "bitonly":
            # Stored as B/C encoding in the bitonly CSV; B=0, C=1.
            toks = seq.split("\x1f") if "\x1f" in seq else list(seq)
            toks = [t for t in toks if t]
            bits = [1 if t == "C" else 0 for t in toks]
            body = " ".join(str(b) for b in bits)
            prompts.append(f"Sequential bits: {body}\nOutcome (0 or 1):")
        else:
            raise ValueError(f"unknown variant: {variant}")
    return prompts


# --------------------------------------------------------------------------- #
# Evaluation with prediction dump                                             #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_with_predictions(
    model, test_loader, device, threshold: float = 0.5
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        out = model(**inputs)
        p = torch.softmax(out["logits"], dim=-1)[:, 1].detach().cpu().float().numpy()
        probs.extend(p.tolist())
        labels.extend(batch["outcome_labels"].cpu().numpy().tolist())
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


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
_MODEL_TAGS = {
    "meta-llama/Llama-3.2-1B": "Llama32-1B",
    "meta-llama/Llama-3.1-8B": "Llama31-8B",
    "meta-llama/Llama-3.1-70B": "Llama31-70B",
    "Qwen/Qwen2.5-14B": "Qwen25-14B",
    "Qwen/Qwen3-4B-Think": "Qwen3-4B",
}


def _model_tag(model_name: str) -> str:
    if model_name in _MODEL_TAGS:
        return _MODEL_TAGS[model_name]
    return model_name.split("/")[-1].replace(".", "").replace("/", "-")


def _run_key(variant: str, size_tag: str, seed: int, anchor: tuple[int, int] | None,
             curriculum_n: int | None = None, init_checkpoint=None,
             model_name: str = "meta-llama/Llama-3.2-1B") -> str:
    tag = _model_tag(model_name)
    if curriculum_n is not None:
        stage = "stage2" if init_checkpoint else "stage1"
        return f"expF_{stage}_{tag}_n{curriculum_n}_seed{seed}"
    if anchor is not None:
        ell, k = anchor
        return f"anchor_l{ell}_k{k}_{tag}_{seed}"
    return f"{variant}_{tag}_{size_tag}_{seed}"


def run(args: argparse.Namespace) -> dict:
    # Use the LLM-file set_seed (benchmark=True). DL-driver cuDNN state is not
    # touched here since this process is LLM-only.
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "device=%s  variant=%s  size=%s  seed=%d",
        device,
        args.variant,
        args.size,
        args.seed,
    )

    anchor: tuple[int, int] | None = None
    curriculum_n: int | None = None
    if args.size.startswith("main_n"):
        # main_n{N} -> main-paper parity (K={W,D,Q,J,X,N}, ell=26) at length N.
        # Used by Experiment F curriculum (Stage 1 n=10, Stage 2 n=20).
        curriculum_n = int(args.size[len("main_n"):])
    elif args.size.startswith("binary"):
        ell = int(args.size.replace("binary", ""))
        k = max(1, ell // 2)
        anchor = (ell, k)

    if curriculum_n is not None:
        tag = f"main_parity_n{curriculum_n}"
        paths = {
            "X_train": Path(args.data_dir) / f"X_train_{tag}.csv",
            "y_train": Path(args.data_dir) / f"y_train_{tag}.csv",
            "X_val":   Path(args.data_dir) / f"X_val_{tag}.csv",
            "y_val":   Path(args.data_dir) / f"y_val_{tag}.csv",
            "X_test":  Path(args.data_dir) / f"X_test_{tag}.csv",
            "y_test":  Path(args.data_dir) / f"y_test_{tag}.csv",
        }
        variant_for_prompt = "raw"
    elif anchor is not None:
        paths = _anchor_paths(Path(args.data_dir), anchor[0], anchor[1])
        # Anchor CSVs store raw letters from a smaller alphabet (e.g. {B,C}
        # for ell=2, {B,C,D,E} for ell=4). Render prompts in raw form so the
        # LLM sees actual letters and can learn the key-subset -> parity
        # mapping exactly like it does on the main 26-letter task, just
        # smaller. Using "bitonly" here would be wrong for ell>2 because it
        # collapses the non-C alphabet letters to 0.
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

    if train_texts:
        logger.info("sample prompt (%s): %r", variant_for_prompt, train_texts[0])

    hf_token = os.environ.get("HF_TOKEN")

    # Reconstruct an args-shaped namespace for load_tokenizer / load_model_causal.
    # Those helpers expect fields like .model_name, .tiny, .peft, etc.
    llm_args = argparse.Namespace(
        model_type="general",
        model_name=args.model_name,
        tiny=False,
        tiny_type="1M",
        peft=args.peft,
        use_quantization=False,
        cold_start=False,
        cache_dir=args.cache_dir,
    )
    tokenizer = load_tokenizer(args.model_name, "general", hf_token, args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    train_ds = TemporalCausalDataset(train_texts, y_train, tokenizer, max_length=args.max_length)
    val_ds = TemporalCausalDataset(val_texts, y_val, tokenizer, max_length=args.max_length)
    test_ds = TemporalCausalDataset(test_texts, y_test, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=0)

    model = load_model_causal(llm_args, tokenizer, hf_token)
    if tokenizer.pad_token is not None:
        model.resize_token_embeddings(len(tokenizer))
    if not llm_args.use_quantization:
        model = model.to(device, dtype=torch.bfloat16)

    if args.init_checkpoint is not None:
        ckpt = Path(args.init_checkpoint)
        logger.info("loading Stage 1 checkpoint from %s", ckpt)
        state = torch.load(ckpt, map_location="cpu")
        # Cast to model's dtype (bf16) before loading.
        cast_state = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                      for k, v in state.items()}
        res = model.load_state_dict(cast_state, strict=False)
        logger.info("ckpt loaded: %d keys applied, %d missing (untouched), "
                    "%d unexpected (skipped)",
                    len(state), len(res.missing_keys), len(res.unexpected_keys))

    # Training arg shim for train_and_evaluate_causal. It accesses
    # .epochs, .patience, .gradient_accumulation_steps, .lr, .early.
    train_args = argparse.Namespace(
        epochs=args.epochs,
        patience=args.patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        early=args.early,
    )

    start = datetime.datetime.now()
    best_auc, best_f1, best_val_loss, epochs_done, val_preds, val_labels = train_and_evaluate_causal(
        model, train_loader, val_loader, train_args, device, use_quantization=False
    )
    optimal_threshold, val_f1_opt = find_optimal_threshold(val_labels, val_preds)
    logger.info("optimal threshold=%.3f (val_f1=%.4f)", optimal_threshold, val_f1_opt)

    test_metrics, test_labels, test_probs, test_preds = evaluate_with_predictions(
        model, test_loader, device, threshold=optimal_threshold
    )
    elapsed = (datetime.datetime.now() - start).total_seconds()

    if args.save_checkpoint is not None:
        ckpt = Path(args.save_checkpoint)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        # Trainable params only: LoRA adapter + classification head. Skip the
        # frozen base (Llama-3.2-1B is ~1B params); these alone are tens of MB.
        trainable_keys = [n for n, p in model.named_parameters() if p.requires_grad]
        # named_parameters() may not include head buffers, so also include any
        # state_dict key matching 'lora_' or 'classification_head' for safety.
        state_full = model.state_dict()
        keep = set(trainable_keys) | {k for k in state_full if "lora_" in k or "classification_head" in k}
        ckpt_state = {k: state_full[k].detach().cpu() for k in keep if k in state_full}
        torch.save(ckpt_state, ckpt)
        logger.info("saved Stage 1 checkpoint (%d tensors, %.1f MB) to %s",
                    len(ckpt_state),
                    sum(t.numel() * t.element_size() for t in ckpt_state.values()) / 1e6,
                    ckpt)

    output_dir = Path(args.output_dir)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    run_key = _run_key(args.variant, args.size, args.seed, anchor,
                       curriculum_n=curriculum_n, init_checkpoint=args.init_checkpoint,
                       model_name=args.model_name)
    pred_path = output_dir / "predictions" / f"{run_key}.csv"
    pd.DataFrame(
        {"y_true": test_labels, "y_prob": test_probs, "y_pred": test_preds}
    ).to_csv(pred_path, index=False)

    summary = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "run_key": run_key,
        "variant": args.variant,
        "variant_for_prompt": variant_for_prompt,
        "model": args.model_name,
        "peft": args.peft,
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
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
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
    p = argparse.ArgumentParser(description="Parity decomposition — LLM driver (Llama-3.2-1B + LoRA)")
    p.add_argument("--variant", choices=VARIANTS, default="raw")
    p.add_argument("--size", required=True,
                   help="Training-size tag: 1K, 40K, 400K, binary2, binary4")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--peft", action="store_true", default=True,
                   help="Use LoRA (default: True).")
    p.add_argument("--no_peft", dest="peft", action="store_false")
    p.add_argument("--data_dir", type=Path, default=Path("data/simulation/parity_decomp"))
    p.add_argument("--output_dir", type=Path, default=Path("results/parity_decomp"))
    p.add_argument("--cache_dir", type=Path,
                   default=Path(os.environ.get("SCRATCH", "/tmp")) / "cache")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--early", choices=["auc", "loss", "f1"], default="loss")
    p.add_argument("--init_checkpoint", type=Path, default=None,
                   help="Optional: load a Stage 1 checkpoint (LoRA adapter + "
                        "classification head state_dict) before training. Used "
                        "by Experiment F curriculum to warm-start Stage 2 from "
                        "Stage 1. RoPE-based models (Llama) need no PE transfer.")
    p.add_argument("--save_checkpoint", type=Path, default=None,
                   help="Optional: after training, save trainable state_dict "
                        "(LoRA adapter + classification head) for warm-start.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
