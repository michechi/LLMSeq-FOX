"""Train the pretrained models on the regenerated OC data with proper
checkpoints, resumable state, and per-epoch matched-completion diagnostics.

Arms:
  --arm bert_lora     bert-base-uncased, AutoModelForSequenceClassification,
                      LoRA r8/a16/drop0.1 on query,key,value (SEQ_CLS)
  --arm llama_lora    meta-llama/Llama-3.2-1B, CausalLMWithClassificationHead
                      (combined LM + classification loss, classification-head
                      inference), LoRA r8/a16/drop0.1 on q,k,v,o_proj
                      (CAUSAL_LM, backbone only - head trains full-rank)
  --arm llama_full    same model / prompt / head / optimizer / loss /
                      early stopping as llama_lora with ALL backbone weights
                      trainable and LoRA disabled

Recipe (paper + repro/src/experiments/{BERT,LLM}_fraction_experiment.py):
  AdamW lr 2e-5, effective batch 16 (micro-batch x grad accumulation),
  max 20 epochs, early stopping patience 3 on ORIGINAL validation loss,
  warmup ratio 0.06, linear decay, threshold tuned on validation F1
  (sweep 0.10..0.89). Prompts, datasets, heads and threshold search are
  IMPORTED from the fraction scripts (single source).

Recorded deviations for the 96-CPU node: max_length 64 (default 512 pads the
~26-35-token prompts more than 10x - padding only, content unaffected; an
assertion verifies no truncation); fp32 instead of bf16 when no GPU present.

The per-epoch completion diagnostic never influences early stopping,
checkpoint selection or the LR schedule.

Usage (from repro/ root, kip-venv):
    DATA_DIR=/root/LLMSeq/data HF_HUB_CACHE=/root/hf_cache \
    python -m src.oc_completion.train_hf --arm bert_lora --task ocdet \
        --seed 9550 [--smoke] [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader

from src.oc_completion.oracle import MECHANISM
from src.oc_completion.scoring import quick_pair_diag
from src.oc_completion.train_dl import (
    DATA_ROOT,
    RESULTS_DIR,
    TASK_DIRNAME,
    append_result,
    load_split,
)

CKPT_ROOT = Path(os.environ.get("LLMSEQ_ROOT", "/root/LLMSeq")) / "checkpoints" / "oc_completion"
HF_CACHE = os.environ.get("HF_HUB_CACHE", "/root/hf_cache")

ARMS = {
    "bert_lora": {"kind": "bert", "model_name": "bert-base-uncased",
                  "peft": True, "micro_batch": 16, "grad_accum": 1},
    "llama_lora": {"kind": "llama", "model_name": "meta-llama/Llama-3.2-1B",
                   "peft": True, "micro_batch": 8, "grad_accum": 2},
    "llama_full": {"kind": "llama", "model_name": "meta-llama/Llama-3.2-1B",
                   "peft": False, "micro_batch": 8, "grad_accum": 2},
}

LR = 2e-5
MAX_EPOCHS = 20
PATIENCE = 3
WARMUP_RATIO = 0.06
MAX_LENGTH = 64


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Replace a checkpoint only after the new file is fully written."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def atomic_json_dump(payload, path: Path) -> None:
    """Write JSON without exposing a partially written destination file."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, path)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_tokenizer(kind: str, model_name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE,
                                        trust_remote_code=True)
    added = 0
    if tok.pad_token is None:
        added = tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok, added


def build_model(kind: str, model_name: str, peft: bool, tokenizer, device,
                dtype):
    from peft import LoraConfig, get_peft_model

    if kind == "bert":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, cache_dir=HF_CACHE,
            trust_remote_code=True)
        model.resize_token_embeddings(len(tokenizer))
        if peft:
            cfg = LoraConfig(r=8, lora_alpha=16,
                             target_modules=["query", "key", "value"],
                             lora_dropout=0.1, bias="none", task_type="SEQ_CLS")
            model = get_peft_model(model, cfg)
    else:
        from transformers import AutoModelForCausalLM

        from src.experiments.LLM_fraction_experiment import (
            CausalLMWithClassificationHead,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, cache_dir=HF_CACHE,
            tie_word_embeddings=True)
        base.config.pad_token_id = tokenizer.pad_token_id
        model = CausalLMWithClassificationHead(base, num_classes=2)
        if peft:
            cfg = LoraConfig(r=8, lora_alpha=16,
                             target_modules=["q_proj", "k_proj", "v_proj",
                                             "o_proj"],
                             lora_dropout=0.1, bias="none",
                             task_type="CAUSAL_LM")
            model.backbone = get_peft_model(model.backbone, cfg)
        model.resize_token_embeddings(len(tokenizer))
    return model.to(device)


def build_texts(kind: str, seqs):
    if kind == "bert":
        from src.experiments.BERT_fraction_experiment import (
            standard_narrative_prompt,
        )
    else:
        from src.experiments.LLM_fraction_experiment import (
            standard_narrative_prompt,
        )
    return [standard_narrative_prompt({"Sequences": s}) for s in seqs]


def build_dataset(kind: str, texts, labels, tokenizer, max_length):
    if kind == "bert":
        from src.experiments.BERT_fraction_experiment import (
            SequenceClassificationDataset,
        )
        return SequenceClassificationDataset(texts, labels, tokenizer,
                                             max_length=max_length)
    from src.experiments.LLM_fraction_experiment import TemporalCausalDataset
    return TemporalCausalDataset(texts, labels, tokenizer,
                                 max_length=max_length)


def forward_batch(kind: str, model, batch, device, with_loss: bool):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if kind == "bert":
        labels = batch["labels"].to(device) if with_loss else None
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    labels=labels)
        return out.loss if with_loss else None, out.logits
    lm_labels = batch["labels"].to(device) if with_loss else None
    outcome = batch["outcome_labels"].to(device) if with_loss else None
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                labels=lm_labels, outcome_labels=outcome)
    return (out["loss"] if with_loss else None), out["logits"]


@torch.no_grad()
def evaluate(kind, model, loader, device):
    model.eval()
    losses, n_loss, probs, labels = [], 0, [], []
    for batch in loader:
        loss, logits = forward_batch(kind, model, batch, device, True)
        bs = batch["input_ids"].shape[0]
        losses.append(loss.item() * bs)
        n_loss += bs
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        probs.extend(p.cpu().numpy())
        key = "labels" if kind == "bert" else "outcome_labels"
        labels.extend(batch[key].numpy())
    probs, labels = np.asarray(probs), np.asarray(labels)
    preds = (probs >= 0.5).astype(int)
    return {
        "loss": float(np.sum(losses) / n_loss),
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.5,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "probs": probs, "labels": labels,
    }


class HFScorer:
    """score_fn interface: sequence strings -> (N, 2) classification logits."""

    def __init__(self, kind, model, tokenizer, max_length, device,
                 batch_size=64):
        self.kind, self.model = kind, model
        self.tokenizer, self.max_length = tokenizer, max_length
        self.device, self.batch_size = device, batch_size

    @torch.no_grad()
    def __call__(self, seqs):
        self.model.eval()
        texts = build_texts(self.kind, seqs)
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(texts[i:i + self.batch_size], truncation=True,
                                 padding="max_length",
                                 max_length=self.max_length,
                                 return_tensors="pt")
            batch = {"input_ids": enc["input_ids"],
                     "attention_mask": enc["attention_mask"]}
            _, logits = forward_batch(self.kind, self.model, batch,
                                      self.device, False)
            out.append(logits.float().cpu().numpy())
        return np.concatenate(out, axis=0)


def load_checkpoint_for_eval(checkpoint: Path, threads: int = 32,
                             batch_size: int = 64):
    torch.set_num_threads(threads)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arm = ck["arm"]
    spec = ARMS[arm]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer, _ = build_tokenizer(spec["kind"], spec["model_name"])
    model = build_model(spec["kind"], spec["model_name"], spec["peft"],
                        tokenizer, device, dtype)
    model.load_state_dict(ck["state_dict"])
    scorer = HFScorer(spec["kind"], model, tokenizer, ck["max_length"], device,
                      batch_size)
    meta = {"model": arm, "seed": ck.get("seed", ""),
            "training_mode": "lora" if spec["peft"] else "full_ft"}
    return scorer, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--task", choices=["ocdet", "ocnoisy"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--diag_pairs", type=int, default=200)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--max_train_rows", type=int, default=0,
                    help="0 = full split (smoke uses the smoke files)")
    ap.add_argument("--results_csv", type=Path,
                    default=RESULTS_DIR / "training_results.csv")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    spec = ARMS[args.arm]
    suffix = "_smoke" if args.smoke else ""

    run_dir = (CKPT_ROOT / TASK_DIRNAME[args.task] / args.arm
               / f"seed_{args.seed}{suffix}")
    run_dir.mkdir(parents=True, exist_ok=True)
    done_marker = run_dir / "done.json"
    if done_marker.exists() and args.resume:
        print(f"[train_hf] {run_dir} already complete - skipping")
        return

    set_seed(args.seed)
    from transformers import get_linear_schedule_with_warmup

    tokenizer, _ = build_tokenizer(spec["kind"], spec["model_name"])
    model = build_model(spec["kind"], spec["model_name"], spec["peft"],
                        tokenizer, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    X_train, y_train, _ = load_split(args.task, "train", args.smoke)
    X_val, y_val, _ = load_split(args.task, "val", args.smoke)
    X_test, y_test, ystar_test = load_split(args.task, "test", args.smoke)
    if args.max_train_rows:
        X_train, y_train = X_train[:args.max_train_rows], y_train[:args.max_train_rows]

    # deviation guard: max_length must not truncate any real content
    longest = max(build_texts(spec["kind"], X_train[:64]), key=len)
    tok_len = len(tokenizer(longest)["input_ids"])
    assert tok_len < args.max_length, \
        f"max_length {args.max_length} would truncate ({tok_len} tokens)"

    texts_train = build_texts(spec["kind"], X_train)
    texts_val = build_texts(spec["kind"], X_val)
    texts_test = build_texts(spec["kind"], X_test)
    ds_train = build_dataset(spec["kind"], texts_train, y_train, tokenizer,
                             args.max_length)
    ds_val = build_dataset(spec["kind"], texts_val, y_val, tokenizer,
                           args.max_length)
    ds_test = build_dataset(spec["kind"], texts_test, y_test, tokenizer,
                            args.max_length)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(ds_train, batch_size=spec["micro_batch"],
                              shuffle=True, num_workers=4, generator=gen)
    val_loader = DataLoader(ds_val, batch_size=args.eval_batch,
                            num_workers=4)
    test_loader = DataLoader(ds_test, batch_size=args.eval_batch,
                             num_workers=4)

    pairs_dir = DATA_ROOT / f"pairs{suffix}"
    diag_path = pairs_dir / f"pairs_two_hole_heldout_val{suffix}.csv"
    diag_pairs = pd.read_csv(diag_path).head(args.diag_pairs)
    scorer = HFScorer(spec["kind"], model, tokenizer, args.max_length, device,
                      args.eval_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    accum = spec["grad_accum"]
    steps_per_epoch = (len(train_loader) + accum - 1) // accum
    total_steps = args.max_epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    start_epoch, best_loss, best_epoch, no_improve = 0, float("inf"), -1, 0
    history, t_prev = [], 0.0
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        expected = {
            "arm": args.arm,
            "task": args.task,
            "seed": args.seed,
            "max_length": args.max_length,
        }
        for key, value in expected.items():
            if ck.get(key) != value:
                raise ValueError(
                    f"resume checkpoint {key}={ck.get(key)!r}, expected {value!r}"
                )
        if ck.get("max_epochs", args.max_epochs) != args.max_epochs:
            raise ValueError(
                "--max_epochs must match the original run when resuming "
                f"({ck.get('max_epochs')} != {args.max_epochs})"
            )
        if ck.get("patience", args.patience) != args.patience:
            raise ValueError(
                "--patience must match the original run when resuming "
                f"({ck.get('patience')} != {args.patience})"
            )
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_loss, best_epoch = ck["best_loss"], ck["best_epoch"]
        no_improve, history = ck["no_improve"], ck["history"]
        t_prev = ck.get("wallclock_s", 0.0)
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        if ck.get("loader_rng") is not None:
            gen.set_state(ck["loader_rng"])
        np.random.set_state(ck["np_rng"])
        random.setstate(ck["py_rng"])
        print(f"[train_hf] resumed at epoch {start_epoch}", flush=True)

    # A timeout can arrive after the epoch checkpoint is committed but before
    # the early-stopping branch below runs.  Preserve the original stopping
    # decision instead of training one extra epoch after requeue.
    if no_improve >= args.patience:
        print(
            f"[train_hf] early-stopping state already reached at epoch "
            f"{start_epoch}; proceeding to final evaluation",
            flush=True,
        )
        start_epoch = args.max_epochs

    config_payload = {
        "arm": args.arm, "task": args.task, "seed": args.seed,
        "model_name": spec["model_name"], "peft": spec["peft"],
        "lora": {"r": 8, "alpha": 16, "dropout": 0.1,
                 "targets": "q,k,v(,o)_proj / query,key,value"}
                if spec["peft"] else None,
        "optimizer": f"AdamW lr={LR}", "micro_batch": spec["micro_batch"],
        "grad_accum": accum, "effective_batch": spec["micro_batch"] * accum,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "early_stopping_metric": "val_loss", "warmup_ratio": WARMUP_RATIO,
        "scheduler": "linear decay", "max_length": args.max_length,
        "dtype": str(dtype), "device": str(device),
        "threshold": "val-F1 sweep 0.10..0.89", "mechanism": MECHANISM,
        "deviations": [
            f"max_length {args.max_length} (code default 512, paper 200; "
            "padding-only change, no truncation - asserted)",
            "fp32 on CPU (code uses bf16 on GPU)" if device.type == "cpu" else None,
        ],
        "smoke": args.smoke, "n_params": n_params,
        "n_trainable": n_trainable,
        "note": "completion diagnostics never influence early stopping, "
                "checkpoint selection or the LR schedule",
    }
    atomic_json_dump(config_payload, run_dir / "config.json")

    t0 = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        ep_loss, seen = 0.0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            loss, _ = forward_batch(spec["kind"], model, batch, device, True)
            (loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            bs = batch["input_ids"].shape[0]
            ep_loss += loss.item() * bs
            seen += bs
        val = evaluate(spec["kind"], model, val_loader, device)
        diag = quick_pair_diag(scorer, diag_pairs)
        rec = {"epoch": epoch + 1, "train_loss": ep_loss / seen,
               "val_loss": val["loss"], "val_auc": val["auc"],
               "val_f1": val["f1"], **diag,
               "seconds": round(t_prev + time.time() - t0, 1)}
        history.append(rec)
        print(f"[train_hf] {args.arm}/{args.task}/s{args.seed} ep{epoch+1} "
              f"train={rec['train_loss']:.4f} val_loss={val['loss']:.4f} "
              f"val_auc={val['auc']:.4f} "
              f"pair_acc={diag['completion_pair_acc']:.4f}", flush=True)

        if val["loss"] < best_loss:
            best_loss, best_epoch, no_improve = val["loss"], epoch + 1, 0
            atomic_torch_save(
                {"state_dict": model.state_dict(), "arm": args.arm,
                 "max_length": args.max_length, "epoch": epoch + 1,
                 "max_epochs": args.max_epochs, "patience": args.patience,
                 "seed": args.seed, "task": args.task,
                 "val": {k: v for k, v in val.items()
                         if k not in ("probs", "labels")}},
                best_path,
            )
        else:
            no_improve += 1

        atomic_torch_save(
            {"state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(),
             "arm": args.arm, "max_length": args.max_length,
             "max_epochs": args.max_epochs, "patience": args.patience,
             "epoch": epoch, "best_loss": best_loss,
             "best_epoch": best_epoch, "no_improve": no_improve,
             "history": history,
             "wallclock_s": t_prev + time.time() - t0,
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": (torch.cuda.get_rng_state_all()
                          if torch.cuda.is_available() else None),
             "loader_rng": gen.get_state(),
             "np_rng": np.random.get_state(),
             "py_rng": random.getstate(),
             "seed": args.seed, "task": args.task},
            last_path,
        )
        atomic_json_dump(history, run_dir / "history.json")

        if no_improve >= args.patience:
            break

    atomic_torch_save(
        {"state_dict": model.state_dict(), "arm": args.arm,
         "max_length": args.max_length, "epoch": len(history),
         "max_epochs": args.max_epochs, "patience": args.patience,
         "seed": args.seed, "task": args.task},
        run_dir / "final.pt",
    )

    # restore best, tune threshold on validation, evaluate on test
    from src.experiments.BERT_fraction_experiment import find_optimal_threshold
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["state_dict"])
    val = evaluate(spec["kind"], model, val_loader, device)
    thr, val_f1_opt = find_optimal_threshold(val["labels"], val["probs"])
    test = evaluate(spec["kind"], model, test_loader, device)
    preds = (test["probs"] >= thr).astype(int)
    auc_latent = (float(roc_auc_score(ystar_test, test["probs"]))
                  if len(set(ystar_test)) > 1 else 0.5)
    wall = round(t_prev + time.time() - t0, 1)

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.arm,
        "training_mode": "lora" if spec["peft"] else "full_ft",
        "task": args.task, "seed": args.seed, "train_rows": len(X_train),
        "epochs_done": len(history), "best_epoch": best_epoch,
        "val_f1": round(val_f1_opt, 6), "val_auc": round(val["auc"], 6),
        "val_loss": round(val["loss"], 6),
        "test_auc_obs": round(test["auc"], 6),
        "test_auc_latent": round(auc_latent, 6),
        "test_f1": round(float(f1_score(test["labels"], preds,
                                        zero_division=0)), 6),
        "test_precision": round(float(precision_score(
            test["labels"], preds, zero_division=0)), 6),
        "test_recall": round(float(recall_score(
            test["labels"], preds, zero_division=0)), 6),
        "threshold": round(float(thr), 4), "n_params": n_params,
        "recipe": f"{args.arm} {spec['model_name']} "
                  f"{'LoRA(r=8,a=16,drop=.1)' if spec['peft'] else 'FULL-FT'} "
                  f"AdamW lr={LR} eff_bs={spec['micro_batch']*accum} "
                  f"warmup=6% linear epochs<={args.max_epochs} "
                  f"patience={args.patience} early=val_loss "
                  f"max_len={args.max_length} dtype={dtype}",
        "wallclock_s": wall, "checkpoint": str(run_dir),
        "smoke": args.smoke, "host": os.uname().nodename,
    }
    append_result(args.results_csv, row)
    atomic_json_dump(
        {"optimal_threshold": float(thr),
         "val_f1_optimal": float(val_f1_opt),
         "val_auc": val["auc"], "val_loss": val["loss"],
         "epochs_done": len(history), "best_epoch": best_epoch,
         "seed": args.seed, "task": args.task,
         "model_name": spec["model_name"], "peft": spec["peft"]},
        run_dir / "meta.json",
    )
    atomic_json_dump(row, done_marker)
    print(f"[train_hf] DONE {args.arm}/{args.task}/s{args.seed}: "
          f"test_auc_obs={test['auc']:.4f} auc_latent={auc_latent:.4f} "
          f"best_epoch={best_epoch} wall={wall}s", flush=True)


if __name__ == "__main__":
    main()
