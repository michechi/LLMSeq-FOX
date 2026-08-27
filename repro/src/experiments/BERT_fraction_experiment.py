"""
BERT Fine-tuning Training Data Fraction Experiment

Tests BERT fine-tuning performance across different training data fractions (1%, 10%, 30%, 50%, 75%, 100%)
using encoder models with sequence classification.

Usage:
    python BERT_fraction_experiment.py --number_to_use 9 --tiny --tiny_type 1M
    python BERT_fraction_experiment.py --number_to_use 9 --model_name bert-base-uncased
"""

import os
import gc
import random
import logging
import datetime
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    BertConfig, get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

torch.set_float32_matmul_precision('high')

# Setup logging with unbuffered output
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def standard_narrative_prompt(row, to_split='\x1f'):
    """Convert row to text format for BERT input."""
    events = row["Sequences"].split(to_split)
    prompt = f'Sequential events: {" ".join(events)}'
    return prompt


def set_seed(seed_value=5550):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = True


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="BERT Fine-tuning Fraction Experiment")

    # Model settings
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of the BERT model from Hugging Face (e.g., bert-base-uncased, "
                             "dmis-lab/biobert-base-cased-v1.2, emilyalsentzer/Bio_ClinicalBERT)")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny model for testing purposes")
    parser.add_argument("--tiny_type", type=str, default="1M",
                        help="Type of tiny model (0.2M, 1M, 5M, 10M, 25M, 50M)")
    parser.add_argument("--peft", action="store_true", help="Use PEFT (LoRA)")
    parser.add_argument("--cold_start", action="store_true",
                        help="Random initialization (no pre-trained weights)")

    # Data paths
    parser.add_argument("--number_to_use", type=str, required=True, help="Dataset ID number")
    parser.add_argument("--path_csv", type=str, default="data/simulation/tested/",
                        help="Path to CSV files (directory containing X_/y_ {train,val,test}_<TAG>.csv)")

    # Output directories
    scratch_dir = os.environ.get("SCRATCH", "/tmp")
    parser.add_argument("--cache_dir", type=str, default=f"{scratch_dir}/cache",
                        help="Cache directory for HuggingFace models")
    parser.add_argument("--output_dir", type=str, default=f"{scratch_dir}/results",
                        help="Directory to save results")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs for training")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--max_length", type=int, default=512, help="Token max length")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--early", type=str, choices=["auc", "loss", "f1"], default="loss",
                        help="Early stopping criterion")
    parser.add_argument("--seed", type=int, default=9550, help="Seed for reproducibility")

    # Experiment settings
    parser.add_argument("--fractions", type=str, default="0.01,0.10,0.30,0.50,0.75,1.0",
                        help="Comma-separated list of training data fractions to test")

    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    # Log arguments
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    return args


class SequenceClassificationDataset(Dataset):
    """Simple dataset for sequence classification with BERT."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_tokenizer(model_name, cache_dir):
    """Load tokenizer for BERT model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    return tokenizer


def load_model(args, tokenizer):
    """Load BERT model for sequence classification."""

    if not args.tiny:
        if not args.cold_start:
            logger.info("Loading pre-trained BERT model (WARM START)")
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=2,
                cache_dir=args.cache_dir,
                trust_remote_code=True
            )
        else:
            logger.info("Loading BERT model (COLD START)")
            config = AutoConfig.from_pretrained(args.model_name, num_labels=2)
            model = AutoModelForSequenceClassification.from_config(config)

        if args.peft:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=['query', 'key', 'value'],
                lora_dropout=0.1,
                bias='none',
                task_type="SEQ_CLS"
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    else:
        # Tiny model for testing
        logger.info(f"Loading TINY BERT model ({args.tiny_type})")

        dimensions = {
            '0.2M': [64, 2, 2, 256],
            '1M': [128, 4, 4, 512],
            '5M': [256, 6, 4, 1024],
            '10M': [384, 6, 6, 1536],
            '25M': [512, 8, 8, 2048],
            '50M': [768, 12, 12, 3072]
        }

        hidden_size, num_layers, num_heads, intermediate_size = dimensions.get(
            args.tiny_type, [128, 4, 4, 512]
        )

        logger.info(f"Tiny BERT config - Hidden: {hidden_size}, Layers: {num_layers}, "
                   f"Heads: {num_heads}, Intermediate: {intermediate_size}")

        tiny_config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=512,
            num_labels=2
        )
        model = AutoModelForSequenceClassification.from_config(tiny_config)

    # Resize embeddings if tokenizer was modified
    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    return model


def subsample_training_data(X_train, y_train, fraction, seed):
    """Stratified subsampling of training data."""
    if fraction >= 1.0:
        return X_train, y_train

    X_subset, _, y_subset, _ = train_test_split(
        X_train, y_train,
        train_size=fraction,
        stratify=y_train['Outcome'],
        random_state=seed
    )
    return X_subset.reset_index(drop=True), y_subset.reset_index(drop=True)


def train_and_evaluate(model, train_loader, val_loader, args, device):
    """Training loop with early stopping."""
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps
    )

    best_auc, best_f1, best_val_loss = 0.0, 0.0, float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            total_train_loss += loss.detach().item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_val_loss += outputs.loss.item()

                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch['labels'].cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_preds_binary = (np.array(val_preds) >= 0.5).astype(int)
        val_f1 = f1_score(val_labels, val_preds_binary, zero_division=0)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Early stopping check
        condition = False
        if args.early == "auc":
            condition = val_auc > best_auc
        elif args.early == "loss":
            condition = avg_val_loss < best_val_loss
        elif args.early == "f1":
            condition = val_f1 > best_f1

        if condition:
            best_auc = val_auc
            best_f1 = val_f1
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"New best model (AUC: {best_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Get final validation predictions for threshold optimization
    model.eval()
    final_val_preds, final_val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            final_val_preds.extend(probs)
            final_val_labels.extend(batch['labels'].cpu().numpy())

    return best_auc, best_f1, best_val_loss, epoch + 1, final_val_preds, final_val_labels


def find_optimal_threshold(labels, preds, thresholds=None):
    """Find threshold that maximizes F1 score on validation data."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.01)

    best_threshold = 0.5
    best_f1 = 0.0

    for thresh in thresholds:
        preds_binary = (np.array(preds) >= thresh).astype(int)
        f1 = f1_score(labels, preds_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold, best_f1


def evaluate_on_test(model, test_loader, device, threshold=0.5):
    """Evaluate model on test set with specified threshold."""
    model.eval()
    test_preds, test_labels_list = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            test_preds.extend(probs)
            test_labels_list.extend(batch['labels'].cpu().numpy())

    test_auc = roc_auc_score(test_labels_list, test_preds)
    test_preds_binary = (np.array(test_preds) >= threshold).astype(int)
    test_f1 = f1_score(test_labels_list, test_preds_binary, zero_division=0)
    test_accuracy = accuracy_score(test_labels_list, test_preds_binary)
    test_precision = precision_score(test_labels_list, test_preds_binary, zero_division=0)
    test_recall = recall_score(test_labels_list, test_preds_binary, zero_division=0)

    return {
        'auc': test_auc,
        'f1': test_f1,
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'threshold': threshold
    }


def run_experiment(args):
    """Main experiment loop over all fractions."""

    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directories
    try:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directories: {e}")
        raise

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)

    # Load data
    logger.info("Loading datasets...")
    X_train = pd.read_csv(f"{args.path_csv}X_train_{args.number_to_use}.csv",
                          na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_train = pd.read_csv(f"{args.path_csv}y_train_{args.number_to_use}.csv",
                          na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    X_val = pd.read_csv(f"{args.path_csv}X_val_{args.number_to_use}.csv",
                        na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_val = pd.read_csv(f"{args.path_csv}y_val_{args.number_to_use}.csv",
                        na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    X_test = pd.read_csv(f"{args.path_csv}X_test_{args.number_to_use}.csv",
                         na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_test = pd.read_csv(f"{args.path_csv}y_test_{args.number_to_use}.csv",
                         na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    logger.info(f"Dataset sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Parse fractions
    fractions = [float(f) for f in args.fractions.split(',')]
    logger.info(f"Testing fractions: {fractions}")

    # Prepare validation and test data (same for all fractions)
    val_texts = X_val.apply(standard_narrative_prompt, axis=1).tolist()
    test_texts = X_test.apply(standard_narrative_prompt, axis=1).tolist()
    val_labels = y_val["Outcome"].tolist()
    test_labels = y_test["Outcome"].tolist()

    val_dataset = SequenceClassificationDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)
    test_dataset = SequenceClassificationDataset(test_texts, test_labels, tokenizer, max_length=args.max_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)

    # Results storage
    results = []

    # Run experiment for each fraction
    for fraction in fractions:
        logger.info("=" * 70)
        logger.info(f"Training with {fraction*100:.0f}% of training data")
        logger.info("=" * 70)

        start_time = datetime.datetime.now()

        try:
            # Subsample training data
            X_train_subset, y_train_subset = subsample_training_data(
                X_train, y_train, fraction, args.seed
            )
            logger.info(f"Training subset size: {len(X_train_subset)}")

            # Prepare training data
            train_texts = X_train_subset.apply(standard_narrative_prompt, axis=1).tolist()
            train_labels = y_train_subset["Outcome"].tolist()

            train_dataset = SequenceClassificationDataset(
                train_texts, train_labels, tokenizer, max_length=args.max_length
            )
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                pin_memory=True, num_workers=0
            )

            # Load fresh model for each fraction
            model = load_model(args, tokenizer)

            # Move model to device
            model = model.to(device)

            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            # Train
            best_auc, best_f1, best_val_loss, epochs_done, val_preds, val_labels_list = train_and_evaluate(
                model, train_loader, val_loader, args, device
            )

            # Find optimal threshold on validation set
            optimal_threshold, val_f1_optimal = find_optimal_threshold(val_labels_list, val_preds)
            logger.info(f"Optimal threshold: {optimal_threshold:.2f} (Val F1: {val_f1_optimal:.4f})")

            # Evaluate on test set with optimal threshold
            test_metrics = evaluate_on_test(model, test_loader, device, threshold=optimal_threshold)

            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

            # Store results
            result = {
                'fraction': fraction,
                'train_samples': len(X_train_subset),
                'val_auc': best_auc,
                'val_f1': best_f1,
                'val_f1_optimal': val_f1_optimal,
                'val_loss': best_val_loss,
                'optimal_threshold': optimal_threshold,
                'test_auc': test_metrics['auc'],
                'test_f1': test_metrics['f1'],
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'epochs': epochs_done,
                'training_time_s': elapsed_time
            }
            results.append(result)

            logger.info(f"Fraction {fraction*100:.0f}% - Test AUC: {test_metrics['auc']:.4f}, "
                       f"Test F1: {test_metrics['f1']:.4f}, Time: {elapsed_time:.1f}s")

            # Cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Error at fraction {fraction}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    results_df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model_name.replace("/", "_")
    tiny_tag = f"_tiny_{args.tiny_type}" if args.tiny else ""
    output_filename = f"bert_fraction_experiment_{model_tag}{tiny_tag}_{args.number_to_use}_{timestamp}.csv"

    output_path = os.path.join(args.output_dir, output_filename)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    args = parse_args()
    results_df = run_experiment(args)
