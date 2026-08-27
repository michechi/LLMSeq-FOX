"""
Attention map analysis for BERT and Llama on the tricky deterministic task.

Trains a model, then extracts and analyzes attention patterns to see if the
model attends to key-letter positions at the correct lag spacing.

Usage:
    # BERT (fast, ~10 min on GPU)
    python -m src.analysis.attention_analysis \
        --path_csv data/simulation/tested/ \
        --number_to_use 9 \
        --output_dir results/attention_bert/ \
        --model_type bert

    # Llama-1B (slower, ~2-3 hours on GPU)
    python -m src.analysis.attention_analysis \
        --path_csv data/simulation/tested/ \
        --number_to_use 9 \
        --output_dir results/attention_llama/ \
        --model_type llama \
        --model_name meta-llama/Llama-3.2-1B \
        --checkpoint_path checkpoints/llama1b_tricky.pt

    # Llama with existing checkpoint (skip training)
    python -m src.analysis.attention_analysis \
        --path_csv data/simulation/tested/ \
        --number_to_use 9 \
        --output_dir results/attention_llama/ \
        --model_type llama \
        --checkpoint_path checkpoints/llama1b_tricky.pt
"""

import os
import gc
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from simulation.do_check_lag import check_lag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

# Key letters and lag from the paper's tricky deterministic setting
KEY_LETTERS = ["W", "D", "Q", "J", "X", "N"]
LAG = 7
SEP = "\x1f"


def set_seed(seed=9550):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standard_narrative_prompt(row, to_split=SEP):
    events = row["Sequences"].split(to_split)
    return f'Sequential events: {" ".join(events)}'


class SeqDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_bert(model, train_loader, val_loader, epochs, lr, patience, device):
    """Quick BERT training with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    best_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())

        auc = roc_auc_score(val_labels, val_preds)
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    logger.info(f"Best validation AUC: {best_auc:.4f}")
    return model


##############################################################################
# LLAMA SUPPORT
##############################################################################

class CausalLMWithClassificationHead(nn.Module):
    """Same wrapper as in LLM_fraction_experiment.py."""

    def __init__(self, backbone_model, num_classes=2):
        super().__init__()
        self.backbone = backbone_model
        self.config = backbone_model.config
        self.num_classes = num_classes

        self.classification_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size // 2, num_classes)
        )
        self.classification_head = self.classification_head.to(dtype=backbone_model.dtype)

    def forward(self, input_ids, attention_mask=None, labels=None,
                outcome_labels=None, output_attentions=False):
        causal_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True
        )

        hidden_states = causal_outputs.hidden_states[-1]
        batch_size = input_ids.shape[0]
        classification_representations = []

        for b in range(batch_size):
            if attention_mask is not None:
                last_token_pos = attention_mask[b].sum().item() - 1
            else:
                last_token_pos = input_ids.shape[1] - 1
            classification_representations.append(hidden_states[b, last_token_pos, :])

        classification_input = torch.stack(classification_representations)
        classification_logits = self.classification_head(classification_input)

        result = {
            'logits': classification_logits,
            'causal_logits': causal_outputs.logits,
            'hidden_states': hidden_states,
        }

        if output_attentions:
            result['attentions'] = causal_outputs.attentions

        if outcome_labels is not None:
            classification_loss = nn.functional.cross_entropy(classification_logits, outcome_labels)
            total_loss = causal_outputs.loss + classification_loss
            result['loss'] = total_loss
        elif labels is not None:
            result['loss'] = causal_outputs.loss

        return result

    def resize_token_embeddings(self, new_num_tokens):
        return self.backbone.resize_token_embeddings(new_num_tokens)


def llama_narrative_prompt(row, to_split=SEP):
    """Prompt format matching LLM_fraction_experiment.py."""
    events = row["Sequences"].split(to_split)
    prompt = f'Sequential events: {" ".join(events)}\n'
    prompt += 'Outcome (0 or 1):'
    return prompt


class CausalSeqDataset(Dataset):
    """Dataset for causal LM with classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        causal_labels = input_ids.clone()
        causal_labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': causal_labels,
            'outcome_labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_llama_model(model_name, tokenizer, peft=True, cache_dir=None):
    """Load Llama with LoRA and classification head."""
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map=None, cache_dir=cache_dir, tie_word_embeddings=True,
        attn_implementation="eager"  # needed for output_attentions=True
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = CausalLMWithClassificationHead(base_model, num_classes=2)
    model.resize_token_embeddings(len(tokenizer))

    if peft:
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.1, bias='none', task_type="CAUSAL_LM"
        )
        model.backbone = get_peft_model(model.backbone, lora_config)
        model.backbone.print_trainable_parameters()

    return model


def train_llama(model, train_loader, val_loader, epochs, lr, patience, device):
    """Training loop for CausalLM with classification head."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    best_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
            outputs['loss'].backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += outputs['loss'].item()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(**batch)
                probs = torch.softmax(outputs['logits'].float(), dim=-1)[:, 1]
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(batch['outcome_labels'].cpu().numpy())

        auc = roc_auc_score(val_labels, val_preds)
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            # Only save trainable params (LoRA + classification head)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                          if v.requires_grad or 'classification_head' in k}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state, strict=False)
    logger.info(f"Best validation AUC: {best_auc:.4f}")
    return model


def extract_attention_llama(model, tokenizer, text, device, max_length=128):
    """Extract attention weights from the CausalLM model."""
    encoding = tokenizer(text, return_tensors='pt', truncation=True,
                         padding='max_length', max_length=max_length)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Build causal labels for the forward pass
    causal_labels = encoding['input_ids'].clone()
    causal_labels[encoding['attention_mask'] == 0] = -100

    model.eval()
    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            labels=causal_labels,
            output_attentions=True
        )

    attentions = [a.squeeze(0).float().cpu().numpy() for a in outputs['attentions']]
    logits = outputs['logits'].float().squeeze(0).cpu().numpy()
    pred = np.argmax(logits)

    return attentions, pred


##############################################################################
# TRUE COMPLIANCE DETECTION
##############################################################################

def is_truly_compliant(seq_str, key_letters, lag, sep=SEP):
    """
    Determine if a sequence is truly compliant (ordered) using deterministic
    labeling, regardless of the noisy label in the stochastic dataset.
    """
    tokens = seq_str.split(sep)
    c_ord = {letter: rank for rank, letter in enumerate(key_letters)}

    lagged = check_lag(tokens, c_ord.keys(), lag)
    if not lagged:
        return False

    def is_ordered_chain(chain):
        for x, y in zip(chain[:-1], chain[1:]):
            if c_ord[x[0]] > c_ord[y[0]]:
                return False
        return True

    if all(isinstance(item, list) for item in lagged):
        return any(is_ordered_chain(chain) for chain in lagged)
    else:
        return is_ordered_chain(lagged)


##############################################################################
# SHARED UTILITIES
##############################################################################

def get_letter_token_positions(text, tokenizer):
    """
    Map each letter in the sequence to its token position in the tokenization.
    Works for both BERT (WordPiece) and Llama (SentencePiece) tokenizers.
    Returns: list of (letter, token_position) for the 20 sequence letters.
    """
    # Extract letters from prompt
    # BERT format: "Sequential events: A B C D ... T"
    # Llama format: "Sequential events: A B C D ... T\nOutcome (0 or 1):"
    parts = text.split("Sequential events: ")
    if len(parts) < 2:
        return []
    after_prefix = parts[1]
    # Remove Llama suffix if present
    if "\nOutcome" in after_prefix:
        after_prefix = after_prefix.split("\nOutcome")[0]
    letters = after_prefix.strip().split()

    # Tokenize the full text
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    letter_positions = []
    letter_idx = 0

    for i, token in enumerate(tokens):
        if letter_idx >= len(letters):
            break

        # Clean token: remove subword markers for both tokenizer types
        # BERT: '##' prefix for subwords
        # Llama/SentencePiece: '▁' (U+2581) prefix for word starts
        clean = token.replace('##', '').replace('▁', '').replace('Ġ', '').upper().strip()

        if clean == letters[letter_idx].upper():
            letter_positions.append((letters[letter_idx], i))
            letter_idx += 1

    return letter_positions


def extract_attention_for_sequence(model, tokenizer, text, device):
    """Extract attention weights for a single sequence."""
    encoding = tokenizer(text, return_tensors='pt', truncation=True,
                         padding='max_length', max_length=128)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**encoding, output_attentions=True)

    # attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
    attentions = [a.squeeze(0).cpu().numpy() for a in outputs.attentions]
    logits = outputs.logits.squeeze(0).cpu().numpy()
    pred = np.argmax(logits)

    return attentions, pred


def analyze_attention_patterns(attentions, letter_positions, key_letters, lag):
    """
    Analyze whether attention concentrates on key-letter positions at lag spacing.

    Returns dict with attention statistics.
    """
    key_set = set(key_letters)
    n_layers = len(attentions)
    n_heads = attentions[0].shape[0]

    # Identify key and non-key letter positions
    key_positions = [(letter, pos) for letter, pos in letter_positions if letter in key_set]
    non_key_positions = [(letter, pos) for letter, pos in letter_positions if letter not in key_set]
    all_letter_pos = [pos for _, pos in letter_positions]

    # Find lag-connected key pairs
    key_pos_set = {pos: letter for letter, pos in letter_positions if letter in key_set}
    lag_pairs = []  # (pos_i, pos_j) where both are key letters and |pos_i - pos_j| corresponds to lag in sequence space

    # We need to map back to sequence positions (not token positions)
    # letter_positions gives us (letter, token_pos) in order of sequence position
    for i, (letter_i, tok_pos_i) in enumerate(letter_positions):
        if letter_i not in key_set:
            continue
        # Check if there's a key letter at sequence position i+lag
        j = i + lag
        if j < len(letter_positions):
            letter_j, tok_pos_j = letter_positions[j]
            if letter_j in key_set:
                lag_pairs.append((tok_pos_i, tok_pos_j))

    results = {
        'per_layer_head': [],  # attention stats per (layer, head)
        'key_to_key_lag': [],  # avg attention between lag-connected key pairs
        'key_to_key_other': [],  # avg attention between non-lag key pairs
        'key_to_nonkey': [],  # avg attention from key to non-key
        'nonkey_to_key': [],  # avg attention from non-key to key
    }

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            attn = attentions[layer_idx][head_idx]  # (seq_len, seq_len)

            # Attention between lag-connected key pairs
            lag_attn = []
            for pos_i, pos_j in lag_pairs:
                lag_attn.append(attn[pos_i, pos_j])
                lag_attn.append(attn[pos_j, pos_i])

            # Attention between key positions (not lag-connected)
            key_tok_positions = [pos for _, pos in key_positions]
            lag_pair_set = set(lag_pairs) | set((b, a) for a, b in lag_pairs)
            other_key_attn = []
            for pi in key_tok_positions:
                for pj in key_tok_positions:
                    if pi != pj and (pi, pj) not in lag_pair_set:
                        other_key_attn.append(attn[pi, pj])

            # Key to non-key attention
            nonkey_tok_positions = [pos for _, pos in non_key_positions]
            k2nk_attn = [attn[pi, pj] for pi in key_tok_positions for pj in nonkey_tok_positions]

            # Non-key to key attention
            nk2k_attn = [attn[pi, pj] for pi in nonkey_tok_positions for pj in key_tok_positions]

            results['per_layer_head'].append({
                'layer': layer_idx,
                'head': head_idx,
                'key_lag_attn': np.mean(lag_attn) if lag_attn else 0,
                'key_other_attn': np.mean(other_key_attn) if other_key_attn else 0,
                'key_to_nonkey_attn': np.mean(k2nk_attn) if k2nk_attn else 0,
                'nonkey_to_key_attn': np.mean(nk2k_attn) if nk2k_attn else 0,
            })

    return results


def plot_attention_heatmap(attentions, letter_positions, key_letters, lag,
                           layer_idx, head_idx, output_path, title_suffix=""):
    """Plot attention heatmap for a specific layer and head, showing only letter positions."""
    key_set = set(key_letters)
    attn = attentions[layer_idx][head_idx]

    # Extract attention submatrix for letter positions only
    letter_tok_positions = [pos for _, pos in letter_positions]
    letters = [letter for letter, _ in letter_positions]
    n = len(letters)

    attn_sub = np.zeros((n, n))
    for i, pi in enumerate(letter_tok_positions):
        for j, pj in enumerate(letter_tok_positions):
            attn_sub[i, j] = attn[pi, pj]

    # Create labels with key letter highlighting
    labels = []
    for i, letter in enumerate(letters):
        if letter in key_set:
            labels.append(f"*{letter}*")
        else:
            labels.append(letter)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn_sub, cmap='Blues', aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Highlight key letter positions
    for i, letter in enumerate(letters):
        if letter in key_set:
            ax.get_xticklabels()[i].set_color('red')
            ax.get_xticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_color('red')
            ax.get_yticklabels()[i].set_fontweight('bold')

    # Mark lag-connected pairs with boxes
    for i in range(n):
        j = i + lag
        if j < n and letters[i] in key_set and letters[j] in key_set:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2,
                                 edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            rect2 = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, linewidth=2,
                                  edgecolor='green', facecolor='none')
            ax.add_patch(rect2)

    ax.set_xlabel("Attending TO (key letters in red, lag pairs boxed in green)")
    ax.set_ylabel("Attending FROM")
    ax.set_title(f"Layer {layer_idx+1}, Head {head_idx+1} {title_suffix}")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_aggregate_analysis(all_results, output_path):
    """Plot aggregate attention analysis across many sequences."""
    # Collect per-layer-head stats
    n_layers = max(r['layer'] for r in all_results[0]['per_layer_head']) + 1
    n_heads = max(r['head'] for r in all_results[0]['per_layer_head']) + 1

    # Average across sequences
    lag_attn = np.zeros((n_layers, n_heads))
    other_attn = np.zeros((n_layers, n_heads))
    k2nk_attn = np.zeros((n_layers, n_heads))
    count = 0

    for seq_results in all_results:
        for r in seq_results['per_layer_head']:
            lag_attn[r['layer'], r['head']] += r['key_lag_attn']
            other_attn[r['layer'], r['head']] += r['key_other_attn']
            k2nk_attn[r['layer'], r['head']] += r['key_to_nonkey_attn']
        count += 1

    lag_attn /= count
    other_attn /= count
    k2nk_attn /= count

    # Ratio: how much more does the model attend to lag-connected key pairs?
    ratio = lag_attn / (other_attn + 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im0 = axes[0].imshow(lag_attn, cmap='Reds', aspect='auto')
    axes[0].set_title("Attention: Key→Key (lag-connected)")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(other_attn, cmap='Blues', aspect='auto')
    axes[1].set_title("Attention: Key→Key (other)")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(ratio, cmap='RdYlGn', aspect='auto',
                          norm=mcolors.LogNorm(vmin=0.5, vmax=max(5, ratio.max())))
    axes[2].set_title("Ratio: Lag-connected / Other key pairs")
    axes[2].set_xlabel("Head")
    axes[2].set_ylabel("Layer")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.suptitle(f"Aggregate attention analysis (n={count} compliant sequences)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_layer_summary(all_results, output_path):
    """Bar plot: average lag-connected vs other attention per layer."""
    n_layers = max(r['layer'] for r in all_results[0]['per_layer_head']) + 1

    lag_by_layer = np.zeros(n_layers)
    other_by_layer = np.zeros(n_layers)
    k2nk_by_layer = np.zeros(n_layers)
    count = 0

    for seq_results in all_results:
        for r in seq_results['per_layer_head']:
            lag_by_layer[r['layer']] += r['key_lag_attn']
            other_by_layer[r['layer']] += r['key_other_attn']
            k2nk_by_layer[r['layer']] += r['key_to_nonkey_attn']
        count += 1

    n_heads = max(r['head'] for r in all_results[0]['per_layer_head']) + 1
    lag_by_layer /= (count * n_heads)
    other_by_layer /= (count * n_heads)
    k2nk_by_layer /= (count * n_heads)

    x = np.arange(n_layers)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, lag_by_layer, width, label='Key→Key (lag-connected)', color='red', alpha=0.7)
    ax.bar(x, other_by_layer, width, label='Key→Key (other)', color='blue', alpha=0.7)
    ax.bar(x + width, k2nk_by_layer, width, label='Key→Non-key', color='gray', alpha=0.7)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Average attention weight')
    ax.set_title('Attention to lag-connected key pairs vs. other positions (averaged across heads & sequences)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i+1}' for i in range(n_layers)])
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_correct_vs_incorrect(correct_results, incorrect_results, output_path,
                              title_suffix=""):
    """Compare lag-attention between correctly and incorrectly classified sequences."""
    n_layers = max(r['layer'] for r in correct_results[0]['per_layer_head']) + 1
    n_heads = max(r['head'] for r in correct_results[0]['per_layer_head']) + 1

    def get_layer_means(results):
        lag_by_layer = np.zeros(n_layers)
        other_by_layer = np.zeros(n_layers)
        count = 0
        for seq_results in results:
            for r in seq_results['per_layer_head']:
                lag_by_layer[r['layer']] += r['key_lag_attn']
                other_by_layer[r['layer']] += r['key_other_attn']
            count += 1
        lag_by_layer /= (count * n_heads)
        other_by_layer /= (count * n_heads)
        return lag_by_layer, other_by_layer

    corr_lag, corr_other = get_layer_means(correct_results)
    incorr_lag, incorr_other = get_layer_means(incorrect_results)

    x = np.arange(n_layers)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5*width, corr_lag, width, label=f'Correct: Key→Key lag (n={len(correct_results)})',
           color='darkred', alpha=0.8)
    ax.bar(x - 0.5*width, corr_other, width, label='Correct: Key→Key other',
           color='darkblue', alpha=0.8)
    ax.bar(x + 0.5*width, incorr_lag, width, label=f'Incorrect: Key→Key lag (n={len(incorrect_results)})',
           color='salmon', alpha=0.8)
    ax.bar(x + 1.5*width, incorr_other, width, label='Incorrect: Key→Key other',
           color='lightblue', alpha=0.8)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Average attention weight')
    base_title = 'Attention to lag-connected key pairs: Correct vs. Incorrect predictions'
    ax.set_title(f"{base_title}{title_suffix}")
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i+1}' for i in range(n_layers)])
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Attention analysis on tricky deterministic task")
    parser.add_argument("--path_csv", type=str, default="data/simulation/tested/")
    parser.add_argument("--number_to_use", type=str, default="9")
    parser.add_argument("--output_dir", type=str, default="results/attention/")
    parser.add_argument("--model_type", type=str, default="bert", choices=["bert", "llama"],
                        help="Model type: 'bert' or 'llama'")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (default: bert-base-uncased or meta-llama/Llama-3.2-1B)")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=9550)
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of compliant sequences to analyze")
    parser.add_argument("--n_heatmaps", type=int, default=3,
                        help="Number of individual sequence heatmaps to save")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to save/load trained model checkpoint")
    parser.add_argument("--train_subsample", type=int, default=50000,
                        help="Subsample training data for speed")

    args = parser.parse_args()

    # Set defaults based on model type
    if args.model_name is None:
        args.model_name = "bert-base-uncased" if args.model_type == "bert" else "meta-llama/Llama-3.2-1B"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}, Model type: {args.model_type}, Model: {args.model_name}")

    is_llama = (args.model_type == "llama")

    # Load data
    na_vals = ['', 'None', 'NaN', 'na', 'nan']
    X_train = pd.read_csv(f"{args.path_csv}X_train_{args.number_to_use}.csv", na_values=na_vals).fillna('')
    y_train = pd.read_csv(f"{args.path_csv}y_train_{args.number_to_use}.csv", na_values=na_vals).fillna('')
    X_test = pd.read_csv(f"{args.path_csv}X_test_{args.number_to_use}.csv", na_values=na_vals).fillna('')
    y_test = pd.read_csv(f"{args.path_csv}y_test_{args.number_to_use}.csv", na_values=na_vals).fillna('')

    # Prepare prompts (different format for BERT vs Llama)
    prompt_fn = llama_narrative_prompt if is_llama else standard_narrative_prompt
    train_texts = X_train.apply(prompt_fn, axis=1).tolist()
    test_texts = X_test.apply(prompt_fn, axis=1).tolist()
    train_labels = y_train["Outcome"].tolist()
    test_labels = y_test["Outcome"].tolist()

    # Subsample training for speed
    if len(train_texts) > args.train_subsample:
        idx = np.random.choice(len(train_texts), args.train_subsample, replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]

    # Split train into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=args.seed, stratify=train_labels
    )
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load or train model
    if is_llama:
        model = load_llama_model(args.model_name, tokenizer, peft=True, cache_dir=args.cache_dir)
        model.classification_head = model.classification_head.to(torch.bfloat16)

        # Load checkpoint if exists
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            logger.info(f"Loading checkpoint from {args.checkpoint_path}")
            state = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(state, strict=False)
        else:
            # Train
            train_dataset = CausalSeqDataset(train_texts, train_labels, tokenizer, args.max_length)
            val_dataset = CausalSeqDataset(val_texts, val_labels, tokenizer, args.max_length)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            model.to(device)
            logger.info("Training Llama...")
            model = train_llama(model, train_loader, val_loader, args.epochs, args.lr, args.patience, device)

            # Save checkpoint
            if args.checkpoint_path:
                os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)
                state = {k: v.cpu() for k, v in model.state_dict().items()
                         if v.requires_grad or 'classification_head' in k}
                torch.save(state, args.checkpoint_path)
                logger.info(f"Checkpoint saved to {args.checkpoint_path}")

        model.to(device)
        extract_fn = lambda text: extract_attention_llama(model, tokenizer, text, device, args.max_length)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2, cache_dir=args.cache_dir
        ).to(device)

        train_dataset = SeqDataset(train_texts, train_labels, tokenizer, args.max_length)
        val_dataset = SeqDataset(val_texts, val_labels, tokenizer, args.max_length)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        logger.info("Training BERT...")
        model = train_bert(model, train_loader, val_loader, args.epochs, args.lr, args.patience, device)
        model.to(device)

        extract_fn = lambda text: extract_attention_for_sequence(model, tokenizer, text, device)

    # Compute true compliance for ALL test sequences (deterministic labeling)
    logger.info("Computing true compliance for test sequences...")
    test_sequences_raw = X_test["Sequences"].tolist()
    true_compliance = [is_truly_compliant(seq, KEY_LETTERS, LAG) for seq in test_sequences_raw]
    n_truly_compliant = sum(true_compliance)
    logger.info(f"Truly compliant: {n_truly_compliant}/{len(true_compliance)} "
                f"({n_truly_compliant/len(true_compliance)*100:.1f}%)")

    # Sample sequences across all 4 categories
    all_indices = list(range(len(test_labels)))
    np.random.shuffle(all_indices)

    # 4 groups: truly_compliant × model_prediction
    # Group names for output
    groups = {
        'true_compliant_pred1': [],     # truly ordered, model says 1 (correct)
        'true_compliant_pred0': [],     # truly ordered, model says 0 (model misses pattern)
        'false_compliant_pred1': [],    # not truly ordered, model says 1 (model fooled by noise label)
        'false_compliant_pred0': [],    # not truly ordered, model says 0
    }
    heatmap_counts = {k: 0 for k in groups}
    analyzed = 0

    # Aggregated correct/incorrect lists across ALL predictions, under two
    # definitions of "correct":
    #   (a) ground truth = is_truly_compliant  -> *_by_truly
    #   (b) noisy training label = test_labels -> *_by_noisy
    correct_by_truly, incorrect_by_truly = [], []
    correct_by_noisy, incorrect_by_noisy = [], []

    logger.info(f"Analyzing attention on test sequences...")

    for idx in all_indices:
        if analyzed >= args.n_samples:
            break

        text = test_texts[idx]
        letter_positions = get_letter_token_positions(text, tokenizer)

        if len(letter_positions) < 10:
            continue

        attentions, pred = extract_fn(text)
        results = analyze_attention_patterns(attentions, letter_positions, KEY_LETTERS, LAG)

        truly_comp = true_compliance[idx]
        pred_label = pred

        if truly_comp and pred_label == 1:
            group = 'true_compliant_pred1'
        elif truly_comp and pred_label == 0:
            group = 'true_compliant_pred0'
        elif not truly_comp and pred_label == 1:
            group = 'false_compliant_pred1'
        else:
            group = 'false_compliant_pred0'

        groups[group].append(results)

        if int(truly_comp) == pred_label:
            correct_by_truly.append(results)
        else:
            incorrect_by_truly.append(results)

        noisy_label = int(test_labels[idx])
        if noisy_label == pred_label:
            correct_by_noisy.append(results)
        else:
            incorrect_by_noisy.append(results)

        analyzed += 1

        # Save heatmaps for each group
        if heatmap_counts[group] < args.n_heatmaps:
            subdir = group
            os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
            for layer_idx in range(len(attentions)):
                head_stats = [r for r in results['per_layer_head'] if r['layer'] == layer_idx]
                best_head = max(head_stats, key=lambda r: r['key_lag_attn'])

                if best_head['key_lag_attn'] > 0.01:
                    plot_attention_heatmap(
                        attentions, letter_positions, KEY_LETTERS, LAG,
                        layer_idx, best_head['head'],
                        os.path.join(args.output_dir, subdir,
                                     f"heatmap_seq{heatmap_counts[group]}_layer{layer_idx+1}_head{best_head['head']+1}.png"),
                        title_suffix=f"({group}, {args.model_type})"
                    )
            heatmap_counts[group] += 1

    # Report group sizes
    logger.info("Group sizes:")
    for name, results_list in groups.items():
        logger.info(f"  {name}: {len(results_list)}")

    # Aggregate plots per group
    for name, results_list in groups.items():
        if len(results_list) >= 5:
            plot_aggregate_analysis(results_list,
                                    os.path.join(args.output_dir, f"aggregate_{name}.png"))
            plot_layer_summary(results_list,
                               os.path.join(args.output_dir, f"layer_summary_{name}.png"))
        else:
            logger.info(f"  {name}: only {len(results_list)} sequences, skipping aggregate")

    # Key comparison: truly compliant (pred=1) vs not truly compliant (pred=1)
    # This shows: does the model attend differently when there IS a real pattern vs not?
    if len(groups['true_compliant_pred1']) >= 5 and len(groups['false_compliant_pred1']) >= 5:
        plot_correct_vs_incorrect(
            groups['true_compliant_pred1'],
            groups['false_compliant_pred1'],
            os.path.join(args.output_dir, "comparison_true_vs_false_compliant_pred1.png")
        )
        logger.info("Saved comparison: truly compliant vs noise-flipped (both predicted as 1)")

    # Also: truly compliant pred=1 vs pred=0
    if len(groups['true_compliant_pred1']) >= 5 and len(groups['true_compliant_pred0']) >= 5:
        plot_correct_vs_incorrect(
            groups['true_compliant_pred1'],
            groups['true_compliant_pred0'],
            os.path.join(args.output_dir, "comparison_true_compliant_correct_vs_missed.png")
        )
        logger.info("Saved comparison: truly compliant correctly vs incorrectly classified")

    # ALL predictions: correct vs incorrect under two definitions of "correct"
    if len(correct_by_truly) >= 5 and len(incorrect_by_truly) >= 5:
        plot_correct_vs_incorrect(
            correct_by_truly,
            incorrect_by_truly,
            os.path.join(args.output_dir,
                         "comparison_all_correct_vs_incorrect_groundtruth.png"),
            title_suffix=" (all preds, ground truth = is_truly_compliant)"
        )
        logger.info(
            f"Saved comparison: all preds, ground-truth correctness "
            f"(correct n={len(correct_by_truly)}, incorrect n={len(incorrect_by_truly)})"
        )

    if len(correct_by_noisy) >= 5 and len(incorrect_by_noisy) >= 5:
        plot_correct_vs_incorrect(
            correct_by_noisy,
            incorrect_by_noisy,
            os.path.join(args.output_dir,
                         "comparison_all_correct_vs_incorrect_noisylabel.png"),
            title_suffix=" (all preds, target = noisy test_labels)"
        )
        logger.info(
            f"Saved comparison: all preds, noisy-label correctness "
            f"(correct n={len(correct_by_noisy)}, incorrect n={len(incorrect_by_noisy)})"
        )

    # Use the largest group with real patterns for main stats
    all_results = groups['true_compliant_pred1'] if groups['true_compliant_pred1'] else \
                  groups['true_compliant_pred0'] if groups['true_compliant_pred0'] else \
                  groups['false_compliant_pred1']

    if not all_results:
        logger.error("No valid results to analyze")
        return

    # Save numerical results
    rows = []
    n_layers = max(r['layer'] for r in all_results[0]['per_layer_head']) + 1
    n_heads = max(r['head'] for r in all_results[0]['per_layer_head']) + 1

    for layer in range(n_layers):
        for head in range(n_heads):
            lag_vals = [seq['per_layer_head'][layer * n_heads + head]['key_lag_attn'] for seq in all_results]
            other_vals = [seq['per_layer_head'][layer * n_heads + head]['key_other_attn'] for seq in all_results]
            k2nk_vals = [seq['per_layer_head'][layer * n_heads + head]['key_to_nonkey_attn'] for seq in all_results]
            rows.append({
                'model_type': args.model_type,
                'layer': layer + 1,
                'head': head + 1,
                'key_lag_attn_mean': np.mean(lag_vals),
                'key_lag_attn_std': np.std(lag_vals),
                'key_other_attn_mean': np.mean(other_vals),
                'key_other_attn_std': np.std(other_vals),
                'key_to_nonkey_mean': np.mean(k2nk_vals),
                'ratio_lag_vs_other': np.mean(lag_vals) / (np.mean(other_vals) + 1e-10),
            })

    df_results = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "attention_stats.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Stats saved to: {csv_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"TOP 10 HEADS BY LAG-ATTENTION RATIO ({args.model_type.upper()})")
    print(f"{'='*70}")
    top = df_results.nlargest(10, 'ratio_lag_vs_other')
    for _, row in top.iterrows():
        print(f"  Layer {int(row['layer']):2d}, Head {int(row['head']):2d}: "
              f"lag={row['key_lag_attn_mean']:.4f}, other={row['key_other_attn_mean']:.4f}, "
              f"ratio={row['ratio_lag_vs_other']:.2f}x")


if __name__ == "__main__":
    main()
