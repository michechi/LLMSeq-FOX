"""
DL Training Data Fraction Experiment

Tests Deep Learning models performance across different training data fractions
(1%, 10%, 30%, 50%, 75%, 100%) using MLP, CNN1D, Transformer, LSTM, BiLSTM, GRU, RNNTransformer.

Usage:
    python DL_fraction_experiment.py --number_to_use 9 --models LSTM,Transformer --optimal
    python DL_fraction_experiment.py --number_to_use 9 --models all --no_search
"""

import os
import gc
import random
import logging
import datetime
import argparse
import json
import warnings

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                            precision_score, recall_score, confusion_matrix)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

# Training data fractions to test
DEFAULT_FRACTIONS = [0.01, 0.10, 0.30, 0.50, 0.75, 1.0] 


# ============================================
# DATASET CLASS
# ============================================
class LetterSequenceDataset(Dataset):
    """Dataset for letter sequences."""
    def __init__(self, sequences, labels, vocab_size=26):
        self.sequences = sequences.tolist() if hasattr(sequences, 'tolist') else sequences
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else labels
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Convert 'A'->0, 'B'->1, ..., 'Z'->25
        encoded = torch.tensor([ord(c) - ord('A') for c in seq], dtype=torch.long)

        return encoded, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Padding for variable length sequences."""
    sequences, labels = zip(*batch)

    max_len = max(len(seq) for seq in sequences)

    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    sequences_tensor = torch.stack(padded_sequences)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return sequences_tensor, labels_tensor


# ============================================
# MODEL ARCHITECTURES
# ============================================
class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron baseline."""
    def __init__(self, vocab_size=26, max_seq_length=30, hidden_dims=[256, 128],
                 num_classes=2, dropout=0.3):
        super(MLPClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        input_dim = max_seq_length * vocab_size

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, seq_len = x.size()

        one_hot = torch.zeros(batch_size, self.max_seq_length, self.vocab_size,
                             device=x.device)

        for i in range(batch_size):
            for j in range(min(seq_len, self.max_seq_length)):
                if x[i, j] > 0:
                    one_hot[i, j, x[i, j]] = 1.0

        flattened = one_hot.view(batch_size, -1)
        logits = self.mlp(flattened)

        return logits


class CNN1DClassifier(nn.Module):
    """1D CNN for sequence classification."""
    def __init__(self, vocab_size=26, embedding_dim=64, num_filters=128,
                 kernel_sizes=[3, 4, 5], num_classes=2, dropout=0.3):
        super(CNN1DClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)

        concatenated = torch.cat(conv_outputs, dim=1)
        dropped = self.dropout(concatenated)
        logits = self.fc(dropped)

        return logits


class TransformerClassifier(nn.Module):
    """Transformer Encoder for sequence classification."""
    def __init__(self, vocab_size=26, embedding_dim=64, num_heads=4,
                 num_layers=2, dim_feedforward=256, num_classes=2,
                 dropout=0.3, max_seq_length=30):
        super(TransformerClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, embedding_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.size()

        embedded = self.embedding(x)

        pos_enc = self.pos_encoding[:, :seq_len, :]
        embedded = embedded + pos_enc

        padding_mask = (x == 0)

        encoded = self.transformer(
            embedded,
            src_key_padding_mask=padding_mask
        )

        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        sum_encoded = (encoded * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1)
        pooled = sum_encoded / (count + 1e-9)

        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        return logits


class LSTMClassifier(nn.Module):
    """Standard LSTM classifier."""
    def __init__(self, vocab_size=26, embedding_dim=32, hidden_dim=64,
                 num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        lstm_out, (h_n, c_n) = self.lstm(embedded, (h0, c0))
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        logits = self.fc(out)

        return logits


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier."""
    def __init__(self, vocab_size=26, embedding_dim=32, hidden_dim=64,
                 num_layers=2, num_classes=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)

        lstm_out, (h_n, c_n) = self.lstm(embedded, (h0, c0))

        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        out = self.dropout(last_hidden)
        logits = self.fc(out)

        return logits


class GRUClassifier(nn.Module):
    """GRU classifier."""
    def __init__(self, vocab_size=26, embedding_dim=32, hidden_dim=64,
                 num_layers=2, num_classes=2, dropout=0.3):
        super(GRUClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        gru_out, h_n = self.gru(embedded, h0)

        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        logits = self.fc(out)

        return logits


class MambaClassifier(nn.Module):
    """
    Mamba (Selective State Space Model) classifier.
    Uses selective state-space layers instead of attention or recurrence.
    Requires: pip install mamba-ssm causal-conv1d
    """
    def __init__(self, vocab_size=26, embedding_dim=64, d_state=16,
                 d_conv=4, expand=2, num_layers=2, num_classes=2,
                 dropout=0.3, max_seq_length=30):
        super(MambaClassifier, self).__init__()

        from mamba_ssm import Mamba

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=embedding_dim, d_state=d_state,
                  d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)

        h = embedded
        for mamba, norm in zip(self.mamba_layers, self.norms):
            h = h + mamba(norm(h))  # pre-norm + residual

        h = self.final_norm(h)

        # Mean pooling excluding padding
        padding_mask = (x != 0).unsqueeze(-1).float()
        pooled = (h * padding_mask).sum(dim=1) / (padding_mask.sum(dim=1) + 1e-9)

        pooled = self.dropout(pooled)
        return self.fc(pooled)


class RNNTransformerClassifier(nn.Module):
    """Hybrid model: BiLSTM + Transformer."""
    def __init__(self, vocab_size=26, embedding_dim=64, rnn_hidden_dim=64,
                 num_heads=4, num_transformer_layers=2, dim_feedforward=256,
                 num_classes=2, dropout=0.3, max_seq_length=30):
        super(RNNTransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        rnn_output_dim = rnn_hidden_dim * 2

        self.projection = nn.Linear(rnn_output_dim, embedding_dim)

        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, embedding_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.size()

        embedded = self.embedding(x)

        rnn_out, _ = self.rnn(embedded)

        projected = self.projection(rnn_out)

        pos_enc = self.pos_encoding[:, :seq_len, :]
        projected = projected + pos_enc

        padding_mask = (x == 0)

        encoded = self.transformer(projected, src_key_padding_mask=padding_mask)

        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)

        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


# ============================================
# MODEL FACTORY
# ============================================
MODEL_CLASSES = {
    'MLP': MLPClassifier,
    'CNN1D': CNN1DClassifier,
    'Transformer': TransformerClassifier,
    'LSTM': LSTMClassifier,
    'BiLSTM': BiLSTMClassifier,
    'GRU': GRUClassifier,
    'RNNTransformer': RNNTransformerClassifier,
    'Mamba': MambaClassifier
}


def create_model(model_name, config):
    """Create model from config."""
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_CLASSES[model_name](**config)


# ============================================
# DEFAULT CONFIGURATIONS
# ============================================
DEFAULT_CONFIGS = {
    'MLP': {
        'vocab_size': 26,
        'max_seq_length': 30,
        'hidden_dims': [256, 128],
        'num_classes': 2,
        'dropout': 0.3
    },
    'CNN1D': {
        'vocab_size': 26,
        'embedding_dim': 64,
        'num_filters': 128,
        'kernel_sizes': [3, 4, 5],
        'num_classes': 2,
        'dropout': 0.3
    },
    'Transformer': {
        'vocab_size': 26,
        'embedding_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'num_classes': 2,
        'dropout': 0.3,
        'max_seq_length': 30
    },
    'LSTM': {
        'vocab_size': 26,
        'embedding_dim': 32,
        'hidden_dim': 64,
        'num_layers': 2,
        'num_classes': 2,
        'dropout': 0.3
    },
    'BiLSTM': {
        'vocab_size': 26,
        'embedding_dim': 32,
        'hidden_dim': 64,
        'num_layers': 2,
        'num_classes': 2,
        'dropout': 0.3
    },
    'GRU': {
        'vocab_size': 26,
        'embedding_dim': 32,
        'hidden_dim': 64,
        'num_layers': 2,
        'num_classes': 2,
        'dropout': 0.3
    },
    'RNNTransformer': {
        'vocab_size': 26,
        'embedding_dim': 64,
        'rnn_hidden_dim': 64,
        'num_heads': 4,
        'num_transformer_layers': 2,
        'dim_feedforward': 256,
        'num_classes': 2,
        'dropout': 0.3,
        'max_seq_length': 30
    },
    'Mamba': {
        'vocab_size': 26,
        'embedding_dim': 64,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'num_layers': 2,
        'num_classes': 2,
        'dropout': 0.3,
        'max_seq_length': 30
    }
}

DEFAULT_LR = {
    'MLP': 0.001,
    'CNN1D': 0.001,
    'Transformer': 0.0005,
    'LSTM': 0.001,
    'BiLSTM': 0.001,
    'GRU': 0.001,
    'RNNTransformer': 0.0005,
    'Mamba': 0.001
}

# ============================================
# OPTIMAL CONFIGURATIONS (tuned hyperparameters)
# ============================================
OPTIMAL_CONFIGS = {
    'MLP': {
        'vocab_size': 26,
        'max_seq_length': 30,
        'hidden_dims': [512, 256, 128],
        'num_classes': 2,
        'dropout': 0.3
    },
    'CNN1D': {
        'vocab_size': 26,
        'embedding_dim': 32,
        'num_filters': 128,
        'kernel_sizes': [3, 4, 5, 6],
        'num_classes': 2,
        'dropout': 0.2
    },
    'Transformer': {
        'vocab_size': 26,
        'embedding_dim': 64,
        'num_heads': 8,
        'num_layers': 2,
        'dim_feedforward': 512,
        'num_classes': 2,
        'dropout': 0.3,
        'max_seq_length': 30
    },
   'LSTM': {
        'vocab_size': 26,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_layers': 2,
        'num_classes': 2,
        'dropout': 0.5
    },
    'BiLSTM': {
        'vocab_size': 26,
        'embedding_dim': 64,
        'hidden_dim': 64,
        'num_layers': 3,
        'num_classes': 2,
        'dropout': 0.2
    },
    'GRU': {
        'vocab_size': 26,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_layers': 2,
        'num_classes': 2,
        'dropout': 0.3
    },
    'RNNTransformer': {
        'vocab_size': 26,
        'embedding_dim': 64,
        'rnn_hidden_dim': 32,
        'num_heads': 2,
        'num_transformer_layers': 1,
        'dim_feedforward': 256,
        'num_classes': 2,
        'dropout': 0.1,
        'max_seq_length': 30
    },
    'Mamba': {
        'vocab_size': 26,
        'embedding_dim': 64,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'num_layers': 4,
        'num_classes': 2,
        'dropout': 0.2,
        'max_seq_length': 30
    }
}

OPTIMAL_LR = {
    'MLP': 0.001,
    'CNN1D': 0.002,
    'Transformer': 0.001,
    'LSTM': 0.0005,
    'BiLSTM': 0.001,
    'GRU': 0.001,
    'RNNTransformer': 0.001,
    'Mamba': 0.001
}




# ============================================
# DATA PREPROCESSING
# ============================================
def preprocess_sequence(seq, separator='\x1f'):
    """Preprocess sequence: split by separator and clean."""
    if pd.isna(seq) or len(str(seq).strip()) == 0:
        return ''

    seq_str = str(seq)
    letters = [c.strip() for c in seq_str.split(separator) if c.strip()]

    return ''.join(letters)


# ============================================
# UTILITY FUNCTIONS
# ============================================
def set_seed(seed_value=42):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="DL Training Data Fraction Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run fraction experiment with optimal configs for all models
  python DL_fraction_experiment.py \\
      --path_csv /path/to/data/ \\
      --number_to_use 1 \\
      --models all \\
      --optimal

  # Run fraction experiment for specific models
  python DL_fraction_experiment.py \\
      --path_csv /path/to/data/ \\
      --number_to_use 1 \\
      --models LSTM,Transformer,RNNTransformer \\
      --optimal

  # Run with custom fractions
  python DL_fraction_experiment.py \\
      --path_csv /path/to/data/ \\
      --number_to_use 1 \\
      --models BiLSTM \\
      --fractions 0.1,0.5,1.0 \\
      --optimal
        """
    )

    # Data arguments
    parser.add_argument('--path_csv', type=str,
                       default='data/simulation/tested/',
                       help='Path to CSV files directory')

    parser.add_argument('--number_to_use', type=str, required=True,
                       help='Number suffix for CSV files')

    parser.add_argument('--sequence_column', type=str, default='Sequences',
                       help='Name of column containing sequences')

    parser.add_argument('--label_column', type=str, default='label',
                       help='Name of column containing labels')

    parser.add_argument('--sequence_separator', type=str, default='\x1f',
                       help='Character used to separate elements in sequences')

    # Model arguments
    parser.add_argument('--models', type=str, default='all',
                       help='Models to train: "all", or comma-separated list')

    # Config selection
    parser.add_argument('--optimal', action='store_true',
                       help='Use optimal (tuned) configurations')

    parser.add_argument('--no_search', action='store_true',
                       help='Use default configurations (no search)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')

    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs for training')

    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Fraction experiment settings
    parser.add_argument('--fractions', type=str, default='0.01,0.10,0.30,0.50,0.75,1.0',
                       help='Comma-separated list of training data fractions to test')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')

    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    # Log arguments
    logger.info("Arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    return args


def subsample_training_data(sequences, labels, fraction, seed):
    """Stratified subsampling of training data."""
    if fraction >= 1.0:
        return sequences, labels

    X_subset, _, y_subset, _ = train_test_split(
        sequences, labels,
        train_size=fraction,
        stratify=labels,
        random_state=seed
    )
    return np.array(X_subset), np.array(y_subset)


def train_model_with_early_stopping(model, train_loader, val_loader,
                                    num_epochs=30, lr=0.001, patience=5,
                                    device='cuda'):
    """Train model with early stopping and return best metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1
    best_val_auc = 0
    best_val_precision = 0
    best_val_recall = 0
    best_epoch = 0
    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    training_history = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0

        for batch_sequences, batch_labels in train_loader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        val_preds = []
        val_labels_list = []
        val_probs = []

        with torch.no_grad():
            for batch_sequences, batch_labels in val_loader:
                batch_sequences = batch_sequences.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_sequences)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(batch_labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)
        val_precision = precision_score(val_labels_list, val_preds, zero_division=0)
        val_recall = recall_score(val_labels_list, val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels_list, val_probs)
        except Exception:
            val_auc = 0.5

        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'val_precision': val_precision,
            'val_recall': val_recall
        }
        training_history.append(epoch_info)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_epoch = epoch + 1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch [{epoch+1}/{num_epochs}] "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)

    return (model, best_val_f1, best_val_auc, best_val_precision,
            best_val_recall, best_epoch, training_history)


def evaluate_on_test(model, test_loader, device='cuda'):
    """Evaluate model on test set and return all metrics."""
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_sequences, batch_labels in test_loader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_sequences)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    except Exception:
        metrics['auc'] = 0.5

    return metrics


def load_data(args):
    """Load train/val/test data from CSV files."""
    logger.info(f"Loading data from: {args.path_csv}")

    X_train = pd.read_csv(
        f"{args.path_csv}X_train_{args.number_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')

    y_train = pd.read_csv(
        f"{args.path_csv}y_train_{args.number_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')

    X_val = pd.read_csv(
        f"{args.path_csv}X_val_{args.number_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')

    y_val = pd.read_csv(
        f"{args.path_csv}y_val_{args.number_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')

    X_test = pd.read_csv(
        f"{args.path_csv}X_test_{args.number_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')

    y_test = pd.read_csv(
        f"{args.path_csv}y_test_{args.number_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')

    # Preprocess sequences
    X_train[args.sequence_column] = X_train[args.sequence_column].apply(
        lambda x: preprocess_sequence(x, args.sequence_separator)
    )
    X_val[args.sequence_column] = X_val[args.sequence_column].apply(
        lambda x: preprocess_sequence(x, args.sequence_separator)
    )
    X_test[args.sequence_column] = X_test[args.sequence_column].apply(
        lambda x: preprocess_sequence(x, args.sequence_separator)
    )

    # Extract sequences and labels
    train_sequences = X_train[args.sequence_column].values
    train_labels = y_train[args.label_column].values

    val_sequences = X_val[args.sequence_column].values
    val_labels = y_val[args.label_column].values

    test_sequences = X_test[args.sequence_column].values
    test_labels = y_test[args.label_column].values

    logger.info(f"Data loaded - Train: {len(train_sequences)}, "
               f"Val: {len(val_sequences)}, Test: {len(test_sequences)}")

    return (train_sequences, train_labels,
            val_sequences, val_labels,
            test_sequences, test_labels)


# ============================================
# MAIN EXPERIMENT
# ============================================
def run_fraction_experiment(args):
    """Main experiment loop over all fractions and models."""

    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    (train_sequences, train_labels,
     val_sequences, val_labels,
     test_sequences, test_labels) = load_data(args)

    # Determine max sequence length
    all_sequences = list(train_sequences) + list(val_sequences) + list(test_sequences)
    max_seq_length = max(len(str(s)) for s in all_sequences)
    logger.info(f"Maximum sequence length: {max_seq_length}")

    # Update configs with max_seq_length
    for model_name in ['MLP', 'Transformer', 'RNNTransformer', 'Mamba']:
        if model_name in DEFAULT_CONFIGS:
            DEFAULT_CONFIGS[model_name]['max_seq_length'] = max_seq_length
        if model_name in OPTIMAL_CONFIGS:
            OPTIMAL_CONFIGS[model_name]['max_seq_length'] = max_seq_length

    # Create validation and test datasets (same for all fractions)
    val_dataset = LetterSequenceDataset(val_sequences, val_labels)
    test_dataset = LetterSequenceDataset(test_sequences, test_labels)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Parse fractions and models
    fractions = [float(f) for f in args.fractions.split(',')]
    logger.info(f"Testing fractions: {fractions}")

    if args.models == 'all':
        models_to_train = ['MLP', 'CNN1D', 'Transformer', 'LSTM', 'BiLSTM', 'GRU', 'RNNTransformer', 'Mamba']
    else:
        models_to_train = args.models.split(',')
    logger.info(f"Models to train: {models_to_train}")

    # Select config source
    if args.optimal:
        config_source = OPTIMAL_CONFIGS
        lr_source = OPTIMAL_LR
        config_name = "OPTIMAL"
    else:
        config_source = DEFAULT_CONFIGS
        lr_source = DEFAULT_LR
        config_name = "DEFAULT"

    logger.info(f"Using {config_name} configurations")

    # Results storage
    all_results = []

    # Run experiment for each model and fraction
    for model_name in models_to_train:
        logger.info("\n" + "=" * 70)
        logger.info(f"MODEL: {model_name}")
        logger.info("=" * 70)

        config = config_source[model_name].copy()
        lr = lr_source.get(model_name, 0.001)

        for fraction in fractions:
            logger.info("-" * 50)
            logger.info(f"Training {model_name} with {fraction*100:.0f}% of training data")
            logger.info("-" * 50)

            start_time = datetime.datetime.now()

            try:
                # Subsample training data
                train_seq_subset, train_labels_subset = subsample_training_data(
                    train_sequences, train_labels, fraction, args.seed
                )
                logger.info(f"Training subset size: {len(train_seq_subset)}")

                # Create training dataset and loader
                train_dataset = LetterSequenceDataset(train_seq_subset, train_labels_subset)
                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size,
                    shuffle=True, collate_fn=collate_fn, num_workers=0
                )

                # Create fresh model
                model = create_model(model_name, config)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                # Train
                (model, best_val_f1, best_val_auc, best_val_precision,
                 best_val_recall, epochs_done, _) = train_model_with_early_stopping(
                    model, train_loader, val_loader,
                    num_epochs=args.epochs, lr=lr, patience=args.patience,
                    device=device
                )

                # Evaluate on test set
                test_metrics = evaluate_on_test(model, test_loader, device=device)

                elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

                # Store results
                result = {
                    'model': model_name,
                    'fraction': fraction,
                    'train_samples': len(train_seq_subset),
                    'n_params': n_params,
                    # Validation metrics
                    'val_f1': best_val_f1,
                    'val_auc': best_val_auc,
                    'val_precision': best_val_precision,
                    'val_recall': best_val_recall,
                    # Test metrics
                    'test_accuracy': test_metrics['accuracy'],
                    'test_precision': test_metrics['precision'],
                    'test_recall': test_metrics['recall'],
                    'test_f1': test_metrics['f1'],
                    'test_auc': test_metrics['auc'],
                    # Confusion matrix components
                    'test_tn': test_metrics['tn'],
                    'test_fp': test_metrics['fp'],
                    'test_fn': test_metrics['fn'],
                    'test_tp': test_metrics['tp'],
                    'confusion_matrix': test_metrics['confusion_matrix'],
                    # Training info
                    'epochs': epochs_done,
                    'training_time_s': elapsed_time
                }
                all_results.append(result)

                logger.info(f"{model_name} @ {fraction*100:.0f}% - "
                           f"Test AUC: {test_metrics['auc']:.4f}, "
                           f"Test F1: {test_metrics['f1']:.4f}, "
                           f"Precision: {test_metrics['precision']:.4f}, "
                           f"Recall: {test_metrics['recall']:.4f}, "
                           f"Time: {elapsed_time:.1f}s")
                logger.info(f"  Confusion Matrix: TN={test_metrics['tn']}, "
                           f"FP={test_metrics['fp']}, FN={test_metrics['fn']}, "
                           f"TP={test_metrics['tp']}")

                # Cleanup
                del model
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                logger.error(f"Error at {model_name} fraction {fraction}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save results to CSV (without confusion_matrix column for cleaner CSV)
    results_for_csv = [{k: v for k, v in r.items() if k != 'confusion_matrix'}
                       for r in all_results]
    results_df = pd.DataFrame(results_for_csv)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"dl_fraction_experiment_{args.number_to_use}_{config_name}_{timestamp}.csv"

    output_path = os.path.join(args.output_dir, output_filename)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    # Save detailed JSON (includes confusion_matrix)
    json_filename = output_filename.replace('.csv', '.json')
    json_path = os.path.join(args.output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': all_results,
            'timestamp': timestamp
        }, f, indent=2)
    logger.info(f"Detailed results saved to: {json_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)

    # Summary table by model
    for model_name in models_to_train:
        model_results = [r for r in all_results if r['model'] == model_name]
        if model_results:
            logger.info(f"\n{model_name}:")
            logger.info(f"{'Fraction':<10} {'Train N':<10} {'Test AUC':<10} "
                       f"{'Test F1':<10} {'Precision':<10} {'Recall':<10}")
            logger.info("-" * 60)
            for r in model_results:
                logger.info(f"{r['fraction']*100:>6.0f}%    "
                           f"{r['train_samples']:<10} "
                           f"{r['test_auc']:<10.4f} "
                           f"{r['test_f1']:<10.4f} "
                           f"{r['test_precision']:<10.4f} "
                           f"{r['test_recall']:<10.4f}")

    return results_df


if __name__ == "__main__":
    args = parse_args()
    results_df = run_fraction_experiment(args)
