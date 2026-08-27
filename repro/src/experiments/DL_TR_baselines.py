"""
RNN Baseline Training with Hyperparameter Search - COMPLETE VERSION
====================================================================
Train MLP, CNN1D, Transformer, LSTM, BiLSTM, and GRU models with automatic hyperparameter tuning.

Author: Michele
Date: 2025-12-28
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, 
                            precision_score, recall_score, confusion_matrix)
import argparse
import os
import csv
import json
import random
from datetime import datetime
import warnings
import logging
import sys
warnings.filterwarnings('ignore')

# ============================================
# LOGGING SETUP
# ============================================
def setup_logging(output_dir, number_to_use):
    """Setup detailed logging to both file and console."""
    logs_dir = os.path.join(output_dir, f'dataset_{number_to_use}', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('RNN_Baseline')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler (detailed)
    log_file = os.path.join(logs_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler (less detailed)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

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
    """
    Multi-Layer Perceptron baseline (NO sequential processing).
    Uses positional one-hot encoding to give MLP access to position info.
    If this performs well, the task doesn't require sequential reasoning.
    """
    def __init__(self, vocab_size=26, max_seq_length=30, hidden_dims=[256, 128], 
                 num_classes=2, dropout=0.3):
        super(MLPClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Input: flattened positional one-hot encoding
        input_dim = max_seq_length * vocab_size
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Create positional one-hot encoding
        one_hot = torch.zeros(batch_size, self.max_seq_length, self.vocab_size, 
                             device=x.device)
        
        # Fill in one-hot vectors (ignoring padding)
        for i in range(batch_size):
            for j in range(min(seq_len, self.max_seq_length)):
                if x[i, j] > 0:  # Ignore padding (0)
                    one_hot[i, j, x[i, j]] = 1.0
        
        # Flatten
        flattened = one_hot.view(batch_size, -1)
        
        # Forward through MLP
        logits = self.mlp(flattened)
        
        return logits


class CNN1DClassifier(nn.Module):
    """
    1D CNN for sequence classification.
    Captures local patterns (n-grams) without recurrence.
    """
    def __init__(self, vocab_size=26, embedding_dim=64, num_filters=128, 
                 kernel_sizes=[3, 4, 5], num_classes=2, dropout=0.3):
        super(CNN1DClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple convolutional layers with different kernel sizes
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
        # Embedding
        embedded = self.embedding(x)
        
        # Transpose for Conv1d
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        # Concatenate
        concatenated = torch.cat(conv_outputs, dim=1)
        dropped = self.dropout(concatenated)
        logits = self.fc(dropped)
        
        return logits


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder for sequence classification.
    Uses multi-head self-attention to capture long-range dependencies.
    """
    def __init__(self, vocab_size=26, embedding_dim=64, num_heads=4, 
                 num_layers=2, dim_feedforward=256, num_classes=2, 
                 dropout=0.3, max_seq_length=30):
        super(TransformerClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, embedding_dim)
        )
        
        # Transformer encoder
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
        
        # Embedding
        embedded = self.embedding(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :]
        embedded = embedded + pos_enc
        
        # Create padding mask
        padding_mask = (x == 0)
        
        # Transformer encoding
        encoded = self.transformer(
            embedded,
            src_key_padding_mask=padding_mask
        )
        
        # Global pooling (mean of non-padded positions)
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        sum_encoded = (encoded * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1)
        pooled = sum_encoded / (count + 1e-9)
        
        # Classification
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


class RNNTransformerClassifier(nn.Module):
    """
    Hybrid model: BiLSTM extracts sequential features,
    then Transformer captures global dependencies.
    Combines the sequential inductive bias of RNNs with
    the global attention mechanism of Transformers.
    """
    def __init__(self, vocab_size=26, embedding_dim=64, rnn_hidden_dim=64,
                 num_heads=4, num_transformer_layers=2, dim_feedforward=256,
                 num_classes=2, dropout=0.3, max_seq_length=30):
        super(RNNTransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # BiLSTM layer (output dim = rnn_hidden_dim * 2 for bidirectional)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        rnn_output_dim = rnn_hidden_dim * 2  # bidirectional

        # Project RNN output to transformer dimension if needed
        self.projection = nn.Linear(rnn_output_dim, embedding_dim)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, embedding_dim)
        )

        # Transformer encoder on top of RNN features
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

        # Embedding
        embedded = self.embedding(x)

        # RNN processing - captures sequential patterns
        rnn_out, _ = self.rnn(embedded)  # [batch, seq, rnn_hidden*2]

        # Project to transformer dimension
        projected = self.projection(rnn_out)  # [batch, seq, embedding_dim]

        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :]
        projected = projected + pos_enc

        # Padding mask
        padding_mask = (x == 0)

        # Transformer encoding - captures global dependencies
        encoded = self.transformer(projected, src_key_padding_mask=padding_mask)

        # Global mean pooling (excluding padding)
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-9)

        # Classification
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


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
    'RNNTransformer': RNNTransformerClassifier
}

def create_model(model_name, config):
    """Create model from config."""
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_CLASSES[model_name](**config)

# ============================================
# TRAINING FUNCTION
# ============================================
def train_model(model, model_name, train_loader, val_loader, 
                num_epochs=20, lr=0.001, device='cuda', logger=None):
    """Train a single model with detailed logging."""
    if logger is None:
        logger = logging.getLogger('RNN_Baseline')
    
    logger.info(f"Starting training for {model_name}")
    logger.debug(f"  Device: {device}")
    logger.debug(f"  Epochs: {num_epochs}")
    logger.debug(f"  Learning rate: {lr}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = -1
    best_epoch = 0
    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}  # Initialize with initial state  
    patience_counter = 0
    patience = 5
    
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
        val_metrics = evaluate_model(model, val_loader, device, compute_all_metrics=False)
        
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc']
        }
        training_history.append(epoch_info)
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.debug(f"  Epoch {epoch+1}/{num_epochs}: NEW BEST - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val F1: {val_metrics['f1']:.4f}, "
                        f"Val AUC: {val_metrics['auc']:.4f}")
        else:
            patience_counter += 1
            logger.debug(f"  Epoch {epoch+1}/{num_epochs}: "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val F1: {val_metrics['f1']:.4f}, "
                        f"Val AUC: {val_metrics['auc']:.4f} "
                        f"(patience: {patience_counter}/{patience})")
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch [{epoch+1}/{num_epochs}] "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val F1: {val_metrics['f1']:.4f}, "
                       f"Val AUC: {val_metrics['auc']:.4f}")
        
        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    logger.info(f"Training completed. Best epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}")
    
    return model, best_val_f1, training_history

# ============================================
# EVALUATION FUNCTION
# ============================================
def evaluate_model(model, data_loader, device='cuda', compute_all_metrics=True, logger=None):
    """Evaluate model and return all metrics."""
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_sequences, batch_labels in data_loader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_sequences)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    metrics = {}
    
    if compute_all_metrics:
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['precision'] = precision_score(all_labels, all_preds, zero_division=0)
        metrics['recall'] = recall_score(all_labels, all_preds, zero_division=0)
    
    metrics['f1'] = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    except:
        metrics['auc'] = 0.5
        if logger:
            logger.warning("AUC calculation failed, setting to 0.5")
    
    if compute_all_metrics:
        metrics['predictions'] = all_preds
        metrics['probabilities'] = all_probs
        metrics['labels'] = all_labels
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
    
    return metrics

# ============================================
# HYPERPARAMETER SEARCH
# ============================================
def random_search(model_name, train_loader, val_loader, n_trials=20, 
                 device='cuda', logger=None):
    """Random search for best hyperparameters."""
    if logger is None:
        logger = logging.getLogger('RNN_Baseline')
    
    logger.info(f"="*60)
    logger.info(f"RANDOM SEARCH for {model_name} ({n_trials} trials)")
    logger.info(f"="*60)
    
    # Define search spaces per model type
    if model_name == 'MLP':
        search_space = {
            'hidden_dims': [
                [256, 128],
                [512, 256],
                [256, 128, 64],
                [512, 256, 128]
            ],
            'dropout': [0.2, 0.3, 0.4, 0.5],
            'lr': [0.0005, 0.001, 0.002]
        }
    elif model_name == 'CNN1D':
        search_space = {
            'embedding_dim': [32, 64, 128],
            'num_filters': [64, 128, 256],
            'kernel_sizes': [
                [3, 4, 5],
                [2, 3, 4],
                [3, 4, 5, 6],
                [2, 3, 4, 5]
            ],
            'dropout': [0.2, 0.3, 0.4],
            'lr': [0.0005, 0.001, 0.002]
        }
    elif model_name == 'Transformer':
        search_space = {
            'embedding_dim': [64, 128],
            'num_heads': [4, 8],
            'num_layers': [2, 3, 4],
            'dim_feedforward': [128, 256, 512],
            'dropout': [0.1, 0.2, 0.3],
            'lr': [0.0001, 0.0005, 0.001]
        }
    elif model_name == 'RNNTransformer':
        search_space = {
            'embedding_dim': [32, 64, 128],
            'rnn_hidden_dim': [32, 64, 128],
            'num_heads': [2, 4, 8],
            'num_transformer_layers': [1, 2, 3],
            'dim_feedforward': [128, 256, 512],
            'dropout': [0.1, 0.2, 0.3],
            'lr': [0.0001, 0.0005, 0.001]
        }
    else:  # RNN models
        search_space = {
            'embedding_dim': [16, 32, 64],
            'hidden_dim': [32, 64, 128],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            'lr': [0.0001, 0.0005, 0.001, 0.002]
        }
    
    results = []
    
    for trial in range(n_trials):
        # Sample configuration
        if model_name == 'MLP':
            config = {
                'vocab_size': 26,
                'max_seq_length': 30,
                'hidden_dims': random.choice(search_space['hidden_dims']),
                'num_classes': 2,
                'dropout': random.choice(search_space['dropout'])
            }
        elif model_name == 'CNN1D':
            config = {
                'vocab_size': 26,
                'embedding_dim': random.choice(search_space['embedding_dim']),
                'num_filters': random.choice(search_space['num_filters']),
                'kernel_sizes': random.choice(search_space['kernel_sizes']),
                'num_classes': 2,
                'dropout': random.choice(search_space['dropout'])
            }
        elif model_name == 'Transformer':
            embedding_dim = random.choice(search_space['embedding_dim'])
            num_heads = random.choice(search_space['num_heads'])

            while embedding_dim % num_heads != 0:
                num_heads = random.choice(search_space['num_heads'])

            config = {
                'vocab_size': 26,
                'embedding_dim': embedding_dim,
                'num_heads': num_heads,
                'num_layers': random.choice(search_space['num_layers']),
                'dim_feedforward': random.choice(search_space['dim_feedforward']),
                'num_classes': 2,
                'dropout': random.choice(search_space['dropout']),
                'max_seq_length': 30
            }
        elif model_name == 'RNNTransformer':
            embedding_dim = random.choice(search_space['embedding_dim'])
            num_heads = random.choice(search_space['num_heads'])

            # Ensure embedding_dim is divisible by num_heads
            while embedding_dim % num_heads != 0:
                num_heads = random.choice(search_space['num_heads'])

            config = {
                'vocab_size': 26,
                'embedding_dim': embedding_dim,
                'rnn_hidden_dim': random.choice(search_space['rnn_hidden_dim']),
                'num_heads': num_heads,
                'num_transformer_layers': random.choice(search_space['num_transformer_layers']),
                'dim_feedforward': random.choice(search_space['dim_feedforward']),
                'num_classes': 2,
                'dropout': random.choice(search_space['dropout']),
                'max_seq_length': 30
            }
        else:  # RNN models
            config = {
                'vocab_size': 26,
                'embedding_dim': random.choice(search_space['embedding_dim']),
                'hidden_dim': random.choice(search_space['hidden_dim']),
                'num_layers': random.choice(search_space['num_layers']),
                'num_classes': 2,
                'dropout': random.choice(search_space['dropout'])
            }
            
            if config['hidden_dim'] < config['embedding_dim']:
                config['hidden_dim'] = config['embedding_dim']
        
        lr = random.choice(search_space['lr'])
        
        logger.info(f"\nTrial {trial+1}/{n_trials}:")
        logger.info(f"  Config: {config}")
        logger.info(f"  LR: {lr}")
        
        model = create_model(model_name, config)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.debug(f"  Model parameters: {n_params:,}")
        
        trained_model, val_f1, _ = train_model(
            model, 
            f"{model_name}_trial{trial}", 
            train_loader, 
            val_loader,
            num_epochs=15,
            lr=lr,
            device=device,
            logger=logger
        )
        
        logger.info(f"  → Val F1: {val_f1:.4f}")
        
        results.append({
            'trial': trial,
            'config': config,
            'lr': lr,
            'val_f1': val_f1,
            'n_params': n_params
        })
    
    best_result = max(results, key=lambda x: x['val_f1'])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST CONFIGURATION (Trial {best_result['trial']+1}):")
    logger.info(f"{'='*60}")
    logger.info(f"Val F1: {best_result['val_f1']:.4f}")
    logger.info(f"Parameters: {best_result['n_params']:,}")
    logger.info(f"Config:")
    for k, v in best_result['config'].items():
        if k not in ['vocab_size', 'num_classes']:
            logger.info(f"  {k}: {v}")
    logger.info(f"  lr: {best_result['lr']}")
    
    return best_result['config'], best_result['lr'], results

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
    }
}

DEFAULT_LR = {
    'MLP': 0.001,
    'CNN1D': 0.001,
    'Transformer': 0.0005,
    'LSTM': 0.001,
    'BiLSTM': 0.001,
    'GRU': 0.001,
    'RNNTransformer': 0.0005
}

# ============================================
# DATA PREPROCESSING & LOADING
# ============================================
def preprocess_sequence(seq, separator='\x1f'):
    """Preprocess sequence: split by separator and clean."""
    if pd.isna(seq) or len(str(seq).strip()) == 0:
        return ''
    
    seq_str = str(seq)
    letters = [c.strip() for c in seq_str.split(separator) if c.strip()]
    
    return ''.join(letters)

def shuffle_sequence(seq):
    """Shuffle a sequence (for data leakage testing)."""
    if pd.isna(seq) or len(str(seq).strip()) == 0:
        return seq
    
    seq_list = list(str(seq))
    random.shuffle(seq_list)
    return ''.join(seq_list)

def load_data_from_csv(path_csv, number_to_use, sequence_column='Sequences', 
                       label_column='label', test_rnd=False, 
                       sequence_separator='\x1f', logger=None):
    """Load train/val/test data from CSV files."""
    if logger is None:
        logger = logging.getLogger('RNN_Baseline')
    
    logger.info(f"Loading data from: {path_csv}")
    logger.info(f"Using dataset number: {number_to_use}")
    logger.info(f"Sequence separator: {repr(sequence_separator)}")
    
    # Load data
    X_train = pd.read_csv(
        f"{path_csv}X_train_{number_to_use}.csv", 
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')
    
    y_train = pd.read_csv(
        f"{path_csv}y_train_{number_to_use}.csv", 
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')
    
    X_val = pd.read_csv(
        f"{path_csv}X_val_{number_to_use}.csv", 
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')
    
    y_val = pd.read_csv(
        f"{path_csv}y_val_{number_to_use}.csv", 
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')
    
    X_test = pd.read_csv(
        f"{path_csv}X_test_{number_to_use}.csv", 
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')
    
    y_test = pd.read_csv(
        f"{path_csv}y_test_{number_to_use}.csv", 
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')
    
    logger.debug(f"CSV files loaded successfully")
    
    # Preprocess sequences
    logger.info("Preprocessing sequences (splitting by separator)...")
    
    if len(X_train) > 0:
        sample_raw = X_train[sequence_column].iloc[0]
        logger.debug(f"  Example raw sequence: {repr(sample_raw)[:100]}")
    
    X_train[sequence_column] = X_train[sequence_column].apply(
        lambda x: preprocess_sequence(x, sequence_separator)
    )
    X_val[sequence_column] = X_val[sequence_column].apply(
        lambda x: preprocess_sequence(x, sequence_separator)
    )
    X_test[sequence_column] = X_test[sequence_column].apply(
        lambda x: preprocess_sequence(x, sequence_separator)
    )
    
    if len(X_train) > 0:
        sample_processed = X_train[sequence_column].iloc[0]
        logger.debug(f"  Example processed sequence: {sample_processed[:100]}")
    
    if test_rnd:
        logger.warning("Shuffling each sequence in test set (data leakage check)")
        X_test[sequence_column] = X_test[sequence_column].apply(shuffle_sequence)
    
    # Extract sequences and labels
    train_sequences = X_train[sequence_column].values
    train_labels = y_train[label_column].values

    val_sequences = X_val[sequence_column].values
    val_labels = y_val[label_column].values

    test_sequences = X_test[sequence_column].values
    test_labels = y_test[label_column].values

    # Balanced dataset reduction if number_to_use == "test_just_pair"
    if number_to_use == "test_just_pair":
        logger.info("Reducing dataset by 50% (balanced by class)...")

        def balanced_subsample(sequences, labels, fraction=0.5):
            """Subsample dataset while preserving class balance."""
            indices_class_0 = np.where(labels == 0)[0]
            indices_class_1 = np.where(labels == 1)[0]

            # Sample 50% from each class
            n_samples_0 = int(len(indices_class_0) * fraction)
            n_samples_1 = int(len(indices_class_1) * fraction)

            sampled_indices_0 = np.random.choice(indices_class_0, n_samples_0, replace=False)
            sampled_indices_1 = np.random.choice(indices_class_1, n_samples_1, replace=False)

            # Combine and shuffle
            sampled_indices = np.concatenate([sampled_indices_0, sampled_indices_1])
            np.random.shuffle(sampled_indices)

            return sequences[sampled_indices], labels[sampled_indices]

        train_sequences, train_labels = balanced_subsample(train_sequences, train_labels)
        val_sequences, val_labels = balanced_subsample(val_sequences, val_labels)
        test_sequences, test_labels = balanced_subsample(test_sequences, test_labels)

        logger.info(f"  Dataset reduced to 50% for each split")

    # Data summary
    logger.info(f"Data loaded successfully:")
    logger.info(f"  Train: {len(train_sequences)} sequences")
    logger.info(f"  Val:   {len(val_sequences)} sequences")
    logger.info(f"  Test:  {len(test_sequences)} sequences")
    logger.info(f"  Class distribution (train): "
               f"{np.sum(train_labels == 0)} (label=0) / "
               f"{np.sum(train_labels == 1)} (label=1)")
    
    # Check for issues
    n_empty_train = np.sum([len(str(s).strip()) == 0 for s in train_sequences])
    if n_empty_train > 0:
        logger.warning(f"Found {n_empty_train} empty sequences in train set")
    
    # Sequence length statistics
    seq_lengths = [len(str(s)) for s in train_sequences if len(str(s)) > 0]
    if len(seq_lengths) > 0:
        logger.info(f"  Sequence length: min={min(seq_lengths)}, "
                   f"max={max(seq_lengths)}, "
                   f"mean={np.mean(seq_lengths):.1f}")
    
    return (train_sequences, train_labels, 
            val_sequences, val_labels, 
            test_sequences, test_labels)

# ============================================
# RESULTS SAVING
# ============================================
def save_results(model_name, test_metrics, config, lr, number_to_use, 
                search_results=None, training_history=None, 
                output_dir='results', logger=None):
    """Save all results to files."""
    if logger is None:
        logger = logging.getLogger('RNN_Baseline')
    
    results_dir = os.path.join(output_dir, f'dataset_{number_to_use}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions
    predictions_file = os.path.join(
        results_dir, 
        f'{model_name}_predictions_dataset{number_to_use}.csv'
    )
    with open(predictions_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['ANSWER', 'PREDICTION', 'PROB'])
        writer.writeheader()
        for true_label, pred_label, prob in zip(
            test_metrics['labels'], 
            test_metrics['predictions'], 
            test_metrics['probabilities']
        ):
            writer.writerow({
                'ANSWER': int(true_label),
                'PREDICTION': int(pred_label),
                'PROB': float(prob)
            })
    
    logger.debug(f"Predictions saved to: {predictions_file}")
    
    # Save detailed summary
    summary = {
        'model': model_name,
        'dataset_number': number_to_use,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'auc': float(test_metrics['auc']),
            'confusion_matrix': test_metrics['confusion_matrix']
        },
        'best_config': {k: v for k, v in config.items() 
                       if k not in ['vocab_size', 'num_classes']},
        'best_lr': lr
    }
    
    if search_results is not None:
        summary['search_trials'] = len(search_results)
        summary['all_search_results'] = [
            {
                'trial': r['trial'],
                'val_f1': float(r['val_f1']),
                'n_params': r['n_params'],
                'config': {k: v for k, v in r['config'].items() 
                          if k not in ['vocab_size', 'num_classes']},
                'lr': r['lr']
            }
            for r in search_results
        ]
    
    if training_history is not None:
        summary['training_history'] = training_history
    
    summary_file = os.path.join(
        results_dir, 
        f'{model_name}_summary_dataset{number_to_use}.json'
    )
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved:")
    logger.info(f"  Predictions: {predictions_file}")
    logger.info(f"  Summary:     {summary_file}")
    
    return summary

# ============================================
# MAIN FUNCTION
# ============================================
def main(args):
    # Setup logging
    logger = setup_logging(args.output_dir, args.number_to_use)
    
    logger.info("="*70)
    logger.info("RNN BASELINE TRAINING SCRIPT")
    logger.info("="*70)
    logger.info(f"Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    (train_sequences, train_labels, 
     val_sequences, val_labels, 
     test_sequences, test_labels) = load_data_from_csv(
        path_csv=args.path_csv,
        number_to_use=args.number_to_use,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
        test_rnd=args.test_rnd,
        sequence_separator=args.sequence_separator,
        logger=logger
    )
    
    # Determine max sequence length
    all_sequences = list(train_sequences) + list(val_sequences) + list(test_sequences)
    max_seq_length = max(len(str(s)) for s in all_sequences)
    logger.info(f"Maximum sequence length in dataset: {max_seq_length}")
    
    # Update configs that need max_seq_length
    for model_name in ['MLP', 'Transformer', 'RNNTransformer']:
        if model_name in DEFAULT_CONFIGS:
            DEFAULT_CONFIGS[model_name]['max_seq_length'] = max_seq_length
    
    # Create datasets
    logger.info("Creating PyTorch datasets...")
    train_dataset = LetterSequenceDataset(train_sequences, train_labels)
    val_dataset = LetterSequenceDataset(val_sequences, val_labels)
    test_dataset = LetterSequenceDataset(test_sequences, test_labels)
    
    # Create dataloaders
    logger.info(f"Creating dataloaders (batch_size={args.batch_size})...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches:   {len(val_loader)}")
    logger.info(f"  Test batches:  {len(test_loader)}")
    
    # Determine which models to train
    if args.models == 'all':
        models_to_train = ['MLP', 'CNN1D', 'Transformer', 'LSTM', 'BiLSTM', 'GRU', 'RNNTransformer']
    else:
        models_to_train = args.models.split(',')
    
    logger.info(f"Models to train: {', '.join(models_to_train)}")
    
    # Store all results
    all_results = {}
    
    # Train each model
    for model_name in models_to_train:
        logger.info("\n" + "#"*70)
        logger.info(f"# TRAINING {model_name}")
        logger.info("#"*70)
        
        # Hyperparameter search or use default
        if args.no_search:
            logger.info(f"Using default configuration (no search)")
            best_config = DEFAULT_CONFIGS[model_name]
            best_lr = DEFAULT_LR.get(model_name, DEFAULT_LR['LSTM'])
            search_results = None
        else:
            best_config, best_lr, search_results = random_search(
                model_name, 
                train_loader, 
                val_loader,
                n_trials=args.search_trials,
                device=device,
                logger=logger
            )
        
        # Final training with best config
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL TRAINING with best configuration")
        logger.info(f"{'='*60}")
        
        final_model = create_model(model_name, best_config)
        n_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")
        
        final_model, final_val_f1, training_history = train_model(
            final_model,
            model_name,
            train_loader,
            val_loader,
            num_epochs=args.final_epochs,
            lr=best_lr,
            device=device,
            logger=logger
        )
        
        logger.info(f"Final Validation F1: {final_val_f1:.4f}")
        
        # Evaluate on test set
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SET EVALUATION")
        logger.info(f"{'='*60}")
        
        test_metrics = evaluate_model(
            final_model, 
            test_loader, 
            device=device,
            compute_all_metrics=True,
            logger=logger
        )
        
        logger.info(f"\nTest Results for {model_name}:")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
        logger.info(f"  AUC:       {test_metrics['auc']:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    {test_metrics['confusion_matrix']}")
        
        # Save results
        summary = save_results(
            model_name, 
            test_metrics, 
            best_config, 
            best_lr, 
            args.number_to_use,
            search_results,
            training_history,
            args.output_dir,
            logger
        )
        
        all_results[model_name] = summary
        
        # Save model checkpoint
        if args.save_models:
            models_dir = os.path.join(args.output_dir, f'dataset_{args.number_to_use}', 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_file = os.path.join(models_dir, f'{model_name}_best.pt')
            torch.save(final_model.state_dict(), model_file)
            logger.info(f"  Model checkpoint: {model_file}")
    
    # Final comparison
    logger.info("\n" + "#"*70)
    logger.info("# FINAL COMPARISON OF ALL MODELS")
    logger.info("#"*70 + "\n")
    
    # Create comparison table
    logger.info(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
               f"{'F1':<10} {'AUC':<10}")
    logger.info("-"*75)
    
    for model_name, summary in all_results.items():
        metrics = summary['test_metrics']
        logger.info(f"{model_name:<15} "
                   f"{metrics['accuracy']:<10.4f} "
                   f"{metrics['precision']:<10.4f} "
                   f"{metrics['recall']:<10.4f} "
                   f"{metrics['f1']:<10.4f} "
                   f"{metrics['auc']:<10.4f}")
    
    # Save comparison
    results_dir = os.path.join(args.output_dir, f'dataset_{args.number_to_use}')
    comparison_file = os.path.join(results_dir, f'comparison_dataset{args.number_to_use}.json')
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nComparison saved to: {comparison_file}")
    
    # Find best model
    best_model_name = max(all_results.items(), 
                         key=lambda x: x[1]['test_metrics']['f1'])[0]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST MODEL: {best_model_name}")
    logger.info(f"  F1 Score: {all_results[best_model_name]['test_metrics']['f1']:.4f}")
    logger.info(f"  AUC:      {all_results[best_model_name]['test_metrics']['auc']:.4f}")
    logger.info(f"{'='*60}")
    
    logger.info("\nTraining completed successfully!")

# ============================================
# COMMAND LINE INTERFACE
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train RNN baselines with hyperparameter search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full search for all models
  python rnn_baseline_trainer.py \\
      --path_csv /path/to/data/ \\
      --number_to_use 1 \\
      --search_trials 20
  
  # Quick test with default configs
  python rnn_baseline_trainer.py \\
      --path_csv /path/to/data/ \\
      --number_to_use 1 \\
      --no_search
  
  # Specific models only
  python rnn_baseline_trainer.py \\
      --path_csv /path/to/data/ \\
      --number_to_use 1 \\
      --models MLP,Transformer,LSTM \\
      --search_trials 15
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
    
    parser.add_argument('--test_rnd', action='store_true',
                       help='Shuffle test sequences (data leakage check)')
    
    parser.add_argument('--sequence_separator', type=str, default='\x1f',
                       help='Character used to separate elements in sequences')
    
    # Model arguments
    parser.add_argument('--models', type=str, default='all',
                       help='Models to train: "all", or comma-separated list (e.g., "MLP,LSTM,Transformer")')
    
    # Search arguments
    parser.add_argument('--no_search', action='store_true',
                       help='Skip hyperparameter search, use default configs')
    
    parser.add_argument('--search_trials', type=int, default=20,
                       help='Number of random search trials per model')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    
    parser.add_argument('--final_epochs', type=int, default=30,
                       help='Number of epochs for final training')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output arguments
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained model checkpoints')
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    main(args)