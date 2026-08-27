"""
XGBoost Training Data Fraction Experiment

Tests XGBoost performance across different training data fractions (1%, 10%, 30%, 50%, 75%, 100%)
with three encoding methods:
- basic: positional encoding (1-26)
- llm: LLM embeddings (Llama-3.1-8B)
- categorical: XGBoost native categorical encoding (20 features, memory efficient)
- tfidf: TF-IDF with n-grams (1,2) - captures letter frequencies and bigram patterns

Usage:
    python XGBoost_fraction_experiment.py --csv_to_use 9 --run_basic --run_llm --run_categorical --run_tfidf
"""

import os
import gc
import time
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional imports for LLM encoding
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XGBoost Training Data Fraction Experiment"
    )

    # Data arguments
    parser.add_argument(
        "--csv_to_use", type=str, default="9",
        help="Dataset ID (e.g., 9)"
    )

    parser.add_argument(
        "--path_csv", type=str,
        default="data/simulation/tested/",
        help="Path to CSV files"
    )

    parser.add_argument(
        "--output_dir", type=str,
        default="results/",
        help="Directory to save results"
    )

    # Cache and embedding directories
    parser.add_argument(
        "--cache_dir", type=str,
        default="cache/",
        help="Cache directory for HuggingFace models"
    )

    parser.add_argument(
        "--embedding_dir", type=str,
        default="cache/embeddings/",
        help="Directory to save/load embeddings"
    )

    # LLM parameters
    parser.add_argument(
        "--model_name", type=str,
        default="meta-llama/Llama-3.1-8B",
        help="LLM model name for embeddings"
    )

    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for LLM encoding"
    )

    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Maximum sequence length for tokenizer"
    )

    # Seed and experiment options
    parser.add_argument(
        "--seed", type=int, default=9550,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--run_basic", action="store_true",
        help="Run basic encoding experiments"
    )

    parser.add_argument(
        "--run_llm", action="store_true",
        help="Run LLM encoding experiments"
    )

    parser.add_argument(
        "--run_categorical", action="store_true",
        help="Run categorical (one-hot) encoding experiments"
    )

    parser.add_argument(
        "--run_tfidf", action="store_true",
        help="Run TF-IDF n-gram encoding experiments"
    )

    if args:
        parsed_args = parser.parse_args(args)
    else:
        parsed_args = parser.parse_args()

    # Log arguments
    for arg, value in sorted(vars(parsed_args).items()):
        logger.info(f"Argument {arg}: {value}")

    return parsed_args


def ordinal_encoding_1to26(X_input, desc="Encoding sequences"):
    """
    Ordinal encoding with values 1-26 (A=1, B=2, ..., Z=26).

    Parameters
    ----------
    X_input : pd.DataFrame
        DataFrame with 'Sequences' column containing letter sequences
    desc : str
        Description for progress bar

    Returns
    -------
    pd.DataFrame
        DataFrame with 20 features (pos_0 to pos_19), values 1-26
    """
    n_samples = len(X_input)
    sequence_length = len(X_input['Sequences'].iloc[0].split('\x1f'))
    X = np.zeros((n_samples, sequence_length), dtype=np.int32)

    for i, seq in enumerate(tqdm(X_input['Sequences'], desc=desc)):
        for j, char in enumerate(seq.split('\x1f')):
            # A=1, B=2, ..., Z=26
            X[i, j] = ord(char.upper()) - ord('A') + 1

    # Create DataFrame with position-based column names
    df = pd.DataFrame(X, dtype=np.int32)
    df = df.add_prefix("pos_")
    return df


def categorical_encoding(X_input, desc="Categorical encoding sequences"):
    """
    Categorical encoding for XGBoost native categorical support.

    Each letter at each position is encoded as a categorical feature (0-25).
    XGBoost handles the categorical splits internally (requires enable_categorical=True).

    Parameters
    ----------
    X_input : pd.DataFrame
        DataFrame with 'Sequences' column containing letter sequences
    desc : str
        Description for progress bar

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical features (pos_0, pos_1, ..., pos_19) as category dtype
    """
    n_samples = len(X_input)
    sequence_length = len(X_input['Sequences'].iloc[0].split('\x1f'))

    # Pre-allocate array
    X = np.zeros((n_samples, sequence_length), dtype=np.int8)

    for i, seq in enumerate(tqdm(X_input['Sequences'], desc=desc)):
        for j, char in enumerate(seq.split('\x1f')):
            # A=0, B=1, ..., Z=25
            X[i, j] = ord(char.upper()) - ord('A')

    # Create DataFrame with position-based column names
    df = pd.DataFrame(X, dtype=np.int8)
    df = df.add_prefix("pos_")

    # Convert to categorical dtype for XGBoost native support
    for col in df.columns:
        df[col] = df[col].astype('category')

    return df


def tfidf_encoding(X_input, vectorizer=None, fit=True):
    """
    TF-IDF encoding with character n-grams (1,2).

    Converts sequences to space-separated letters and applies TF-IDF.
    Captures both single letter frequencies and bigram patterns.

    Parameters
    ----------
    X_input : pd.DataFrame
        DataFrame with 'Sequences' column containing letter sequences
    vectorizer : TfidfVectorizer or None
        Pre-fitted vectorizer (for val/test sets). If None, creates new one.
    fit : bool
        If True, fit the vectorizer on the data (for training set).
        If False, only transform (for val/test sets).

    Returns
    -------
    tuple or np.ndarray
        If fit=True: (encoded_array, fitted_vectorizer)
        If fit=False: encoded_array
    """
    # Convert sequences: "A\x1fB\x1fC" -> "A B C"
    texts = X_input['Sequences'].str.replace('\x1f', ' ')

    if fit:
        # Create and fit vectorizer on training data
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 2),  # unigrams + bigrams
            lowercase=False,
            token_pattern=r'[A-Z]'  # only letters
        )
        X_encoded = vectorizer.fit_transform(texts)
        return X_encoded.toarray(), vectorizer
    else:
        # Transform using pre-fitted vectorizer
        X_encoded = vectorizer.transform(texts)
        return X_encoded.toarray()


def standard_narrative_prompt(row, to_split='\x1f', column_name="Sequences"):
    """
    Create a narrative prompt from a sequence.

    Parameters
    ----------
    row : pd.Series
        Row containing the sequence
    to_split : str
        Delimiter character
    column_name : str
        Name of the column containing sequences

    Returns
    -------
    str
        Formatted prompt
    """
    events = row[column_name].split(to_split)
    prompt = f'Sequential events: {" ".join(events)}\n'
    prompt += 'Outcome (0 or 1):'
    return prompt


def get_embeddings(texts, model_name, batch_size, max_length, device, cache_dir):
    """
    Generate embeddings using a HuggingFace model with mean pooling.

    Parameters
    ----------
    texts : list
        List of text strings to embed
    model_name : str
        Name of the HuggingFace model
    batch_size : int
        Batch size for processing
    max_length : int
        Maximum sequence length
    device : str
        Device to use ('cuda' or 'cpu')
    cache_dir : str
        Cache directory for models

    Returns
    -------
    np.ndarray
        Array of embeddings with shape (len(texts), hidden_size)
    """
    logger.info(f"Loading {model_name} on {device}...")
    logger.info(f"Batch size: {batch_size}, Max length: {max_length}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
        trust_remote_code=True
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)

        # Convert to CPU and numpy
        all_embeddings.append(embeddings.cpu().float().numpy())

        # Clear GPU memory
        del outputs, embeddings, inputs
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings)


def subsample_training_data(X_train, y_train, fraction, seed):
    """
    Stratified subsampling of training data to maintain class balance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.DataFrame or np.ndarray
        Training labels
    fraction : float
        Fraction of data to use (0.0 to 1.0)
    seed : int
        Random seed

    Returns
    -------
    tuple
        (X_subset, y_subset) subsampled data
    """
    if fraction >= 1.0:
        return X_train.copy(), y_train.copy()

    # Convert to numpy if needed
    y_train_array = y_train.values.ravel() if isinstance(y_train, pd.DataFrame) else y_train.ravel()

    # Stratified sampling
    X_subset, _, y_subset, _ = train_test_split(
        X_train, y_train_array,
        train_size=fraction,
        stratify=y_train_array,
        random_state=seed
    )

    logger.info(f"Subsampled {len(X_subset)} samples ({fraction*100:.1f}%) from {len(X_train)}")
    if isinstance(y_subset, np.ndarray):
        y_subset = pd.DataFrame(y_subset, columns=['Outcome'])
    class_dist = np.bincount(y_subset.values.ravel().astype(int))
    logger.info(f"Class distribution: {class_dist} ({class_dist[1]/len(y_subset)*100:.2f}% positive)")

    return X_subset, pd.DataFrame(y_subset, columns=['Outcome']) if not isinstance(y_subset, pd.DataFrame) else y_subset


def get_llm_embeddings_for_fraction(
    X_input, fraction, dataset_id, model_name, batch_size,
    max_length, device, cache_dir, embedding_dir, is_subset=True
):
    """
    Generate or load LLM embeddings for a dataset/fraction.

    Parameters
    ----------
    X_input : pd.DataFrame
        Input data
    fraction : float
        Fraction (only used for naming when is_subset=True)
    dataset_id : str
        Dataset ID
    model_name : str
        LLM model name
    batch_size : int
        Batch size for embedding
    max_length : int
        Max sequence length
    device : str
        Device to use
    cache_dir : str
        Cache directory
    embedding_dir : str
        Embedding storage directory
    is_subset : bool
        If True, uses fraction in filename; if False, saves full embeddings

    Returns
    -------
    np.ndarray
        Embeddings array
    """
    # Create filename
    safe_model_name = model_name.replace("/", "_")
    if is_subset:
        fraction_str = f"{fraction:.2f}".replace(".", "")  # 0.01 -> "001"
        embedding_path = f"{embedding_dir}X_train_{dataset_id}_llama_{fraction_str}_embeddings.npy"
    else:
        # For val/test, we don't include fraction
        set_type = "val" if len(X_input) < 100000 else "test"
        embedding_path = f"{embedding_dir}X_{set_type}_{dataset_id}_llama_embeddings.npy"

    # Check if embeddings already exist
    if os.path.exists(embedding_path):
        logger.info(f"Loading cached embeddings from {embedding_path}")
        return np.load(embedding_path)

    # Create prompts
    logger.info("Creating narrative prompts...")
    X_input_copy = X_input.copy()
    X_input_copy['prompt'] = X_input_copy.apply(standard_narrative_prompt, axis=1)

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(X_input_copy)} samples")
    embeddings = get_embeddings(
        X_input_copy['prompt'].tolist(),
        model_name, batch_size, max_length, device, cache_dir
    )

    # Create directory if needed
    os.makedirs(embedding_dir, exist_ok=True)

    # Cache embeddings
    np.save(embedding_path, embeddings)
    logger.info(f"Saved embeddings to {embedding_path}, shape: {embeddings.shape}")

    return embeddings


def run_experiment(args):
    """
    Run the complete XGBoost fraction experiment.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    """
    # Check if LLM is requested but not available
    if args.run_llm and (not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE):
        logger.error("LLM encoding requires torch and transformers. "
                    f"torch available: {TORCH_AVAILABLE}, "
                    f"transformers available: {TRANSFORMERS_AVAILABLE}")
        raise ImportError("Required packages for LLM encoding are not installed")

    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"CUDA available: {TORCH_AVAILABLE and torch.cuda.is_available()}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.embedding_dir, exist_ok=True)

    # Load datasets
    logger.info("Loading datasets...")
    X_train = pd.read_csv(
        f"{args.path_csv}X_train_{args.csv_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')
    y_train = pd.read_csv(
        f"{args.path_csv}y_train_{args.csv_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')

    X_val = pd.read_csv(
        f"{args.path_csv}X_val_{args.csv_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')
    y_val = pd.read_csv(
        f"{args.path_csv}y_val_{args.csv_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')

    X_test = pd.read_csv(
        f"{args.path_csv}X_test_{args.csv_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')
    y_test = pd.read_csv(
        f"{args.path_csv}y_test_{args.csv_to_use}.csv",
        na_values=['', 'None', 'NaN', 'na', 'nan']
    ).fillna('')

    logger.info(f"Train set: {len(X_train)}, Val set: {len(X_val)}, Test set: {len(X_test)}")

    # If csv_to_use is "test_just_pair", sample 50% of all sets (stratified)
    if args.csv_to_use == "test_just_pair":
        logger.info("Detected 'test_just_pair' dataset - sampling 50% of all sets (stratified)...")

        # Stratified sample of train set
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=0.5,
            stratify=y_train['Outcome'].values,
            random_state=args.seed
        )
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        # Stratified sample of val set
        X_val, _, y_val, _ = train_test_split(
            X_val, y_val,
            train_size=0.5,
            stratify=y_val['Outcome'].values,
            random_state=args.seed
        )
        X_val = X_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        # Stratified sample of test set
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test,
            train_size=0.5,
            stratify=y_test['Outcome'].values,
            random_state=args.seed
        )
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        logger.info(f"After 50% sampling - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Log class distributions
        train_pos = y_train['Outcome'].sum() / len(y_train) * 100
        val_pos = y_val['Outcome'].sum() / len(y_val) * 100
        test_pos = y_test['Outcome'].sum() / len(y_test) * 100
        logger.info(f"Class balance (% positive) - Train: {train_pos:.2f}%, Val: {val_pos:.2f}%, Test: {test_pos:.2f}%")

    # Data fractions to test
    fractions = [0.01, 0.10, 0.30, 0.50, 0.75, 1.00]
    results = []

    # Determine which encodings to run
    encodings_to_run = []
    if args.run_basic:
        encodings_to_run.append('basic')
    if args.run_llm:
        encodings_to_run.append('llm')
    if args.run_categorical:
        encodings_to_run.append('categorical')
    if args.run_tfidf:
        encodings_to_run.append('tfidf')

    if not encodings_to_run:
        logger.error("No encoding methods selected. Use --run_basic, --run_llm, --run_categorical, and/or --run_tfidf")
        return

    logger.info(f"Will run: {encodings_to_run}")

    # Experimental loop
    for encoding_method in encodings_to_run:
        for fraction in fractions:
            logger.info("=" * 70)
            logger.info(f"Encoding: {encoding_method.upper()}, Fraction: {fraction*100:.0f}%")
            logger.info("=" * 70)

            start_time = time.time()

            try:
                # 1. Subsample training data (stratified)
                X_train_subset, y_train_subset = subsample_training_data(
                    X_train, y_train, fraction, args.seed
                )

                # 2. Encode data
                if encoding_method == 'basic':
                    logger.info("Encoding with basic method (1-26)...")
                    X_train_encoded = ordinal_encoding_1to26(
                        X_train_subset, desc=f"Encoding train ({fraction*100:.0f}%)"
                    )
                    X_val_encoded = ordinal_encoding_1to26(
                        X_val, desc="Encoding val"
                    )
                    X_test_encoded = ordinal_encoding_1to26(
                        X_test, desc="Encoding test"
                    )

                elif encoding_method == 'llm':
                    logger.info("Encoding with LLM method...")
                    X_train_encoded = get_llm_embeddings_for_fraction(
                        X_train_subset, fraction, args.csv_to_use,
                        args.model_name, args.batch_size, args.max_length,
                        device, args.cache_dir, args.embedding_dir, is_subset=True
                    )
                    X_val_encoded = get_llm_embeddings_for_fraction(
                        X_val, fraction, args.csv_to_use,
                        args.model_name, args.batch_size, args.max_length,
                        device, args.cache_dir, args.embedding_dir, is_subset=False
                    )
                    X_test_encoded = get_llm_embeddings_for_fraction(
                        X_test, fraction, args.csv_to_use,
                        args.model_name, args.batch_size, args.max_length,
                        device, args.cache_dir, args.embedding_dir, is_subset=False
                    )

                elif encoding_method == 'categorical':
                    logger.info("Encoding with XGBoost native categorical method...")
                    X_train_encoded = categorical_encoding(
                        X_train_subset, desc=f"Categorical train ({fraction*100:.0f}%)"
                    )
                    X_val_encoded = categorical_encoding(
                        X_val, desc="Categorical val"
                    )
                    X_test_encoded = categorical_encoding(
                        X_test, desc="Categorical test"
                    )

                elif encoding_method == 'tfidf':
                    logger.info("Encoding with TF-IDF n-gram method...")
                    # Fit vectorizer on training data
                    X_train_encoded, tfidf_vectorizer = tfidf_encoding(
                        X_train_subset, fit=True
                    )
                    # Transform val and test with fitted vectorizer
                    X_val_encoded = tfidf_encoding(
                        X_val, vectorizer=tfidf_vectorizer, fit=False
                    )
                    X_test_encoded = tfidf_encoding(
                        X_test, vectorizer=tfidf_vectorizer, fit=False
                    )
                    logger.info(f"TF-IDF features: {X_train_encoded.shape[1]}")

                # 3. Calculate scale_pos_weight for class imbalance
                y_train_subset_array = y_train_subset.values.ravel()
                n_negative = np.sum(y_train_subset_array == 0)
                n_positive = np.sum(y_train_subset_array == 1)
                scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
                logger.info(f"Scale pos weight: {scale_pos_weight:.4f}")

                # 4. Train XGBoost with fixed parameters
                logger.info("Training XGBoost...")
                xgb_device = 'cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'

                # Enable categorical support for categorical encoding
                use_categorical = (encoding_method == 'categorical')

                xgb_model = XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=1,
                    gamma=0.1,
                    reg_alpha=0.5,
                    reg_lambda=1.5,
                    tree_method='hist',
                    device=xgb_device,
                    scale_pos_weight=scale_pos_weight,
                    random_state=args.seed,
                    verbosity=0,
                    enable_categorical=use_categorical,
                    early_stopping_rounds=10
                )

                # Train with native early stopping
                y_val_array = y_val.values.ravel()
                xgb_model.fit(
                    X_train_encoded,
                    y_train_subset_array,
                    eval_set=[(X_val_encoded, y_val_array)],
                    verbose=False
                )
                logger.info(f"Best iteration: {xgb_model.best_iteration}")

                # 5. Evaluate on full test set
                y_test_array = y_test.values.ravel()
                y_pred_proba = xgb_model.predict_proba(X_test_encoded)[:, 1]
                y_pred = xgb_model.predict(X_test_encoded)

                # 6. Calculate metrics
                elapsed_time = time.time() - start_time
                metrics = {
                    'encoding_method': encoding_method,
                    'fraction': fraction,
                    'train_samples': len(X_train_subset),
                    'test_samples': len(X_test),
                    'auc': roc_auc_score(y_test_array, y_pred_proba),
                    'f1': f1_score(y_test_array, y_pred),
                    'accuracy': accuracy_score(y_test_array, y_pred),
                    'precision': precision_score(y_test_array, y_pred, zero_division=0),
                    'recall': recall_score(y_test_array, y_pred, zero_division=0),
                    'training_time_sec': elapsed_time,
                    'seed': args.seed,
                    'dataset_id': args.csv_to_use,
                    'timestamp': datetime.datetime.now().isoformat()
                }

                results.append(metrics)

                logger.info(f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | "
                           f"Accuracy: {metrics['accuracy']:.4f} | Time: {elapsed_time:.2f}s")

                # Clean up memory
                del X_train_encoded, X_val_encoded, X_test_encoded, xgb_model
                gc.collect()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error in {encoding_method} with fraction {fraction}: {e}")
                raise


    # Save results
    logger.info("=" * 70)
    logger.info("Saving results...")
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.output_dir, 'xgb_fraction_experiment_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

    logger.info("\nFinal Results Summary:")
    logger.info(results_df.to_string())

    return results_df


if __name__ == "__main__":
    args = parse_args()
    results_df = run_experiment(args)
