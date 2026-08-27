import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import random
import collections
import sys

from pathlib import Path
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from collections import Counter
from scipy.stats import bernoulli
from itertools import product
from typing import List, Union
from tqdm import tqdm

def plot_ngram_vignette(X_train, y_train, number_csv, max_n=7, top_n=700):
    """
    Create a vignette (faceted) plot showing n-gram distributions from 2-grams to max_n-grams.
    Each subplot shows the top_n most discriminative n-grams.
    """
    
    def extract_ngrams(seq, n):
        """Extract n-grams from a sequence"""
        letters = seq.split('\x1f')
        if len(letters) < n:
            return []
        return ["-".join(letters[i:i+n]) for i in range(len(letters)-n+1)]
    
    # Prepare subplots
    n_grams = list(range(1, max_n + 1))
    n_plots = len(n_grams)
    n_cols = 2  # 2 columns
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, n in enumerate(n_grams):
        ax = axes[idx]
        
        # Count n-grams per class
        ngrams_0 = Counter()
        ngrams_1 = Counter()
        
        for seq, label in zip(pd.DataFrame(X_train)['Sequences'], pd.DataFrame(y_train)['Outcome']):
            ngrams = extract_ngrams(seq, n)
            if label == 0:
                ngrams_0.update(ngrams)
            else:
                ngrams_1.update(ngrams)
        
        # Find most discriminative n-grams
        all_ngrams = set(ngrams_0.keys()) | set(ngrams_1.keys())
        ngram_diff = {}
        
        for ng in all_ngrams:
            freq_0 = ngrams_0.get(ng, 0)
            freq_1 = ngrams_1.get(ng, 0)
            diff = abs(freq_0 - freq_1)
            ngram_diff[ng] = (freq_0, freq_1, diff)
        
        # Sort and take top N
        top_ngrams = sorted(ngram_diff.items(), key=lambda x: x[1][2], reverse=True)[:top_n]
        
        # Prepare data
        ngram_labels = [ng for ng, _ in top_ngrams]
        counts_0 = [data[0] for _, data in top_ngrams]
        counts_1 = [data[1] for _, data in top_ngrams]
        
        # Plot
        x = np.arange(len(ngram_labels))
        width = 0.35
        
        ax.bar(x - width/2, counts_1, width, label='Label=1', alpha=0.8, color='C0')
        ax.bar(x + width/2, counts_0, width, label='Label=0', alpha=0.8, color='C1')
        
        ngram_name = {2: 'Bigrams', 3: 'Trigrams', 4: '4-grams', 5: '5-grams', 6: '6-grams', 7: '7-grams'}
        ax.set_title(f'Top {len(ngram_labels)} {ngram_name.get(n, f"{n}-grams")}', fontsize=12, fontweight='bold')
        
        ax.set_xlabel(f'{ngram_name.get(n, f"{n}-grams")}', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        
        # Don't show x-tick labels for top_n=100 (too crowded)
        ax.set_xticks([])
        
        if idx == 0:  # Legend only on first plot
            ax.legend(fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        
        # Print top 5 for each n-gram
        print(f"\nTop 5 most discriminative {ngram_name.get(n, f'{n}-grams')}:")
        print(f"{f'{n}-gram':<30} {'Count(0)':<12} {'Count(1)':<12} {'Diff':<10}")
        print("-" * 70)
        for ng, (c0, c1, diff) in top_ngrams[:5]:
            print(f"{ng:<30} {c0:<12} {c1:<12} {diff:<10.0f}")
    
    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'N-gram Distribution Analysis (Top {top_n} Most Discriminative) - data: {number_csv}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('ngram_vignette.png', dpi=300, bbox_inches='tight')
    plt.show()


def sample_with_proportion(df, n, pi, label_col='Outcome', random_state=9550):
    """
    Sample n rows from df with specified proportion of positive class.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    n : int
        Total number of samples to return
    pi : float
        Proportion of positive class (label=1), between 0 and 1
    label_col : str
        Name of the label column
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Sampled dataframe with desired proportion
    """
    # Calculate number of samples per class
    n_pos = int(n * pi)
    n_neg = n - n_pos
    
    # Separate by class
    df_pos = df[df[label_col] == 1]
    df_neg = df[df[label_col] == 0]
    
    # Check if we have enough samples
    if len(df_pos) < n_pos:
        raise ValueError(f"Not enough positive samples: need {n_pos}, have {len(df_pos)}")
    if len(df_neg) < n_neg:
        raise ValueError(f"Not enough negative samples: need {n_neg}, have {len(df_neg)}")
    
    # Sample from each class
    sampled_pos = df_pos.sample(n=n_pos, random_state=random_state)
    sampled_neg = df_neg.sample(n=n_neg, random_state=random_state)
    
    # Combine and shuffle
    sampled_df = pd.concat([sampled_pos, sampled_neg], axis=0)
    sampled_df = sampled_df.sample(frac=1, random_state=random_state)  # shuffle
    
    return sampled_df.reset_index(drop=True)
