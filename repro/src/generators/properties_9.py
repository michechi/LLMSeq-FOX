import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import random
import collections

from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from collections import Counter
from scipy.stats import bernoulli
from itertools import product
from typing import List, Union
from tqdm import tqdm
from src.generators._internals.do_check_lag import check_lag

def rm_all():
    [globals().pop(var) for var in list(globals()) if not var.startswith('_')]

def generate_sequences(
    letters:list, n:int=10, m:int=10_000, replacement:bool=False, seed:int=None, batch_size:int=None, duplicates:bool=False
):
    """
    Generate exactly m unique sequences (rows) of length n where each event is 'Letter' + 'Digit' (A1, Z0, H4, ...).
    - If replacement=False, each *row* has no repeated letters and no repeated digits.
    - Across rows, duplicates are removed; generation continues until m unique rows are collected.
    """
    rng = np.random.default_rng(seed)
    L = np.asarray(letters, dtype='<U8')
    Llen = len(L)

    if not replacement and n > Llen:
        raise ValueError("n must be <= len(letters) when replacement=False")

    # Choose a sensible batch size (oversample a bit to reduce iterations)
    if batch_size is None:
        batch_size = min(10_000, m // 10)

    # Store unique rows via a hash set of compact string keys
    # Use a separator that won't appear in tokens (ASCII Unit Separator)
    SEP = "\x1f"
    seen = set()
    out_rows = []

    def _gen_batch(k):
        """k batch dimension"""
        if replacement:
            print("replacement!\n")
            li = rng.integers(0, Llen, size=(k, n))
        else:
            print("no replacement!\n")
            # Per-row permutations (no repeats within a row)
            l_scores = rng.random((k, Llen))
            li = np.argsort(l_scores, axis=1)[:, :n]
        seq = L[li]  # shape (k, n), dtype '<U...'
        return seq

    while len(out_rows) < m:
        need = m - len(out_rows)
        k = max(batch_size, need)  # at least batch_size to amortize costs
        seq = _gen_batch(k)        # (k, n)

        # Build vectorized keys for dedup: "A|B|..."
        keys = [SEP.join(key) for key in seq.tolist()]
        # Inefficient! (?)
        # for j in range(1, n):
        #     keys = np.char.add(np.char.add(keys, SEP), seq[:, j])

        # Filter rows not seen before
        mask_new = np.fromiter((key not in seen for key in keys), count=k, dtype=bool)
        if not mask_new.any():
            continue

        # new_keys = keys[mask_new] # Now keys it's a list not an array so this won't work
        new_keys = [key for key,to_keep in zip(keys, mask_new) if to_keep]

        out_rows+=new_keys
        for key in new_keys:
            seen.add(key)

    return out_rows

# For parallelization
def worker_generate(args):
    letters, n, m_chunk, replacement, seed = args 
    return generate_sequences(letters, n, m_chunk, replacement, seed)

def assign_outcome_positional(
    seq:str, 
    c_ord:dict, 
    lags:Union[int, List], 
    rnd:bool=False, 
    sep:str="\x1f", 
    already_splitted=False, 
    pr_1 = 1, # No random 
    tolerance=True
    ) -> int:
    """
    function that, given a sequence, says 1 or 0, depending on ordering.
    """
    # test_seq = 'D7\x1fH5\x1fA7\x1fR5\x1fL1\x1fE4\x1fF8\x1fC0\x1fA8\x1fN0' # sequences[0]
    # test_seq_splt = test_seq.split("\x1f")
    # np.random.seed(seed=123456)

    l_keys = c_ord.keys()

    if not already_splitted:
        test_seq_splt = seq.split(sep) # avoiding noising letters
    else: 
        test_seq_splt = seq
    
    # Check if there are key letters separated by lags.
    are_lagged_keys = check_lag(test_seq_splt, l_keys, lags)

    if are_lagged_keys:
        n_seq=len(are_lagged_keys)
        all_ordered = [True]*n_seq
        tol=tolerance # For the cyclic ordering
        for pos, subseq in enumerate(are_lagged_keys):
            # Check whether are ordered
            for x,y in zip(subseq[:-1], subseq[1:]):
                if ((2*c_ord[x[0]])>(2*c_ord[y[0]])):
                        if tol:
                            tol=False
                        else:
                            all_ordered[pos]=False
        if any(all_ordered):
            # If there is at least one true, then there is one ordered sequence=> high probabilities of 1
            pr_to_simulate = pr_1
        else:
            # If there is no true, then there are no one ordered sequence=> low probabilities of 1
            pr_to_simulate = 1-pr_1
    else:
        # If there no keys, then there are no one ordered sequence=> low probabilities of 1
        all_ordered=False
        pr_to_simulate = 1-pr_1 # if test_seq_splt is void then there are no keys so not ordered
        
    if rnd:
        # Stochastics outcome
        outcome = bernoulli.rvs(pr_to_simulate)
    else:
        # Deterministic outcome
        outcome = int(np.where(pr_to_simulate==pr_1, 1, 0))
    
    # Returning more, to be able to inspect results
    results_2_debug = {
        'outcome':[outcome],
        'seq':seq,
        'pr_2_sim':[pr_to_simulate],
        'lagged_keys':[are_lagged_keys],
        'all_ordered':[all_ordered]
    }
    return(results_2_debug)
    
# seq4 = ['A', 'A', 'C', 'D', 'B', 'A', 'Z', 'H', 'C']
# # seqT = ['M','V','D','V','M','T','L','C','C','G','X','J','C','Y','J','B','C','Q','F','M']
# keys = ['A', 'B', 'C']
# c_vocab4 = {w:p for p,w in enumerate(keys,start=0)}
# # check_lag(seqT, ["W", "D", "Q", "J", "U"], 7)
# assign_outcome_positional(seq4, c_vocab4, lags=3, already_splitted=True)

random.seed(959693)

n_events = 30 # More
n_seq = 100_000_000
k =4
n_0s = 250_000
n_1s = 250_000
n_tot = n_0s + n_1s
n_train = n_tot * 0.80 # 80% of n_tot
n_val = n_tot * 0.10
n_test = n_tot * 0.10
generate = False
parallel = True
#sum([n_train, n_test, n_val]) == n_tot

letters = list(string.ascii_uppercase)
# letters_4_key = letters.copy()
# random.shuffle(letters_4_key)
letters_4_key = ["W", "D", "Q", "J", "X", "U"] # Added X
# digits = list(map(str, range(10)))

# random.shuffle(letters_4_key)
# random.shuffle(digits)

c_vocab = {w:p for p,w in enumerate(letters_4_key,start=0)}

# This is sequential
if generate:
    sequences = generate_sequences(letters=letters, n=n_events, m=n_seq, replacement=True)
else:
    # Legacy: load pre-generated sequences from a previous run.
    # Set DATA_DIR_LEGACY env var to override.
    _legacy_dir = os.environ.get("DATA_DIR_LEGACY", "data/simulation")
    sequences = pd.read_csv(f"{_legacy_dir}/X_test_5.csv")["Sequences"].tolist()
    labels = pd.read_csv(f"{_legacy_dir}/y_test_5.csv")["Outcome"].tolist()

    sequences += pd.read_csv(f"{_legacy_dir}/X_train_5.csv")["Sequences"].tolist()
    labels += pd.read_csv(f"{_legacy_dir}/y_train_5.csv")["Outcome"].tolist()

    sequences += pd.read_csv(f"{_legacy_dir}/X_val_5.csv")["Sequences"].tolist()
    labels += pd.read_csv(f"{_legacy_dir}/y_val_5.csv")["Outcome"].tolist()

    n_seq = len(sequences)


# # Parallel version
# n_cores = cpu_count()-1
# rng = np.random.default_rng(999)
# seeds = rng.integers(0, 2**31, size=n_cores)
# m_per_core = int(n_seq * 1.2 / n_cores)  # 20% oversample
# tasks = [(letters, n_events, m_per_core, True, seed) 
#              for seed in seeds]

# with Pool(processes=n_cores) as pool:
#     results = pool.map(worker_generate, tasks)

# # Deduplicate and trim
# all_sequences = [seq for result in results for seq in result]
# sequences = list(dict.fromkeys(all_sequences))[:n_seq]

set_seq = set(sequences)
n_set_seq = len(set_seq)

if n_seq != n_set_seq:
    print(f"Attention! There are {n_seq-n_set_seq} duplicates!\nRemoving them..")
    sequences = set_seq.copy()
    n_seq = n_set_seq
    del set_seq, n_set_seq
    print("Done!")

def efficient_check_v1(sequences, c_vocab, assign_outcome_positional, tolerance=True, lags=7, rnd=False):
    """Calcola una volta sola e poi usa i risultati"""
    # Calcola UNA SOLA VOLTA per ogni sequenza
    print(f"Tolerance: {tolerance}, lags: {lags}!\n")
    outcomes = []

    df_2_monitor = pd.DataFrame({
        'outcome':[],
        'seq':[],
        'pr_2_sim':[],
        'lagged_keys':[],
        'all_ordered':[]
    })

    for seq in tqdm(sequences):
        df_results = pd.DataFrame(assign_outcome_positional(seq, c_vocab, already_splitted=False, lags=lags, tolerance=tolerance, rnd=rnd))
        df_2_monitor = pd.concat([df_2_monitor, df_results])
        # outcomes += assign_outcome_positional(seq, c_vocab, already_splitted=False, lags=lags, tolerance=tolerance)
    # outcomes = [assign_outcome_positional(seq, c_vocab, already_splitted=False, lags=8) for seq in sequences]
    
    sequences, outcomes = df_2_monitor["seq"], df_2_monitor["outcome"]
    
    # Ora usa zip per associare sequenze ai loro outcomes
    sequences_with_outcomes = list(zip(sequences, outcomes))
    
    # Filtra basandoti sui risultati già calcolati
    which_1s = [seq for seq, outcome in sequences_with_outcomes if outcome == 1]
    which_0s = [seq for seq, outcome in sequences_with_outcomes if outcome == 0]
    
    n_1s = len(which_1s)
    n_0s = len(which_0s)
    
    if n_1s > 0:
        print(f"There are {n_1s} ordered sequences! ({n_1s/len(sequences)*100:.2f}%)")
        
        if (n_0s + n_1s) == len(sequences):
            print("All good!")
        else:
            print("Figures do not add up!")
    else:
        print("No valid sequences found")
    
    return ({
        'valid_sequences': which_1s,
        'invalid_sequences': which_0s,
        'outcomes': outcomes
    }, df_2_monitor)


def process_single_sequence(args):
    """Funzione helper per il multiprocessing"""
    seq, c_vocab, assign_outcome_positional, lags, tolerance, rnd = args
    result = assign_outcome_positional(
        seq, c_vocab, 
        already_splitted=False, 
        lags=lags, 
        tolerance=tolerance, 
        rnd=rnd
    )
    # Converti il dizionario in DataFrame
    return pd.DataFrame(result)

def efficient_check_parallel(sequences, c_vocab, assign_outcome_positional, 
                             tolerance=True, lags=7, rnd=False, n_workers=None):
    """Versione parallelizzata con multiprocessing.Pool"""
    print(f"Tolerance: {tolerance}, lags: {lags}!\n")
    
    # Use all cores-1 if not specified
    if n_workers is None:
        n_workers = cpu_count()
    
    # Prepara gli argomenti per ogni sequenza
    args_list = [
        (seq, c_vocab, assign_outcome_positional, lags, tolerance, rnd) 
        for seq in sequences
    ]
    
    # Parallelize with Pool
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_sequence, args_list), 
            total=len(sequences)
        ))
    
    # Combine all DF
    df_2_monitor = pd.concat(results, ignore_index=True)
    
    sequences, outcomes = df_2_monitor["seq"], df_2_monitor["outcome"]
    sequences_with_outcomes = list(zip(sequences, outcomes))
    
    which_1s = [seq for seq, outcome in sequences_with_outcomes if outcome == 1]
    which_0s = [seq for seq, outcome in sequences_with_outcomes if outcome == 0]
    
    n_1s = len(which_1s)
    n_0s = len(which_0s)
    
    if n_1s > 0:
        print(f"There are {n_1s} ordered sequences! ({n_1s/len(sequences)*100:.2f}%)")
        if (n_0s + n_1s) == len(sequences):
            print("All good!")
        else:
            print("Figures do not add up!")
    else:
        print("No valid sequences found")
    
    return ({
        'valid_sequences': which_1s,
        'invalid_sequences': which_0s,
        'outcomes': outcomes
    }, df_2_monitor)

# Which lags do we prefer?
# lags = [7,4,2] # is valid
lags = 7

if parallel:
    # Parallel:
    n_cores = cpu_count()-1
    
    check, df_2_monitor = efficient_check_parallel(
            sequences=sequences, 
            c_vocab=c_vocab, 
            assign_outcome_positional=assign_outcome_positional, 
            tolerance=False, 
            lags=lags, 
            rnd=True,
            n_workers=n_cores  # oppure None per usare tutti i core
        )
else:
    # Sequential:
    check, df_2_monitor = efficient_check_v1(
        sequences=sequences, 
        c_vocab=c_vocab, 
        assign_outcome_positional=assign_outcome_positional, 
        tolerance=False, 
        lags=lags, 
        rnd=True)

def extract_characters(seq:str, sep:str="\x1f") -> str:
    seq_char = "-".join([x[0] for x in seq.split(sep)])
    return(seq_char)

# Check with reality
if (not generate):
    print(pd.Series(labels).value_counts()/len(labels))

# GRAPHICAL STUFF
# check.keys()
chr_seq_1s=list(map(extract_characters, check['valid_sequences']))
chr_seq_0s=list(map(extract_characters, check['invalid_sequences']))

i = 0
stats_1s, stats_0s = collections.Counter(x[i*2] for x in chr_seq_1s), collections.Counter(x[i*2] for x in chr_seq_0s)

letters_ord = string.ascii_uppercase
counts_1s = [stats_1s.get(letter, 0) for letter in letters_ord]
counts_0s = [stats_0s.get(letter, 0) for letter in letters_ord]

# Plot affiancato
x = np.arange(len(letters_ord))
width = 0.35  # Larghezza barre

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, counts_1s, width, label='Label=1', alpha=0.8)
ax.bar(x + width/2, counts_0s, width, label='Label=0', alpha=0.8)
ax.set_xlabel('Letters')
ax.set_ylabel('Count')
ax.set_title(f'{i+1}st/nd Letter Frequency by Label')
ax.set_xticks(x)
ax.set_xticklabels(letters_ord)
ax.legend()
plt.show()


# Create Dataset
df = pd.DataFrame({
    "Sequences":df_2_monitor['seq'],
    "Outcome":df_2_monitor['outcome']
})

df_1s = df.loc[df["Outcome"]==1, ]
df_0s = df.loc[df["Outcome"]==0, ]
df_full = pd.concat([df_1s, df_0s], ignore_index=True)

# Let's try to reduce to 500_000 sample
df_full = df_full.sample(n_tot)

# Diagnostics graphs
which_1s_sel = list(df_full.loc[df_full["Outcome"]==1, "Sequences"])
which_0s_sel = list(df_full.loc[df_full["Outcome"]==0, "Sequences"])

chr_seq_1s=list(map(extract_characters, which_1s_sel))
chr_seq_0s=list(map(extract_characters, which_0s_sel))

i = 0
stats_1s, stats_0s = collections.Counter(x[i*2] for x in chr_seq_1s), collections.Counter(x[i*2] for x in chr_seq_0s)

letters_ord = string.ascii_uppercase
counts_1s = [stats_1s.get(letter, 0) for letter in letters_ord]
counts_0s = [stats_0s.get(letter, 0) for letter in letters_ord]

# Plot affiancato
x = np.arange(len(letters_ord))
width = 0.35  # Larghezza barre

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, counts_1s, width, label='Label=1', alpha=0.8)
ax.bar(x + width/2, counts_0s, width, label='Label=0', alpha=0.8)
ax.set_xlabel('Letters')
ax.set_ylabel('Count')
ax.set_title(f'{i+1}st/nd Letter Frequency by Label')
ax.set_xticks(x)
ax.set_xticklabels(letters_ord)
ax.legend()
plt.show()

# Splitting Train Val Test
# Now train_val_test split:
X, y = df_full["Sequences"], df_full["Outcome"]
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=0.80, random_state=999)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.50, random_state=999)

# Uncomment just if you want to save data!
# number_csv = 9
# for df, name in zip([X_train, X_val, X_test, y_train, y_val, y_test], [f"X_train_{number_csv}", f"X_val_{number_csv}", f"X_test_{number_csv}", f"y_train_{number_csv}", f"y_val_{number_csv}", f"y_test_{number_csv}"]):
#     df.to_csv(f"data/simulation/{name}.csv", index=False)

# GRAPHICAL STUFF
# Let's visualize bigrams
def plot_bigram_distribution(X_train, y_train, from_n=0, top_n=30):
    """
    Crea un grafico che mostra i bigrammi più discriminativi
    tra le due classi
    """
    
    # Funzione per estrarre bigrammi
    def extract_bigrams(seq):
        letters = seq.split('\x1f')
        return [f"{letters[i]}-{letters[i+1]}" for i in range(len(letters)-1)]
    
    # Conta bigrammi per classe
    bigrams_0 = Counter()
    bigrams_1 = Counter()
    
    for seq, label in zip(pd.DataFrame(X_train)['Sequences'], pd.DataFrame(y_train)['Outcome']):
        bigrams = extract_bigrams(seq)
        if label == 0:
            bigrams_0.update(bigrams)
        else:
            bigrams_1.update(bigrams)
    
    # Calcola frequenze normalizzate
    total_0 = sum(bigrams_0.values())
    total_1 = sum(bigrams_1.values())
    
    # Trova bigrammi più discriminativi
    all_bigrams = set(bigrams_0.keys()) | set(bigrams_1.keys())
    bigram_diff = {}
    
    for bg in all_bigrams:
        freq_0 = bigrams_0.get(bg, 0)
        freq_1 = bigrams_1.get(bg, 0)
        diff = abs(freq_0 - freq_1)
        bigram_diff[bg] = (freq_0, freq_1, diff)
    
    # Ordina per differenza e prendi top N
    top_bigrams = sorted(bigram_diff.items(), key=lambda x: x[1][2], reverse=True)[from_n:top_n]
    
    # Prepara dati per il grafico
    bigram_labels = [bg for bg, _ in top_bigrams]
    counts_0 = [data[0] for _, data in top_bigrams]
    counts_1 = [data[1] for _, data in top_bigrams]
    
    # Crea il grafico
    x = np.arange(len(bigram_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width/2, counts_1, width, label='Label=1 (Ordered)', alpha=0.8)
    bars0 = ax.bar(x + width/2, counts_0, width, label='Label=0 (Unordered)', alpha=0.8)
    
    ax.set_xlabel('Bigrams', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Top {top_n} Most Discriminative Bigrams by Label', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bigram_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bigram_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Stampa statistiche
    print(f"\nTop {min(10, top_n)} most discriminative bigrams:")
    print(f"{'Bigram':<10} {'Count(Label=0)':<15} {'Count(Label=1)':<15} {'Difference':<12}")
    print("-" * 60)
    for bg, (c0, c1, diff) in top_bigrams[:10]:
        print(f"{bg:<10} {c0:<15} {c1:<15} {diff:<12.0f}")

# Esegui
plot_bigram_distribution(X_train, y_train, from_n=1, top_n=100)
