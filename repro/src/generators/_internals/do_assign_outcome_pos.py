import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
from scipy.stats import bernoulli
from typing import List, Union
from tqdm import tqdm
from src.generators._internals.do_check_lag import check_lag



def assign_outcome_positional(
    seq: str,
    c_ord: dict,
    lags,
    rnd: bool = False,
    sep: str = "\x1f",
    already_splitted: bool = False,
    pr_1: float = 0.7,
    debugging: bool = True,
    min_chain_length: int = 2,
    tolerance: bool = True
) -> dict:
    """
    Assigns outcome ∈ {0,1}
    - even length → 1 if ordered
    - odd length  → 1 if NOT ordered
    """

    # --- split sequence ---
    tokens = seq if already_splitted else seq.split(sep)

    # --- parity ---
    is_even = (len(tokens) % 2 == 0)
    is_odd = not is_even

    # --- find lagged keys ---
    lagged = check_lag(tokens, c_ord.keys(), lags)

    def is_ordered_chain(chain):
        """Check ordering with optional single tolerance."""
        if len(chain) < min_chain_length:
            return False

        tol = tolerance
        for x, y in zip(chain[:-1], chain[1:]):
            if c_ord[x[0]] > c_ord[y[0]]:
                if tol:
                    tol = False
                else:
                    return False
        return True

    # --- check ordering ---
    is_ordered = False

    if lagged:
        if all(isinstance(item, list) for item in lagged):
            is_ordered = any(is_ordered_chain(chain) for chain in lagged)
        else:
            is_ordered = is_ordered_chain(lagged)

    # --- outcome ---
    if rnd:
        p = pr_1 if is_ordered else 1 - pr_1
        outcome = bernoulli.rvs(p)
    else:
        outcome = int(is_ordered)

    # --- invert for odd sequences ---
    if is_odd:
        outcome = 1 - outcome

    # --- safety ---
    assert outcome in (0, 1), f"Outcome non binario: {outcome}"

    # --- debug output ---
    return {
        'outcome': [outcome],
        'seq': seq,
        'odd': is_odd,
        'pr_2_sim': [pr_1 if is_ordered else 1 - pr_1],
        'lagged_keys': [lagged],
        'all_ordered': [is_ordered]
    }

def assign_outcome_positional_2_key(
    seq: str,
    c_ord: dict,
    second_key,
    lags,
    rnd: bool = False,
    sep: str = "\x1f",
    already_splitted: bool = False,
    pr_1: float = 0.7,
    debugging: bool = True,
    min_chain_length: int = 2,
    tolerance: bool = True
) -> dict:
    """
    Outcome = 1 iff:
    - sequence satisfies ordering
    - number of tokens belonging to second_key is EVEN
    """

    # --- split sequence ---
    tokens = seq if already_splitted else seq.split(sep)

    # --- count second key occurrences ---
    n_second_key = sum(1 for t in tokens if t[0] in second_key)
    second_key_even = (n_second_key % 2 == 0)

    # --- find lagged keys ---
    lagged = check_lag(tokens, c_ord.keys(), lags)

    def is_ordered_chain(chain):
        """Check ordering with optional single tolerance."""
        if len(chain) < min_chain_length:
            return False

        tol = tolerance
        for x, y in zip(chain[:-1], chain[1:]):
            if c_ord[x[0]] > c_ord[y[0]]:
                if tol:
                    tol = False
                else:
                    return False
        return True

    # --- check ordering ---
    is_ordered = False
    if lagged:
        if all(isinstance(item, list) for item in lagged):
            is_ordered = any(is_ordered_chain(chain) for chain in lagged)
        else:
            is_ordered = is_ordered_chain(lagged)

    # --- final logical condition ---
    valid = is_ordered and second_key_even

    # --- outcome ---
    if rnd:
        p = pr_1 if valid else 1 - pr_1
        outcome = bernoulli.rvs(p)
    else:
        outcome = int(valid)

    # --- safety ---
    assert outcome in (0, 1), f"Outcome non binario: {outcome}"

    # --- debug output ---
    return {
        'outcome': [outcome],
        'seq': seq,
        'n_second_key': [n_second_key],
        'second_key_even': [second_key_even],
        'pr_2_sim': [pr_1 if valid else 1 - pr_1],
        'lagged_keys': [lagged],
        'all_ordered': [is_ordered]
    }

def efficient_check_v1(sequences, c_vocab, assign_outcome_positional, tolerance=True, lags=7, rnd=False, debugging=False):
    """Calcola una volta sola e poi usa i risultati"""
    # Calcola UNA SOLA VOLTA per ogni sequenza
    print(f"Tolerance: {tolerance}, lags: {lags}!\n")
    outcomes = []

    if not debugging:
        df_2_monitor = pd.DataFrame({
            'outcome':[],
            'seq':[],
            'odd':[],
            'pr_2_sim':[],
            # 'lagged_keys':[],
            # 'all_ordered':[]
        })
    else:
        df_2_monitor = pd.DataFrame({
            'outcome':[],
            'seq':[],
            'odd':[],
            'pr_2_sim':[],
            'lagged_keys':[],
            'all_ordered':[],
            'lags':[]
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
    seq, c_vocab, assign_outcome_positional, lags, tolerance, rnd, debugging, min_chain_length = args
    result = assign_outcome_positional(
        seq, c_vocab, 
        already_splitted=False, 
        lags=lags, 
        tolerance=tolerance, 
        rnd=rnd,
        debugging=debugging,
        min_chain_length=min_chain_length
    )
    # Converti il dizionario in DataFrame
    return pd.DataFrame(result)

def efficient_check_parallel(sequences, c_vocab, assign_outcome_positional, 
                             tolerance=True, lags=7, rnd=False, debugging=True, min_chain_length=3, n_workers=None):
    """Versione parallelizzata con multiprocessing.Pool"""
    print(f"Tolerance: {tolerance}, lags: {lags}!\n")
    
    # Use all cores-1 if not specified
    if n_workers is None:
        n_workers = cpu_count()
    
    # Prepara gli argomenti per ogni sequenza
    args_list = [
        (seq, c_vocab, assign_outcome_positional, lags, tolerance, rnd, debugging, min_chain_length) 
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
