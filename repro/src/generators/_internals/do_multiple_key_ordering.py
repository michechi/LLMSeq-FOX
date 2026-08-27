from scipy.stats import bernoulli
import numpy as np
import random
import string

letters = list(string.ascii_uppercase)

def do_keys(letters:list, seed:int) -> dict:
    """Create a different ordering for each letter of the alphabet"""
    key_dict = dict()
    for i, letter in enumerate(letters):
        letters_2_shuffle = letters.copy()
        random.seed(seed+i)
        random.shuffle(letters_2_shuffle)
        key_dict[letter] = {w:p for p,w in enumerate(letters_2_shuffle, start=0)}

    return key_dict

def do_lags(letters:list, seed:int) -> dict:
    """
    Associate for each letter a different lag.
    - 7 -> 0.45
    - 3 -> 0.3
    - 9 -> 0.1
    - 4 -> 0.1
    - 2 -> 0.05
    """
    random.seed(seed)
    lag_dict = dict()
    v = [7, 3, 9, 4, 2] # lags
    n_lags = 5 # no hard_coded
    # How can I distribute probability mass better and in a more automatical way?
    p = np.array(
        [[0.45, 0.2, 0.2, 0.1, 0.05], # first lag
        [0.35, 0.3, 0.1, 0.2, 0.05], # second lag
        [0.1, 0.4, 0.1, 0.3, 0.1], # third lags
        [0.05, 0.3, 0.05, 0.4, 0.2],
        [0.05, 0.3, 0.05, 0.1, 0.5]]) # probabilities for each lag

    n = len(letters) # number of letters
    
    for i in range(n_lags):
        indexes_lags = np.random.multinomial(1, p[i], n)
        all = v * indexes_lags
        mask = all > 0 
        lags = all[mask]
        lag_dict[i] = dict(zip(letters, lags))

    return lag_dict

def do_order_multiple_key(test_seq_splt:list, lags:dict, c_ord:dict):
    ordered = True
    current_position = 0
    lag_step = 0
    n_max = len(test_seq_splt) - 1
    max_lag_steps = len(lags)  # Numero massimo di step disponibili
    
    while ordered:
        # Controlla se abbiamo esaurito i lag disponibili
        if lag_step >= max_lag_steps:
            return ordered  # Ritorna True perché non abbiamo trovato violazioni
        
        pivot_letter = test_seq_splt[current_position]
        current_dict = c_ord[pivot_letter]
        current_lag = lags[lag_step][pivot_letter]
        next_position = current_position + current_lag
        
        # Se andiamo out of range, ritorna ordered (che è ancora True)
        if next_position > n_max:
            return ordered
        
        next_letter = test_seq_splt[next_position]
        
        # Verifica ordinamento
        ordered = (current_dict[pivot_letter] <= current_dict[next_letter])
        
        # Prepara per la prossima iterazione
        current_position = next_position
        lag_step += 1
    
    return ordered  # Sarà False se siamo usciti dal loop

def do_multiple_key_ordering(
    seq:str, 
    c_ord: dict, 
    lags:dict,
    rnd:bool,
    sep:str='\x1f',
    already_splitted=False,
    pr_1=0.7,
    tolerance=True,
    debugging=False):
    """
    Function to do multiple key ordering strategy on a sequence.
    seq: input sequence
    key_dict: dictionary containing ordering information for different keys (one for each letter)
    lag_dict: dictionary containing lag information for each step
    """
    if not already_splitted:
        test_seq_splt = seq.split(sep) # avoiding noising letters
    else: 
        test_seq_splt = seq
    # Caso base: sequenza troppo corta

    if len(test_seq_splt) < 2:
        return True
    
    order = do_order_multiple_key(test_seq_splt, lags, c_ord)
    if order:
        pr_to_simulate = pr_1
    else:
        pr_to_simulate = 1-pr_1
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
            'pr_2_sim':[pr_to_simulate]
        }
    
    return(results_2_debug)
    
