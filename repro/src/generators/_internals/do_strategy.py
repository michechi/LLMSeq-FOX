from src.generators._internals.do_check_lag import check_lag

def do_chek_order(are_lagged_keys, key_letters, cycle):
    """Check if lagged keys are in correct order."""
    n_seq=len(are_lagged_keys)
    all_ordered = [True]*n_seq
    tol=cycle # For the cyclic ordering
    for pos, subseq in enumerate(are_lagged_keys):
        # Check whether are ordered
        for x,y in zip(subseq[:-1], subseq[1:]):
            if ((2*key_letters[x[0]])>(2*key_letters[y[0]])):
                    if tol:
                        tol=False
                    else:
                        all_ordered[pos]=False
                        break
    
    # Return True if at least one ordered sequence is ordered
    return all_ordered, any(all_ordered)

def do_strategy(subseq, lags1_2, key_letters, debugging, cycle=False):
    """Determine strategy based on first subsequence."""
    lag_1, lag_2 = lags1_2
    
    # Check both lags
    are_lagged_keys_1 = check_lag(subseq, key_letters, lag_1)
    are_lagged_keys_2 = check_lag(subseq, key_letters, lag_2)
    
    # Initialize
    all_ordered_1, all_ordered_2 = []
    first_order, second_order = False

    # Check orders
    if are_lagged_keys_1:
        all_ordered_1, first_order = do_chek_order(are_lagged_keys_1, key_letters, cycle)
    if are_lagged_keys_2:
        all_ordered_2, second_order = do_chek_order(are_lagged_keys_2, key_letters, cycle)
    
    # Build info for debugging
    info_2_debug = None
    if debugging:
        info_2_debug = dict(
            lagged_1 = are_lagged_keys_1,
            lagged_2 = are_lagged_keys_2,
            order = [all_ordered_1, all_ordered_2]
            )

    # Let us decide the strategy
    if first_order and second_order:
        return info_2_debug, "both_orders"
    elif first_order:
        return info_2_debug, "first_order"
    elif second_order:
        return info_2_debug, "second_order"
    else:
        return info_2_debug, "no_order"
    
def do_order(subseq,lag, key_letters, strategy, debugging, cycle=False):
    """Check order in second subsequence based on strategy."""
    
    info_2_debug = None

    if strategy == "no_order":
        if debugging:
            info_2_debug = dict(
                lagged_3 = False,
                final_order =  [False]
            )
        return info_2_debug, False
    
    # Handle different strategies
    are_lagged_keys = False

    if strategy == "both_orders":
        # Try first key set, then second
        are_lagged_keys = check_lag(subseq, key_letters[0], lag)
        key_to_use = key_letters[0]
        
        if not are_lagged_keys:
            are_lagged_keys = check_lag(subseq, key_letters[1], lag)
            key_to_use = key_letters[1]
    else:  # first_order or second_order
        are_lagged_keys = check_lag(subseq, key_letters, lag)
        key_to_use = key_letters
    
    # Build debug info
    if debugging:
        info_2_debug = {
            'lagged_3': are_lagged_keys,
            'final_order': [False]
        }
    
    # Check ordering
    if are_lagged_keys:
        all_ordered_3, is_ordered = do_check_order(are_lagged_keys, key_to_use, cycle)
        if debugging:
            info_2_debug['final_order'] = all_ordered_3
        return info_2_debug, is_ordered
    
    return info_2_debug, False


    
