from typing import List, Union

def check_lag(seq: List, keys: List, lags: Union[int, List[int]] = 6, 
              overflow_strategy: str = 'repeat_last') -> Union[List[List], bool]:
    """
    Trova catene di elementi con distanze specifiche (fisse o variabili).
    
    Args:
        seq: sequenza di elementi
        keys: elementi da cercare nella sequenza
        lags: può essere:
            - int: distanza fissa tra tutti gli elementi
            - List[int]: distanze variabili [d1, d2, d3, ...]
                dove d1 = distanza tra elem 1 e 2
                      d2 = distanza tra elem 2 e 3, etc.
        overflow_strategy: cosa fare quando gli elementi sono più dei lag:
            - 'repeat_last': riusa l'ultimo lag per elementi successivi
            - 'cycle': ricomincia dal primo lag (pattern ciclico)
            - 'any': qualsiasi distanza va bene dopo i lag specificati
            - 'strict': accetta solo catene lunghe quanto i lag + 1
    
    Returns:
        False se non trova pattern validi
        Lista di liste con i pattern trovati
    
    Examples:
        >>> seq = ['A', 'X', 'X', 'X', 'X', 'X', 'X', 'B', 'Y', 'Y', 'Y', 'C', 'Z', 'Z', 'D']
        >>> keys = ['A', 'B', 'C', 'D']
        >>> 
        >>> # Lag fisso (comportamento originale)
        >>> check_lag(seq, keys, lags=3)
        False  # Non c'è pattern con lag fisso 3
        >>> 
        >>> # Lag variabili [7, 4, 3]
        >>> check_lag(seq, keys, lags=[7, 4, 3])
        [['A', 'B', 'C', 'D']]  # A->B dist 7, B->C dist 4, C->D dist 3
    """
    
    # Converti lag singolo in lista per uniformità
    if isinstance(lags, int):
        lag_list = [lags]
        is_fixed_lag = True
    else:
        lag_list = lags
        is_fixed_lag = False

    # Validazione input
    if not lag_list:
        raise ValueError("La lista dei lag non può essere vuota")
    
    if lags == 0:
        return [elem for elem in seq if elem in keys]
    
    # Trova tutte le posizioni degli elementi chiave
    key_positions = [(i, elem) for i, elem in enumerate(seq) if elem in keys]
    
    if not key_positions:
        return False
    
    chains = []
    used_positions = set()
    
    # Funzione helper per ottenere il lag per una data posizione nella catena
    def get_lag_for_position(chain_index: int) -> Union[int, None]:
        """
        Ritorna il lag da usare per la posizione chain_index nella catena.
        chain_index 0 significa che stiamo cercando il secondo elemento (lag tra 1° e 2°)
        """
        if is_fixed_lag:
            return lag_list[0]
        
        if chain_index < len(lag_list):
            return lag_list[chain_index]
        
        # Gestione overflow: abbiamo più elementi che lag specificati
        if overflow_strategy == 'repeat_last':
            return lag_list[-1]
        elif overflow_strategy == 'cycle':
            return lag_list[chain_index % len(lag_list)]
        elif overflow_strategy == 'any':
            return None  # Segnale che qualsiasi distanza va bene
        elif overflow_strategy == 'strict':
            return False  # Non continuare la catena
        else:
            raise ValueError(f"Strategia overflow non valida: {overflow_strategy}")
    
    # Per ogni posizione di partenza possibile
    for start_pos, start_elem in key_positions:
        if start_pos in used_positions:
            continue
        
        # Costruisci la catena
        chain = [start_elem]
        chain_positions = [start_pos]
        current_pos = start_pos
        chain_index = 0  # Indice per sapere quale lag usare
        
        # Continua a cercare elementi secondo i lag specificati
        while True:
            # Ottieni il lag per questa posizione
            expected_lag = get_lag_for_position(chain_index)
            
            if expected_lag is False:  # Strategia 'strict' - stop
                break
            
            found = False
            
            if expected_lag is None:  # Strategia 'any' - accetta qualsiasi elemento chiave
                # Trova il prossimo elemento chiave a qualsiasi distanza
                for pos, elem in key_positions:
                    if pos > current_pos and elem in keys and pos not in chain_positions:
                        chain.append(elem)
                        chain_positions.append(pos)
                        current_pos = pos
                        found = True
                        break
            else:  # Lag specifico
                next_pos = current_pos + expected_lag
                # Cerca un elemento chiave alla posizione attesa
                for pos, elem in key_positions:
                    if pos == next_pos and elem in keys:
                        chain.append(elem)
                        chain_positions.append(pos)
                        current_pos = next_pos
                        found = True
                        break
            
            if not found:
                break
            
            chain_index += 1
        
        # Salva la catena se ha almeno 2 elementi (pattern valido)
        if len(chain) >= 2:
            # Verifica che non sia sottoinsieme di una catena esistente
            is_subset = False
            for i, existing_chain in enumerate(chains):
                existing_positions = [p for p, _ in existing_chain]
                if set(chain_positions).issubset(set(existing_positions)):
                    is_subset = True
                    break
                # Se questa catena contiene una esistente, sostituiscila
                elif set(existing_positions).issubset(set(chain_positions)):
                    chains[i] = list(zip(chain_positions, chain))
                    used_positions.update(chain_positions)
                    is_subset = True
                    break
            
            if not is_subset:
                chains.append(list(zip(chain_positions, chain)))
                used_positions.update(chain_positions)
    
    if not chains:
        return False
    
    # Ritorna solo le liste di elementi (senza le posizioni)

    return [[elem for _, elem in chain] for chain in chains]