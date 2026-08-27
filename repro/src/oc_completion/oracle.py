"""Canonical Ordered-Compliance (OC) oracle and mechanism parameters.

This module is the SINGLE entry point for the OC latent-label function used by
the matched-completion analysis. It does not re-implement the rule: it wraps
`rule_greedy_monotone_impl` from `src.analysis.mechanism_id.scripts.common`,
which reproduces the generator chain (`check_lag` + no-tolerance ordering
check, min chain length 2) that produced the paper's OC-Noisy dataset
(tag `_9`; label-audit accuracy 0.699 ~= 1 - pi at pi = 0.3, see
`src/analysis/mechanism_id/report.md` section 2.3).

Mechanism parameters (paper Section 4.1, "medium-difficulty anchor"):
    alphabet   A..Z            (l = 26)
    S          W,D,Q,J,X,U     (m = 6)
    kappa      kappa(W)=0 < kappa(D)=1 < kappa(Q)=2 < kappa(J)=3
               < kappa(X)=4 < kappa(U)=5
    lambda     7
    n          20

Y* = 1 iff the sequence contains at least one maximal lag-lambda run of key
symbols with q >= 2 whose kappa values are non-decreasing.
"""
from __future__ import annotations

import string
from typing import Dict, List, Sequence, Tuple

from src.analysis.mechanism_id.scripts.common import (
    KAPPA_PAPER,
    KEY_LETTERS_PAPER,
    rule_greedy_monotone_impl,
)

SEP = "\x1f"
ALPHABET: Tuple[str, ...] = tuple(string.ascii_uppercase)
N_EVENTS = 20
LAG = 7

KEY_LETTERS: Tuple[str, ...] = KEY_LETTERS_PAPER          # ("W","D","Q","J","X","U")
KAPPA: Dict[str, int] = dict(KAPPA_PAPER)                  # enumeration order
KEY_SET = frozenset(KEY_LETTERS)
NON_KEY_LETTERS: Tuple[str, ...] = tuple(l for l in ALPHABET if l not in KEY_SET)

MECHANISM = {
    "task_family": "ordered_compliance",
    "alphabet_size": len(ALPHABET),
    "n_events": N_EVENTS,
    "key_letters": list(KEY_LETTERS),
    "kappa": {k: KAPPA[k] for k in KEY_LETTERS},
    "lag": LAG,
    "oracle": "rule_greedy_monotone_impl(tolerance=False, min_chain_length=2)",
    "oracle_source": "src/analysis/mechanism_id/scripts/common.py",
}


def oc_label_tokens(toks: Sequence[str]) -> int:
    """Latent label Y* for an already-tokenised sequence."""
    return rule_greedy_monotone_impl(
        toks, LAG, KEY_SET, KAPPA, tolerance=False, min_chain_length=2
    )


def oc_label(seq: str, sep: str = SEP) -> int:
    """Latent label Y* for a separator-joined sequence string."""
    return oc_label_tokens(seq.split(sep))


def to_string(toks: Sequence[str], sep: str = SEP) -> str:
    return sep.join(toks)


def tokens_of(seq: str, sep: str = SEP) -> List[str]:
    return seq.split(sep)
