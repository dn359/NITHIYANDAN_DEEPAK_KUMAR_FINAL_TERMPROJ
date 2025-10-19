from __future__ import annotations
import itertools
from typing import List, Dict, Tuple, Iterable, Set, FrozenSet

def _support_count(transactions: List[List[str]], itemset: FrozenSet[str]) -> int:
    cnt = 0
    for t in transactions:
        if itemset.issubset(t):
            cnt += 1
    return cnt

def frequent_itemsets_bruteforce(
    transactions: List[List[str]],
    min_support: float,
) -> Tuple[List[Dict], Dict[FrozenSet[str], float]]:
    """
    Brute force enumeration of k-itemsets for k = 1..K.
    Returns:
      - list of dicts: {"itemset": tuple(...), "support": float}
      - support_map: {frozenset(items): support_float}
    """
    n_tx = len(transactions)
    unique_items: List[str] = sorted({i for t in transactions for i in t})
    support_map: Dict[FrozenSet[str], float] = {}
    results: List[Dict] = []

    k = 1
    while True:
        candidates = [frozenset(c) for c in itertools.combinations(unique_items, k)]
        freq_this_k: List[FrozenSet[str]] = []
        for cand in candidates:
            sup_count = _support_count(transactions, cand)
            sup = sup_count / n_tx
            if sup >= min_support:
                support_map[cand] = sup
                freq_this_k.append(cand)
                results.append({"itemset": tuple(cand), "support": sup})
        if not freq_this_k:
            break
        k += 1

    # stop when no frequent itemsets for current k (so last non-empty was k-1)
    return results, support_map

def association_rules_from_frequents(
    support_map: Dict[FrozenSet[str], float],
    min_confidence: float,
) -> List[Dict]:
    """
    Generate rules X -> Y for all frequent itemsets L with |L|>=2.
    confidence(L: X->Y) = support(L) / support(X)
    """
    rules: List[Dict] = []
    frequents = [iset for iset in support_map.keys() if len(iset) >= 2]
    for L in frequents:
        L_support = support_map[L]
        items = list(L)
        # proper non-empty subsets as antecedents
        for r in range(1, len(items)):
            for antecedent in itertools.combinations(items, r):
                A = frozenset(antecedent)
                B = L - A
                if not B:
                    continue
                sup_A = support_map.get(A)
                if sup_A is None or sup_A == 0:
                    continue
                confidence = L_support / sup_A
                if confidence >= min_confidence:
                    rules.append({
                        "antecedent": tuple(sorted(A)),
                        "consequent": tuple(sorted(B)),
                        "support": L_support,
                        "confidence": confidence,
                    })
    # sort for pretty output
    rules.sort(key=lambda x: (-x["confidence"], -x["support"], x["antecedent"]))
    return rules
