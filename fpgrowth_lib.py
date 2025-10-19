from __future__ import annotations
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def run_fpgrowth(one_hot_df: pd.DataFrame, min_support: float, min_confidence: float):
    freq = fpgrowth(one_hot_df, min_support=min_support, use_colnames=True)
    freq = freq.sort_values("support", ascending=False).reset_index(drop=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules[["antecedents", "consequents", "support", "confidence"]].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: tuple(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: tuple(sorted(s)))
    freq["itemset"] = freq["itemsets"].apply(lambda s: tuple(sorted(s)))
    freq = freq.drop(columns=["itemsets"])
    rules = rules.sort_values(["confidence", "support"], ascending=[False, False]).reset_index(drop=True)
    return freq[["itemset", "support"]], rules
