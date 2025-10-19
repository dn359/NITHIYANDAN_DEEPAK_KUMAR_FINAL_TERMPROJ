from __future__ import annotations
import os
import pandas as pd

def dataset_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name

def load_transactions(csv_path: str) -> list[list[str]]:
    """
    Robust loader for the TID,Items CSV where Items contains many commas.
    We read lines and split ONLY on the first comma.
    """
    txns: list[list[str]] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        header = f.readline()  # skip header line: TID,Items
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split once: left = TID, right = full items string (may contain commas)
            try:
                _tid, items_str = line.split(",", 1)
            except ValueError:
                # malformed line; skip
                continue
            items = [x.strip() for x in items_str.split(",") if x.strip()]
            txns.append(items)
    return txns

def one_hot_encode(transactions: list[list[str]]) -> pd.DataFrame:
    """
    Returns a one-hot encoded DataFrame (rows = transactions, cols = unique items).
    """
    all_items = sorted({item for txn in transactions for item in txn})
    data = []
    for txn in transactions:
        row = {item: 1 if item in txn else 0 for item in all_items}
        data.append(row)
    return pd.DataFrame(data, columns=all_items)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_df(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
