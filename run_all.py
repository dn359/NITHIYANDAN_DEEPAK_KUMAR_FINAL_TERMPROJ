# run_all.py  (Jupyter + Windows safe; no rich, no unicode)
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List, Any

import pandas as pd
import typer

from src.io_utils import (
    load_transactions,
    one_hot_encode,
    dataset_name_from_path,
    ensure_dir,
    save_df,
)
from src.metrics import timer
from src.brute_force import (
    frequent_itemsets_bruteforce,
    association_rules_from_frequents,
)
from src.apriori_lib import run_apriori
from src.fpgrowth_lib import run_fpgrowth

app = typer.Typer(add_completion=False)

def print_table(title: str, rows: List[List[Any]], headers: List[str]) -> None:
    print("\n" + str(title))
    print(" | ".join(headers))
    for r in rows:
        print(" | ".join(str(x) for x in r))

def prompt_for_missing_args(
    data: Optional[str], minsup: Optional[float], minconf: Optional[float]
) -> Tuple[str, float, float]:
    # Dataset
    if not data:
        datasets = sorted([f[:-4] for f in os.listdir("data") if f.endswith(".csv")])
        if not datasets:
            raise FileNotFoundError("No CSV files found under ./data")
        print("\nAvailable datasets:")
        for idx, name in enumerate(datasets, 1):
            print(f"  {idx}. {name}")
        sel = input("\nSelect dataset by number or name: ").strip()
        if sel.isdigit():
            i = int(sel)
            if i < 1 or i > len(datasets):
                raise ValueError("Invalid dataset number.")
            chosen = datasets[i - 1]
        else:
            if sel not in datasets:
                raise ValueError(f"'{sel}' is not in {datasets}.")
            chosen = sel
        data = f"data/{chosen}.csv"

    # minsup
    if minsup is None:
        while True:
            try:
                minsup = float(input("Enter minsup (0-1): ").strip())
                if not (0 < minsup <= 1):
                    raise ValueError
                break
            except ValueError:
                print("  Invalid input. Please enter a number between 0 and 1.")

    # minconf
    if minconf is None:
        while True:
            try:
                minconf = float(input("Enter minconf (0-1): ").strip())
                if not (0 < minconf <= 1):
                    raise ValueError
                break
            except ValueError:
                print("  Invalid input. Please enter a number between 0 and 1.")

    return data, minsup, minconf

@app.command()
def main(
    data: Optional[str] = typer.Option(None, "--data", "-d", help="Path to dataset CSV"),
    minsup: Optional[float] = typer.Option(None, "--minsup", "-s", help="Minimum support in (0,1]"),
    minconf: Optional[float] = typer.Option(None, "--minconf", "-c", help="Minimum confidence in (0,1]"),
    outdir: str = typer.Option("outputs", "--outdir", "-o", help="Directory to write outputs"),
):
    """
    Runs Brute Force, Apriori, and FP-Growth on the selected dataset.
    If any of --data/--minsup/--minconf are omitted, you will be prompted interactively.
    """
    data, minsup, minconf = prompt_for_missing_args(data, minsup, minconf)

    if not os.path.exists(data):
        raise FileNotFoundError(f"Dataset not found: {data}")
    if not (0 < minsup <= 1 and 0 < minconf <= 1):
        raise ValueError("Support and confidence must be in (0, 1].")

    print(f"\nRunning with data='{data}', minsup={minsup}, minconf={minconf}\n")

    # Load and prep data
    transactions = load_transactions(data)
    onehot = one_hot_encode(transactions).astype(bool)
    dataset = dataset_name_from_path(data)
    base_out = Path(outdir) / dataset
    ensure_dir(base_out.as_posix())

    # ---------- Brute Force ----------
    with timer() as t:
        bf_freq, bf_support_map = frequent_itemsets_bruteforce(transactions, minsup)
        bf_rules = association_rules_from_frequents(bf_support_map, minconf)
    bf_time = t()

    bf_freq_df = pd.DataFrame(bf_freq)
    bf_rules_df = pd.DataFrame(bf_rules)
    save_df(bf_freq_df, (base_out / "bruteforce" / "frequent_itemsets.csv"))
    save_df(bf_rules_df, (base_out / "bruteforce" / "association_rules.csv"))

    print_table(
        "Brute Force: Frequent Itemsets (top 15)",
        [(r["itemset"], f'{r["support"]:.3f}') for r in bf_freq[:15]],
        ["itemset", "support"],
    )
    print_table(
        "Brute Force: Rules (top 15)",
        [
            (
                r["antecedent"],
                r["consequent"],
                f'{r["support"]:.3f}',
                f'{r["confidence"]:.3f}',
            )
            for r in bf_rules[:15]
        ],
        ["antecedent", "consequent", "support", "confidence"],
    )
    print(f"Brute Force time: {bf_time:.4f} sec")

    # ---------- Apriori ----------
    with timer() as t:
        ap_freq_df, ap_rules_df = run_apriori(onehot, minsup, minconf)
    ap_time = t()
    save_df(ap_freq_df, (base_out / "apriori" / "frequent_itemsets.csv"))
    save_df(ap_rules_df, (base_out / "apriori" / "association_rules.csv"))
    print(f"Apriori time: {ap_time:.4f} sec")

    # ---------- FP-Growth ----------
    with timer() as t:
        fp_freq_df, fp_rules_df = run_fpgrowth(onehot, minsup, minconf)
    fp_time = t()
    save_df(fp_freq_df, (base_out / "fpgrowth" / "frequent_itemsets.csv"))
    save_df(fp_rules_df, (base_out / "fpgrowth" / "association_rules.csv"))
    print(f"FP-Growth time: {fp_time:.4f} sec")

    # ---------- Timings ----------
    timings = pd.DataFrame(
        [
            {"algorithm": "bruteforce", "seconds": round(bf_time, 4)},
            {"algorithm": "apriori", "seconds": round(ap_time, 4)},
            {"algorithm": "fpgrowth", "seconds": round(fp_time, 4)},
        ]
    )
    save_df(timings, (base_out / "timings.csv"))
    print("\nTiming Summary (seconds)")
    print(timings.to_string(index=False))

    print(f"\nOutputs saved under {base_out.as_posix()}\n")

if __name__ == "__main__":
    app()
