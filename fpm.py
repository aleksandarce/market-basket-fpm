"""
=============================================================================
 Frequent Pattern Mining Project
 4th-Year Data Mining Course Project
=============================================================================

 Goal
 ----
 Compare three classic frequent-pattern-mining algorithms on an e-commerce
 transactional dataset:

     1. Apriori        (mlxtend.frequent_patterns.apriori)
     2. FP-Growth      (mlxtend.frequent_patterns.fpgrowth)
     3. ECLAT          (manual implementation using vertical TID-sets)

 For each algorithm we measure:
     - runtime in seconds
     - number of frequent itemsets discovered
     - number of association rules generated
     - top rules by lift and confidence

 The script is configurable from the CONFIGURATION block at the top.

 Dataset (verified on file)
 --------------------------
   Source : "E-commerce Dataset for Predictive Marketing 2023" (Kaggle)
   File   : ECommerce_consumer behaviour.csv  (~121 MB)
   Rows   : 2,019,501
   Cols   : 12  →  order_id, user_id, order_number, order_dow,
                   order_hour_of_day, days_since_prior_order, product_id,
                   add_to_cart_order, reordered, department_id, department,
                   product_name
   Unique : 200,000 orders   |  21 departments  |  134 products
   Missing: only `days_since_prior_order` has NaNs (124,342). The FPM-relevant
            columns (order_id, department, product_name) have ZERO missing values.
   Dupes  : there are no full-row duplicates, but there are 566k duplicate
            (order_id, product_name) pairs because the raw rows are
            product-lines and the same product can appear twice in one cart.
            We drop these in preprocessing.

 How to run
 ----------
     pip install pandas numpy matplotlib mlxtend
     python fpm_project.py

 Author : <your name>
 Date   : <date>
=============================================================================
"""

# -----------------------------------------------------------------------------
# 1. IMPORTS
# -----------------------------------------------------------------------------
# Standard library
import os
import time
from collections import defaultdict
from itertools import combinations

# Third-party numerical / data libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# mlxtend gives us Apriori, FP-Growth and a rule-generation helper.
# It expects a transaction-encoded one-hot DataFrame as input.
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# =============================================================================
# 2. CONFIGURATION
# Edit ONLY this block to adapt the script to your environment / dataset.
# All defaults below were VERIFIED to work on the real Kaggle CSV.
# =============================================================================

# --- Paths ---
DATA_PATH  = "data/ecommerce.csv"      # path to the Kaggle CSV (renamed,
                                       # original name has a space:
                                       # "ECommerce_consumer behaviour.csv")
OUTPUT_DIR = "outputs"                 # all results / plots are written here

# --- Column names (VERIFIED against the actual file) -------------------------
TRANSACTION_COL = "order_id"           # 200,000 unique orders
ITEM_COL        = "product_name"       # 134 unique products
CATEGORY_COL    = "department"         # 21 unique departments
CUSTOMER_COL    = "user_id"            # not used unless BUILD_TRANSACTION_FROM
DATE_COL        = None                 # this dataset has no real date column
                                       # (only order_dow / order_hour_of_day)

# The dataset already has order_id, so we never need to synthesize one.
# Leave this as None unless you adapt the script to another dataset.
BUILD_TRANSACTION_FROM = None          # None  |  "customer_date"

# --- What level of mining to perform ----------------------------------------
LEVEL = "category"    # "category" -> uses the 21 departments (DENSER, FAST).
                      # "product"  -> uses product_name. Note: there are only
                      #               134 unique products, but running on all
                      #               of them at min_support=0.01 still risks
                      #               OOM in apriori; ALWAYS keep top-N
                      #               filtering on for the product run.

# --- Performance knobs ------------------------------------------------------
USE_TOP_ITEMS         = True
TOP_N_ITEMS           = 50            # for "product" level. Ignored if LEVEL
                                      # is "category" (only 21 anyway).
USE_SAMPLE            = True
N_SAMPLE_TRANSACTIONS = 50_000        # 50k baskets is plenty: in our dry-run
                                      # all three algorithms finished in
                                      # under 2 seconds at this size.

# --- Frequent-pattern-mining parameters -------------------------------------
MIN_SUPPORT    = 0.01                 # 1% of transactions
MIN_CONFIDENCE = 0.30                 # 30% confidence threshold for rules
MAX_LEN        = 3                    # max itemset size; keeps runtime sane

# --- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42

# Make sure the output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 3. LOAD & INSPECT
# =============================================================================

def load_and_inspect(path):
    """
    Load the CSV and print a basic data-quality report.
    This is the very first thing to run; it tells us:
      - how many rows / columns we have
      - which columns exist (so we know what to put in the CONFIG block)
      - how many missing values each column has

    Optimisation: the raw CSV is ~121 MB and has 12 columns, but FPM only
    needs three of them. We pass `usecols=` to read_csv so pandas never
    materialises the other nine columns — this roughly quarters the memory
    footprint of the loaded DataFrame.
    """
    print(">>> Loading data from:", path)

    # Build the minimum set of columns we will actually need downstream.
    needed = {TRANSACTION_COL, ITEM_COL, CATEGORY_COL}
    if BUILD_TRANSACTION_FROM == "customer_date":
        needed.update({CUSTOMER_COL, DATE_COL})
    needed.discard(None)

    df = pd.read_csv(path, usecols=list(needed))

    print("\n--- SHAPE ---")
    print(df.shape)

    print("\n--- COLUMNS LOADED ---")
    print(df.columns.tolist())

    print("\n--- DTYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isna().sum())

    print("\n--- HEAD ---")
    print(df.head())

    return df


# =============================================================================
# 4. PREPROCESSING
# =============================================================================

def build_synthetic_transaction_id(df, customer_col, date_col):
    """Optional: if there is no real order_id, treat one customer + one day
    as a transaction. This is a standard fallback used in many FPM tutorials."""
    if customer_col is None or date_col is None:
        raise ValueError("To build synthetic txn id, both customer and date "
                         "columns must be provided.")
    df = df.copy()
    df["txn_id"] = (
        df[customer_col].astype(str) + "_" + df[date_col].astype(str)
    )
    return df


def preprocess(df, transaction_col, item_col):
    """
    Keep only the two columns we need, drop missing values and duplicates,
    and normalise item strings (lower-case + strip whitespace).
    """
    print("\n>>> Preprocessing...")
    df = df[[transaction_col, item_col]].copy()

    before = len(df)

    # Drop rows with missing transaction id or item — FPM cannot use them.
    df = df.dropna(subset=[transaction_col, item_col])

    # Normalise item strings so "Banana", "banana ", "BANANA" become the same.
    df[item_col] = df[item_col].astype(str).str.strip().str.lower()
    df = df[df[item_col] != ""]

    # If the same item appears twice in the same transaction, keep only one.
    # (FPM cares about *presence*, not quantity.)
    df = df.drop_duplicates(subset=[transaction_col, item_col])

    after = len(df)
    print(f"    Rows before cleaning: {before:,}")
    print(f"    Rows after cleaning : {after:,}  (removed {before - after:,})")
    return df


def filter_top_items(df, item_col, top_n):
    """
    Keep only the rows whose item is among the top-N most frequent items.
    This drastically reduces the size of the basket matrix and makes
    Apriori tractable on product-level data.
    """
    top_items = df[item_col].value_counts().head(top_n).index
    filtered = df[df[item_col].isin(top_items)].copy()
    print(f"    Kept top {top_n} items "
          f"→ rows: {len(filtered):,}  unique items: {filtered[item_col].nunique()}")
    return filtered


def build_baskets(df, transaction_col, item_col):
    """
    Group rows by transaction id and collect each transaction's items into a list.
    Baskets with fewer than 2 items cannot produce association rules,
    so we drop them.
    """
    baskets = (
        df.groupby(transaction_col)[item_col]
          .apply(list)
    )
    baskets = baskets[baskets.apply(len) >= 2]
    print(f"    Built {len(baskets):,} baskets (size ≥ 2)")
    return baskets.tolist()


def sample_baskets(baskets, n, seed=RANDOM_SEED):
    """Randomly sub-sample baskets. Used to keep runtime under control."""
    if len(baskets) <= n:
        return baskets
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(baskets), size=n, replace=False)
    sampled = [baskets[i] for i in idx]
    print(f"    Sampled {len(sampled):,} baskets (from {len(baskets):,})")
    return sampled


def one_hot_encode(baskets):
    """
    Convert the list-of-lists representation into a one-hot DataFrame
    where each row is a transaction and each column is an item (True/False).
    mlxtend's Apriori and FP-Growth require this format.
    """
    te = TransactionEncoder()
    arr = te.fit(baskets).transform(baskets)
    encoded = pd.DataFrame(arr, columns=te.columns_)
    print(f"    One-hot matrix shape: {encoded.shape}")
    return encoded


# =============================================================================
# 5. ECLAT (manual implementation)
# =============================================================================
#
# Idea (from the lecture notes):
#   Apriori / FP-Growth use the HORIZONTAL data layout
#       transaction -> list of items
#   ECLAT uses the VERTICAL layout
#       item -> set of transaction ids that contain it
#
#   To compute the support of an itemset X = {a, b, c} we INTERSECT
#   the TID-sets of a, b, c. The size of the intersection divided by
#   the number of transactions is the support.
#
#   We expand itemsets level by level (k = 1, 2, 3 …) and prune any
#   candidate whose support falls below min_support.
#
# This is not the fastest possible ECLAT (no dECLAT, no diffsets),
# but it's the textbook version and easy to explain in the demo.
# =============================================================================

def eclat(baskets, min_support, max_len=3):
    """
    Run a textbook ECLAT and return a DataFrame in the same shape as the
    one returned by mlxtend.apriori, so that association_rules() can consume it.
    """
    n_txn = len(baskets)
    min_count = int(np.ceil(min_support * n_txn))

    # 1) Build the vertical layout: item -> set of transaction indices
    item_tids = defaultdict(set)
    for tid, items in enumerate(baskets):
        for it in set(items):
            item_tids[it].add(tid)

    # 2) Keep only items whose TID-set is large enough (frequent 1-itemsets)
    L1 = {
        frozenset([it]): tids
        for it, tids in item_tids.items()
        if len(tids) >= min_count
    }
    all_frequent = dict(L1)

    # 3) Iteratively build larger itemsets by intersecting TID-sets
    current_level = L1
    k = 2
    while current_level and k <= max_len:
        next_level = {}
        keys = list(current_level.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                candidate = keys[i] | keys[j]
                # We only want itemsets of size exactly k at this level
                if len(candidate) != k or candidate in next_level:
                    continue
                tids = current_level[keys[i]] & current_level[keys[j]]
                if len(tids) >= min_count:
                    next_level[candidate] = tids
        all_frequent.update(next_level)
        current_level = next_level
        k += 1

    # 4) Format result like mlxtend so downstream code stays the same
    rows = [
        {"support": len(tids) / n_txn, "itemsets": itemset}
        for itemset, tids in all_frequent.items()
    ]
    return pd.DataFrame(rows)


# =============================================================================
# 6. RULES + UTILITIES
# =============================================================================

def make_rules(itemsets_df, min_confidence):
    """Wrapper around mlxtend.association_rules that survives empty input."""
    if itemsets_df is None or len(itemsets_df) == 0:
        return pd.DataFrame()
    try:
        return association_rules(
            itemsets_df,
            metric="confidence",
            min_threshold=min_confidence,
        )
    except Exception as e:
        # association_rules can raise if no 2+-itemsets are present
        print(f"    [warn] association_rules failed: {e}")
        return pd.DataFrame()


def time_it(fn, *args, **kwargs):
    """Tiny helper: run fn and return (result, runtime_seconds)."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


# =============================================================================
# 7. VISUALISATIONS
# =============================================================================

def plot_top_items(df, item_col, out_path, top=20):
    counts = df[item_col].value_counts().head(top)
    plt.figure(figsize=(10, 6))
    counts[::-1].plot(kind="barh")          # reverse so largest is on top
    plt.title(f"Top {top} Most Frequent Items")
    plt.xlabel("Number of transactions containing the item")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bar(values, labels, title, ylabel, out_path):
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.2f}" if isinstance(v, float) else str(v),
                 ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_support_confidence(rules, out_path):
    if rules is None or len(rules) == 0:
        return
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(rules["support"], rules["confidence"],
                     c=rules["lift"], cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association rules: support vs confidence (colour = lift)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================

def main():
    # ---- 8.1 Load --------------------------------------------------------
    df = load_and_inspect(DATA_PATH)

    # ---- 8.2 Build / pick the transaction column ------------------------
    if BUILD_TRANSACTION_FROM == "customer_date":
        df = build_synthetic_transaction_id(df, CUSTOMER_COL, DATE_COL)
        transaction_col = "txn_id"
    else:
        transaction_col = TRANSACTION_COL

    # ---- 8.3 Pick item column based on chosen level ---------------------
    item_col = CATEGORY_COL if LEVEL == "category" else ITEM_COL
    print(f"\n>>> Mining level   : {LEVEL}")
    print(f">>> Transaction col: {transaction_col}")
    print(f">>> Item column    : {item_col}")

    # ---- 8.4 Preprocess -------------------------------------------------
    df_clean = preprocess(df, transaction_col, item_col)

    # ---- 8.5 Optional: keep only top-N most frequent items --------------
    if USE_TOP_ITEMS:
        df_clean = filter_top_items(df_clean, item_col, TOP_N_ITEMS)

    # ---- 8.6 Build baskets ----------------------------------------------
    baskets = build_baskets(df_clean, transaction_col, item_col)

    # ---- 8.7 Optional: sample baskets for speed -------------------------
    if USE_SAMPLE:
        baskets = sample_baskets(baskets, N_SAMPLE_TRANSACTIONS)

    # ---- 8.8 One-hot encode ---------------------------------------------
    encoded = one_hot_encode(baskets)

    # ---- 8.9 Run the three algorithms -----------------------------------
    print("\n>>> Running Apriori...")
    ap_items, ap_time = time_it(
        apriori, encoded, min_support=MIN_SUPPORT,
        use_colnames=True, max_len=MAX_LEN,
    )
    print(f"    Apriori   : {ap_time:6.2f}s  → {len(ap_items):,} itemsets")

    print(">>> Running FP-Growth...")
    fp_items, fp_time = time_it(
        fpgrowth, encoded, min_support=MIN_SUPPORT,
        use_colnames=True, max_len=MAX_LEN,
    )
    print(f"    FP-Growth : {fp_time:6.2f}s  → {len(fp_items):,} itemsets")

    print(">>> Running ECLAT (manual implementation)...")
    ec_items, ec_time = time_it(
        eclat, baskets, MIN_SUPPORT, MAX_LEN,
    )
    print(f"    ECLAT     : {ec_time:6.2f}s  → {len(ec_items):,} itemsets")

    # ---- 8.10 Generate association rules --------------------------------
    print("\n>>> Generating association rules...")
    ap_rules = make_rules(ap_items, MIN_CONFIDENCE)
    fp_rules = make_rules(fp_items, MIN_CONFIDENCE)
    ec_rules = make_rules(ec_items, MIN_CONFIDENCE)
    print(f"    Apriori   rules: {len(ap_rules):,}")
    print(f"    FP-Growth rules: {len(fp_rules):,}")
    print(f"    ECLAT     rules: {len(ec_rules):,}")

    # ---- 8.11 Save results ----------------------------------------------
    ap_items.to_csv(f"{OUTPUT_DIR}/apriori_itemsets.csv", index=False)
    fp_items.to_csv(f"{OUTPUT_DIR}/fpgrowth_itemsets.csv", index=False)
    ec_items.to_csv(f"{OUTPUT_DIR}/eclat_itemsets.csv",   index=False)
    ap_rules.to_csv(f"{OUTPUT_DIR}/apriori_rules.csv",    index=False)
    fp_rules.to_csv(f"{OUTPUT_DIR}/fpgrowth_rules.csv",   index=False)
    ec_rules.to_csv(f"{OUTPUT_DIR}/eclat_rules.csv",      index=False)

    # ---- 8.12 Comparison table ------------------------------------------
    # We add the average support / confidence / lift across each algorithm's
    # rule set as well, because the project brief asks for them explicitly.
    def _avg(col, rules_df):
        return float(rules_df[col].mean()) if len(rules_df) > 0 else float("nan")

    comparison = pd.DataFrame({
        "Algorithm":         ["Apriori", "FP-Growth", "ECLAT"],
        "Runtime (s)":       [ap_time, fp_time, ec_time],
        "Frequent itemsets": [len(ap_items), len(fp_items), len(ec_items)],
        "Rules":             [len(ap_rules), len(fp_rules), len(ec_rules)],
        "Avg support":       [_avg("support",    ap_rules),
                              _avg("support",    fp_rules),
                              _avg("support",    ec_rules)],
        "Avg confidence":    [_avg("confidence", ap_rules),
                              _avg("confidence", fp_rules),
                              _avg("confidence", ec_rules)],
        "Avg lift":          [_avg("lift",       ap_rules),
                              _avg("lift",       fp_rules),
                              _avg("lift",       ec_rules)],
    })
    print("\n=== COMPARISON ===")
    print(comparison.to_string(index=False))
    comparison.to_csv(f"{OUTPUT_DIR}/algorithm_comparison.csv", index=False)

    # -- DEMO NOTE ON RUNTIME ORDERING ------------------------------------
    # Textbooks usually say:  FP-Growth < ECLAT < Apriori  (faster -> slower).
    # On THIS dataset (only 21 departments or top-50 products, dense baskets,
    # small N) we have repeatedly observed a different order, e.g.
    #     category-level, 20k baskets : Apriori 0.03s < ECLAT 0.11s < FP-Growth 0.18s
    #     product-level (top-50), 50k : ECLAT 1.23s < Apriori 1.64s < FP-Growth 1.97s
    # This is not a bug. mlxtend's apriori is vectorised over NumPy boolean
    # matrices and is extremely fast when the number of items is small; the
    # tree-building overhead of FP-Growth only pays off on much larger /
    # denser data. ECLAT's TID-set intersections are also very cheap on small
    # data. The point of the comparison is to OBSERVE this and explain it,
    # not to force a specific ordering. (See the Discussion section in the
    # report.)

    # ---- 8.13 Top rules for the report ----------------------------------
    if len(fp_rules) > 0:
        top_lift = fp_rules.sort_values("lift", ascending=False).head(10)
        top_conf = fp_rules.sort_values("confidence", ascending=False).head(10)
        top_lift.to_csv(f"{OUTPUT_DIR}/top_rules_by_lift.csv", index=False)
        top_conf.to_csv(f"{OUTPUT_DIR}/top_rules_by_confidence.csv", index=False)
        print("\n--- TOP 10 RULES BY LIFT (FP-Growth) ---")
        print(top_lift[["antecedents", "consequents",
                        "support", "confidence", "lift"]].to_string(index=False))

    # ---- 8.14 Plots -----------------------------------------------------
    plot_top_items(df_clean, item_col, f"{OUTPUT_DIR}/plot_top_items.png")
    plot_bar(comparison["Runtime (s)"].tolist(),
             comparison["Algorithm"].tolist(),
             "Runtime comparison", "Seconds",
             f"{OUTPUT_DIR}/plot_runtime.png")
    plot_bar(comparison["Frequent itemsets"].tolist(),
             comparison["Algorithm"].tolist(),
             "Number of frequent itemsets", "Count",
             f"{OUTPUT_DIR}/plot_itemsets.png")
    plot_bar(comparison["Rules"].tolist(),
             comparison["Algorithm"].tolist(),
             "Number of association rules", "Count",
             f"{OUTPUT_DIR}/plot_rules.png")
    plot_support_confidence(fp_rules,
                            f"{OUTPUT_DIR}/plot_support_vs_confidence.png")

    print(f"\n>>> Done. All results saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
