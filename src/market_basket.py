"""
market_basket.py
─────────────────────────────────────────────
Market Basket Analysis — Association Rules
─────────────────────────────────────────────
PURPOSE : Applies Apriori algorithm to find
          product categories frequently
          purchased together.
          Reveals cross-selling opportunities.
AUTHOR  : Mrudula
─────────────────────────────────────────────
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# ── STEP 1: Build Transactions ───────────────────────────
def build_transactions(df: pd.DataFrame,
                       col: str = "Purchase_Categories",
                       sep: str = ";") -> list:
    """
    Converts multi-value category column into
    a list of transactions for Apriori.

    WHY: Apriori needs each row as a list of items.
    'Clothing;Electronics' → ['Clothing', 'Electronics']

    Args:
        df  : cleaned DataFrame
        col : column with purchase categories
        sep : separator character

    Returns:
        List of transactions (list of lists)
    """
    transactions = []
    for entry in df[col].dropna():
        items = [
            i.strip().title()
            for i in str(entry).split(sep)
            if i.strip() and i.strip() not in
            ("Unknown", "Nan", "")
        ]
        if items:
            transactions.append(items)

    print(f"✅ {len(transactions)} transactions built")
    return transactions


# ── STEP 2: Encode Transactions ──────────────────────────
def encode_transactions(transactions: list) -> pd.DataFrame:
    """
    Converts list of transactions into a
    one-hot encoded DataFrame for Apriori.

    WHY: Apriori needs a boolean matrix —
    True if item is in transaction, False if not.

    Returns:
        Boolean DataFrame (rows=transactions,
        cols=unique categories)
    """
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    basket_df = pd.DataFrame(
        te_array,
        columns=te.columns_
    )
    print(f"✅ Encoded: {basket_df.shape[0]} transactions,"
          f" {basket_df.shape[1]} unique categories")
    return basket_df


# ── STEP 3: Run Apriori ──────────────────────────────────
def run_apriori(basket_df: pd.DataFrame,
                min_support: float = 0.05) -> pd.DataFrame:
    """
    Finds frequent itemsets using Apriori algorithm.

    WHY min_support=0.05:
    A category pair must appear in at least 5%
    of transactions to be considered meaningful.
    Too low = noise. Too high = miss real patterns.

    Args:
        basket_df   : encoded transaction DataFrame
        min_support : minimum support threshold

    Returns:
        DataFrame of frequent itemsets with support
    """
    frequent_items = apriori(
        basket_df,
        min_support=min_support,
        use_colnames=True
    )
    print(f"✅ Frequent itemsets found: {len(frequent_items)}")
    return frequent_items


# ── STEP 4: Generate Association Rules ───────────────────
def get_association_rules(frequent_items: pd.DataFrame,
                          min_lift: float = 1.0) -> pd.DataFrame:
    """
    Generates association rules from frequent itemsets.

    Three key metrics:
        Support    → how often the pair appears together
        Confidence → how often the rule is correct
        Lift > 1   → the pair occurs more than by chance

    Args:
        frequent_items : output from run_apriori()
        min_lift       : minimum lift threshold

    Returns:
        DataFrame of rules sorted by lift descending
    """
    rules = association_rules(
        frequent_items,
        metric="lift",
        min_threshold=min_lift
    )
    rules = rules.sort_values("lift", ascending=False)
    print(f"✅ Association rules found: {len(rules)}")
    return rules


# ── MASTER FUNCTION ───────────────────────────────────────
def run_market_basket(df: pd.DataFrame,
                      min_support: float = 0.05,
                      min_lift: float = 1.0) -> pd.DataFrame:
    """
    Runs full Market Basket Analysis pipeline.
    Call this from main.py or notebook.

    Args:
        df          : cleaned DataFrame
        min_support : Apriori support threshold
        min_lift    : Association rule lift threshold

    Returns:
        rules DataFrame sorted by lift
    """
    print("\n" + "="*50)
    print("  MARKET BASKET ANALYSIS — APRIORI")
    print("="*50)

    transactions  = build_transactions(df)
    basket_df     = encode_transactions(transactions)
    frequent_items = run_apriori(basket_df, min_support)
    rules         = get_association_rules(frequent_items,
                                          min_lift)

    print("\n📊 Top 5 Cross-Selling Opportunities:")
    print(rules[["antecedents", "consequents",
                 "support", "confidence", "lift"]
               ].head(5).to_string())

    return rules
