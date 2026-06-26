"""
helpers.py
─────────────────────────────────────────────
Shared Utility Functions
─────────────────────────────────────────────
PURPOSE : Reusable helper functions used across
          multiple tasks. Avoids repeating the
          same code in different files.
AUTHOR  : Mrudula
─────────────────────────────────────────────
"""

import pandas as pd
from collections import Counter


# ── HELPER 1: Explode Purchase Categories ────────────────
def explode_categories(df: pd.DataFrame,
                       col: str = "Purchase_Categories",
                       sep: str = ";") -> pd.DataFrame:
    """
    Converts multi-value category column into a count table.

    WHY WE NEED THIS:
    One customer can select multiple categories stored as:
    'Clothing and Fashion;Electronics;Groceries'

    We need to split and count each category separately
    to find the most popular ones.

    Args:
        df  : cleaned DataFrame
        col : column with multi-values
        sep : separator character (semicolon)

    Returns:
        DataFrame with ['Category', 'Count'] columns
        sorted by Count descending

    Example Output:
        Category                  Count
        Clothing And Fashion        450
        Home And Kitchen            391
        Beauty And Personal Care    383
    """
    all_items = []

    for entry in df[col].dropna():
        for item in str(entry).split(sep):
            item = item.strip().title()
            # Skip empty or invalid entries
            if item and item not in ("Nan", "Unknown", ""):
                all_items.append(item)

    # Count how many times each category appears
    counts = Counter(all_items)

    # Convert to DataFrame and sort by count
    result = (
        pd.DataFrame(
            counts.items(),
            columns=["Category", "Count"]
        )
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    print(f"✅ Categories exploded → {len(result)} unique found")
    return result


# ── HELPER 2: Explode Cart Abandonment Factors ───────────
def explode_factors(df: pd.DataFrame,
                    col: str = "Cart_Abandonment_Factors",
                    sep: str = ";") -> pd.Series:
    """
    Converts multi-value abandonment column into
    a sorted value counts Series.

    WHY WE NEED THIS:
    One customer can have multiple abandonment reasons:
    'High Shipping Costs;Found Better Price Elsewhere'

    We split and count each reason to find
    the most common cart abandonment factors.

    Args:
        df  : cleaned DataFrame
        col : column to explode
        sep : separator character

    Returns:
        pd.Series sorted by count descending

    Example Output:
        High Shipping Costs        224
        Found A Better Price       206
        Changed My Mind            194
    """
    all_factors = []

    for entry in df[col].dropna():
        for factor in str(entry).split(sep):
            factor = factor.strip().title()
            if factor and factor not in ("Nan", "Unknown", ""):
                all_factors.append(factor)

    result = pd.Series(
        Counter(all_factors)
    ).sort_values(ascending=False)

    print(f"✅ Factors exploded → {len(result)} unique found")
    return result


# ── HELPER 3: Print Section Header ───────────────────────
def section_header(title: str) -> None:
    """
    Prints a clean section header in terminal output.

    WHY: Makes terminal output readable when
    running the full pipeline in main.py.

    Example Output:
    ══════════════════════════════════════════════════
      TASK 2: DESCRIPTIVE BEHAVIOR ANALYSIS
    ══════════════════════════════════════════════════
    """
    line = "═" * 50
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")