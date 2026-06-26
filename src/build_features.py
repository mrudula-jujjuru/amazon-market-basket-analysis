"""
build_features.py
─────────────────────────────────────────────
Task 3: Feature Engineering
─────────────────────────────────────────────
PURPOSE : Converts raw categorical columns into
          numeric features needed for KMeans
          clustering and rule-based segmentation.
AUTHOR  : Mrudula
─────────────────────────────────────────────
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── Purchase Frequency Text → Number Mapping ─────────────
FREQ_MAP = {
    "Few Times A Week":       5,
    "Once A Week":            4,
    "Few Times A Month":      3,
    "Once A Month":           2,
    "Less Than Once A Month": 1
}


# ── FEATURE 1: Add Purchase Frequency Score ───────────────
def add_purchase_freq_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts text Purchase_Frequency into numeric score 1-5.

    WHY WE NEED THIS:
    KMeans only works with numbers.
    'Once A Week' means nothing to the algorithm
    but the number 4 does.

    Mapping:
        Few Times A Week        → 5 (most frequent)
        Once A Week             → 4
        Few Times A Month       → 3
        Once A Month            → 2
        Less Than Once A Month  → 1 (least frequent)
    """
    df["Purchase_Freq_Score"] = df["Purchase_Frequency"].map(FREQ_MAP)
    print("✅ Purchase_Freq_Score added (scale 1–5)")
    return df


# ── FEATURE 2: Rule-Based Customer Segmentation ──────────
def segment_customer(row: pd.Series) -> str:
    """
    Assigns customer segment based on two factors:
      1. Purchase_Freq_Score  → how often they buy
      2. Shopping_Satisfaction → how happy they are

    Segmentation Rules:
    ┌──────────────────────────────────────────────────┐
    │ Freq ≥ 4 AND Satisfaction ≥ 4 → Frequent Buyer   │
    │ Satisfaction ≤ 2 OR Freq ≤ 1  → At-Risk Customer │
    │ Everything else               → Occasional Shopper│
    └──────────────────────────────────────────────────┘

    WHY THESE THRESHOLDS:
    Freq ≥ 4 = buys at least once a week (active user)
    Sat ≥ 4  = happy with experience (loyal customer)
    Sat ≤ 2  = unhappy (high churn risk)
    """
    freq = row["Purchase_Freq_Score"]
    sat  = row["Shopping_Satisfaction"]

    if freq >= 4 and sat >= 4:
        return "Frequent Buyer"
    elif sat <= 2 or freq <= 1:
        return "At-Risk Customer"
    else:
        return "Occasional Shopper"


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies segment_customer() to every row
    and adds a new 'Segment' column to DataFrame.
    """
    df["Segment"] = df.apply(segment_customer, axis=1)

    print("✅ Customer segments assigned:")
    print(df["Segment"].value_counts().to_string())
    return df


# ── FEATURE 3: Scale Features for KMeans ─────────────────
def get_scaled_features(df: pd.DataFrame) -> tuple:
    """
    Selects 4 key behavioral features and scales them
    for KMeans clustering.

    WHY SCALING IS NEEDED:
    KMeans uses distance between points to form clusters.
    If 'age' ranges 18-65 and 'satisfaction' ranges 1-5,
    age will dominate just because its numbers are bigger.

    StandardScaler brings everything to same scale
    (mean=0, std=1) so all features contribute equally
    to the clustering decision.

    Features selected:
        Purchase_Freq_Score         → buying frequency
        Shopping_Satisfaction       → happiness level
        Customer_Reviews_Importance → trust in reviews
        Rating_Accuracy             → trust in ratings

    Returns:
        X_scaled   : scaled numpy array for KMeans
        scaler     : fitted scaler object
        cluster_df : original unscaled feature DataFrame
    """
    features = [
        "Purchase_Freq_Score",
        "Shopping_Satisfaction",
        "Customer_Reviews_Importance",
        "Rating_Accuracy"
    ]

    # Select only these columns, drop rows with nulls
    cluster_df = df[features].dropna()

    # Scale all features to same range
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    print(f"✅ Features scaled for clustering:")
    for f in features:
        print(f"   → {f}")
    print(f"   Total rows for clustering: {len(cluster_df)}")

    return X_scaled, scaler, cluster_df