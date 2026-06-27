"""
main.py
─────────────────────────────────────────────
Amazon Market Basket Analysis — Main Pipeline
─────────────────────────────────────────────
PURPOSE : Entry point for the entire project.
          Runs all tasks in correct order by
          calling functions from src/ modules.

HOW TO RUN:
          python main.py

AUTHOR  : Mrudula
─────────────────────────────────────────────
"""

import yaml
import pandas as pd

# ── Import all modules from src/ ─────────────────────────
from src.clean_data      import run_cleaning_pipeline
from src.helpers         import (explode_categories,
                                  explode_factors,
                                  section_header)
from src.build_features  import (add_purchase_freq_score,
                                  add_segments,
                                  get_scaled_features)
from src.clustering      import (find_optimal_k,
                                  train_kmeans,
                                  save_model)
from src.plot_helpers    import (plot_age_distribution,
                                  plot_gender_distribution,
                                  plot_purchase_frequency,
                                  plot_top_categories,
                                  plot_cart_abandonment,
                                  plot_browsing_frequency,
                                  plot_satisfaction_distribution,
                                  plot_customer_segments,
                                  plot_kmeans_clusters,
                                  plot_segment_satisfaction,
                                  plot_rec_vs_satisfaction,
                                  plot_review_reliability,
                                  plot_recommendation_engagement,
                                  plot_dashboard)


# ── Load Configuration ────────────────────────────────────
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

RAW_PATH       = cfg["data"]["raw_path"]
PROCESSED_PATH = cfg["data"]["processed_path"]
N_CLUSTERS     = cfg["kmeans"]["n_clusters"]
RANDOM_STATE   = cfg["kmeans"]["random_state"]


# ══════════════════════════════════════════════════════════
# TASK 1 — DATA CLEANING AND PREPARATION
# ══════════════════════════════════════════════════════════
section_header("TASK 1: DATA CLEANING AND PREPARATION")

df = run_cleaning_pipeline(RAW_PATH, PROCESSED_PATH)


# ══════════════════════════════════════════════════════════
# TASK 2 — DESCRIPTIVE BEHAVIOR ANALYSIS
# ══════════════════════════════════════════════════════════
section_header("TASK 2: DESCRIPTIVE BEHAVIOR ANALYSIS")

# ── Demographics ──────────────────────────────────────────
print("\n📊 Customer Demographics:")
print(f"   Total customers : {len(df)}")
print(f"   Age range       : {df['age'].min()} – {df['age'].max()}")
print(f"   Average age     : {df['age'].mean():.1f} years")
print(f"\n   Gender breakdown:")
print(df["Gender"].value_counts().to_string())

plot_age_distribution(df)
plot_gender_distribution(df)

# ── Purchase Behavior ─────────────────────────────────────
print("\n📊 Purchase Behavior:")
print(df["Purchase_Frequency"].value_counts().to_string())
plot_purchase_frequency(df)

# ── Product Categories ────────────────────────────────────
print("\n📊 Top Product Categories:")
cat_df = explode_categories(df)
print(cat_df.head(10).to_string())
plot_top_categories(cat_df)

# ── Cart Abandonment ──────────────────────────────────────
print("\n📊 Cart Abandonment Factors:")
factors = explode_factors(df)
print(factors.head(8).to_string())
plot_cart_abandonment(factors)

# ── Browsing & Satisfaction ───────────────────────────────
plot_browsing_frequency(df)
plot_satisfaction_distribution(df)

# ── Summary Statistics ────────────────────────────────────
print("\n📊 Key Metrics Summary:")
metrics = ["Customer_Reviews_Importance",
           "Shopping_Satisfaction",
           "Rating_Accuracy"]
print(df[metrics].agg(["mean", "median", "std"]).round(2))


# ══════════════════════════════════════════════════════════
# TASK 3 — CUSTOMER SEGMENTATION AND PROFILING
# ══════════════════════════════════════════════════════════
section_header("TASK 3: CUSTOMER SEGMENTATION")

# ── Feature Engineering ───────────────────────────────────
df = add_purchase_freq_score(df)
df = add_segments(df)

# ── Rule Based Segments ───────────────────────────────────
print("\n📊 Rule-Based Segment Profiles:")
print(df.groupby("Segment")[
    ["age", "Shopping_Satisfaction",
     "Purchase_Freq_Score"]
].mean().round(2).to_string())

plot_customer_segments(df)
plot_segment_satisfaction(df)

# ── KMeans Clustering ─────────────────────────────────────
print("\n📊 KMeans Clustering:")
X_scaled, scaler, cluster_df = get_scaled_features(df)

# Find best K
find_optimal_k(X_scaled)

# Train with K from config
km, labels = train_kmeans(
    X_scaled,
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE
)

# Add cluster labels to DataFrame
df.loc[cluster_df.index, "KMeans_Cluster"] = labels

# Save trained model
save_model(km, path="reports/kmeans_model.pkl")

plot_kmeans_clusters(df)

# ── Cluster Profiles ──────────────────────────────────────
print("\n📊 KMeans Cluster Profiles:")
print(df.groupby("KMeans_Cluster")[
    ["age", "Shopping_Satisfaction",
     "Purchase_Freq_Score",
     "Customer_Reviews_Importance"]
].mean().round(2).to_string())

# ══════════════════════════════════════════════════════════
#  MARKET BASKET ANALYSIS
# ══════════════════════════════════════════════════════════

# ── Market Basket Analysis ────────────────────────────────
from src.market_basket import run_market_basket

section_header("MARKET BASKET ANALYSIS")
rules = run_market_basket(
    df,
    min_support=cfg["apriori"]["min_support"],
    min_lift=cfg["apriori"]["min_lift"]
)


# ══════════════════════════════════════════════════════════
# TASK 4 — RECOMMENDATION AND REVIEW INSIGHTS
# ══════════════════════════════════════════════════════════
section_header("TASK 4: RECOMMENDATION AND REVIEW INSIGHTS")

# ── Recommendation vs Satisfaction ────────────────────────
print("\n📊 Avg Satisfaction by Recommendation Helpfulness:")
print(df.groupby("Recommendation_Helpfulness")
      ["Shopping_Satisfaction"]
      .mean()
      .sort_values(ascending=False)
      .round(2)
      .to_string())

plot_rec_vs_satisfaction(df)

# ── Review Reliability ────────────────────────────────────
print("\n📊 Rating Accuracy by Review Reliability:")
print(df.groupby("Review_Reliability")
      ["Rating_Accuracy"]
      .mean()
      .sort_values(ascending=False)
      .round(2)
      .to_string())

plot_review_reliability(df)
plot_recommendation_engagement(df)

# ── Actionable Insights ───────────────────────────────────
print("\n💡 Actionable Insights:")
insights = [
    "1. Customers who find recommendations helpful have"
    "   higher satisfaction → improve recommendation algorithms",
    "2. High shipping cost is #1 abandonment reason"
    "   → introduce free shipping thresholds",
    "3. At-Risk customers engage less with recommendations"
    "   → send targeted discount notifications",
    "4. Keyword search dominates → enhance semantic search",
    "5. Heavy review readers show better rating accuracy"
    "   → incentivise review reading with tooltips"
]
for insight in insights:
    print(f"   {insight}")


# ══════════════════════════════════════════════════════════
# TASK 5 — SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════
section_header("TASK 5: SUMMARY DASHBOARD")

plot_dashboard(df, cat_df)


# ══════════════════════════════════════════════════════════
# PIPELINE COMPLETE
# ══════════════════════════════════════════════════════════
print("\n" + "═"*50)
print("  ✅ FULL PIPELINE COMPLETE!")
print("═"*50)
print(f"\n  Total customers analysed : {len(df)}")
print(f"  Charts saved to          : reports/figures/")
print(f"  Cleaned data saved to    : {PROCESSED_PATH}")
print(f"  Segments created         : "
      f"{df['Segment'].nunique()}")
print(f"  KMeans clusters          : {N_CLUSTERS}")
print(f"\n  Top category     : {cat_df.iloc[0]['Category']}")
print(f"  Avg satisfaction : "
      f"{df['Shopping_Satisfaction'].mean():.2f} / 5")
print("\n" + "═"*50)