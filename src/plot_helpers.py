"""
plot_helpers.py
─────────────────────────────────────────────
Tasks 2, 3, 4, 5: All Visualizations
─────────────────────────────────────────────
PURPOSE : Contains all chart/plot functions.
          Every function saves its chart to
          reports/figures/ automatically.
AUTHOR  : Mrudula
─────────────────────────────────────────────
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Global Settings ───────────────────────────────────────
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Amazon brand colors
COLORS = [
    "#FF9900",  # Amazon orange
    "#232F3E",  # Amazon dark
    "#146EB4",  # Amazon blue
    "#2ecc71",  # green
    "#e74c3c",  # red
    "#3498db",  # light blue
    "#9b59b6",  # purple
    "#e67e22"   # dark orange
]


def save_figure(filename: str) -> None:
    """Save current plot to reports/figures/ and show it."""
    path = f"{FIGURES_DIR}/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ Chart saved → {path}")


# ══════════════════════════════════════════════════════════
# TASK 2 — DESCRIPTIVE BEHAVIOR ANALYSIS
# ══════════════════════════════════════════════════════════

def plot_age_distribution(df: pd.DataFrame) -> None:
    """
    Task 2.1 — Age distribution of surveyed customers.
    Shows which age groups shop most on Amazon.
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(df["age"], bins=15, kde=True,
                 color=COLORS[2])
    plt.title("Age Distribution of Amazon Customers",
              fontweight="bold")
    plt.xlabel("Age")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    save_figure("01_age_distribution.png")


def plot_gender_distribution(df: pd.DataFrame) -> None:
    """
    Task 2.2 — Gender breakdown of surveyed customers.
    Pie chart showing Male/Female/Other split.
    """
    gender_counts = df["Gender"].value_counts()

    plt.figure(figsize=(6, 5))
    plt.pie(gender_counts,
            labels=gender_counts.index,
            autopct="%1.1f%%",
            colors=COLORS[:len(gender_counts)],
            startangle=140)
    plt.title("Gender Distribution",
              fontweight="bold")
    plt.tight_layout()
    save_figure("02_gender_distribution.png")


def plot_purchase_frequency(df: pd.DataFrame) -> None:
    """
    Task 2.3 — How often customers make purchases.
    Shows the distribution of buying frequency.
    """
    freq_order = [
        "Few Times A Week",
        "Once A Week",
        "Few Times A Month",
        "Once A Month",
        "Less Than Once A Month"
    ]

    freq_counts = (df["Purchase_Frequency"]
                   .value_counts()
                   .reindex(freq_order, fill_value=0))

    plt.figure(figsize=(9, 4))
    freq_counts.plot(kind="bar", color=COLORS[0],
                     edgecolor="white")
    plt.title("Purchase Frequency Distribution",
              fontweight="bold")
    plt.xlabel("Frequency")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_figure("03_purchase_frequency.png")


def plot_top_categories(cat_df: pd.DataFrame,
                        top_n: int = 10) -> None:
    """
    Task 2.4 — Most purchased product categories.
    Horizontal bar chart of top N categories.
    """
    plt.figure(figsize=(10, 5))
    top = cat_df.head(top_n)
    plt.barh(top["Category"], top["Count"],
             color=COLORS[0], edgecolor="white")
    plt.title(f"Top {top_n} Product Categories Purchased",
              fontweight="bold")
    plt.xlabel("Number of Customers")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_figure("04_top_categories.png")


def plot_cart_abandonment(factors_series: pd.Series) -> None:
    """
    Task 2.5 — Why customers abandon their carts.
    Bar chart of most common abandonment reasons.
    """
    plt.figure(figsize=(10, 4))
    factors_series.head(8).plot(kind="bar",
                                 color=COLORS[4],
                                 edgecolor="white")
    plt.title("Cart Abandonment Factors",
              fontweight="bold")
    plt.xlabel("Reason")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    save_figure("05_cart_abandonment.png")


def plot_browsing_frequency(df: pd.DataFrame) -> None:
    """
    Task 2.6 — How often customers browse without buying.
    Shows passive engagement with Amazon platform.
    """
    plt.figure(figsize=(8, 4))
    df["Browsing_Frequency"].value_counts().plot(
        kind="bar", color=COLORS[2], edgecolor="white")
    plt.title("Browsing Frequency Distribution",
              fontweight="bold")
    plt.xlabel("Browsing Frequency")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_figure("06_browsing_frequency.png")


def plot_satisfaction_distribution(df: pd.DataFrame) -> None:
    """
    Task 2.7 — Overall shopping satisfaction levels.
    Bar chart of satisfaction scores 1-5.
    """
    sat_counts = (df["Shopping_Satisfaction"]
                  .value_counts()
                  .sort_index())

    plt.figure(figsize=(7, 4))
    sat_counts.plot(kind="bar",
                    color=COLORS[0],
                    edgecolor="white")
    plt.title("Shopping Satisfaction Levels (Scale 1–5)",
              fontweight="bold")
    plt.xlabel("Satisfaction Score")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_figure("07_satisfaction_levels.png")


# ══════════════════════════════════════════════════════════
# TASK 3 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════

def plot_customer_segments(df: pd.DataFrame) -> None:
    """
    Task 3.1 — Rule-based customer segment distribution.
    Shows how many customers fall in each segment.
    """
    seg_counts = df["Segment"].value_counts()

    plt.figure(figsize=(7, 4))
    seg_counts.plot(
        kind="bar",
        color=[COLORS[3], COLORS[0], COLORS[4]],
        edgecolor="white"
    )
    plt.title("Customer Segments Distribution",
              fontweight="bold")
    plt.xlabel("Segment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_figure("08_customer_segments.png")


def plot_kmeans_clusters(df: pd.DataFrame) -> None:
    """
    Task 3.2 — KMeans cluster distribution.
    Shows customer count in each cluster (0, 1, 2).
    """
    plt.figure(figsize=(7, 4))
    df["KMeans_Cluster"].value_counts().sort_index().plot(
        kind="bar", color=COLORS[2], edgecolor="white")
    plt.title("KMeans Cluster Distribution",
              fontweight="bold")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_figure("09_kmeans_clusters.png")


def plot_segment_satisfaction(df: pd.DataFrame) -> None:
    """
    Task 3.3 — Satisfaction levels across segments.
    Boxplot showing satisfaction spread per segment.
    """
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df,
                x="Segment",
                y="Shopping_Satisfaction",
                palette="Set2")
    plt.title("Satisfaction by Customer Segment",
              fontweight="bold")
    plt.xlabel("Segment")
    plt.ylabel("Shopping Satisfaction (1–5)")
    plt.tight_layout()
    save_figure("10_segment_satisfaction.png")


# ══════════════════════════════════════════════════════════
# TASK 4 — RECOMMENDATION AND REVIEW INSIGHTS
# ══════════════════════════════════════════════════════════

def plot_rec_vs_satisfaction(df: pd.DataFrame) -> None:
    """
    Task 4.1 — Does recommendation helpfulness
    affect shopping satisfaction?
    Higher helpfulness → higher satisfaction?
    """
    avg_sat = (df.groupby("Recommendation_Helpfulness")
               ["Shopping_Satisfaction"]
               .mean()
               .sort_values())

    plt.figure(figsize=(8, 4))
    avg_sat.plot(kind="bar",
                 color=COLORS[0],
                 edgecolor="white")
    plt.title("Avg Satisfaction by Recommendation Helpfulness",
              fontweight="bold")
    plt.xlabel("Recommendation Helpfulness")
    plt.ylabel("Avg Shopping Satisfaction (1–5)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save_figure("11_rec_vs_satisfaction.png")


def plot_review_reliability(df: pd.DataFrame) -> None:
    """
    Task 4.2 — How review reliability affects
    rating accuracy perception.
    """
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df,
                x="Review_Reliability",
                y="Rating_Accuracy",
                palette="coolwarm")
    plt.title("Rating Accuracy by Review Reliability",
              fontweight="bold")
    plt.xlabel("Review Reliability")
    plt.ylabel("Rating Accuracy (1–5)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    save_figure("12_review_reliability.png")


def plot_recommendation_engagement(df: pd.DataFrame) -> None:
    """
    Task 4.3 — How often customers engage with
    personalized recommendations.
    """
    rec_counts = (df["Personalized_Recommendation_Frequency"]
                  .value_counts())

    plt.figure(figsize=(7, 5))
    plt.pie(rec_counts,
            labels=rec_counts.index,
            autopct="%1.1f%%",
            colors=sns.color_palette("pastel"),
            startangle=140)
    plt.title("Personalized Recommendation Engagement",
              fontweight="bold")
    plt.tight_layout()
    save_figure("13_recommendation_engagement.png")


# ══════════════════════════════════════════════════════════
# TASK 5 — SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════

def plot_dashboard(df: pd.DataFrame,
                   cat_df: pd.DataFrame) -> None:
    """
    Task 5 — 2x3 Summary Dashboard combining
    key insights from all tasks into one view.

    Panels:
        [0,0] Top Purchase Categories
        [0,1] Browsing Frequency
        [0,2] Gender Distribution
        [1,0] Satisfaction Levels
        [1,1] Rec Helpfulness vs Satisfaction
        [1,2] Correlation Heatmap
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Amazon Customer Behaviour — Summary Dashboard",
        fontsize=16, fontweight="bold"
    )

    # ── Panel 1: Top 5 Categories ─────────────────────────
    top5 = cat_df.head(5)
    axes[0, 0].barh(top5["Category"], top5["Count"],
                    color=COLORS[0])
    axes[0, 0].set_title("Top Purchase Categories")
    axes[0, 0].invert_yaxis()
    axes[0, 0].tick_params(axis="y", labelsize=7)

    # ── Panel 2: Browsing Frequency ───────────────────────
    browse = df["Browsing_Frequency"].value_counts()
    axes[0, 1].pie(browse,
                   labels=browse.index,
                   autopct="%1.0f%%",
                   colors=sns.color_palette("pastel"))
    axes[0, 1].set_title("Browsing Frequency")

    # ── Panel 3: Gender Split ─────────────────────────────
    gender = df["Gender"].value_counts()
    axes[0, 2].pie(gender,
                   labels=gender.index,
                   autopct="%1.0f%%",
                   colors=COLORS[:len(gender)])
    axes[0, 2].set_title("Gender Distribution")

    # ── Panel 4: Satisfaction Levels ──────────────────────
    sat = df["Shopping_Satisfaction"].value_counts().sort_index()
    axes[1, 0].bar(sat.index, sat.values, color=COLORS[0])
    axes[1, 0].set_title("Satisfaction Levels (1–5)")
    axes[1, 0].set_xlabel("Score")
    axes[1, 0].set_ylabel("Customers")

    # ── Panel 5: Rec Helpfulness vs Satisfaction ──────────
    avg = (df.groupby("Recommendation_Helpfulness")
           ["Shopping_Satisfaction"].mean().sort_values())
    axes[1, 1].bar(avg.index, avg.values, color=COLORS[2])
    axes[1, 1].set_title("Rec. Helpfulness → Satisfaction")
    axes[1, 1].set_ylabel("Avg Satisfaction")
    axes[1, 1].tick_params(axis="x", rotation=20,
                            labelsize=7)

    # ── Panel 6: Correlation Heatmap ──────────────────────
    corr_cols = [
        "Purchase_Freq_Score",
        "Shopping_Satisfaction",
        "Customer_Reviews_Importance",
        "Rating_Accuracy"
    ]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f",
                cmap="YlOrRd", ax=axes[1, 2],
                linewidths=0.5,
                annot_kws={"size": 8})
    axes[1, 2].set_title("Correlation Heatmap")
    axes[1, 2].tick_params(axis="x", rotation=30,
                            labelsize=7)
    axes[1, 2].tick_params(axis="y", rotation=0,
                            labelsize=7)

    plt.tight_layout()
    save_figure("00_summary_dashboard.png")
    print("✅ Dashboard saved!")