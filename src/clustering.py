"""
clustering.py
─────────────────────────────────────────────
Task 3: K-Means Clustering Model
─────────────────────────────────────────────
PURPOSE : Trains KMeans model on customer
          behaviour features, finds optimal K,
          and saves trained model for reuse.
AUTHOR  : Mrudula
─────────────────────────────────────────────
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ── STEP 1: Find Best K ───────────────────────────────────
def find_optimal_k(X_scaled,
                   k_range: range = range(2, 8),
                   save_path: str = "reports/figures/elbow_method.png"):
    """
    Tests K=2 to K=7 using two methods:

    METHOD 1 — Elbow Method:
        Plots inertia for each K value.
        Inertia = sum of squared distances from
        each customer to their cluster center.
        Where curve bends like elbow = best K.

    METHOD 2 — Silhouette Score:
        Measures how well each customer fits
        their assigned cluster vs other clusters.
        Score closer to 1.0 = better clusters.

    WHY USE BOTH:
        Elbow = visual and intuitive
        Silhouette = mathematical confirmation
        Together they give more confidence in K.

    Args:
        X_scaled  : scaled features from build_features.py
        k_range   : K values to test (default 2 to 7)
        save_path : where to save the plot

    Returns:
        inertias   : inertia value for each K
        sil_scores : silhouette score for each K
    """
    inertias   = []
    sil_scores = []

    print("🔍 Testing K values...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))
        print(f"   K={k} | Inertia: {km.inertia_:.1f} | "
              f"Silhouette: {silhouette_score(X_scaled, km.labels_):.3f}")

    # ── Plot Elbow + Silhouette side by side ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Finding Best K for KMeans Clustering",
                 fontweight="bold")

    # Elbow curve
    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2)
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    # Silhouette scores
    axes[1].plot(list(k_range), sil_scores, "gs-", linewidth=2)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Score (higher = better)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Elbow plot saved → {save_path}")

    return inertias, sil_scores


# ── STEP 2: Train Final KMeans Model ─────────────────────
def train_kmeans(X_scaled,
                 n_clusters: int = 3,
                 random_state: int = 42) -> tuple:
    """
    Trains final KMeans with chosen K value.

    WHY random_state=42:
        KMeans starts with random cluster centers.
        Fixing random_state makes results reproducible
        — same clusters every time you run the code.

    WHY n_init=10:
        Runs 10 times with different starting points
        and picks the best result. Avoids getting
        stuck in a bad local solution.

    Args:
        X_scaled     : scaled feature array
        n_clusters   : K value from config.yaml
        random_state : seed for reproducibility

    Returns:
        km     : trained KMeans model
        labels : cluster ID for each customer (0,1,2)
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = km.fit_predict(X_scaled)
    score  = silhouette_score(X_scaled, labels)

    print(f"\n✅ KMeans trained!")
    print(f"   K                = {n_clusters}")
    print(f"   Silhouette Score = {score:.3f}")
    print(f"\n   Cluster Distribution:")
    for cid, count in (pd.Series(labels)
                       .value_counts()
                       .sort_index()
                       .items()):
        print(f"     Cluster {cid} → {count} customers")

    return km, labels


# ── STEP 3: Save Trained Model ───────────────────────────
def save_model(model,
               path: str = "models/saved_models/kmeans.pkl"
               ) -> None:
    """
    Saves trained KMeans model to disk as .pkl file.

    WHY WE SAVE THE MODEL:
        In production you train once and predict many times.
        Without saving you would retrain every single time
        — wasteful and slow.

        joblib.dump() = clicking Save on your model.
        joblib.load() = opening it again instantly.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved → {path}")


# ── STEP 4: Load Saved Model ─────────────────────────────
def load_model(path: str = "models/saved_models/kmeans.pkl"):
    """
    Loads a previously saved KMeans model.
    Use this to predict clusters for new customers
    without retraining from scratch.

    Args:
        path : path to saved .pkl file

    Returns:
        Loaded KMeans model ready for prediction
    """
    model = joblib.load(path)
    print(f"✅ Model loaded → {path}")
    return model