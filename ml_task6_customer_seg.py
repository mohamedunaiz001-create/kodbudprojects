"""
================================================================
  Kodbud ML Internship — Task 6: Customer Segmentation (K-Means)
  Dataset : Synthetic Mall Customer data (realistic)
  Output  : task6_plots/
================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
os.makedirs("task6_plots", exist_ok=True)

DARK = "#0D1117"; CARD = "#161B22"
TXT  = "#CDE4F5"; GRID = "#21262D"
CLUSTER_COLORS = ["#00E5FF","#FF3B6B","#39FF14","#FFB300","#AB47BC"]

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": GRID,   "axes.labelcolor": TXT,
    "xtick.color": TXT,       "ytick.color": TXT,
    "text.color": TXT,        "grid.color": GRID,
    "grid.alpha": 0.5,        "font.family": "monospace",
})

print("=" * 60)
print("  TASK 6: Customer Segmentation — K-Means Clustering")
print("=" * 60)

# ── 1. Generate Realistic Mall Customer Dataset ───────────────────
print("\n[1/6] Generating mall customer dataset...")
np.random.seed(42)
N = 500

# 5 distinct customer segments
segments = {
    "Low Income Low Spend":    dict(inc=(20,35),  spend=(10,30),  n=100),
    "Low Income High Spend":   dict(inc=(20,40),  spend=(60,99),  n=80),
    "Mid Income Mid Spend":    dict(inc=(40,70),  spend=(40,65),  n=140),
    "High Income Low Spend":   dict(inc=(70,100), spend=(5,30),   n=90),
    "High Income High Spend":  dict(inc=(70,100), spend=(70,99),  n=90),
}

rows = []
for seg_name, params in segments.items():
    n = params["n"]
    inc   = np.random.uniform(*params["inc"],   n)
    spend = np.random.uniform(*params["spend"],  n)
    age   = np.random.randint(18, 70, n)
    gender = np.random.choice(["Male","Female"], n)
    rows.append(pd.DataFrame({
        "CustomerID":       range(len(rows)*100, len(rows)*100+n),
        "Gender":           gender,
        "Age":              age,
        "AnnualIncome_k":   inc.round(1),
        "SpendingScore":    spend.round(1),
        "TrueSegment":      seg_name,
    }))

df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
df["CustomerID"] = range(1, len(df)+1)
print(f"      Dataset: {df.shape[0]} customers, {df.shape[1]} features")
print(df[["Age","AnnualIncome_k","SpendingScore"]].describe().round(2).to_string())

# ── 2. EDA ────────────────────────────────────────────────────────
print("\n[2/6] Generating EDA plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
fig.suptitle("Task 6 — Mall Customer EDA", color="#00E5FF", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.hist(df["AnnualIncome_k"], bins=30, color="#00E5FF", alpha=0.8, edgecolor=DARK)
ax.set_title("Annual Income Distribution", color="#00E5FF")
ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Count")

ax = axes[0, 1]
ax.hist(df["SpendingScore"], bins=30, color="#39FF14", alpha=0.8, edgecolor=DARK)
ax.set_title("Spending Score Distribution", color="#00E5FF")
ax.set_xlabel("Spending Score (1-100)"); ax.set_ylabel("Count")

ax = axes[1, 0]
male = df[df["Gender"]=="Male"]["SpendingScore"]
female = df[df["Gender"]=="Female"]["SpendingScore"]
ax.hist(male, bins=25, color="#00E5FF", alpha=0.7, label="Male", edgecolor=DARK)
ax.hist(female, bins=25, color="#FF3B6B", alpha=0.7, label="Female", edgecolor=DARK)
ax.set_title("Spending Score by Gender", color="#00E5FF")
ax.set_xlabel("Spending Score"); ax.legend(fontsize=9)

ax = axes[1, 1]
ax.scatter(df["AnnualIncome_k"], df["SpendingScore"],
           c="#00E5FF", alpha=0.4, s=12, edgecolors="none")
ax.set_title("Income vs Spending (Raw)", color="#00E5FF")
ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score")

plt.tight_layout()
plt.savefig("task6_plots/eda.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("      Saved: task6_plots/eda.png")

# ── 3. Elbow Method + Silhouette ──────────────────────────────────
print("\n[3/6] Finding optimal K (Elbow + Silhouette)...")
features_2d = ["AnnualIncome_k", "SpendingScore"]
X_2d = df[features_2d].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_2d)

k_range = range(2, 11)
inertias, silhouettes = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK)
fig.suptitle("Task 6 — Optimal K Selection", color="#00E5FF", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(list(k_range), inertias, "o-", color="#00E5FF", lw=2.5, markersize=7)
ax.axvline(x=5, color="#FF3B6B", lw=1.5, linestyle="--", label="Optimal K=5")
ax.set_title("Elbow Method — Inertia", color="#00E5FF")
ax.set_xlabel("Number of Clusters (K)"); ax.set_ylabel("Inertia")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(list(k_range), silhouettes, "s-", color="#39FF14", lw=2.5, markersize=7)
ax.axvline(x=5, color="#FF3B6B", lw=1.5, linestyle="--", label="Optimal K=5")
ax.set_title("Silhouette Score", color="#00E5FF")
ax.set_xlabel("Number of Clusters (K)"); ax.set_ylabel("Silhouette Score")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("task6_plots/elbow.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"      Saved: task6_plots/elbow.png")
print(f"      Best K by silhouette: {list(k_range)[np.argmax(silhouettes)]}")

# ── 4. Final Clustering (K=5) ─────────────────────────────────────
print("\n[4/6] Running K-Means with K=5...")
OPTIMAL_K = 5
km_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df["Cluster"] = km_final.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, df["Cluster"])
print(f"      Silhouette Score: {sil:.4f}")

# Cluster labels based on centroids
centroids = scaler.inverse_transform(km_final.cluster_centers_)
cluster_labels = {}
for i, (inc, spend) in enumerate(centroids):
    if inc < 45 and spend < 45:
        cluster_labels[i] = "Budget Savers"
    elif inc < 45 and spend >= 45:
        cluster_labels[i] = "Impulsive Buyers"
    elif 45 <= inc <= 70:
        cluster_labels[i] = "Average Customers"
    elif inc > 70 and spend < 45:
        cluster_labels[i] = "Careful Spenders"
    else:
        cluster_labels[i] = "Premium Customers"

df["Segment"] = df["Cluster"].map(cluster_labels)

# ── 5. Cluster Visualization ──────────────────────────────────────
print("\n[5/6] Generating cluster visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK)
fig.suptitle("Task 6 — K-Means Clustering Results (K=5)", color="#00E5FF",
             fontsize=14, fontweight="bold")

# 2D cluster plot
ax = axes[0]
for i in range(OPTIMAL_K):
    mask = df["Cluster"] == i
    ax.scatter(df[mask]["AnnualIncome_k"], df[mask]["SpendingScore"],
               c=CLUSTER_COLORS[i], label=cluster_labels[i],
               alpha=0.75, s=35, edgecolors="none")
# Plot centroids
for i, (inc, spend) in enumerate(centroids):
    ax.scatter(inc, spend, c=CLUSTER_COLORS[i], s=250, marker="*",
               edgecolors="white", linewidths=1.5, zorder=10)
ax.set_title("Customer Segments\n(★ = Cluster Centroid)", color="#00E5FF", fontsize=11)
ax.set_xlabel("Annual Income (k$)"); ax.set_ylabel("Spending Score (1-100)")
ax.legend(fontsize=8, loc="upper left")
ax.text(0.98, 0.02, f"Silhouette: {sil:.3f}", transform=ax.transAxes,
        ha="right", color="#39FF14", fontsize=9)

# Cluster size + stats bar chart
ax = axes[1]
seg_stats = df.groupby("Segment").agg(
    Count=("Cluster","count"),
    AvgIncome=("AnnualIncome_k","mean"),
    AvgSpend=("SpendingScore","mean"),
).reset_index()
x = np.arange(len(seg_stats))
w = 0.3
bars1 = ax.bar(x-w, seg_stats["AvgIncome"], w, color="#00E5FF", alpha=0.85,
               label="Avg Income (k$)", edgecolor=DARK)
bars2 = ax.bar(x,   seg_stats["AvgSpend"],  w, color="#39FF14", alpha=0.85,
               label="Avg Spending Score", edgecolor=DARK)
bars3 = ax.bar(x+w, seg_stats["Count"],     w, color="#FFB300", alpha=0.85,
               label="Count", edgecolor=DARK)
ax.set_title("Segment Statistics", color="#00E5FF", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(seg_stats["Segment"], rotation=15, ha="right", fontsize=8)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("task6_plots/clusters.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("      Saved: task6_plots/clusters.png")

# ── 6. Summary ────────────────────────────────────────────────────
print("\n[6/6] Cluster Summary:")
print("-" * 60)
summary = df.groupby("Segment").agg(
    Count=("Cluster","count"),
    Avg_Income=("AnnualIncome_k","mean"),
    Avg_Spend=("SpendingScore","mean"),
    Avg_Age=("Age","mean"),
).round(1)
print(summary.to_string())
print("-" * 60)
print(f"  Optimal K        : {OPTIMAL_K}")
print(f"  Silhouette Score : {sil:.4f}  (>0.5 = good separation)")
print("-" * 60)
print("\n✅ Task 6 Complete — plots saved to task6_plots/")
