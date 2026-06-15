"""
================================================================
  Kodbud ML Internship — Task 8: Diabetes Prediction
  Model   : Logistic Regression + Random Forest (comparison)
  Dataset : PIMA Indians Diabetes (sklearn synthetic equivalent)
  Output  : task8_plots/
================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, accuracy_score, precision_recall_curve)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
os.makedirs("task8_plots", exist_ok=True)

DARK = "#0D1117"; CARD = "#161B22"; ACC = "#00E5FF"
RED  = "#FF3B6B"; GRN  = "#39FF14"; YLW = "#FFB300"; TXT = "#CDE4F5"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": "#21262D", "axes.labelcolor": TXT,
    "xtick.color": TXT, "ytick.color": TXT, "text.color": TXT,
    "grid.color": "#21262D", "grid.alpha": 0.5, "font.family": "monospace",
})

print("=" * 60)
print("  TASK 8: Diabetes Prediction — Logistic Regression")
print("=" * 60)

# ── 1. Generate Realistic PIMA-style Dataset ──────────────────────
print("\n[1/6] Building PIMA-style diabetes dataset...")
np.random.seed(42)
N = 800

def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)

# Non-diabetic (class 0)
n0 = 530
g0 = np.column_stack([
    clip(np.random.normal(2.5, 2.5, n0), 0, 12),     # Pregnancies
    clip(np.random.normal(109, 20, n0),  50, 165),    # Glucose
    clip(np.random.normal(70, 12, n0),   40, 110),    # BloodPressure
    clip(np.random.normal(25, 10, n0),   7, 60),      # SkinThickness
    clip(np.random.normal(90, 60, n0),   15, 400),    # Insulin
    clip(np.random.normal(27, 5, n0),    17, 55),     # BMI
    clip(np.random.normal(0.35, 0.2, n0),0.08, 1.5), # DiabetesPedigreeFunction
    clip(np.random.normal(32, 11, n0),   21, 65),     # Age
])

# Diabetic (class 1)
n1 = 270
g1 = np.column_stack([
    clip(np.random.normal(4.5, 3.0, n1), 0, 15),
    clip(np.random.normal(143, 25, n1),  70, 200),
    clip(np.random.normal(75, 13, n1),   40, 110),
    clip(np.random.normal(32, 12, n1),   7, 65),
    clip(np.random.normal(175, 90, n1),  15, 600),
    clip(np.random.normal(35, 6, n1),    18, 65),
    clip(np.random.normal(0.55, 0.28, n1),0.08, 2.4),
    clip(np.random.normal(40, 12, n1),   21, 75),
])

cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age"]

df = pd.DataFrame(
    np.vstack([g0, g1]),
    columns=cols
).round(2)
df["Outcome"] = [0]*n0 + [1]*n1
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"      Dataset: {df.shape[0]} patients | Diabetic: {df['Outcome'].mean()*100:.1f}%")
print(df.describe().round(2).to_string())

# ── 2. EDA ────────────────────────────────────────────────────────
print("\n[2/6] Generating EDA plots...")
fig, axes = plt.subplots(2, 4, figsize=(18, 10), facecolor=DARK)
fig.suptitle("Task 8 — Diabetes Dataset EDA", color=ACC, fontsize=14, fontweight="bold")

for i, col in enumerate(cols):
    ax = axes[i//4][i%4]
    df[df["Outcome"]==0][col].plot.hist(ax=ax, alpha=0.7, bins=25, color=GRN,
                                         label="No Diabetes", edgecolor=DARK)
    df[df["Outcome"]==1][col].plot.hist(ax=ax, alpha=0.7, bins=25, color=RED,
                                         label="Diabetes", edgecolor=DARK)
    ax.set_title(col, color=ACC, fontsize=9)
    ax.set_xlabel(""); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("task8_plots/eda.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("      Saved: task8_plots/eda.png")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK)
corr = df.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, center=0, ax=ax, annot=True, fmt=".2f",
            annot_kws={"size": 9}, linewidths=0.5, linecolor=DARK,
            square=True)
ax.set_title("Feature Correlation Matrix", color=ACC, fontsize=13)
plt.tight_layout()
plt.savefig("task8_plots/correlation.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("      Saved: task8_plots/correlation.png")

# ── 3. Preprocessing ──────────────────────────────────────────────
print("\n[3/6] Preprocessing...")
X = df[cols].values
y = df["Outcome"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 4. Train Models ───────────────────────────────────────────────
print("\n[4/6] Training models...")
lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
lr.fit(X_train_s, y_train)
rf.fit(X_train_s, y_train)

results = {}
for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
    y_p   = model.predict(X_test_s)
    y_pr  = model.predict_proba(X_test_s)[:,1]
    cv    = cross_val_score(model, X_train_s, y_train, cv=5).mean()
    fp, tp, _ = roc_curve(y_test, y_pr)
    results[name] = {
        "y_pred": y_p, "y_prob": y_pr,
        "acc": accuracy_score(y_test, y_p),
        "cv":  cv, "fpr": fp, "tpr": tp,
        "auc": auc(fp, tp),
    }
    print(f"      {name:<25}  Acc={results[name]['acc']:.4f}  "
          f"AUC={results[name]['auc']:.4f}  CV={cv:.4f}")

best_name = max(results, key=lambda k: results[k]["acc"])
best      = results[best_name]

# ── 5. Results Plots ──────────────────────────────────────────────
print("\n[5/6] Generating results plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
fig.suptitle(f"Task 8 — Diabetes Prediction Results ({best_name})",
             color=ACC, fontsize=13, fontweight="bold")

# Confusion matrix
ax = axes[0, 0]
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Diabetes","Diabetes"],
            yticklabels=["No Diabetes","Diabetes"],
            annot_kws={"size": 14, "color": "white"}, linewidths=1.5, linecolor=DARK)
ax.set_title(f"Confusion Matrix\nAccuracy: {best['acc']*100:.1f}%", color=ACC)
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

# ROC curves
ax = axes[0, 1]
colors_roc = [ACC, YLW]
for (name, res), col in zip(results.items(), colors_roc):
    ax.plot(res["fpr"], res["tpr"], color=col, lw=2.5,
            label=f"{name} (AUC={res['auc']:.3f})")
ax.fill_between(best["fpr"], best["tpr"], alpha=0.08, color=ACC)
ax.plot([0,1],[0,1], color=RED, lw=1, linestyle="--", label="Random")
ax.set_title("ROC Curve", color=ACC)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8)

# Feature importances
ax = axes[1, 0]
coef_abs = np.abs(lr.coef_[0])
importance = pd.Series(coef_abs, index=cols).sort_values()
bar_colors = [ACC if v == importance.max() else "#1E3A5F" for v in importance.values]
ax.barh(importance.index, importance.values, color=bar_colors, edgecolor=DARK)
ax.set_title("Feature Importance\n(|LR Coefficient|)", color=ACC)
ax.set_xlabel("Absolute Coefficient")

# Precision-Recall curve
ax = axes[1, 1]
prec, rec, _ = precision_recall_curve(y_test, best["y_prob"])
ax.plot(rec, prec, color=ACC, lw=2.5, label=best_name)
ax.fill_between(rec, prec, alpha=0.1, color=ACC)
ax.axhline(y_test.mean(), color=RED, lw=1, linestyle="--",
           label=f"Baseline (prevalence={y_test.mean():.2f})")
ax.set_title("Precision-Recall Curve", color=ACC)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("task8_plots/results.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("      Saved: task8_plots/results.png")

# ── 6. Summary ────────────────────────────────────────────────────
print(f"\n[6/6] Classification Report ({best_name}):")
print(classification_report(y_test, best["y_pred"],
      target_names=["No Diabetes","Diabetes"]))
print("-" * 60)
print(f"  Best Model   : {best_name}")
print(f"  Accuracy     : {best['acc']*100:.2f}%")
print(f"  ROC AUC      : {best['auc']:.4f}")
print(f"  CV Accuracy  : {best['cv']*100:.2f}%")
print("-" * 60)
print("\n  Top predictive features (by LR coefficient):")
top = pd.Series(lr.coef_[0], index=cols).abs().sort_values(ascending=False)
for feat, val in top.items():
    print(f"    {feat:<30} {val:.4f}")
print("-" * 60)
print("\n✅ Task 8 Complete — plots saved to task8_plots/")
