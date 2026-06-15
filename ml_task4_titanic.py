"""
================================================================
  Kodbud ML Internship — Task 4: Titanic Survival Prediction
  Models  : Logistic Regression + Random Forest
  Dataset : Titanic (seaborn built-in)
  Output  : task4_plots/
================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, accuracy_score)

warnings.filterwarnings("ignore")
os.makedirs("task4_plots", exist_ok=True)

DARK = "#0D1117"; CARD = "#161B22"; ACC = "#00E5FF"
RED  = "#FF3B6B"; GRN  = "#39FF14"; YLW = "#FFB300"; TXT = "#CDE4F5"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD,
    "axes.edgecolor": "#21262D", "axes.labelcolor": TXT,
    "xtick.color": TXT, "ytick.color": TXT, "text.color": TXT,
    "grid.color": "#21262D", "grid.alpha": 0.5, "font.family": "monospace",
})

print("=" * 60)
print("  TASK 4: Titanic Survival Prediction")
print("=" * 60)

# ── 1. Load Data ──────────────────────────────────────────────────
print("\n[1/6] Loading Titanic dataset...")
df = sns.load_dataset("titanic")
print(f"      Shape: {df.shape}  |  Survived: {df['survived'].mean()*100:.1f}%")
print(f"      Missing values:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")

# ── 2. EDA ────────────────────────────────────────────────────────
print("\n[2/6] Generating EDA plots...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=DARK)
fig.suptitle("Task 4 — Titanic EDA", color=ACC, fontsize=14, fontweight="bold")

# Survival count
ax = axes[0, 0]
surv_counts = df["survived"].value_counts()
bars = ax.bar(["Died", "Survived"], surv_counts.values,
              color=[RED, GRN], edgecolor=DARK, width=0.5)
ax.set_title("Survival Count", color=ACC)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+5,
            str(b.get_height()), ha="center", color=TXT, fontsize=11)

# Survival by class
ax = axes[0, 1]
surv_class = df.groupby("pclass")["survived"].mean() * 100
ax.bar(["1st Class", "2nd Class", "3rd Class"], surv_class.values,
       color=[GRN, YLW, RED], edgecolor=DARK, width=0.5)
ax.set_title("Survival Rate by Class", color=ACC)
ax.set_ylabel("Survival Rate (%)"); ax.set_ylim(0, 100)

# Survival by sex
ax = axes[0, 2]
surv_sex = df.groupby("sex")["survived"].mean() * 100
ax.bar(surv_sex.index, surv_sex.values, color=[ACC, RED], edgecolor=DARK, width=0.4)
ax.set_title("Survival Rate by Sex", color=ACC)
ax.set_ylabel("Survival Rate (%)"); ax.set_ylim(0, 100)

# Age distribution by survival
ax = axes[1, 0]
df[df["survived"]==1]["age"].dropna().plot.hist(ax=ax, alpha=0.7, bins=30,
    color=GRN, label="Survived", edgecolor=DARK)
df[df["survived"]==0]["age"].dropna().plot.hist(ax=ax, alpha=0.7, bins=30,
    color=RED, label="Died", edgecolor=DARK)
ax.set_title("Age Distribution by Survival", color=ACC)
ax.set_xlabel("Age"); ax.legend(fontsize=8)

# Fare distribution
ax = axes[1, 1]
df[df["survived"]==1]["fare"].plot.hist(ax=ax, alpha=0.7, bins=40,
    color=GRN, label="Survived", edgecolor=DARK)
df[df["survived"]==0]["fare"].plot.hist(ax=ax, alpha=0.7, bins=40,
    color=RED, label="Died", edgecolor=DARK)
ax.set_title("Fare Distribution by Survival", color=ACC)
ax.set_xlabel("Fare"); ax.legend(fontsize=8)

# Embarkation survival
ax = axes[1, 2]
surv_emb = df.groupby("embark_town")["survived"].mean() * 100
surv_emb.plot.bar(ax=ax, color=ACC, edgecolor=DARK)
ax.set_title("Survival Rate by Embarkation", color=ACC)
ax.set_ylabel("Survival Rate (%)"); ax.set_xticklabels(ax.get_xticklabels(), rotation=20)

plt.tight_layout()
plt.savefig("task4_plots/eda.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("      Saved: task4_plots/eda.png")

# ── 3. Preprocessing ──────────────────────────────────────────────
print("\n[3/6] Preprocessing...")
df_clean = df[["survived","pclass","sex","age","sibsp","parch","fare","embark_town"]].copy()
df_clean["age"].fillna(df_clean["age"].median(), inplace=True)
df_clean["fare"].fillna(df_clean["fare"].median(), inplace=True)
df_clean["embark_town"].fillna("Southampton", inplace=True)
df_clean["sex"] = LabelEncoder().fit_transform(df_clean["sex"])
df_clean["embark_town"] = LabelEncoder().fit_transform(df_clean["embark_town"])
df_clean.dropna(inplace=True)

X = df_clean.drop("survived", axis=1).values
y = df_clean["survived"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")
print(f"      Class balance — Survived: {y_train.mean()*100:.1f}%")

# ── 4. Train Models ───────────────────────────────────────────────
print("\n[4/6] Training models...")
feature_names = ["pclass","sex","age","sibsp","parch","fare","embark_town"]
lr  = LogisticRegression(max_iter=500, random_state=42)
rf  = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
lr.fit(X_train_s, y_train)
rf.fit(X_train_s, y_train)

lr_acc  = accuracy_score(y_test, lr.predict(X_test_s))
rf_acc  = accuracy_score(y_test, rf.predict(X_test_s))
lr_cv   = cross_val_score(lr, X_train_s, y_train, cv=5).mean()
rf_cv   = cross_val_score(rf, X_train_s, y_train, cv=5).mean()
print(f"      Logistic Regression — Accuracy: {lr_acc:.4f}  CV: {lr_cv:.4f}")
print(f"      Random Forest       — Accuracy: {rf_acc:.4f}  CV: {rf_cv:.4f}")

best_model = rf if rf_acc >= lr_acc else lr
best_name  = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
y_pred     = best_model.predict(X_test_s)
y_prob     = best_model.predict_proba(X_test_s)[:, 1]

# ── 5. Results Plots ──────────────────────────────────────────────
print("\n[5/6] Generating results plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=DARK)
fig.suptitle(f"Task 4 — Results: {best_name}", color=ACC, fontsize=14, fontweight="bold")

# Confusion matrix
ax = axes[0, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Died","Survived"], yticklabels=["Died","Survived"],
            annot_kws={"size": 14, "color": "white"}, linewidths=1, linecolor=DARK)
ax.set_title("Confusion Matrix", color=ACC)
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

# ROC curve (both models)
ax = axes[0, 1]
for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
    prob = model.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    col = ACC if name == "Random Forest" else YLW
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
ax.plot([0,1],[0,1], color=RED, lw=1, linestyle="--", label="Random")
ax.set_title("ROC Curve", color=ACC)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Feature importance (RF)
ax = axes[1, 0]
importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
colors_fi = [ACC if v == importance.max() else "#1E3A5F" for v in importance.values]
ax.barh(importance.index, importance.values, color=colors_fi, edgecolor=DARK)
ax.set_title("Feature Importances (Random Forest)", color=ACC)
ax.set_xlabel("Importance Score")

# Model accuracy comparison
ax = axes[1, 1]
names  = ["Logistic\nRegression", "Random\nForest"]
accs   = [lr_acc, rf_acc]
cvs    = [lr_cv, rf_cv]
x = np.arange(len(names)); w = 0.35
ax.bar(x-w/2, accs, w, color=ACC, alpha=0.85, label="Test Accuracy", edgecolor=DARK)
ax.bar(x+w/2, cvs, w, color=YLW, alpha=0.85, label="CV Accuracy", edgecolor=DARK)
ax.set_title("Model Comparison", color=ACC)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("Accuracy"); ax.set_ylim(0.6, 1.0)
ax.legend(fontsize=8)
for i, (a, c) in enumerate(zip(accs, cvs)):
    ax.text(i-w/2, a+0.005, f"{a:.3f}", ha="center", fontsize=9, color=TXT)
    ax.text(i+w/2, c+0.005, f"{c:.3f}", ha="center", fontsize=9, color=TXT)

plt.tight_layout()
plt.savefig("task4_plots/results.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()
print("      Saved: task4_plots/results.png")

print(f"\n[6/6] Classification Report ({best_name}):")
print(classification_report(y_test, y_pred, target_names=["Died","Survived"]))
print("-" * 60)
print(f"  Best Model  : {best_name}")
print(f"  Accuracy    : {max(lr_acc,rf_acc)*100:.2f}%")
print(f"  CV Accuracy : {max(lr_cv,rf_cv)*100:.2f}%")
print("-" * 60)
print("\n✅ Task 4 Complete — plots saved to task4_plots/")
