import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 12: XGBOOST CLASSIFIER")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────
print("\n[1/7] Loading Breast Cancer dataset...")
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {list(cancer.target_names)}")

# ── 2. EDA ────────────────────────────────────────────────────
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(X.columns[:10]):
    axes[i].hist(X[col], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig("outputs/01_feature_distributions.png", dpi=150)
plt.close()
print("  Saved: 01_feature_distributions.png")

fig, ax = plt.subplots(figsize=(10, 8))
corr = X.copy()
corr["target"] = y
sns.heatmap(corr.corr(), annot=False, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=150)
plt.close()
print("  Saved: 02_correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(8, 5))
class_counts = pd.Series(y).value_counts()
ax.bar([cancer.target_names[i] for i in class_counts.index], class_counts.values,
       color=["coral", "steelblue"], edgecolor="black")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.set_title("Class Distribution")
plt.tight_layout()
plt.savefig("outputs/03_class_distribution.png", dpi=150)
plt.close()
print("  Saved: 03_class_distribution.png")

# ── 3. Preprocessing ──────────────────────────────────────────
print("\n[3/7] Preprocessing data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  Train: {X_train_scaled.shape[0]} samples")
print(f"  Test:  {X_test_scaled.shape[0]} samples")

# ── 4. Hyperparameter Tuning ──────────────────────────────────
print("\n[4/7] Tuning max_depth and learning_rate...")
max_depths = [3, 5, 7, 9, 11]
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]

results = {}
for md in max_depths:
    for lr in learning_rates:
        model = xgb.XGBClassifier(
            max_depth=md,
            learning_rate=lr,
            n_estimators=200,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
        )
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        results[(md, lr)] = score

best_params = max(results, key=results.get)
best_md, best_lr = best_params
print(f"  Best max_depth: {best_md}")
print(f"  Best learning_rate: {best_lr}")
print(f"  Best Accuracy: {results[best_params]:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
for md in max_depths:
    scores = [results[(md, lr)] for lr in learning_rates]
    ax.plot(learning_rates, scores, "o-", label=f"max_depth={md}", linewidth=2)
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Accuracy")
ax.set_title("max_depth & learning_rate Tuning")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/04_hyperparameter_tuning.png", dpi=150)
plt.close()
print("  Saved: 04_hyperparameter_tuning.png")

# ── 5. Train Final Model ──────────────────────────────────────
print("\n[5/7] Training final model...")
model = xgb.XGBClassifier(
    max_depth=best_md,
    learning_rate=best_lr,
    n_estimators=200,
    random_state=42,
    verbosity=0,
    use_label_encoder=False,
)
model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    verbose=False,
)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ── 6. Metrics ────────────────────────────────────────────────
print("\n[6/7] Computing metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# ── 7. Visualizations ─────────────────────────────────────────
print("\n[7/7] Generating visualizations...")

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/05_confusion_matrix.png", dpi=150)
plt.close()
print("  Saved: 05_confusion_matrix.png")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"ROC (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=2)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/06_roc_curve.png", dpi=150)
plt.close()
print("  Saved: 06_roc_curve.png")

fig, ax = plt.subplots(figsize=(10, 8))
importances = model.feature_importances_
sorted_idx = np.argsort(importances)
ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="steelblue")
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([cancer.feature_names[i] for i in sorted_idx], fontsize=9)
ax.set_xlabel("Importance")
ax.set_title("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/07_feature_importance.png", dpi=150)
plt.close()
print("  Saved: 07_feature_importance.png")

train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    xgb.XGBClassifier(
        max_depth=best_md,
        learning_rate=best_lr,
        n_estimators=200,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    ),
    X_train_scaled,
    y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy",
    n_jobs=-1,
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_scores_lc.mean(axis=1), "o-", label="Train", color="steelblue")
ax.plot(train_sizes, val_scores_lc.mean(axis=1), "s-", label="Validation", color="coral")
ax.fill_between(
    train_sizes,
    train_scores_lc.mean(axis=1) - train_scores_lc.std(axis=1),
    train_scores_lc.mean(axis=1) + train_scores_lc.std(axis=1),
    alpha=0.15, color="steelblue",
)
ax.fill_between(
    train_sizes,
    val_scores_lc.mean(axis=1) - val_scores_lc.std(axis=1),
    val_scores_lc.mean(axis=1) + val_scores_lc.std(axis=1),
    alpha=0.15, color="coral",
)
ax.set_xlabel("Training Samples")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/08_learning_curve.png", dpi=150)
plt.close()
print("  Saved: 08_learning_curve.png")

results_dict = model.evals_result()
train_logloss = results_dict["validation_0"]["logloss"]
val_logloss = results_dict["validation_1"]["logloss"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(train_logloss) + 1), train_logloss, "o-", label="Train LogLoss", color="steelblue")
ax.plot(range(1, len(val_logloss) + 1), val_logloss, "s-", label="Validation LogLoss", color="coral")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("Log Loss")
ax.set_title("Training History")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/09_training_history.png", dpi=150)
plt.close()
print("  Saved: 09_training_history.png")

# ── 8. Save Model ─────────────────────────────────────────────
joblib.dump(model, "outputs/xgboost_classifier.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
print("\n  Saved: xgboost_classifier.pkl")
print("  Saved: scaler.pkl")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
