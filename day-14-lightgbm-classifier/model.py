import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import lightgbm as lgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("DAY 14: LightGBM Classifier - Wine Dataset")
print("=" * 60)

# --- 1. Load Data ---
print("\n[1/7] Loading Wine dataset...")
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="Class")

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Classes: {wine.target_names}")
print(f"Class distribution:\n{y.value_counts().sort_index()}")

# --- 2. EDA ---
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(3, 5, figsize=(22, 14))
axes = axes.flatten()
for i, col in enumerate(X.columns):
    sns.histplot(X[col], kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(f"{col}\n(skew={X[col].skew():.2f})")
    axes[i].set_xlabel("")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/01_feature_distributions.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(12, 10))
corr = X.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            square=True, linewidths=0.5)
plt.tight_layout()
plt.savefig("outputs/02_correlation_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x=y, palette="Set2", ax=ax)
ax.set_xticklabels([wine.target_names[i] for i in range(3)])
ax.set_xlabel("Wine Class")
ax.set_ylabel("Count")
ax.set_title("Class Distribution")
plt.tight_layout()
plt.savefig("outputs/03_class_distribution.png", dpi=150)
plt.close()

print("  EDA plots saved.")

# --- 3. Preprocessing ---
print("\n[3/7] Preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "outputs/scaler.pkl")
print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

# --- 4. Model Training ---
print("\n[4/7] Training LightGBM Classifier...")

param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}

best_score = 0
best_params = {}

for md in param_grid["max_depth"]:
    for lr in param_grid["learning_rate"]:
        model = lgb.LGBMClassifier(
            max_depth=md,
            learning_rate=lr,
            n_estimators=200,
            random_state=42,
            verbose=-1,
        )
        scores = cross_val_score(
            model, X_train_scaled, y_train, cv=3,
            scoring="accuracy"
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = {"max_depth": md, "learning_rate": lr}

print(f"  Best params: {best_params}")
print(f"  Best CV Accuracy: {best_score:.4f}")

model = lgb.LGBMClassifier(
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    n_estimators=200,
    random_state=42,
    verbose=-1,
)
model.fit(X_train_scaled, y_train)

# --- 5. Evaluation ---
print("\n[5/7] Evaluating model...")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

# --- 6. Visualizations ---
print("\n[6/7] Generating visualizations...")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=[wine.target_names[i] for i in range(3)],
            yticklabels=[wine.target_names[i] for i in range(3)])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/04_confusion_matrix.png", dpi=150)
plt.close()

# ROC Curve (One-vs-Rest)
fig, ax = plt.subplots(figsize=(8, 6))
for i, class_name in enumerate(wine.target_names):
    fpr, tpr, _ = RocCurveDisplay.from_predictions(
        y_test == i, y_prob[:, i], ax=ax, name=f"Class {i} ({class_name})"
    ).figure.axes[0].lines[-1].get_xydata().T if False else (None, None, None)
    RocCurveDisplay.from_predictions(y_test == i, y_prob[:, i], ax=ax,
                                      name=f"Class {i} ({class_name})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_title("ROC Curve (One-vs-Rest)")
plt.tight_layout()
plt.savefig("outputs/05_roc_curve.png", dpi=150)
plt.close()

# Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)
ax.barh(range(len(indices)), importances[indices], color="teal")
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([X.columns[i] for i in indices])
ax.set_xlabel("Feature Importance")
ax.set_title("LightGBM Feature Importance")
plt.tight_layout()
plt.savefig("outputs/06_feature_importance.png", dpi=150)
plt.close()

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    lgb.LGBMClassifier(
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        n_estimators=200,
        random_state=42,
        verbose=-1,
    ),
    X_train_scaled, y_train, cv=3,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Train Accuracy")
ax.plot(train_sizes, val_mean, "s-", color="darkorange", label="CV Accuracy")
ax.set_xlabel("Training Samples")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/07_learning_curve.png", dpi=150)
plt.close()

# Training History Plot
eval_results = {}
model_hist = lgb.LGBMClassifier(
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    n_estimators=200,
    random_state=42,
    verbose=-1,
)
model_hist.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    callbacks=[lgb.record_evaluation(eval_results)],
)

fig, ax = plt.subplots(figsize=(8, 6))
iterations = range(1, len(eval_results["training"]["multi_logloss"]) + 1)
ax.plot(iterations, eval_results["training"]["multi_logloss"],
        "o-", color="steelblue", label="Train Log Loss")
ax.plot(iterations, eval_results["valid_0"]["multi_logloss"],
        "s-", color="darkorange", label="Validation Log Loss")
ax.set_xlabel("Iteration")
ax.set_ylabel("Multi Log Loss")
ax.set_title("Training History by Iteration")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/08_training_history.png", dpi=150)
plt.close()

print("  All plots saved to outputs/")

# --- 7. Save Model ---
print("\n[7/7] Saving model...")
joblib.dump(model, "outputs/lightgbm_classifier.pkl")
print("  Model saved to outputs/lightgbm_classifier.pkl")

print("\n" + "=" * 60)
print("DONE — LightGBM Classifier complete!")
print("=" * 60)
