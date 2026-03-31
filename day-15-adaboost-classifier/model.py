import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("DAY 15: AdaBoost Classifier - Breast Cancer")
print("=" * 60)

# --- 1. Load Data ---
print("\n[1/7] Loading Breast Cancer dataset...")
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name="Diagnosis")

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Classes: {cancer.target_names}")
print(f"Class distribution:\n{y.value_counts().sort_index()}")

# --- 2. EDA ---
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(3, 5, figsize=(22, 14))
axes = axes.flatten()
for i, col in enumerate(X.columns[:15]):
    sns.histplot(X[col], kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(f"{col[:15]}")
    axes[i].set_xlabel("")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/01_feature_distributions.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(14, 12))
corr = X.corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax,
            square=True, linewidths=0.3)
plt.tight_layout()
plt.savefig("outputs/02_correlation_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x=y, palette="Set2", ax=ax)
ax.set_xticklabels(cancer.target_names)
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Count")
ax.set_title("Class Distribution (0=Malignant, 1=Benign)")
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
print("\n[4/7] Training AdaBoost Classifier...")

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.5, 1.0],
}

best_score = 0
best_params = {}

for ne in param_grid["n_estimators"]:
    for lr in param_grid["learning_rate"]:
        base_estimator = DecisionTreeClassifier(max_depth=2, random_state=42)
        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=ne,
            learning_rate=lr,
            random_state=42,
        )
        scores = cross_val_score(
            model, X_train_scaled, y_train, cv=3,
            scoring="accuracy"
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = {"n_estimators": ne, "learning_rate": lr}

print(f"  Best params: {best_params}")
print(f"  Best CV Accuracy: {best_score:.4f}")

base_estimator = DecisionTreeClassifier(max_depth=2, random_state=42)
model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# --- 5. Evaluation ---
print("\n[5/7] Evaluating model...")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

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

# --- 6. Visualizations ---
print("\n[6/7] Generating visualizations...")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/04_confusion_matrix.png", dpi=150)
plt.close()

# ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="AdaBoost ROC")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_title("ROC Curve")
plt.tight_layout()
plt.savefig("outputs/05_roc_curve.png", dpi=150)
plt.close()

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        random_state=42,
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
plt.savefig("outputs/06_learning_curve.png", dpi=150)
plt.close()

# Estimator Weights Plot
fig, ax = plt.subplots(figsize=(10, 6))
weights = model.estimator_weights_
ax.bar(range(len(weights)), weights, color="teal", alpha=0.7)
ax.set_xlabel("Estimator Index")
ax.set_ylabel("Weight")
ax.set_title("AdaBoost Estimator Weights")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/07_estimator_weights.png", dpi=150)
plt.close()

# Decision Boundary with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model_pca = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    random_state=42,
)
model_pca.fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 8))
ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                     c=y_train, cmap="RdYlBu", edgecolor="k", s=50)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("Decision Boundary (PCA-reduced to 2D)")
plt.colorbar(scatter, ax=ax, label="Class")
plt.tight_layout()
plt.savefig("outputs/08_decision_boundary_pca.png", dpi=150)
plt.close()

print("  All plots saved to outputs/")

# --- 7. Save Model ---
print("\n[7/7] Saving model...")
joblib.dump(model, "outputs/adaboost_classifier.pkl")
print("  Model saved to outputs/adaboost_classifier.pkl")

print("\n" + "=" * 60)
print("DONE — AdaBoost Classifier complete!")
print("=" * 60)
