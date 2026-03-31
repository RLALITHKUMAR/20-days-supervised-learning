import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize
import joblib
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 10: RANDOM FOREST CLASSIFIER")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────
print("\n[1/7] Loading Digits dataset...")
digits = load_digits()
X = pd.DataFrame(digits.data)
y = digits.target

print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(digits.target_names)}")

# ── 2. EDA ────────────────────────────────────────────────────
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/01_sample_digits.png", dpi=150)
plt.close()
print("  Saved: 01_sample_digits.png")

fig, ax = plt.subplots(figsize=(8, 5))
class_counts = pd.Series(y).value_counts().sort_index()
ax.bar(class_counts.index, class_counts.values, color="steelblue", edgecolor="black")
ax.set_xlabel("Digit")
ax.set_ylabel("Count")
ax.set_title("Class Distribution")
ax.set_xticks(range(10))
plt.tight_layout()
plt.savefig("outputs/02_class_distribution.png", dpi=150)
plt.close()
print("  Saved: 02_class_distribution.png")

fig, ax = plt.subplots(figsize=(10, 8))
sample_data = pd.DataFrame(X.values, columns=[f"pixel_{i}" for i in range(64)])
sample_data["target"] = y
corr = sample_data.corr()["target"].iloc[:-1].abs()
top_features = corr.nlargest(20).index
sns.heatmap(X[top_features].corr(), cmap="coolwarm", center=0, ax=ax)
ax.set_title("Top 20 Features Correlation")
plt.tight_layout()
plt.savefig("outputs/03_feature_correlation.png", dpi=150)
plt.close()
print("  Saved: 03_feature_correlation.png")

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
print("\n[4/7] Tuning n_estimators...")
estimator_range = [50, 100, 200, 300, 500]
train_scores = []
val_scores = []

for n in estimator_range:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    train_scores.append(rf.score(X_train_scaled, y_train))
    val_scores.append(rf.score(X_test_scaled, y_test))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(estimator_range, train_scores, "o-", label="Train Accuracy", color="steelblue")
ax.plot(estimator_range, val_scores, "s-", label="Val Accuracy", color="coral")
ax.set_xlabel("Number of Estimators")
ax.set_ylabel("Accuracy")
ax.set_title("n_estimators Tuning")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/04_n_estimators_tuning.png", dpi=150)
plt.close()
print("  Saved: 04_n_estimators_tuning.png")

best_n = estimator_range[np.argmax(val_scores)]
print(f"  Best n_estimators: {best_n}")

# ── 5. Train Final Model ──────────────────────────────────────
print("\n[5/7] Training final model...")
model = RandomForestClassifier(
    n_estimators=best_n,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# ── 6. Metrics ────────────────────────────────────────────────
print("\n[6/7] Computing metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

y_test_bin = label_binarize(y_test, classes=range(10))
roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="weighted")

print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred))

# ── 7. Visualizations ─────────────────────────────────────────
print("\n[7/7] Generating visualizations...")

fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/05_confusion_matrix.png", dpi=150)
plt.close()
print("  Saved: 05_confusion_matrix.png")

fig, axes = plt.subplots(3, 5, figsize=(12, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    idx = np.random.choice(range(len(X_test)))
    ax.imshow(X_test.iloc[idx].values.reshape(8, 8), cmap="gray")
    pred = y_pred[idx]
    actual = y_test.iloc[idx]
    color = "green" if pred == actual else "red"
    ax.set_title(f"Pred:{pred}\nAct:{actual}", color=color, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/06_sample_predictions.png", dpi=150)
plt.close()
print("  Saved: 06_sample_predictions.png")

fig, ax = plt.subplots(figsize=(10, 6))
importances = model.feature_importances_
pixel_importance = importances.reshape(8, 8)
im = ax.imshow(pixel_importance, cmap="YlOrRd")
ax.set_title("Feature Importance (Pixel Level)")
plt.colorbar(im, ax=ax, label="Importance")
plt.tight_layout()
plt.savefig("outputs/07_feature_importance.png", dpi=150)
plt.close()
print("  Saved: 07_feature_importance.png")

train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    RandomForestClassifier(n_estimators=best_n, random_state=42, n_jobs=-1),
    X_train_scaled, y_train,
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

fig, ax = plt.subplots(figsize=(8, 6))
y_test_bin_full = label_binarize(y_test, classes=range(10))
for i in range(10):
    fpr, tpr, _ = [], [], []
    from sklearn.metrics import roc_curve as sk_roc_curve
    fpr_i, tpr_i, _ = sk_roc_curve(y_test_bin_full[:, i], y_prob[:, i])
    auc_i = roc_auc_score(y_test_bin_full[:, i], y_prob[:, i])
    ax.plot(fpr_i, tpr_i, lw=2, label=f"Digit {i} (AUC={auc_i:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=2)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (One-vs-Rest)")
ax.legend(fontsize=7, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/09_roc_curve.png", dpi=150)
plt.close()
print("  Saved: 09_roc_curve.png")

# ── 8. Save Model ─────────────────────────────────────────────
joblib.dump(model, "outputs/random_forest_classifier.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
print("\n  Saved: random_forest_classifier.pkl")
print("  Saved: scaler.pkl")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
