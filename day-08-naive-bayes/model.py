import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("DAY 8: NAIVE BAYES CLASSIFIER")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1] Loading Iris Dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="Species")

print(f"    Samples: {X.shape[0]}")
print(f"    Features: {X.shape[1]}")
print(f"    Classes: {dict(zip(iris.target_names, [np.sum(y==i) for i in range(3)]))}")

# ============================================================
# 2. EDA
# ============================================================
print("\n[2] Performing Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Class distribution
class_counts = y.value_counts().sort_index()
colors = ["#e74c3c", "#3498db", "#2ecc71"]
axes[0, 0].bar(iris.target_names, class_counts.values, color=colors, edgecolor="black")
axes[0, 0].set_title("Class Distribution")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_xticklabels(iris.target_names, rotation=15)
for i, v in enumerate(class_counts.values):
    axes[0, 0].text(i, v + 1, str(v), ha="center", fontweight="bold")

# Correlation heatmap
corr = X.copy()
corr["Species"] = y
sns.heatmap(corr.corr(), annot=True, cmap="coolwarm", ax=axes[0, 1],
            cbar_kws={"shrink": 0.8}, fmt=".2f")
axes[0, 1].set_title("Feature Correlation Heatmap")

# Feature distributions by class
for i, feat in enumerate(iris.feature_names[:2]):
    for cls in range(3):
        mask = y == cls
        axes[1, 0].hist(X.loc[mask, feat], bins=15, alpha=0.5,
                        label=iris.target_names[cls], color=colors[cls], edgecolor="black")
axes[1, 0].set_title("Feature Distributions by Class")
axes[1, 0].set_xlabel("Value")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].legend(fontsize=8)

# Scatter plot of two most discriminative features
axes[1, 1].scatter(X["petal length (cm)"], X["petal width (cm)"],
                   c=y, cmap="viridis", edgecolors="black", s=50, alpha=0.8)
axes[1, 1].set_title("Petal Length vs Petal Width")
axes[1, 1].set_xlabel("Petal Length (cm)")
axes[1, 1].set_ylabel("Petal Width (cm)")
scatter_legend = axes[1, 1].scatter([], [], c=[], cmap="viridis")
axes[1, 1].legend(handles=[plt.Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=colors[i], markersize=8, label=iris.target_names[i])
                           for i in range(3)], fontsize=8)

plt.tight_layout()
plt.savefig("outputs/01_eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/01_eda_plots.png")

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n[3] Preprocessing Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    Train set: {X_train_scaled.shape[0]} samples")
print(f"    Test set:  {X_test_scaled.shape[0]} samples")

# ============================================================
# 4. TRAIN MODEL
# ============================================================
print("\n[4] Training Gaussian Naive Bayes...")
model = GaussianNB()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# ============================================================
# 5. METRICS
# ============================================================
print("\n[5] Evaluation Metrics:")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# ROC-AUC (one-vs-rest for multiclass)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
roc_auc = roc_auc_score(y_test_bin, y_prob, average="weighted", multi_class="ovr")

print(f"    Accuracy:  {accuracy:.4f}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall:    {recall:.4f}")
print(f"    F1 Score:  {f1:.4f}")
print(f"    ROC-AUC:   {roc_auc:.4f}")

# Per-class metrics
print("\n    Per-class Metrics:")
for cls in range(3):
    p = precision_score(y_test, y_pred, labels=[cls], average="micro")
    r = recall_score(y_test, y_pred, labels=[cls], average="micro")
    f = f1_score(y_test, y_pred, labels=[cls], average="micro")
    print(f"      {iris.target_names[cls]:<12}: P={p:.3f}, R={r:.3f}, F1={f:.3f}")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n[6] Generating Visualizations...")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            cbar_kws={"shrink": 0.8})
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/02_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/02_confusion_matrix.png")

# Feature Distributions by Class
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, feature in enumerate(iris.feature_names):
    for cls in range(3):
        mask = y == cls
        sns.kdeplot(data=X.loc[mask], x=feature, ax=axes[i],
                    fill=True, alpha=0.4, label=iris.target_names[cls], color=colors[cls])
    axes[i].set_title(f"{feature} by Class")
    axes[i].legend(fontsize=8)
    axes[i].set_xlabel("Value")

plt.tight_layout()
plt.savefig("outputs/03_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/03_feature_distributions.png")

# ROC Curves (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

plt.figure(figsize=(8, 6))
for i, cls_name in enumerate(iris.target_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, color=colors[i], linewidth=2,
             label=f"{cls_name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.savefig("outputs/04_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/04_roc_curves.png")

# Decision Boundary with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model_2d = GaussianNB()
model_2d.fit(X_train_pca, y_train)

h = 0.02
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train,
                       cmap="viridis", edgecolors="black", s=50)
plt.title(f"Gaussian NB Decision Boundary (PCA 2D)\nExplained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.colorbar(scatter, label="Class")
plt.legend(handles=[plt.Line2D([0], [0], marker="o", color="w",
                                markerfacecolor=colors[i], markersize=8, label=iris.target_names[i])
                    for i in range(3)], loc="lower right")
plt.savefig("outputs/05_decision_boundary.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/05_decision_boundary.png")

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy", n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red")
plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training Accuracy")
plt.plot(train_sizes, val_mean, "o-", color="red", label="Validation Accuracy")
plt.title("Learning Curve - Gaussian Naive Bayes")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/06_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/06_learning_curve.png")

# ============================================================
# 7. GAUSSIAN PARAMETERS
# ============================================================
print("\n[7] Gaussian Parameters (Mean and Variance per class):")
for i, cls_name in enumerate(iris.target_names):
    print(f"\n    {cls_name}:")
    print(f"      Means:     {model.theta_[i]}")
    print(f"      Variances: {model.var_[i]}")

# ============================================================
# 8. SAVE MODEL
# ============================================================
print("\n[8] Saving Model...")
joblib.dump(model, "outputs/naive_bayes_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
print("    Saved: outputs/naive_bayes_model.pkl")
print("    Saved: outputs/scaler.pkl")

print("\n" + "=" * 60)
print("DAY 8 COMPLETE")
print("=" * 60)
