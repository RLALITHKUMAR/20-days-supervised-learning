import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("DAY 6: SVM CLASSIFIER")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1] Loading Breast Cancer Dataset...")
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name="Diagnosis")

print(f"    Samples: {X.shape[0]}")
print(f"    Features: {X.shape[1]}")
print(f"    Classes: {dict(zip(cancer.target_names, [np.sum(y==0), np.sum(y==1)]))}")

# ============================================================
# 2. EDA
# ============================================================
print("\n[2] Performing Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Class distribution
class_counts = y.value_counts()
colors = ["#e74c3c", "#3498db"]
axes[0, 0].bar(["Malignant", "Benign"], class_counts.values, color=colors, edgecolor="black")
axes[0, 0].set_title("Class Distribution")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(class_counts.values):
    axes[0, 0].text(i, v + 5, str(v), ha="center", fontweight="bold")

# Correlation heatmap (sample of features to avoid clutter)
sample_features = cancer.feature_names[::3]
corr_data = X[sample_features].copy()
corr_data["Target"] = y
sns.heatmap(corr_data.corr(), annot=False, cmap="coolwarm", ax=axes[0, 1], cbar_kws={"shrink": 0.8})
axes[0, 1].set_title("Feature Correlation Heatmap (Sample)")

# Feature distributions by class
top_features = ["mean radius", "mean texture", "mean perimeter"]
for i, feat in enumerate(top_features):
    sns.kdeplot(data=X, x=feat, hue=y, ax=axes[1, 0], fill=True, alpha=0.3, palette=["red", "blue"])
axes[1, 0].set_title("Feature Distributions by Class")
axes[1, 0].set_xlabel("Value")

# Box plot of first 10 features
X.boxplot(ax=axes[1, 1], rot=45, fontsize=6)
axes[1, 1].set_title("Feature Box Plots (First 10)")
axes[1, 1].set_ylabel("Value")

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
print("\n[4] Training SVM Classifier (RBF Kernel)...")
model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ============================================================
# 5. METRICS
# ============================================================
print("\n[5] Evaluation Metrics:")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"    Accuracy:  {accuracy:.4f}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall:    {recall:.4f}")
print(f"    F1 Score:  {f1:.4f}")
print(f"    ROC-AUC:   {roc_auc:.4f}")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n[6] Generating Visualizations...")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names,
            cbar_kws={"shrink": 0.8})
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/02_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/02_confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkblue", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random Classifier")
plt.fill_between(fpr, tpr, alpha=0.2, color="darkblue")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig("outputs/03_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/03_roc_curve.png")

# Decision Boundary with PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model_2d = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
model_2d.fit(X_train_pca, y_train)

h = 0.02
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train,
                       cmap="coolwarm", edgecolors="black", s=30)
plt.title(f"SVM Decision Boundary (PCA 2D)\nExplained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.colorbar(scatter, label="Class")
plt.savefig("outputs/04_decision_boundary.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/04_decision_boundary.png")

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5,
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
plt.title("Learning Curve - SVM Classifier")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/05_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/05_learning_curve.png")

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color="darkgreen", linewidth=2,
         label=f"PR Curve (AP = {avg_precision:.4f})")
plt.fill_between(recall_vals, precision_vals, alpha=0.2, color="darkgreen")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.savefig("outputs/06_precision_recall_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/06_precision_recall_curve.png")

# ============================================================
# 7. SAVE MODEL
# ============================================================
print("\n[7] Saving Model...")
joblib.dump(model, "outputs/svm_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
print("    Saved: outputs/svm_model.pkl")
print("    Saved: outputs/scaler.pkl")

print("\n" + "=" * 60)
print("DAY 6 COMPLETE")
print("=" * 60)
