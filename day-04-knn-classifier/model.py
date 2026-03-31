"""
Day 4: KNN Classifier
Dataset: Wine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

print("="*60)
print("DAY 4: K-NEAREST NEIGHBORS CLASSIFIER")
print("="*60)

# Load Data
print("\n[1] Loading Wine Dataset...")
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print(f"    Samples: {X.shape[0]}")
print(f"    Features: {X.shape[1]}")
print(f"    Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

# EDA
print("\n[2] Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

sns.countplot(x=y, ax=axes[0,0], palette='Set2')
axes[0,0].set_xticks([0, 1, 2])
axes[0,0].set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
axes[0,0].set_title('Class Distribution')

sns.boxplot(data=X[['alcohol', 'malic_acid', 'ash']], ax=axes[0,1])
axes[0,1].set_title('Feature Distributions')
axes[0,1].tick_params(axis='x', rotation=45)

corr = X.corr().iloc[:5, :5]
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[0,2], fmt='.2f')
axes[0,2].set_title('Feature Correlation (Top 5)')

sns.scatterplot(x=X['alcohol'], y=X['proline'], hue=y, ax=axes[1,0], 
                palette='Set1', alpha=0.6)
axes[1,0].set_title('Alcohol vs Proline by Class')

sns.histplot(X['flavanoids'], kde=True, ax=axes[1,1], color='purple')
axes[1,1].set_title('Flavanoids Distribution')

sns.violinplot(x=y, y=X['color_intensity'], ax=axes[1,2], palette='Set2')
axes[1,2].set_xticks([0, 1, 2])
axes[1,2].set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
axes[1,2].set_title('Color Intensity by Class')

plt.tight_layout()
plt.savefig('outputs/04_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: outputs/04_eda.png")

# Preprocessing
print("\n[3] Preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    Train set: {X_train.shape[0]} samples")
print(f"    Test set: {X_test.shape[0]} samples")

# Find optimal K
print("\n[4] Finding Optimal K...")
k_range = range(1, 21)
k_scores = []
for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    cv_score = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    k_scores.append(cv_score)
    print(f"    K={k:2d}, CV Accuracy={cv_score:.4f}")

best_k = k_range[np.argmax(k_scores)]
print(f"\n    Best K: {best_k}")

# Plot K vs Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_range, k_scores, marker='o', color='blue')
ax.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
ax.set_xlabel('Number of Neighbors (K)')
ax.set_ylabel('Cross-Validation Accuracy')
ax.set_title('K vs Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/04_k_vs_accuracy.png', dpi=150)
plt.close()
print("    Saved: outputs/04_k_vs_accuracy.png")

# Training
print("\n[5] Training KNN Model...")
model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
model.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"    CV Accuracy Scores: {cv_scores}")
print(f"    Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Evaluation
print("\n[6] Evaluation Metrics...")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"    Accuracy:  {accuracy:.4f}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall:    {recall:.4f}")
print(f"    F1-Score:  {f1:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Visualizations
print("\n[7] Generating Visualizations...")

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=wine.target_names, yticklabels=wine.target_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/04_confusion_matrix.png', dpi=150)
plt.close()
print("    Saved: outputs/04_confusion_matrix.png")

# Plot 2: Decision Boundary (using first 2 features)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

knn_2d = KNeighborsClassifier(n_neighbors=best_k)
knn_2d.fit(X_pca, y_train)
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 8))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='Set1', edgecolors='black', s=50)
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_title(f'KNN Decision Boundary (K={best_k})')
plt.tight_layout()
plt.savefig('outputs/04_decision_boundary.png', dpi=150)
plt.close()
print("    Saved: outputs/04_decision_boundary.png")

# Plot 3: Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
ax.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
ax.set_xlabel('Training Examples')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curve')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/04_learning_curve.png', dpi=150)
plt.close()
print("    Saved: outputs/04_learning_curve.png")

# Plot 4: Feature Importance (distance-based)
fig, ax = plt.subplots(figsize=(10, 6))
importances = np.abs(model._fit_X.mean(axis=0))
sorted_idx = np.argsort(importances)
ax.barh(range(len(importances)), importances[sorted_idx], color='teal')
ax.set_yticks(range(len(importances)))
ax.set_yticklabels([wine.feature_names[i] for i in sorted_idx])
ax.set_xlabel('Mean Feature Value')
ax.set_title('Feature Scale Analysis')
plt.tight_layout()
plt.savefig('outputs/04_feature_analysis.png', dpi=150)
plt.close()
print("    Saved: outputs/04_feature_analysis.png")

# Save Model
print("\n[8] Saving Model...")
joblib.dump(model, 'outputs/knn_classifier_model.pkl')
joblib.dump(scaler, 'outputs/scaler.pkl')
print("    Saved: outputs/knn_classifier_model.pkl")
print("    Saved: outputs/scaler.pkl")

print("\n" + "="*60)
print("DONE! All outputs saved to outputs/ folder")
print("="*60)
