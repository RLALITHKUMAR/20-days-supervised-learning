"""
Day 2: Logistic Regression
Dataset: Breast Cancer Wisconsin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, 
                             roc_curve, precision_recall_curve)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

print("="*60)
print("DAY 2: LOGISTIC REGRESSION")
print("="*60)

# Load Data
print("\n[1] Loading Breast Cancer Dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"    Samples: {X.shape[0]}")
print(f"    Features: {X.shape[1]}")
print(f"    Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

# EDA
print("\n[2] Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

sns.countplot(x=y, ax=axes[0,0], palette=['red', 'green'])
axes[0,0].set_xticks([0, 1])
axes[0,0].set_xticklabels(['Malignant', 'Benign'])
axes[0,0].set_title('Class Distribution')

top_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
sns.boxplot(data=X[top_features], ax=axes[0,1])
axes[0,1].set_title('Top Features Distribution')
axes[0,1].tick_params(axis='x', rotation=45)

corr = X.corr().iloc[:5, :5]
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[0,2], fmt='.2f')
axes[0,2].set_title('Feature Correlation (Top 5)')

sns.histplot(X['mean radius'], kde=True, ax=axes[1,0], color='blue')
axes[1,0].set_title('Mean Radius Distribution')

sns.histplot(X['mean area'], kde=True, ax=axes[1,1], color='orange')
axes[1,1].set_title('Mean Area Distribution')

sns.scatterplot(x=X['mean radius'], y=X['mean area'], hue=y, ax=axes[1,2], 
                palette={0: 'red', 1: 'green'}, alpha=0.5)
axes[1,2].set_title('Radius vs Area by Class')

plt.tight_layout()
plt.savefig('outputs/02_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: outputs/02_eda.png")

# Preprocessing
print("\n[3] Preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    Train set: {X_train.shape[0]} samples")
print(f"    Test set: {X_test.shape[0]} samples")

# Training
print("\n[4] Training Logistic Regression Model...")
model = LogisticRegression(max_iter=10000, C=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"    CV Accuracy Scores: {cv_scores}")
print(f"    Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Evaluation
print("\n[5] Evaluation Metrics...")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"    Accuracy:  {accuracy:.4f}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall:    {recall:.4f}")
print(f"    F1-Score:  {f1:.4f}")
print(f"    ROC-AUC:   {roc_auc:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Visualizations
print("\n[6] Generating Visualizations...")

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=data.target_names, yticklabels=data.target_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/02_confusion_matrix.png', dpi=150)
plt.close()
print("    Saved: outputs/02_confusion_matrix.png")

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/02_roc_curve.png', dpi=150)
plt.close()
print("    Saved: outputs/02_roc_curve.png")

# Plot 3: Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall_vals, precision_vals, color='green', lw=2)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/02_precision_recall.png', dpi=150)
plt.close()
print("    Saved: outputs/02_precision_recall.png")

# Plot 4: Feature Coefficients
fig, ax = plt.subplots(figsize=(10, 8))
coefficients = model.coef_[0]
sorted_idx = np.argsort(coefficients)
ax.barh(range(len(coefficients)), coefficients[sorted_idx], color='teal')
ax.set_yticks(range(len(coefficients)))
ax.set_yticklabels([data.feature_names[i] for i in sorted_idx])
ax.set_xlabel('Coefficient Value')
ax.set_title('Feature Coefficients')
plt.tight_layout()
plt.savefig('outputs/02_coefficients.png', dpi=150)
plt.close()
print("    Saved: outputs/02_coefficients.png")

# Plot 5: Learning Curve
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
plt.savefig('outputs/02_learning_curve.png', dpi=150)
plt.close()
print("    Saved: outputs/02_learning_curve.png")

# Save Model
print("\n[7] Saving Model...")
joblib.dump(model, 'outputs/logistic_regression_model.pkl')
joblib.dump(scaler, 'outputs/scaler.pkl')
print("    Saved: outputs/logistic_regression_model.pkl")
print("    Saved: outputs/scaler.pkl")

print("\n" + "="*60)
print("DONE! All outputs saved to outputs/ folder")
print("="*60)
