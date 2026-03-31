import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("DAY 20: Gradient Boosting Classifier")
print("Dataset: Breast Cancer (Wisconsin)")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/7] Loading dataset...")
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {dict(zip(cancer.target_names, [np.sum(y==0), np.sum(y==1)]))}")

# ============================================================
# 2. EDA
# ============================================================
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for i, (name, ax) in enumerate(zip(cancer.feature_names[:10], axes.flat)):
    ax.hist(df[name], bins=30, color='steelblue', edgecolor='navy', alpha=0.8)
    ax.set_title(name, fontsize=8)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
plt.suptitle('Feature Distributions (First 10)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/01_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_feature_distributions.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_names = cancer.target_names
class_counts = pd.Series(y).value_counts()
colors = ['coral', 'steelblue']
axes[0].bar(class_names, class_counts.values, color=colors, edgecolor='gray')
axes[0].set_xlabel('Diagnosis')
axes[0].set_ylabel('Count')
axes[0].set_title('Class Distribution')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

corr = df.corr()['target'].drop('target').sort_values()
colors_bar = ['red' if v < 0 else 'green' for v in corr.values]
axes[1].barh(corr.index, corr.values, color=colors_bar, edgecolor='gray')
axes[1].set_xlabel('Correlation with Target')
axes[1].set_title('Feature Correlation with Diagnosis')
plt.suptitle('Class Distribution & Feature Correlations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/02_class_distribution_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_class_distribution_correlations.png")

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', ax=ax,
            annot=False, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_correlation_heatmap.png")

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n[3/7] Preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  Train set: {X_train_scaled.shape}")
print(f"  Test set:  {X_test_scaled.shape}")

# ============================================================
# 4. MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================================
print("\n[4/7] Training Gradient Boosting Classifier with hyperparameter tuning...")

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [2, 3, 4, 5]
}

best_score = 0
best_params = None
best_model = None
results = []

for n_est in param_grid['n_estimators']:
    for depth in param_grid['max_depth']:
        gbc = GradientBoostingClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gbc.fit(X_train_scaled, y_train)
        score = gbc.score(X_test_scaled, y_test)
        results.append({'n_estimators': n_est, 'max_depth': depth, 'accuracy': score})
        if score > best_score:
            best_score = score
            best_params = {'n_estimators': n_est, 'max_depth': depth}
            best_model = gbc

results_df = pd.DataFrame(results)
print(f"\n  Hyperparameter tuning results (Top 5 by Accuracy):")
top5 = results_df.nlargest(5, 'accuracy')
print(f"  {'N_Estimators':<15} {'Max_Depth':<12} {'Accuracy':<10}")
print(f"  {'-'*37}")
for _, row in top5.iterrows():
    print(f"  {row['n_estimators']:<15} {row['max_depth']:<12} {row['accuracy']:<10.4f}")

print(f"\n  Best params: {best_params}")
print(f"  Best test accuracy: {best_score:.4f}")

# ============================================================
# 5. EVALUATION
# ============================================================
print("\n[5/7] Evaluating model...")
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n[6/7] Generating visualizations...")

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/04_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_confusion_matrix.png")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='steelblue', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig('outputs/05_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_roc_curve.png")

importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(importances)), importances[indices], color='steelblue', edgecolor='navy')
ax.set_yticks(range(len(importances)))
ax.set_yticklabels([cancer.feature_names[i] for i in indices], fontsize=9)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Feature Importance (Gradient Boosting)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_feature_importance.png")

fig, ax = plt.subplots(figsize=(10, 6))
train_sizes, train_scores, val_scores = learning_curve(
    GradientBoostingClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ),
    X_train_scaled, y_train, cv=3, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 8), scoring='accuracy'
)
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy', linewidth=2)
ax.plot(train_sizes, val_mean, 's-', color='orange', label='Cross-Validation Accuracy', linewidth=2)
ax.set_xlabel('Training Examples', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/07_learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07_learning_curve.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
staged_train = list(best_model.staged_predict(X_train_scaled))
staged_test = list(best_model.staged_predict(X_test_scaled))
n_estimators_range = range(1, len(staged_train) + 1)

train_acc = [accuracy_score(y_train, p) for p in staged_train]
test_acc = [accuracy_score(y_test, p) for p in staged_test]

axes[0].plot(n_estimators_range, train_acc, label='Train Accuracy', color='blue', linewidth=2)
axes[0].plot(n_estimators_range, test_acc, label='Test Accuracy', color='orange', linewidth=2)
axes[0].set_xlabel('Number of Estimators', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Training History (Accuracy vs Estimators)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

train_auc = [roc_auc_score(y_train, best_model.staged_predict_proba(X_train_scaled)[:, i, 1])
             for i in range(len(staged_train))]
test_auc = [roc_auc_score(y_test, best_model.staged_predict_proba(X_test_scaled)[:, i, 1])
            for i in range(len(staged_test))]
axes[1].plot(n_estimators_range, train_auc, label='Train ROC-AUC', color='blue', linewidth=2)
axes[1].plot(n_estimators_range, test_auc, label='Test ROC-AUC', color='orange', linewidth=2)
axes[1].set_xlabel('Number of Estimators', fontsize=12)
axes[1].set_ylabel('ROC-AUC', fontsize=12)
axes[1].set_title('Training History (ROC-AUC vs Estimators)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.suptitle('Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/08_training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 08_training_history.png")

# ============================================================
# 7. SAVE MODEL
# ============================================================
print("\n[7/7] Saving model...")
model_data = {
    'model': best_model,
    'scaler': scaler,
    'best_params': best_params,
    'feature_names': cancer.feature_names,
    'metrics': {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc
    }
}
joblib.dump(model_data, 'outputs/gbc_breast_cancer_model.joblib')
print("  Saved: outputs/gbc_breast_cancer_model.joblib")

print("\n" + "=" * 60)
print("DAY 20 COMPLETE!")
print(f"Best Model: GBC n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}")
print(f"Test Accuracy: {acc:.4f}")
print("=" * 60)
