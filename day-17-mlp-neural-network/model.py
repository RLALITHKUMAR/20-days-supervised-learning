import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import joblib
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("DAY 17: MLP Neural Network (Classification)")
print("Dataset: Digits (8x8 handwritten digits)")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/7] Loading dataset...")
digits = load_digits()
X, y = digits.data, digits.target
feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {np.unique(y)}")
print(f"  Class distribution:\n{pd.Series(y).value_counts().sort_index().to_string()}")

# ============================================================
# 2. EDA
# ============================================================
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Label: {y[i]}', fontsize=12)
    ax.axis('off')
plt.suptitle('Sample Digits (First 10)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/01_sample_digits.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_sample_digits.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_counts = pd.Series(y).value_counts().sort_index()
axes[0].bar(class_counts.index, class_counts.values, color='steelblue', edgecolor='navy')
axes[0].set_xlabel('Digit')
axes[0].set_ylabel('Count')
axes[0].set_title('Class Distribution')
axes[0].set_xticks(range(10))

pixel_means = df[feature_names].mean()
pixel_means.index = range(len(pixel_means))
axes[1].plot(pixel_means.index, pixel_means.values, color='coral', linewidth=1.5)
axes[1].set_xlabel('Pixel Index')
axes[1].set_ylabel('Mean Intensity')
axes[1].set_title('Average Pixel Intensity Across All Images')
axes[1].set_xlim(0, 63)
plt.suptitle('EDA: Class Distribution & Pixel Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/02_class_distribution_pixel_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_class_distribution_pixel_analysis.png")

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[feature_names].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Pixel Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/03_pixel_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_pixel_correlation_heatmap.png")

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
print("\n[4/7] Training MLP Classifier with hyperparameter tuning...")
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
    'alpha': [0.0001, 0.001, 0.01]
}

best_score = 0
best_params = None
best_model = None
results = []

for hl in param_grid['hidden_layer_sizes']:
    for alpha in param_grid['alpha']:
        mlp = MLPClassifier(
            hidden_layer_sizes=hl,
            alpha=alpha,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate='adaptive',
            solver='adam'
        )
        mlp.fit(X_train_scaled, y_train)
        score = mlp.score(X_test_scaled, y_test)
        results.append({'hidden_layers': hl, 'alpha': alpha, 'accuracy': score})
        if score > best_score:
            best_score = score
            best_params = {'hidden_layer_sizes': hl, 'alpha': alpha}
            best_model = mlp

results_df = pd.DataFrame(results)
print(f"\n  Hyperparameter tuning results:")
print(f"  {'Hidden Layers':<20} {'Alpha':<10} {'Accuracy':<10}")
print(f"  {'-'*40}")
for _, row in results_df.iterrows():
    print(f"  {str(row['hidden_layers']):<20} {row['alpha']:<10} {row['accuracy']:<10.4f}")

print(f"\n  Best params: {best_params}")
print(f"  Best test accuracy: {best_score:.4f}")

# ============================================================
# 5. EVALUATION
# ============================================================
print("\n[5/7] Evaluating model...")
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

print(f"\n  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n[6/7] Generating visualizations...")

fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=range(10), yticklabels=range(10))
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/04_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_confusion_matrix.png")

fig, axes = plt.subplots(3, 5, figsize=(15, 10))
test_indices = np.random.choice(len(X_test), 15, replace=False)
for i, idx in enumerate(test_indices):
    row, col = divmod(i, 5)
    axes[row, col].imshow(X_test[idx].reshape(8, 8), cmap='gray')
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    color = 'green' if true_label == pred_label else 'red'
    axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}',
                             color=color, fontsize=10)
    axes[row, col].axis('off')
plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/05_sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_sample_predictions.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(best_model.loss_curve_, color='navy', linewidth=2)
ax.set_xlabel('Iterations', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/06_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_loss_curve.png")

fig, ax = plt.subplots(figsize=(10, 6))
train_sizes, train_scores, val_scores = learning_curve(
    MLPClassifier(hidden_layer_sizes=best_params['hidden_layer_sizes'],
                  alpha=best_params['alpha'], max_iter=500, random_state=42,
                  early_stopping=False, solver='adam'),
    X_train_scaled, y_train, cv=3, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
ax.plot(train_sizes, val_mean, 's-', color='orange', label='Cross-Validation Score', linewidth=2)
ax.set_xlabel('Training Examples', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/07_learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07_learning_curve.png")

y_test_bin = label_binarize(y_test, classes=range(10))
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(10):
    fpr, tpr, _ = [], [], []
    from sklearn.metrics import roc_curve
    fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    auc_i = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
    ax.plot(fpr_i, tpr_i, linewidth=2, label=f'Digit {i} (AUC={auc_i:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve (One-vs-Rest)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/08_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 08_roc_curve.png")

# ============================================================
# 7. SAVE MODEL
# ============================================================
print("\n[7/7] Saving model...")
model_data = {
    'model': best_model,
    'scaler': scaler,
    'best_params': best_params,
    'feature_names': feature_names,
    'metrics': {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc
    }
}
joblib.dump(model_data, 'outputs/mlp_digits_model.joblib')
print("  Saved: outputs/mlp_digits_model.joblib")

print("\n" + "=" * 60)
print("DAY 17 COMPLETE!")
print(f"Best Model: MLP{best_params['hidden_layer_sizes']} alpha={best_params['alpha']}")
print(f"Test Accuracy: {acc:.4f}")
print("=" * 60)
