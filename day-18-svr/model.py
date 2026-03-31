import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("DAY 18: Support Vector Regressor (SVR)")
print("Dataset: California Housing")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/7] Loading dataset...")
california = fetch_california_housing()
X, y = california.data, california.target
df = pd.DataFrame(X, columns=california.feature_names)
df['MedHouseVal'] = y
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
print(f"  Target mean: {y.mean():.2f}")

# ============================================================
# 2. EDA
# ============================================================
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
for i, (name, ax) in enumerate(zip(california.feature_names, axes.flat)):
    ax.hist(df[name], bins=50, color='steelblue', edgecolor='navy', alpha=0.8)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/01_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_feature_distributions.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y, bins=50, color='coral', edgecolor='darkred', alpha=0.8)
axes[0].set_xlabel('Median House Value ($100,000s)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Target Distribution')

corr = df.corr()['MedHouseVal'].drop('MedHouseVal').sort_values()
colors = ['red' if v < 0 else 'green' for v in corr.values]
axes[1].barh(corr.index, corr.values, color=colors, edgecolor='gray')
axes[1].set_xlabel('Correlation with Target')
axes[1].set_title('Feature Correlation with House Value')
plt.suptitle('Target Analysis & Feature Correlations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/02_target_analysis_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_target_analysis_correlations.png")

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  Train set: {X_train_scaled.shape}")
print(f"  Test set:  {X_test_scaled.shape}")

# ============================================================
# 4. MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================================
print("\n[4/7] Training SVR with hyperparameter tuning...")

param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'epsilon': [0.01, 0.1, 0.5, 1.0]
}

best_score = float('inf')
best_params = None
best_model = None
results = []

for C in param_grid['C']:
    for epsilon in param_grid['epsilon']:
        svr = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma='scale')
        svr.fit(X_train_scaled, y_train)
        y_pred_temp = svr.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred_temp)
        results.append({'C': C, 'epsilon': epsilon, 'MSE': mse})
        if mse < best_score:
            best_score = mse
            best_params = {'C': C, 'epsilon': epsilon}
            best_model = svr

results_df = pd.DataFrame(results)
print(f"\n  Hyperparameter tuning results (Top 5 by MSE):")
top5 = results_df.nsmallest(5, 'MSE')
print(f"  {'C':<10} {'Epsilon':<10} {'MSE':<15}")
print(f"  {'-'*35}")
for _, row in top5.iterrows():
    print(f"  {row['C']:<10} {row['epsilon']:<10} {row['MSE']:<15.4f}")

print(f"\n  Best params: {best_params}")
print(f"  Best test MSE: {best_score:.4f}")

# ============================================================
# 5. EVALUATION
# ============================================================
print("\n[5/7] Evaluating model...")
y_pred = best_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n[6/7] Generating visualizations...")

fig, ax = plt.subplots(figsize=(10, 8))
sample_idx = np.random.choice(len(y_test), 500, replace=False)
ax.scatter(y_test[sample_idx], y_pred[sample_idx], alpha=0.5, color='steelblue', edgecolors='navy', linewidth=0.5)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Median House Value ($100,000s)', fontsize=12)
ax.set_ylabel('Predicted Median House Value ($100,000s)', fontsize=12)
ax.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/04_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_predicted_vs_actual.png")

residuals = y_test - y_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred, residuals, alpha=0.5, color='coral', edgecolors='darkred', linewidth=0.5)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Values', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='navy', alpha=0.8)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residual Value', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/05_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_residual_analysis.png")

fig, ax = plt.subplots(figsize=(10, 6))
train_sizes, train_scores, val_scores = learning_curve(
    SVR(kernel='rbf', C=best_params['C'], epsilon=best_params['epsilon'], gamma='scale'),
    X_train_scaled, y_train, cv=3, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 8), scoring='r2'
)
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training R²', linewidth=2)
ax.plot(train_sizes, val_mean, 's-', color='orange', label='Cross-Validation R²', linewidth=2)
ax.set_xlabel('Training Examples', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/06_learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_learning_curve.png")

fig, ax = plt.subplots(figsize=(10, 6))
kernels = ['rbf', 'linear', 'poly']
kernel_colors = {'rbf': 'steelblue', 'linear': 'coral', 'poly': 'green'}
kernel_scores = {}
for kernel in kernels:
    svr_temp = SVR(kernel=kernel, C=best_params['C'], epsilon=best_params['epsilon'])
    svr_temp.fit(X_train_scaled[:2000], y_train[:2000])
    y_pred_temp = svr_temp.predict(X_test_scaled[:2000])
    score = r2_score(y_test[:2000], y_pred_temp)
    kernel_scores[kernel] = score
    ax.scatter(kernel, score, s=200, color=kernel_colors[kernel], edgecolors='black', zorder=5)
    ax.annotate(f'{score:.3f}', (kernel, score), textcoords="offset points",
                xytext=(0, 15), ha='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Kernel Type', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Kernel Comparison (Subset: 2000 samples)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('outputs/07_kernel_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07_kernel_comparison.png")

# ============================================================
# 7. SAVE MODEL
# ============================================================
print("\n[7/7] Saving model...")
model_data = {
    'model': best_model,
    'scaler': scaler,
    'best_params': best_params,
    'feature_names': california.feature_names,
    'metrics': {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
}
joblib.dump(model_data, 'outputs/svr_california_model.joblib')
print("  Saved: outputs/svr_california_model.joblib")

print("\n" + "=" * 60)
print("DAY 18 COMPLETE!")
print(f"Best Model: SVR(RBF) C={best_params['C']}, epsilon={best_params['epsilon']}")
print(f"Test R²: {r2:.4f}")
print("=" * 60)
