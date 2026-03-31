import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("DAY 19: Gradient Boosting Regressor")
print("Dataset: Diabetes")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/7] Loading dataset...")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
print(f"  Target mean: {y.mean():.2f}")

# ============================================================
# 2. EDA
# ============================================================
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
for i, (name, ax) in enumerate(zip(diabetes.feature_names, axes.flat)):
    ax.hist(df[name], bins=30, color='steelblue', edgecolor='navy', alpha=0.8)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/01_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_feature_distributions.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y, bins=40, color='coral', edgecolor='darkred', alpha=0.8)
axes[0].set_xlabel('Disease Progression')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Target Distribution')

corr = df.corr()['target'].drop('target').sort_values()
colors = ['red' if v < 0 else 'green' for v in corr.values]
axes[1].barh(corr.index, corr.values, color=colors, edgecolor='gray')
axes[1].set_xlabel('Correlation with Target')
axes[1].set_title('Feature Correlation with Diabetes Progression')
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
print("\n[4/7] Training Gradient Boosting Regressor with hyperparameter tuning...")

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

best_score = float('inf')
best_params = None
best_model = None
results = []

for n_est in param_grid['n_estimators']:
    for lr in param_grid['learning_rate']:
        gbr = GradientBoostingRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )
        gbr.fit(X_train_scaled, y_train)
        y_pred_temp = gbr.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred_temp)
        results.append({'n_estimators': n_est, 'learning_rate': lr, 'MSE': mse})
        if mse < best_score:
            best_score = mse
            best_params = {'n_estimators': n_est, 'learning_rate': lr}
            best_model = gbr

results_df = pd.DataFrame(results)
print(f"\n  Hyperparameter tuning results (Top 5 by MSE):")
top5 = results_df.nsmallest(5, 'MSE')
print(f"  {'N_Estimators':<15} {'Learning_Rate':<15} {'MSE':<15}")
print(f"  {'-'*45}")
for _, row in top5.iterrows():
    print(f"  {row['n_estimators']:<15} {row['learning_rate']:<15} {row['MSE']:<15.4f}")

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
ax.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='navy', linewidth=0.5)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Disease Progression', fontsize=12)
ax.set_ylabel('Predicted Disease Progression', fontsize=12)
ax.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/04_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_predicted_vs_actual.png")

residuals = y_test - y_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred, residuals, alpha=0.6, color='coral', edgecolors='darkred', linewidth=0.5)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Values', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='navy', alpha=0.8, density=True)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
x_range = np.linspace(residuals.min(), residuals.max(), 100)
from scipy import stats
kde = stats.gaussian_kde(residuals)
axes[1].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
axes[1].set_xlabel('Residual Value', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.suptitle('Residual Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/05_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_residual_analysis.png")

importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(importances)), importances[indices], color='steelblue', edgecolor='navy')
ax.set_yticks(range(len(importances)))
ax.set_yticklabels([diabetes.feature_names[i] for i in indices])
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Feature Importance (Gradient Boosting)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_feature_importance.png")

fig, ax = plt.subplots(figsize=(10, 6))
train_sizes, train_scores, val_scores = learning_curve(
    GradientBoostingRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=4,
        subsample=0.8,
        random_state=42
    ),
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
plt.savefig('outputs/07_learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07_learning_curve.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
staged_train = list(best_model.staged_predict(X_train_scaled))
staged_test = list(best_model.staged_predict(X_test_scaled))
n_estimators_range = range(1, len(staged_train) + 1)

train_errors = [mean_squared_error(y_train, p) for p in staged_train]
test_errors = [mean_squared_error(y_test, p) for p in staged_test]

axes[0].plot(n_estimators_range, train_errors, label='Train MSE', color='blue', linewidth=2)
axes[0].plot(n_estimators_range, test_errors, label='Test MSE', color='orange', linewidth=2)
axes[0].set_xlabel('Number of Estimators', fontsize=12)
axes[0].set_ylabel('MSE', fontsize=12)
axes[0].set_title('Training History (MSE vs Estimators)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

train_r2 = [r2_score(y_train, p) for p in staged_train]
test_r2 = [r2_score(y_test, p) for p in staged_test]
axes[1].plot(n_estimators_range, train_r2, label='Train R²', color='blue', linewidth=2)
axes[1].plot(n_estimators_range, test_r2, label='Test R²', color='orange', linewidth=2)
axes[1].set_xlabel('Number of Estimators', fontsize=12)
axes[1].set_ylabel('R² Score', fontsize=12)
axes[1].set_title('Training History (R² vs Estimators)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.suptitle('Staged Predictions / Training History', fontsize=14, fontweight='bold')
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
    'feature_names': diabetes.feature_names,
    'metrics': {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
}
joblib.dump(model_data, 'outputs/gbr_diabetes_model.joblib')
print("  Saved: outputs/gbr_diabetes_model.joblib")

print("\n" + "=" * 60)
print("DAY 19 COMPLETE!")
print(f"Best Model: GBR n_estimators={best_params['n_estimators']}, lr={best_params['learning_rate']}")
print(f"Test R²: {r2:.4f}")
print("=" * 60)
