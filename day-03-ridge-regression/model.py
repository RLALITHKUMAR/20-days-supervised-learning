"""
Day 3: Ridge Regression
Dataset: Diabetes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

print("="*60)
print("DAY 3: RIDGE REGRESSION")
print("="*60)

# Load Data
print("\n[1] Loading Diabetes Dataset...")
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

print(f"    Samples: {X.shape[0]}")
print(f"    Features: {X.shape[1]}")
print(f"    Target range: {y.min():.0f} - {y.max():.0f}")

# EDA
print("\n[2] Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

sns.histplot(y, kde=True, ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Target Distribution (Disease Progression)')

sns.boxplot(data=X[['bmi', 'bp', 's5']], ax=axes[0,1])
axes[0,1].set_title('Key Features Distribution')
axes[0,1].tick_params(axis='x', rotation=45)

corr = X.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=axes[0,2], cbar_kws={'shrink': 0.8})
axes[0,2].set_title('Feature Correlation Matrix')

sns.scatterplot(x=X['bmi'], y=y, alpha=0.5, ax=axes[1,0], color='purple')
axes[1,0].set_title('BMI vs Disease Progression')
axes[1,0].set_xlabel('BMI')

sns.scatterplot(x=X['bp'], y=y, alpha=0.5, ax=axes[1,1], color='orange')
axes[1,1].set_title('Blood Pressure vs Disease Progression')
axes[1,1].set_xlabel('Blood Pressure')

sns.scatterplot(x=X['s5'], y=y, alpha=0.5, ax=axes[1,2], color='red')
axes[1,2].set_title('S5 (Serum 5) vs Disease Progression')
axes[1,2].set_xlabel('S5')

plt.tight_layout()
plt.savefig('outputs/03_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: outputs/03_eda.png")

# Preprocessing
print("\n[3] Preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    Train set: {X_train.shape[0]} samples")
print(f"    Test set: {X_test.shape[0]} samples")

# Training with alpha tuning
print("\n[4] Training Ridge Regression Model...")
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
best_r2 = -np.inf
best_alpha = 1.0

for alpha in alphas:
    model_temp = Ridge(alpha=alpha)
    model_temp.fit(X_train_scaled, y_train)
    cv_r2 = cross_val_score(model_temp, X_train_scaled, y_train, cv=5, scoring='r2').mean()
    print(f"    Alpha={alpha:.2f}, CV R²={cv_r2:.4f}")
    if cv_r2 > best_r2:
        best_r2 = cv_r2
        best_alpha = alpha

print(f"\n    Best Alpha: {best_alpha}")

model = Ridge(alpha=best_alpha)
model.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"    CV R² Scores: {cv_scores}")
print(f"    Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Evaluation
print("\n[5] Evaluation Metrics...")
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"    MSE:  {mse:.4f}")
print(f"    RMSE: {rmse:.4f}")
print(f"    MAE:  {mae:.4f}")
print(f"    R²:   {r2:.4f}")

# Visualizations
print("\n[6] Generating Visualizations...")

# Plot 1: Predicted vs Actual
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.5, color='blue', edgecolors='none')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Ridge Regression: Predicted vs Actual')
plt.tight_layout()
plt.savefig('outputs/03_predicted_vs_actual.png', dpi=150)
plt.close()
print("    Saved: outputs/03_predicted_vs_actual.png")

# Plot 2: Residuals
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_pred, residuals, alpha=0.5, color='green', edgecolors='none')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
plt.tight_layout()
plt.savefig('outputs/03_residuals.png', dpi=150)
plt.close()
print("    Saved: outputs/03_residuals.png")

# Plot 3: Coefficient Shrinkage
fig, ax = plt.subplots(figsize=(10, 6))
coef_ols = []
coef_ridge = []
for a in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    m = Ridge(alpha=a)
    m.fit(X_train_scaled, y_train)
    coef_ridge.append(m.coef_)

for i in range(len(diabetes.feature_names)):
    ax.plot([0.001, 0.01, 0.1, 1.0, 10.0, 100.0], [c[i] for c in coef_ridge], marker='o')
ax.set_xscale('log')
ax.set_xlabel('Alpha (log scale)')
ax.set_ylabel('Coefficient Value')
ax.set_title('Coefficient Shrinkage vs Alpha')
ax.legend(diabetes.feature_names, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('outputs/03_coefficient_shrinkage.png', dpi=150)
plt.close()
print("    Saved: outputs/03_coefficient_shrinkage.png")

# Plot 4: Feature Coefficients
fig, ax = plt.subplots(figsize=(10, 6))
coefficients = model.coef_
sorted_idx = np.argsort(coefficients)
ax.barh(range(len(coefficients)), coefficients[sorted_idx], color='teal')
ax.set_yticks(range(len(coefficients)))
ax.set_yticklabels([diabetes.feature_names[i] for i in sorted_idx])
ax.set_xlabel('Coefficient Value')
ax.set_title('Feature Coefficients (Ridge)')
plt.tight_layout()
plt.savefig('outputs/03_coefficients.png', dpi=150)
plt.close()
print("    Saved: outputs/03_coefficients.png")

# Plot 5: Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
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
ax.set_ylabel('R² Score')
ax.set_title('Learning Curve')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/03_learning_curve.png', dpi=150)
plt.close()
print("    Saved: outputs/03_learning_curve.png")

# Save Model
print("\n[7] Saving Model...")
joblib.dump(model, 'outputs/ridge_regression_model.pkl')
joblib.dump(scaler, 'outputs/scaler.pkl')
print("    Saved: outputs/ridge_regression_model.pkl")
print("    Saved: outputs/scaler.pkl")

print("\n" + "="*60)
print("DONE! All outputs saved to outputs/ folder")
print("="*60)
