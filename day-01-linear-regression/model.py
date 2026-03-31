"""
Day 1: Linear Regression
Dataset: California Housing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

print("="*60)
print("DAY 1: LINEAR REGRESSION")
print("="*60)

# Load Data
print("\n[1] Loading California Housing Dataset...")
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

print(f"    Samples: {X.shape[0]}")
print(f"    Features: {X.shape[1]}")
print(f"    Target range: ${y.min()*100k:.0f} - ${y.max()*100k:.0f}")

# EDA
print("\n[2] Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

sns.histplot(y, kde=True, ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Target Distribution (House Prices)')
axes[0,0].set_xlabel('Price ($100,000s)')

sns.boxplot(x=X['MedInc'], ax=axes[0,1], color='lightgreen')
axes[0,1].set_title('Median Income Distribution')

corr = X.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=axes[0,2], cbar_kws={'shrink': 0.8})
axes[0,2].set_title('Feature Correlation Matrix')

sns.scatterplot(x=X['MedInc'], y=y, alpha=0.3, ax=axes[1,0], color='purple')
axes[1,0].set_title('Income vs House Price')
axes[1,0].set_xlabel('Median Income')

sns.scatterplot(x=X['AveRooms'], y=y, alpha=0.3, ax=axes[1,1], color='orange')
axes[1,1].set_title('Avg Rooms vs House Price')
axes[1,1].set_xlabel('Average Rooms')

sns.scatterplot(x=X['Latitude'], y=y, alpha=0.3, ax=axes[1,2], color='red')
axes[1,2].set_title('Latitude vs House Price')
axes[1,2].set_xlabel('Latitude')

plt.tight_layout()
plt.savefig('outputs/01_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: outputs/01_eda.png")

# Preprocessing
print("\n[3] Preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    Train set: {X_train.shape[0]} samples")
print(f"    Test set: {X_test.shape[0]} samples")

# Training
print("\n[4] Training Linear Regression Model...")
model = LinearRegression()
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
ax.scatter(y_test, y_pred, alpha=0.3, color='blue', edgecolors='none')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Prices ($100,000s)')
ax.set_ylabel('Predicted Prices ($100,000s)')
ax.set_title('Linear Regression: Predicted vs Actual')
plt.tight_layout()
plt.savefig('outputs/01_predicted_vs_actual.png', dpi=150)
plt.close()
print("    Saved: outputs/01_predicted_vs_actual.png")

# Plot 2: Residuals
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_pred, residuals, alpha=0.3, color='green', edgecolors='none')
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
plt.tight_layout()
plt.savefig('outputs/01_residuals.png', dpi=150)
plt.close()
print("    Saved: outputs/01_residuals.png")

# Plot 3: Feature Coefficients
fig, ax = plt.subplots(figsize=(10, 6))
coefficients = model.coef_
features = california.feature_names
sorted_idx = np.argsort(coefficients)
ax.barh(range(len(features)), coefficients[sorted_idx], color='teal')
ax.set_yticks(range(len(features)))
ax.set_yticklabels([features[i] for i in sorted_idx])
ax.set_xlabel('Coefficient Value')
ax.set_title('Feature Coefficients')
plt.tight_layout()
plt.savefig('outputs/01_coefficients.png', dpi=150)
plt.close()
print("    Saved: outputs/01_coefficients.png")

# Plot 4: Learning Curve
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
plt.savefig('outputs/01_learning_curve.png', dpi=150)
plt.close()
print("    Saved: outputs/01_learning_curve.png")

# Plot 5: Residual Distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(residuals, kde=True, ax=ax, color='purple')
ax.axvline(x=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Residual Value')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution')
plt.tight_layout()
plt.savefig('outputs/01_residual_distribution.png', dpi=150)
plt.close()
print("    Saved: outputs/01_residual_distribution.png")

# Save Model
print("\n[7] Saving Model...")
joblib.dump(model, 'outputs/linear_regression_model.pkl')
joblib.dump(scaler, 'outputs/scaler.pkl')
print("    Saved: outputs/linear_regression_model.pkl")
print("    Saved: outputs/scaler.pkl")

print("\n" + "="*60)
print("DONE! All outputs saved to outputs/ folder")
print("="*60)
