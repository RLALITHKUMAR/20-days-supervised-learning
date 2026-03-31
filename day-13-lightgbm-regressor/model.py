import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("DAY 13: LightGBM Regressor - California Housing")
print("=" * 60)

# ─── 1. Load Data ────────────────────────────────────────────────
print("\n[1/7] Loading California Housing dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="MedHouseVal")

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# ─── 2. EDA ──────────────────────────────────────────────────────
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, col in enumerate(X.columns):
    sns.histplot(X[col], kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(f"{col}\n(skew={X[col].skew():.2f})")
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig("outputs/01_feature_distributions.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            square=True, linewidths=0.5)
plt.tight_layout()
plt.savefig("outputs/02_correlation_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(y, kde=True, color="darkgreen", ax=ax)
ax.set_title("Target Distribution (Median House Value)")
ax.set_xlabel("Median House Value ($100,000s)")
plt.tight_layout()
plt.savefig("outputs/03_target_distribution.png", dpi=150)
plt.close()

print("  EDA plots saved.")

# ─── 3. Preprocessing ───────────────────────────────────────────
print("\n[3/7] Preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "outputs/scaler.pkl")
print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

# ─── 4. Model Training ──────────────────────────────────────────
print("\n[4/7] Training LightGBM Regressor...")

param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
}

best_score = -np.inf
best_params = {}

for lr in param_grid["learning_rate"]:
    for ne in param_grid["n_estimators"]:
        model = lgb.LGBMRegressor(
            learning_rate=lr,
            n_estimators=ne,
            random_state=42,
            verbose=-1,
        )
        scores = cross_val_score(
            model, X_train_scaled, y_train, cv=3,
            scoring="neg_mean_squared_error"
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = {"learning_rate": lr, "n_estimators": ne}

print(f"  Best params: {best_params}")
print(f"  Best CV MSE: {-best_score:.4f}")

model = lgb.LGBMRegressor(
    learning_rate=best_params["learning_rate"],
    n_estimators=best_params["n_estimators"],
    random_state=42,
    verbose=-1,
)
model.fit(X_train_scaled, y_train)

# ─── 5. Evaluation ──────────────────────────────────────────────
print("\n[5/7] Evaluating model...")
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")

# ─── 6. Visualizations ──────────────────────────────────────────
print("\n[6/7] Generating visualizations...")

# Predicted vs Actual
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
ax.set_xlabel("Actual Median House Value")
ax.set_ylabel("Predicted Median House Value")
ax.set_title("Predicted vs Actual Values")
plt.tight_layout()
plt.savefig("outputs/04_predicted_vs_actual.png", dpi=150)
plt.close()

# Residuals plot
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_pred, residuals, alpha=0.3, s=10, color="darkorange")
ax.axhline(y=0, color="red", linestyle="--", lw=2)
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig("outputs/05_residuals_plot.png", dpi=150)
plt.close()

# Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)
ax.barh(range(len(indices)), importances[indices], color="teal")
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([X.columns[i] for i in indices])
ax.set_xlabel("Feature Importance")
ax.set_title("LightGBM Feature Importance")
plt.tight_layout()
plt.savefig("outputs/06_feature_importance.png", dpi=150)
plt.close()

# Learning curve
train_sizes, train_scores, val_scores = learning_curve(
    lgb.LGBMRegressor(
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"],
        random_state=42,
        verbose=-1,
    ),
    X_train_scaled, y_train, cv=3,
    scoring="neg_mean_squared_error",
    train_sizes=np.linspace(0.1, 1.0, 10),
)
train_mean = -train_scores.mean(axis=1)
val_mean = -val_scores.mean(axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Train MSE")
ax.plot(train_sizes, val_mean, "s-", color="darkorange", label="CV MSE")
ax.set_xlabel("Training Samples")
ax.set_ylabel("MSE")
ax.set_title("Learning Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/07_learning_curve.png", dpi=150)
plt.close()

# Training history / iteration plot
eval_results = {}
model_hist = lgb.LGBMRegressor(
    learning_rate=best_params["learning_rate"],
    n_estimators=best_params["n_estimators"],
    random_state=42,
    verbose=-1,
)
model_hist.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    callbacks=[lgb.record_evaluation(eval_results)],
)

fig, ax = plt.subplots(figsize=(8, 6))
iterations = range(1, len(eval_results["training"]["l2"]) + 1)
ax.plot(iterations, eval_results["training"]["l2"], "o-", color="steelblue", label="Train MSE")
ax.plot(iterations, eval_results["valid_0"]["l2"], "s-", color="darkorange", label="Validation MSE")
ax.set_xlabel("Iteration")
ax.set_ylabel("MSE")
ax.set_title("Training History by Iteration")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/08_training_history.png", dpi=150)
plt.close()

# Residual distribution
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(residuals, kde=True, color="purple", ax=ax)
ax.axvline(x=0, color="red", linestyle="--", lw=2)
ax.set_xlabel("Residual Value")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution")
plt.tight_layout()
plt.savefig("outputs/09_residual_distribution.png", dpi=150)
plt.close()

print("  All plots saved to outputs/")

# ─── 7. Save Model ──────────────────────────────────────────────
print("\n[7/7] Saving model...")
joblib.dump(model, "outputs/lightgbm_regressor.pkl")
print("  Model saved to outputs/lightgbm_regressor.pkl")

print("\n" + "=" * 60)
print("DONE — LightGBM Regressor complete!")
print("=" * 60)
