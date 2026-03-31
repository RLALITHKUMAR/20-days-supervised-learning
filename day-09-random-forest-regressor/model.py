import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 9: RANDOM FOREST REGRESSOR")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────
print("\n[1/7] Loading California Housing dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")

# ── 2. EDA ────────────────────────────────────────────────────
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, col in enumerate(X.columns):
    axes[i].hist(X[col], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig("outputs/01_feature_distributions.png", dpi=150)
plt.close()
print("  Saved: 01_feature_distributions.png")

fig, ax = plt.subplots(figsize=(10, 8))
corr = X.copy()
corr["target"] = y
sns.heatmap(corr.corr(), annot=False, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=150)
plt.close()
print("  Saved: 02_correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=X, ax=ax, color="steelblue")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_title("Feature Box Plots")
plt.tight_layout()
plt.savefig("outputs/03_box_plots.png", dpi=150)
plt.close()
print("  Saved: 03_box_plots.png")

# ── 3. Preprocessing ──────────────────────────────────────────
print("\n[3/7] Preprocessing data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  Train: {X_train_scaled.shape[0]} samples")
print(f"  Test:  {X_test_scaled.shape[0]} samples")

# ── 4. Hyperparameter Tuning ──────────────────────────────────
print("\n[4/7] Tuning n_estimators...")
estimator_range = [50, 100, 200, 300, 500]
train_scores = []
val_scores = []

for n in estimator_range:
    rf = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    train_scores.append(rf.score(X_train_scaled, y_train))
    val_scores.append(rf.score(X_test_scaled, y_test))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(estimator_range, train_scores, "o-", label="Train R²", color="steelblue")
ax.plot(estimator_range, val_scores, "s-", label="Val R²", color="coral")
ax.set_xlabel("Number of Estimators")
ax.set_ylabel("R² Score")
ax.set_title("n_estimators Tuning")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/04_oob_score_plot.png", dpi=150)
plt.close()
print("  Saved: 04_oob_score_plot.png")

best_n = estimator_range[np.argmax(val_scores)]
print(f"  Best n_estimators: {best_n}")

# ── 5. Train Final Model ──────────────────────────────────────
print("\n[5/7] Training final model...")
model = RandomForestRegressor(
    n_estimators=best_n,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)
print(f"  OOB Score: {model.oob_score:.4f}")

y_pred = model.predict(X_test_scaled)

# ── 6. Metrics ────────────────────────────────────────────────
print("\n[6/7] Computing metrics...")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")

# ── 7. Visualizations ─────────────────────────────────────────
print("\n[7/7] Generating visualizations...")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Predicted vs Actual")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/05_predicted_vs_actual.png", dpi=150)
plt.close()
print("  Saved: 05_predicted_vs_actual.png")

residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_pred, residuals, alpha=0.3, s=10, color="coral")
ax.axhline(0, color="red", linestyle="--", lw=2)
ax.set_xlabel("Predicted")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/06_residual_plot.png", dpi=150)
plt.close()
print("  Saved: 06_residual_plot.png")

fig, ax = plt.subplots(figsize=(10, 6))
importances = model.feature_importances_
sorted_idx = np.argsort(importances)
ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="steelblue")
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([housing.feature_names[i] for i in sorted_idx])
ax.set_xlabel("Importance")
ax.set_title("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/07_feature_importance.png", dpi=150)
plt.close()
print("  Saved: 07_feature_importance.png")

train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    RandomForestRegressor(n_estimators=best_n, random_state=42, n_jobs=-1),
    X_train_scaled,
    y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="r2",
    n_jobs=-1,
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    train_sizes,
    train_scores_lc.mean(axis=1),
    "o-",
    label="Train",
    color="steelblue",
)
ax.plot(
    train_sizes,
    val_scores_lc.mean(axis=1),
    "s-",
    label="Validation",
    color="coral",
)
ax.fill_between(
    train_sizes,
    train_scores_lc.mean(axis=1) - train_scores_lc.std(axis=1),
    train_scores_lc.mean(axis=1) + train_scores_lc.std(axis=1),
    alpha=0.15,
    color="steelblue",
)
ax.fill_between(
    train_sizes,
    val_scores_lc.mean(axis=1) - val_scores_lc.std(axis=1),
    val_scores_lc.mean(axis=1) + val_scores_lc.std(axis=1),
    alpha=0.15,
    color="coral",
)
ax.set_xlabel("Training Samples")
ax.set_ylabel("R² Score")
ax.set_title("Learning Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/08_learning_curve.png", dpi=150)
plt.close()
print("  Saved: 08_learning_curve.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(residuals, bins=50, kde=True, color="steelblue", ax=ax)
ax.axvline(0, color="red", linestyle="--", lw=2)
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution")
plt.tight_layout()
plt.savefig("outputs/09_residual_distribution.png", dpi=150)
plt.close()
print("  Saved: 09_residual_distribution.png")

# ── 8. Save Model ─────────────────────────────────────────────
joblib.dump(model, "outputs/random_forest_regressor.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
print("\n  Saved: random_forest_regressor.pkl")
print("  Saved: scaler.pkl")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
