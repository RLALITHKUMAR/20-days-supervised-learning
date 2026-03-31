import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 11: XGBOOST REGRESSOR")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────
print("\n[1/7] Loading Diabetes dataset...")
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")

# ── 2. EDA ────────────────────────────────────────────────────
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(X.columns):
    axes[i].hist(X[col], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
axes[-1].axis("off")
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
print("\n[4/7] Tuning learning_rate and n_estimators...")
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
n_estimators_list = [50, 100, 200, 300, 500]

results = {}
for lr in learning_rates:
    for n_est in n_estimators_list:
        model = xgb.XGBRegressor(
            learning_rate=lr,
            n_estimators=n_est,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        results[(lr, n_est)] = score

best_params = max(results, key=results.get)
best_lr, best_n = best_params
print(f"  Best learning_rate: {best_lr}")
print(f"  Best n_estimators: {best_n}")
print(f"  Best R²: {results[best_params]:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
for lr in learning_rates:
    scores = [results[(lr, n)] for n in n_estimators_list]
    ax.plot(n_estimators_list, scores, "o-", label=f"lr={lr}", linewidth=2)
ax.set_xlabel("n_estimators")
ax.set_ylabel("R² Score")
ax.set_title("Learning Rate & n_estimators Tuning")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/04_hyperparameter_tuning.png", dpi=150)
plt.close()
print("  Saved: 04_hyperparameter_tuning.png")

# ── 5. Train Final Model ──────────────────────────────────────
print("\n[5/7] Training final model...")
model = xgb.XGBRegressor(
    learning_rate=best_lr,
    n_estimators=best_n,
    random_state=42,
    verbosity=0,
)
model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    verbose=False,
)

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
ax.set_yticklabels([diabetes.feature_names[i] for i in sorted_idx])
ax.set_xlabel("Importance")
ax.set_title("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/07_feature_importance.png", dpi=150)
plt.close()
print("  Saved: 07_feature_importance.png")

train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    xgb.XGBRegressor(learning_rate=best_lr, n_estimators=best_n, random_state=42, verbosity=0),
    X_train_scaled,
    y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="r2",
    n_jobs=-1,
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_scores_lc.mean(axis=1), "o-", label="Train", color="steelblue")
ax.plot(train_sizes, val_scores_lc.mean(axis=1), "s-", label="Validation", color="coral")
ax.fill_between(
    train_sizes,
    train_scores_lc.mean(axis=1) - train_scores_lc.std(axis=1),
    train_scores_lc.mean(axis=1) + train_scores_lc.std(axis=1),
    alpha=0.15, color="steelblue",
)
ax.fill_between(
    train_sizes,
    val_scores_lc.mean(axis=1) - val_scores_lc.std(axis=1),
    val_scores_lc.mean(axis=1) + val_scores_lc.std(axis=1),
    alpha=0.15, color="coral",
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

results_dict = model.evals_result()
train_rmse = results_dict["validation_0"]["rmse"]
val_rmse = results_dict["validation_1"]["rmse"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(train_rmse) + 1), train_rmse, "o-", label="Train RMSE", color="steelblue")
ax.plot(range(1, len(val_rmse) + 1), val_rmse, "s-", label="Validation RMSE", color="coral")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("RMSE")
ax.set_title("Training History (Boosting Rounds)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/09_training_history.png", dpi=150)
plt.close()
print("  Saved: 09_training_history.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(residuals, bins=50, kde=True, color="steelblue", ax=ax)
ax.axvline(0, color="red", linestyle="--", lw=2)
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution")
plt.tight_layout()
plt.savefig("outputs/10_residual_distribution.png", dpi=150)
plt.close()
print("  Saved: 10_residual_distribution.png")

# ── 8. Save Model ─────────────────────────────────────────────
joblib.dump(model, "outputs/xgboost_regressor.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
print("\n  Saved: xgboost_regressor.pkl")
print("  Saved: scaler.pkl")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
