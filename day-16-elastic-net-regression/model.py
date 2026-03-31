import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("DAY 16: Elastic Net Regression - Diabetes")
print("=" * 60)

# --- 1. Load Data ---
print("\n[1/7] Loading Diabetes dataset...")
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="Progression")

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# --- 2. EDA ---
print("\n[2/7] Performing EDA...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(X.columns):
    sns.histplot(X[col], kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(f"{col}")
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
ax.set_title("Target Distribution (Disease Progression)")
ax.set_xlabel("Disease Progression Score")
plt.tight_layout()
plt.savefig("outputs/03_target_distribution.png", dpi=150)
plt.close()

print("  EDA plots saved.")

# --- 3. Preprocessing ---
print("\n[3/7] Preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "outputs/scaler.pkl")
print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

# --- 4. Model Training ---
print("\n[4/7] Training Elastic Net Regression...")

param_grid = {
    "alpha": [0.01, 0.1, 0.5, 1.0, 5.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

best_score = -np.inf
best_params = {}

for alpha in param_grid["alpha"]:
    for l1_ratio in param_grid["l1_ratio"]:
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10000,
            random_state=42,
        )
        scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5,
            scoring="neg_mean_squared_error"
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = {"alpha": alpha, "l1_ratio": l1_ratio}

print(f"  Best params: {best_params}")
print(f"  Best CV MSE: {-best_score:.4f}")

model = ElasticNet(
    alpha=best_params["alpha"],
    l1_ratio=best_params["l1_ratio"],
    max_iter=10000,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# --- 5. Evaluation ---
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

# --- 6. Visualizations ---
print("\n[6/7] Generating visualizations...")

# Predicted vs Actual
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.5, s=20, color="steelblue")
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
ax.set_xlabel("Actual Disease Progression")
ax.set_ylabel("Predicted Disease Progression")
ax.set_title("Predicted vs Actual Values")
plt.tight_layout()
plt.savefig("outputs/04_predicted_vs_actual.png", dpi=150)
plt.close()

# Residuals Plot
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_pred, residuals, alpha=0.5, s=20, color="darkorange")
ax.axhline(y=0, color="red", linestyle="--", lw=2)
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig("outputs/05_residuals_plot.png", dpi=150)
plt.close()

# Coefficient Paths for Different L1 Ratios
alphas = np.logspace(-3, 2, 50)
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, l1_ratio in enumerate(l1_ratios):
    coefs = []
    for alpha in alphas:
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
        en.fit(X_train_scaled, y_train)
        coefs.append(en.coef_)
    coefs = np.array(coefs)
    axes[0].plot(alphas, coefs[:, :5], color=colors[idx], alpha=0.6,
                 label=f"l1_ratio={l1_ratio}")

axes[0].set_xscale("log")
axes[0].set_xlabel("Alpha (log scale)")
axes[0].set_ylabel("Coefficient Values")
axes[0].set_title("Coefficient Paths (First 5 Features)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Final coefficients for best model
coefs = model.coef_
axes[1].barh(range(len(coefs)), coefs, color="teal")
axes[1].set_yticks(range(len(coefs)))
axes[1].set_yticklabels(X.columns)
axes[1].set_xlabel("Coefficient Value")
axes[1].set_title(f"Elastic Net Coefficients (alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']})")
axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("outputs/06_coefficient_paths.png", dpi=150)
plt.close()

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    ElasticNet(
        alpha=best_params["alpha"],
        l1_ratio=best_params["l1_ratio"],
        max_iter=10000,
        random_state=42,
    ),
    X_train_scaled, y_train, cv=5,
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

# Residual Distribution
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(residuals, kde=True, color="purple", ax=ax)
ax.axvline(x=0, color="red", linestyle="--", lw=2)
ax.set_xlabel("Residual Value")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution")
plt.tight_layout()
plt.savefig("outputs/08_residual_distribution.png", dpi=150)
plt.close()

# Alpha Comparison Plot
alpha_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
l1_ratios_compare = [0.2, 0.5, 0.8]
results = {}

for l1_r in l1_ratios_compare:
    mse_scores = []
    for a in alpha_values:
        en = ElasticNet(alpha=a, l1_ratio=l1_r, max_iter=10000, random_state=42)
        scores = cross_val_score(en, X_train_scaled, y_train, cv=5,
                                 scoring="neg_mean_squared_error")
        mse_scores.append(-scores.mean())
    results[l1_r] = mse_scores

fig, ax = plt.subplots(figsize=(10, 6))
for l1_r, scores in results.items():
    ax.plot(alpha_values, scores, "o-", label=f"l1_ratio={l1_r}")
ax.set_xscale("log")
ax.set_xlabel("Alpha (log scale)")
ax.set_ylabel("Cross-Validated MSE")
ax.set_title("Alpha Comparison Across L1 Ratios")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/09_alpha_comparison.png", dpi=150)
plt.close()

print("  All plots saved to outputs/")

# --- 7. Save Model ---
print("\n[7/7] Saving model...")
joblib.dump(model, "outputs/elastic_net_model.pkl")
print("  Model saved to outputs/elastic_net_model.pkl")

print("\n" + "=" * 60)
print("DONE — Elastic Net Regression complete!")
print("=" * 60)
