import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("DAY 5: LASSO REGRESSION")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1] Loading California Housing Dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="MedHouseVal")

print(f"    Samples: {X.shape[0]}")
print(f"    Features: {X.shape[1]}")
print(f"    Features: {list(X.columns)}")

# ============================================================
# 2. EDA
# ============================================================
print("\n[2] Performing Exploratory Data Analysis...")

# Distribution of target
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(y, bins=50, edgecolor="black", color="steelblue")
axes[0, 0].set_title("Distribution of Median House Value")
axes[0, 0].set_xlabel("Price ($100,000s)")
axes[0, 0].set_ylabel("Frequency")

# Correlation heatmap
corr = X.copy()
corr["MedHouseVal"] = y
sns.heatmap(corr.corr(), annot=False, cmap="coolwarm", ax=axes[0, 1], cbar_kws={"shrink": 0.8})
axes[0, 1].set_title("Feature Correlation Heatmap")

# Feature vs target scatter (top 2 correlated)
feature_importance = X.corrwith(y).abs().sort_values(ascending=False)
top_feature = feature_importance.index[0]
axes[1, 0].scatter(X[top_feature], y, alpha=0.1, s=5, color="darkgreen")
axes[1, 0].set_title(f"{top_feature} vs House Price")
axes[1, 0].set_xlabel(top_feature)
axes[1, 0].set_ylabel("MedHouseVal")

# Box plot of features
X.boxplot(ax=axes[1, 1], rot=45, fontsize=7)
axes[1, 1].set_title("Feature Box Plots")
axes[1, 1].set_ylabel("Value")

plt.tight_layout()
plt.savefig("outputs/01_eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/01_eda_plots.png")

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n[3] Preprocessing Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    Train set: {X_train_scaled.shape[0]} samples")
print(f"    Test set:  {X_test_scaled.shape[0]} samples")

# ============================================================
# 4. ALPHA TUNING
# ============================================================
print("\n[4] Tuning Lasso Alpha...")
alphas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
best_alpha = None
best_score = -np.inf
alpha_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, scoring="r2")
    mean_score = scores.mean()
    alpha_scores.append(mean_score)
    print(f"    alpha={alpha:<8} | CV R2={mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_alpha = alpha

print(f"\n    Best alpha: {best_alpha} (R2={best_score:.4f})")

# Alpha tuning plot
plt.figure(figsize=(10, 5))
plt.semilogx(alphas, alpha_scores, marker="o", color="darkblue", linewidth=2)
plt.axvline(best_alpha, color="red", linestyle="--", label=f"Best alpha={best_alpha}")
plt.title("Lasso Alpha Tuning (Cross-Validation R2)")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Mean R2 Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/02_alpha_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/02_alpha_tuning.png")

# ============================================================
# 5. TRAIN FINAL MODEL
# ============================================================
print("\n[5] Training Final Lasso Model...")
model = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# ============================================================
# 6. METRICS
# ============================================================
print("\n[6] Evaluation Metrics:")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"    MSE:  {mse:.4f}")
print(f"    RMSE: {rmse:.4f}")
print(f"    MAE:  {mae:.4f}")
print(f"    R2:   {r2:.4f}")

# ============================================================
# 7. VISUALIZATIONS
# ============================================================
print("\n[7] Generating Visualizations...")

# Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.15, s=10, color="steelblue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=2, label="Perfect Prediction")
plt.title("Predicted vs Actual House Prices")
plt.xlabel("Actual Price ($100,000s)")
plt.ylabel("Predicted Price ($100,000s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/03_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/03_predicted_vs_actual.png")

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.15, s=10, color="darkgreen")
plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True, alpha=0.3)
plt.savefig("outputs/04_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/04_residuals.png")

# Coefficient paths showing feature selection
plt.figure(figsize=(10, 6))
alphas_path = np.logspace(-4, 2, 100)
coefs = []
for a in alphas_path:
    lasso = Lasso(alpha=a, max_iter=10000, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    coefs.append(lasso.coef_)

coefs = np.array(coefs)
for i, feature in enumerate(housing.feature_names):
    plt.plot(alphas_path, coefs[:, i], label=feature, linewidth=1.5)

plt.xscale("log")
plt.axvline(best_alpha, color="black", linestyle="--", label=f"Best alpha={best_alpha}")
plt.title("Lasso Coefficient Paths (Feature Selection)")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Coefficient Value")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/05_coefficient_paths.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/05_coefficient_paths.png")

# Learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="r2", n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red")
plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training R2")
plt.plot(train_sizes, val_mean, "o-", color="red", label="Validation R2")
plt.title("Learning Curve - Lasso Regression")
plt.xlabel("Training Set Size")
plt.ylabel("R2 Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/06_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/06_learning_curve.png")

# Residual distribution
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=50, kde=True, color="purple", edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--", linewidth=2)
plt.title("Distribution of Residuals")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.savefig("outputs/07_residual_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/07_residual_distribution.png")

# ============================================================
# 8. FEATURE IMPORTANCE (COEFFICIENTS)
# ============================================================
print("\n[8] Feature Coefficients:")
coef_df = pd.DataFrame({"Feature": housing.feature_names, "Coefficient": model.coef_})
coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False)
for _, row in coef_df.iterrows():
    print(f"    {row['Feature']:<12}: {row['Coefficient']:.4f}")

# ============================================================
# 9. SAVE MODEL
# ============================================================
print("\n[9] Saving Model...")
joblib.dump(model, "outputs/lasso_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
print("    Saved: outputs/lasso_model.pkl")
print("    Saved: outputs/scaler.pkl")

print("\n" + "=" * 60)
print("DAY 5 COMPLETE")
print("=" * 60)
