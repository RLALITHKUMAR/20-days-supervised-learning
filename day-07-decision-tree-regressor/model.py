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
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("DAY 7: DECISION TREE REGRESSOR")
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution of target
axes[0, 0].hist(y, bins=50, edgecolor="black", color="steelblue")
axes[0, 0].set_title("Distribution of Median House Value")
axes[0, 0].set_xlabel("Price ($100,000s)")
axes[0, 0].set_ylabel("Frequency")

# Correlation heatmap
corr = X.copy()
corr["MedHouseVal"] = y
sns.heatmap(corr.corr(), annot=False, cmap="coolwarm", ax=axes[0, 1], cbar_kws={"shrink": 0.8})
axes[0, 1].set_title("Feature Correlation Heatmap")

# Feature vs target scatter
feature_importance_corr = X.corrwith(y).abs().sort_values(ascending=False)
top_feature = feature_importance_corr.index[0]
axes[1, 0].scatter(X[top_feature], y, alpha=0.1, s=5, color="darkgreen")
axes[1, 0].set_title(f"{top_feature} vs House Price")
axes[1, 0].set_xlabel(top_feature)
axes[1, 0].set_ylabel("MedHouseVal")

# Pairwise scatter of top features
axes[1, 1].scatter(X["MedInc"], X["AveRooms"], alpha=0.1, s=5, c=y, cmap="viridis")
axes[1, 1].set_title("MedInc vs AveRooms (colored by Price)")
axes[1, 1].set_xlabel("MedInc")
axes[1, 1].set_ylabel("AveRooms")
plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label="MedHouseVal", shrink=0.8)

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
# 4. MAX_DEPTH TUNING
# ============================================================
print("\n[4] Tuning max_depth...")
max_depths = [3, 5, 7, 10, 15, 20, 25, 30, None]
best_depth = None
best_score = -np.inf
depth_scores = []

for depth in max_depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    scores = cross_val_score(tree, X_train_scaled, y_train, cv=5, scoring="r2")
    mean_score = scores.mean()
    depth_scores.append(mean_score)
    depth_label = "None" if depth is None else str(depth)
    print(f"    max_depth={depth_label:<6} | CV R2={mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_depth = depth

print(f"\n    Best max_depth: {best_depth} (R2={best_score:.4f})")

# Depth tuning plot
depth_labels = ["None" if d is None else str(d) for d in max_depths]
plt.figure(figsize=(10, 5))
x_vals = range(len(max_depths))
plt.plot(x_vals, depth_scores, marker="o", color="darkblue", linewidth=2)
best_idx = max_depths.index(best_depth)
plt.axvline(best_idx, color="red", linestyle="--", label=f"Best depth={best_depth}")
plt.xticks(x_vals, depth_labels)
plt.title("Decision Tree max_depth Tuning (Cross-Validation R2)")
plt.xlabel("max_depth")
plt.ylabel("Mean R2 Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/02_depth_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/02_depth_tuning.png")

# ============================================================
# 5. TRAIN FINAL MODEL
# ============================================================
print("\n[5] Training Final Decision Tree Regressor...")
model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
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

# Feature Importance Bar Chart
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), importances[indices], color="steelblue", edgecolor="black")
plt.yticks(range(X.shape[1]), [housing.feature_names[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Feature Importance - Decision Tree Regressor")
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("outputs/05_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/05_feature_importance.png")

# Tree Visualization (limited depth for readability)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model, max_depth=3, feature_names=housing.feature_names,
          filled=True, rounded=True, fontsize=9, ax=ax)
plt.title("Decision Tree Structure (First 3 Levels)")
plt.savefig("outputs/06_tree_visualization.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/06_tree_visualization.png")

# Learning Curve
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
plt.title("Learning Curve - Decision Tree Regressor")
plt.xlabel("Training Set Size")
plt.ylabel("R2 Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/07_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: outputs/07_learning_curve.png")

# ============================================================
# 8. FEATURE IMPORTANCE SUMMARY
# ============================================================
print("\n[8] Feature Importances:")
for i in indices:
    print(f"    {housing.feature_names[i]:<12}: {importances[i]:.4f}")

# ============================================================
# 9. SAVE MODEL
# ============================================================
print("\n[9] Saving Model...")
joblib.dump(model, "outputs/decision_tree_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
print("    Saved: outputs/decision_tree_model.pkl")
print("    Saved: outputs/scaler.pkl")

print("\n" + "=" * 60)
print("DAY 7 COMPLETE")
print("=" * 60)
