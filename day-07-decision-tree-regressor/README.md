# Day 7: Decision Tree Regressor

## Overview
Decision Tree Regressor is a non-parametric supervised learning algorithm that predicts continuous target values by recursively partitioning the feature space into regions. Each internal node represents a decision based on a feature threshold, and each leaf node contains a predicted value (typically the mean of training samples in that region).

## Dataset
- **Name:** California Housing
- **Samples:** 20,640
- **Features:** 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Target:** Median house value (in $100,000s)

## Results
Typical metrics achieved:
- **MSE:** ~0.35
- **RMSE:** ~0.59
- **MAE:** ~0.38
- **R² Score:** ~0.73

## How to Run
```bash
cd day-07-decision-tree-regressor
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:
- `01_eda_plots.png` - Exploratory data analysis (distribution, heatmap, scatter plots)
- `02_depth_tuning.png` - Cross-validation R2 vs max_depth values
- `03_predicted_vs_actual.png` - Predicted vs actual house prices scatter plot
- `04_residuals.png` - Residuals vs predicted values
- `05_feature_importance.png` - Feature importance bar chart
- `06_tree_visualization.png` - Visual representation of the decision tree structure (first 3 levels)
- `07_learning_curve.png` - Learning curve (training vs validation R2)
- `decision_tree_model.pkl` - Trained Decision Tree model (joblib)
- `scaler.pkl` - Fitted StandardScaler (joblib)

## Key Concepts
- **Recursive Binary Splitting:** The tree recursively splits the data into two subsets at each node, choosing the split that minimizes the variance (MSE) of the target in each child node
- **max_depth:** Controls the maximum depth of the tree; limiting depth prevents overfitting by restricting the tree's complexity
- **Feature Importance:** Computed as the total reduction in criterion (variance) brought by each feature across all splits; normalized to sum to 1
- **Overfitting:** Decision trees are prone to overfitting, especially with deep trees; pruning via max_depth, min_samples_split, or min_samples_leaf helps control this
- **Interpretability:** Decision trees are highly interpretable - the prediction path for any sample can be traced from root to leaf, making decisions transparent
- **Non-linear Relationships:** Trees naturally capture non-linear relationships and feature interactions without requiring explicit feature engineering
- **No Scaling Required (in theory):** While decision trees are invariant to monotonic transformations of features, StandardScaler is applied here for consistency and when comparing with other models
