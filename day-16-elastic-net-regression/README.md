# Day 16: Elastic Net Regression — Diabetes

## Overview
This project builds an **Elastic Net Regression** model to predict disease progression in diabetes patients. Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization, offering the benefits of both: feature selection from Lasso and handling of correlated features from Ridge.

## Dataset
- **Name**: Diabetes
- **Samples**: 442
- **Features**: 10 (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6 — standardized physiological measurements)
- **Target**: Quantitative measure of disease progression one year after baseline

## Results
Typical metrics after hyperparameter tuning:

| Metric | Value |
|--------|-------|
| MSE    | ~2900 |
| RMSE   | ~54   |
| MAE    | ~43   |
| R²     | ~0.49 |

## How to Run
```bash
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:

| File | Description |
|------|-------------|
| `01_feature_distributions.png` | Histograms of all 10 features |
| `02_correlation_matrix.png` | Feature correlation heatmap |
| `03_target_distribution.png` | Target variable distribution |
| `04_predicted_vs_actual.png` | Scatter plot of predicted vs actual values |
| `05_residuals_plot.png` | Residuals vs predicted values |
| `06_coefficient_paths.png` | Coefficient paths across alphas + final coefficients |
| `07_learning_curve.png` | Learning curve (train vs CV MSE) |
| `08_residual_distribution.png` | Histogram of residuals |
| `09_alpha_comparison.png` | MSE comparison across alpha values and L1 ratios |
| `elastic_net_model.pkl` | Saved model (joblib) |
| `scaler.pkl` | Fitted StandardScaler |

## Key Concepts
- **Elastic Net**: Linear regression with combined L1 and L2 penalties: `penalty = alpha * (l1_ratio * L1 + (1 - l1_ratio) * L2)`
- **alpha**: Overall regularization strength; higher values shrink coefficients more aggressively
- **l1_ratio**: Mix between Lasso (l1_ratio=1) and Ridge (l1_ratio=0); controls the balance between feature selection and coefficient shrinkage
- **L1 Regularization (Lasso)**: Drives some coefficients to exactly zero, performing automatic feature selection
- **L2 Regularization (Ridge)**: Shrinks all coefficients proportionally, handles multicollinearity well
- **Coefficient Paths**: Tracking how coefficients change as alpha varies reveals which features are most robust to regularization
- **StandardScaler**: Critical for regularized models since penalty terms are sensitive to feature scales
