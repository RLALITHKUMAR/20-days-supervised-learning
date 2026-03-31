# Day 18: Support Vector Regressor (SVR)

## Overview

This project implements a Support Vector Regressor (SVR) for predicting California housing prices using scikit-learn. The model uses the RBF kernel with hyperparameter tuning, comprehensive evaluation metrics, and detailed visualizations.

## Dataset

- **Name:** California Housing (`fetch_california_housing` from sklearn)
- **Samples:** 20,640
- **Features:** 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Target:** Median house value (in $100,000s)
- **Type:** Regression

## Model Configuration

- **Algorithm:** Support Vector Regressor (SVR)
- **Kernel:** RBF (Radial Basis Function)
- **Hyperparameters Tuned:**
  - `C` (regularization): 0.1, 1.0, 10.0, 100.0
  - `epsilon` (insensitive zone): 0.01, 0.1, 0.5, 1.0
- **Gamma:** 'scale' (auto-calculated)

## Results

Typical performance metrics:

| Metric | Value   |
|--------|---------|
| MSE    | ~0.35   |
| RMSE   | ~0.59   |
| MAE    | ~0.38   |
| R²     | ~0.65   |

## How to Run

```bash
cd day-18-svr
python model.py
```

**Requirements:**
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

## Outputs

All outputs are saved to the `outputs/` directory:

| File                                | Description                                    |
|-------------------------------------|------------------------------------------------|
| `01_feature_distributions.png`      | Histograms of all 8 features                   |
| `02_target_analysis_correlations.png` | Target distribution & feature correlations   |
| `03_correlation_heatmap.png`        | Upper-triangle correlation heatmap             |
| `04_predicted_vs_actual.png`        | Scatter plot of predicted vs actual values     |
| `05_residual_analysis.png`          | Residuals vs predicted + residual distribution |
| `06_learning_curve.png`             | Learning curve (train vs CV R²)                |
| `07_kernel_comparison.png`          | RBF vs Linear vs Poly kernel comparison        |
| `svr_california_model.joblib`       | Saved model (model + scaler + params + metrics) |

## Key Concepts

- **Support Vector Regression:** Extends SVM to regression by finding a tube of width 2ε around the prediction function
- **RBF Kernel:** Maps data to infinite-dimensional space, capturing non-linear relationships
- **C Parameter:** Controls trade-off between model complexity and training error tolerance
- **Epsilon (ε):** Defines the insensitive zone; errors within ε are not penalized
- **Feature Scaling:** Critical for SVR since it relies on distance calculations
- **Residual Analysis:** Checking for patterns in residuals to validate model assumptions
