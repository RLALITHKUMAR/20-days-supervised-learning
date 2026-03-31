# Day 5: Lasso Regression

## Overview
Lasso (Least Absolute Shrinkage and Selection Operator) Regression is a linear regression technique that performs both regularization and feature selection. By adding an L1 penalty to the loss function, Lasso drives some coefficients to exactly zero, effectively selecting the most important features.

## Dataset
- **Name:** California Housing
- **Samples:** 20,640
- **Features:** 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Target:** Median house value (in $100,000s)

## Results
Typical metrics achieved:
- **MSE:** ~0.52
- **RMSE:** ~0.72
- **MAE:** ~0.52
- **R² Score:** ~0.60

## How to Run
```bash
cd day-05-lasso-regression
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:
- `01_eda_plots.png` - Exploratory data analysis (distribution, heatmap, scatter, box plots)
- `02_alpha_tuning.png` - Cross-validation R2 vs alpha values
- `03_predicted_vs_actual.png` - Predicted vs actual house prices scatter plot
- `04_residuals.png` - Residuals vs predicted values
- `05_coefficient_paths.png` - Coefficient paths showing feature selection across alpha values
- `06_learning_curve.png` - Learning curve (training vs validation R2)
- `07_residual_distribution.png` - Histogram of residual distribution
- `lasso_model.pkl` - Trained Lasso model (joblib)
- `scaler.pkl` - Fitted StandardScaler (joblib)

## Key Concepts
- **L1 Regularization:** Adds the absolute value of coefficients as penalty term, encouraging sparsity
- **Feature Selection:** Lasso naturally performs feature selection by shrinking less important feature coefficients to exactly zero
- **Alpha (λ):** Controls the strength of regularization; higher values lead to more coefficients being zeroed out
- **StandardScaler:** Essential for Lasso since it penalizes coefficient magnitudes; features must be on the same scale
- **Coefficient Paths:** Visualizing how coefficients change with different alpha values reveals which features survive regularization
- **Bias-Variance Tradeoff:** Lasso reduces model variance at the cost of introducing some bias, improving generalization
