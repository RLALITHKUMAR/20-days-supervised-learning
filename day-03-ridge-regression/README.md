# Day 3: Ridge Regression

## Overview
Ridge regression (L2 regularization) for predicting diabetes disease progression. Demonstrates how L2 penalty shrinks coefficients to prevent overfitting.

## Dataset
- **Diabetes Dataset** (sklearn built-in)
- 442 samples
- 10 features: age, sex, bmi, bp, s1-s6 (blood serum measurements)
- Target: Disease progression one year after baseline

## Results
- **R² Score**: ~0.48
- **RMSE**: ~56.5
- **MAE**: ~44.5

## How to Run
```bash
python model.py
```

## Outputs
- `03_eda.png` - Exploratory data analysis plots
- `03_predicted_vs_actual.png` - Predicted vs actual scatter
- `03_residuals.png` - Residual plot
- `03_coefficient_shrinkage.png` - Coefficient paths vs alpha
- `03_coefficients.png` - Final feature coefficients
- `03_learning_curve.png` - Learning curve
- `ridge_regression_model.pkl` - Saved model

## Key Concepts
- L2 regularization prevents overfitting
- Alpha hyperparameter controls regularization strength
- Coefficient shrinkage visualization
- Feature scaling is critical for regularized models
