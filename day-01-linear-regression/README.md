# Day 1: Linear Regression

## Overview
Simple linear regression model to predict California housing prices based on features like median income, house age, average rooms, etc.

## Dataset
- **California Housing Dataset** (sklearn built-in)
- 20,640 samples
- 8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- Target: Median house value (in $100,000s)

## Results
- **R² Score**: ~0.60
- **RMSE**: ~0.72
- **MAE**: ~0.52

## How to Run
```bash
python model.py
```

## Outputs
- `01_eda.png` - Exploratory data analysis plots
- `01_predicted_vs_actual.png` - Predicted vs actual values scatter
- `01_residuals.png` - Residual plot
- `01_coefficients.png` - Feature coefficient values
- `01_learning_curve.png` - Learning curve
- `01_residual_distribution.png` - Residual distribution histogram
- `linear_regression_model.pkl` - Saved model
- `scaler.pkl` - Fitted scaler

## Key Concepts
- Feature scaling with StandardScaler
- Cross-validation for robust evaluation
- Coefficient interpretation
- Residual analysis for model diagnostics
