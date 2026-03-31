# Day 13: LightGBM Regressor — California Housing

## Overview
This project builds a **LightGBM Regressor** to predict median house values in California. LightGBM is a gradient boosting framework that uses tree-based learning algorithms with leaf-wise growth for faster training and better accuracy.

## Dataset
- **Name**: California Housing
- **Samples**: 20,640
- **Features**: 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Target**: Median house value (in $100,000s)

## Results
Typical metrics after hyperparameter tuning:

| Metric | Value |
|--------|-------|
| MSE    | ~0.21 |
| RMSE   | ~0.46 |
| MAE    | ~0.32 |
| R²     | ~0.73 |

## How to Run
```bash
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:

| File | Description |
|------|-------------|
| `01_feature_distributions.png` | Histograms of all 8 features |
| `02_correlation_matrix.png` | Feature correlation heatmap |
| `03_target_distribution.png` | Target variable distribution |
| `04_predicted_vs_actual.png` | Scatter plot of predicted vs actual values |
| `05_residuals_plot.png` | Residuals vs predicted values |
| `06_feature_importance.png` | Feature importance bar chart |
| `07_learning_curve.png` | Learning curve (train vs CV MSE) |
| `08_training_history.png` | Training MSE by iteration |
| `09_residual_distribution.png` | Histogram of residuals |
| `lightgbm_regressor.pkl` | Saved model (joblib) |
| `scaler.pkl` | Fitted StandardScaler |

## Key Concepts
- **LightGBM**: Gradient boosting with leaf-wise tree growth, histogram-based splitting, and GPU support
- **Learning Rate**: Controls step size; lower values need more estimators but generalize better
- **n_estimators**: Number of boosting iterations; tuned alongside learning rate
- **Leaf-wise Growth**: LightGBM grows trees leaf-by-leaf (vs level-wise in XGBoost), enabling faster convergence
- **MSE/RMSE/MAE**: Regression loss metrics; RMSE penalizes large errors more heavily
- **R² Score**: Proportion of variance explained by the model (best = 1.0)
