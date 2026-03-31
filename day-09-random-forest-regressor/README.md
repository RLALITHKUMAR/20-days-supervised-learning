# Day 9: Random Forest Regressor

## Overview
This project implements a Random Forest Regressor for predicting California housing prices. It includes comprehensive EDA, hyperparameter tuning, model evaluation, and detailed visualizations.

## Dataset
- **Name**: California Housing
- **Samples**: 20,640
- **Features**: 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Target**: Median house value (in $100,000s)

## Results
| Metric | Value |
|--------|-------|
| MSE    | ~0.25 |
| RMSE   | ~0.50 |
| MAE    | ~0.35 |
| R²     | ~0.82 |

## How to Run
```bash
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:
- `01_feature_distributions.png` - Histograms of all features
- `02_correlation_heatmap.png` - Feature correlation matrix
- `03_box_plots.png` - Box plots for outlier detection
- `04_oob_score_plot.png` - n_estimators tuning curve
- `05_predicted_vs_actual.png` - Predicted vs actual scatter plot
- `06_residual_plot.png` - Residuals vs predicted values
- `07_feature_importance.png` - Feature importance bar chart
- `08_learning_curve.png` - Learning curve with confidence bands
- `09_residual_distribution.png` - Residual distribution with KDE
- `random_forest_regressor.pkl` - Trained model
- `scaler.pkl` - Fitted StandardScaler

## Key Concepts
- **Random Forest**: Ensemble of decision trees using bagging and random feature selection
- **OOB Score**: Out-of-bag error estimate using samples not in bootstrap
- **Feature Importance**: Mean decrease in impurity across all trees
- **StandardScaler**: Z-score normalization for consistent feature scaling
