# Day 11: XGBoost Regressor

## Overview
This project implements an XGBoost Regressor for predicting diabetes disease progression. It includes comprehensive EDA, hyperparameter tuning (learning rate and n_estimators), model evaluation, and detailed visualizations including training history.

## Dataset
- **Name**: Diabetes (sklearn)
- **Samples**: 442
- **Features**: 10 (age, sex, bmi, bp, s1-s6 blood serum measurements)
- **Target**: Quantitative measure of disease progression one year after baseline

## Results
| Metric | Value  |
|--------|--------|
| MSE    | ~2800  |
| RMSE   | ~53    |
| MAE    | ~42    |
| R²     | ~0.52  |

## How to Run
```bash
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:
- `01_feature_distributions.png` - Histograms of all features
- `02_correlation_heatmap.png` - Feature correlation matrix
- `03_box_plots.png` - Box plots for outlier detection
- `04_hyperparameter_tuning.png` - Learning rate & n_estimators tuning grid
- `05_predicted_vs_actual.png` - Predicted vs actual scatter plot
- `06_residual_plot.png` - Residuals vs predicted values
- `07_feature_importance.png` - Feature importance bar chart
- `08_learning_curve.png` - Learning curve with confidence bands
- `09_training_history.png` - Training/validation RMSE over boosting rounds
- `10_residual_distribution.png` - Residual distribution with KDE
- `xgboost_regressor.pkl` - Trained model
- `scaler.pkl` - Fitted StandardScaler

## Key Concepts
- **XGBoost**: Gradient boosting framework with regularization to prevent overfitting
- **Learning Rate**: Controls contribution of each tree; smaller values need more estimators
- **Boosting Rounds**: Sequential tree building where each tree corrects previous errors
- **Feature Importance**: Gain-based importance showing contribution to predictions
- **Early Stopping**: Can halt training when validation metric stops improving
