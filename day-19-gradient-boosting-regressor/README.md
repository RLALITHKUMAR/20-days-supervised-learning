# Day 19: Gradient Boosting Regressor

## Overview

This project implements a Gradient Boosting Regressor for predicting diabetes disease progression using scikit-learn. The model uses sequential ensemble learning with hyperparameter tuning, comprehensive evaluation metrics, and detailed visualizations.

## Dataset

- **Name:** Diabetes Dataset (`load_diabetes` from sklearn)
- **Samples:** 442
- **Features:** 10 (age, sex, bmi, bp, s1-s6 blood serum measurements)
- **Target:** Quantitative measure of disease progression one year after baseline
- **Type:** Regression

## Model Configuration

- **Algorithm:** GradientBoostingRegressor
- **Hyperparameters Tuned:**
  - `n_estimators`: 50, 100, 200, 300
  - `learning_rate`: 0.01, 0.05, 0.1, 0.2
- **Fixed Parameters:**
  - `max_depth`: 4
  - `subsample`: 0.8
  - `random_state`: 42

## Results

Typical performance metrics:

| Metric | Value   |
|--------|---------|
| MSE    | ~2500   |
| RMSE   | ~50     |
| MAE    | ~38     |
| R²     | ~0.50   |

## How to Run

```bash
cd day-19-gradient-boosting-regressor
python model.py
```

**Requirements:**
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- joblib

## Outputs

All outputs are saved to the `outputs/` directory:

| File                                | Description                                    |
|-------------------------------------|------------------------------------------------|
| `01_feature_distributions.png`      | Histograms of all 10 features                  |
| `02_target_analysis_correlations.png` | Target distribution & feature correlations   |
| `03_correlation_heatmap.png`        | Upper-triangle correlation heatmap             |
| `04_predicted_vs_actual.png`        | Scatter plot of predicted vs actual values     |
| `05_residual_analysis.png`          | Residuals vs predicted + residual distribution with KDE |
| `06_feature_importance.png`         | Feature importance bar chart                   |
| `07_learning_curve.png`             | Learning curve (train vs CV R²)                |
| `08_training_history.png`           | Staged predictions (MSE & R² vs estimators)    |
| `gbr_diabetes_model.joblib`         | Saved model (model + scaler + params + metrics) |

## Key Concepts

- **Gradient Boosting:** Sequential ensemble method where each new tree corrects errors of the previous ensemble
- **Learning Rate:** Shrinks contribution of each tree; lower values need more estimators but generalize better
- **Staged Predictions:** Access predictions at each boosting stage to analyze training dynamics
- **Feature Importance:** Computed as total reduction of criterion (MSE) brought by each feature across all trees
- **Subsampling:** Using a fraction of training data for each tree reduces variance and speeds up training
- **Bias-Variance Tradeoff:** More estimators reduce bias but risk overfitting; learning rate controls this balance
