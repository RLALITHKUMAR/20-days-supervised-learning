# Day 12: XGBoost Classifier

## Overview
This project implements an XGBoost Classifier for breast cancer diagnosis (malignant vs benign). It includes comprehensive EDA, hyperparameter tuning (max_depth and learning_rate), model evaluation, and detailed visualizations including training history.

## Dataset
- **Name**: Breast Cancer Wisconsin (sklearn)
- **Samples**: 569
- **Features**: 30 (mean, standard error, and worst of 10 cell nucleus measurements)
- **Target**: Diagnosis (Malignant = 0, Benign = 1)

## Results
| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~0.98  |
| Precision | ~0.97  |
| Recall    | ~0.99  |
| F1 Score  | ~0.98  |
| ROC-AUC   | ~0.99  |

## How to Run
```bash
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:
- `01_feature_distributions.png` - Histograms of top 10 features
- `02_correlation_heatmap.png` - Feature correlation matrix
- `03_class_distribution.png` - Class balance bar chart
- `04_hyperparameter_tuning.png` - max_depth & learning_rate tuning grid
- `05_confusion_matrix.png` - Confusion matrix heatmap
- `06_roc_curve.png` - ROC curve with AUC score
- `07_feature_importance.png` - Feature importance bar chart
- `08_learning_curve.png` - Learning curve with confidence bands
- `09_training_history.png` - Training/validation log loss over boosting rounds
- `xgboost_classifier.pkl` - Trained model
- `scaler.pkl` - Fitted StandardScaler

## Key Concepts
- **XGBoost Classifier**: Gradient boosting for binary classification with regularization
- **max_depth**: Controls tree complexity; deeper trees capture more interactions but risk overfitting
- **Log Loss**: Proper scoring rule that penalizes confident wrong predictions
- **Stratified Split**: Maintains class ratio in train/test for reliable evaluation
- **Feature Importance**: Gain-based importance showing which cell measurements matter most
