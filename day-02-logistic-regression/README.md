# Day 2: Logistic Regression

## Overview
Logistic regression classifier for breast cancer diagnosis (malignant vs benign) based on cell nucleus features.

## Dataset
- **Breast Cancer Wisconsin Dataset** (sklearn built-in)
- 569 samples
- 30 features computed from digitized images of fine needle aspirates
- Target: Malignant (0) or Benign (1)

## Results
- **Accuracy**: ~0.98
- **Precision**: ~0.98
- **Recall**: ~0.97
- **F1-Score**: ~0.97
- **ROC-AUC**: ~0.99

## How to Run
```bash
python model.py
```

## Outputs
- `02_eda.png` - Exploratory data analysis plots
- `02_confusion_matrix.png` - Confusion matrix heatmap
- `02_roc_curve.png` - ROC curve with AUC
- `02_precision_recall.png` - Precision-Recall curve
- `02_coefficients.png` - Feature coefficient values
- `02_learning_curve.png` - Learning curve
- `logistic_regression_model.pkl` - Saved model
- `scaler.pkl` - Fitted scaler

## Key Concepts
- Binary classification with logistic regression
- Class stratification in train/test split
- ROC-AUC evaluation metric
- Precision-Recall tradeoff analysis
- Feature importance via coefficients
