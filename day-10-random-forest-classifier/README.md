# Day 10: Random Forest Classifier

## Overview
This project implements a Random Forest Classifier for handwritten digit recognition using the Digits dataset. It includes comprehensive EDA, hyperparameter tuning, model evaluation, and detailed visualizations.

## Dataset
- **Name**: Digits (sklearn)
- **Samples**: 1,797
- **Features**: 64 (8x8 pixel values, 0-16)
- **Target**: Digit class (0-9)

## Results
| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~0.98  |
| Precision | ~0.98  |
| Recall    | ~0.98  |
| F1 Score  | ~0.98  |
| ROC-AUC   | ~0.99  |

## How to Run
```bash
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:
- `01_sample_digits.png` - Sample digit images with labels
- `02_class_distribution.png` - Class balance bar chart
- `03_feature_correlation.png` - Top features correlation heatmap
- `04_n_estimators_tuning.png` - n_estimators tuning curve
- `05_confusion_matrix.png` - Confusion matrix heatmap
- `06_sample_predictions.png` - Sample predictions grid (green=correct, red=wrong)
- `07_feature_importance.png` - Pixel-level feature importance heatmap
- `08_learning_curve.png` - Learning curve with confidence bands
- `09_roc_curve.png` - One-vs-rest ROC curves for all digits
- `random_forest_classifier.pkl` - Trained model
- `scaler.pkl` - Fitted StandardScaler

## Key Concepts
- **Random Forest Classifier**: Ensemble of decision trees for multi-class classification
- **Stratified Split**: Preserves class distribution in train/test splits
- **Multi-class ROC-AUC**: One-vs-rest approach for evaluating each class
- **Confusion Matrix**: Shows per-class misclassification patterns
- **Pixel Importance**: Visualizing which pixels matter most for classification
