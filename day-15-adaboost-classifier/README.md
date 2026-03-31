# Day 15: AdaBoost Classifier — Breast Cancer

## Overview
This project builds an **AdaBoost Classifier** to predict breast cancer diagnosis (malignant vs benign). AdaBoost (Adaptive Boosting) combines weak learners (decision stumps) sequentially, giving more weight to misclassified samples at each iteration.

## Dataset
- **Name**: Breast Cancer Wisconsin (Diagnostic)
- **Samples**: 569
- **Features**: 30 (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each measured as mean, standard error, and worst)
- **Target**: Diagnosis (0 = Malignant, 1 = Benign)

## Results
Typical metrics after hyperparameter tuning:

| Metric    | Value |
|-----------|-------|
| Accuracy  | ~0.97 |
| Precision | ~0.97 |
| Recall    | ~0.98 |
| F1 Score  | ~0.98 |
| ROC-AUC   | ~0.99 |

## How to Run
```bash
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:

| File | Description |
|------|-------------|
| `01_feature_distributions.png` | Histograms of first 15 features |
| `02_correlation_matrix.png` | Feature correlation heatmap |
| `03_class_distribution.png` | Class count bar chart |
| `04_confusion_matrix.png` | Confusion matrix heatmap |
| `05_roc_curve.png` | ROC curve with AUC score |
| `06_learning_curve.png` | Learning curve (train vs CV accuracy) |
| `07_estimator_weights.png` | Bar chart of AdaBoost estimator weights |
| `08_decision_boundary_pca.png` | 2D decision boundary using PCA |
| `adaboost_classifier.pkl` | Saved model (joblib) |
| `scaler.pkl` | Fitted StandardScaler |

## Key Concepts
- **AdaBoost**: Adaptive boosting that sequentially trains weak learners, increasing weights on misclassified samples
- **n_estimators**: Number of weak learners (decision stumps) in the ensemble
- **learning_rate**: Shrinks contribution of each classifier; balances between under/over-fitting
- **Estimator Weights**: Each weak learner gets a weight based on its accuracy; more accurate classifiers get higher weight
- **Decision Stump**: A one-level decision tree used as the default base estimator in AdaBoost
- **PCA Decision Boundary**: Projects 30 features to 2D using PCA to visualize the classifier's decision regions
