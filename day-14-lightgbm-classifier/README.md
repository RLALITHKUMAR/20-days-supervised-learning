# Day 14: LightGBM Classifier — Wine Dataset

## Overview
This project builds a **LightGBM Classifier** to classify wine cultivars into three classes. LightGBM uses gradient boosting with leaf-wise tree growth for fast, accurate multi-class classification.

## Dataset
- **Name**: Wine
- **Samples**: 178
- **Features**: 13 (Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315, Proline)
- **Target**: Wine cultivar class (3 classes: Class 0, Class 1, Class 2)

## Results
Typical metrics after hyperparameter tuning:

| Metric    | Value |
|-----------|-------|
| Accuracy  | ~0.97 |
| Precision | ~0.97 |
| Recall    | ~0.97 |
| F1 Score  | ~0.97 |
| ROC-AUC   | ~0.99 |

## How to Run
```bash
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:

| File | Description |
|------|-------------|
| `01_feature_distributions.png` | Histograms of all 13 features |
| `02_correlation_matrix.png` | Feature correlation heatmap |
| `03_class_distribution.png` | Class count bar chart |
| `04_confusion_matrix.png` | Confusion matrix heatmap |
| `05_roc_curve.png` | One-vs-rest ROC curves for all 3 classes |
| `06_feature_importance.png` | Feature importance bar chart |
| `07_learning_curve.png` | Learning curve (train vs CV accuracy) |
| `08_training_history.png` | Training log loss by iteration |
| `lightgbm_classifier.pkl` | Saved model (joblib) |
| `scaler.pkl` | Fitted StandardScaler |

## Key Concepts
- **LightGBM Classifier**: Gradient boosting framework optimized for classification with leaf-wise growth
- **max_depth**: Controls tree depth; prevents overfitting on small datasets
- **learning_rate**: Shrinks contribution of each tree; lower values need more estimators
- **One-vs-Rest ROC-AUC**: Multi-class ROC computed by treating each class as positive vs all others
- **Multi Log Loss**: Classification loss function used during LightGBM training
- **Stratified Split**: Preserves class distribution in train/test splits for imbalanced data
