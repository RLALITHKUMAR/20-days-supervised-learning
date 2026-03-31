# Day 4: K-Nearest Neighbors Classifier

## Overview
KNN classifier for wine cultivar classification. Demonstrates instance-based learning with optimal K selection.

## Dataset
- **Wine Dataset** (sklearn built-in)
- 178 samples
- 13 features: chemical analysis results
- Target: 3 cultivar classes (0, 1, 2)

## Results
- **Accuracy**: ~0.97
- **Precision**: ~0.97
- **Recall**: ~0.97
- **F1-Score**: ~0.97

## How to Run
```bash
python model.py
```

## Outputs
- `04_eda.png` - Exploratory data analysis
- `04_k_vs_accuracy.png` - K value optimization
- `04_confusion_matrix.png` - Confusion matrix
- `04_decision_boundary.png` - Decision boundary (PCA)
- `04_learning_curve.png` - Learning curve
- `04_feature_analysis.png` - Feature analysis
- `knn_classifier_model.pkl` - Saved model

## Key Concepts
- Instance-based (lazy) learning
- K value selection via cross-validation
- Distance weighting for better predictions
- Feature scaling is critical for KNN
