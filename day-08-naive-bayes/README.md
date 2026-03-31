# Day 8: Naive Bayes Classifier

## Overview
Naive Bayes is a family of probabilistic classifiers based on Bayes' Theorem with the "naive" assumption of conditional independence between features. Gaussian Naive Bayes assumes that continuous features follow a normal (Gaussian) distribution within each class, making it particularly well-suited for datasets where features are approximately normally distributed.

## Dataset
- **Name:** Iris
- **Samples:** 150
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Target:** Species (Setosa, Versicolor, Virginica) - 3 classes, 50 samples each

## Results
Typical metrics achieved:
- **Accuracy:** ~0.93
- **Precision:** ~0.93
- **Recall:** ~0.93
- **F1 Score:** ~0.93
- **ROC-AUC:** ~0.98

## How to Run
```bash
cd day-08-naive-bayes
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:
- `01_eda_plots.png` - Exploratory data analysis (class distribution, heatmap, histograms, scatter plot)
- `02_confusion_matrix.png` - Confusion matrix heatmap
- `03_feature_distributions.png` - Feature distributions by class (KDE plots for all 4 features)
- `04_roc_curves.png` - ROC curves (one-vs-rest) for all 3 classes
- `05_decision_boundary.png` - 2D decision boundary visualization using PCA
- `06_learning_curve.png` - Learning curve (training vs validation accuracy)
- `naive_bayes_model.pkl` - Trained Gaussian NB model (joblib)
- `scaler.pkl` - Fitted StandardScaler (joblib)

## Key Concepts
- **Bayes' Theorem:** P(class|features) = P(features|class) * P(class) / P(features); the classifier computes the posterior probability of each class given the features
- **Conditional Independence Assumption:** Features are assumed to be independent given the class label; this "naive" assumption simplifies computation but is rarely true in practice
- **Gaussian Likelihood:** For continuous features, the likelihood P(feature|class) is modeled using a Gaussian distribution with class-specific mean (theta) and variance (sigma^2)
- **No Hyperparameters:** Gaussian NB has no tunable hyperparameters - it simply estimates the mean and variance of each feature for each class from the training data
- **Multiclass ROC-AUC:** Computed using a one-vs-rest approach, where each class is treated as positive and all others as negative
- **Efficiency:** Extremely fast to train and predict, making it ideal for large datasets and real-time applications
- **StandardScaler:** While Naive Bayes is theoretically invariant to feature scaling (since it estimates per-class distributions), scaling can improve numerical stability and is essential when comparing with other models
