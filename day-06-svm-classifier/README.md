# Day 6: SVM Classifier

## Overview
Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification tasks. SVM finds the optimal hyperplane that maximally separates classes in a high-dimensional feature space. Using the RBF (Radial Basis Function) kernel, SVM can capture complex non-linear decision boundaries by implicitly mapping data to infinite-dimensional space.

## Dataset
- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Samples:** 569
- **Features:** 30 (computed from digitized images of fine needle aspirates of breast masses)
- **Target:** Diagnosis (Malignant = 0, Benign = 1)

## Results
Typical metrics achieved:
- **Accuracy:** ~0.97
- **Precision:** ~0.98
- **Recall:** ~0.97
- **F1 Score:** ~0.97
- **ROC-AUC:** ~0.99

## How to Run
```bash
cd day-06-svm-classifier
python model.py
```

## Outputs
All files are saved in the `outputs/` directory:
- `01_eda_plots.png` - Exploratory data analysis (class distribution, heatmap, KDE plots, box plots)
- `02_confusion_matrix.png` - Confusion matrix heatmap
- `03_roc_curve.png` - Receiver Operating Characteristic curve with AUC
- `04_decision_boundary.png` - 2D decision boundary visualization using PCA
- `05_learning_curve.png` - Learning curve (training vs validation accuracy)
- `06_precision_recall_curve.png` - Precision-Recall curve with Average Precision
- `svm_model.pkl` - Trained SVM model (joblib)
- `scaler.pkl` - Fitted StandardScaler (joblib)

## Key Concepts
- **Maximum Margin Hyperplane:** SVM finds the decision boundary that maximizes the margin between classes, leading to better generalization
- **Support Vectors:** Only the data points closest to the decision boundary (support vectors) influence the model, making SVM memory efficient
- **RBF Kernel:** The Radial Basis Function kernel computes similarity using a Gaussian function, enabling SVM to handle non-linearly separable data by mapping it to infinite-dimensional space
- **C Parameter:** Controls the trade-off between achieving a low training error and a low testing error (regularization strength)
- **Gamma Parameter:** Defines how far the influence of a single training example reaches; low gamma means far reach, high gamma means close reach
- **Probability Estimates:** Setting `probability=True` enables Platt scaling to convert decision function outputs to calibrated probabilities, required for ROC-AUC computation
- **StandardScaler:** Critical for SVM since it relies on distance calculations; unscaled features with larger ranges would dominate the kernel computation
