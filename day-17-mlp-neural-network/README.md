# Day 17: MLP Neural Network (Classification)

## Overview

This project implements a Multi-Layer Perceptron (MLP) Neural Network for handwritten digit classification using scikit-learn's `MLPClassifier`. The model is trained on the Digits dataset and includes comprehensive EDA, hyperparameter tuning, evaluation metrics, and visualizations.

## Dataset

- **Name:** Digits Dataset (`load_digits` from sklearn)
- **Samples:** 1,797
- **Features:** 64 (8×8 pixel images, flattened)
- **Target:** 10 classes (digits 0-9)
- **Type:** Multiclass classification

## Model Architecture

- **Algorithm:** MLPClassifier (Feedforward Neural Network)
- **Activation:** ReLU (default)
- **Solver:** Adam optimizer
- **Learning Rate:** Adaptive
- **Early Stopping:** Enabled (validation_fraction=0.15)
- **Hyperparameters Tuned:**
  - `hidden_layer_sizes`: (50,), (100,), (100, 50), (200, 100)
  - `alpha` (L2 regularization): 0.0001, 0.001, 0.01

## Results

Typical performance metrics:

| Metric     | Value  |
|------------|--------|
| Accuracy   | ~0.98  |
| Precision  | ~0.98  |
| Recall     | ~0.98  |
| F1-Score   | ~0.98  |
| ROC-AUC    | ~0.99  |

## How to Run

```bash
cd day-17-mlp-neural-network
python model.py
```

**Requirements:**
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

## Outputs

All outputs are saved to the `outputs/` directory:

| File                            | Description                                    |
|---------------------------------|------------------------------------------------|
| `01_sample_digits.png`          | Grid of first 10 digit images                  |
| `02_class_distribution_pixel_analysis.png` | Class balance & average pixel intensity  |
| `03_pixel_correlation_heatmap.png` | Correlation between pixel features          |
| `04_confusion_matrix.png`       | Confusion matrix heatmap                       |
| `05_sample_predictions.png`     | 15 sample predictions (green=correct, red=wrong) |
| `06_loss_curve.png`             | Training loss over iterations                  |
| `07_learning_curve.png`         | Learning curve (train vs CV accuracy)          |
| `08_roc_curve.png`              | One-vs-Rest ROC curves for all 10 digits       |
| `mlp_digits_model.joblib`       | Saved model (model + scaler + params + metrics) |

## Key Concepts

- **MLP Neural Networks:** Feedforward networks with input, hidden, and output layers using backpropagation for weight updates
- **Adam Optimizer:** Adaptive learning rate optimization combining momentum and RMSProp
- **Regularization (alpha):** L2 penalty to prevent overfitting by constraining weight magnitudes
- **Early Stopping:** Halts training when validation performance stops improving
- **One-vs-Rest ROC:** Extends binary ROC-AUC to multiclass by computing one curve per class
- **StandardScaler:** Z-score normalization critical for neural network convergence
