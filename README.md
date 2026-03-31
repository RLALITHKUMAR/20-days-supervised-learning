# 20 Days of Supervised Learning

A comprehensive collection of 20 supervised learning models, built one per day. Each model is self-contained with EDA, preprocessing, training, evaluation, and visualizations.

## Setup

```bash
pip install -r requirements.txt
```

## Daily Progress

| Day | Model | Type | Dataset | Status |
|-----|-------|------|---------|--------|
| 1 | Linear Regression | Regression | California Housing | ✅ |
| 2 | Logistic Regression | Classification | Breast Cancer | ✅ |
| 3 | Ridge Regression | Regression | Diabetes | ✅ |
| 4 | KNN Classifier | Classification | Wine | ✅ |
| 5 | Lasso Regression | Regression | California Housing | ✅ |
| 6 | SVM Classifier | Classification | Breast Cancer | ✅ |
| 7 | Decision Tree Regressor | Regression | California Housing | ✅ |
| 8 | Naive Bayes | Classification | Iris | ✅ |
| 9 | Random Forest Regressor | Regression | California Housing | ✅ |
| 10 | Random Forest Classifier | Classification | Digits | ✅ |
| 11 | XGBoost Regressor | Regression | Diabetes | ✅ |
| 12 | XGBoost Classifier | Classification | Breast Cancer | ✅ |
| 13 | LightGBM Regressor | Regression | California Housing | ✅ |
| 14 | LightGBM Classifier | Classification | Wine | ✅ |
| 15 | AdaBoost Classifier | Classification | Breast Cancer | ✅ |
| 16 | Elastic Net Regression | Regression | Diabetes | ✅ |
| 17 | MLP Neural Network | Classification | Digits | ✅ |
| 18 | SVR | Regression | California Housing | ✅ |
| 19 | Gradient Boosting Regressor | Regression | Diabetes | ✅ |
| 20 | Gradient Boosting Classifier | Classification | Breast Cancer | ✅ |

## How to Run

```bash
cd day-01-linear-regression
python model.py
```

Each script will:
- Load and explore the dataset
- Preprocess the data
- Train the model with hyperparameter tuning
- Evaluate performance
- Save visualizations to the `outputs/` folder
- Save the trained model using joblib

## Tech Stack

- Python 3.8+
- scikit-learn
- XGBoost
- LightGBM
- Matplotlib & Seaborn
- NumPy & Pandas

## Author

Built for resume portfolio showcasing supervised learning expertise.
