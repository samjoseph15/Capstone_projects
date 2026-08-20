# Intelligent-E-Commerce-Customer-Analytics-Platform


## 📌 Overview

This project uses the **Olist Brazilian E-Commerce dataset** to analyze customer behavior and build machine-learning models for:

* Predicting repeat purchases within 90 days.
* Predicting future customer revenue.

## 🔄 Workflow

```text
Raw Olist Data
      ↓
Data Preprocessing
      ↓
Data Cleaning & Merging
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Model Evaluation
```

## 📂 Notebooks

### `data_preprocessing.ipynb`

* Loads and cleans Olist datasets.
* Handles missing values.
* Merges customer, order, payment, product, and review data.
* Performs feature engineering.
* Creates customer-level datasets.

### `ml_analysis_1.ipynb`

* Performs classification and regression.
* Compares multiple ML models.
* Evaluates model performance.
* Saves the trained models.

## 🤖 Machine Learning

### Classification

**Goal:** Predict whether a customer will make another purchase within 90 days.

**Best model:** XGBoost

Output:

```text
xgb_model.pkl
```

### Regression

**Goal:** Predict future customer revenue.

**Best model:** Linear Regression

Output:

```text
lr_model.pkl
```

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Matplotlib
* Jupyter Notebook

## 📊 Key Tasks

* Data preprocessing
* Data cleaning
* Feature engineering
* Customer segmentation
* Classification
* Regression
* Model evaluation
* Model saving

## 🚀 Future Improvements

* Hyperparameter tuning
* Feature importance and SHAP analysis
* Streamlit dashboard
* Model deployment
