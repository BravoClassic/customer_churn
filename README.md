# Customer Churn Prediction

A machine learning project for predicting whether a customer is likely to churn based on customer account information, usage behavior, and service-related features.

## Overview

Customer churn prediction helps businesses identify customers who are at risk of leaving. This project uses supervised machine learning techniques to classify customers as likely to churn or not churn, giving teams a stronger foundation for customer retention strategy.

## Problem Statement

Customer churn directly affects revenue, growth, and long-term customer value. The goal of this project is to build a predictive model that can identify high-risk customers before they leave, using historical customer data and classification models.

## Objectives

- Explore customer behavior and churn patterns
- Clean and preprocess customer data
- Encode categorical variables and prepare features for modeling
- Train machine learning classification models
- Evaluate model performance using reliable classification metrics
- Identify key factors that influence customer churn

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Machine Learning Workflow

1. Load and inspect the dataset
2. Perform exploratory data analysis
3. Clean missing or inconsistent values
4. Encode categorical features
5. Split the data into training and testing sets
6. Train classification models
7. Evaluate model performance
8. Analyze feature importance and churn drivers

## Models

This project can support multiple classification models, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

## Evaluation Metrics

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC Score

For churn prediction, recall and F1-score are especially important because the business goal is often to correctly identify customers who are likely to leave.

## Project Structure

```text
customer_churn/
├── data/
│   └── customer_churn.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── README.md
└── requirements.txt
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/BravoClassic/customer_churn.git
cd customer_churn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook
```

Or run the training script if available:

```bash
python src/train_model.py
```

## Results

The final model predicts customer churn and provides insight into the customer attributes most associated with churn. This project demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, model evaluation, and churn risk interpretation.

## Future Improvements

- Add hyperparameter tuning
- Handle class imbalance with class weights or SMOTE
- Track experiments with MLflow
- Deploy the model with FastAPI
- Build a dashboard for churn risk monitoring
- Add automated model retraining

## Resume Summary

Built a customer churn prediction pipeline using Python, Pandas, and scikit-learn, applying preprocessing, feature engineering, and classification models to identify high-risk customers and evaluate retention signals using precision, recall, F1-score, and ROC-AUC.

## Author

Gerald Akorli
