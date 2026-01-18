# Credit Default Prediction – End-to-End ML Deployment

## 📌 Project Overview
This project is an **end-to-end Machine Learning system** for predicting **credit card default risk**.  
It covers the complete ML lifecycle — **data preprocessing, model training, evaluation, and real-time deployment using FastAPI**.

The system exposes a REST API that accepts customer financial attributes and returns:
- Probability of default
- Binary default prediction

---

## 🎯 Problem Statement
Financial institutions must identify customers likely to default on credit card payments to reduce financial risk and improve credit decision-making.

This project solves the problem using supervised machine learning and deploys it as a production-ready API.

---

## 🧠 Solution Approach

### 1️⃣ Data Processing
- Structured credit card transaction dataset
- Feature selection and cleaning
- Numerical feature scaling
- Target variable: `default` (0 = No Default, 1 = Default)

---

### 2️⃣ Model Training
- Baseline: Logistic Regression
- Final model: **Random Forest Classifier**
- Reasoning:
  - Captures non-linear relationships
  - Performs well on tabular financial data
  - Robust to noise and outliers

A **scikit-learn Pipeline** is used to bundle preprocessing and modeling, ensuring training–serving consistency.

---

### 3️⃣ Model Evaluation
Evaluation metrics:
- ROC-AUC
- Recall (important for default detection)
- Confusion Matrix

The final model balances detection performance with false positives.

---

## 🚀 Deployment Architecture

