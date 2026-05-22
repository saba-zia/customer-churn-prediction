# Customer Churn Prediction

## Project Overview

This project uses a real-world customer dataset to understand customer churn and predict whether a customer is likely to leave a company.

Customer churn happens when a customer stops using a company's service. Predicting churn is important because keeping existing customers is usually cheaper than finding new ones.

The goal of this project is not only to build a machine learning model, but also to understand the full workflow of a real ML project step by step — from loading raw data to evaluating model performance.

---

## Dataset

Dataset used:

**Telco Customer Churn Dataset (Kaggle)**

The dataset contains information about customers such as:

- Customer demographics
- Services being used
- Contract type
- Internet service
- Payment methods
- Monthly charges
- Total charges
- Customer tenure
- Churn status

Each row represents one customer.

---

## Main Question

Can customer information and service usage help predict whether a customer will churn?

---

# Project Progress

---

## Stage 1 — Data Loading and First Exploration ✔

In this stage I started by loading the dataset and getting a first understanding of the data.

Steps completed:

- Loaded dataset with Pandas
- Checked dataset size
- Viewed first rows
- Checked data types
- Looked for missing values
- Generated basic statistics

Purpose:

Before building any model, it is important to understand what the data looks like and whether there are any obvious problems.

---

## Stage 2 — Exploratory Data Analysis (EDA) ✔

In this stage I explored the relationship between customer information and churn.

Visualizations created:

- Churn distribution
- Contract type vs Churn
- Internet service vs Churn
- Payment method vs Churn
- Correlation analysis
- Customer charge patterns

Main observations:

### Contract type

Customers with month-to-month contracts had the highest churn rate.

Customers with one-year and two-year contracts were more likely to stay.

Possible explanation:

Long-term contracts may increase customer stability and reduce churn.

---

### Internet service

Customers using Fiber Optic internet showed higher churn rates.

Customers without internet service had the lowest churn rate.

Possible explanation:

Fiber users may have higher expectations regarding price or service quality.

---

### Payment methods

Customers using Electronic Check showed noticeably higher churn.

Automatic payment methods had lower churn rates.

Possible explanation:

Customers with automatic payments may have more stable subscriptions.

---

### Correlation analysis

Some interesting relationships appeared:

- Tenure had a negative relationship with churn
- Monthly charges had a weak positive relationship with churn
- Total charges had a weak negative relationship with churn

Meaning:

Customers who stay longer generally churn less.

---

## Stage 3 — Data Cleaning and Preprocessing ✔

Raw data cannot directly be used for machine learning, so preprocessing was necessary.

Steps completed:

### Converted TotalCharges

Converted values to numeric format:

```python
pd.to_numeric()
```

Reason:

Some values were stored as text.

---

### Missing values

Found missing values:

```text
TotalCharges → 11 missing values
```

Removed missing rows:

```python
dropna()
```

Reason:

Machine learning models cannot work with missing values.

---

### Removed customerID

Removed:

```text
customerID
```

Reason:

Customer IDs do not contain useful information for prediction.

---

### Converted target labels

Converted:

```text
No → 0
Yes → 1
```

Reason:

Machine learning models work with numbers.

---

### One-hot encoding

Applied:

```python
pd.get_dummies()
```

Reason:

Machine learning models cannot understand text categories directly.

Dataset after preprocessing:

Rows:

```text
7032
```

Features:

```text
30
```

---

## Stage 4 — Model Training ✔

For the first model I used Logistic Regression.

Steps:

- Split dataset into training and testing data
- Applied feature scaling
- Trained Logistic Regression model

Train-test split:

```python
test_size=0.2
random_state=42
```

Reason:

Training data teaches the model.

Testing data checks whether the model learned correctly on unseen data.

Feature scaling:

```python
StandardScaler()
```

Reason:

Features have different scales and large values can dominate smaller ones.

---

## Stage 5 — Model Evaluation ✔

After training the model, I evaluated its performance.

Accuracy:

```text
78.75%
```

Confusion Matrix:

```text
[[915 118]
 [181 193]]
```

Classification Report:

```text
Precision (Churn): 0.62
Recall (Churn): 0.52
F1-score (Churn): 0.56
```

Result interpretation:

The model performs reasonably well overall.

The model predicts customers who stay quite well, but it still misses many customers who actually leave.

This means there is still room for improvement.

Possible future improvements:

- Random Forest
- Decision Tree
- XGBoost
- Hyperparameter tuning
- Feature importance analysis
- Improve recall score

---

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git
- GitHub

---

## Current Status

Stage 1 ✔

Stage 2 ✔

Stage 3 ✔

Stage 4 ✔

Stage 5 ✔

Current step:

Preparing model improvements and model comparison.
