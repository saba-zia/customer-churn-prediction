# Customer Churn Prediction

## Project Overview

This project uses a real-world customer dataset to understand customer churn and predict whether a customer is likely to leave a company.

Customer churn happens when customers stop using a company's services. Predicting churn is important because keeping existing customers is often cheaper than acquiring new ones.

The goal of this project is not only to build a machine learning model but also to understand and practice the complete workflow of a real machine learning project step by step.

The project follows the process from:

- Loading raw data
- Data exploration
- Data preprocessing
- Model training
- Model evaluation
- Project organization

---

## Dataset

Dataset used:

**Telco Customer Churn Dataset (Kaggle)**

The dataset contains customer information such as:

- Customer demographics
- Internet services
- Contract type
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

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── raw/
│       └── archive/
│           └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── eda.ipynb
├── README.md
└── .gitignore
```

### Folder Explanation

**data/**

Contains the raw dataset used in the project.

**src/data_preprocessing.py**

Cleans and preprocesses the dataset:

- Converts TotalCharges to numeric
- Removes missing values
- Removes customerID
- Converts labels
- Performs one-hot encoding
- Creates X and y

**src/train_model.py**

Responsible for:

- Loading data
- Splitting train and test data
- Scaling features
- Training Logistic Regression

**src/evaluate_model.py**

Responsible for:

- Accuracy calculation
- Confusion Matrix
- Classification Report

**eda.ipynb**

Contains:

- Exploratory Data Analysis
- Visualizations
- Experiments
- Learning notes

---

## Project Progress

## Stage 1 — Data Loading and First Exploration ✔

Completed:

- Loaded dataset with Pandas
- Checked dataset size
- Viewed first rows
- Checked data types
- Looked for missing values
- Generated statistics

Purpose:

Understand the dataset before building any model.

---

## Stage 2 — Exploratory Data Analysis (EDA) ✔

Completed:

- Churn distribution
- Contract type analysis
- Internet service analysis
- Payment method analysis
- Correlation analysis

Main findings:

- Month-to-month contracts showed higher churn
- Fiber optic users showed higher churn
- Electronic check users showed higher churn
- Longer customer tenure reduced churn

---

## Stage 3 — Data Cleaning and Preprocessing ✔

Completed:

- Converted TotalCharges to numeric
- Removed missing values
- Removed customerID
- Converted target labels
- Applied one-hot encoding

Final dataset:

Rows:

7032

Features:

30

---

## Stage 4 — Model Training ✔

Model used:

Logistic Regression

Completed:

- Train/Test split
- Feature scaling
- Model training

Train/Test split:

```python
test_size=0.2
random_state=42
```

Feature scaling:

```python
StandardScaler()
```

---

## Stage 5 — Model Evaluation ✔

Results:

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

Interpretation:

The model performs reasonably well but still misses several customers that actually churn.

---

## Stage 6 — Recall Improvement and ROC Analysis ✔

Completed:

- Probability prediction
- Threshold tuning
- ROC curve analysis
- AUC score calculation

Results:

AUC Score:

```text
0.832
```

Observation:

The model can distinguish customers reasonably well and performs much better than random guessing.

---

## Stage 7 — Project Organization and GitHub Cleanup ✔

Completed:

- Created src folder
- Moved preprocessing code
- Moved training code
- Moved evaluation code
- Improved project structure

Purpose:

Make the repository cleaner and more professional.

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

Completed:

✔ Stage 1  
✔ Stage 2  
✔ Stage 3  
✔ Stage 4  
✔ Stage 5  
✔ Stage 6  
✔ Stage 7  

Current step:

Preparing model comparison and future improvements.