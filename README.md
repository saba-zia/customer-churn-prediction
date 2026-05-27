## Live Demo

Try the deployed application here:

https://customer-churn-prediction-saba.streamlit.app
# Customer Churn Prediction

## Project Overview

This project uses a real-world customer dataset to understand customer churn and predict whether a customer is likely to leave a company.

Customer churn happens when customers stop using a company's services. Predicting churn is important because keeping existing customers is often cheaper than acquiring new ones.

The main goal of this project is not only to build a machine learning model, but also to understand the complete workflow of a real machine learning project step by step.

The workflow includes:

- Loading raw data
- Data exploration
- Data preprocessing
- Model training
- Model evaluation
- Model improvement
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

Can customer information and customer service usage help predict whether a customer will churn?

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
│   ├── evaluate_model.py
│   └── feature_importance.py
│
├── eda.ipynb
├── README.md
├── .gitignore
└── requirements.txt
```

### Folder Explanation

**data/**

Contains the original dataset.

**src/data_preprocessing.py**

Responsible for:

- Converting TotalCharges into numeric values
- Removing missing values
- Removing customerID
- Converting labels
- One-hot encoding
- Creating X and y

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

**src/feature_importance.py**

Responsible for:

- Understanding which features influence churn the most

**eda.ipynb**

Contains:

- Exploratory Data Analysis
- Visualizations
- Experiments
- Learning notes

---

## Project Progress

---

## Stage 1 — Data Loading and First Exploration ✔

Completed:

- Loaded dataset using Pandas
- Checked dataset size
- Viewed first rows
- Checked data types
- Looked for missing values
- Generated statistics

Purpose:

Understand the dataset before building models.

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

```text
7032
```

Features:

```text
30
```

---

## Stage 4 — Model Training ✔

Model used:

**Logistic Regression**

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

Reason:

Features have different scales and scaling helps the model learn better.

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

The model performs reasonably well overall but still misses some customers who actually churn.

---

## Stage 6 — Model Improvement and ROC Analysis ✔

Completed:

- Probability prediction
- Threshold tuning
- Recall improvement
- ROC Curve
- AUC calculation

Results:

Recall improved:

```text
0.52 → 0.71
```

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
- Added feature importance analysis
- Improved project structure

Purpose:

Create a cleaner and more professional project structure.

---

## Key Results

Model performance:

Accuracy:

```text
78.75%
```

Recall after threshold tuning:

```text
71%
```

AUC Score:

```text
0.832
```

Important features identified:

- TotalCharges
- InternetService_Fiber optic
- StreamingMovies
- StreamingTV
- MultipleLines

---
## Model Comparison Results

Two machine learning models were tested and compared:

### Logistic Regression

Results:

- Accuracy: 78.75%
- Precision: 0.62
- Recall: 0.52
- F1-score: 0.56

Strengths:

- Better recall for churn prediction
- Simpler and easier to interpret
- Faster training time

Weaknesses:

- Assumes mostly linear relationships


### Random Forest

Results:

- Accuracy: 78.46%
- Precision: 0.63
- Recall: 0.47
- F1-score: 0.54

Strengths:

- Can capture more complex patterns
- Handles non-linear relationships

Weaknesses:

- Lower recall in this project
- More difficult to interpret


### Final Model Selection

Although both models achieved similar accuracy, Logistic Regression was selected as the final model because it achieved better recall and F1-score.

For churn prediction, recall is important because missing customers who may leave the service can be more costly than predicting extra churn cases.

---
## Future Improvements

Possible future work:

- Random Forest
- XGBoost
- Hyperparameter tuning
- Save trained model using joblib
- Build prediction system for new customers
- Create a Streamlit application

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

## How to Run the Project

Clone repository:

```bash
git clone <repository-link>
```

Install required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python src/train_model.py
```

---

## Final Learning Outcome

Through this project I learned:

- Data preprocessing
- Exploratory Data Analysis
- Logistic Regression
- Model evaluation
- Threshold tuning
- ROC and AUC interpretation
- Feature importance analysis
- GitHub project organization
- Writing cleaner and reusable code