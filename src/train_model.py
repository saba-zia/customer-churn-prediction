import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from feature_importance import show_feature_importance
from evaluate_model import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data_preprocessing import preprocess_data


df = pd.read_csv(
    "data/raw/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)
evaluate_model(
    model,
    X_test_scaled,
    y_test
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

print("Random Forest training finished")
show_feature_importance(
    rf_model,
    X
)
print("\nRandom Forest Results:")
evaluate_model(
    rf_model,
    X_test_scaled,
    y_test
)
logistic_pred = model.predict(X_test_scaled)
random_forest_pred = rf_model.predict(X_test_scaled)

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, logistic_pred),
        accuracy_score(y_test, random_forest_pred)
    ],
    "Precision_Churn": [
        precision_score(y_test, logistic_pred),
        precision_score(y_test, random_forest_pred)
    ],
    "Recall_Churn": [
        recall_score(y_test, logistic_pred),
        recall_score(y_test, random_forest_pred)
    ],
    "F1_Churn": [
        f1_score(y_test, logistic_pred),
        f1_score(y_test, random_forest_pred)
    ]
})

print("\nModel Comparison:")
print(comparison)
print("Training finished successfully")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)