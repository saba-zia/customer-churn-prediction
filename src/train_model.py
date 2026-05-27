import pandas as pd
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
show_feature_importance(
    model,
    X
)

print("Training finished successfully")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)