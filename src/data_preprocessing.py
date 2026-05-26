import pandas as pd


def preprocess_data(df):
    """
    Clean and prepare the raw customer churn dataset for machine learning.
    """

    df_ml = df.copy()

    df_ml["TotalCharges"] = pd.to_numeric(
        df_ml["TotalCharges"],
        errors="coerce"
    )

    df_ml = df_ml.dropna()

    df_ml = df_ml.drop(
        "customerID",
        axis=1
    )

    df_ml["Churn"] = df_ml["Churn"].map(
        {
            "No": 0,
            "Yes": 1
        }
    )

    df_ml = pd.get_dummies(
        df_ml,
        drop_first=True
    )

    X = df_ml.drop(
        "Churn",
        axis=1
    )

    y = df_ml["Churn"]

    return X, y