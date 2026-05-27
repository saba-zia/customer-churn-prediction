import pandas as pd


def show_feature_importance(model, X):

    importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.coef_[0]
        }
    )

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(
        importance.head(10)
    )