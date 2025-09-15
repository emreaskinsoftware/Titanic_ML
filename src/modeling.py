import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

def build_pipeline() -> Pipeline:
    """
    Titanic için Logistic Regression pipeline'ı oluşturur.
    """
    categorical = ["Sex", "Embarked", "Pclass", "Title", "IsAlone"]
    numerical   = ["Age", "Fare", "FamilySize"]

    # Kategorik: eksik -> en sık, one-hot encode
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Sayısal: eksik -> median, scale
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical),
            ("num", num_transformer, numerical)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    return model

def evaluate_model(train: pd.DataFrame) -> dict:

    """
    Stratified K-Fold ile Logistic Regression'ı değerlendirir.
    """
    X = train[["Sex", "Embarked", "Pclass", "Age", "Fare", "FamilySize", "IsAlone", "Title"]]
    y = train["Survived"]

    model = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

    return {
        "accuracy_mean": scores.mean(),
        "accuracy_std": scores.std(),
        "f1_mean": f1_scores.mean(),
        "f1_std": f1_scores.std()
    }
