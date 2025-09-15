import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Ek kütüphaneler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def evaluate_models(train: pd.DataFrame) -> dict:
    X = train[["Sex", "Embarked", "Pclass", "Age", "Fare", "FamilySize", "IsAlone", "Title"]]
    y = train["Survived"]

    categorical = ["Sex", "Embarked", "Pclass", "Title", "IsAlone"]
    numerical   = ["Age", "Fare", "FamilySize"]

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("cat", cat_transformer, categorical),
        ("num", num_transformer, numerical)
    ])

    models = {
        "LogisticRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", __import__("sklearn.linear_model").linear_model.LogisticRegression(max_iter=1000))
        ]),
        "RandomForest": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        "GradientBoosting": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False
            ))
        ]),
        "LightGBM": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", LGBMClassifier(
                n_estimators=800,        # biraz artırdık
                learning_rate=0.03,      # daha küçük lr
                num_leaves=31,           # default ama net belirtelim
                min_child_samples=10,    # varsayılan 20 -> biraz esnet
                min_split_gain=0.0,      # pozitif kazanç eşiğini sıfırla
                subsample=0.9,           # hafif randomization
                colsample_bytree=0.9,
                max_depth=-1,
                random_state=42,
                n_jobs=-1,
                verbosity=-1             # warningleri kapat
            ))
        ])
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        f1 = cross_val_score(model, X, y, cv=cv, scoring="f1").mean()
        results[name] = {"accuracy": acc, "f1": f1}

    return results
