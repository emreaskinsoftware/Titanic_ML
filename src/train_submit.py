# src/train_submit.py
import pandas as pd
from pathlib import Path
from typing import Optional
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from .data_loader import load_data
from .feature_engineering import add_features
from .tuning import build_preprocessor  # aynı preprocessor'u yeniden kullan

FEATURES = ["Sex", "Embarked", "Pclass", "Age", "Fare", "FamilySize", "IsAlone", "Title"]

def train_full_and_predict(best_estimator: Optional[Pipeline] = None,
                           out_path: Path = Path("submission.csv")) -> Path:
    train, test = load_data()
    train = add_features(train)
    test  = add_features(test)

    X_train = train[FEATURES]
    y_train = train["Survived"]
    X_test  = test[FEATURES]

    if best_estimator is None:
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("clf", XGBClassifier(
                tree_method="hist",
                n_estimators=800,
                learning_rate=0.02,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=1.0,
                min_child_weight=1,
                gamma=0.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            ))
        ])
    else:
        pipe = best_estimator

    pipe.fit(X_train, y_train)
    # threshold=0.5 ile sınıf tahmini
    preds = (pipe.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": preds
    })
    submission.to_csv(out_path, index=False)
    return out_path
