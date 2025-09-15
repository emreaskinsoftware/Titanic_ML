# src/tuning.py
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
from scipy.stats import uniform, randint


# ---- Ortak sütunlar ----
CATEGORICAL = ["Sex", "Embarked", "Pclass", "Title", "IsAlone"]
NUMERICAL   = ["Age", "Fare", "FamilySize"]
FEATURES    = CATEGORICAL + NUMERICAL


def build_preprocessor() -> ColumnTransformer:
    """Kategori/sayısal ön işlem adımlarını döndürür."""
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer([
        ("cat", cat, CATEGORICAL),
        ("num", num, NUMERICAL)
    ])


def build_xgb_pipeline() -> Pipeline:
    """XGBoost + preprocessor pipeline’ı (skor/aramalarda kullanılır)."""
    pre = build_preprocessor()
    xgb = XGBClassifier(
        # modern ve hızlı ağaç yapısı
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1  # Windows çoklu-proses sorunları için güvenli
    )
    return Pipeline([
        ("preprocessor", pre),
        ("clf", xgb)
    ])


def xgb_random_search(train: pd.DataFrame, n_iter: int = 60) -> Dict[str, Any]:
    """XGBoost için RandomizedSearchCV (hızlı & etkili)."""
    X = train[FEATURES]
    y = train["Survived"]

    pipe = build_xgb_pipeline()

    # Geniş aralıklar: az denemede iyi sonuç arar
    param_dist = {
        "clf__n_estimators": randint(250, 900),
        "clf__learning_rate": uniform(0.01, 0.09),   # 0.01–0.10
        "clf__max_depth": randint(2, 6),             # 2–5
        "clf__subsample": uniform(0.7, 0.3),         # 0.7–1.0
        "clf__colsample_bytree": uniform(0.7, 0.3),  # 0.7–1.0
        "clf__min_child_weight": randint(1, 6),
        "clf__gamma": uniform(0.0, 0.2)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=1,    # güvenli
        verbose=2
    )
    rs.fit(X, y)
    return {
        "best_params": rs.best_params_,
        "best_score_f1_cv": rs.best_score_,
        "best_estimator": rs.best_estimator_
    }


# (İstersen) Eski GridSearch sürümü de dursun:
def xgb_grid_search(train: pd.DataFrame) -> Dict[str, Any]:
    X = train[FEATURES]
    y = train["Survived"]

    pipe = build_xgb_pipeline()
    param_grid = {
        "clf__n_estimators": [300, 500, 800],
        "clf__learning_rate": [0.05, 0.03, 0.02],
        "clf__max_depth": [3, 4, 5],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
        "clf__min_child_weight": [1, 3],
        "clf__gamma": [0.0, 0.1],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        verbose=1
    )
    gs.fit(X, y)
    return {
        "best_params": gs.best_params_,
        "best_score_f1_cv": gs.best_score_,
        "best_estimator": gs.best_estimator_
    }
