# Titanic - Machine Learning from Disaster 🚢

Bu proje, Kaggle üzerindeki klasik **Titanic** yarışmasını temel alır.  
Amaç: Yolcuların hayatta kalıp kalmadığını tahmin eden bir makine öğrenmesi modeli geliştirmek.

---

## 🔑 Özellikler
- **Modüler proje yapısı** (`src/` klasörü altında reusable kodlar)
- **EDA**: Eksik değerler, dağılımlar, görseller
- **Feature Engineering**: `FamilySize`, `IsAlone`, `Title` vb.
- **Pipeline**: Scikit-learn `Pipeline` + `ColumnTransformer` (sızıntısız veri işleme)
- **Model Karşılaştırma**: Logistic Regression, RandomForest, GradientBoosting, XGBoost, LightGBM
- **Hyperparameter Tuning**: `RandomizedSearchCV` ile en iyi XGBoost parametreleri
- **Submission**: Kaggle’a yüklenebilir `submission.csv`

---

## 📂 Proje Yapısı

