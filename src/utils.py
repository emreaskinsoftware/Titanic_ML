import pandas as pd

def audit_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Bir DataFrame için hızlı özet çıkarır:
    - sütun tipi
    - benzersiz değer sayısı
    - eksik değer sayısı ve oranı
    """
    total = len(df)
    summary = []
    for col in df.columns:
        missing = df[col].isna().sum()
        summary.append({
            "column": col,
            "dtype": df[col].dtype.name,
            "n_unique": df[col].nunique(dropna=True),
            "missing": int(missing),
            "missing_rate": float(missing / total) if total else 0.0,
        })
    return pd.DataFrame(summary).sort_values(by=["missing", "column"], 
                                             ascending=[False, True], 
                                             ignore_index=True)
