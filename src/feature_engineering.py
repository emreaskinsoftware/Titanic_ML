import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Titanic için ek özellikler üretir:
    - FamilySize
    - IsAlone
    - Title (Name kolonundan)
    """
    df = df.copy()

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # IsAlone
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Title extraction
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")  # "Mr", "Mrs", "Miss", "Master", vs.
    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare", 
        "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
        "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare"
    })

    return df
