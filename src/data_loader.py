import pandas as pd
from .config import TRAIN_PATH, TEST_PATH
from .utils import audit_dataframe

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """train.csv ve test.csv dosyalarını okur."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def quick_audit(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Veri setleri için hızlı kalite kontrol raporu basar."""
    print("Train shape:", train.shape)
    print("Test  shape:", test.shape)

    print("\n--- Train head ---")
    print(train.head())

    print("\n--- Eksik değer özeti (train) ---")
    print(audit_dataframe(train, "train"))

    print("\n--- Eksik değer özeti (test) ---")
    print(audit_dataframe(test, "test"))

    if "Survived" in train.columns:
        target_counts = train["Survived"].value_counts().sort_index()
        print("\n--- Hedef dağılımı (Survived) ---")
        print(target_counts)
        print("Class ratio (1 rate):", target_counts.get(1, 0) / target_counts.sum())
