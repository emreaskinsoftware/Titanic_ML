from pathlib import Path

# Proje kök dizinini otomatik bul
BASE_DIR = Path(__file__).resolve().parent.parent

# Data klasör yolları
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"
