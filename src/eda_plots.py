import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_categorical(df: pd.DataFrame, col: str, target: str = "Survived") -> None:
    """
    Kategorik değişken ile hedef arasındaki ilişkiyi çubuk grafikle gösterir.
    """
    plt.figure(figsize=(6,4))
    sns.barplot(x=col, y=target, data=df, ci=None)
    plt.title(f"{col} vs {target} (mean survival rate)")
    plt.show()

def plot_numerical(df: pd.DataFrame, col: str, target: str = "Survived") -> None:
    """
    Sayısal değişkenin dağılımını hedefe göre kutu grafiğiyle gösterir.
    """
    plt.figure(figsize=(6,4))
    sns.boxplot(x=target, y=col, data=df)
    plt.title(f"{col} distribution by {target}")
    plt.show()
