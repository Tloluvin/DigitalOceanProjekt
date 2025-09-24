import pandas as pd

def analyze_missing_values(df: pd.DataFrame) -> pd.Series:
    missing = df.isna().sum()
    return missing[missing > 0]

def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()
