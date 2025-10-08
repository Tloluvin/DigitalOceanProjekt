import pandas as pd
import numpy as np
from typing import Tuple

def clean_data_for_modeling(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Kompleksowe czyszczenie danych do modelowania
    
    Args:
        df: DataFrame z surowymi danymi
        year: Rok zawodów (2023 lub 2024)
    
    Returns:
        Oczyszczony DataFrame
    """
    df_clean = df.copy()
    
    # 1. Usunięcie osób, które nie ukończyły biegu
    print(f"Przed filtrowaniem: {len(df_clean)} wierszy")
    df_clean = df_clean[df_clean['Miejsce'].notna()].copy()
    print(f"Po usunięciu DNF/DNS: {len(df_clean)} wierszy")
    
    # 2. Konwersja czasów na sekundy
    time_columns = ['Czas', '5 km Czas', '10 km Czas', '15 km Czas', '20 km Czas']
    for col in time_columns:
        if col in df_clean.columns:
            df_clean[f'{col}_seconds'] = df_clean[col].apply(_convert_time_to_seconds)
    
    # 3. Obliczanie wieku
    df_clean['Wiek'] = year - df_clean['Rocznik']
    
    # 4. Czyszczenie wieku - usunięcie nieprawidłowych wartości
    df_clean['Wiek'] = df_clean['Wiek'].where(
        (df_clean['Wiek'] > 10) & (df_clean['Wiek'] < 100), 
        np.nan
    )
    
    # 5. Czyszczenie płci - tylko M i K
    df_clean = df_clean[df_clean['Płeć'].isin(['M', 'K'])].copy()
    
    # 6. Usunięcie outlierów w tempie (metoda IQR)
    df_clean = _remove_outliers_iqr(df_clean, 'Tempo', factor=3.0)
    
    # 7. Usunięcie wierszy z brakującymi wartościami w kluczowych kolumnach
    key_columns = ['Tempo', 'Płeć', 'Wiek', '5 km Tempo']
    df_clean = df_clean.dropna(subset=key_columns)
    
    # 8. Tworzenie nowych feature'ów
    df_clean = _create_features(df_clean)
    
    print(f"Po oczyszczeniu: {len(df_clean)} wierszy")
    print(f"Usunięto: {len(df) - len(df_clean)} wierszy ({((len(df) - len(df_clean))/len(df)*100):.2f}%)")
    
    return df_clean


def _convert_time_to_seconds(time_str: str) -> float:
    """Konwertuje czas w formacie HH:MM:SS lub MM:SS na sekundy"""
    try:
        if pd.isna(time_str):
            return np.nan
        parts = str(time_str).split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
        else:
            return np.nan
    except:
        return np.nan


def _remove_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Usuwa outliery z DataFrame używając metody IQR
    
    Args:
        df: DataFrame
        column: Nazwa kolumny
        factor: Współczynnik IQR (domyślnie 1.5, dla bardziej agresywnego czyszczenia: 3.0)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    
    removed = len(df) - len(df_filtered)
    print(f"Usunięto {removed} outlierów z kolumny '{column}'")
    
    return df_filtered


def _create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tworzy nowe feature'y do modelowania"""
    
    # 1. Różnica tempa między pierwszą a drugą połową
    if '10 km Tempo' in df.columns and '20 km Tempo' in df.columns:
        df['Tempo_Drop'] = df['20 km Tempo'] - df['10 km Tempo']
    
    # 2. Czy tempo rosło (pozytywne) czy malało (negatywne)
    if 'Tempo_Drop' in df.columns:
        df['Negative_Split'] = (df['Tempo_Drop'] < 0).astype(int)
    
    # 3. Kategorie wiekowe jako liczby
    age_category_mapping = {
        'M16': 1, 'K16': 1,
        'M20': 2, 'K20': 2,
        'M30': 3, 'K30': 3,
        'M40': 4, 'K40': 4,
        'M50': 5, 'K50': 5,
        'M60': 6, 'K60': 6,
        'M70': 7, 'K70': 7
    }
    df['Age_Category_Numeric'] = df['Kategoria wiekowa'].map(age_category_mapping)
    
    # 4. Grupa wiekowa (co 10 lat)
    df['Age_Group'] = pd.cut(df['Wiek'], bins=[0, 20, 30, 40, 50, 60, 100], labels=['<20', '20-29', '30-39', '40-49', '50-59', '60+'])
    
    # 5. Binary encoding płci (0 = K, 1 = M)
    df['Gender_Numeric'] = (df['Płeć'] == 'M').astype(int)
    
    # USUNIĘTE FEATURE'Y (nieużywane w modelu):
    # - Has_Team (redundantny, 0.04% importance)
    # - First_5km_Fast (korelacja -0.786 z 5 km Tempo)
    
    # 6. Stabilność kategoryczna (opcjonalna, do wizualizacji)
    if 'Tempo Stabilność' in df.columns:
        df['Stability_Category'] = pd.cut(
            df['Tempo Stabilność'], 
            bins=[0, 0.05, 0.1, 1.0], 
            labels=['Bardzo_stabilny', 'Stabilny', 'Niestabilny']
        )
    
    return df


def prepare_features_for_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Przygotowuje DataFrame z feature'ami do modelowania
    
    Returns:
        Tuple: (DataFrame z feature'ami, lista nazw feature'ów)
    """
    
    # Lista feature'ów do wykorzystania w modelu
    feature_columns = [
        'Gender_Numeric',      # Płeć (0=K, 1=M)
        'Wiek',               # Wiek
        '5 km Tempo',         # Tempo na pierwszych 5km
        'Tempo Stabilność',   # Stabilność tempa
        'Age_Category_Numeric', # Kategoria wiekowa jako liczba
        'Has_Team',           # Czy należy do drużyny
        'First_5km_Fast'      # Czy pierwsze 5km było szybkie
    ]
    
    # Sprawdzenie czy wszystkie kolumny istnieją
    available_features = [col for col in feature_columns if col in df.columns]
    
    print(f"Dostępne feature'y: {available_features}")
    
    # Usunięcie wierszy z NaN w wybranych feature'ach
    df_features = df[available_features + ['Tempo']].dropna()
    
    print(f"Liczba próbek po usunięciu NaN: {len(df_features)}")
    
    return df_features, available_features


def merge_years_data(df_2023: pd.DataFrame, df_2024: pd.DataFrame) -> pd.DataFrame:
    """
    Łączy dane z obu lat, dodając kolumnę 'Year'
    """
    df_2023_clean = clean_data_for_modeling(df_2023, 2023)
    df_2024_clean = clean_data_for_modeling(df_2024, 2024)
    
    df_2023_clean['Year'] = 2023
    df_2024_clean['Year'] = 2024
    
    df_combined = pd.concat([df_2023_clean, df_2024_clean], ignore_index=True)
    
    print(f"\nPołączone dane:")
    print(f"- Rok 2023: {len(df_2023_clean)} wierszy")
    print(f"- Rok 2024: {len(df_2024_clean)} wierszy")
    print(f"- Łącznie: {len(df_combined)} wierszy")
    
    return df_combined


def get_data_summary(df: pd.DataFrame) -> dict:
    """Zwraca podsumowanie oczyszczonych danych"""
    summary = {
        'total_records': len(df),
        'male_count': len(df[df['Płeć'] == 'M']),
        'female_count': len(df[df['Płeć'] == 'K']),
        'avg_age': df['Wiek'].mean(),
        'avg_tempo': df['Tempo'].mean(),
        'avg_stability': df['Tempo Stabilność'].mean(),
        'missing_values': df.isna().sum().sum()
    }
    
    return summary