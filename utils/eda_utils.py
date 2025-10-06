import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict

# Ustawienia stylu dla wykresów
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analiza brakujących wartości z procentami"""
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    return missing_df.reset_index(drop=True)


def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Podstawowe statystyki numeryczne"""
    return df.describe()


def convert_time_to_seconds(time_str: str) -> float:
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


def prepare_data_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Przygotowuje dane do analizy - konwersje typów"""
    df_clean = df.copy()
    
    # Konwersja czasów na sekundy
    time_columns = ['Czas', '5 km Czas', '10 km Czas', '15 km Czas', '20 km Czas']
    for col in time_columns:
        if col in df_clean.columns:
            df_clean[f'{col}_seconds'] = df_clean[col].apply(convert_time_to_seconds)
    
    # Obliczanie wieku z rocznika
    current_year = 2023 if df_clean['Rocznik'].median() < 2010 else 2024
    df_clean['Wiek'] = current_year - df_clean['Rocznik']
    df_clean['Wiek'] = df_clean['Wiek'].where((df_clean['Wiek'] > 0) & (df_clean['Wiek'] < 100), np.nan)
    
    # Filtrowanie tylko ukończonych biegów
    df_clean['Finished'] = df_clean['Miejsce'].notna()
    
    return df_clean


def analyze_gender_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Analiza rozkładu płci"""
    gender_stats = df.groupby('Płeć').agg({
        'Numer startowy': 'count',
        'Miejsce': 'count',
        'Tempo': 'mean'
    }).rename(columns={
        'Numer startowy': 'Registered',
        'Miejsce': 'Finished',
        'Tempo': 'Avg Tempo (min/km)'
    })
    
    gender_stats['Finish Rate %'] = (gender_stats['Finished'] / gender_stats['Registered'] * 100).round(2)
    gender_stats['Avg Tempo (min/km)'] = gender_stats['Avg Tempo (min/km)'].round(2)
    
    return gender_stats


def analyze_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Analiza grup wiekowych"""
    df_with_age = df[df['Wiek'].notna()].copy()
    
    age_bins = [0, 20, 30, 40, 50, 60, 100]
    age_labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
    df_with_age['Age_Group'] = pd.cut(df_with_age['Wiek'], bins=age_bins, labels=age_labels)
    
    age_stats = df_with_age.groupby('Age_Group').agg({
        'Numer startowy': 'count',
        'Tempo': 'mean',
        'Tempo Stabilność': 'mean'
    }).rename(columns={
        'Numer startowy': 'Count',
        'Tempo': 'Avg Tempo (min/km)',
        'Tempo Stabilność': 'Avg Stability'
    })
    
    age_stats['Avg Tempo (min/km)'] = age_stats['Avg Tempo (min/km)'].round(2)
    age_stats['Avg Stability'] = age_stats['Avg Stability'].round(4)
    
    return age_stats


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, int]:
    """Wykrywa outliery metodą IQR"""
    df_clean = df[df[column].notna()].copy()
    
    Q1 = df_clean[column].quantile(0.25)
    Q3 = df_clean[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_clean[(df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)]
    
    return outliers, len(outliers)


def compare_years_summary(df_2023: pd.DataFrame, df_2024: pd.DataFrame) -> pd.DataFrame:
    """Porównanie podstawowych statystyk między latami"""
    
    def get_year_stats(df, year):
        finished = df[df['Miejsce'].notna()]
        return {
            'Year': year,
            'Total Registered': len(df),
            'Total Finished': len(finished),
            'Finish Rate %': round(len(finished) / len(df) * 100, 2),
            'Avg Tempo (min/km)': round(finished['Tempo'].mean(), 2),
            'Avg Stability': round(finished['Tempo Stabilność'].mean(), 4),
            'Male %': round(len(df[df['Płeć'] == 'M']) / len(df[df['Płeć'].notna()]) * 100, 2),
            'Female %': round(len(df[df['Płeć'] == 'K']) / len(df[df['Płeć'].notna()]) * 100, 2)
        }
    
    stats_2023 = get_year_stats(df_2023, 2023)
    stats_2024 = get_year_stats(df_2024, 2024)
    
    comparison = pd.DataFrame([stats_2023, stats_2024])
    
    return comparison


def plot_time_distribution(df: pd.DataFrame, year: int) -> plt.Figure:
    """Wykres rozkładu czasów ukończenia"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_finished = df[df['Tempo'].notna()]
    
    # Histogram czasu
    axes[0].hist(df_finished['Tempo'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Tempo (min/km)', fontsize=12)
    axes[0].set_ylabel('Liczba uczestników', fontsize=12)
    axes[0].set_title(f'Rozkład tempa - {year}', fontsize=14, fontweight='bold')
    axes[0].axvline(df_finished['Tempo'].median(), color='red', linestyle='--', linewidth=2, label=f'Mediana: {df_finished["Tempo"].median():.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot tempa według płci
    df_gender = df_finished[df_finished['Płeć'].isin(['M', 'K'])]
    df_gender.boxplot(column='Tempo', by='Płeć', ax=axes[1])
    axes[1].set_xlabel('Płeć', fontsize=12)
    axes[1].set_ylabel('Tempo (min/km)', fontsize=12)
    axes[1].set_title(f'Tempo według płci - {year}', fontsize=14, fontweight='bold')
    axes[1].get_figure().suptitle('')
    
    plt.tight_layout()
    return fig


def plot_age_distribution(df: pd.DataFrame, year: int) -> plt.Figure:
    """Wykres rozkładu wieku"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_age = df[df['Wiek'].notna() & (df['Wiek'] > 0) & (df['Wiek'] < 100)]
    
    # Histogram wieku
    axes[0].hist(df_age['Wiek'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Wiek', fontsize=12)
    axes[0].set_ylabel('Liczba uczestników', fontsize=12)
    axes[0].set_title(f'Rozkład wieku uczestników - {year}', fontsize=14, fontweight='bold')
    axes[0].axvline(df_age['Wiek'].median(), color='red', linestyle='--', linewidth=2, label=f'Mediana: {df_age["Wiek"].median():.0f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Tempo vs Wiek
    df_tempo_age = df_age[df_age['Tempo'].notna()]
    axes[1].scatter(df_tempo_age['Wiek'], df_tempo_age['Tempo'], alpha=0.3, s=10, color='coral')
    axes[1].set_xlabel('Wiek', fontsize=12)
    axes[1].set_ylabel('Tempo (min/km)', fontsize=12)
    axes[1].set_title(f'Tempo vs Wiek - {year}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Dodanie linii trendu
    z = np.polyfit(df_tempo_age['Wiek'], df_tempo_age['Tempo'], 2)
    p = np.poly1d(z)
    axes[1].plot(sorted(df_tempo_age['Wiek']), p(sorted(df_tempo_age['Wiek'])), "r--", linewidth=2, label='Trend')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def plot_pace_stability(df: pd.DataFrame, year: int) -> plt.Figure:
    """Wykres stabilności tempa"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df_stab = df[df['Tempo Stabilność'].notna()]
    
    # Histogram stabilności
    axes[0].hist(df_stab['Tempo Stabilność'], bins=50, color='green', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Tempo Stabilność', fontsize=12)
    axes[0].set_ylabel('Liczba uczestników', fontsize=12)
    axes[0].set_title(f'Rozkład stabilności tempa - {year}', fontsize=14, fontweight='bold')
    axes[0].axvline(df_stab['Tempo Stabilność'].median(), color='red', linestyle='--', linewidth=2, label=f'Mediana: {df_stab["Tempo Stabilność"].median():.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Tempo vs Stabilność
    df_both = df[(df['Tempo'].notna()) & (df['Tempo Stabilność'].notna())]
    axes[1].scatter(df_both['Tempo'], df_both['Tempo Stabilność'], alpha=0.3, s=10, color='green')
    axes[1].set_xlabel('Tempo (min/km)', fontsize=12)
    axes[1].set_ylabel('Tempo Stabilność', fontsize=12)
    axes[1].set_title(f'Tempo vs Stabilność - {year}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_split_times(df: pd.DataFrame, year: int) -> plt.Figure:
    """Wykres czasów pośrednich"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    splits = ['5 km Tempo', '10 km Tempo', '15 km Tempo', '20 km Tempo', 'Tempo']
    split_labels = ['5km', '10km', '15km', '20km', 'Finish']
    
    df_splits = df[splits].dropna()
    
    # Box plot dla każdego splitu
    bp = ax.boxplot([df_splits[col] for col in splits], labels=split_labels, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Punkt pomiaru', fontsize=12)
    ax.set_ylabel('Tempo (min/km)', fontsize=12)
    ax.set_title(f'Tempo na poszczególnych odcinkach - {year}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig