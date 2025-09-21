import boto3
import pandas as pd
import pathlib
import streamlit as st
from dotenv import load_dotenv

BUCKET_NAME = "dane-modul9"

load_dotenv()

s3 = boto3.client(
    "s3",
)

wroclaw_2023_df = pd.read_csv(f"s3://{BUCKET_NAME}/dane-zadanie_modul9/halfmarathon_wroclaw_2023__final.csv", sep=";")
wroclaw_2024_df = pd.read_csv(f"s3://{BUCKET_NAME}/dane-zadanie_modul9/halfmarathon_wroclaw_2024__final.csv", sep=";")

print ("Podstawowe informacje o ramce danych")
wroclaw_2023_df.info()
wroclaw_2024_df.info()

print ("Zliczenie braków w każdej kolumnie")
wroclaw_2023_df.isna().sum()
wroclaw_2024_df.isna().sum()

print ("Typy i liczba braków w jednej tabeli")
pd.DataFrame({
    "dtype": wroclaw_2023_df.dtypes,
    "missing": wroclaw_2023_df.isna().sum()
})

pd.DataFrame({
    "dtype": wroclaw_2024_df.dtypes,
    "missing": wroclaw_2024_df.isna().sum()
})

print("Podgląd wartości w kolumnach czasowych")

wroclaw_2023_df[["5 km Czas", "10 km Czas", "15 km Czas", "20 km Czas", "Czas"]].head(10)

wroclaw_2024_df[["5 km Czas", "10 km Czas", "15 km Czas", "20 km Czas", "Czas"]].head(10)

print("Sprawdzenie unikalnych wartości (formatów)")
wroclaw_2023_df["Czas"].dropna().unique()[:20]
wroclaw_2024_df["Czas"].dropna().unique()[:20]

print("Test konwersji na timedelta")
pd.to_timedelta(wroclaw_2023_df["Czas"].dropna().head(10))
pd.to_timedelta(wroclaw_2023_df["Czas"].dropna().head(10))



st.set_page_config(page_title="Zadanie z modułu 9", page_icon="📊", layout="wide")
st.title("Zadanie z modułu 9")
