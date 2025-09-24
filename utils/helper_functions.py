from dotenv import load_dotenv
import boto3
import pandas as pd
import streamlit as st

BUCKET_NAME = "dane-modul9"
load_dotenv()

s3 = boto3.client(
    "s3",
)

@st.cache_data
def load_data():
    # Wczytuje dane CSV tylko raz, potem wynik jest buforowany przez Streamlit cache
    wroclaw_2023_df = pd.read_csv(f"s3://{BUCKET_NAME}/dane-zadanie_modul9/halfmarathon_wroclaw_2023__final.csv", sep=";")
    wroclaw_2024_df = pd.read_csv(f"s3://{BUCKET_NAME}/dane-zadanie_modul9/halfmarathon_wroclaw_2024__final.csv", sep=";")
    return wroclaw_2023_df, wroclaw_2024_df
