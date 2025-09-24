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

st.set_page_config(page_title="Zadanie z modułu 9", page_icon="📊", layout="wide")

wroclaw_2023_df = pd.read_csv(f"s3://{BUCKET_NAME}/dane-zadanie_modul9/halfmarathon_wroclaw_2023__final.csv", sep=";")
wroclaw_2024_df = pd.read_csv(f"s3://{BUCKET_NAME}/dane-zadanie_modul9/halfmarathon_wroclaw_2024__final.csv", sep=";")

st.set_page_config(layout="wide")

# Sidebar menu
st.sidebar.title("Menu")
menu_selection = st.sidebar.radio(
    "Wybierz sekcję:",
    ("Przegląd danych", "Analiza EDA", "Model predykcji")
)

# Dynamiczny tytuł w zależności od sekcji
if menu_selection == "Przegląd danych":
    st.title("Przegląd danych półmaratonu")
    st.subheader("Dane za rok 2023")
    st.dataframe(wroclaw_2023_df)
    st.subheader("Dane za rok 2024")
    st.dataframe(wroclaw_2024_df)

elif menu_selection == "Analiza EDA":
    st.title("Eksploracyjna analiza danych (EDA)")
    st.write("Tu pojawi się szczegółowa analiza danych.")

elif menu_selection == "Model predykcji":
    st.title("Model przewidujący wyniki biegu")
    st.write("Tu pojawi się narzędzie przewidujące rezultaty na podstawie podanych danych zawodnika.")