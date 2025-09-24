import streamlit as st
from app_pages import data_overview, eda_analysis, prediction_model
from utils.helper_functions import load_data

st.set_page_config(
    page_title="Halfmarathon Wrocław Analysis",  # tytuł zakładki
    page_icon="🏃‍♂️",  # ikona zakładki (możesz użyć emoji lub ścieżki do pliku .png)
    layout="wide"
)

wroclaw_2023_df, wroclaw_2024_df = load_data()

menu = {
    "Data Overview": lambda: data_overview.show(wroclaw_2023_df, wroclaw_2024_df),
    "EDA Analysis": lambda: eda_analysis.show(wroclaw_2023_df, wroclaw_2024_df),
    "Prediction Model": prediction_model.show
}

st.sidebar.title("Menu")
choice = st.sidebar.radio("Choose section:", list(menu.keys()))
menu[choice]()
