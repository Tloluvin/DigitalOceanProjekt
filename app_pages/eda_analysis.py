import streamlit as st
import pandas as pd
from utils import eda_utils

def show_missing_data(missing: pd.Series):
    if missing.empty:
        st.write("No missing values detected.")
        return

    missing_df = missing.reset_index(drop=False)
    missing_df.columns = ["Column", "Missing Count"]
    st.table(missing_df)  # statyczna tabela bez indeksu

def show(wroclaw_2023_df, wroclaw_2024_df):
    st.title("Exploratory Data Analysis (EDA)")

    for df, year in [(wroclaw_2023_df, 2023), (wroclaw_2024_df, 2024)]:
        st.header(f"Year {year} - Basic statistics")
        st.write(eda_utils.basic_statistics(df))

        missing = eda_utils.analyze_missing_values(df)
        st.write(f"Missing values for year {year}:")
        show_missing_data(missing)
