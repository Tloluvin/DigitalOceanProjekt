import streamlit as st
import pandas as pd
from utils import eda_utils
from itables.streamlit import interactive_table

def show_missing_data(missing: pd.Series):
    if missing.empty:
        st.write("No missing values detected.")
        return
    
    # Jeśli missing ma indeks typu int (czyli nie zawiera nazw kolumn), odtwórz kolumnę 'Column'
    if missing.index.dtype == 'int64':
        missing_df = pd.DataFrame({
            "Missing Count": missing.values,
            "Column": missing.index.astype(str)
        })[["Column", "Missing Count"]]
    else:
        missing_df = missing.reset_index(drop=False)
        missing_df.columns = ["Column", "Missing Count"]
    
    interactive_table(missing_df)


def show(wroclaw_2023_df, wroclaw_2024_df):
    st.title("Exploratory Data Analysis (EDA)")

    for df, year in [(wroclaw_2023_df, 2023), (wroclaw_2024_df, 2024)]:
        st.header(f"Year {year} - Basic statistics")
        st.write(eda_utils.basic_statistics(df))  # indeks tutaj zostaje

        missing = eda_utils.analyze_missing_values(df)
        st.write(f"Missing values for year {year}:")
        show_missing_data(missing)  # a tutaj indeks jest resetowany i usuwany
