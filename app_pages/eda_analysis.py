import streamlit as st
from utils import eda_utils
from itables.streamlit import interactive_table


def show_missing_data(missing):
    if missing.empty:
        st.write("No missing values detected.")
        return

    missing_df = missing.reset_index(drop=False)
    missing_df.columns = ["Column", "Missing Count"]
    interactive_table(missing_df)


def show(wroclaw_2023_df, wroclaw_2024_df):
    st.title("Exploratory Data Analysis (EDA)")

    tabs = st.tabs(["Year 2023", "Year 2024"])

    with tabs[0]:
        st.header("Year 2023 - Full analysis")
        st.subheader("Basic statistics")
        st.write(eda_utils.basic_statistics(wroclaw_2023_df))

        missing_2023 = eda_utils.analyze_missing_values(wroclaw_2023_df)
        st.subheader("Missing values")
        show_missing_data(missing_2023)

    with tabs[1]:
        st.header("Year 2024 - Full analysis")
        st.subheader("Basic statistics")
        st.write(eda_utils.basic_statistics(wroclaw_2024_df))

        missing_2024 = eda_utils.analyze_missing_values(wroclaw_2024_df)
        st.subheader("Missing values")
        show_missing_data(missing_2024)
