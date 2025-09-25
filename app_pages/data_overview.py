import streamlit as st
from itables.streamlit import interactive_table

def show(wroclaw_2023_df, wroclaw_2024_df):
    st.title("Data Overview")

    st.subheader("Data 2023")
    interactive_table(wroclaw_2023_df)

    st.subheader("Data 2024")
    interactive_table(wroclaw_2024_df)
