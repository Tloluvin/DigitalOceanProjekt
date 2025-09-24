import streamlit as st

def show(wroclaw_2023_df, wroclaw_2024_df):
    st.title("Data Overview")
    st.subheader("Data 2023")
    st.dataframe(wroclaw_2023_df)
    st.subheader("Data 2024")
    st.dataframe(wroclaw_2024_df)
