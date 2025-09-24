# Halfmarathon Wroclaw Analysis App

This project is a Streamlit web application for analyzing and comparing the results of the Wroclaw Halfmarathon for years 2023 and 2024. The app provides:

- Data overview of race results
- Exploratory Data Analysis (EDA) including data quality checks and statistics
- Prediction model to forecast race results based on participant input (planned)

## Project Structure

The project is organized to facilitate modular development and easier maintenance:
/
├─ /app_pages
│ ├─ data_overview.py # Data overview page code
│ ├─ eda_analysis.py # Exploratory Data Analysis page code
│ └─ prediction_model.py # Prediction model page code
├─ /utils
│ ├─ eda_utils.py #
│ └─ helper_functions.py # Helper functions for data loading, preprocessing etc.
├─ .env # Environment variables and secrets
├─ .gitignore # Git ignore file
├─ app.py # Main app file managing navigation and page rendering
├─ README.md # This file
└─ requirements.txt # Python dependencies