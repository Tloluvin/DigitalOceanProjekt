import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import boto3
from dotenv import load_dotenv
from utils.llm_integration import (
    extract_runner_data_with_gemini,
    validate_extracted_data,
    convert_to_model_input,
    log_prediction_to_langfuse
)

# Załadowanie zmiennych środowiskowych
load_dotenv()

BUCKET_NAME = "dane-modul9"

def load_model_from_local():
    """Ładuje model z lokalnego katalogu"""
    try:
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl') and 'model' in f]
        if not model_files:
            return None, None, None

        model_path = f'models/{model_files[0]}'
        scaler_path = 'models/scaler.pkl'
        info_path = 'models/model_info.json'

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        with open(info_path, 'r') as f:
            model_info = json.load(f)

        return model, scaler, model_info
    except Exception as e:
        st.error(f"Błąd ładowania modelu lokalnie: {e}")
        return None, None, None

def load_model_from_digitalocean():
    """Ładuje model z DigitalOcean Spaces"""
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3")
        )

        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix='models/')
        model_file = None
        for obj in response.get('Contents', []):
            if 'halfmarathon_model' in obj['Key'] and obj['Key'].endswith('.pkl'):
                model_file = obj['Key']
                break

        if not model_file:
            return None, None, None

        os.makedirs('temp_models', exist_ok=True)
        s3_client.download_file(BUCKET_NAME, model_file, 'temp_models/model.pkl')
        s3_client.download_file(BUCKET_NAME, 'models/scaler.pkl', 'temp_models/scaler.pkl')
        s3_client.download_file(BUCKET_NAME, 'models/model_info.json', 'temp_models/model_info.json')

        model = joblib.load('temp_models/model.pkl')
        scaler = joblib.load('temp_models/scaler.pkl')

        with open('temp_models/model_info.json', 'r') as f:
            model_info = json.load(f)

        return model, scaler, model_info
    except Exception as e:
        st.error(f"Błąd ładowania modelu z DigitalOcean: {e}")
        return None, None, None

def predict_race_time(model, scaler, input_data, model_info):
    """Wykonuje predykcję czasu biegu"""
    try:
        feature_order = model_info['features']
        input_df = pd.DataFrame([input_data])[feature_order]

        if model_info['model_name'] in ['Linear Regression', 'Ridge Regression'] and scaler:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
        else:
            prediction = model.predict(input_df)[0]

        return prediction
    except Exception as e:
        st.error(f"Błąd predykcji: {e}")
        return None

def tempo_to_finish_time(tempo_min_per_km):
    """Konwertuje tempo (min/km) na czas ukończenia półmaratonu"""
    total_minutes = tempo_min_per_km * 21.0975
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    seconds = int((total_minutes % 1) * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def calculate_5km_tempo(time_5km_minutes):
    """Oblicza tempo na 5km w min/km"""
    return time_5km_minutes / 5.0

def show():
    st.title("🎯 Model Predykcyjny")
    st.markdown("Przewidywanie czasu ukończenia Półmaratonu Wrocławskiego")
    st.markdown("---")

    # Wybór źródła modelu
    model_source = st.radio("Źródło modelu:", ["Lokalny katalog", "DigitalOcean Spaces"], horizontal=True)

    with st.spinner("Ładowanie modelu..."):
        if model_source == "Lokalny katalog":
            model, scaler, model_info = load_model_from_local()
        else:
            model, scaler, model_info = load_model_from_digitalocean()

    if model is None or model_info is None:
        st.error("❌ Nie można załadować modelu. Upewnij się, że model został wytrenowany i zapisany.")
        return

    st.success(f"✅ Model załadowany: **{model_info['model_name']}**")

    # Tab: Manual Input
    input_tabs = st.tabs(["📝 Formularz", "💬 Opis tekstowy (LLM)"])
    with input_tabs[0]:
        st.subheader("Wprowadź dane zawodnika")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Płeć:", ["Mężczyzna", "Kobieta"])
            age = st.number_input("Wiek:", 15, 100, 30)
            time_5km_min = st.number_input("Czas na 5km (minuty):", 10.0, 60.0, 25.0, 0.5)
        with col2:
            pace_stability = st.slider("Stabilność tempa:", 0.0, 0.2, 0.06, 0.01)

        if st.button("🚀 Przewiduj czas", type="primary", key="predict_manual"):
            tempo_5km = calculate_5km_tempo(time_5km_min)
            input_data = {
                'Gender_Numeric': 1 if gender == "Mężczyzna" else 0,
                'Wiek': age,
                '5 km Tempo': tempo_5km,
                'Tempo Stabilność': pace_stability
            }
            predicted_tempo = predict_race_time(model, scaler, input_data, model_info)
            if predicted_tempo:
                finish_time = tempo_to_finish_time(predicted_tempo)
                col1, col2, col3 = st.columns(3)
                col1.metric("Przewidywane tempo", f"{predicted_tempo:.2f} min/km")
                col2.metric("Przewidywany czas ukończenia", finish_time)
                col3.metric("Dystans", f"{21.0975:.2f} km")

    # Tab: LLM Input
    with input_tabs[1]:
        st.subheader("💬 Opisz się tekstem - użyjemy AI")
        user_text = st.text_area("Twój opis:", height=150, placeholder="Opisz swoje dane: wiek, płeć, czas na 5km...")
        if st.button("🤖 Wyślij do analizy AI", key="predict_ai"):
            if not user_text.strip():
                st.warning("⚠️ Proszę wpisać opis!")
            else:
                if not os.getenv("GOOGLE_API_KEY") or not os.getenv("LANGFUSE_PUBLIC_KEY"):
                    st.error("❌ Brak konfiguracji API")
                else:
                    try:
                        with st.spinner("🤖 Analizuję tekst za pomocą Google Gemini..."):
                            extraction_result = extract_runner_data_with_gemini(user_text)
                        extracted_data = extraction_result.get('output')
                        if not extracted_data:
                            st.error("❌ Błąd ekstrakcji danych")
                        else:
                            st.success("✅ Dane wyekstrahowane!")
                            validation = validate_extracted_data(extracted_data)
                            if not validation['is_valid']:
                                st.warning(validation['message'])
                            else:
                                st.success(validation['message'])
                                model_input = convert_to_model_input(extracted_data)
                                model_input['Tempo Stabilność'] = 0.06
                                predicted_tempo = predict_race_time(model, scaler, model_input, model_info)
                                if predicted_tempo:
                                    finish_time = tempo_to_finish_time(predicted_tempo)
                                    log_prediction_to_langfuse(
                                        user_text=user_text,
                                        extracted_data=extracted_data,
                                        prediction=predicted_tempo,
                                        model_name=model_info['model_name'],
                                        success=True
                                    )
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Przewidywane tempo", f"{predicted_tempo:.2f} min/km")
                                    col2.metric("Przewidywany czas ukończenia", finish_time)
                                    col3.metric("Dystans", f"{21.0975:.2f} km")
                    except Exception as e:
                        st.error(f"❌ Nieoczekiwany błąd: {e}")
