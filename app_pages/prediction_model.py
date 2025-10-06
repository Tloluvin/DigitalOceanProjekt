import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import boto3
from dotenv import load_dotenv

# Załadowanie zmiennych środowiskowych
load_dotenv()

BUCKET_NAME = "dane-modul9"

def load_model_from_local():
    """Ładuje model z lokalnego katalogu"""
    try:
        # Znajdź plik modelu w katalogu models/
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
        
        # Pobierz listę plików z modelem
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix='models/')
        
        model_file = None
        for obj in response.get('Contents', []):
            if 'halfmarathon_model' in obj['Key'] and obj['Key'].endswith('.pkl'):
                model_file = obj['Key']
                break
        
        if not model_file:
            return None, None, None
        
        # Pobierz pliki tymczasowo
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
        # Przygotowanie danych wejściowych
        feature_order = model_info['features']
        input_df = pd.DataFrame([input_data])[feature_order]
        
        # Predykcja
        # Sprawdź czy model wymaga skalowania (Linear/Ridge)
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
    total_minutes = tempo_min_per_km * 21.0975  # Długość półmaratonu
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
    model_source = st.radio(
        "Źródło modelu:",
        ["Lokalny katalog", "DigitalOcean Spaces"],
        horizontal=True
    )
    
    # Ładowanie modelu
    with st.spinner("Ładowanie modelu..."):
        if model_source == "Lokalny katalog":
            model, scaler, model_info = load_model_from_local()
        else:
            model, scaler, model_info = load_model_from_digitalocean()
    
    if model is None or model_info is None:
        st.error("❌ Nie można załadować modelu. Upewnij się, że model został wytrenowany i zapisany.")
        st.info("""
        **Aby wytrenować model:**
        1. Uruchom notebook `train_model.ipynb`
        2. Model zostanie automatycznie zapisany lokalnie i na DigitalOcean
        3. Odśwież tę stronę
        """)
        return
    
    # Informacje o modelu
    st.success(f"✅ Model załadowany: **{model_info['model_name']}**")
    
    with st.expander("📊 Informacje o modelu"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE", f"{model_info['mae']:.4f} min/km")
        with col2:
            st.metric("RMSE", f"{model_info['rmse']:.4f} min/km")
        with col3:
            st.metric("R² Score", f"{model_info['r2']:.4f}")
        
        st.markdown(f"**Data trenowania:** {model_info['training_date']}")
        st.markdown(f"**Liczba próbek treningowych:** {model_info['training_samples']}")
        st.markdown(f"**Użyte feature'y:** {', '.join(model_info['features'])}")
    
    st.markdown("---")
    
    # Tabs: Manual Input vs Text Input (z LLM)
    input_tabs = st.tabs(["📝 Formularz", "💬 Opis tekstowy (LLM)"])
    
    # ========== TAB 1: MANUAL INPUT ==========
    with input_tabs[0]:
        st.subheader("Wprowadź dane zawodnika")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Płeć:", ["Mężczyzna", "Kobieta"])
            age = st.number_input("Wiek:", min_value=15, max_value=100, value=30, step=1)
            time_5km_min = st.number_input(
                "Czas na 5km (minuty):", 
                min_value=10.0, 
                max_value=60.0, 
                value=25.0, 
                step=0.5,
                help="Podaj swój ostatni czas na 5km"
            )
        
        with col2:
            has_team = st.selectbox("Czy należysz do drużyny?", ["Nie", "Tak"])
            pace_stability = st.slider(
                "Stabilność tempa (przewidywana):",
                min_value=0.0,
                max_value=0.2,
                value=0.05,
                step=0.01,
                help="0.0 = bardzo stabilne tempo, 0.2 = bardzo niestabilne tempo"
            )
        
        # Button predykcji
        if st.button("🚀 Przewiduj czas", type="primary", key="predict_manual"):
            # Przygotowanie danych wejściowych
            tempo_5km = calculate_5km_tempo(time_5km_min)
            
            input_data = {
                'Gender_Numeric': 1 if gender == "Mężczyzna" else 0,
                'Wiek': age,
                '5 km Tempo': tempo_5km,
                'Tempo Stabilność': pace_stability,
                'Has_Team': 1 if has_team == "Tak" else 0,
                'First_5km_Fast': 1 if tempo_5km < 5.0 else 0  # Arbitralnie: <5 min/km = szybki
            }
            
            # Predykcja
            predicted_tempo = predict_race_time(model, scaler, input_data, model_info)
            
            if predicted_tempo:
                finish_time = tempo_to_finish_time(predicted_tempo)
                
                st.markdown("---")
                st.subheader("🏁 Wynik predykcji")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Przewidywane tempo", f"{predicted_tempo:.2f} min/km")
                with col2:
                    st.metric("Przewidywany czas ukończenia", finish_time)
                with col3:
                    total_km = 21.0975
                    st.metric("Dystans", f"{total_km:.2f} km")
                
                # Interpretacja wyniku
                st.markdown("---")
                st.subheader("📊 Interpretacja")
                
                if predicted_tempo < 4.5:
                    st.success("🏆 Wynik profesjonalny! To świetny czas!")
                elif predicted_tempo < 5.5:
                    st.success("🥇 Bardzo dobry wynik! Jesteś w czołówce amatorów!")
                elif predicted_tempo < 6.5:
                    st.info("🏃 Dobry wynik! Solidne tempo dla amatora.")
                else:
                    st.warning("🚶 Spokojne tempo. Pamiętaj, że najważniejsze to ukończyć!")
                
                # Porównanie z danymi historycznymi
                st.markdown("---")
                st.subheader("📈 Porównanie z danymi historycznymi")
                
                # Tutaj możesz dodać porównanie z medianą, percentylami itp.
                st.info(f"""
                **Twoje przewidywane tempo:** {predicted_tempo:.2f} min/km
                
                **Dla porównania (dane historyczne):**
                - Mediana wszystkich uczestników: ~5.7 min/km
                - Top 10%: <4.5 min/km
                - Top 25%: <5.1 min/km
                """)
    
    # ========== TAB 2: TEXT INPUT WITH LLM ==========
    with input_tabs[1]:
        st.subheader("💬 Opisz się tekstem - użyjemy AI do ekstrakcji danych")
        st.info("""
        **Przykład:**
        "Cześć, mam na imię Jan, mam 32 lata, jestem mężczyzną i ostatnio 
        przebiegłem 5km w czasie 24 minuty. Należę do klubu biegowego."
        """)
        
        user_text = st.text_area(
            "Twój opis:",
            height=150,
            placeholder="Opisz swoje dane: wiek, płeć, czas na 5km..."
        )
        
        # LLM Integration
        if st.button("🤖 Wyślij do analizy AI", type="primary", key="predict_ai"):
            if not user_text.strip():
                st.warning("⚠️ Proszę wpisać opis!")
            else:
                # Sprawdzenie czy klucze API są skonfigurowane
                if not os.getenv("GOOGLE_API_KEY") or not os.getenv("LANGFUSE_PUBLIC_KEY"):
                    st.error("""
                    ❌ **Brak konfiguracji API**
                    
                    Aby użyć funkcji ekstrakcji z AI, należy skonfigurować:
                    1. `GOOGLE_API_KEY` - klucz do Google Gemini API (DARMOWE!)
                    2. `LANGFUSE_PUBLIC_KEY` i `LANGFUSE_SECRET_KEY` - klucze do Langfuse
                    
                    **Jak uzyskać Google API Key:**
                    1. Przejdź na: https://aistudio.google.com/app/apikey
                    2. Zaloguj się kontem Google
                    3. Kliknij "Create API Key"
                    4. Skopiuj klucz i dodaj do `.env`: `GOOGLE_API_KEY=your_key_here`
                    
                    **Darmowy limit Gemini:**
                    - 15 zapytań/minutę
                    - 1500 zapytań/dzień
                    - 1 milion tokenów/miesiąc
                    
                    Dodaj te klucze do pliku `.env` lub użyj formularza w zakładce "Formularz".
                    """)
                else:
                    try:
                        # Import modułu LLM
                        from utils.llm_integration import (
                            extract_runner_data_with_openai,
                            validate_extracted_data,
                            convert_to_model_input,
                            log_prediction_to_langfuse
                        )
                        
                        with st.spinner("🤖 Analizuję tekst za pomocą AI..."):
                            # Ekstrakcja danych
                            extraction_result = extract_runner_data_with_openai(user_text)
                        
                        if not extraction_result['success']:
                            st.error(f"❌ Błąd ekstrakcji: {extraction_result['error']}")
                            st.info("💡 Spróbuj przeformułować swój opis lub użyj formularza ręcznego.")
                        else:
                            extracted_data = extraction_result['data']
                            
                            # Wyświetlenie wyekstrahowanych danych
                            st.success("✅ Dane wyekstrahowane!")
                            
                            with st.expander("📋 Wyekstrahowane dane", expanded=True):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    gender_display = "Mężczyzna" if extracted_data.get('gender') == 'M' else "Kobieta" if extracted_data.get('gender') == 'K' else "Nie podano"
                                    st.metric("Płeć", gender_display)
                                
                                with col2:
                                    age_display = extracted_data.get('age', 'Nie podano')
                                    st.metric("Wiek", age_display)
                                
                                with col3:
                                    time_display = f"{extracted_data.get('time_5km_minutes', 'Nie podano')} min" if extracted_data.get('time_5km_minutes') else "Nie podano"
                                    st.metric("Czas 5km", time_display)
                                
                                if extracted_data.get('has_team') is not None:
                                    st.info(f"🏃‍♂️ Klub: {'Tak' if extracted_data['has_team'] else 'Nie'}")
                                
                                st.caption(f"🔢 Użyto {extraction_result.get('tokens_used', 'N/A')} tokenów")
                            
                            # Walidacja danych
                            validation = validate_extracted_data(extracted_data)
                            
                            if not validation['is_valid']:
                                st.warning(validation['message'])
                            else:
                                st.success(validation['message'])
                                
                                # Konwersja do formatu modelu
                                model_input = convert_to_model_input(extracted_data)
                                
                                # Dodanie domyślnej stabilności
                                model_input['Tempo Stabilność'] = 0.06  # Mediana
                                
                                # Predykcja
                                predicted_tempo = predict_race_time(model, scaler, model_input, model_info)
                                
                                if predicted_tempo:
                                    finish_time = tempo_to_finish_time(predicted_tempo)
                                    
                                    # Logowanie do Langfuse
                                    log_prediction_to_langfuse(
                                        user_text=user_text,
                                        extracted_data=extracted_data,
                                        prediction=predicted_tempo,
                                        model_name=model_info['model_name'],
                                        success=True
                                    )
                                    
                                    st.markdown("---")
                                    st.subheader("🏁 Wynik predykcji")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Przewidywane tempo", f"{predicted_tempo:.2f} min/km")
                                    with col2:
                                        st.metric("Przewidywany czas ukończenia", finish_time)
                                    with col3:
                                        total_km = 21.0975
                                        st.metric("Dystans", f"{total_km:.2f} km")
                                    
                                    # Interpretacja wyniku
                                    st.markdown("---")
                                    st.subheader("📊 Interpretacja")
                                    
                                    if predicted_tempo < 4.5:
                                        st.success("🏆 Wynik profesjonalny! To świetny czas!")
                                    elif predicted_tempo < 5.5:
                                        st.success("🥇 Bardzo dobry wynik! Jesteś w czołówce amatorów!")
                                    elif predicted_tempo < 6.5:
                                        st.info("🏃 Dobry wynik! Solidne tempo dla amatora.")
                                    else:
                                        st.warning("🚶 Spokojne tempo. Pamiętaj, że najważniejsze to ukończyć!")
                                    
                                    st.success("✅ Interakcja zapisana w Langfuse do analizy jakości AI")
                    
                    except ImportError:
                        st.error("""
                        ❌ **Moduł LLM nie jest dostępny**
                        
                        Upewnij się, że plik `utils/llm_integration.py` istnieje.
                        """)
                    except Exception as e:
                        st.error(f"❌ Nieoczekiwany błąd: {str(e)}")
                        st.info("💡 Użyj formularza ręcznego w zakładce 'Formularz'")
    
    st.markdown("---")
    
    # Dodatkowe informacje
    with st.expander("ℹ️ Jak działa model?"):
        st.markdown("""
        ### 🧠 Jak działa model predykcyjny?
        
        Model został wytrenowany na danych z Półmaratonu Wrocławskiego z lat 2023 i 2024.
        Używa algorytmu **Machine Learning** do przewidywania tempa na podstawie:
        
        **Wejściowe dane (features):**
        1. **Płeć** - statystycznie mężczyźni biegają szybciej
        2. **Wiek** - tempo zmienia się z wiekiem
        3. **Tempo na 5km** - najważniejszy wskaźnik przygotowania
        4. **Stabilność tempa** - jak równomiernie biegniesz
        5. **Przynależność do drużyny** - członkowie klubów często są lepiej przygotowani
        6. **Szybkość pierwszych 5km** - czy startujesz zbyt szybko
        
        **Wynik:**
        - Przewidywane tempo na całym dystansie (min/km)
        - Szacowany czas ukończenia półmaratonu
        
        **Dokładność modelu:**
        - Średni błąd (MAE): ~{model_info['mae']:.2f} min/km
        - To oznacza, że predykcja może się różnić o ±{model_info['mae']*21:.1f} minut od rzeczywistego czasu
        
        ### 📊 Na czym został wytrenowany?
        
        - **Liczba próbek:** {model_info['training_samples']:,} zawodników
        - **Lata:** 2023, 2024
        - **Algorytm:** {model_info['model_name']}
        
        ### ⚠️ Ograniczenia
        
        - Model nie uwzględnia warunków pogodowych
        - Nie uwzględnia kontuzji czy formy dnia
        - Zakłada podobny profil trasy jak w latach 2023-2024
        - Najlepiej sprawdza się dla typowych biegaczy amatorów (tempo 4-7 min/km)
        """)
    
    with st.expander("🎓 Jak poprawić swój wynik?"):
        st.markdown("""
        ### 💪 Wskazówki treningowe
        
        **1. Regularność treningów**
        - Trenuj 3-4 razy w tygodniu
        - Zróżnicowane dystanse i intensywność
        
        **2. Długie biegi**
        - Przynajmniej jeden długi bieg tygodniowo (15-18 km)
        - W spokojnym tempie, budującym wytrzymałość
        
        **3. Tempo runs**
        - Biegi w tempie docelowym lub nieco szybciej
        - Uczą organizm biegać efektywnie w tym tempie
        
        **4. Intervalowe**
        - Poprawa VO2max i szybkości
        - Krótkie, intensywne odcinki z regeneracją
        
        **5. Stabilność tempa**
        - Nie startuj zbyt szybko!
        - Pierwsze 5-10 km w tempie lub wolniej
        - Negatywny split = szybsza druga połowa
        
        **6. Regeneracja**
        - Sen 7-9 godzin
        - Odpowiednie odżywianie
        - Dni odpoczynku
        
        **7. Znajdź klub biegowy**
        - Motywacja grupy
        - Porady doświadczonych biegaczy
        - Dane pokazują, że członkowie klubów osiągają lepsze wyniki!
        """)