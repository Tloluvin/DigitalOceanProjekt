# 🏃‍♂️ Halfmarathon Wrocław Analysis App

Aplikacja Streamlit do analizy i przewidywania wyników Półmaratonu Wrocławskiego dla lat 2023 i 2024.

## 📋 Opis projektu

Projekt składa się z trzech głównych modułów:
1. **Data Overview** - Przeglądanie surowych danych z zawodów
2. **EDA Analysis** - Szczegółowa analiza eksploracyjna danych
3. **Prediction Model** - Model predykcyjny czasu ukończenia biegu

## 🗂️ Struktura projektu

```
/
├── app_pages/
│   ├── data_overview.py       # Strona z przeglądaniem danych
│   ├── eda_analysis.py         # Rozbudowana analiza EDA
│   └── prediction_model.py     # Model predykcyjny (w budowie)
├── utils/
│   ├── eda_utils.py           # Funkcje pomocnicze do EDA i wizualizacji
│   ├── helper_functions.py    # Ładowanie danych z DigitalOcean Spaces
│   └── data_preprocessing.py  # Czyszczenie i przygotowanie danych
├── .env                        # Zmienne środowiskowe (AWS/DigitalOcean credentials)
├── .gitignore                 # Pliki ignorowane przez Git
├── app.py                     # Główny plik aplikacji
├── README.md                  # Ten plik
└── requirements.txt           # Zależności Python
```

## 🚀 Instalacja

### 1. Klonowanie repozytorium
```bash
git clone <repository-url>
cd halfmarathon-wroclaw-analysis
```

### 2. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 3. Konfiguracja zmiennych środowiskowych

Utwórz plik `.env` w głównym katalogu projektu:

```env
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_ENDPOINT_URL_S3=https://your-endpoint.digitaloceanspaces.com
```

### 4. Uruchomienie aplikacji
```bash
streamlit run app.py
```

## 📊 Funkcjonalności EDA

### 1. Overview & Comparison
- Porównanie podstawowych statystyk między rokiem 2023 i 2024
- Metryki: liczba uczestników, wskaźnik ukończenia, średnie tempo, stabilność

### 2. Data Quality
- Analiza brakujących wartości
- Typy danych i podstawowe statystyki
- Wizualizacja braków danych

### 3. Distributions
- **Tempo/Czas ukończenia**: Rozkład tempa, porównanie płci
- **Wiek uczestników**: Rozkład wieku, korelacja wiek-tempo
- **Stabilność tempa**: Analiza równomierności biegu
- **Czasy na odcinkach**: Tempo na 5km, 10km, 15km, 20km

### 4. Demographics
- Rozkład według płci (liczba, wskaźnik ukończenia, średnie tempo)
- Analiza grup wiekowych (<20, 20-29, 30-39, 40-49, 50-59, 60+)
- Porównanie demografii między latami

### 5. Performance Analysis
- Statystyki czasów na poszczególnych odcinkach
- Top 10 najszybszych uczestników
- Korelacja wiek vs tempo
- Analiza stabilności (bardzo stabilni, stabilni, niestabilni)

### 6. Outliers Detection
- Wykrywanie wartości odstających metodą IQR
- Analiza outlierów dla: tempo, stabilność, wiek, czasy na odcinkach
- Statystyki porównawcze (wszystkie dane vs outliery)

## 🛠️ Technologie

- **Python 3.9+**
- **Streamlit** - Framework webowy
- **Pandas** - Przetwarzanie danych
- **NumPy** - Obliczenia numeryczne
- **Matplotlib & Seaborn** - Wizualizacje
- **Scikit-learn** - Machine Learning (Random Forest, Gradient Boosting)
- **boto3/s3fs** - Integracja z DigitalOcean Spaces
- **Google Gemini** - Ekstrakcja danych z tekstu (DARMOWE!) ✨
- **Langfuse** - Monitorowanie LLM

## 📈 Roadmap

### ✅ Zrealizowane
- [x] Struktura projektu i setup
- [x] Ładowanie danych z DigitalOcean Spaces
- [x] Podstawowa prezentacja danych (Data Overview)
- [x] Rozbudowana analiza EDA z wizualizacjami
- [x] Moduł czyszczenia danych

### 🔄 W trakcie
- [ ] Notebook do trenowania modelu
- [ ] Pipeline czyszczenia danych
- [ ] Feature engineering
- [ ] Model predykcyjny (Random Forest / XGBoost)

### 📋 Planowane
- [ ] Integracja z OpenAI (ekstrakcja danych z tekstu użytkownika)
- [ ] Integracja z Langfuse (monitoring skuteczności LLM)
- [ ] Deployment na DigitalOcean App Platform
- [ ] API endpoint do predykcji
- [ ] Zapisywanie modelu do DigitalOcean Spaces

## 📝 Czyszczenie danych

Moduł `data_preprocessing.py` wykonuje:

1. **Filtrowanie**: Usunięcie osób, które nie ukończyły biegu (DNF/DNS)
2. **Konwersje**: Przekształcenie czasów na sekundy
3. **Obliczenia**: Wiek z rocznika
4. **Czyszczenie**: Usunięcie nieprawidłowych wartości wieku (<10, >100)
5. **Filtrowanie płci**: Tylko M i K
6. **Outliery**: Usunięcie wartości odstających (metoda IQR)
7. **Feature engineering**: Tworzenie nowych zmiennych:
   - `Tempo_Drop` - różnica tempa między połowami
   - `Negative_Split` - czy tempo rosło
   - `Age_Category_Numeric` - numeryczna kategoria wiekowa
   - `Gender_Numeric` - binarna płeć (0=K, 1=M)
   - `Has_Team` - czy uczestnik należy do drużyny
   - `First_5km_Fast` - czy pierwsze 5km było szybkie
   - `Stability_Category` - kategoryczna stabilność

## 🎯 Model predykcyjny (planowany)

Model będzie przewidywał **tempo ukończenia biegu** na podstawie:

### Input features:
- Płeć (M/K)
- Wiek
- Czas na 5km (jako referencja przygotowania)

### Target:
- Tempo na mecie (min/km)

### Metryki ewaluacji:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

## 🤖 Integracja z LLM (planowana)

Użytkownik wprowadzi dane w formie tekstowej:
```
"Cześć, mam na imię Jan, mam 32 lata, jestem mężczyzną 
i ostatnio przebiegłem 5km w czasie 24 minuty"
```

System:
1. Użyje OpenAI do ekstrakcji danych (wiek, płeć, czas 5km)
2. Przetworzy dane przez model predykcyjny
3. Zwróci przewidywany czas ukończenia półmaratonu
4. Loguje interakcję w Langfuse do monitorowania