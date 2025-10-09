import os
import json
from typing import Dict
from dotenv import load_dotenv
import google.generativeai as genai
from langfuse import Langfuse, observe

# Załadowanie zmiennych środowiskowych
load_dotenv()

# Inicjalizacja klientów
# Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"))


# Langfuse (inicjalizacja klienta)
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)


def create_extraction_prompt(user_text: str) -> str:
    """Tworzy prompt do ekstrakcji danych z tekstu użytkownika"""
    prompt = f"""
Jesteś asystentem, który pomaga wyekstrahować dane z opisu użytkownika dotyczącego jego profilu biegacza.

OPIS UŻYTKOWNIKA:
{user_text}

ZADANIE:
Wyekstrahuj następujące informacje z tekstu:
1. Płeć (gender): "M" dla mężczyzny, "K" dla kobiety
2. Wiek (age): liczba całkowita
3. Czas na 5km w minutach (time_5km_minutes): liczba zmiennoprzecinkowa

ZASADY:
- Jeśli jakiejś informacji brak, ustaw wartość null
- Zwróć TYLKO poprawny JSON bez dodatkowego tekstu
- Format: {{"gender": "M/K/null", "age": number/null, "time_5km_minutes": number/null}}

PRZYKŁADY:

Input: "Cześć, mam 32 lata, jestem mężczyzną i ostatnio przebiegłem 5km w 24 minuty"
Output: {{"gender": "M", "age": 32, "time_5km_minutes": 24.0}}

Input: "Jestem kobietą, mam 28 lat, biegam 5km w około 30 minut"
Output: {{"gender": "K", "age": 28, "time_5km_minutes": 30.0}}

Input: "Mam 45 lat, mój ostatni czas na 5 km to 27 minut"
Output: {{"gender": null, "age": 45, "time_5km_minutes": 27.0}}

Teraz wyekstrahuj dane z podanego opisu użytkownika.
"""
    return prompt


@observe(as_type="generation")
def extract_runner_data_with_gemini(user_text: str) -> Dict:
    """
    Ekstrahuje dane biegacza z tekstu używając Google Gemini API
    Dane wejściowe i wyjściowe zostaną automatycznie zarejestrowane w Langfuse.
    """
    prompt = create_extraction_prompt(user_text)
    response = gemini_model.generate_content(prompt)
    raw_response = response.text.strip()

    # Czyszczenie potencjalnych bloków markdown
    if raw_response.startswith("```json"):
        raw_response = raw_response[7:]
    if raw_response.startswith("```"):
        raw_response = raw_response[3:]
    if raw_response.endswith("```"):
        raw_response = raw_response[:-3]

    try:
        extracted_data = json.loads(raw_response.strip())
    except json.JSONDecodeError as e:
        # Langfuse automatycznie zarejestruje wyjątek dzięki dekoratorowi @observe
        raise ValueError(f"Błąd parsowania JSON: {str(e)}")

    estimated_tokens = len(prompt.split()) + len(raw_response.split())

    return {
        "input": user_text,
        "prompt": prompt,
        "output": extracted_data,
        "metadata": {
            "model": "gemini-1.5-flash",
            "estimated_tokens": estimated_tokens,
            "provider": "google"
        }
    }


def validate_extracted_data(extracted_data: Dict) -> Dict:
    """Waliduje wyekstrahowane dane"""
    required_fields = {
        "gender": "Płeć",
        "age": "Wiek",
        "time_5km_minutes": "Czas na 5km"
    }

    missing_fields = []
    invalid_fields = []

    for field, display_name in required_fields.items():
        value = extracted_data.get(field)
        if value is None:
            missing_fields.append(display_name)
        elif field == "gender" and value not in ["M", "K"]:
            invalid_fields.append(f"{display_name} (musi być M lub K)")
        elif field == "age" and (not isinstance(value, (int, float)) or not (10 <= value <= 100)):
            invalid_fields.append(f"{display_name} (musi być liczbą 10-100)")
        elif field == "time_5km_minutes" and (not isinstance(value, (int, float)) or not (10 <= value <= 60)):
            invalid_fields.append(f"{display_name} (musi być liczbą 10-60 minut)")

    is_valid = not missing_fields and not invalid_fields

    return {
        "is_valid": is_valid,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "message": _create_validation_message(is_valid, missing_fields, invalid_fields)
    }


def _create_validation_message(is_valid: bool, missing_fields: list, invalid_fields: list) -> str:
    """Tworzy czytelną wiadomość walidacyjną"""
    if is_valid:
        return "✅ Wszystkie wymagane dane zostały poprawnie wyekstrahowane!"

    parts = []
    if missing_fields:
        parts.append(f"❌ **Brakujące dane:** {', '.join(missing_fields)}")
    if invalid_fields:
        parts.append(f"⚠️ **Nieprawidłowe dane:** {', '.join(invalid_fields)}")
    parts.append("\n💡 **Proszę podać brakujące informacje:**")

    if "Płeć" in missing_fields:
        parts.append("- Czy jesteś mężczyzną czy kobietą?")
    if "Wiek" in missing_fields:
        parts.append("- Ile masz lat?")
    if "Czas na 5km" in missing_fields:
        parts.append("- Jaki jest Twój ostatni czas na 5km?")

    return "\n".join(parts)


def convert_to_model_input(extracted_data: Dict) -> Dict:
    """Konwertuje dane do formatu wejściowego modelu"""
    tempo_5km = extracted_data['time_5km_minutes'] / 5.0
    return {
        'Gender_Numeric': 1 if extracted_data['gender'] == 'M' else 0,
        'Wiek': int(extracted_data['age']),
        '5 km Tempo': tempo_5km,
        'Tempo Stabilność': 0.06
    }


@observe(as_type="generation")
def log_prediction_to_langfuse(
    user_text: str,
    extracted_data: Dict,
    prediction: float,
    model_name: str,
    success: bool = True
):
    """
    Loguje interakcję (ekstrakcja + predykcja) do Langfuse
    Dzięki dekoratorowi @observe dane są automatycznie rejestrowane.
    """
    return {
        "input": user_text,
        "output": {
            "extracted_data": extracted_data,
            "predicted_tempo": prediction,
            "model": model_name
        },
        "metadata": {
            "success": success,
            "model_type": "ml_prediction"
        }
    }


def test_extraction(user_text: str):
    """Testuje ekstrakcję danych"""
    print(f"\n{'='*60}")
    print("TEST EKSTRAKCJI DANYCH (Google Gemini)")
    print(f"{'='*60}")
    print(f"\nInput: {user_text}")

    result = extract_runner_data_with_gemini(user_text)
    print(f"\nWynik: {json.dumps(result, indent=2, ensure_ascii=False)}")

    validation = validate_extracted_data(result["output"])
    print(f"\nWalidacja:\n{validation['message']}")

    if validation["is_valid"]:
        model_input = convert_to_model_input(result["output"])
        print(f"\nDane do modelu:\n{json.dumps(model_input, indent=2, ensure_ascii=False)}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    test_cases = [
        "Cześć, mam na imię Jan, mam 32 lata, jestem mężczyzną i ostatnio przebiegłem 5km w 24 minuty",
        "Jestem kobietą, mam 28 lat, biegam 5km w około 30 minut i należę do klubu biegowego",
        "Mam 45 lat, mój ostatni czas na 5 km to 27 minut"
    ]
    for text in test_cases:
        test_extraction(text)
