import streamlit as st
import json
from pathlib import Path
from PIL import Image

# -------------------------------------------------------
# 🔧 Pomocnicze funkcje
# -------------------------------------------------------

def load_plots_manifest():
    """Wczytuje manifest z informacjami o zapisanych wykresach"""
    manifest_path = Path('data/plots_manifest.json')

    if not manifest_path.exists():
        return None

    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_plot_path(relative_path: str) -> Path:
    """
    Normalizuje ścieżkę do pliku wykresu:
    - zamienia backslashe na slashe
    - dokleja katalog 'data' jako prefix
    """
    clean_path = Path("data") / Path(relative_path.replace("\\", "/"))
    return clean_path


# -------------------------------------------------------
# 🖼️ Strona główna sekcji wyników
# -------------------------------------------------------

def show():
    st.title("📊 Wyniki Trenowania Modelu")
    st.markdown("Wizualizacje i analiza z procesu trenowania modelu predykcyjnego")
    st.markdown("---")

    manifest = load_plots_manifest()

    if manifest is None or not manifest.get('plots'):
        st.warning("⚠️ Brak zapisanych wykresów. Uruchom najpierw train_model.ipynb")
        st.info("""
        Aby wygenerować wykresy:
        1. Uruchom notebook `train_model.ipynb`
        2. Wykresy zostaną automatycznie zapisane w katalogu `data/training_plots/`
        3. Wróć na tę stronę - wykresy pojawią się tutaj
        """)
        return

    st.success(f"✅ Znaleziono {len(manifest['plots'])} wykresów")
    st.caption(f"Wygenerowane: {manifest['created']}")
    st.markdown("---")

    plots = manifest['plots']

    # -------------------------------------------------------
    # 1. Analiza danych wejściowych
    # -------------------------------------------------------
    st.header("📈 1. Analiza danych wejściowych")
    exploratory = next((p for p in plots if 'exploratory' in p['filename']), None)

    if exploratory:
        st.subheader(exploratory['title'])
        st.markdown(f"*{exploratory['description']}*")

        plot_path = resolve_plot_path(exploratory['path'])
        if plot_path.exists():
            image = Image.open(plot_path)
            st.image(image, use_container_width=True)
            st.caption(f"📁 {exploratory['filename']}")
        else:
            st.error(f"Nie znaleziono pliku: {plot_path}")

    st.markdown("---")

    # -------------------------------------------------------
    # 2. Porównanie wydajności modeli
    # -------------------------------------------------------
    st.header("⚖️ 2. Porównanie wydajności modeli")
    comparison = next((p for p in plots if 'comparison' in p['filename']), None)

    if comparison:
        st.subheader(comparison['title'])
        st.markdown(f"*{comparison['description']}*")

        col1, col2 = st.columns([3, 1])

        with col1:
            plot_path = resolve_plot_path(comparison['path'])
            if plot_path.exists():
                image = Image.open(plot_path)
                st.image(image, use_container_width=True)
            else:
                st.error(f"Nie znaleziono pliku: {plot_path}")

        with col2:
            st.info("""
            **Metryki:**

            **MAE** - Mean Absolute Error  
            (przeciętna absolutna różnica)

            **RMSE** - Root Mean Squared Error  
            (pierwiastek średniego kwadratu błędu)

            **R²** - Współczynnik determinacji  
            (procent wariancji wyjaśnionej przez model)
            """)

        st.caption(f"📁 {comparison['filename']}")

    st.markdown("---")

    # -------------------------------------------------------
    # 3. Ważność cech (Feature Importance)
    # -------------------------------------------------------
    st.header("🎯 3. Ważność cech (Feature Importance)")
    importance = next((p for p in plots if 'importance' in p['filename']), None)

    if importance:
        st.subheader(importance['title'])
        st.markdown(f"*{importance['description']}*")

        col1, col2 = st.columns([3, 1])

        with col1:
            plot_path = resolve_plot_path(importance['path'])
            if plot_path.exists():
                image = Image.open(plot_path)
                st.image(image, use_container_width=True)
            else:
                st.error(f"Nie znaleziono pliku: {plot_path}")

        with col2:
            st.success("""
            **Kluczowe zmienne:**

            1️⃣ **5 km Tempo**  
               ~87.7% wpływu

            2️⃣ **Tempo Stabilność**  
               ~11.9% wpływu

            3️⃣ **Wiek**  
               ~0.3% wpływu

            4️⃣ **Płeć**  
               ~0.0% wpływu
            """)

        st.caption(f"📁 {importance['filename']}")

    st.markdown("---")

    # -------------------------------------------------------
    # 4. Analiza dokładności predykcji
    # -------------------------------------------------------
    st.header("🔮 4. Analiza dokładności predykcji")
    predictions = next((p for p in plots if 'predictions' in p['filename']), None)

    if predictions:
        st.subheader(predictions['title'])
        st.markdown(f"*{predictions['description']}*")

        plot_path = resolve_plot_path(predictions['path'])
        if plot_path.exists():
            image = Image.open(plot_path)
            st.image(image, use_container_width=True)
        else:
            st.error(f"Nie znaleziono pliku: {plot_path}")

        st.info("""
        **Interpretacja wykresu:**

        - **Lewy panel**: Punkty blisko linii czerwonej oznaczają dokładne predykcje  
        - **Prawy panel**: Histogram błędów skupiony wokół zera wskazuje na systematyczną dokładność
        """)

        st.caption(f"📁 {predictions['filename']}")

    st.markdown("---")

    # -------------------------------------------------------
    # 📋 Podsumowanie
    # -------------------------------------------------------
    st.header("📋 Podsumowanie i wnioski")

    st.success("""
    Model predykcyjny został pomyślnie wytrenowany i oceniony.  
    Analiza wykresów potwierdza jego stabilność i dobrą jakość predykcji.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🔍 Najważniejsze obserwacje:
        - Wysoka zgodność predykcji z wartościami rzeczywistymi  
        - Dominujący wpływ **tempa na 5 km** na wynik końcowy  
        - Niewielki wpływ wieku i płci
        """)

    with col2:
        st.markdown("""
        ### 💡 Rekomendacje:
        - Skup się na poprawie **stabilności tempa** w treningach  
        - Można uprościć model — część cech ma znikomy wpływ  
        - Model można wykorzystać do personalizowania planów biegowych
        """)

    st.markdown("---")

    st.info("""
    **Jak interpretować te wyniki?**

    Wizualizacje pozwalają szybko zrozumieć, które cechy decydują o czasie biegu
    oraz jak dokładne są predykcje modelu.  
    Wyniki mogą być użyte w raportach, prezentacjach lub do dalszej optymalizacji modelu.
    """)
