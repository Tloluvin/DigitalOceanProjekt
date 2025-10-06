import streamlit as st
from utils import eda_utils
import pandas as pd

def show(wroclaw_2023_df, wroclaw_2024_df):
    st.title("🔍 Exploratory Data Analysis (EDA)")
    st.markdown("---")
    
    # Przygotowanie danych
    df_2023_prep = eda_utils.prepare_data_for_analysis(wroclaw_2023_df)
    df_2024_prep = eda_utils.prepare_data_for_analysis(wroclaw_2024_df)
    
    # Menu główne EDA
    eda_section = st.selectbox(
        "Wybierz sekcję analizy:",
        ["📊 Overview & Comparison", "🔢 Data Quality", "📈 Distributions", "👥 Demographics", "⚡ Performance Analysis", "🎯 Outliers Detection"]
    )
    
    # ========== OVERVIEW & COMPARISON ==========
    if eda_section == "📊 Overview & Comparison":
        st.header("📊 Overview & Comparison")
        st.markdown("Porównanie podstawowych statystyk między rokiem 2023 i 2024")
        
        comparison = eda_utils.compare_years_summary(df_2023_prep, df_2024_prep)
        
        # Wyświetlenie w ładnej tabeli
        st.dataframe(comparison.set_index('Year'), use_container_width=True)
        
        # Metryki w kolumnach
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Wzrost uczestników 2023→2024",
                f"+{comparison.loc[1, 'Total Registered'] - comparison.loc[0, 'Total Registered']}",
                f"{((comparison.loc[1, 'Total Registered'] / comparison.loc[0, 'Total Registered'] - 1) * 100):.1f}%"
            )
        
        with col2:
            st.metric(
                "Zmiana % ukończenia",
                f"{comparison.loc[1, 'Finish Rate %']}%",
                f"{(comparison.loc[1, 'Finish Rate %'] - comparison.loc[0, 'Finish Rate %']):.2f}%"
            )
        
        with col3:
            tempo_change = comparison.loc[1, 'Avg Tempo (min/km)'] - comparison.loc[0, 'Avg Tempo (min/km)']
            st.metric(
                "Zmiana średniego tempa",
                f"{comparison.loc[1, 'Avg Tempo (min/km)']} min/km",
                f"{tempo_change:+.2f} min/km"
            )
        
        with col4:
            stab_change = comparison.loc[1, 'Avg Stability'] - comparison.loc[0, 'Avg Stability']
            st.metric(
                "Zmiana stabilności",
                f"{comparison.loc[1, 'Avg Stability']:.4f}",
                f"{stab_change:+.4f}"
            )
    
    # ========== DATA QUALITY ==========
    elif eda_section == "🔢 Data Quality":
        st.header("🔢 Data Quality Analysis")
        
        tabs = st.tabs(["2023", "2024"])
        
        with tabs[0]:
            st.subheader("📅 Rok 2023 - Jakość danych")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Podstawowe informacje:**")
                st.write(f"- **Liczba uczestników:** {len(df_2023_prep)}")
                st.write(f"- **Ukończonych biegów:** {df_2023_prep['Finished'].sum()}")
                st.write(f"- **Wskaźnik ukończenia:** {(df_2023_prep['Finished'].sum() / len(df_2023_prep) * 100):.2f}%")
                st.write(f"- **Liczba kolumn:** {len(df_2023_prep.columns)}")
            
            with col2:
                st.markdown("**Typy danych:**")
                type_counts = df_2023_prep.dtypes.value_counts()
                for dtype, count in type_counts.items():
                    st.write(f"- **{dtype}:** {count} kolumn")
            
            st.markdown("---")
            st.markdown("**Brakujące wartości:**")
            missing_2023 = eda_utils.analyze_missing_values(wroclaw_2023_df)
            
            if missing_2023.empty:
                st.success("✅ Brak brakujących wartości!")
            else:
                st.dataframe(missing_2023, use_container_width=True, hide_index=True)
                
                # Wizualizacja brakujących wartości
                st.markdown("**Top 10 kolumn z brakującymi wartościami:**")
                top_missing = missing_2023.head(10)
                st.bar_chart(top_missing.set_index('Column')['Missing %'])
        
        with tabs[1]:
            st.subheader("📅 Rok 2024 - Jakość danych")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Podstawowe informacje:**")
                st.write(f"- **Liczba uczestników:** {len(df_2024_prep)}")
                st.write(f"- **Ukończonych biegów:** {df_2024_prep['Finished'].sum()}")
                st.write(f"- **Wskaźnik ukończenia:** {(df_2024_prep['Finished'].sum() / len(df_2024_prep) * 100):.2f}%")
                st.write(f"- **Liczba kolumn:** {len(df_2024_prep.columns)}")
            
            with col2:
                st.markdown("**Typy danych:**")
                type_counts = df_2024_prep.dtypes.value_counts()
                for dtype, count in type_counts.items():
                    st.write(f"- **{dtype}:** {count} kolumn")
            
            st.markdown("---")
            st.markdown("**Brakujące wartości:**")
            missing_2024 = eda_utils.analyze_missing_values(wroclaw_2024_df)
            
            if missing_2024.empty:
                st.success("✅ Brak brakujących wartości!")
            else:
                st.dataframe(missing_2024, use_container_width=True, hide_index=True)
                
                # Wizualizacja brakujących wartości
                st.markdown("**Top 10 kolumn z brakującymi wartościami:**")
                top_missing = missing_2024.head(10)
                st.bar_chart(top_missing.set_index('Column')['Missing %'])
    
    # ========== DISTRIBUTIONS ==========
    elif eda_section == "📈 Distributions":
        st.header("📈 Rozkłady zmiennych")
        
        year_choice = st.radio("Wybierz rok:", ["2023", "2024"], horizontal=True)
        
        df = df_2023_prep if year_choice == "2023" else df_2024_prep
        year = 2023 if year_choice == "2023" else 2024
        
        # Wybór typu rozkładu
        dist_type = st.selectbox(
            "Wybierz typ rozkładu:",
            ["Tempo/Czas ukończenia", "Wiek uczestników", "Stabilność tempa", "Czasy na odcinkach"]
        )
        
        if dist_type == "Tempo/Czas ukończenia":
            st.subheader(f"Rozkład tempa - {year}")
            fig = eda_utils.plot_time_distribution(df, year)
            st.pyplot(fig)
            
            # Statystyki
            col1, col2, col3, col4 = st.columns(4)
            tempo_stats = df[df['Tempo'].notna()]['Tempo']
            with col1:
                st.metric("Średnie tempo", f"{tempo_stats.mean():.2f} min/km")
            with col2:
                st.metric("Mediana", f"{tempo_stats.median():.2f} min/km")
            with col3:
                st.metric("Najszybsze", f"{tempo_stats.min():.2f} min/km")
            with col4:
                st.metric("Najwolniejsze", f"{tempo_stats.max():.2f} min/km")
        
        elif dist_type == "Wiek uczestników":
            st.subheader(f"Rozkład wieku - {year}")
            fig = eda_utils.plot_age_distribution(df, year)
            st.pyplot(fig)
            
            # Statystyki
            col1, col2, col3, col4 = st.columns(4)
            age_stats = df[df['Wiek'].notna()]['Wiek']
            with col1:
                st.metric("Średni wiek", f"{age_stats.mean():.1f} lat")
            with col2:
                st.metric("Mediana", f"{age_stats.median():.0f} lat")
            with col3:
                st.metric("Najmłodszy", f"{age_stats.min():.0f} lat")
            with col4:
                st.metric("Najstarszy", f"{age_stats.max():.0f} lat")
        
        elif dist_type == "Stabilność tempa":
            st.subheader(f"Stabilność tempa - {year}")
            fig = eda_utils.plot_pace_stability(df, year)
            st.pyplot(fig)
            
            st.info("💡 **Tempo Stabilność** - niższa wartość oznacza bardziej równomierne tempo przez cały bieg")
            
            # Statystyki
            col1, col2, col3 = st.columns(3)
            stab_stats = df[df['Tempo Stabilność'].notna()]['Tempo Stabilność']
            with col1:
                st.metric("Średnia stabilność", f"{stab_stats.mean():.4f}")
            with col2:
                st.metric("Mediana", f"{stab_stats.median():.4f}")
            with col3:
                st.metric("Odchylenie std", f"{stab_stats.std():.4f}")
        
        elif dist_type == "Czasy na odcinkach":
            st.subheader(f"Tempo na poszczególnych odcinkach - {year}")
            fig = eda_utils.plot_split_times(df, year)
            st.pyplot(fig)
            
            st.info("💡 Wykres pokazuje jak tempo zmienia się na kolejnych odcinkach 5km, 10km, 15km, 20km i na mecie")
    
    # ========== DEMOGRAPHICS ==========
    elif eda_section == "👥 Demographics":
        st.header("👥 Analiza demograficzna")
        
        year_choice = st.radio("Wybierz rok:", ["2023", "2024", "Porównanie"], horizontal=True)
        
        if year_choice in ["2023", "2024"]:
            df = df_2023_prep if year_choice == "2023" else df_2024_prep
            year = 2023 if year_choice == "2023" else 2024
            
            # Analiza płci
            st.subheader(f"Rozkład według płci - {year}")
            gender_stats = eda_utils.analyze_gender_distribution(df)
            st.dataframe(gender_stats, use_container_width=True)
            
            # Wykres kołowy
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Udział płci w zawodach:**")
                gender_counts = df['Płeć'].value_counts()
                st.bar_chart(gender_counts)
            
            with col2:
                st.markdown("**Wskaźnik ukończenia według płci:**")
                finish_rate = gender_stats['Finish Rate %']
                st.bar_chart(finish_rate)
            
            st.markdown("---")
            
            # Analiza grup wiekowych
            st.subheader(f"Rozkład według grup wiekowych - {year}")
            age_stats = eda_utils.analyze_age_groups(df)
            st.dataframe(age_stats, use_container_width=True)
            
            # Wykresy
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Liczba uczestników w grupach wiekowych:**")
                st.bar_chart(age_stats['Count'])
            
            with col2:
                st.markdown("**Średnie tempo w grupach wiekowych:**")
                st.bar_chart(age_stats['Avg Tempo (min/km)'])
        
        else:  # Porównanie
            st.subheader("Porównanie demografii 2023 vs 2024")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**2023 - Płeć:**")
                gender_2023 = eda_utils.analyze_gender_distribution(df_2023_prep)
                st.dataframe(gender_2023, use_container_width=True)
            
            with col2:
                st.markdown("**2024 - Płeć:**")
                gender_2024 = eda_utils.analyze_gender_distribution(df_2024_prep)
                st.dataframe(gender_2024, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**2023 - Grupy wiekowe:**")
                age_2023 = eda_utils.analyze_age_groups(df_2023_prep)
                st.dataframe(age_2023, use_container_width=True)
            
            with col2:
                st.markdown("**2024 - Grupy wiekowe:**")
                age_2024 = eda_utils.analyze_age_groups(df_2024_prep)
                st.dataframe(age_2024, use_container_width=True)
    
    # ========== PERFORMANCE ANALYSIS ==========
    elif eda_section == "⚡ Performance Analysis":
        st.header("⚡ Analiza wydajności")
        
        year_choice = st.radio("Wybierz rok:", ["2023", "2024"], horizontal=True)
        
        df = df_2023_prep if year_choice == "2023" else df_2024_prep
        year = 2023 if year_choice == "2023" else 2024
        
        df_finished = df[df['Finished'] == True].copy()
        
        st.subheader("Statystyki czasów na poszczególnych odcinkach")
        
        # Statystyki dla każdego odcinka
        split_stats = df_finished[['5 km Tempo', '10 km Tempo', '15 km Tempo', '20 km Tempo', 'Tempo']].describe()
        st.dataframe(split_stats, use_container_width=True)
        
        st.markdown("---")
        
        # Top 10 najszybszych
        st.subheader("🏆 Top 10 najszybszych uczestników")
        
        top_10 = df_finished.nsmallest(10, 'Tempo')[['Imię', 'Nazwisko', 'Płeć', 'Wiek', 'Czas', 'Tempo', 'Tempo Stabilność']]
        top_10 = top_10.reset_index(drop=True)
        top_10.index = top_10.index + 1
        st.dataframe(top_10, use_container_width=True)
        
        st.markdown("---")
        
        # Analiza korelacji między wiekiem a tempem
        st.subheader("📊 Korelacja: Wiek vs Tempo")
        
        df_corr = df_finished[(df_finished['Wiek'].notna()) & (df_finished['Tempo'].notna())]
        
        if len(df_corr) > 0:
            correlation = df_corr['Wiek'].corr(df_corr['Tempo'])
            st.metric("Współczynnik korelacji", f"{correlation:.4f}")
            
            if abs(correlation) < 0.3:
                st.info("📌 Słaba korelacja")
            elif abs(correlation) < 0.7:
                st.warning("📌 Umiarkowana korelacja")
            else:
                st.error("📌 Silna korelacja")
        
        st.markdown("---")
        
        # Analiza stabilności
        st.subheader("📈 Analiza stabilności tempa")
        
        col1, col2, col3 = st.columns(3)
        
        df_stab = df_finished[df_finished['Tempo Stabilność'].notna()]
        
        # Kategorie stabilności
        very_stable = len(df_stab[df_stab['Tempo Stabilność'] < 0.05])
        stable = len(df_stab[(df_stab['Tempo Stabilność'] >= 0.05) & (df_stab['Tempo Stabilność'] < 0.1)])
        unstable = len(df_stab[df_stab['Tempo Stabilność'] >= 0.1])
        
        with col1:
            st.metric("Bardzo stabilni", very_stable, f"{(very_stable/len(df_stab)*100):.1f}%")
        with col2:
            st.metric("Stabilni", stable, f"{(stable/len(df_stab)*100):.1f}%")
        with col3:
            st.metric("Niestabilni", unstable, f"{(unstable/len(df_stab)*100):.1f}%")
    
    # ========== OUTLIERS DETECTION ==========
    elif eda_section == "🎯 Outliers Detection":
        st.header("🎯 Wykrywanie outlierów")
        
        year_choice = st.radio("Wybierz rok:", ["2023", "2024"], horizontal=True)
        
        df = df_2023_prep if year_choice == "2023" else df_2024_prep
        year = 2023 if year_choice == "2023" else 2024
        
        st.markdown("Wykrywanie outlierów metodą **IQR (Interquartile Range)**")
        st.info("💡 Outliery to wartości odstające, które mogą wskazywać na błędy w danych lub wyjątkowe przypadki")
        
        # Wybór zmiennej do analizy
        variable = st.selectbox(
            "Wybierz zmienną do analizy:",
            ["Tempo", "Tempo Stabilność", "Wiek", "5 km Tempo", "10 km Tempo", "15 km Tempo", "20 km Tempo"]
        )
        
        if variable in df.columns:
            outliers, count = eda_utils.detect_outliers_iqr(df, variable)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Liczba outlierów", count)
            with col2:
                total = df[variable].notna().sum()
                st.metric("% outlierów", f"{(count/total*100):.2f}%")
            with col3:
                st.metric("Całkowita liczba obserwacji", total)
            
            st.markdown("---")
            
            if count > 0:
                st.subheader(f"Outliery dla zmiennej: {variable}")
                
                # Wyświetl top 20 outlierów
                outliers_display = outliers.nsmallest(20, variable) if variable == 'Tempo' else outliers.nlargest(20, variable)
                outliers_display = outliers_display[['Imię', 'Nazwisko', 'Płeć', 'Wiek', variable, 'Czas']]
                
                st.dataframe(outliers_display.reset_index(drop=True), use_container_width=True)
                
                st.markdown("---")
                
                # Statystyki outlierów
                st.subheader("Statystyki outlierów")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Wszystkie dane:**")
                    all_stats = df[variable].describe()
                    st.write(all_stats)
                
                with col2:
                    st.markdown("**Tylko outliery:**")
                    outlier_stats = outliers[variable].describe()
                    st.write(outlier_stats)
                
            else:
                st.success("✅ Nie wykryto outlierów dla tej zmiennej!")
        
        else:
            st.error(f"❌ Zmienna '{variable}' nie istnieje w danych")
    
    st.markdown("---")
    st.markdown("### 📝 Podsumowanie EDA")
    st.info("""
    **Co dalej?**
    
    1. ✅ Przeanalizowaliśmy jakość danych i zidentyfikowaliśmy braki
    2. ✅ Zbadaliśmy rozkłady kluczowych zmiennych
    3. ✅ Przeanalizowaliśmy demografię uczestników
    4. ✅ Oceniliśmy wydajność i stabilność tempa
    5. ✅ Wykryliśmy potencjalne outliery
    
    **Następne kroki:**
    - Oczyszczenie danych (usunięcie/imputacja braków, obsługa outlierów)
    - Feature engineering (tworzenie nowych zmiennych)
    - Budowa modelu predykcyjnego
    """)