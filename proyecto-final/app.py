import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Datos de Spotify",
    page_icon="🎵",
    layout="wide"
)

# Título principal
st.title("🎵 Análisis Completo de Datos de Spotify")

# Sidebar 
st.sidebar.title("Panel de Control")

# Carga de archivo
file = st.sidebar.file_uploader("Carga tu archivo CSV aquí", type=['csv'])
st.sidebar.caption("Sube tu archivo de datos de Spotify para comenzar el análisis")

if file is None:
    st.warning("⚠️ Inserta tu archivo CSV para visualizar los datos de análisis")
    st.info("**Formato esperado del archivo:**")
    st.code("""
    danceability,energy,key,loudness,mode,speechiness,acousticness,
    instrumentalness,liveness,valence,tempo,duration_ms,time_signature,liked
    """)
    st.stop()

# Lectura de datos
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["duration_min"] = df["duration_ms"] / 60000
    return df

df = load_data(file)

# Opciones de filtrado en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Opciones de Filtrado")

# Filtro por liked
liked_filter = st.sidebar.selectbox(
    "Filtrar por preferencia:",
    options=["Todas", "Solo gustadas (1)", "Solo no gustadas (0)"]
)

# Filtro por duración
duration_range = st.sidebar.slider(
    "Duración de canciones (minutos):",
    min_value=float(df['duration_min'].min()),
    max_value=float(df['duration_min'].max()),
    value=(float(df['duration_min'].min()), float(df['duration_min'].max())),
    step=0.1
)

# Filtro por tempo
tempo_range = st.sidebar.slider(
    "Rango de Tempo (BPM):",
    min_value=float(df['tempo'].min()),
    max_value=float(df['tempo'].max()),
    value=(float(df['tempo'].min()), float(df['tempo'].max())),
    step=1.0
)

# Aplicar filtros
df_filtered = df.copy()

if liked_filter == "Solo gustadas (1)":
    df_filtered = df_filtered[df_filtered['liked'] == 1]
elif liked_filter == "Solo no gustadas (0)":
    df_filtered = df_filtered[df_filtered['liked'] == 0]

df_filtered = df_filtered[
    (df_filtered['duration_min'] >= duration_range[0]) & 
    (df_filtered['duration_min'] <= duration_range[1]) &
    (df_filtered['tempo'] >= tempo_range[0]) & 
    (df_filtered['tempo'] <= tempo_range[1])
]

# Información del dataset en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Info del Dataset")
st.sidebar.metric("Total de canciones originales", len(df))
st.sidebar.metric("Canciones filtradas", len(df_filtered))
st.sidebar.metric("Canciones gustadas", len(df[df['liked'] == 1]))
st.sidebar.metric("Canciones no gustadas", len(df[df['liked'] == 0]))

# KPI's principales
st.header("Métricas Principales")

# Calcular KPIs
features = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence']

avg_duration = df_filtered['duration_min'].mean()
most_common_key = df_filtered['key'].mode()[0] if not df_filtered.empty else 0
avg_tempo = df_filtered['tempo'].mean()
liked_percentage = (len(df_filtered[df_filtered['liked'] == 1]) / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
energy_avg = df_filtered['energy'].mean()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Duración Promedio", f"{avg_duration:.2f} min")
with c2:
    st.metric("Tonalidad Común", f"Key {most_common_key}")
with c3:
    st.metric("Tempo Promedio", f"{avg_tempo:.1f} BPM")
with c4:
    st.metric("% Canciones Gustadas", f"{liked_percentage:.1f}%")
with c5:
    st.metric("Energía Promedio", f"{energy_avg:.3f}")

st.markdown("---")

# Análisis principal
tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribuciones", "🎯 Correlaciones", "🎨 Características", "🔍 Comparativas"])

with tab1:
    st.subheader("Distribución de Características Musicales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma de danceability
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df_filtered, x="danceability", hue="liked", kde=True, 
                    element="step", stat="percent", ax=ax)
        plt.title("Distribución de Danceability")
        plt.xlabel("Danceability")
        plt.ylabel("Porcentaje")
        st.pyplot(fig, clear_figure=True)
    
    with col2:
        # Boxplot de valence
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df_filtered, x="liked", y="valence", ax=ax)
        plt.title("Comparación de Valencia según Preferencia")
        plt.xlabel("Liked (0 = No, 1 = Sí)")
        plt.ylabel("Valence")
        st.pyplot(fig, clear_figure=True)
    
    # Distribución de tonalidades
    st.subheader("Distribución de Tonalidades por Modo")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df_filtered, x="key", hue="mode", ax=ax)
    plt.title("Distribución de Tonalidades (Key) por Modo")
    plt.xlabel("Key (0 = C, 1 = C#, 2 = D, 3 = D#, 4 = E, 5 = F, 6 = F#, 7 = G, 8 = G#, 9 = A, 10 = A#, 11 = B)")
    plt.ylabel("Número de canciones")
    st.pyplot(fig, clear_figure=True)

with tab2:
    st.subheader("Matriz de Correlaciones")
    
    # Matriz de correlación
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    corr = df_filtered[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                cbar=True, square=True, ax=ax)
    plt.title("Matriz de Correlación entre Métricas Musicales")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    
    # Correlaciones más fuertes
    st.subheader("Correlaciones más Significativas")
    
    # Obtener correlaciones excluyendo la diagonal
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_values = corr.mask(mask).stack().reset_index()
    corr_values.columns = ['Variable 1', 'Variable 2', 'Correlación']
    corr_values = corr_values[corr_values['Variable 1'] != corr_values['Variable 2']]
    
    strongest_corr = corr_values.reindex(corr_values['Correlación'].abs().sort_values(ascending=False).index).head(10)
    
    for _, row in strongest_corr.iterrows():
        correlation = row['Correlación']
        if abs(correlation) > 0.3:  # Solo mostrar correlaciones significativas
            strength = "Fuerte" if abs(correlation) > 0.7 else "Moderada" if abs(correlation) > 0.5 else "Débil"
            direction = "positiva" if correlation > 0 else "negativa"
            st.write(f"**{row['Variable 1']}** vs **{row['Variable 2']}**: {correlation:.3f} ({strength} {direction})")

with tab3:
    st.subheader("Análisis de Características Musicales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Duración promedio por característica (tu gráfico original mejorado)
        features = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence']
        
        duration_by_feature = {}
        for feature in features:
            if len(df_filtered) > 0:
                duration_by_feature[feature] = df_filtered.groupby(pd.cut(df_filtered[feature], bins=5))['duration_ms'].mean().mean()
        
        if duration_by_feature:
            res_df = pd.DataFrame(list(duration_by_feature.items()), columns=['Feature', 'Avg_duration_ms'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = plt.bar(res_df['Feature'], res_df['Avg_duration_ms'], 
                          color=plt.cm.viridis(np.linspace(0, 1, len(res_df))))
            plt.xticks(rotation=45, ha='right')
            plt.title("Duración Promedio por Característica Musical")
            plt.ylabel("Duración Promedio (ms)")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
    
    with col2:
        # Tempo promedio por característica (tu gráfico original mejorado)
        tempo_by_feature = {}
        for feature in features:
            if len(df_filtered) > 0 and df_filtered[feature].mean() != 0:
                tempo_by_feature[feature] = (df_filtered['tempo'] * df_filtered[feature]).mean() / df_filtered[feature].mean()
        
        if tempo_by_feature:
            tempo_df = pd.Series(tempo_by_feature)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            tempo_df.plot(kind='bar', ax=ax, color=plt.cm.plasma(np.linspace(0, 1, len(tempo_df))))
            plt.title("Tempo Promedio por Característica")
            plt.ylabel("Tempo Promedio (BPM)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

with tab4:
    st.subheader("Comparativa: Canciones Gustadas vs No Gustadas")
    
    if len(df_filtered[df_filtered['liked'] == 1]) > 0 and len(df_filtered[df_filtered['liked'] == 0]) > 0:
        # Radar Chart
        categories = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "liveness"]
        stats = df_filtered.groupby("liked")[categories].mean()
        
        labels = list(stats.columns)
        num_vars = len(labels)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['#FF6B6B', '#4ECDC4']
        for i, (idx, row) in enumerate(stats.iterrows()):
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=f"{'Gustadas' if idx == 1 else 'No Gustadas'}", 
                   color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        plt.title("Perfil Promedio: Canciones Gustadas vs No Gustadas", pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        st.pyplot(fig, clear_figure=True)
        
        # Tabla comparativa
        st.subheader("Estadísticas Comparativas")
        comparison_stats = df_filtered.groupby("liked")[["danceability", "energy", "valence", "tempo", "duration_min"]].mean()
        comparison_stats.index = ['No Gustadas', 'Gustadas']
        st.dataframe(comparison_stats.round(3))
    else:
        st.warning("No hay suficientes datos en ambas categorías (gustadas/no gustadas) para realizar la comparación.")

# Footer con información adicional
st.markdown("---")
st.markdown("**💡 Consejos de uso:**")
st.markdown("- Utiliza los filtros en la barra lateral para explorar subconjuntos específicos de datos")
st.markdown("- Las correlaciones te ayudan a entender qué características están relacionadas")
st.markdown("- El radar chart muestra el 'perfil musical' de las canciones que te gustan vs las que no")