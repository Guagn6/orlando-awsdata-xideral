import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título de la app
st.title("Visualización de Datos de Spotify")

# Sidebar 
st.sidebar.title("Datos")
file = st.sidebar.file_uploader("Carga tu archivo csv aquí")
st.sidebar.caption("Otras opciones para cargar tus datos...")
if file is None:
    st.warning("Inserta tu archivo para visualizar sus datos")
    st.stop()

# READ DATA
df = pd.read_csv(file)

# KPI's
print("Tiempo promedio de canciones: " + str(df['duration_ms'].mean()))

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Características",2)
with c2:
    st.metric("Mínimo por Característica",2)
with c3:
    st.metric("Máximo por Característica",2)
with c4:
    st.metric("Duración promedio por característica",2)
with c5:
    st.metric("Característica con mayor contenido",2)

st.markdown("---")

cols = st.columns(2)

with cols[0]:
    features = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence']
    resultados = {}
    for colum in features:
        resultados[colum] = df.groupby(colum)['duration_ms'].sum().mean()  
    
    res_df = pd.DataFrame(list(resultados.items()), columns=['Feature', 'Total_duration_ms'])
    fig = plt.figure(figsize=(10,6))
    plt.bar(res_df['Feature'], res_df['Total_duration_ms'])
    plt.xticks(rotation=45)
    plt.title("Duración Promedio por Característica")
    plt.ylabel("Duración Total (ms)")
    # plt.show()
    st.pyplot(fig, clear_figure=True)
with cols[1]:
    features = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence']
    tempoMedias = {}
    for colum in features:
        tempoMedias[colum] = (df['tempo'] * df[colum]).mean() / df[colum].mean()

    tempoMediasDf = pd.Series(tempoMedias)

    fig = plt.figure(figsize=(10,6))
    tempoMediasDf.plot(kind='bar')

    plt.title("Tempo promedio por Género")
    plt.ylabel("Tempo promedio")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig, clear_figure=True)