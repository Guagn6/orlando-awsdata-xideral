import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título de la app
st.title("Visualización de Datos de Netflix")

# Sidebar 
st.sidebar.title("Datos")
file = st.sidebar.file_uploader("Aquí ouedes subir tu csv")
st.sidebar.caption("Si usas otra opción abajo puedes mapear tus datos")
if file is None:
    st.warning("Inserte un archivo CSV")
    st.stop()

# READ DATA
df = pd.read_csv(file)

# KPI's
titulos = df["title"].value_counts().sum()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("titulos",titulos)
with c2:
    st.metric("Total series",2)
with c3:
    st.metric("Total peliculas",2)
with c4:
    st.metric("Media duración peliculas",2)
with c5:
    st.metric("Media duración Series",2)

st.markdown("---")

cols = st.columns(2)

with cols[0]:
    st.write("Titulos por año de estreno")
    counts = df["release_year"].dropna().astype(int).value_counts().sort_index()
    fig = plt.figure(figsize=(10,4))
    plt.plot(counts.index, counts.values)
    plt.xlabel("Año")
    plt.ylabel("Títulos")
    plt.title("Títulos por año de estreno")
    #plt.show()
    st.pyplot(fig, clear_figure=True)
with cols[1]:
    st.write("Cobertura por clasificación Top 10")
    rating_ct = df["rating"].value_counts().sort_values(ascending=False).head(10)
    fig = plt.figure(figsize=(10,4))
    plt.bar(rating_ct.index.astype(str), rating_ct.values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Rating")
    plt.ylabel("Títulos")
    plt.title("Cobertura por rating (Top 10)")
    plt.tight_layout()
    # plt.show()
    st.pyplot(fig, clear_figure=True)