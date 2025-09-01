import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Función de simulación
# ======================
def simular_vendedor(simulaciones, precio_venta, precio_compra, precio_reciclaje, max_oferta=100):
    """
    Ejecuta la simulación del problema del vendedor de periódicos.
    
    Retorna:
        df_resultados: DataFrame con columnas
                       ['Oferta', 'Ganancia promedio', 'Ganancias (lista por simulación)']
    """
    rng = np.random.default_rng(42)
    # Suponemos una demanda aleatoria uniforme entre 0 y max_oferta (puedes cambiar si tu notebook usa otra dist)
    demandas = rng.integers(low=0, high=max_oferta+1, size=simulaciones)

    resultados = []
    for oferta in range(1, max_oferta+1):
        ganancias = []
        for d in demandas:
            vendidos = min(oferta, d)
            sobrante = max(0, oferta - d)

            ingreso = vendidos * precio_venta
            costo = oferta * precio_compra
            reciclaje = sobrante * precio_reciclaje

            ganancia = ingreso - costo + reciclaje
            ganancias.append(ganancia)

        resultados.append({
            "Oferta": oferta,
            "Ganancia promedio": np.mean(ganancias),
            "Ganancias": ganancias
        })

    df_resultados = pd.DataFrame(resultados)
    return df_resultados


# ======================
# Interfaz Streamlit
# ======================
st.set_page_config(page_title="Lab 3 – Vendedor de Periódicos", layout="centered")
st.title("Laboratorio 3 – Problema del Vendedor de Periódicos")

st.sidebar.header("Parámetros de simulación")
simulaciones = st.sidebar.number_input("Cantidad de simulaciones", min_value=100, value=1000, step=100)
precio_venta = st.sidebar.number_input("Precio de venta", min_value=0.0, value=1.5, step=0.1)
precio_compra = st.sidebar.number_input("Precio de compra", min_value=0.0, value=1.0, step=0.1)
precio_reciclaje = st.sidebar.number_input("Precio de reciclaje", min_value=0.0, value=0.2, step=0.1)

ejecutar = st.sidebar.button("Iniciar simulación")

if ejecutar:
    st.subheader("Resultados de la simulación")
    df_res = simular_vendedor(simulaciones, precio_venta, precio_compra, precio_reciclaje)

    # ======================
    # Gráfico de línea
    # ======================
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df_res["Oferta"], df_res["Ganancia promedio"], marker="o")
    ax1.set_title("Ganancia promedio por oferta")
    ax1.set_xlabel("Oferta")
    ax1.set_ylabel("Ganancia promedio")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

    # ======================
    # Gráfico de violín
    # ======================
    # Expandimos el dataframe para seaborn
    df_expand = df_res.explode("Ganancias")
    df_expand["Ganancias"] = df_expand["Ganancias"].astype(float)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df_expand, x="Oferta", y="Ganancias", ax=ax2, inner="quartile", scale="width")
    ax2.set_title("Distribución de ganancias por oferta")
    ax2.set_xlabel("Oferta")
    ax2.set_ylabel("Ganancia")
    st.pyplot(fig2)

    # ======================
    # Mostrar tabla resumen
    # ======================
    st.subheader("Tabla resumen")
    st.dataframe(df_res[["Oferta", "Ganancia promedio"]].round(2))
else:
    st.info("Configura los parámetros en la barra lateral y haz clic en **Iniciar simulación**.")