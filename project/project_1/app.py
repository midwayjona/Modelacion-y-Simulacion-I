# -------------------------------
# Proyecto 1 – Perinola (Streamlit)
# -------------------------------
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib" # Evita error en algunos sistemas
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple

import streamlit as st

# -------------------------------
# Modelo y utilidades
# -------------------------------
ACCIONES = ["PON_1", "PON_2", "TOMA_1", "TOMA_2", "TOMA_TODO", "TODOS_PONEN"]

@dataclass
class ConfigJuego:
    n_jugadores: int = 4
    capital_inicial: int = 10
    max_jugadas: int = 100_000
    seed: Optional[int] = 123

@dataclass
class ResultadoJuego:
    historial_saldos: pd.DataFrame
    historial_pozo: pd.Series
    jugada_primera_eliminacion: Optional[int]
    jugada_ganador: int
    ganador_id: int
    resumen_final: pd.DataFrame

def jugadores_activos(saldos: np.ndarray) -> np.ndarray:
    return np.where(saldos > 0)[0]

def aplicar_pago(saldo: int, monto: int) -> Tuple[int, int, bool]:
    pagado = min(saldo, monto)
    nuevo_saldo = saldo - pagado
    eliminado = (nuevo_saldo == 0)
    return pagado, nuevo_saldo, eliminado

def aplicar_cobro(pozo: int, monto: int) -> Tuple[int, int]:
    cobrado = min(pozo, monto)
    return cobrado, pozo - cobrado

def tirar_perinola(rng: np.random.Generator) -> str:
    return rng.choice(ACCIONES)

def simular_un_juego(cfg: ConfigJuego) -> ResultadoJuego:
    rng_local = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    n = cfg.n_jugadores

    saldos = np.full(n, cfg.capital_inicial, dtype=int)
    pozo = 0
    turno = 0

    hist_saldos = []
    hist_pozo = []
    jugada_primera_elim = None
    jugada = 0

    while jugada < cfg.max_jugadas:
        activos = jugadores_activos(saldos)
        if len(activos) <= 1:
            break

        if saldos[turno] <= 0:
            turno = (turno + 1) % n
            continue

        accion = tirar_perinola(rng_local)

        if accion == "PON_1":
            pagado, nuevo, elim = aplicar_pago(saldos[turno], 1)
            saldos[turno] = nuevo
            pozo += pagado
            if elim and jugada_primera_elim is None:
                jugada_primera_elim = jugada + 1

        elif accion == "PON_2":
            pagado, nuevo, elim = aplicar_pago(saldos[turno], 2)
            saldos[turno] = nuevo
            pozo += pagado
            if elim and jugada_primera_elim is None:
                jugada_primera_elim = jugada + 1

        elif accion == "TOMA_1":
            cobrado, pozo = aplicar_cobro(pozo, 1)
            saldos[turno] += cobrado

        elif accion == "TOMA_2":
            cobrado, pozo = aplicar_cobro(pozo, 2)
            saldos[turno] += cobrado

        elif accion == "TOMA_TODO":
            cobrado, pozo = pozo, 0
            saldos[turno] += cobrado

        elif accion == "TODOS_PONEN":
            total_aportado = 0
            for j in range(n):
                if saldos[j] > 0:
                    pagado, nuevo, elim = aplicar_pago(saldos[j], 1)
                    saldos[j] = nuevo
                    total_aportado += pagado
                    if elim and jugada_primera_elim is None:
                        jugada_primera_elim = jugada + 1
            pozo += total_aportado

        hist_saldos.append(saldos.copy())
        hist_pozo.append(pozo)

        turno = (turno + 1) % n
        jugada += 1

        if len(jugadores_activos(saldos)) <= 1:
            break

    activos = jugadores_activos(saldos)
    ganador = int(activos[0]) if len(activos) == 1 else int(np.argmax(saldos))
    jugada_ganador = jugada

    hist_saldos_df = pd.DataFrame(hist_saldos, columns=[f"J{i}" for i in range(n)])
    hist_pozo_sr = pd.Series(hist_pozo, name="Pozo")

    saldo_final = pd.Series(saldos, index=[f"J{i}" for i in range(n)], name="Saldo final")
    ganancia_neta = saldo_final - cfg.capital_inicial
    resumen_final = pd.DataFrame({"Saldo final": saldo_final, "Ganancia/Pérdida": ganancia_neta})

    return ResultadoJuego(
        historial_saldos=hist_saldos_df,
        historial_pozo=hist_pozo_sr,
        jugada_primera_eliminacion=jugada_primera_elim,
        jugada_ganador=jugada_ganador,
        ganador_id=ganador,
        resumen_final=resumen_final
    )

def experimentar(n_repeticiones: int, cfg_base: ConfigJuego) -> pd.DataFrame:
    registros = []
    for i in range(n_repeticiones):
        seed_i = (cfg_base.seed or 0) + i + 1
        cfg_i = ConfigJuego(
            n_jugadores=cfg_base.n_jugadores,
            capital_inicial=cfg_base.capital_inicial,
            max_jugadas=cfg_base.max_jugadas,
            seed=seed_i
        )
        r = simular_un_juego(cfg_i)
        registros.append({
            "juego": i + 1,
            "jugada_primera_eliminacion": r.jugada_primera_eliminacion,
            "jugada_ganador": r.jugada_ganador,
            "ganador_id": r.ganador_id
        })
    return pd.DataFrame(registros)

def estudio_vs_njugadores(lista_n: List[int], repeticiones: int, capital_inicial: int = 10, seed: int = 123) -> pd.DataFrame:
    resultados = []
    base = 0
    for n in lista_n:
        cfg = ConfigJuego(n_jugadores=n, capital_inicial=capital_inicial, seed=seed + base)
        df = experimentar(repeticiones, cfg)
        resultados.append({
            "n_jugadores": n,
            "jugadas_promedio_hasta_ganador": df["jugada_ganador"].mean(),
            "jugadas_p95_hasta_ganador": df["jugada_ganador"].quantile(0.95)
        })
        base += repeticiones
    return pd.DataFrame(resultados)

# -------------------------------
# Interfaz Streamlit
# -------------------------------
st.set_page_config(page_title="Proyecto 1 – Perinola", layout="wide")
st.title("Proyecto 1 — Simulación del Juego de la Perinola")

with st.sidebar:
    st.header("Parámetros generales")
    n_jugadores = st.number_input("Número de jugadores", min_value=2, max_value=20, value=5, step=1)
    capital_inicial = st.number_input("Capital inicial por jugador", min_value=1, max_value=1000, value=10, step=1)
    max_jugadas = st.number_input("Máx. jugadas (límite de seguridad)", min_value=100, max_value=200000, value=100000, step=100)
    seed = st.number_input("Semilla aleatoria", min_value=0, max_value=10_000_000, value=123, step=1)

tab1, tab2 = st.tabs(["Simular un juego", "Experimentos (múltiples juegos)"])

# -------------------------------
# Tab 1: Un juego
# -------------------------------
with tab1:
    st.subheader("Simulación de un juego")
    cfg = ConfigJuego(n_jugadores=n_jugadores, capital_inicial=capital_inicial, max_jugadas=max_jugadas, seed=seed)
    if st.button("Ejecutar simulación (un juego)"):
        res = simular_un_juego(cfg)

        colA, colB, colC = st.columns(3)
        colA.metric("Jugada 1ª eliminación", f"{res.jugada_primera_eliminacion}")
        colB.metric("Jugada fin (ganador)", f"{res.jugada_ganador}")
        colC.metric("Ganador", f"J{res.ganador_id}")

        # Saldos por jugador
        fig1, ax1 = plt.subplots(figsize=(9, 5))
        for col in res.historial_saldos.columns:
            ax1.plot(res.historial_saldos.index + 1, res.historial_saldos[col], label=col, linewidth=1.8)
        ax1.set_title("Evolución del saldo por jugador")
        ax1.set_xlabel("Jugada")
        ax1.set_ylabel("Saldo")
        ax1.grid(alpha=0.3)
        ax1.legend(ncol=2)
        st.pyplot(fig1)

        # Pozo
        fig2, ax2 = plt.subplots(figsize=(9, 3.8))
        ax2.plot(res.historial_pozo.index + 1, res.historial_pozo.values, linewidth=2)
        ax2.set_title("Evolución del pozo")
        ax2.set_xlabel("Jugada")
        ax2.set_ylabel("Pozo")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

        # Ganancia / Pérdida
        fig3, ax3 = plt.subplots(figsize=(6.5, 3.8))
        final_gp = res.resumen_final["Ganancia/Pérdida"]
        ax3.bar(final_gp.index, final_gp.values)
        ax3.set_title("Ganancia / Pérdida por jugador (final del juego)")
        ax3.set_xlabel("Jugador")
        ax3.set_ylabel("Ganancia/Pérdida")
        ax3.grid(axis='y', alpha=0.3)
        st.pyplot(fig3)

        st.subheader("Resumen final")
        st.dataframe(res.resumen_final)

# -------------------------------
# Tab 2: Experimentos (independientes)
# -------------------------------
with tab2:
    st.subheader("Experimentos (múltiples juegos)")

    # ---- Form 1: Estadísticas de múltiples juegos ----
    with st.form("form_experimentos"):
        repeticiones = st.number_input(
            "Número de juegos a simular",
            min_value=100, max_value=20000, value=1000, step=100,
            key="rep_experimentos"
        )
        submitted_exp = st.form_submit_button("Ejecutar experimentos")

    if submitted_exp:
        cfg_base = ConfigJuego(
            n_jugadores=n_jugadores,
            capital_inicial=capital_inicial,
            max_jugadas=max_jugadas,
            seed=seed
        )
        df_exp = experimentar(int(repeticiones), cfg_base)

        col1, col2 = st.columns(2)
        col1.metric("Prom. jugadas hasta 1ª eliminación",
                    f"{df_exp['jugada_primera_eliminacion'].mean():.3f}")
        col2.metric("Prom. jugadas hasta ganador",
                    f"{df_exp['jugada_ganador'].mean():.3f}")

        st.markdown("**Resumen estadístico (jugadas):**")
        st.dataframe(df_exp[["jugada_primera_eliminacion", "jugada_ganador"]].describe())

    st.markdown("---")
    st.markdown("### Efecto del número de jugadores (promedio y P95 de jugadas hasta ganador)")

    # ---- Form 2: Estudio vs N (independiente del anterior) ----
    with st.form("form_vs_n"):
        n_min = st.number_input(
            "N jugadores (mín)",
            min_value=2, max_value=20, value=3, step=1, key="n_min"
        )
        n_max = st.number_input(
            "N jugadores (máx)",
            min_value=int(n_min), max_value=30, value=10, step=1, key="n_max"
        )
        rep_vs_n = st.number_input(
            "Repeticiones por cada N",
            min_value=100, max_value=5000, value=400, step=100, key="rep_vs_n"
        )
        submitted_vs_n = st.form_submit_button("Ejecutar estudio vs N")

    if submitted_vs_n:
        lista_n = list(range(int(n_min), int(n_max) + 1))
        df_n = estudio_vs_njugadores(
            lista_n=lista_n,
            repeticiones=int(rep_vs_n),
            capital_inicial=capital_inicial,
            seed=seed
        )
        st.dataframe(df_n)

        figN, axN = plt.subplots(figsize=(7.5, 4))
        axN.plot(df_n["n_jugadores"],
                 df_n["jugadas_promedio_hasta_ganador"],
                 marker="o", linewidth=2, label="Promedio")
        axN.plot(df_n["n_jugadores"],
                 df_n["jugadas_p95_hasta_ganador"],
                 marker="^", linewidth=1.8, label="P95")
        axN.set_title("Jugadas hasta encontrar un ganador vs. número de jugadores")
        axN.set_xlabel("Número de jugadores")
        axN.set_ylabel("Jugadas hasta ganador")
        axN.grid(alpha=0.3)
        axN.legend()
        st.pyplot(figN)