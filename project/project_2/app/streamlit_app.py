from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core.model import generate_problem, evaluate, greedy_baseline
from core.ga import run_ga
from core.sa import run_sa
from core.tabu import run_tabu


st.set_page_config(page_title="Programa de Cosecha (Metaheurísticas)", layout="wide")

st.title("Programa de Cosecha • Knapsack Binario con Doble Restricción")
st.caption("Maximización de caña recolectada bajo restricciones de horas (W) y capacidad total (K*C). "
           "Se grafica la *disminución* del costo penalizado para observar la convergencia.")


@st.cache_data(show_spinner=False)
def build_problem(seed: int, N: int, workers: int, K: int, C: int):
    prob = generate_problem(N=N, workers=workers, K=K, C=C, seed=seed)
    df = pd.DataFrame({
        "parcela": np.arange(prob.N, dtype=int),
        "Vi_kg": prob.V.astype(int),
        "Pi_horas": prob.P.astype(float),
        "ratio_Vi_Pi": (prob.V / np.maximum(prob.P, 1e-9))
    }).sort_values("parcela").reset_index(drop=True)
    return prob, df


def _interp_text(prob, ev, history_best_cost) -> str:
    start = history_best_cost[0]
    end = history_best_cost[-1]
    drop = (start - end) / abs(start) * 100 if start != 0 else 0.0
    feas = "Sí" if ev["feasible_all"] else "No"
    advice = []
    if not ev["feasible_hours"]:
        advice.append("aumentar `workers` (↑W) o seleccionar menos parcelas")
    if not ev["feasible_capacity"]:
        advice.append("aumentar `K` o `C` (↑K*C) o seleccionar menos parcelas")
    advice_txt = " • Recomendación: " + "; ".join(advice) if advice else ""
    return (f"El costo penalizado disminuyó **{drop:.2f}%** (menor es mejor). "
            f"Factibilidad total: **{feas}**. "
            f"Horas usadas: **{ev['hours_used']:.2f}** de **{prob.W:.2f}**; "
            f"Kg transportados: **{ev['kg_used']:,}** de **{prob.capacity_total:,}**."
            f"{advice_txt}")


# ---------------- Sidebar: parámetros con restricciones ----------------
with st.sidebar:
    st.header("Parámetros del Problema")
    # Restricciones solicitadas + keys únicos
    seed = st.number_input("seed", value=42, step=1, key="seed")
    N = st.number_input("N (número de parcelas)", value=80, min_value=20, max_value=200, step=1, key="N")
    workers = st.number_input("workers (≥1)", value=10, min_value=1, max_value=500, step=1, key="workers")
    K = st.number_input("K (vehículos, ≥1)", value=3, min_value=1, max_value=200, step=1, key="K")
    C = st.number_input("C (capacidad por vehículo, kg)", value=1800, min_value=1000, max_value=2500, step=50,
                        help="Restricción: 1000 ≤ C ≤ 2500", key="C")

    st.divider()
    st.header("Hiperparámetros por algoritmo")

    with st.expander("GA (Algoritmo Genético)"):
        ga_params: Dict[str, Any] = {}
        ga_params["pop"] = st.number_input("Población", value=80, min_value=10, max_value=2000, step=1, key="ga_pop")
        ga_params["gens"] = st.number_input("Generaciones", value=300, min_value=10, max_value=5000, step=10, key="ga_gens")
        ga_params["pc"] = st.slider("Prob. crossover (pc)", min_value=0.0, max_value=1.0, value=0.8, step=0.01, key="ga_pc")
        ga_params["pm"] = st.slider("Prob. mutación (pm)", min_value=0.0, max_value=1.0, value=0.02, step=0.001, key="ga_pm")
        ga_params["elite"] = st.number_input("Elitismo (#)", value=1, min_value=0, max_value=50, step=1, key="ga_elite")
        ga_params["tournament_k"] = st.number_input("Torneo k", value=3, min_value=2, max_value=10, step=1, key="ga_tk")

    with st.expander("SA (Recocido Simulado)"):
        sa_params: Dict[str, Any] = {}
        sa_params["T0"] = st.number_input("T0 (temperatura inicial)", value=1.0, min_value=1e-6, max_value=10.0, step=0.1, format="%.4f", key="sa_T0")
        sa_params["Tend"] = st.number_input("Tend (temperatura final)", value=1e-3, min_value=1e-8, max_value=1.0, step=1e-4, format="%.6f", key="sa_Tend")
        sa_params["iters"] = st.number_input("Iteraciones (SA)", value=int(80 * 300), min_value=100, max_value=1000000, step=100, key="sa_iters")
        sa_params["alpha"] = st.number_input("alpha (enfriamiento geom.)", value=0.99, min_value=0.90, max_value=0.9999, step=0.0005, format="%.4f", key="sa_alpha")

    with st.expander("TS (Búsqueda Tabú)"):
        ts_params: Dict[str, Any] = {}
        ts_params["iters"] = st.number_input("Iteraciones (TS)", value=int(80 * 300), min_value=100, max_value=1000000, step=100, key="ts_iters")
        ts_params["tabu_size"] = st.number_input("Tamaño lista tabú", value=max(1, int(np.ceil(0.1 * 80))), min_value=1, max_value=5000, step=1, key="ts_tabu")
        ts_params["consider_all"] = st.checkbox("Evaluar todos los vecinos (lento pero robusto)", value=True, key="ts_consider_all")

    st.caption("Nota: Al cambiar N, conviene ajustar iters/tabu_size proporcionalmente.")

    run_btn = st.button("Resolver", type="primary", key="run_btn")


# ---------------- Main ----------------
prob, data_df = build_problem(seed=int(seed), N=int(N), workers=int(workers), K=int(K), C=int(C))

# Ajusta iters y tabu_size por si quedaron desfasados del N actual
if run_btn:
    if sa_params["iters"] < N*50:  # asegura suficientes iteraciones
        sa_params["iters"] = int(N * 300)
    if ts_params["iters"] < N*50:
        ts_params["iters"] = int(N * 300)
    # limitar tabu_size a [1, N]
    ts_params["tabu_size"] = int(np.clip(ts_params["tabu_size"], 1, N))

# Datos y parámetros globales
col1, col2 = st.columns([2, 1], gap="large")
with col1:
    st.subheader("Datos de parcelas")
    st.dataframe(data_df, use_container_width=True, hide_index=True)
with col2:
    st.subheader("Parámetros globales")
    st.metric("W (horas)", f"{prob.W:.2f}")
    st.metric("K", f"{prob.K}")
    st.metric("C (kg)", f"{prob.C}")
    st.metric("Capacidad total K*C (kg)", f"{prob.capacity_total:,}")

st.divider()

# Tabs para cada algoritmo
tab_ga, tab_sa, tab_ts = st.tabs(["Algoritmo Genético (GA)", "Recocido Simulado (SA)", "Búsqueda Tabú (TS)"])

# Ejecuta todos si el usuario pulsa Resolver (permite comparar y mostrar en cada tab)
ga_res = sa_res = ts_res = None
if run_btn:
    ga_res = run_ga(prob, params=ga_params, seed=int(seed))
    sa_res = run_sa(prob, params=sa_params, seed=int(seed))
    ts_res = run_tabu(prob, params=ts_params, seed=int(seed))

# ----- TAB GA -----
with tab_ga:
    st.markdown("**¿Qué hace?** GA explora poblaciones de soluciones binarias; "
                "usa selección por torneo, cruce de 1 punto y mutación bit-flip con elitismo "
                "para mejorar el costo penalizado generación a generación.")
    if ga_res is None:
        st.info("Configura parámetros y pulsa **Resolver** para ejecutar GA.")
    else:
        ev = evaluate(prob, ga_res.best_X)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Z (kg recolectados)", f"{ev['Z']:,}")
            st.metric("Costo penalizado (↓)", f"{ga_res.best_cost:,.2f}")
        with c2:
            st.metric("Horas usadas", f"{ev['hours_used']:.2f}")
            st.metric("Límite horas (W)", f"{prob.W:.2f}")
        with c3:
            st.metric("Kg transportados", f"{ev['kg_used']:,}")
            st.metric("Límite K*C (kg)", f"{prob.capacity_total:,}")

        # Gráfico de convergencia
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ga_res.history_best_cost)
        ax.set_title("GA • Convergencia (Costo penalizado, debe disminuir)")
        ax.set_xlabel("Generación")
        ax.set_ylabel("Costo")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig, clear_figure=True)

        # Parcelas y CSV
        chosen_idx = np.where(ga_res.best_X == 1)[0]
        st.subheader("Parcelas seleccionadas (GA)")
        st.text(f"Índices: {chosen_idx.tolist()}")

        sol_df = data_df.copy()
        sol_df["X"] = ga_res.best_X
        csv = sol_df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV (GA)", data=csv, file_name="solucion_cosecha_GA.csv", mime="text/csv")

        # Interpretación
        st.markdown(f"**Interpretación:** {_interp_text(prob, ev, ga_res.history_best_cost)}")

# ----- TAB SA -----
with tab_sa:
    st.markdown("**¿Qué hace?** SA explora vecinos por flips de un bit y acepta peores movimientos "
                "según una probabilidad controlada por la temperatura; la temperatura decrece "
                "geométricamente para refinar la búsqueda.")
    if sa_res is None:
        st.info("Configura parámetros y pulsa **Resolver** para ejecutar SA.")
    else:
        ev = evaluate(prob, sa_res.best_X)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Z (kg recolectados)", f"{ev['Z']:,}")
            st.metric("Costo penalizado (↓)", f"{sa_res.best_cost:,.2f}")
        with c2:
            st.metric("Horas usadas", f"{ev['hours_used']:.2f}")
            st.metric("Límite horas (W)", f"{prob.W:.2f}")
        with c3:
            st.metric("Kg transportados", f"{ev['kg_used']:,}")
            st.metric("Límite K*C (kg)", f"{prob.capacity_total:,}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sa_res.history_best_cost)
        ax.set_title("SA • Convergencia (Costo penalizado, debe disminuir)")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Costo")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig, clear_figure=True)

        chosen_idx = np.where(sa_res.best_X == 1)[0]
        st.subheader("Parcelas seleccionadas (SA)")
        st.text(f"Índices: {chosen_idx.tolist()}")

        sol_df = data_df.copy()
        sol_df["X"] = sa_res.best_X
        csv = sol_df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV (SA)", data=csv, file_name="solucion_cosecha_SA.csv", mime="text/csv")

        st.markdown(f"**Interpretación:** {_interp_text(prob, ev, sa_res.history_best_cost)}")

# ----- TAB TS -----
with tab_ts:
    st.markdown("**¿Qué hace?** TS explora el mejor vecino permitido por una **lista tabú** "
                "que evita ciclos recientes; permite aspiración si un movimiento mejora el mejor global.")
    if ts_res is None:
        st.info("Configura parámetros y pulsa **Resolver** para ejecutar TS.")
    else:
        ev = evaluate(prob, ts_res.best_X)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Z (kg recolectados)", f"{ev['Z']:,}")
            st.metric("Costo penalizado (↓)", f"{ts_res.best_cost:,.2f}")
        with c2:
            st.metric("Horas usadas", f"{ev['hours_used']:.2f}")
            st.metric("Límite horas (W)", f"{prob.W:.2f}")
        with c3:
            st.metric("Kg transportados", f"{ev['kg_used']:,}")
            st.metric("Límite K*C (kg)", f"{prob.capacity_total:,}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ts_res.history_best_cost)
        ax.set_title("TS • Convergencia (Costo penalizado, debe disminuir)")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Costo")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig, clear_figure=True)

        chosen_idx = np.where(ts_res.best_X == 1)[0]
        st.subheader("Parcelas seleccionadas (TS)")
        st.text(f"Índices: {chosen_idx.tolist()}")

        sol_df = data_df.copy()
        sol_df["X"] = ts_res.best_X
        csv = sol_df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV (TS)", data=csv, file_name="solucion_cosecha_TS.csv", mime="text/csv")

        st.markdown(f"**Interpretación:** {_interp_text(prob, ev, ts_res.history_best_cost)}")

# ----- Comparativa e interpretación global -----
st.divider()
st.subheader("Comparativa e interpretación global")
if not run_btn:
    st.info("Pulsa **Resolver** para comparar GA, SA y TS.")
else:
    rows = []
    for name, res in [("GA", ga_res), ("SA", sa_res), ("TS", ts_res)]:
        ev = evaluate(prob, res.best_X)
        rows.append({
            "algoritmo": name,
            "Z": ev["Z"],
            "hours_used": round(ev["hours_used"], 2),
            "kg_used": ev["kg_used"],
            "feasible": ev["feasible_all"],
            "best_cost": round(res.history_best_cost[-1], 2),
        })
    comp_df = pd.DataFrame(rows).sort_values(["feasible", "Z", "best_cost"], ascending=[False, False, True])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    best_name = comp_df.iloc[0]["algoritmo"]
    st.markdown(
        f"**Mejor desempeño (según Z factible y costo menor):** {best_name}. "
        "Si hay empates, priorizamos factibilidad y mayor Z."
    )
