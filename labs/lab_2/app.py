import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pathlib
from sklearn.linear_model import LinearRegression

# BestFit (como en clase)
from fitter import Fitter
import scipy.stats as st_stats


# =========================
# Utilidades comunes
# =========================
def fmt_money(x: float) -> str:
    return f"${x:,.2f}"

# =========================
# EJERCICIO 1 (Ahorro)
# =========================
ANIOS = 17
APORTE = 20_000.0
MEDIA = 0.04
STD   = 0.10

def simular_ruta_ej1(anios=ANIOS, aporte=APORTE, media=MEDIA, std=STD, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    r = rng.normal(media, std, anios)
    s = np.zeros(anios, dtype=float)
    saldo = 0.0
    for t in range(anios):
        saldo += aporte
        saldo *= (1.0 + r[t])
        s[t] = saldo
    return r, s

def simular_montecarlo_ej1(trials=1000, anios=ANIOS, aporte=APORTE, media=MEDIA, std=STD, seed=42):
    rng = np.random.default_rng(seed)
    rend_all = np.zeros((trials, anios), dtype=float)
    saldo_all = np.zeros((trials, anios), dtype=float)
    for i in range(trials):
        r, s = simular_ruta_ej1(anios, aporte, media, std, rng)
        rend_all[i] = r
        saldo_all[i] = s
    return rend_all, saldo_all

def vista_ej1():
    st.subheader("Ejercicio 1 — Ahorro Universitario (Montecarlo)")
    st.caption("Aporte de $20,000 al inicio de cada año, rendimiento anual ~ Normal(4%, 10%), horizonte 17 años.")
    trials = st.number_input("Simulaciones (mín. 1,000)", min_value=1000, value=1000, step=100)
    seed = st.number_input("Semilla aleatoria", min_value=0, value=42, step=1)
    ejecutar = st.button("Ejecutar simulación")

    if not ejecutar:
        return

    rend_all, saldo_all = simular_montecarlo_ej1(trials=trials, seed=seed)
    monto_final = saldo_all[:, -1]

    # Métricas solicitadas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rend. anual promedio", f"{rend_all.mean():.2%}")
    col2.metric("Monto final promedio", fmt_money(float(monto_final.mean())))
    col3.metric("Escenario pesimista (P5)", fmt_money(float(np.percentile(monto_final, 5))))
    col4.metric("Escenario optimista (P95)", fmt_money(float(np.percentile(monto_final, 95))))

    # Gráficas solicitadas
    anios_axis = np.arange(1, ANIOS + 1)

    # 1) Rendimientos por año (promedio)
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(anios_axis, rend_all.mean(axis=0), linewidth=2)
    ax1.set_title("Rendimientos promedio por año")
    ax1.set_xlabel("Año")
    ax1.set_ylabel("Rendimiento")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

    # 2) Monto ahorrado por año (flujo promedio)
    flujo_all = np.empty_like(saldo_all)
    flujo_all[:, 0] = saldo_all[:, 0]
    if ANIOS > 1:
        flujo_all[:, 1:] = saldo_all[:, 1:] - saldo_all[:, :-1]
    flujo_prom = flujo_all.mean(axis=0)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(anios_axis, flujo_prom, linewidth=2)
    ax2.set_title("Monto ahorrado por año (promedio)")
    ax2.set_xlabel("Año")
    ax2.set_ylabel("USD")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    # 3) Monto acumulado por año (promedio)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(anios_axis, saldo_all.mean(axis=0), linewidth=2)
    ax3.set_title("Monto acumulado promedio por año")
    ax3.set_xlabel("Año")
    ax3.set_ylabel("USD")
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)


# =========================
# EJERCICIO 2 (Publicidad)
# =========================
CANDIDATAS = ['norm', 'lognorm', 'beta', 'expon', 'gamma', 'weibull_min']

def ajustar_regresion(df: pd.DataFrame) -> LinearRegression:
    X = df[["TV", "Radio", "Newspaper"]].values
    y = df["Sales"].values
    modelo = LinearRegression().fit(X, y)
    return modelo

def bestfit_variable(x: pd.Series, candidatas=CANDIDATAS, bins=50):
    datos = x.dropna().values
    f = Fitter(datos, distributions=candidatas, bins=bins)
    f.fit()
    try:
        best = f.get_best(method='ks')
    except Exception:
        best = f.get_best()
    dist_name = list(best.keys())[0]
    raw = best[dist_name]
    shapes = tuple(raw['args']) if 'args' in raw else ()
    params = {'shapes': shapes, 'loc': float(raw['loc']), 'scale': float(raw['scale'])}
    return dist_name, params

def triangular_params(x: pd.Series):
    a = float(x.min())
    b = float(x.max())
    m = float(x.median())
    m = max(min(m, b), a)
    return a, m, b

def sample_from_fit(dist_name: str, params: dict, size: int, rng: np.random.Generator):
    shapes = params.get('shapes', ())
    loc = params.get('loc', 0.0)
    scale = params.get('scale', 1.0)
    dist_map = {
        'norm': st_stats.norm,
        'lognorm': st_stats.lognorm,
        'beta': st_stats.beta,
        'expon': st_stats.expon,
        'gamma': st_stats.gamma,
        'weibull_min': st_stats.weibull_min,
    }
    if dist_name not in dist_map:
        raise ValueError("Distribución no soportada")
    dist = dist_map[dist_name]
    return dist.rvs(*shapes, loc=loc, scale=scale, size=size, random_state=rng)

def muestrear_variable(x: pd.Series, dist_name: str, params: dict, size: int, rng: np.random.Generator):
    try:
        m = sample_from_fit(dist_name, params, size, rng)
        return m, dist_name
    except Exception:
        a, mo, b = triangular_params(x)
        m = rng.triangular(a, mo, b, size)
        return m, "triangular"

def vista_ej2():
    st.subheader("Ejercicio 2 — Publicidad (Regresión + BestFit + Montecarlo)")
    st.caption("Se usa BestFit para TV/Radio/Newspaper; si falla, Triangular (min, moda≈mediana, máx).")

    # Carga de datos
    st.write("Carga automática de `Advertising.csv` (debe estar junto a este archivo).")
    try:
        csv_path = pathlib.Path(__file__).parent / "Advertising.csv"
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"No se pudo leer Advertising.csv: {e}")
        return

    # Regresión
    modelo = ajustar_regresion(df)
    coefs = dict(zip(["TV","Radio","Newspaper"], modelo.coef_))
    inter = float(modelo.intercept_)
    st.code(
        f"Sales = {inter:.4f} + {coefs['TV']:.4f}*TV + {coefs['Radio']:.4f}*Radio + {coefs['Newspaper']:.4f}*Newspaper",
        language="text"
    )

    # BestFit por variable
    dist_tv, params_tv = bestfit_variable(df["TV"])
    dist_ra, params_ra = bestfit_variable(df["Radio"])
    dist_ne, params_ne = bestfit_variable(df["Newspaper"])

    trials = st.number_input("Simulaciones (mín. 1,000)", min_value=1000, value=1000, step=100)
    seed = st.number_input("Semilla aleatoria", min_value=0, value=123, step=1)
    ejecutar = st.button("Ejecutar simulación")

    if not ejecutar:
        return

    rng = np.random.default_rng(seed)
    tv_sim, used_tv = muestrear_variable(df["TV"], dist_tv, params_tv, trials, rng)
    ra_sim, used_ra = muestrear_variable(df["Radio"], dist_ra, params_ra, trials, rng)
    ne_sim, used_ne = muestrear_variable(df["Newspaper"], dist_ne, params_ne, trials, rng)

    Xsim = np.column_stack([tv_sim, ra_sim, ne_sim])
    ventas_pred = modelo.predict(Xsim)

    idx = int(np.argmax(ventas_pred))
    tv_opt, ra_opt, ne_opt = float(tv_sim[idx]), float(ra_sim[idx]), float(ne_sim[idx])
    ventas_opt = float(ventas_pred[idx])

    total = tv_opt + ra_opt + ne_opt
    pct_tv = tv_opt/total if total > 0 else 0.0
    pct_ra = ra_opt/total if total > 0 else 0.0
    pct_ne = ne_opt/total if total > 0 else 0.0

    st.write("**Distribuciones utilizadas para muestreo**")
    st.write({
        "TV": used_tv,
        "Radio": used_ra,
        "Newspaper": used_ne
    })

    st.write("**Resultado óptimo (ventas máximas predichas)**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ventas estimadas", f"{ventas_opt:.3f}")
    col2.metric("TV óptimo", f"{tv_opt:.2f}")
    col3.metric("Radio óptimo", f"{ra_opt:.2f}")
    col4.metric("Periódico óptimo", f"{ne_opt:.2f}")

    st.write("**Porcentajes normalizados**")
    tabla = pd.DataFrame({
        "Canal": ["TV","Radio","Newspaper"],
        "Inversión óptima": [tv_opt, ra_opt, ne_opt],
        "Porcentaje": [pct_tv, pct_ra, pct_ne]
    })
    st.dataframe(tabla.style.format({"Inversión óptima": "{:.2f}", "Porcentaje": "{:.2%}"}))

    # (Opcional) Histograma simple de ventas predichas
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ventas_pred, bins=30)
    ax.set_title("Distribución de ventas predichas (simulación)")
    ax.set_xlabel("Ventas predichas")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)


# =========================
# LANDING PAGE
# =========================
st.set_page_config(page_title="Lab 2 - Montecarlo", layout="centered")
st.title("Laboratorio 2 — Simulación de Montecarlo")

eleccion = st.radio(
    "Elige el ejercicio para simular:",
    ("—", "Ejercicio 1: Ahorro Universitario", "Ejercicio 2: Publicidad")
)

if eleccion == "Ejercicio 1: Ahorro Universitario":
    vista_ej1()
elif eleccion == "Ejercicio 2: Publicidad":
    vista_ej2()
else:
    st.markdown(
        "Bienvenido 👋\n\n"
        "Este aplicativo te permite ejecutar las simulaciones del **Laboratorio 2**:\n\n"
        "1) **Ahorro Universitario:** Montecarlo con aportes anuales y rendimiento normal.\n"
        "2) **Publicidad:** Regresión `Sales ~ TV + Radio + Newspaper`, BestFit por canal y simulación para encontrar la mejor combinación."
    )