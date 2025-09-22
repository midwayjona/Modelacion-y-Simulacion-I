# Programa de Cosecha (Knapsack Binario con Doble Restricción)

Este proyecto implementa un **Programa de Cosecha** de caña de azúcar modelado como un **knapsack binario** con **dos restricciones**: horas totales de trabajo y capacidad total de transporte. El objetivo es **maximizar los kilogramos recolectados**. Para visualizar la convergencia de las metaheurísticas como una curva que **disminuye**, optimizamos un **costo penalizado** (minimización) que incluye penalizaciones por violaciones de horas o capacidad.

Se incluyen **tres metaheurísticas**: **Algoritmo Genético (GA)**, **Recocido Simulado (SA)** y **Búsqueda Tabú (TS)**. El flujo de trabajo se muestra tanto en una **notebook de Python** como en una **aplicación Streamlit**.

---

## Demo en línea
- **App en Streamlit:** https://proyecto-de-cosecha.streamlit.app

## Demo en línea
- **Notebook:** `harvest_metaheuristics.ipynb`](./notebooks/harvest_metaheuristics.ipynb)

---

## Modelo (resumen)

- Conjunto de parcelas \( i=1..N \).
- Variables: \( V_i \) (kg), \( P_i \) (horas), \( X_i \in \{0,1\} \).
- Parámetros: \( W = \text{workers} \times 40 \), \( K \) vehículos, \( C \) kg por vehículo.

**Objetivo:**

$$
\max\ Z \;=\; \sum_{i=1}^{N} V_i\,X_i
$$

**Restricciones:**

$$
\sum_{i=1}^{N} P_i\,X_i \;\le\; W
\qquad\text{y}\qquad
\sum_{i=1}^{N} V_i\,X_i \;\le\; K\,C
$$

**Dominio:**

$$
X_i \in \{0,1\}\quad \forall i
$$

**Costo penalizado (a minimizar para graficar):**

$$
\text{cost}(X) \;=\; -Z(X)\;+\;\lambda_h\,\max\!\Big(0,\ \textstyle\sum_i P_i X_i - W\Big)\;+\;\lambda_c\,\max\!\Big(0,\ \textstyle\sum_i V_i X_i - K C\Big).
$$

> Nota: se mantiene la formulación binaria global (capacidad total \(K\,C\)). Extensiones a **multiple knapsack** o **múltiples viajes** se comentan en el código pero **no** se implementan.

---

## Instalación y ejecución

### 1) Crear y activar entorno
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2) Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3) Ejecutar la Notebook
Abra `notebooks/harvest_metaheuristics.ipynb` y ejecute todas las celdas.

### 4) Ejecutar la App
```bash
streamlit run app/streamlit_app.py
```

---

## Parámetros y datos
- \( N \): número de parcelas.
- \( V_i \): kilogramos por parcela, enteros en \([100,1200]\).
- \( P_i \): horas por parcela, reales en \([1.0,10.0]\) redondeadas a 2 decimales.
- \( W = \text{workers} \times 40.0 \).
- \( K \), \( C \): vehículos y capacidad por vehículo. Capacidad total: \(K \times C\).
- Valores por defecto en la app con validaciones: \(N\in[20,200]\), \(C\in[1000,2500]\), \(K\ge 1\), \(\text{workers}\ge 1\).

La generación de datos es **determinista** y **reproducible** vía `seed`.

---

## Uso (app)
1. Ajuste `seed`, `N`, `workers`, `K`, `C` en la barra lateral (con rangos validados).
2. Elija algoritmo (**GA**, **SA**, **TS**) y configure sus hiperparámetros.
3. Pulse **Resolver**.
4. Interprete el gráfico: el **costo penalizado** debe **disminuir** con las iteraciones.
5. Descargue el CSV con la solución desde cada pestaña.

---

## Reproducibilidad
El generador usa `numpy.random.default_rng(seed)`. Con la misma `seed` y parámetros, obtendrá los mismos datos y resultados (sujeto a la naturaleza estocástica de los algoritmos).

---

## Estructura del repositorio
```
project-root/
├─ notebooks/
│  └─ harvest_metaheuristics.ipynb
├─ app/
│  └─ streamlit_app.py
├─ core/
│  ├─ model.py
│  ├─ ga.py
│  ├─ sa.py
│  ├─ tabu.py
│  └─ utils.py
├─ artifacts/
│  └─ .gitkeep
├─ tests/
│  └─ test_model.py
├─ README.md
└─ requirements.txt
```

---

## Créditos
- **Jonathan Amado – 14002284**
