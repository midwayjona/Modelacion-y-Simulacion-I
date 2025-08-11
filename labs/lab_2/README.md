# Laboratorio 2 – Simulación de Montecarlo  
Curso: **Modelación y Simulación I – Maestría en Investigación de Operaciones**  
Universidad Galileo – FISICC  

**Notebook del laboratorio:** [`lab_2.ipynb`](./lab_2.ipynb)  
**Archivo de datos:** [`Advertising.csv`](./Advertising.csv)  

---

## Descripción  
Este laboratorio tiene como objetivo aplicar técnicas de **Simulación de Montecarlo** para resolver dos problemas diferentes:  
1. **Planeación de ahorro a largo plazo con rendimientos variables**.  
2. **Optimización del presupuesto de publicidad para maximizar ventas**.  

El laboratorio se implementará en **Python (Jupyter Notebook)** y también deberá contar con una **aplicación en Streamlit** para interactuar con las simulaciones y visualizar los resultados.

---

## Ejercicio 1 – Ahorro Universitario con Rendimiento Variable  
Juan y Juana Doe desean ahorrar $500,000 para la universidad de su hijo en 17 años.  
- Aportarán $20,000 al inicio de cada año.  
- El rendimiento anual sigue una distribución normal con media 4% y desviación estándar 10%.  
- Cada intento de simulación generará 17 tasas de rendimiento distintas (una por año).  

**Tareas a realizar:**  
- Ejecutar la simulación de Montecarlo con al menos 1,000 intentos.  
- Calcular:  
  - Rendimiento promedio total al final de los 17 años.  
  - Monto promedio acumulado al final de los 17 años.  
  - Escenarios pesimista y optimista.  
- Graficar:  
  - Rendimientos obtenidos por año.  
  - Monto ahorrado por año.  
  - Monto acumulado por año.

---

## Ejercicio 2 – Optimización de Presupuesto Publicitario  
La empresa **ABC Corp.** invierte en publicidad a través de **TV**, **Radio** y **Periódico**.  
Se busca encontrar la combinación óptima de presupuesto para maximizar las ventas.

**Tareas a realizar:**  
- Construir un **modelo de regresión lineal**:  
  - Variable dependiente: `Sales`.  
  - Variables independientes: `TV`, `Newspaper`, `Radio`.  
- Determinar la **distribución de probabilidad** de `TV`, `Radio` y `Newspaper` usando *bestfit*;  
  - Si no es posible, asumir distribución **Triangular** (mínimo, máximo, más probable).  
- Generar números aleatorios para cada variable y validar la distribución.  
- Realizar una **simulación de Montecarlo** para estimar los valores de inversión que maximizan las ventas.  
- Determinar el presupuesto óptimo de cada tipo de publicidad en porcentaje.

---

## Requerimientos Técnicos  
- Implementación en **Python (Jupyter Notebook)**.  
- Aplicación interactiva en **Streamlit** para correr la simulación y mostrar los resultados.  
- Graficar y reportar todos los resultados solicitados.  

---

## Archivos incluidos  
- `lab_2.ipynb` – Notebook con el desarrollo del laboratorio.  
- `Advertising.csv` – Datos de inversión en publicidad y ventas.  
- `Lab_2.pdf` – Guía oficial del laboratorio.

---
**Autor:** *Jonathan Amado [14002285]* – Maestría en Investigación de Operaciones – Universidad Galileo