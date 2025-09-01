# Laboratorio 3 – Problema del Vendedor de Periódicos (Newsvendor Problem)

Curso: **Modelación y Simulación I – Maestría en Investigación de Operaciones**  
Universidad Galileo – FISICC  

---

**Ejecuta las simulaciones en línea:**  
[**App de Streamlit – Laboratorio 3**](https://newsvendor-problem.streamlit.app)  

## Descripción  
Este laboratorio aborda el clásico **problema del vendedor de periódicos (Newsvendor Problem)**, que consiste en determinar la cantidad óptima de periódicos a ofrecer para maximizar la utilidad esperada considerando:

- **Precio de venta**
- **Precio de compra**
- **Precio de reciclaje**
- **Demanda aleatoria**

La resolución se hace mediante **Simulación de Montecarlo** para distintos valores de oferta, evaluando las ganancias en múltiples escenarios aleatorios.

---

## Implementación  

El laboratorio cuenta con dos componentes:

1. **Notebook:** [`tsp.ipynb`](./tsp.ipynb)  
   - Desarrollo paso a paso de la simulación en Python. (Resuelto en clase).
   - Permite entender la lógica y validar resultados.  

2. **Aplicación de Streamlit:** [`app.py`](./app.py)  
   - Interfaz interactiva donde el usuario puede:  
     - Definir parámetros de entrada:  
       - Cantidad de simulaciones  
       - Precio de venta  
       - Precio de compra  
       - Precio de reciclaje  
   - Resultados mostrados:  
     - **Gráfico de línea** con el valor promedio de las ganancias para cada cantidad de oferta.  
     - **Gráfico de violín** mostrando la distribución de probabilidad de ganancias para cada caso de oferta.
