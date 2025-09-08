# Proyecto 1 – Simulación del Juego de la Perinola  

Curso: **Modelación y Simulación I – Maestría en Investigación de Operaciones**  
Universidad Galileo – FISICC  

---

## Descripción  

Este proyecto implementa una simulación del **juego de la perinola**, un juego de azar en el que cada tirada determina la acción de los jugadores sobre un pozo común de dinero.  

### Resultados de la perinola:  
- **Pon 1**  
- **Pon 2**  
- **Toma 1**  
- **Toma 2**  
- **Toma todo**  
- **Todos ponen**  

### Reglas básicas:  
- Cada jugador inicia con la misma cantidad de dinero.  
- Existe un **pozo común** donde se realizan aportes y del cual se pueden tomar ganancias.  
- El juego termina cuando solo queda un jugador con dinero.  

---

## Simulación  

La simulación considera:  
- **N jugadores**.  
- **M juegos** (número de repeticiones).  
- Registro de las **ganancias y pérdidas por jugador** en cada simulación.  

---

## Preguntas a responder  

Con los resultados de la simulación se busca responder:  

1. ¿Cuántos juegos son necesarios para que un jugador se quede sin dinero?  
2. ¿En cuántos juegos, en promedio, hay un ganador definido?  
3. ¿Cómo afecta el número de jugadores al número de juegos para que un jugador se gane todo el dinero?  
4. ¿Cuál es la gráfica de ganancias y pérdidas por jugador al término de la simulación?  

---

## Entregables  

1. **Notebook:** [`perinola.ipynb`](./perinola.ipynb)  
   - Implementa la simulación del juego con parámetros configurables.  
   - Incluye gráficas de ganancias/pérdidas y análisis de los resultados.  

2. **Aplicación de Streamlit:** [`app.py`](./app.py)  
   - Interfaz interactiva que permite configurar:  
     - Número de jugadores.  
     - Número de juegos (iteraciones).  
     - Monto inicial de cada jugador.  
   - Al ejecutar, la aplicación muestra:  
     - Evolución de la simulación.  
     - Gráficas por jugador de ganancia y pérdida.  
     - Estadísticas clave para responder las preguntas planteadas.  

---

## Ejecución de la aplicación  

1. Instalar dependencias:
   ```bash
   pip install streamlit numpy pandas matplotlib seaborn