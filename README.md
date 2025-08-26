# Optimización de Resonadores Acústicos

## 📌 Descripción General
Este software está diseñado para analizar y optimizar la geometría de resonadores acústicos, como los de las flautas, con el objetivo de evaluar y minimizar su **inarmonicidad**. El sistema simula cómo se comportaría el instrumento al acortar su longitud efectiva, un efecto análogo al de abrir agujeros de digitación.

El proyecto combina un motor de cálculo de impedancia acústica (basado en [OpenWind](https://github.com/openwind)) con una **interfaz gráfica interactiva (PyQt6)** y herramientas de **optimización por línea de comandos** para una exploración exhaustiva del espacio de diseño.

---

## ⚙️ Componentes Principales

### 1. `resonador_armonico.py`
- **Aplicación Principal con GUI (PyQt6)**.
- Permite la definición interactiva de la geometría del instrumento (cilindros, conos, embocadura).
- Organizada en pestañas para diferentes análisis:
  - **Geometría y Análisis de Impedancia:** Visualización del diseño y de la curva de impedancia.
  - **Escaneo por Longitud:** Ejecuta un escaneo no bloqueante que acorta progresivamente el tubo, generando gráficos de inarmonicidad (Δ2, Δ3, etc.) en función de la frecuencia fundamental. Los resultados se pueden exportar a CSV y PNG.
  - **Optimización ("Curva Azul"):** Lanza un optimizador en segundo plano para encontrar la geometría que minimiza la inarmonicidad del segundo armónico (la "curva azul"). Muestra el progreso en tiempo real y una visualización en vivo de la geometría y la curva de inharmonicidad del mejor candidato encontrado hasta el momento.

### 2. `ui_builders.py`
- Módulo auxiliar que contiene funciones para construir los distintos componentes de la interfaz gráfica (paneles, pestañas, widgets), manteniendo el código de la GUI modular y organizado.

### 3. `workers.py`
- Implementa los workers (`ScanWorker`, `BlueOptWorker`) que se ejecutan en hilos separados (`QThread`).
- Esto permite que las tareas computacionalmente intensivas, como el escaneo y la optimización, se realicen en segundo plano sin congelar la interfaz de usuario, proporcionando una experiencia fluida.

---

### 2. `escaneo_longitud.py`
- Contiene la lógica central para el análisis acústico.
- Proporciona la función `escanear_por_longitud`, que simula las resonancias de un tubo a diferentes longitudes efectivas y calcula las inarmonicidades. Es utilizado tanto por la GUI interactiva como por los workers de optimización.

---

### 3. `analisis_cvs_optimizacion.py`
- **Visualizador Interactivo de Resultados.**
- Es una aplicación PyQt separada que carga un archivo CSV (típicamente generado por un proceso de optimización) y muestra un gráfico de dispersión del espacio de búsqueda.
- Al hacer clic en un punto (una simulación), se muestra un gráfico de barras con la inarmonicidad de esa configuración específica, facilitando el análisis post-mortem de los resultados.

---

### 4. `configurando_gui.py`
- Una utilidad de GUI independiente para **crear archivos de configuración JSON** para experimentos de optimización. Facilita la definición de rangos de parámetros y pesos para la función de costo.

---

### 5. `bayes_explorer.py`
- **Herramienta de Optimización por Línea de Comandos.**
- Implementa un explorador de parámetros basado en **Optimización Bayesiana** (usando `scikit-optimize`).
- Toma un archivo de configuración JSON como entrada para definir el espacio de búsqueda, los objetivos y los parámetros de ejecución.
- Es ideal para búsquedas automáticas y sistemáticas que pueden correr durante horas, guardando un registro detallado (`trace.jsonl`) y la mejor configuración encontrada.

---

## 🚀 Flujos de Trabajo

Existen dos flujos de trabajo principales para utilizar este software:

### A. Flujo Interactivo (GUI)

1.  **Ejecutar la aplicación principal:**
    ```bash
    python resonador_armonico.py
    ```
2.  **Definir una geometría inicial** en el panel izquierdo de la GUI.
3.  **Usar la pestaña "Escaneo por Longitud"** para analizar la inarmonicidad de la geometría actual.
4.  **Usar la pestaña "Optimización"** para que el software busque automáticamente una mejor geometría. Se puede observar el proceso en vivo.
5.  **Exportar** los resultados de interés (curvas, geometrías) a archivos CSV/PNG.

### B. Flujo por Línea de Comandos (Optimización Bayesiana)

1.  **Crear un archivo de configuración JSON** (p. ej., `bayes_config.json`). Este archivo define las variables a optimizar, sus rangos (`space`), los parámetros del modelo y los pesos de la función objetivo.
    ```json
    {
      "meta": { "name": "traverso_cone_search" },
      "model": {
        "L_min_frac": 0.5,
        "scan_steps": 10,
        "temperature_C": 20.0
      },
      "space": {
        "L_total": [0.5, 0.7],
        "r1_longitud": [0.2, 0.4],
        "r1_radio": [0.008, 0.012],
        "radio_cono_final": [0.004, 0.008],
        "agujero_posicion": [0.01, 0.05],
        "agujero_radio_in": [0.004, 0.006],
        "agujero_radio_out": [0.004, 0.006],
        "agujero_largo": [0.001, 0.004]
      },
      "objective": {
        "weights": { "w_mag": 1.0, "w_var": 0.5, "w_max": 0.5, "w_miss": 50.0 }
      },
      "runtime": {
        "output_dir": "bayes_results",
        "n_calls": 200,
        "random_starts": 20,
        "seed": 42
      }
    }
    ```
2.  **Lanzar el explorador bayesiano** desde la terminal:
    ```bash
    python bayes_explorer.py bayes_config.json
    ```
3.  El script ejecutará el número de evaluaciones especificadas (`n_calls`) y guardará los resultados en el directorio de salida (`output_dir`). Esto incluye un `trace.jsonl` con cada evaluación y un `best.json` con el mejor resultado.
4.  **Analizar los resultados** posteriormente, por ejemplo, cargando el `trace.jsonl` en un script de análisis personalizado o usando el `analisis_cvs_optimizacion.py` (requiere convertir el JSONL a CSV primero).

---

## 🛠️ Tecnologías Usadas

- **Python 3.10+**
- **PyQt6**: Para las interfaces gráficas.
- **Matplotlib**: Para la visualización de datos.
- **NumPy / SciPy**: Para cálculo numérico y optimización local (Powell).
- **scikit-optimize**: Para la Optimización Bayesiana.
- **OpenWind**: Librería base para la simulación de acústica de resonadores.

---

## 📥 Instalación

```bash
# Crear y activar un entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

`requirements.txt` incluye:
```
pyqt5
matplotlib
numpy
scipy
```

(OpenWind debe estar instalado por separado).

---

## ▶️ Uso

```bash
python lanzador_final.py
```

La GUI se abrirá con todas las pestañas.

---

## ✨ Créditos
- Desarrollo: Patricio de la Cuadra  
- Asistencia en software: ChatGPT (OpenAI)  
- Basado en librería **OpenWind**

---

## 📌 Futuras Extensiones
- Inclusión de agujeros reales (posición, diámetro).
- Optimización multi-objetivo (no solo f₂/f₁).
- Algoritmos de optimización más avanzados (genéticos, bayesianos).
- Exportación directa a formatos CAD para prototipado.
- Se planea un sistema de exploración exhaustiva de parámetros definido por JSON con guardado automático de resultados para análisis posterior.
