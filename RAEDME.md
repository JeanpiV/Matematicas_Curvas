# Visualización y Cálculo del Movimiento en Curvas (Matemáticas III)

## 📌 Resumen y Objetivo General
Este proyecto implementa un algoritmo en **Python** para analizar el movimiento cinemático y la geometría diferencial de curvas en el espacio 3D. El objetivo es comprobar la comprensión teórica mediante la programación de cálculos vectoriales y su visualización interactiva.

El sistema permite:
1. **Calcular vectores cinemáticos:** Posición r(t), Velocidad v(t) y Aceleración a(t).
2. **Construir el Triedro de Frenet-Serret:** Vectores Tangente (T), Normal (N) y Binormal (B).
3. **Calcular Curvatura y Torsión:** Implementación numérica de las fórmulas \kappa(t) y \tau(t).
4. **Validación Numérica:** Comparación de derivadas analíticas (exactas) vs. numéricas (diferencias finitas).
5. **Visualización:** Gráficas 3D interactivas, animación del recorrido y gráficas 2D de las propiedades escalares.

---

## ⚙️ Requisitos del Sistema

* **Lenguaje:** Python 3.8 o superior.
* **Librerías:** * `numpy`: Para cálculos matriciales y álgebra lineal.
  * `matplotlib`: Para la generación de gráficas 2D/3D y animación.
  * `pandas`: Para la tabulación y presentación de datos.

---

## 🚀 Instrucciones de Instalación y Ejecución

Sigue estos pasos en tu terminal para configurar el entorno correctamente:

### 1. Configuración del Entorno (Recomendado)
Para evitar conflictos con otras librerías, crea y activa un entorno virtual.

**En Windows (PowerShell/CMD):**
```bash
python -m venv venv
.\venv\Scripts\activate
```
### 2. Activar el Entorno
* #### En Windows
```bash
.\venv\Scripts
```
(Deberás ver un *`(venv)`* al inicio de tu linea de comandos).

### 3. Instalar Dependencias

Una vez instalado todo, corre el script principal:

```bash
python proyecto_curvas.py
```
## 🛠️ Configuración y Modificación de la Curva

El código viene configurado por defecto con una **Hélice Circular**, ideal para validación por tener curvatura y torsión constantes.

### Modificar Parámetros de la Hélice

En el archivo `proyecto_curvas.py`, busca la sección `--- CONFIGURACIÓN DE PARÁMETROS ---`:

* `a_val`: Radio de la hélice (Defecto; 2).
* `b_val`: Paso vertical de la hélice (Defecto:  0.5).
* `t_final`: Intervalo de tiempo (Defecto: 4*pi, dos vueltas).

### cambiar la Curva (Avanzado)

El algoritmo funciona para cualquier curva paramétrica r(t). Para cambiarla, edita las funciones en la sección `--- DEFINICIÓN DE LA CURVA ---`:

1. `obtener_r(t)`: Define la nueva ecuación vectorial (x(t), y(t), z(t)).
2. `obtener_v_analitica(t)`: Define la primera derivada exacta para validación.
3. `obtener_a_analatica(t)`: Define la segunda derivada exacta.

> **Nota:** Si cambias la curva, asegúrate de actualizar también las derivadas analíticas para que el cálculo ddel procentaje de error sea correcto.

---

## 📊 Resultados Esperados
Al ejecutar el programa, se generarán:

1. **Ventana Gráfica:**
    * **Panel Izquierdo (3D):** Muestra
la trayectoria de la hélice. Un punto rojo animado recorre la curva. En puntos clave, se muestra el Triedro de Frenet (Rojo=Tangente, Verde=Normal, Azul=Binormal).

    * **Panel Derecho(2D):** Muestra la evolución de la Curvatura y Torsión. Para la hélice, estas deben ser lineas horizontales (constantes).

2. ** Consola (Terminal):**
    * **Error Medio:** Confirmación de que la derivada numérica es precisa (valor cercano a 0).

    * **Prueba de Ortoganilidad:** Confirmación de que *$T \cdot N \approx 0$*

    * **Tabla de Coordenadas:** Conversión de puntos clave a sistemas Cilindricos y Esferico.