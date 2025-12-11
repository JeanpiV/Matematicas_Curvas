import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.animation import FuncAnimation

# --- CONFIGURACIÓN DE PARÁMETROS ---
t_inicio, t_final = 0, 4 * np.pi  # Dos vueltas completas
n_puntos = 100
a_val = 2  # Radio de la hélice
b_val = 0.5  # Paso vertical

# --- DEFINICIÓN DE LA CURVA (ANALÍTICA) ---
def obtener_r(t, a=a_val, b=b_val):
    """Retorna la posición r(t) = (x, y, z)."""
    return np.array([a * np.cos(t), a * np.sin(t), b * t])

def obtener_v_analitica(t, a=a_val, b=b_val):
    """Derivada exacta (a mano): v(t) = (-a sin t, a cos t, b)."""
    return np.array([-a * np.sin(t), a * np.cos(t), np.full_like(t, b)])

def obtener_a_analitica(t, a=a_val, b=b_val):
    """Segunda derivada exacta: a(t) = (-a cos t, -a sin t, 0)."""
    return np.array([-a * np.cos(t), -a * np.sin(t), np.zeros_like(t)])

# --- CÁLCULOS NUMÉRICOS Y GEOMETRÍA ---
def calcular_derivada_numerica(valores, dt):
    """Calcula derivada usando diferencias centrales (como pide el PDF)."""
    return np.gradient(valores, dt, axis=1)

def calcular_triedro_curvatura(v, a_vec):
    """
    Calcula T, N, B, Curvatura y Torsión.
    v: vector velocidad (3, n)
    a_vec: vector aceleración (3, n)
    """
    # Norma de la velocidad ||v||
    norm_v = np.linalg.norm(v, axis=0)
    
    # 1. Vector Tangente: T = v / ||v||
    T = v / norm_v

    # Derivada de T para hallar N (Numéricamente)
    dT = np.gradient(T, axis=1)
    norm_dT = np.linalg.norm(dT, axis=0)
    
    # 2. Vector Normal: N = dT / ||dT||
    # Evitamos división por cero con un pequeño epsilon
    N = dT / (norm_dT + 1e-9) 

    # 3. Vector Binormal: B = T x N
    B = np.cross(T.T, N.T).T

    # Producto cruz v x a
    v_cross_a = np.cross(v.T, a_vec.T).T
    norm_v_cross_a = np.linalg.norm(v_cross_a, axis=0)

    # 4. Curvatura: k = ||v x a|| / ||v||^3
    kappa = norm_v_cross_a / (norm_v**3)

    # Para la torsión necesitamos la derivada de la aceleración (Jerk)
    # tau = ((v x a) . a') / ||v x a||^2
    jerk = np.gradient(a_vec, axis=1) # a_prima numérica
    dot_prod = np.sum(v_cross_a * jerk, axis=0)
    tau = dot_prod / (norm_v_cross_a**2)

    return T, N, B, kappa, tau

# --- TRANSFORMACIÓN DE COORDENADAS ---
def cartesianas_a_otras(x, y, z):
    # Cilíndricas (rho, phi, z)
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Esféricas (r, theta, phi) - Convención física
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / (r + 1e-9)) # Ángulo polar
    
    return (rho, phi, z), (r, theta, phi)

# --- EJECUCIÓN PRINCIPAL ---
# 1. Generar datos
t = np.linspace(t_inicio, t_final, n_puntos)
dt = t[1] - t[0]

r = obtener_r(t)
x, y, z = r

# 2. Derivadas
v_ana = obtener_v_analitica(t)
a_ana = obtener_a_analitica(t)

v_num = calcular_derivada_numerica(r, dt)
a_num = calcular_derivada_numerica(v_num, dt)

# 3. Calcular Geometría (Usamos la numérica para probar el algoritmo)
T, N, B, kappa, tau = calcular_triedro_curvatura(v_num, a_num)

# --- VALIDACIÓN (PARTE D - Pruebas) ---
error_v = np.mean(np.linalg.norm(v_ana - v_num, axis=0))
print(f"Error medio entre Velocidad Analítica y Numérica: {error_v:.6f}")

# Ortogonalidad T . N debe ser 0
ortogonalidad = np.mean(np.abs(np.sum(T * N, axis=0)))
print(f"Prueba de Ortogonalidad (T . N medio): {ortogonalidad:.6f} (Debe ser cercano a 0)")

# --- VISUALIZACIÓN (PARTE C) ---
fig = plt.figure(figsize=(14, 6))

# Gráfica 1: Trayectoria 3D con Triedro
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x, y, z, label='Hélice', color='blue')

# --- ANIMACIÓN (Requisito del Informe del profesor : Recorriendo la curva) ---
# Creamos un punto rojo que se moverá
punto_anim, = ax1.plot([], [], [], 'ro', markersize=8, label='Punto P(t)')

def init():
    punto_anim.set_data([], [])
    punto_anim.set_3d_properties([])
    return punto_anim,

def update(frame):
    # frame es el índice del punto actual
    punto_anim.set_data([x[frame]], [y[frame]])     # Actualiza X e Y
    punto_anim.set_3d_properties([z[frame]])        # Actualiza Z
    return punto_anim,

# Crear la animación
# frames=n_puntos: cuántos pasos tiene la animación
# interval=50: milisegundos entre cada paso (velocidad)
anim = FuncAnimation(fig, update, frames=n_puntos, init_func=init, interval=50, blit=False)

#--- culminacion de la ANIMMACION --- SI ALGUIEN SABE MEJOR ANIMAR QUE CAMBIE EL CODIGO PLS

# Dibujar vectores T, N, B en 4 puntos específicos
indices = [0, 33, 66, 99] # Puntos seleccionados
colors = ['r', 'g', 'b'] # T=Rojo, N=Verde, B=Azul
labels = ['T', 'N', 'B']

for i in indices:
    # Origen del vector
    ox, oy, oz = x[i], y[i], z[i]
    # Vectores (escalados para que se vean bien)
    vecs = [T[:, i], N[:, i], B[:, i]]
    for v_vec, c, l in zip(vecs, colors, labels):
        ax1.quiver(ox, oy, oz, v_vec[0], v_vec[1], v_vec[2], length=1.0, color=c, normalize=True)

ax1.set_title('Trayectoria y Triedro de Frenet')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.legend()

# Gráfica 2: Curvatura y Torsión
ax2 = fig.add_subplot(122)
ax2.plot(t, kappa, label=r'Curvatura ($\kappa$)', color='purple')
ax2.plot(t, tau, label='Torsión ($\\tau$)', color='orange')
ax2.set_title('Curvatura y Torsión vs Tiempo')
ax2.set_ylim(0, max(np.max(kappa), np.max(tau)) * 1.5)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# --- TABLA COMPARATIVA (PARTE C.4) ---
puntos_idx = [0, n_puntos//2, -1] # Inicio, medio, fin
data = []
for i in puntos_idx:
    xi, yi, zi = x[i], y[i], z[i]
    (rho, phi, _), (r_sf, theta, _) = cartesianas_a_otras(xi, yi, zi)
    data.append({
        "t": t[i],
        "Cart (x,y,z)": f"({xi:.2f}, {yi:.2f}, {zi:.2f})",
        "Cil (rho,phi,z)": f"({rho:.2f}, {phi:.2f}, {zi:.2f})",
        "Esf (r,theta,phi)": f"({r_sf:.2f}, {theta:.2f}, {phi:.2f})"
    })

df = pd.DataFrame(data)
print("\n--- TABLA DE COORDENADAS ---")
print(df)
#agregar cuadrica para un triedro. NOTA PERSONAL