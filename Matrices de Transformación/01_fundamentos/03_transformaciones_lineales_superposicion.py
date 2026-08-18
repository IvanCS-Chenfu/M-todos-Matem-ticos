from pathlib import Path
import sys

import numpy as np


# Permite importar módulos desde la carpeta Matrices_de_Transformacion/
CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def formatear_vector(vector):
    """Devuelve un vector 2D con formato compacto."""

    vector = np.asarray(vector, dtype=float)
    return f"[{vector[0]:6.2f}, {vector[1]:6.2f}]"


def interpolar(inicio, fin, cantidad):
    """Genera valores entre inicio y fin, incluyendo ambos extremos."""

    return np.linspace(inicio, fin, cantidad)


def suavizar(progreso):
    """Interpolación cosenoidal para movimientos suaves."""

    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def crear_cuadricula(matriz, limite=3.0, divisiones=7, muestras=61):
    """Calcula una cuadrícula 2D transformada por la matriz indicada."""

    matriz = np.asarray(matriz, dtype=float)
    valores = np.linspace(-limite, limite, divisiones)
    parametro = np.linspace(-limite, limite, muestras)
    lineas = []

    for valor in valores:
        vertical = np.column_stack((np.full_like(parametro, valor), parametro))
        horizontal = np.column_stack((parametro, np.full_like(parametro, valor)))
        lineas.extend([
            (matriz @ vertical.T).T,
            (matriz @ horizontal.T).T,
        ])

    return lineas


def crear_estado_aditividad(matriz_actual, matriz_final, u, v, fase, mensaje):
    """Visualiza la igualdad A(u+v) = Au + Av."""

    matriz_actual = np.asarray(matriz_actual, dtype=float)
    matriz_final = np.asarray(matriz_final, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    u_t = matriz_actual @ u
    v_t = matriz_actual @ v
    suma_t = matriz_actual @ (u + v)

    u_final = matriz_final @ u
    v_final = matriz_final @ v
    suma_final = matriz_final @ (u + v)

    vertices = np.array([
        [0.0, 0.0],
        u_t,
        u_t + v_t,
        v_t,
    ])

    return {
        "polygons": [
            {
                "points": vertices,
                "facecolor": "#DCEAF7",
                "edgecolor": "#7A9CC6",
                "alpha": 0.24,
                "linewidth": 1.4,
            },
        ],
        "vectors": [
            {
                "name": "Au",
                "origin": np.zeros(2),
                "value": u_t,
                "color": "#B23A48",
                "linewidth": 3.0,
            },
            {
                "name": "Av",
                "origin": np.zeros(2),
                "value": v_t,
                "color": "#2D7F5E",
                "linewidth": 3.0,
            },
            {
                "name": "Av (trasladado)",
                "origin": u_t,
                "value": v_t,
                "color": "#2D7F5E",
                "alpha": 0.75,
                "linewidth": 2.3,
                "linestyle": "--",
                "label_offset": (0.12, -0.20),
            },
            {
                "name": "A(u+v)",
                "origin": np.zeros(2),
                "value": suma_t,
                "color": "#7B2CBF",
                "linewidth": 3.2,
            },
        ],
        "points": [
            {
                "name": "0",
                "position": np.zeros(2),
                "color": "#111827",
                "size": 65,
                "label_offset": (0.10, -0.20),
            },
        ],
        "message": mensaje,
        "info_title": "Principio de superposición",
        "info_lines": [
            {"text": "VECTORES ORIGINALES", "bold": True},
            f"u     = {formatear_vector(u)}",
            f"v     = {formatear_vector(v)}",
            f"u + v = {formatear_vector(u + v)}",
            "",
            {"text": "TRANSFORMACIÓN ACTUAL", "bold": True},
            f"Au       = {formatear_vector(u_t)}",
            f"Av       = {formatear_vector(v_t)}",
            f"A(u+v)   = {formatear_vector(suma_t)}",
            f"Au + Av  = {formatear_vector(u_t + v_t)}",
            "",
            {"text": "RESULTADO FINAL", "bold": True},
            f"A(u+v) = {formatear_vector(suma_final)}",
            f"Au+Av  = {formatear_vector(u_final + v_final)}",
        ],
        "phase": fase,
        "info_line_height": 0.046,
        "legend": [
            {"kind": "line", "label": "Au", "color": "#B23A48"},
            {"kind": "line", "label": "Av", "color": "#2D7F5E"},
            {"kind": "line", "label": "A(u+v)", "color": "#7B2CBF"},
        ],
        "legend_ncol": 3,
        "legend_fontsize": 8.3,
    }


def crear_estado_homogeneidad(matriz, u, lambda_actual, lambda_final, fase, mensaje):
    """Visualiza la igualdad A(lambda*u) = lambda*A(u)."""

    matriz = np.asarray(matriz, dtype=float)
    u = np.asarray(u, dtype=float)

    au = matriz @ u
    lambda_u = lambda_actual * u
    a_lambda_u = matriz @ lambda_u
    lambda_au = lambda_actual * au

    return {
        "vectors": [
            {
                "name": "u",
                "origin": np.zeros(2),
                "value": u,
                "color": "#6B7280",
                "alpha": 0.55,
                "linewidth": 2.0,
            },
            {
                "name": "Au",
                "origin": np.zeros(2),
                "value": au,
                "color": "#1F77B4",
                "alpha": 0.70,
                "linewidth": 2.5,
            },
            {
                "name": "λu",
                "origin": np.zeros(2),
                "value": lambda_u,
                "color": "#E07A1F",
                "linewidth": 3.0,
            },
            {
                "name": "A(λu)=λAu",
                "origin": np.zeros(2),
                "value": a_lambda_u,
                "color": "#7B2CBF",
                "linewidth": 3.2,
            },
        ],
        "message": mensaje,
        "info_title": "Homogeneidad",
        "info_lines": [
            {"text": "DATOS", "bold": True},
            f"u       = {formatear_vector(u)}",
            f"Au      = {formatear_vector(au)}",
            f"λ       = {lambda_actual:6.2f}",
            "",
            f"λu      = {formatear_vector(lambda_u)}",
            f"A(λu)   = {formatear_vector(a_lambda_u)}",
            f"λAu     = {formatear_vector(lambda_au)}",
            "",
            {"text": "COMPROBACIÓN", "bold": True},
            f"objetivo λ = {lambda_final:5.2f}",
            "A(λu) y λAu coinciden.",
        ],
        "phase": fase,
        "info_line_height": 0.050,
        "legend": [
            {"kind": "line", "label": "u", "color": "#6B7280"},
            {"kind": "line", "label": "λu", "color": "#E07A1F"},
            {"kind": "line", "label": "A(λu)", "color": "#7B2CBF"},
        ],
        "legend_ncol": 3,
        "legend_fontsize": 8.4,
    }


def crear_estado_origen(matriz_actual, matriz_final, fase, mensaje):
    """Muestra que una transformación lineal deforma el plano pero deja 0 fijo."""

    lineas_originales = crear_cuadricula(np.eye(2))
    lineas_transformadas = crear_cuadricula(matriz_actual)

    polylines = []
    for linea in lineas_originales:
        polylines.append({
            "points": linea,
            "color": "#9CA3AF",
            "alpha": 0.16,
            "linewidth": 0.9,
        })
    for linea in lineas_transformadas:
        polylines.append({
            "points": linea,
            "color": "#1F77B4",
            "alpha": 0.58,
            "linewidth": 1.1,
        })

    cero = np.zeros(2)
    cero_actual = matriz_actual @ cero

    return {
        "polylines": polylines,
        "points": [
            {
                "name": "A·0 = 0",
                "position": cero_actual,
                "color": "#111827",
                "size": 100,
                "label_offset": (0.12, -0.28),
            },
        ],
        "vectors": [
            {
                "name": "col 1",
                "origin": cero,
                "value": matriz_actual[:, 0],
                "color": "#B23A48",
                "linewidth": 2.6,
            },
            {
                "name": "col 2",
                "origin": cero,
                "value": matriz_actual[:, 1],
                "color": "#2D7F5E",
                "linewidth": 2.6,
            },
        ],
        "message": mensaje,
        "info_title": "El origen permanece fijo",
        "info_lines": [
            {"text": "PROPIEDAD", "bold": True},
            "f(p) = A p",
            "f(0) = A 0 = 0",
            "",
            f"A(t)·0 = {formatear_vector(cero_actual)}",
            "",
            {"text": "CONSECUENCIA", "bold": True},
            "Rotación, escalado, reflexión",
            "y cizalla pueden ser lineales.",
            "Una traslación pura no puede",
            "serlo porque mueve el origen.",
            "",
            f"det(A final) = {np.linalg.det(matriz_final):.3f}",
        ],
        "phase": fase,
        "info_line_height": 0.049,
        "legend": [
            {"kind": "line", "label": "cuadrícula original", "color": "#9CA3AF"},
            {"kind": "line", "label": "cuadrícula transformada", "color": "#1F77B4"},
            {"kind": "point", "label": "origen fijo", "color": "#111827"},
        ],
        "legend_ncol": 3,
        "legend_fontsize": 8.0,
    }


def crear_estado_conclusion(matriz, u, v):
    """Combina cuadrícula y superposición en el fotograma final."""

    base = crear_estado_origen(
        matriz_actual=matriz,
        matriz_final=matriz,
        fase="Conclusión",
        mensaje=(
            "Linealidad significa preservar combinaciones lineales. Además, "
            "toda transformación p↦Ap mantiene fijo el origen."
        ),
    )

    u_t = matriz @ u
    v_t = matriz @ v
    suma_t = matriz @ (u + v)

    base["polygons"] = [
        {
            "points": np.array([[0.0, 0.0], u_t, u_t + v_t, v_t]),
            "facecolor": "#E9DDF4",
            "edgecolor": "#7B2CBF",
            "alpha": 0.20,
            "linewidth": 1.5,
            "zorder": 18,
        },
    ]
    base["vectors"].extend([
        {
            "name": "Au",
            "origin": np.zeros(2),
            "value": u_t,
            "color": "#B23A48",
            "linewidth": 3.0,
        },
        {
            "name": "Av",
            "origin": u_t,
            "value": v_t,
            "color": "#2D7F5E",
            "linewidth": 2.6,
            "linestyle": "--",
        },
        {
            "name": "A(u+v)",
            "origin": np.zeros(2),
            "value": suma_t,
            "color": "#7B2CBF",
            "linewidth": 3.2,
        },
    ])
    base["info_lines"] = [
        {"text": "ADITIVIDAD", "bold": True},
        f"A(u+v) = {formatear_vector(suma_t)}",
        f"Au+Av  = {formatear_vector(u_t + v_t)}",
        "",
        {"text": "HOMOGENEIDAD", "bold": True},
        "A(λu) = λAu",
        "",
        {"text": "ORIGEN", "bold": True},
        "A·0 = 0",
        "",
        "Las tres propiedades se ven",
        "en una única transformación A.",
    ]
    base["info_line_height"] = 0.052
    base["legend"] = [
        {"kind": "line", "label": "cuadrícula transformada", "color": "#1F77B4"},
        {"kind": "line", "label": "Au", "color": "#B23A48"},
        {"kind": "line", "label": "A(u+v)", "color": "#7B2CBF"},
        {"kind": "point", "label": "origen fijo", "color": "#111827"},
    ]
    base["legend_ncol"] = 2
    return base


def crear_estados_demostracion():
    """Construye la animación del apartado 1.3."""

    matriz = np.array([
        [1.20, 0.55],
        [-0.35, 1.10],
    ])
    u = np.array([2.0, 0.7])
    v = np.array([-0.4, 1.7])
    lambda_final = -1.5

    estados = []
    identidad = np.eye(2)

    for _ in range(32):
        estados.append(
            crear_estado_aditividad(
                matriz_actual=identidad,
                matriz_final=matriz,
                u=u,
                v=v,
                fase="1/5 · Aditividad antes de transformar",
                mensaje=(
                    "u+v es la diagonal del paralelogramo. Aplicaremos la "
                    "misma matriz A a u, v y a su suma."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 85):
        suave = suavizar(progreso)
        matriz_actual = (1.0 - suave) * identidad + suave * matriz
        estados.append(
            crear_estado_aditividad(
                matriz_actual=matriz_actual,
                matriz_final=matriz,
                u=u,
                v=v,
                fase="2/5 · Transformar u, v y u+v",
                mensaje=(
                    "La transformación deforma todo el paralelogramo de forma "
                    "coherente: las sumas de vectores se conservan."
                ),
            )
        )

    for _ in range(34):
        estados.append(
            crear_estado_aditividad(
                matriz_actual=matriz,
                matriz_final=matriz,
                u=u,
                v=v,
                fase="3/5 · A(u+v) = Au + Av",
                mensaje=(
                    "La diagonal transformada A(u+v) coincide exactamente con "
                    "la suma de los vectores transformados Au y Av."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 75):
        suave = suavizar(progreso)
        lambda_actual = (1.0 - suave) * 1.0 + suave * lambda_final
        estados.append(
            crear_estado_homogeneidad(
                matriz=matriz,
                u=u,
                lambda_actual=lambda_actual,
                lambda_final=lambda_final,
                fase="4/5 · Homogeneidad",
                mensaje=(
                    "Escalar el vector antes de aplicar A produce el mismo "
                    "resultado que escalar Au después: A(λu)=λAu."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 80):
        suave = suavizar(progreso)
        matriz_actual = (1.0 - suave) * identidad + suave * matriz
        estados.append(
            crear_estado_origen(
                matriz_actual=matriz_actual,
                matriz_final=matriz,
                fase="5/5 · El origen permanece fijo",
                mensaje=(
                    "Aunque el plano se deforme, el punto 0 no se mueve. "
                    "Esta propiedad distingue una transformación lineal de una traslación."
                ),
            )
        )

    final = crear_estado_conclusion(matriz, u, v)
    for _ in range(48):
        estados.append(final)

    return {
        "states": estados,
        "matrix": matriz,
        "u": u,
        "v": v,
        "lambda": lambda_final,
    }


def imprimir_resultado(resultado):
    """Imprime las comprobaciones numéricas de linealidad."""

    matriz = resultado["matrix"]
    u = resultado["u"]
    v = resultado["v"]
    lam = resultado["lambda"]

    print("\n=== 1.3. Transformaciones lineales y superposición ===")
    print("\nMatriz A:")
    print(matriz)
    print("\nAditividad:")
    print(f"  A(u+v) = {formatear_vector(matriz @ (u + v))}")
    print(f"  Au+Av  = {formatear_vector(matriz @ u + matriz @ v)}")
    print("\nHomogeneidad:")
    print(f"  A(λu)  = {formatear_vector(matriz @ (lam * u))}")
    print(f"  λAu    = {formatear_vector(lam * (matriz @ u))}")
    print("\nOrigen:")
    print(f"  A0     = {formatear_vector(matriz @ np.zeros(2))}")


def main():
    resultado = crear_estados_demostracion()
    imprimir_resultado(resultado)

    animador = TransformAnimator(
        figsize=(15, 8.5),
        interval=50,
    )

    image_path = (
        MATRICES_DIR
        / "assets"
        / "01_fundamentos"
        / "03_transformaciones_lineales_superposicion.png"
    )

    video_path = (
        MATRICES_DIR
        / "assets"
        / "01_fundamentos"
        / "03_transformaciones_lineales_superposicion.webm"
    )

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="1.3. Transformaciones lineales y principio de superposición",
        limits=(-4.8, 5.5, -4.8, 5.5),
        final_image_path=image_path,
        video_path=video_path,
        repeat=False,
        fps=20,
        dpi=125,
        show=True,
    )

    _ = animacion


if __name__ == "__main__":
    main()
