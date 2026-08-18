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


def crear_cuadricula(matriz, limite=2.5, divisiones=7, muestras=61):
    """Calcula una cuadrícula transformada por una matriz 2x2."""

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


def crear_estado_columnas(matriz_actual, matriz_final, alpha_area, fase, mensaje):
    """
    Muestra e1, e2 y sus imágenes, que son las columnas de la matriz.
    """

    matriz_actual = np.asarray(matriz_actual, dtype=float)
    matriz_final = np.asarray(matriz_final, dtype=float)

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    c1 = matriz_actual[:, 0]
    c2 = matriz_actual[:, 1]

    polylines = []
    for linea in crear_cuadricula(np.eye(2)):
        polylines.append({
            "points": linea,
            "color": "#9CA3AF",
            "alpha": 0.13,
            "linewidth": 0.8,
        })
    for linea in crear_cuadricula(matriz_actual):
        polylines.append({
            "points": linea,
            "color": "#1F77B4",
            "alpha": 0.43,
            "linewidth": 1.0,
        })

    vertices = np.array([
        [0.0, 0.0],
        c1,
        c1 + c2,
        c2,
    ])

    return {
        "polylines": polylines,
        "polygons": [
            {
                "points": vertices,
                "facecolor": "#F2D7A7",
                "edgecolor": "#B7791F",
                "alpha": 0.34 * alpha_area,
                "linewidth": 1.7,
                "zorder": 15,
            },
        ],
        "vectors": [
            {
                "name": "e1",
                "origin": np.zeros(2),
                "value": e1,
                "color": "#9CA3AF",
                "alpha": 0.50,
                "linewidth": 2.0,
            },
            {
                "name": "e2",
                "origin": np.zeros(2),
                "value": e2,
                "color": "#9CA3AF",
                "alpha": 0.50,
                "linewidth": 2.0,
            },
            {
                "name": "Ae1 = col 1",
                "origin": np.zeros(2),
                "value": c1,
                "color": "#B23A48",
                "linewidth": 3.0,
            },
            {
                "name": "Ae2 = col 2",
                "origin": np.zeros(2),
                "value": c2,
                "color": "#2D7F5E",
                "linewidth": 3.0,
            },
        ],
        "message": mensaje,
        "info_title": "Columnas e interpretación geométrica",
        "info_lines": [
            {"text": "BASE CANÓNICA", "bold": True},
            f"e1 = {formatear_vector(e1)}",
            f"e2 = {formatear_vector(e2)}",
            "",
            {"text": "IMÁGENES POR A(t)", "bold": True},
            f"A(t)e1 = {formatear_vector(c1)}",
            f"A(t)e2 = {formatear_vector(c2)}",
            "",
            {"text": "ÁREA", "bold": True},
            f"|det A(t)| = {abs(np.linalg.det(matriz_actual)):.3f}",
            f"|det A|    = {abs(np.linalg.det(matriz_final)):.3f}",
            "",
            "Las columnas indican dónde",
            "terminan e1 y e2 tras aplicar A.",
        ],
        "phase": fase,
        "info_line_height": 0.046,
        "legend": [
            {"kind": "line", "label": "base canónica", "color": "#9CA3AF"},
            {"kind": "line", "label": "columna 1", "color": "#B23A48"},
            {"kind": "line", "label": "columna 2", "color": "#2D7F5E"},
            {"kind": "line", "label": "cuadrícula transformada", "color": "#1F77B4"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.2,
    }


def crear_estado_cambio_base(base_actual, base_final, punto, fase, mensaje):
    """
    Mantiene fijo un vector físico y cambia la base usada para describirlo.

    Si B=[b1 b2], las coordenadas del vector p en esa base son B^{-1}p.
    """

    base_actual = np.asarray(base_actual, dtype=float)
    base_final = np.asarray(base_final, dtype=float)
    punto = np.asarray(punto, dtype=float)

    coordenadas = np.linalg.solve(base_actual, punto)
    coordenadas_finales = np.linalg.solve(base_final, punto)

    b1 = base_actual[:, 0]
    b2 = base_actual[:, 1]
    contribucion_1 = coordenadas[0] * b1
    contribucion_2 = coordenadas[1] * b2

    return {
        "polygons": [
            {
                "points": np.array([
                    [0.0, 0.0],
                    b1,
                    b1 + b2,
                    b2,
                ]),
                "facecolor": "#DCEAF7",
                "edgecolor": "#7A9CC6",
                "alpha": 0.18,
                "linewidth": 1.4,
            },
        ],
        "vectors": [
            {
                "name": "b1",
                "origin": np.zeros(2),
                "value": b1,
                "color": "#B23A48",
                "linewidth": 3.0,
            },
            {
                "name": "b2",
                "origin": np.zeros(2),
                "value": b2,
                "color": "#2D7F5E",
                "linewidth": 3.0,
            },
            {
                "name": "p físico",
                "origin": np.zeros(2),
                "value": punto,
                "color": "#7B2CBF",
                "linewidth": 3.3,
            },
            {
                "name": "q1·b1",
                "origin": np.zeros(2),
                "value": contribucion_1,
                "color": "#E07A1F",
                "linewidth": 2.5,
                "linestyle": "--",
            },
            {
                "name": "q2·b2",
                "origin": contribucion_1,
                "value": contribucion_2,
                "color": "#1F77B4",
                "linewidth": 2.5,
                "linestyle": "--",
            },
        ],
        "message": mensaje,
        "info_title": "Cambio de base: p no se mueve",
        "info_lines": [
            {"text": "VECTOR FÍSICO", "bold": True},
            f"p = {formatear_vector(punto)}",
            "",
            {"text": "BASE ACTUAL B(t)", "bold": True},
            f"b1 = {formatear_vector(b1)}",
            f"b2 = {formatear_vector(b2)}",
            "",
            {"text": "COORDENADAS [p]_B(t)", "bold": True},
            f"q = {formatear_vector(coordenadas)}",
            "",
            {"text": "BASE FINAL", "bold": True},
            f"[p]_B = {formatear_vector(coordenadas_finales)}",
            "",
            "B q = p: cambia la descripción,",
            "no el vector físico.",
        ],
        "phase": fase,
        "info_line_height": 0.045,
        "legend": [
            {"kind": "line", "label": "b1", "color": "#B23A48"},
            {"kind": "line", "label": "b2", "color": "#2D7F5E"},
            {"kind": "line", "label": "p físico", "color": "#7B2CBF"},
            {"kind": "line", "label": "componentes en B", "color": "#E07A1F", "linestyle": "--"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.2,
    }


def crear_estado_activa_pasiva(matriz, punto):
    """Fotograma final que contrasta transformación activa y cambio de base."""

    matriz = np.asarray(matriz, dtype=float)
    punto = np.asarray(punto, dtype=float)
    punto_activo = matriz @ punto
    coordenadas_pasivas = np.linalg.solve(matriz, punto)

    b1 = matriz[:, 0]
    b2 = matriz[:, 1]
    contribucion_1 = coordenadas_pasivas[0] * b1
    contribucion_2 = coordenadas_pasivas[1] * b2

    return {
        "polygons": [
            {
                "points": np.array([[0.0, 0.0], b1, b1 + b2, b2]),
                "facecolor": "#DCEAF7",
                "edgecolor": "#7A9CC6",
                "alpha": 0.20,
                "linewidth": 1.4,
            },
        ],
        "vectors": [
            {
                "name": "b1 = col 1",
                "origin": np.zeros(2),
                "value": b1,
                "color": "#B23A48",
                "linewidth": 2.8,
            },
            {
                "name": "b2 = col 2",
                "origin": np.zeros(2),
                "value": b2,
                "color": "#2D7F5E",
                "linewidth": 2.8,
            },
            {
                "name": "p",
                "origin": np.zeros(2),
                "value": punto,
                "color": "#7B2CBF",
                "linewidth": 3.1,
            },
            {
                "name": "Ap (activa)",
                "origin": np.zeros(2),
                "value": punto_activo,
                "color": "#D97706",
                "linewidth": 3.1,
            },
            {
                "name": "q1·b1",
                "origin": np.zeros(2),
                "value": contribucion_1,
                "color": "#1F77B4",
                "linewidth": 2.2,
                "linestyle": "--",
            },
            {
                "name": "q2·b2",
                "origin": contribucion_1,
                "value": contribucion_2,
                "color": "#1F77B4",
                "linewidth": 2.2,
                "linestyle": "--",
            },
        ],
        "message": (
            "La misma matriz puede usarse en preguntas distintas: activamente "
            "mueve p a Ap; como base, deja p fijo y cambia sus coordenadas."
        ),
        "info_title": "Activa frente a cambio de base",
        "info_lines": [
            {"text": "TRANSFORMACIÓN ACTIVA", "bold": True},
            f"p  = {formatear_vector(punto)}",
            f"Ap = {formatear_vector(punto_activo)}",
            "El vector físico cambia.",
            "",
            {"text": "CAMBIO DE BASE", "bold": True},
            f"p físico = {formatear_vector(punto)}",
            f"[p]_B    = {formatear_vector(coordenadas_pasivas)}",
            "El vector físico no cambia.",
            "",
            {"text": "ÁREA DE LA BASE", "bold": True},
            f"|det A| = {abs(np.linalg.det(matriz)):.3f}",
        ],
        "phase": "Conclusión",
        "info_line_height": 0.049,
        "legend": [
            {"kind": "line", "label": "p", "color": "#7B2CBF"},
            {"kind": "line", "label": "Ap activa", "color": "#D97706"},
            {"kind": "line", "label": "base B", "color": "#B23A48"},
            {"kind": "line", "label": "descomposición pasiva", "color": "#1F77B4", "linestyle": "--"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.1,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 1.4."""

    matriz = np.array([
        [1.40, -0.45],
        [0.55, 1.15],
    ])
    punto = np.array([2.60, 2.00])
    identidad = np.eye(2)

    estados = []

    for _ in range(32):
        estados.append(
            crear_estado_columnas(
                matriz_actual=identidad,
                matriz_final=matriz,
                alpha_area=0.0,
                fase="1/5 · Base canónica",
                mensaje=(
                    "Partimos de e1 y e2. Una matriz queda determinada por "
                    "dónde envía estos dos vectores de la base canónica."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 90):
        suave = suavizar(progreso)
        matriz_actual = (1.0 - suave) * identidad + suave * matriz
        estados.append(
            crear_estado_columnas(
                matriz_actual=matriz_actual,
                matriz_final=matriz,
                alpha_area=0.0,
                fase="2/5 · Las columnas son Ae1 y Ae2",
                mensaje=(
                    "Al deformar la base, Ae1 termina en la primera columna y "
                    "Ae2 en la segunda. Así se puede leer geométricamente A."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 45):
        estados.append(
            crear_estado_columnas(
                matriz_actual=matriz,
                matriz_final=matriz,
                alpha_area=suavizar(progreso),
                fase="3/5 · Determinante y área",
                mensaje=(
                    "Las columnas generan un paralelogramo. Su área es "
                    "|det(A)| veces el área del cuadrado unidad."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 95):
        suave = suavizar(progreso)
        base_actual = (1.0 - suave) * identidad + suave * matriz
        estados.append(
            crear_estado_cambio_base(
                base_actual=base_actual,
                base_final=matriz,
                punto=punto,
                fase="4/5 · Cambiar la base",
                mensaje=(
                    "El vector p permanece físicamente fijo. Al cambiar b1 y b2, "
                    "sus coordenadas q=B^{-1}p cambian para seguir describiendo p."
                ),
            )
        )

    for _ in range(36):
        estados.append(
            crear_estado_cambio_base(
                base_actual=matriz,
                base_final=matriz,
                punto=punto,
                fase="5/5 · p = B[p]_B",
                mensaje=(
                    "Las componentes [p]_B indican cuánto de b1 y b2 hace falta "
                    "para reconstruir el mismo vector físico p."
                ),
            )
        )

    final = crear_estado_activa_pasiva(matriz, punto)
    for _ in range(48):
        estados.append(final)

    return {
        "states": estados,
        "matrix": matriz,
        "point": punto,
        "coordinates_in_basis": np.linalg.solve(matriz, punto),
    }


def imprimir_resultado(resultado):
    """Imprime los datos principales de la interpretación geométrica."""

    matriz = resultado["matrix"]
    punto = resultado["point"]
    coords = resultado["coordinates_in_basis"]

    print("\n=== 1.4. Interpretación geométrica y cambio de base ===")
    print("\nMatriz A / base B:")
    print(matriz)
    print(f"\nPrimera columna  = {formatear_vector(matriz[:, 0])}")
    print(f"Segunda columna  = {formatear_vector(matriz[:, 1])}")
    print(f"|det(A)|         = {abs(np.linalg.det(matriz)):.4f}")
    print(f"\np físico          = {formatear_vector(punto)}")
    print(f"[p]_B             = {formatear_vector(coords)}")
    print(f"B[p]_B            = {formatear_vector(matriz @ coords)}")
    print(f"Ap (activa)       = {formatear_vector(matriz @ punto)}")


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
        / "04_interpretacion_geometrica_matriz_cambio_base.png"
    )

    video_path = (
        MATRICES_DIR
        / "assets"
        / "01_fundamentos"
        / "04_interpretacion_geometrica_matriz_cambio_base.webm"
    )

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="1.4. Interpretación geométrica de una matriz y cambio de base",
        limits=(-4.8, 5.8, -4.3, 5.5),
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
