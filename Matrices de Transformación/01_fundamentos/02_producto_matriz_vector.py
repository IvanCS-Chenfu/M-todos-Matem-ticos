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


def formatear_matriz(matriz):
    """Devuelve las dos filas de una matriz 2x2 como texto."""

    matriz = np.asarray(matriz, dtype=float)
    return (
        f"[[{matriz[0, 0]:5.2f}, {matriz[0, 1]:5.2f}],",
        f" [{matriz[1, 0]:5.2f}, {matriz[1, 1]:5.2f}]]",
    )


def interpolar(inicio, fin, cantidad):
    """Genera valores entre inicio y fin, incluyendo ambos extremos."""

    return np.linspace(inicio, fin, cantidad)


def suavizar(progreso):
    """Interpolación cosenoidal para movimientos suaves."""

    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def crear_cuadricula_transformada(matriz, limite=2.0, divisiones=5, muestras=61):
    """
    Calcula las líneas de una cuadrícula y sus imágenes por una matriz 2x2.

    El animador no conoce la transformación matemática: recibe simplemente
    los puntos ya calculados y se limita a dibujarlos.
    """

    matriz = np.asarray(matriz, dtype=float)
    valores = np.linspace(-limite, limite, divisiones)
    parametro = np.linspace(-limite, limite, muestras)

    originales = []
    transformadas = []

    for valor in valores:
        vertical = np.column_stack((np.full_like(parametro, valor), parametro))
        horizontal = np.column_stack((parametro, np.full_like(parametro, valor)))

        originales.extend([vertical, horizontal])
        transformadas.extend([
            (matriz @ vertical.T).T,
            (matriz @ horizontal.T).T,
        ])

    return originales, transformadas


def crear_estado_columnas(
    matriz,
    vector,
    escala_primera,
    escala_segunda,
    fase,
    mensaje,
):
    """
    Construye un estado que interpreta Ap como combinación de las columnas.

    Si A = [a1 a2] y p = [x y]^T, entonces:
        Ap = x*a1 + y*a2.
    """

    matriz = np.asarray(matriz, dtype=float)
    vector = np.asarray(vector, dtype=float)

    columna_1 = matriz[:, 0]
    columna_2 = matriz[:, 1]

    contribucion_1 = vector[0] * columna_1
    contribucion_2 = vector[1] * columna_2
    resultado = matriz @ vector

    parte_1 = escala_primera * contribucion_1
    parte_2 = escala_segunda * contribucion_2
    origen_parte_2 = parte_1

    matriz_linea_1, matriz_linea_2 = formatear_matriz(matriz)

    return {
        "vectors": [
            {
                "name": "p",
                "origin": np.zeros(2),
                "value": vector,
                "color": "#6B7280",
                "alpha": 0.65,
                "linewidth": 2.2,
            },
            {
                "name": "a1",
                "origin": np.zeros(2),
                "value": columna_1,
                "color": "#B23A48",
                "alpha": 0.85,
                "linewidth": 2.2,
            },
            {
                "name": "a2",
                "origin": np.zeros(2),
                "value": columna_2,
                "color": "#2D7F5E",
                "alpha": 0.85,
                "linewidth": 2.2,
            },
            {
                "name": "x·a1",
                "origin": np.zeros(2),
                "value": parte_1,
                "color": "#E07A1F",
                "alpha": 0.95 if escala_primera > 0 else 0.0,
                "linewidth": 3.0,
            },
            {
                "name": "y·a2",
                "origin": origen_parte_2,
                "value": parte_2,
                "color": "#1F77B4",
                "alpha": 0.95 if escala_segunda > 0 else 0.0,
                "linewidth": 3.0,
            },
        ],
        "points": [
            {
                "name": "Ap" if escala_primera >= 0.999 and escala_segunda >= 0.999 else "",
                "position": parte_1 + parte_2,
                "color": "#7B2CBF",
                "alpha": 1.0 if escala_primera > 0 or escala_segunda > 0 else 0.0,
                "size": 75,
            },
        ],
        "segments": [
            {
                "start": np.zeros(2),
                "end": resultado,
                "color": "#7B2CBF",
                "alpha": 0.32 if escala_primera >= 0.999 and escala_segunda >= 0.999 else 0.0,
                "linestyle": "--",
                "linewidth": 1.6,
            },
        ],
        "message": mensaje,
        "info_title": "Producto matriz-vector",
        "info_lines": [
            {"text": "DATOS", "bold": True},
            f"A = {matriz_linea_1}",
            f"    {matriz_linea_2}",
            f"p = {formatear_vector(vector)}",
            "",
            {"text": "COLUMNAS DE A", "bold": True},
            f"a1 = {formatear_vector(columna_1)}",
            f"a2 = {formatear_vector(columna_2)}",
            "",
            {"text": "COMBINACIÓN LINEAL", "bold": True},
            f"x·a1 = {formatear_vector(contribucion_1)}",
            f"y·a2 = {formatear_vector(contribucion_2)}",
            f"Ap    = {formatear_vector(resultado)}",
        ],
        "phase": fase,
        "info_line_height": 0.047,
        "legend": [
            {"kind": "line", "label": "vector p", "color": "#6B7280"},
            {"kind": "line", "label": "x·a1", "color": "#E07A1F"},
            {"kind": "line", "label": "y·a2", "color": "#1F77B4"},
            {"kind": "point", "label": "resultado Ap", "color": "#7B2CBF"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.5,
    }


def crear_estado_plano(matriz_actual, matriz_final, vector, fase, mensaje):
    """Crea un estado donde la matriz deforma una cuadrícula del plano."""

    matriz_actual = np.asarray(matriz_actual, dtype=float)
    matriz_final = np.asarray(matriz_final, dtype=float)
    vector = np.asarray(vector, dtype=float)

    originales, transformadas = crear_cuadricula_transformada(matriz_actual)
    resultado_actual = matriz_actual @ vector
    resultado_final = matriz_final @ vector

    polylines = []

    # Cuadrícula original tenue como referencia.
    originales_identidad, _ = crear_cuadricula_transformada(np.eye(2))
    for linea in originales_identidad:
        polylines.append({
            "points": linea,
            "color": "#9CA3AF",
            "alpha": 0.18,
            "linewidth": 0.9,
        })

    for linea in transformadas:
        polylines.append({
            "points": linea,
            "color": "#1F77B4",
            "alpha": 0.55,
            "linewidth": 1.1,
        })

    matriz_linea_1, matriz_linea_2 = formatear_matriz(matriz_actual)

    return {
        "polylines": polylines,
        "vectors": [
            {
                "name": "p",
                "origin": np.zeros(2),
                "value": vector,
                "color": "#6B7280",
                "alpha": 0.60,
                "linewidth": 2.2,
            },
            {
                "name": "A(t)p",
                "origin": np.zeros(2),
                "value": resultado_actual,
                "color": "#7B2CBF",
                "alpha": 1.0,
                "linewidth": 3.1,
            },
            {
                "name": "col 1",
                "origin": np.zeros(2),
                "value": matriz_actual[:, 0],
                "color": "#B23A48",
                "alpha": 0.9,
                "linewidth": 2.4,
            },
            {
                "name": "col 2",
                "origin": np.zeros(2),
                "value": matriz_actual[:, 1],
                "color": "#2D7F5E",
                "alpha": 0.9,
                "linewidth": 2.4,
            },
        ],
        "message": mensaje,
        "info_title": "Matriz como transformación del plano",
        "info_lines": [
            {"text": "MATRIZ ACTUAL A(t)", "bold": True},
            matriz_linea_1,
            matriz_linea_2,
            "",
            f"p       = {formatear_vector(vector)}",
            f"A(t)p   = {formatear_vector(resultado_actual)}",
            f"Ap final= {formatear_vector(resultado_final)}",
            "",
            {"text": "INTERPRETACIÓN", "bold": True},
            "Cada punto q pasa a A(t)q.",
            "Las rectas de la cuadrícula",
            "se transforman con la misma regla.",
        ],
        "phase": fase,
        "info_line_height": 0.050,
        "legend": [
            {"kind": "line", "label": "cuadrícula original", "color": "#9CA3AF"},
            {"kind": "line", "label": "cuadrícula transformada", "color": "#1F77B4"},
            {"kind": "line", "label": "p", "color": "#6B7280"},
            {"kind": "line", "label": "A(t)p", "color": "#7B2CBF"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.2,
    }


def crear_estados_demostracion():
    """
    Construye la animación del apartado 1.2 con el ejemplo numérico de la wiki.

    La demo muestra dos interpretaciones complementarias del producto Ap:
    1. cálculo como combinación de las columnas de A;
    2. matriz como función que transforma todo el plano.
    """

    matriz = np.array([
        [2.0, -1.0],
        [3.0, 4.0],
    ])
    vector = np.array([5.0, 2.0])

    estados = []

    for _ in range(28):
        estados.append(
            crear_estado_columnas(
                matriz=matriz,
                vector=vector,
                escala_primera=0.0,
                escala_segunda=0.0,
                fase="1/5 · A y p",
                mensaje=(
                    "A transforma p. Antes de multiplicar fila por columna, "
                    "observamos A como dos vectores columna a1 y a2."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 55):
        estados.append(
            crear_estado_columnas(
                matriz=matriz,
                vector=vector,
                escala_primera=suavizar(progreso),
                escala_segunda=0.0,
                fase="2/5 · Primera contribución",
                mensaje=(
                    "La componente x=5 multiplica la primera columna: "
                    "5·a1 = [10, 15]."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 55):
        estados.append(
            crear_estado_columnas(
                matriz=matriz,
                vector=vector,
                escala_primera=1.0,
                escala_segunda=suavizar(progreso),
                fase="3/5 · Segunda contribución",
                mensaje=(
                    "La componente y=2 multiplica la segunda columna y se suma "
                    "a la primera contribución."
                ),
            )
        )

    for _ in range(30):
        estados.append(
            crear_estado_columnas(
                matriz=matriz,
                vector=vector,
                escala_primera=1.0,
                escala_segunda=1.0,
                fase="4/5 · Resultado Ap",
                mensaje=(
                    "Ap = x·a1 + y·a2 = [8, 23]. El producto matriz-vector "
                    "es una combinación lineal de las columnas de A."
                ),
            )
        )

    identidad = np.eye(2)
    for progreso in interpolar(0.0, 1.0, 85):
        suave = suavizar(progreso)
        matriz_actual = (1.0 - suave) * identidad + suave * matriz

        estados.append(
            crear_estado_plano(
                matriz_actual=matriz_actual,
                matriz_final=matriz,
                vector=vector,
                fase="5/5 · La matriz transforma el plano",
                mensaje=(
                    "Interpolamos desde I hasta A. La misma multiplicación "
                    "se aplica a cada punto de la cuadrícula y al vector p."
                ),
            )
        )

    for _ in range(45):
        estados.append(
            crear_estado_plano(
                matriz_actual=matriz,
                matriz_final=matriz,
                vector=vector,
                fase="Conclusión",
                mensaje=(
                    "Una matriz 2x2 puede leerse como una función lineal del "
                    "plano y como las imágenes de los dos vectores de la base."
                ),
            )
        )

    return {
        "states": estados,
        "matrix": matriz,
        "vector": vector,
        "result": matriz @ vector,
    }


def imprimir_resultado(resultado):
    """Muestra por terminal el cálculo principal del apartado 1.2."""

    matriz = resultado["matrix"]
    vector = resultado["vector"]
    result = resultado["result"]

    columna_1 = matriz[:, 0]
    columna_2 = matriz[:, 1]

    print("\n=== 1.2. Producto matriz-vector ===")
    print("\nMatriz A:")
    print(matriz)
    print(f"\np = {formatear_vector(vector)}")
    print(f"\na1 = {formatear_vector(columna_1)}")
    print(f"a2 = {formatear_vector(columna_2)}")
    print(f"\nx*a1 = {formatear_vector(vector[0] * columna_1)}")
    print(f"y*a2 = {formatear_vector(vector[1] * columna_2)}")
    print(f"\nAp = {formatear_vector(result)}")


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
        / "02_producto_matriz_vector.png"
    )

    video_path = (
        MATRICES_DIR
        / "assets"
        / "01_fundamentos"
        / "02_producto_matriz_vector.webm"
    )

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="1.2. Producto matriz-vector",
        limits=(-7.0, 12.0, -5.0, 26.0),
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
