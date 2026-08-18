from pathlib import Path
import sys

import numpy as np


# Permite importar módulos desde la carpeta Matrices de Transformación/
CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def formatear_vector(vector):
    """Devuelve un vector 2D con formato compacto."""

    vector = np.asarray(vector, dtype=float)
    return f"[{vector[0]:6.2f}, {vector[1]:6.2f}]"


def formatear_matriz(matriz):
    """Devuelve una matriz 2x2 como dos líneas de texto."""

    matriz = np.asarray(matriz, dtype=float)
    return (
        f"[[{matriz[0, 0]:6.3f}, {matriz[0, 1]:6.3f}],",
        f" [{matriz[1, 0]:6.3f}, {matriz[1, 1]:6.3f}]]",
    )


def interpolar(inicio, fin, cantidad):
    """Genera valores entre inicio y fin, incluyendo ambos extremos."""

    return np.linspace(inicio, fin, cantidad)


def suavizar(progreso):
    """Interpolación cosenoidal para movimientos suaves."""

    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def matriz_rotacion(angulo):
    """Matriz activa 2D con ángulo positivo antihorario."""

    c = np.cos(angulo)
    s = np.sin(angulo)

    return np.array([
        [c, -s],
        [s, c],
    ])


def matriz_escalado(sx, sy):
    """Matriz diagonal de escalado 2D."""

    return np.array([
        [sx, 0.0],
        [0.0, sy],
    ])


def transformar_puntos(puntos, matriz):
    """Aplica una matriz 2x2 a una colección de puntos."""

    puntos = np.asarray(puntos, dtype=float)
    matriz = np.asarray(matriz, dtype=float)

    return (matriz @ puntos.T).T


def crear_estado_secuencia(
    figura,
    punto,
    matriz_actual,
    matriz_total,
    primera,
    segunda,
    fase,
    mensaje,
):
    """
    Muestra una secuencia de dos transformaciones lineales.

    `matriz_actual` es la transformación que se está visualizando en el frame
    actual; `matriz_total` es el producto final de ese orden.
    """

    figura = np.asarray(figura, dtype=float)
    punto = np.asarray(punto, dtype=float)
    matriz_actual = np.asarray(matriz_actual, dtype=float)
    matriz_total = np.asarray(matriz_total, dtype=float)

    figura_actual = transformar_puntos(figura, matriz_actual)
    punto_actual = matriz_actual @ punto
    punto_final = matriz_total @ punto

    m1, m2 = formatear_matriz(matriz_actual)
    mt1, mt2 = formatear_matriz(matriz_total)

    return {
        "polygons": [
            {
                "points": figura,
                "facecolor": "#D1D5DB",
                "edgecolor": "#6B7280",
                "alpha": 0.14,
                "linewidth": 1.4,
            },
            {
                "points": figura_actual,
                "facecolor": "#DCEAF7",
                "edgecolor": "#1F77B4",
                "alpha": 0.32,
                "linewidth": 2.0,
            },
        ],
        "points": [
            {
                "name": "p",
                "position": punto,
                "color": "#6B7280",
                "alpha": 0.62,
                "size": 70,
            },
            {
                "name": "A(t)p",
                "position": punto_actual,
                "color": "#1F77B4",
                "size": 90,
            },
        ],
        "vectors": [
            {
                "name": "col 1",
                "origin": np.zeros(2),
                "value": matriz_actual[:, 0],
                "color": "#B23A48",
                "linewidth": 2.5,
            },
            {
                "name": "col 2",
                "origin": np.zeros(2),
                "value": matriz_actual[:, 1],
                "color": "#2D7F5E",
                "linewidth": 2.5,
            },
        ],
        "message": mensaje,
        "info_title": "Composición de transformaciones",
        "info_lines": [
            {"text": "ORDEN", "bold": True},
            f"1º {primera}",
            f"2º {segunda}",
            "",
            {"text": "MATRIZ ACTUAL", "bold": True},
            m1,
            m2,
            "",
            {"text": "PRODUCTO FINAL", "bold": True},
            mt1,
            mt2,
            "",
            f"p final = {formatear_vector(punto_final)}",
            "La operación inicial queda",
            "más a la derecha del producto.",
        ],
        "phase": fase,
        "info_line_height": 0.044,
        "legend": [
            {"kind": "line", "label": "figura original", "color": "#6B7280"},
            {"kind": "line", "label": "figura actual", "color": "#1F77B4"},
            {"kind": "line", "label": "columna 1", "color": "#B23A48"},
            {"kind": "line", "label": "columna 2", "color": "#2D7F5E"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.1,
    }


def crear_estado_comparacion(
    figura,
    punto,
    rotacion,
    escalado,
):
    """Superpone R S y S R para hacer visible la no conmutatividad."""

    figura = np.asarray(figura, dtype=float)
    punto = np.asarray(punto, dtype=float)

    rs = rotacion @ escalado
    sr = escalado @ rotacion

    figura_rs = transformar_puntos(figura, rs)
    figura_sr = transformar_puntos(figura, sr)

    punto_rs = rs @ punto
    punto_sr = sr @ punto

    rs1, rs2 = formatear_matriz(rs)
    sr1, sr2 = formatear_matriz(sr)

    diferencia = np.linalg.norm(punto_rs - punto_sr)

    return {
        "polygons": [
            {
                "points": figura,
                "facecolor": "none",
                "edgecolor": "#6B7280",
                "alpha": 0.55,
                "linewidth": 1.5,
            },
            {
                "points": figura_rs,
                "facecolor": "#DCEAF7",
                "edgecolor": "#1F77B4",
                "alpha": 0.20,
                "linewidth": 2.2,
            },
            {
                "points": figura_sr,
                "facecolor": "#F2D7A7",
                "edgecolor": "#D97706",
                "alpha": 0.20,
                "linewidth": 2.2,
            },
        ],
        "points": [
            {
                "name": "R S p",
                "position": punto_rs,
                "color": "#1F77B4",
                "size": 90,
            },
            {
                "name": "S R p",
                "position": punto_sr,
                "color": "#D97706",
                "size": 90,
            },
        ],
        "segments": [
            {
                "start": punto_rs,
                "end": punto_sr,
                "color": "#7B2CBF",
                "alpha": 0.72,
                "linestyle": "--",
                "linewidth": 1.8,
            },
        ],
        "message": (
            "Con escalado no uniforme, R S y S R son matrices distintas. "
            "Por eso aplicar primero S y luego R no equivale al orden contrario."
        ),
        "info_title": "No conmutatividad",
        "info_lines": [
            {"text": "PRIMERO S, DESPUÉS R", "bold": True},
            "A = R S",
            rs1,
            rs2,
            f"R S p = {formatear_vector(punto_rs)}",
            "",
            {"text": "PRIMERO R, DESPUÉS S", "bold": True},
            "B = S R",
            sr1,
            sr2,
            f"S R p = {formatear_vector(punto_sr)}",
            "",
            f"||RSp-SRp|| = {diferencia:.3f}",
        ],
        "phase": "4/5 · R S != S R",
        "info_line_height": 0.043,
        "legend": [
            {"kind": "line", "label": "original", "color": "#6B7280"},
            {"kind": "line", "label": "R S", "color": "#1F77B4"},
            {"kind": "line", "label": "S R", "color": "#D97706"},
            {"kind": "line", "label": "diferencia", "color": "#7B2CBF", "linestyle": "--"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.1,
    }


def crear_estado_traslacion_externa(
    figura,
    punto,
    matriz_total,
    traslacion,
):
    """
    Introduce una traslación después de una composición lineal.

    Este estado no usa coordenadas homogéneas: enseña exactamente el problema
    del subapartado 2.5.3, es decir, que aparece una suma externa +t.
    """

    figura = np.asarray(figura, dtype=float)
    punto = np.asarray(punto, dtype=float)
    matriz_total = np.asarray(matriz_total, dtype=float)
    traslacion = np.asarray(traslacion, dtype=float)

    figura_lineal = transformar_puntos(figura, matriz_total)
    figura_afin = figura_lineal + traslacion

    punto_lineal = matriz_total @ punto
    punto_afin = punto_lineal + traslacion

    mt1, mt2 = formatear_matriz(matriz_total)

    return {
        "polygons": [
            {
                "points": figura_lineal,
                "facecolor": "#DCEAF7",
                "edgecolor": "#1F77B4",
                "alpha": 0.18,
                "linewidth": 2.0,
            },
            {
                "points": figura_afin,
                "facecolor": "#E9DDF4",
                "edgecolor": "#7B2CBF",
                "alpha": 0.24,
                "linewidth": 2.2,
            },
        ],
        "points": [
            {
                "name": "A p",
                "position": punto_lineal,
                "color": "#1F77B4",
                "size": 82,
            },
            {
                "name": "A p + t",
                "position": punto_afin,
                "color": "#7B2CBF",
                "size": 95,
            },
        ],
        "vectors": [
            {
                "name": "t",
                "origin": punto_lineal,
                "value": traslacion,
                "color": "#E07A1F",
                "linewidth": 3.0,
            },
        ],
        "message": (
            "Las transformaciones lineales se reducen a un producto A_total. "
            "Al añadir una traslación aparece A_total p + t: la suma queda fuera "
            "de la matriz 2x2 y motiva las coordenadas homogéneas."
        ),
        "info_title": "Preparación para homogéneas",
        "info_lines": [
            {"text": "PARTE LINEAL", "bold": True},
            "A_total = R S",
            mt1,
            mt2,
            f"A p     = {formatear_vector(punto_lineal)}",
            "",
            {"text": "AÑADIR TRASLACIÓN", "bold": True},
            f"t       = {formatear_vector(traslacion)}",
            f"A p + t = {formatear_vector(punto_afin)}",
            "",
            "Con matrices 2x2 seguimos",
            "necesitando arrastrar t aparte.",
        ],
        "phase": "Conclusión · Falta integrar la traslación",
        "info_line_height": 0.048,
        "legend": [
            {"kind": "line", "label": "resultado lineal", "color": "#1F77B4"},
            {"kind": "line", "label": "resultado + traslación", "color": "#7B2CBF"},
            {"kind": "line", "label": "vector t", "color": "#E07A1F"},
        ],
        "legend_ncol": 3,
        "legend_fontsize": 8.0,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 2.5."""

    angulo = np.radians(45.0)
    rotacion = matriz_rotacion(angulo)
    escalado = matriz_escalado(2.0, 0.5)

    figura = np.array([
        [0.4, 0.4],
        [2.5, 0.4],
        [2.1, 1.3],
        [2.7, 2.0],
        [0.8, 2.5],
    ])
    punto = np.array([2.0, 1.0])

    estados = []
    identidad = np.eye(2)

    # --------------------------------------------------------------
    # Orden A: primero escalar S, después rotar R -> R S
    # --------------------------------------------------------------
    for progreso in interpolar(0.0, 1.0, 75):
        suave = suavizar(progreso)
        sx = 1.0 + suave * 1.0
        sy = 1.0 - suave * 0.5
        matriz_actual = matriz_escalado(sx, sy)

        estados.append(
            crear_estado_secuencia(
                figura=figura,
                punto=punto,
                matriz_actual=matriz_actual,
                matriz_total=rotacion @ escalado,
                primera="escalar S",
                segunda="rotar R",
                fase="1/5 · Primero S",
                mensaje=(
                    "Aplicamos primero el escalado no uniforme. En el producto "
                    "final R S, S aparece a la derecha."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 85):
        theta = suavizar(progreso) * angulo
        matriz_actual = matriz_rotacion(theta) @ escalado

        estados.append(
            crear_estado_secuencia(
                figura=figura,
                punto=punto,
                matriz_actual=matriz_actual,
                matriz_total=rotacion @ escalado,
                primera="escalar S",
                segunda="rotar R",
                fase="2/5 · Después R",
                mensaje=(
                    "La segunda transformación multiplica por la izquierda: "
                    "p_final = R(S p) = (R S)p."
                ),
            )
        )

    # Volver suavemente a la identidad para comenzar el segundo orden.
    rs = rotacion @ escalado
    for progreso in interpolar(0.0, 1.0, 42):
        suave = suavizar(progreso)
        matriz_actual = (1.0 - suave) * rs + suave * identidad

        estados.append(
            crear_estado_secuencia(
                figura=figura,
                punto=punto,
                matriz_actual=matriz_actual,
                matriz_total=escalado @ rotacion,
                primera="rotar R",
                segunda="escalar S",
                fase="3/5 · Cambiar el orden",
                mensaje=(
                    "Regresamos a la figura inicial para repetir el proceso en "
                    "el orden contrario."
                ),
            )
        )

    # --------------------------------------------------------------
    # Orden B: primero rotar R, después escalar S -> S R
    # --------------------------------------------------------------
    for progreso in interpolar(0.0, 1.0, 80):
        theta = suavizar(progreso) * angulo
        matriz_actual = matriz_rotacion(theta)

        estados.append(
            crear_estado_secuencia(
                figura=figura,
                punto=punto,
                matriz_actual=matriz_actual,
                matriz_total=escalado @ rotacion,
                primera="rotar R",
                segunda="escalar S",
                fase="3/5 · Primero R",
                mensaje=(
                    "Ahora rotamos primero. En el producto final S R, R aparece "
                    "a la derecha porque actúa antes."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 85):
        suave = suavizar(progreso)
        sx = 1.0 + suave * 1.0
        sy = 1.0 - suave * 0.5
        matriz_actual = matriz_escalado(sx, sy) @ rotacion

        estados.append(
            crear_estado_secuencia(
                figura=figura,
                punto=punto,
                matriz_actual=matriz_actual,
                matriz_total=escalado @ rotacion,
                primera="rotar R",
                segunda="escalar S",
                fase="3/5 · Después S",
                mensaje=(
                    "El escalado se aplica sobre la figura ya rotada: "
                    "p_final = S(R p) = (S R)p."
                ),
            )
        )

    for _ in range(45):
        estados.append(
            crear_estado_comparacion(
                figura=figura,
                punto=punto,
                rotacion=rotacion,
                escalado=escalado,
            )
        )

    traslacion = np.array([2.2, -1.2])
    for _ in range(48):
        estados.append(
            crear_estado_traslacion_externa(
                figura=figura,
                punto=punto,
                matriz_total=rotacion @ escalado,
                traslacion=traslacion,
            )
        )

    return {
        "states": estados,
        "point": punto,
        "rotation": rotacion,
        "scale": escalado,
        "translation": traslacion,
    }


def imprimir_resultado(resultado):
    """Muestra por terminal los productos de los dos órdenes."""

    punto = resultado["point"]
    rotacion = resultado["rotation"]
    escalado = resultado["scale"]
    traslacion = resultado["translation"]

    rs = rotacion @ escalado
    sr = escalado @ rotacion

    print("\n=== 2.5. Composición y orden ===")

    print("\nPrimero S y después R -> R S:")
    print(rs)
    print(f"R S p = {formatear_vector(rs @ punto)}")

    print("\nPrimero R y después S -> S R:")
    print(sr)
    print(f"S R p = {formatear_vector(sr @ punto)}")

    print("\n¿Coinciden las matrices?")
    print(np.allclose(rs, sr))

    print("\nAñadiendo una traslación después de R S:")
    print(f"R S p + t = {formatear_vector(rs @ punto + traslacion)}")
    print("La suma +t queda fuera de la matriz 2x2.")


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
        / "02_transformaciones_2d"
        / "05_composicion_orden.png"
    )

    video_path = (
        MATRICES_DIR
        / "assets"
        / "02_transformaciones_2d"
        / "05_composicion_orden.webm"
    )

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="2.5. Composición y orden de las transformaciones 2D",
        limits=(-4.5, 8.2, -4.8, 6.8),
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
