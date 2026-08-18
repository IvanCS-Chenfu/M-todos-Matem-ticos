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
        f"[[{matriz[0, 0]:6.2f}, {matriz[0, 1]:6.2f}],",
        f" [{matriz[1, 0]:6.2f}, {matriz[1, 1]:6.2f}]]",
    )


def interpolar(inicio, fin, cantidad):
    """Genera valores entre inicio y fin, incluyendo ambos extremos."""

    return np.linspace(inicio, fin, cantidad)


def suavizar(progreso):
    """Interpolación cosenoidal para movimientos suaves."""

    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def matriz_escalado(sx, sy):
    """Construye la matriz diagonal de escalado."""

    return np.array([
        [sx, 0.0],
        [0.0, sy],
    ])


def transformar_puntos(puntos, matriz):
    """Aplica una matriz 2x2 a una colección de puntos."""

    puntos = np.asarray(puntos, dtype=float)
    matriz = np.asarray(matriz, dtype=float)

    return (matriz @ puntos.T).T


def crear_cuadricula(matriz, limite=2.5, divisiones=7, muestras=61):
    """Calcula una cuadrícula transformada por una matriz 2x2."""

    matriz = np.asarray(matriz, dtype=float)
    valores = np.linspace(-limite, limite, divisiones)
    parametro = np.linspace(-limite, limite, muestras)

    lineas = []

    for valor in valores:
        vertical = np.column_stack((
            np.full_like(parametro, valor),
            parametro,
        ))
        horizontal = np.column_stack((
            parametro,
            np.full_like(parametro, valor),
        ))

        lineas.extend([
            transformar_puntos(vertical, matriz),
            transformar_puntos(horizontal, matriz),
        ])

    return lineas


def crear_estado_escalado(
    figura,
    punto,
    sx,
    sy,
    fase,
    mensaje,
):
    """Crea un estado de escalado con cuadrícula, figura y punto."""

    figura = np.asarray(figura, dtype=float)
    punto = np.asarray(punto, dtype=float)

    matriz = matriz_escalado(sx, sy)
    figura_t = transformar_puntos(figura, matriz)
    punto_t = matriz @ punto

    polylines = []

    for linea in crear_cuadricula(np.eye(2)):
        polylines.append({
            "points": linea,
            "color": "#9CA3AF",
            "alpha": 0.14,
            "linewidth": 0.8,
        })

    for linea in crear_cuadricula(matriz):
        polylines.append({
            "points": linea,
            "color": "#1F77B4",
            "alpha": 0.48,
            "linewidth": 1.0,
        })

    m1, m2 = formatear_matriz(matriz)

    return {
        "polylines": polylines,
        "polygons": [
            {
                "points": figura,
                "facecolor": "#D1D5DB",
                "edgecolor": "#6B7280",
                "alpha": 0.14,
                "linewidth": 1.4,
            },
            {
                "points": figura_t,
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
                "name": "S p",
                "position": punto_t,
                "color": "#7B2CBF",
                "size": 95,
            },
        ],
        "vectors": [
            {
                "name": "S e1",
                "origin": np.zeros(2),
                "value": matriz[:, 0],
                "color": "#B23A48",
                "linewidth": 2.7,
            },
            {
                "name": "S e2",
                "origin": np.zeros(2),
                "value": matriz[:, 1],
                "color": "#2D7F5E",
                "linewidth": 2.7,
            },
        ],
        "message": mensaje,
        "info_title": "Escalado 2D",
        "info_lines": [
            {"text": "FACTORES", "bold": True},
            f"sx = {sx:6.2f}",
            f"sy = {sy:6.2f}",
            "",
            {"text": "MATRIZ S", "bold": True},
            m1,
            m2,
            "",
            {"text": "PUNTO", "bold": True},
            f"p       = {formatear_vector(punto)}",
            f"S p     = {formatear_vector(punto_t)}",
            "",
            {"text": "ÁREA", "bold": True},
            f"det(S)  = {np.linalg.det(matriz):6.3f}",
            f"|det(S)|= {abs(np.linalg.det(matriz)):6.3f}",
        ],
        "phase": fase,
        "info_line_height": 0.045,
        "legend": [
            {"kind": "line", "label": "cuadrícula original", "color": "#9CA3AF"},
            {"kind": "line", "label": "cuadrícula escalada", "color": "#1F77B4"},
            {"kind": "line", "label": "S e1", "color": "#B23A48"},
            {"kind": "line", "label": "S e2", "color": "#2D7F5E"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.1,
    }


def crear_estado_comparacion(figura, punto):
    """
    Superpone tres resultados para comparar tipos de escalado.

    Se muestran:
    - uniforme: diag(1.6, 1.6),
    - no uniforme: diag(2, 0.5),
    - negativo: diag(-1, 1).
    """

    figura = np.asarray(figura, dtype=float)
    punto = np.asarray(punto, dtype=float)

    matrices = {
        "uniforme": matriz_escalado(1.6, 1.6),
        "no_uniforme": matriz_escalado(2.0, 0.5),
        "negativo": matriz_escalado(-1.0, 1.0),
    }

    figura_uniforme = transformar_puntos(figura, matrices["uniforme"])
    figura_no_uniforme = transformar_puntos(
        figura,
        matrices["no_uniforme"],
    )
    figura_negativa = transformar_puntos(figura, matrices["negativo"])

    punto_uniforme = matrices["uniforme"] @ punto
    punto_no_uniforme = matrices["no_uniforme"] @ punto
    punto_negativo = matrices["negativo"] @ punto

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
                "points": figura_uniforme,
                "facecolor": "#DCEAF7",
                "edgecolor": "#1F77B4",
                "alpha": 0.17,
                "linewidth": 2.0,
            },
            {
                "points": figura_no_uniforme,
                "facecolor": "#F2D7A7",
                "edgecolor": "#D97706",
                "alpha": 0.17,
                "linewidth": 2.0,
            },
            {
                "points": figura_negativa,
                "facecolor": "#E9DDF4",
                "edgecolor": "#7B2CBF",
                "alpha": 0.17,
                "linewidth": 2.0,
            },
        ],
        "points": [
            {
                "name": "S_u p",
                "position": punto_uniforme,
                "color": "#1F77B4",
                "size": 70,
            },
            {
                "name": "S_nu p",
                "position": punto_no_uniforme,
                "color": "#D97706",
                "size": 70,
            },
            {
                "name": "S_- p",
                "position": punto_negativo,
                "color": "#7B2CBF",
                "size": 70,
            },
        ],
        "message": (
            "Escalado uniforme: misma forma. No uniforme: deformación. "
            "Factor negativo: reflexión en el eje correspondiente."
        ),
        "info_title": "Comparación de escalados",
        "info_lines": [
            {"text": "UNIFORME", "bold": True},
            "S = diag(1.6, 1.6)",
            f"det = {np.linalg.det(matrices['uniforme']):.2f}",
            "",
            {"text": "NO UNIFORME", "bold": True},
            "S = diag(2.0, 0.5)",
            f"det = {np.linalg.det(matrices['no_uniforme']):.2f}",
            "",
            {"text": "FACTOR NEGATIVO", "bold": True},
            "S = diag(-1.0, 1.0)",
            f"det = {np.linalg.det(matrices['negativo']):.2f}",
            "",
            "El signo del determinante",
            "indica si cambia la orientación.",
        ],
        "phase": "Conclusión · Tres tipos de escalado",
        "info_line_height": 0.046,
        "legend": [
            {"kind": "line", "label": "original", "color": "#6B7280"},
            {"kind": "line", "label": "uniforme", "color": "#1F77B4"},
            {"kind": "line", "label": "no uniforme", "color": "#D97706"},
            {"kind": "line", "label": "escala negativa", "color": "#7B2CBF"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.1,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 2.3."""

    punto = np.array([3.0, 4.0])

    figura = np.array([
        [0.4, 0.5],
        [2.4, 0.5],
        [2.4, 2.1],
        [0.9, 2.8],
        [0.4, 2.1],
    ])

    estados = []

    for progreso in interpolar(0.0, 1.0, 85):
        suave = suavizar(progreso)
        s = 1.0 + suave * 0.60

        estados.append(
            crear_estado_escalado(
                figura=figura,
                punto=punto,
                sx=s,
                sy=s,
                fase="1/4 · Escalado uniforme",
                mensaje=(
                    "sx=sy: todas las direcciones se multiplican por el mismo "
                    "factor. La figura cambia de tamaño sin deformarse."
                ),
            )
        )

    for _ in range(20):
        estados.append(
            crear_estado_escalado(
                figura=figura,
                punto=punto,
                sx=1.0,
                sy=1.0,
                fase="2/4 · Preparar escalado no uniforme",
                mensaje=(
                    "Volvemos a la identidad para comparar con el ejemplo "
                    "S=diag(2, 0.5)."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 95):
        suave = suavizar(progreso)
        sx = 1.0 + suave * 1.0
        sy = 1.0 - suave * 0.5

        estados.append(
            crear_estado_escalado(
                figura=figura,
                punto=punto,
                sx=sx,
                sy=sy,
                fase="2/4 · Escalado no uniforme",
                mensaje=(
                    "El eje x se estira mientras y se comprime. En el ejemplo "
                    "final p=[3,4] pasa a [6,2]."
                ),
            )
        )

    for _ in range(20):
        estados.append(
            crear_estado_escalado(
                figura=figura,
                punto=punto,
                sx=1.0,
                sy=1.0,
                fase="3/4 · Preparar escala negativa",
                mensaje=(
                    "Una escala negativa invierte la dirección del eje "
                    "correspondiente y produce una reflexión."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 95):
        suave = suavizar(progreso)
        sx = 1.0 - 2.0 * suave
        sy = 1.0

        estados.append(
            crear_estado_escalado(
                figura=figura,
                punto=punto,
                sx=sx,
                sy=sy,
                fase="3/4 · Factor negativo",
                mensaje=(
                    "sx atraviesa 0 y termina en -1: la figura queda reflejada "
                    "respecto al eje y y det(S) cambia de signo."
                ),
            )
        )

    for _ in range(45):
        estados.append(
            crear_estado_comparacion(
                figura=figura,
                punto=punto,
            )
        )

    return {
        "states": estados,
        "point": punto,
        "example_matrix": matriz_escalado(2.0, 0.5),
    }


def imprimir_resultado(resultado):
    """Muestra por terminal el ejemplo numérico del apartado 2.3."""

    punto = resultado["point"]
    matriz = resultado["example_matrix"]
    transformado = matriz @ punto

    print("\n=== 2.3. Escalado en 2D ===")
    print("\nS =")
    print(matriz)
    print(f"\np  = {formatear_vector(punto)}")
    print(f"Sp = {formatear_vector(transformado)}")
    print(f"\ndet(S) = {np.linalg.det(matriz):.3f}")
    print("El ejemplo cambia la forma, pero conserva el área porque |det(S)|=1.")


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
        / "03_escalado_2d.png"
    )

    video_path = (
        MATRICES_DIR
        / "assets"
        / "02_transformaciones_2d"
        / "03_escalado_2d.webm"
    )

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="2.3. Escalado en 2D",
        limits=(-5.8, 7.3, -4.8, 6.8),
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
