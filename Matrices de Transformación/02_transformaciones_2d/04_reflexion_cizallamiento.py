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


def transformar_puntos(puntos, matriz):
    """Aplica una matriz 2x2 a una colección de puntos."""

    puntos = np.asarray(puntos, dtype=float)
    matriz = np.asarray(matriz, dtype=float)

    return (matriz @ puntos.T).T


def crear_cuadricula(matriz, limite=3.0, divisiones=7, muestras=61):
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


def area_firmada(poligono):
    """
    Calcula el área firmada de un polígono.

    El signo permite visualizar si la orientación de los vértices se conserva
    o se invierte tras una transformación.
    """

    poligono = np.asarray(poligono, dtype=float)
    x = poligono[:, 0]
    y = poligono[:, 1]

    return 0.5 * (
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )


def crear_estado_reflexion(
    figura,
    punto,
    factor_y,
    fase,
    mensaje,
):
    """Visualiza una reflexión progresiva respecto al eje x."""

    figura = np.asarray(figura, dtype=float)
    punto = np.asarray(punto, dtype=float)

    matriz = np.array([
        [1.0, 0.0],
        [0.0, factor_y],
    ])

    figura_t = transformar_puntos(figura, matriz)
    punto_t = matriz @ punto

    polylines = []

    for linea in crear_cuadricula(np.eye(2)):
        polylines.append({
            "points": linea,
            "color": "#9CA3AF",
            "alpha": 0.13,
            "linewidth": 0.8,
        })

    for linea in crear_cuadricula(matriz):
        polylines.append({
            "points": linea,
            "color": "#7B2CBF",
            "alpha": 0.44,
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
                "facecolor": "#E9DDF4",
                "edgecolor": "#7B2CBF",
                "alpha": 0.28,
                "linewidth": 2.0,
            },
        ],
        "points": [
            {
                "name": "p",
                "position": punto,
                "color": "#6B7280",
                "alpha": 0.6,
                "size": 70,
            },
            {
                "name": "F p",
                "position": punto_t,
                "color": "#7B2CBF",
                "size": 90,
            },
        ],
        "message": mensaje,
        "info_title": "Reflexión respecto al eje x",
        "info_lines": [
            {"text": "MATRIZ ACTUAL", "bold": True},
            m1,
            m2,
            "",
            {"text": "PUNTO", "bold": True},
            f"p       = {formatear_vector(punto)}",
            f"F p     = {formatear_vector(punto_t)}",
            "",
            {"text": "ORIENTACIÓN", "bold": True},
            f"det(F)  = {np.linalg.det(matriz):6.3f}",
            f"área original = {area_firmada(figura):6.3f}",
            f"área actual   = {area_firmada(figura_t):6.3f}",
            "",
            "Al terminar, det(Fx)=-1.",
        ],
        "phase": fase,
        "info_line_height": 0.046,
        "legend": [
            {"kind": "line", "label": "original", "color": "#6B7280"},
            {"kind": "line", "label": "reflexión", "color": "#7B2CBF"},
            {"kind": "line", "label": "cuadrícula reflejada", "color": "#7B2CBF"},
        ],
        "legend_ncol": 3,
        "legend_fontsize": 8.0,
    }


def crear_estado_cizalla(
    figura,
    punto,
    k,
    fase,
    mensaje,
):
    """Visualiza una cizalla horizontal Hx(k)."""

    figura = np.asarray(figura, dtype=float)
    punto = np.asarray(punto, dtype=float)

    matriz = np.array([
        [1.0, k],
        [0.0, 1.0],
    ])

    figura_t = transformar_puntos(figura, matriz)
    punto_t = matriz @ punto

    polylines = []

    for linea in crear_cuadricula(np.eye(2)):
        polylines.append({
            "points": linea,
            "color": "#9CA3AF",
            "alpha": 0.13,
            "linewidth": 0.8,
        })

    for linea in crear_cuadricula(matriz):
        polylines.append({
            "points": linea,
            "color": "#D97706",
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
                "facecolor": "#F2D7A7",
                "edgecolor": "#D97706",
                "alpha": 0.30,
                "linewidth": 2.0,
            },
        ],
        "points": [
            {
                "name": "p",
                "position": punto,
                "color": "#6B7280",
                "alpha": 0.60,
                "size": 70,
            },
            {
                "name": "H p",
                "position": punto_t,
                "color": "#D97706",
                "size": 90,
            },
        ],
        "message": mensaje,
        "info_title": "Cizallamiento horizontal",
        "info_lines": [
            {"text": "PARÁMETRO", "bold": True},
            f"k = {k:6.2f}",
            "",
            {"text": "MATRIZ Hx(k)", "bold": True},
            m1,
            m2,
            "",
            {"text": "PUNTO", "bold": True},
            f"p       = {formatear_vector(punto)}",
            f"H p     = {formatear_vector(punto_t)}",
            "",
            {"text": "PROPIEDADES", "bold": True},
            f"det(H)  = {np.linalg.det(matriz):6.3f}",
            "y' = y",
            "x' = x + k y",
        ],
        "phase": fase,
        "info_line_height": 0.047,
        "legend": [
            {"kind": "line", "label": "original", "color": "#6B7280"},
            {"kind": "line", "label": "cizalla", "color": "#D97706"},
            {"kind": "line", "label": "cuadrícula cizallada", "color": "#D97706"},
        ],
        "legend_ncol": 3,
        "legend_fontsize": 8.0,
    }


def crear_estado_comparacion(figura, punto):
    """Superpone original, reflexión final y cizalla final."""

    figura = np.asarray(figura, dtype=float)
    punto = np.asarray(punto, dtype=float)

    reflexion = np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ])
    cizalla = np.array([
        [1.0, 2.0],
        [0.0, 1.0],
    ])

    figura_reflejada = transformar_puntos(figura, reflexion)
    figura_cizallada = transformar_puntos(figura, cizalla)

    punto_reflejado = reflexion @ punto
    punto_cizallado = cizalla @ punto

    return {
        "polygons": [
            {
                "points": figura,
                "facecolor": "none",
                "edgecolor": "#6B7280",
                "alpha": 0.70,
                "linewidth": 1.6,
            },
            {
                "points": figura_reflejada,
                "facecolor": "#E9DDF4",
                "edgecolor": "#7B2CBF",
                "alpha": 0.20,
                "linewidth": 2.0,
            },
            {
                "points": figura_cizallada,
                "facecolor": "#F2D7A7",
                "edgecolor": "#D97706",
                "alpha": 0.20,
                "linewidth": 2.0,
            },
        ],
        "points": [
            {
                "name": "Fx p",
                "position": punto_reflejado,
                "color": "#7B2CBF",
                "size": 80,
            },
            {
                "name": "Hx p",
                "position": punto_cizallado,
                "color": "#D97706",
                "size": 80,
            },
        ],
        "message": (
            "Reflexión: invierte orientación y det=-1. Cizalla: inclina la "
            "figura, conserva paralelismo y en Hx(k) se mantiene det=1."
        ),
        "info_title": "Reflexión frente a cizalla",
        "info_lines": [
            {"text": "REFLEXIÓN Fx", "bold": True},
            "Fx = diag(1, -1)",
            f"det(Fx) = {np.linalg.det(reflexion):.1f}",
            f"Fx p = {formatear_vector(punto_reflejado)}",
            "",
            {"text": "CIZALLA Hx(2)", "bold": True},
            "Hx = [[1, 2], [0, 1]]",
            f"det(Hx) = {np.linalg.det(cizalla):.1f}",
            f"Hx p = {formatear_vector(punto_cizallado)}",
            "",
            "La reflexión cambia el signo",
            "del área firmada.",
            "La cizalla mantiene el área.",
        ],
        "phase": "Conclusión · Comparación",
        "info_line_height": 0.048,
        "legend": [
            {"kind": "line", "label": "original", "color": "#6B7280"},
            {"kind": "line", "label": "reflexión Fx", "color": "#7B2CBF"},
            {"kind": "line", "label": "cizalla Hx(2)", "color": "#D97706"},
        ],
        "legend_ncol": 3,
        "legend_fontsize": 8.0,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 2.4."""

    figura = np.array([
        [0.4, 0.4],
        [2.4, 0.4],
        [2.4, 2.1],
        [1.0, 2.8],
        [0.4, 2.1],
    ])

    # Se conserva el ejemplo numérico del subapartado de cizalla.
    punto = np.array([1.0, 3.0])

    estados = []

    for progreso in interpolar(0.0, 1.0, 105):
        suave = suavizar(progreso)
        factor_y = 1.0 - 2.0 * suave

        estados.append(
            crear_estado_reflexion(
                figura=figura,
                punto=punto,
                factor_y=factor_y,
                fase="1/3 · Reflexión respecto al eje x",
                mensaje=(
                    "La coordenada y cambia progresivamente de signo. "
                    "Al final Fx=diag(1,-1) y la orientación se invierte."
                ),
            )
        )

    for _ in range(22):
        estados.append(
            crear_estado_cizalla(
                figura=figura,
                punto=punto,
                k=0.0,
                fase="2/3 · Preparar cizallamiento",
                mensaje=(
                    "Ahora partimos de la identidad y hacemos crecer el "
                    "parámetro k de Hx(k)."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 110):
        k = 2.0 * suavizar(progreso)

        estados.append(
            crear_estado_cizalla(
                figura=figura,
                punto=punto,
                k=k,
                fase="2/3 · Cizallamiento horizontal",
                mensaje=(
                    "x'=x+k y mientras y permanece fijo. Las rectas paralelas "
                    "siguen siendo paralelas aunque cambien ángulos y longitudes."
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
    }


def imprimir_resultado(resultado):
    """Muestra por terminal las dos transformaciones finales."""

    punto = resultado["point"]

    reflexion = np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ])
    cizalla = np.array([
        [1.0, 2.0],
        [0.0, 1.0],
    ])

    print("\n=== 2.4. Reflexión y cizallamiento ===")
    print(f"\np = {formatear_vector(punto)}")
    print(f"Fx p = {formatear_vector(reflexion @ punto)}")
    print(f"Hx(2) p = {formatear_vector(cizalla @ punto)}")
    print(f"\ndet(Fx) = {np.linalg.det(reflexion):.1f}")
    print(f"det(Hx) = {np.linalg.det(cizalla):.1f}")


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
        / "04_reflexion_cizallamiento.png"
    )

    video_path = (
        MATRICES_DIR
        / "assets"
        / "02_transformaciones_2d"
        / "04_reflexion_cizallamiento.webm"
    )

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="2.4. Reflexión y cizallamiento",
        limits=(-4.2, 9.0, -4.5, 5.2),
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
