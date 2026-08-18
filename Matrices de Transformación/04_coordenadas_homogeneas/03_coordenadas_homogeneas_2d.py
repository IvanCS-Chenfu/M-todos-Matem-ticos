from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rotacion_2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def suavizar(progreso):
    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def formatear_vector(v):
    v = np.asarray(v, dtype=float)
    return "[" + ", ".join(f"{x:6.2f}" for x in v) + "]"


def matriz_homogenea_2d(A, t):
    H = np.eye(3)
    H[:2, :2] = A
    H[:2, 2] = t
    return H


def crear_figura():
    return np.array([
        [1.1, 1.8],
        [2.9, 1.8],
        [3.3, 2.8],
        [2.2, 3.7],
        [1.0, 3.0],
    ], dtype=float)


def crear_estado_equivalencia(p, lambda_actual, fase, mensaje):
    p_h = np.array([lambda_actual * p[0], lambda_actual * p[1], lambda_actual])
    normalizado = p_h[:2] / p_h[2]

    return {
        "points": [
            {"name": "p cartesiano", "position": p, "color": "#7B2CBF", "size": 100},
        ],
        "segments": [
            {
                "start": np.zeros(2),
                "end": p,
                "color": "#7B2CBF",
                "alpha": 0.30,
                "linestyle": "--",
            }
        ],
        "message": mensaje,
        "info_title": "Equivalencia homogénea",
        "info_lines": [
            {"text": "PUNTO CARTESIANO", "bold": True},
            f"p = {formatear_vector(p)}",
            "",
            {"text": "REPRESENTACIÓN HOMOGÉNEA", "bold": True},
            f"lambda = {lambda_actual:5.2f}",
            f"p_h = {formatear_vector(p_h)}",
            "",
            {"text": "NORMALIZACIÓN", "bold": True},
            f"p_h / w -> {formatear_vector(normalizado)}",
            "",
            "Cambiar lambda modifica",
            "el triplete homogéneo,",
            "no el punto representado.",
        ],
        "phase": fase,
        "info_line_height": 0.048,
        "legend": [
            {"kind": "point", "label": "mismo punto físico", "color": "#7B2CBF"},
        ],
        "legend_ncol": 1,
    }


def crear_estado_transformacion(A_actual, A_final, t_actual, t_final, p, fase, mensaje):
    H = matriz_homogenea_2d(A_actual, t_actual)
    p_h = np.array([p[0], p[1], 1.0])

    p_afin = A_actual @ p + t_actual
    p_h_resultado = H @ p_h
    p_h_cart = p_h_resultado[:2] / p_h_resultado[2]

    figura = crear_figura()
    figura_afin = (A_actual @ figura.T).T + t_actual

    return {
        "polygons": [
            {
                "points": figura,
                "facecolor": "#CBD5E1",
                "edgecolor": "#64748B",
                "alpha": 0.10,
                "linewidth": 1.0,
            },
            {
                "points": figura_afin,
                "facecolor": "#BFDBFE",
                "edgecolor": "#2563EB",
                "alpha": 0.32,
                "linewidth": 1.5,
            },
        ],
        "points": [
            {"name": "p", "position": p, "color": "#6B7280", "alpha": 0.45, "size": 55},
            {"name": "Ap+t", "position": p_afin, "color": "#D97706", "size": 95},
            {
                "name": "H p_h",
                "position": p_h_cart,
                "color": "#7B2CBF",
                "size": 38,
                "label_offset": (0.15, -0.28),
            },
        ],
        "vectors": [
            {"name": "t", "origin": np.zeros(2), "value": t_actual, "color": "#E07A1F", "linewidth": 2.8},
        ],
        "message": mensaje,
        "info_title": "Ap+t frente a H p_h",
        "info_lines": [
            {"text": "COORDENADAS", "bold": True},
            f"p   = {formatear_vector(p)}",
            f"p_h = {formatear_vector(p_h)}",
            "",
            {"text": "CÁLCULO AFÍN", "bold": True},
            f"Ap+t = {formatear_vector(p_afin)}",
            "",
            {"text": "CÁLCULO HOMOGÉNEO", "bold": True},
            f"H p_h = {formatear_vector(p_h_resultado)}",
            f"cart. = {formatear_vector(p_h_cart)}",
            "",
            "La tercera coordenada = 1",
            "hace entrar t dentro",
            "del producto matricial.",
        ],
        "phase": fase,
        "info_line_height": 0.0435,
        "info_fontsize": 9.1,
        "legend": [
            {"kind": "point", "label": "resultado Ap+t", "color": "#D97706"},
            {"kind": "point", "label": "resultado H p_h", "color": "#7B2CBF"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.2,
    }


def crear_estado_conclusion(A, t, p):
    H = matriz_homogenea_2d(A, t)
    p_h = np.r_[p, 1.0]
    resultado = H @ p_h
    p_cart = resultado[:2]

    figura = crear_figura()
    figura_final = (A @ figura.T).T + t

    return {
        "polygons": [
            {
                "points": figura_final,
                "facecolor": "#BFDBFE",
                "edgecolor": "#2563EB",
                "alpha": 0.34,
                "linewidth": 1.5,
            }
        ],
        "points": [
            {"name": "p'", "position": p_cart, "color": "#7B2CBF", "size": 100},
        ],
        "vectors": [
            {"name": "t", "origin": np.zeros(2), "value": t, "color": "#E07A1F", "linewidth": 2.8},
        ],
        "message": (
            "La matriz 3x3 reúne el bloque lineal A, la traslación t y una última "
            "fila que mantiene w=1 para transformaciones afines."
        ),
        "info_title": "Matriz homogénea 2D",
        "info_lines": [
            {"text": "H = [ A  t ]", "bold": True},
            f"[{H[0,0]:6.3f}, {H[0,1]:6.3f}, {H[0,2]:6.3f}]",
            f"[{H[1,0]:6.3f}, {H[1,1]:6.3f}, {H[1,2]:6.3f}]",
            f"[{H[2,0]:6.3f}, {H[2,1]:6.3f}, {H[2,2]:6.3f}]",
            "",
            {"text": "BLOQUES", "bold": True},
            "A: transformación lineal",
            f"t: {formatear_vector(t)}",
            "última fila: [0, 0, 1]",
            "",
            f"p_h  = {formatear_vector(p_h)}",
            f"H p_h= {formatear_vector(resultado)}",
        ],
        "phase": "Conclusión",
        "info_line_height": 0.043,
        "info_fontsize": 9.2,
        "legend": [
            {"kind": "line", "label": "figura transformada", "color": "#2563EB"},
            {"kind": "line", "label": "traslación t", "color": "#E07A1F"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.2,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 4.3."""

    p_equiv = np.array([2.0, 3.0])
    A = rotacion_2d(np.radians(35.0))
    t = np.array([2.0, -1.0])
    p = np.array([2.0, 1.2])
    I = np.eye(2)

    estados = []

    for _ in range(25):
        estados.append(
            crear_estado_equivalencia(p_equiv, 1.0, "1/4 · Añadir w=1", "El punto [x,y]^T pasa a [x,y,1]^T sin convertir físicamente el plano en 3D.")
        )

    for lambda_actual in np.linspace(1.0, 5.0, 85):
        estados.append(
            crear_estado_equivalencia(p_equiv, lambda_actual, "2/4 · Equivalencia por escala", "[2,3,1], [4,6,2] y [10,15,5] describen el mismo punto porque al dividir entre w se recupera [2,3].")
        )

    for progreso in np.linspace(0.0, 1.0, 100):
        s = suavizar(progreso)
        A_actual = (1.0 - s) * I + s * A
        t_actual = s * t
        estados.append(
            crear_estado_transformacion(A_actual, A, t_actual, t, p, "3/4 · Construir H", "Comparamos Ap+t con H p_h mientras aparecen simultáneamente la parte lineal y la traslación.")
        )

    for _ in range(35):
        estados.append(
            crear_estado_transformacion(A, A, t, t, p, "4/4 · Mismo resultado", "Los dos cálculos coinciden: H permite absorber la suma externa dentro de un único producto matricial.")
        )

    for _ in range(55):
        estados.append(crear_estado_conclusion(A, t, p))

    return {"states": estados, "A": A, "t": t, "p": p}


def imprimir_resultado(resultado):
    A, t, p = resultado["A"], resultado["t"], resultado["p"]
    H = matriz_homogenea_2d(A, t)
    p_h = np.r_[p, 1.0]
    print("\n=== 4.3. Coordenadas homogéneas en 2D ===")
    print("\nH =\n", H)
    print("Ap+t =", A @ p + t)
    print("H p_h =", H @ p_h)
    print("error =", np.linalg.norm((A @ p + t) - (H @ p_h)[:2]))


def main():
    resultado = crear_estados_demostracion()
    imprimir_resultado(resultado)

    animador = TransformAnimator(figsize=(15.5, 8.8), interval=50)
    image_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "03_coordenadas_homogeneas_2d.png"
    video_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "03_coordenadas_homogeneas_2d.webm"

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="4.3. Coordenadas homogéneas en 2D",
        limits=(-2.5, 6.0, -3.0, 5.5),
        final_image_path=image_path,
        video_path=video_path,
        repeat=False,
        fps=20,
        dpi=130,
        show=True,
    )
    _ = animacion


if __name__ == "__main__":
    main()
