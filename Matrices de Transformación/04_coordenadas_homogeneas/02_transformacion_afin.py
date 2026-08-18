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
    return f"[{v[0]:6.2f}, {v[1]:6.2f}]"


def formatear_matriz(A):
    A = np.asarray(A, dtype=float)
    return (
        f"[{A[0,0]:6.3f}, {A[0,1]:6.3f}]",
        f"[{A[1,0]:6.3f}, {A[1,1]:6.3f}]",
    )


def crear_figura():
    return np.array([
        [-0.5, -0.4],
        [1.2, -0.35],
        [1.45, 0.55],
        [0.25, 1.25],
        [-0.8, 0.55],
    ], dtype=float)


def aplicar_afin(puntos, A, t):
    puntos = np.asarray(puntos, dtype=float)
    return (A @ puntos.T).T + t


def crear_estado_primera(A1_actual, A1_final, t1_actual, t1_final, p, fase, mensaje):
    figura = crear_figura()
    figura_1 = aplicar_afin(figura, A1_actual, t1_actual)
    p1 = A1_actual @ p + t1_actual

    a11, a12 = formatear_matriz(A1_actual)

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
                "points": figura_1,
                "facecolor": "#BFDBFE",
                "edgecolor": "#2563EB",
                "alpha": 0.34,
                "linewidth": 1.5,
            },
        ],
        "points": [
            {"name": "p", "position": p, "color": "#6B7280", "alpha": 0.45, "size": 55},
            {"name": "p2", "position": p1, "color": "#7B2CBF", "size": 85},
        ],
        "vectors": [
            {
                "name": "t1",
                "origin": np.zeros(2),
                "value": t1_actual,
                "color": "#E07A1F",
                "linewidth": 2.8,
            }
        ],
        "message": mensaje,
        "info_title": "Primera transformación afín",
        "info_lines": [
            {"text": "p2 = A1 p1 + t1", "bold": True},
            "",
            {"text": "A1(t)", "bold": True},
            a11,
            a12,
            "",
            f"t1(t) = {formatear_vector(t1_actual)}",
            "",
            f"p1 = {formatear_vector(p)}",
            f"p2 = {formatear_vector(p1)}",
            "",
            "A actúa primero sobre",
            "la geometría y t desplaza",
            "el resultado.",
        ],
        "phase": fase,
        "info_line_height": 0.046,
        "legend": [
            {"kind": "line", "label": "figura original", "color": "#64748B"},
            {"kind": "line", "label": "A1 p + t1", "color": "#2563EB"},
            {"kind": "line", "label": "t1", "color": "#E07A1F"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.2,
    }


def crear_estado_segunda(A1, t1, A2_actual, A2_final, t2_actual, t2_final, p, fase, mensaje):
    figura = crear_figura()
    figura_1 = aplicar_afin(figura, A1, t1)
    figura_2 = aplicar_afin(figura_1, A2_actual, t2_actual)

    p2 = A1 @ p + t1
    p3 = A2_actual @ p2 + t2_actual

    a21, a22 = formatear_matriz(A2_actual)

    return {
        "polygons": [
            {
                "points": figura_1,
                "facecolor": "#BFDBFE",
                "edgecolor": "#2563EB",
                "alpha": 0.10,
                "linewidth": 1.0,
            },
            {
                "points": figura_2,
                "facecolor": "#FDE68A",
                "edgecolor": "#D97706",
                "alpha": 0.36,
                "linewidth": 1.5,
            },
        ],
        "points": [
            {"name": "p2", "position": p2, "color": "#2563EB", "alpha": 0.45, "size": 50},
            {"name": "p3", "position": p3, "color": "#7B2CBF", "size": 85},
        ],
        "vectors": [
            {
                "name": "t2",
                "origin": np.zeros(2),
                "value": t2_actual,
                "color": "#E07A1F",
                "linewidth": 2.8,
            }
        ],
        "message": mensaje,
        "info_title": "Segunda transformación afín",
        "info_lines": [
            {"text": "p3 = A2 p2 + t2", "bold": True},
            "",
            {"text": "A2(t)", "bold": True},
            a21,
            a22,
            "",
            f"t2(t) = {formatear_vector(t2_actual)}",
            "",
            f"p2 = {formatear_vector(p2)}",
            f"p3 = {formatear_vector(p3)}",
            "",
            "La segunda matriz también",
            "actúa sobre la traslación",
            "que venía de la primera.",
        ],
        "phase": fase,
        "info_line_height": 0.045,
        "legend": [
            {"kind": "line", "label": "tras primera afín", "color": "#2563EB"},
            {"kind": "line", "label": "tras segunda afín", "color": "#D97706"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.2,
    }


def crear_estado_composicion(A1, t1, A2, t2, p):
    figura = crear_figura()
    secuencial = aplicar_afin(aplicar_afin(figura, A1, t1), A2, t2)

    A3 = A2 @ A1
    t3 = A2 @ t1 + t2
    compuesta = aplicar_afin(figura, A3, t3)

    p_secuencial = A2 @ (A1 @ p + t1) + t2
    p_compuesto = A3 @ p + t3

    a31, a32 = formatear_matriz(A3)

    return {
        "polygons": [
            {
                "points": secuencial,
                "facecolor": "#FDE68A",
                "edgecolor": "#D97706",
                "alpha": 0.30,
                "linewidth": 2.0,
            },
            {
                "points": compuesta,
                "facecolor": "none",
                "edgecolor": "#7B2CBF",
                "alpha": 0.95,
                "linewidth": 1.8,
                "linestyle": "--",
            },
        ],
        "points": [
            {"name": "secuencial", "position": p_secuencial, "color": "#D97706", "size": 85},
            {
                "name": "compuesta",
                "position": p_compuesto,
                "color": "#7B2CBF",
                "size": 38,
                "label_offset": (0.15, -0.26),
            },
        ],
        "message": (
            "Las dos transformaciones pueden resumirse en A3=A2A1 y "
            "t3=A2t1+t2, pero todavía debemos transportar matriz y vector por separado."
        ),
        "info_title": "Composición sin homogéneas",
        "info_lines": [
            {"text": "RESULTADO COMPUESTO", "bold": True},
            "A3 = A2 A1",
            a31,
            a32,
            "",
            "t3 = A2 t1 + t2",
            f"t3 = {formatear_vector(t3)}",
            "",
            f"p3 sec. = {formatear_vector(p_secuencial)}",
            f"p3 comp.= {formatear_vector(p_compuesto)}",
            "",
            {"text": "PROBLEMA", "bold": True},
            "Hay que conservar A y t",
            "como dos objetos distintos.",
            "Queremos una sola matriz H.",
        ],
        "phase": "Conclusión",
        "info_line_height": 0.0415,
        "info_fontsize": 9.1,
        "legend": [
            {"kind": "line", "label": "aplicación secuencial", "color": "#D97706"},
            {"kind": "line", "label": "A3 p + t3", "color": "#7B2CBF", "linestyle": "--"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.1,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 4.2."""

    A1 = rotacion_2d(np.radians(35.0))
    t1 = np.array([1.5, 0.6])
    A2 = np.array([[1.15, 0.40], [0.0, 0.80]])
    t2 = np.array([-0.8, 1.4])
    p = np.array([1.0, 0.7])

    estados = []
    I = np.eye(2)

    for _ in range(28):
        estados.append(
            crear_estado_primera(I, A1, np.zeros(2), t1, p, "1/4 · Forma afín", "Partimos de p'=Ap+t: una parte lineal y una traslación separadas.")
        )

    for progreso in np.linspace(0.0, 1.0, 90):
        s = suavizar(progreso)
        A_actual = (1.0 - s) * I + s * A1
        t_actual = s * t1
        estados.append(
            crear_estado_primera(A_actual, A1, t_actual, t1, p, "2/4 · Primera afín", "Aplicamos progresivamente A1 y t1. El punto final es p2=A1p1+t1.")
        )

    for progreso in np.linspace(0.0, 1.0, 95):
        s = suavizar(progreso)
        A2_actual = (1.0 - s) * I + s * A2
        t2_actual = s * t2
        estados.append(
            crear_estado_segunda(A1, t1, A2_actual, A2, t2_actual, t2, p, "3/4 · Segunda afín", "La segunda transformación actúa sobre todo p2, incluida la traslación generada en el primer paso.")
        )

    for _ in range(35):
        estados.append(
            crear_estado_segunda(A1, t1, A2, A2, t2, t2, p, "4/4 · Sustitución", "Al sustituir p2 en p3 aparece A2A1p1 + A2t1 + t2.")
        )

    for _ in range(55):
        estados.append(crear_estado_composicion(A1, t1, A2, t2, p))

    return {"states": estados, "A1": A1, "t1": t1, "A2": A2, "t2": t2, "p": p}


def imprimir_resultado(resultado):
    A1, t1, A2, t2, p = resultado["A1"], resultado["t1"], resultado["A2"], resultado["t2"], resultado["p"]
    A3 = A2 @ A1
    t3 = A2 @ t1 + t2
    p3_seq = A2 @ (A1 @ p + t1) + t2
    p3_cmp = A3 @ p + t3

    print("\n=== 4.2. Transformación afín: p' = Ap + t ===")
    print("\nA3 = A2 A1 =\n", A3)
    print("t3 = A2 t1 + t2 =", t3)
    print("p3 secuencial =", p3_seq)
    print("p3 compuesta   =", p3_cmp)
    print("error =", np.linalg.norm(p3_seq - p3_cmp))


def main():
    resultado = crear_estados_demostracion()
    imprimir_resultado(resultado)

    animador = TransformAnimator(figsize=(15.5, 8.8), interval=50)
    image_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "02_transformacion_afin.png"
    video_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "02_transformacion_afin.webm"

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="4.2. Transformación afín: p' = Ap + t",
        limits=(-3.0, 5.2, -2.6, 5.0),
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
