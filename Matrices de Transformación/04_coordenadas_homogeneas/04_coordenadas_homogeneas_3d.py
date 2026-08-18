from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def suavizar(progreso):
    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def formatear_vector(v):
    v = np.asarray(v, dtype=float)
    return "[" + ", ".join(f"{x:5.2f}" for x in v) + "]"


def matriz_homogenea_3d(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def crear_estado_rotacion(theta, P, fase, mensaje):
    R = rz(theta)
    RP = R @ P

    return {
        "frames3d": [
            {
                "name": "0",
                "origin": np.zeros(3),
                "rotation": np.eye(3),
                "length": 1.5,
                "alpha": 0.20,
                "colors": ("#9CA3AF", "#9CA3AF", "#9CA3AF"),
            },
            {
                "name": "R",
                "origin": np.zeros(3),
                "rotation": R,
                "length": 1.8,
                "alpha": 1.0,
            },
        ],
        "points3d": [
            {"name": "P", "position": P, "color": "#6B7280", "alpha": 0.35, "size": 45},
            {"name": "RP", "position": RP, "color": "#7B2CBF", "size": 75},
        ],
        "vectors3d": [
            {"name": "P", "origin": np.zeros(3), "value": P, "color": "#6B7280", "alpha": 0.30, "linewidth": 1.8},
            {"name": "RP", "origin": np.zeros(3), "value": RP, "color": "#7B2CBF", "linewidth": 2.8},
        ],
        "message": mensaje,
        "info_title": "Paso 1: rotación alrededor de Z",
        "info_lines": [
            {"text": "Rz(theta)", "bold": True},
            f"theta = {np.degrees(theta):6.1f}°",
            "",
            f"P  = {formatear_vector(P)}",
            f"RP = {formatear_vector(RP)}",
            "",
            "Con 90°:",
            "[X,Y,Z] -> [-Y,X,Z]",
            "",
            "La coordenada Z",
            "permanece inalterada.",
        ],
        "phase": fase,
        "info_line_height": 0.050,
        "legend": [
            {"kind": "line", "label": "P original", "color": "#6B7280"},
            {"kind": "line", "label": "RP", "color": "#7B2CBF"},
        ],
        "legend_ncol": 1,
    }


def crear_estado_traslacion(R, t_actual, t_final, P, fase, mensaje):
    RP = R @ P
    P_actual = RP + t_actual

    return {
        "frames3d": [
            {
                "name": "R",
                "origin": np.zeros(3),
                "rotation": R,
                "length": 1.5,
                "alpha": 0.18,
                "colors": ("#9CA3AF", "#9CA3AF", "#9CA3AF"),
            },
            {
                "name": "T",
                "origin": t_actual,
                "rotation": R,
                "length": 1.8,
                "alpha": 1.0,
            },
        ],
        "points3d": [
            {"name": "RP", "position": RP, "color": "#6B7280", "alpha": 0.35, "size": 45},
            {"name": "RP+t", "position": P_actual, "color": "#7B2CBF", "size": 75},
        ],
        "vectors3d": [
            {"name": "t", "origin": np.zeros(3), "value": t_actual, "color": "#E07A1F", "linewidth": 3.0},
        ],
        "segments3d": [
            {"start": RP, "end": RP + t_final, "color": "#7B2CBF", "alpha": 0.25, "linestyle": "--"},
        ],
        "message": mensaje,
        "info_title": "Paso 2: traslación",
        "info_lines": [
            {"text": "DATOS", "bold": True},
            f"RP = {formatear_vector(RP)}",
            f"t  = {formatear_vector(t_final)}",
            "",
            {"text": "ESTADO ACTUAL", "bold": True},
            f"t(t)  = {formatear_vector(t_actual)}",
            f"P'(t) = {formatear_vector(P_actual)}",
            "",
            "El frame final conserva",
            "la orientación R y su",
            "origen pasa a t.",
        ],
        "phase": fase,
        "info_line_height": 0.050,
        "legend": [
            {"kind": "line", "label": "vector t", "color": "#E07A1F"},
            {"kind": "point", "label": "punto transformado", "color": "#7B2CBF"},
        ],
        "legend_ncol": 1,
    }


def crear_estado_conclusion(R, t, P):
    T = matriz_homogenea_3d(R, t)
    P_h = np.r_[P, 1.0]
    P_h_out = T @ P_h
    P_secuencial = R @ P + t

    return {
        "frames3d": [
            {
                "name": "0",
                "origin": np.zeros(3),
                "rotation": np.eye(3),
                "length": 1.35,
                "alpha": 0.18,
                "colors": ("#9CA3AF", "#9CA3AF", "#9CA3AF"),
            },
            {
                "name": "T",
                "origin": t,
                "rotation": R,
                "length": 1.9,
                "alpha": 1.0,
            },
        ],
        "points3d": [
            {"name": "P", "position": P, "color": "#6B7280", "alpha": 0.35, "size": 45},
            {"name": "secuencial", "position": P_secuencial, "color": "#D97706", "size": 85},
            {
                "name": "T P_h",
                "position": P_h_out[:3],
                "color": "#7B2CBF",
                "size": 36,
                "label_offset": (0.14, -0.18, 0.12),
            },
        ],
        "vectors3d": [
            {"name": "t", "origin": np.zeros(3), "value": t, "color": "#E07A1F", "linewidth": 2.8},
        ],
        "message": (
            "Rotar y después trasladar produce exactamente el mismo resultado que "
            "multiplicar una sola vez por la matriz homogénea 4x4."
        ),
        "info_title": "Matriz homogénea 4x4",
        "info_lines": [
            {"text": "T", "bold": True},
            "[ 0, -1,  0,  4]",
            "[ 1,  0,  0,  0]",
            "[ 0,  0,  1, -1]",
            "[ 0,  0,  0,  1]",
            "",
            f"P_h = {formatear_vector(P_h)}",
            f"T P = {formatear_vector(P_h_out)}",
            "",
            f"RP+t= {formatear_vector(P_secuencial)}",
            "",
            "Ambos métodos coinciden.",
        ],
        "phase": "Conclusión",
        "info_line_height": 0.0435,
        "info_fontsize": 9.0,
        "legend": [
            {"kind": "point", "label": "RP+t", "color": "#D97706"},
            {"kind": "point", "label": "T P_h", "color": "#7B2CBF"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.1,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 4.4 con el ejemplo exacto de la wiki."""

    theta_final = np.radians(90.0)
    R = rz(theta_final)
    t = np.array([4.0, 0.0, -1.0])
    P = np.array([1.0, 2.0, 3.0])

    estados = []

    for _ in range(25):
        estados.append(crear_estado_rotacion(0.0, P, "1/4 · Punto homogéneo 3D", "Partimos de P=[1,2,3] y lo representaremos después como [1,2,3,1]."))

    for progreso in np.linspace(0.0, 1.0, 95):
        theta = suavizar(progreso) * theta_final
        estados.append(crear_estado_rotacion(theta, P, "2/4 · Rz(90°)", "Primero aplicamos la rotación activa alrededor de Z. El punto pasa de [1,2,3] a [-2,1,3]."))

    for progreso in np.linspace(0.0, 1.0, 95):
        t_actual = suavizar(progreso) * t
        estados.append(crear_estado_traslacion(R, t_actual, t, P, "3/4 · Añadir t", "Después sumamos t=[4,0,-1]. La orientación del frame ya no cambia; solo se desplaza su origen."))

    for _ in range(35):
        estados.append(crear_estado_traslacion(R, t, t, P, "4/4 · Resultado secuencial", "El resultado secuencial es RP+t=[2,1,2]. Ahora lo compararemos con la matriz 4x4 completa."))

    for _ in range(60):
        estados.append(crear_estado_conclusion(R, t, P))

    return {"states": estados, "R": R, "t": t, "P": P}


def imprimir_resultado(resultado):
    R, t, P = resultado["R"], resultado["t"], resultado["P"]
    T = matriz_homogenea_3d(R, t)
    P_h = np.r_[P, 1.0]
    print("\n=== 4.4. Coordenadas homogéneas en 3D ===")
    print("\nT =\n", T)
    print("RP+t =", R @ P + t)
    print("T P_h =", T @ P_h)


def main():
    resultado = crear_estados_demostracion()
    imprimir_resultado(resultado)

    animador = TransformAnimator(figsize=(15.5, 8.8), interval=50)
    image_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "04_coordenadas_homogeneas_3d.png"
    video_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "04_coordenadas_homogeneas_3d.webm"

    animacion = animador.animate_3d_states(
        states=resultado["states"],
        title="4.4. Coordenadas homogéneas en 3D",
        limits=(-3.2, 6.2, -3.2, 4.2, -2.0, 5.2),
        view=(24.0, -58.0),
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
