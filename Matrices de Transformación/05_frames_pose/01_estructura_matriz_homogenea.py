from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def suavizar(p):
    return 0.5 - 0.5 * np.cos(np.pi * p)


def T3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def formatear(v):
    v = np.asarray(v, dtype=float)
    return "[" + ", ".join(f"{x:5.2f}" for x in v) + "]"


def crear_estado_2d(progreso, fase, mensaje):
    theta = suavizar(progreso) * np.radians(35.0)
    t = suavizar(progreso) * np.array([2.0, -1.0])
    R2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    H = np.eye(3)
    H[:2, :2] = R2
    H[:2, 2] = t

    p_B = np.array([1.2, 0.7])
    p_A = R2 @ p_B + t
    R3 = np.eye(3)
    R3[:2, :2] = R2

    return {
        "frames3d": [
            {"name": "A", "origin": np.zeros(3), "rotation": np.eye(3), "length": 1.45, "alpha": 0.25,
             "colors": ("#9CA3AF", "#9CA3AF", "#9CA3AF")},
            {"name": "B", "origin": np.r_[t, 0.0], "rotation": R3, "length": 1.35, "alpha": 1.0},
        ],
        "points3d": [
            {"name": "p", "position": np.r_[p_A, 0.0], "color": "#7B2CBF", "size": 75},
        ],
        "segments3d": [
            {"start": np.r_[t, 0.0], "end": np.r_[p_A, 0.0], "color": "#7B2CBF", "alpha": 0.55, "linestyle": "--"},
        ],
        "message": mensaje,
        "info_title": "Estructura homogénea 2D",
        "info_lines": [
            {"text": "H = [ R | t ]", "bold": True},
            f"[{H[0,0]:5.2f}, {H[0,1]:5.2f}, {H[0,2]:5.2f}]",
            f"[{H[1,0]:5.2f}, {H[1,1]:5.2f}, {H[1,2]:5.2f}]",
            f"[{H[2,0]:5.2f}, {H[2,1]:5.2f}, {H[2,2]:5.2f}]",
            "",
            {"text": "PUNTO", "bold": True},
            f"p_B = {formatear(p_B)}",
            f"p_A = {formatear(p_A)}",
            "",
            "bloque 2x2 -> orientación",
            "columna t -> posición",
            "última fila -> homogénea",
        ],
        "phase": fase,
        "info_line_height": 0.047,
        "legend": [
            {"kind": "point", "label": "punto expresado en A", "color": "#7B2CBF"},
        ],
        "legend_fontsize": 8.2,
    }


def crear_estado_3d(progreso, fase, mensaje):
    s = suavizar(progreso)
    R = rz(s * np.radians(40.0)) @ ry(s * np.radians(-20.0)) @ rx(s * np.radians(15.0))
    t = s * np.array([2.0, -1.0, 1.2])
    T = T3(R, t)
    p_B = np.array([1.0, 0.5, 0.8])
    p_A = R @ p_B + t

    return {
        "frames3d": [
            {"name": "A", "origin": np.zeros(3), "rotation": np.eye(3), "length": 1.35, "alpha": 0.20,
             "colors": ("#9CA3AF", "#9CA3AF", "#9CA3AF")},
            {"name": "B", "origin": t, "rotation": R, "length": 1.65, "alpha": 1.0},
        ],
        "points3d": [
            {"name": "p", "position": p_A, "color": "#7B2CBF", "size": 78},
        ],
        "segments3d": [
            {"start": t, "end": p_A, "color": "#7B2CBF", "alpha": 0.55, "linestyle": "--"},
        ],
        "vectors3d": [
            {"name": "t", "origin": np.zeros(3), "value": t, "color": "#E07A1F", "linewidth": 2.8},
        ],
        "message": mensaje,
        "info_title": "Estructura homogénea 3D",
        "info_lines": [
            {"text": "T = [ R | t ]", "bold": True},
            f"[{T[0,0]:5.2f},{T[0,1]:5.2f},{T[0,2]:5.2f},{T[0,3]:5.2f}]",
            f"[{T[1,0]:5.2f},{T[1,1]:5.2f},{T[1,2]:5.2f},{T[1,3]:5.2f}]",
            f"[{T[2,0]:5.2f},{T[2,1]:5.2f},{T[2,2]:5.2f},{T[2,3]:5.2f}]",
            "[ 0.00, 0.00, 0.00, 1.00]",
            "",
            f"p_B = {formatear(p_B)}",
            f"p_A = {formatear(p_A)}",
            "",
            "p_A = R p_B + t",
            "T agrupa ambos términos.",
        ],
        "phase": fase,
        "info_line_height": 0.043,
        "info_fontsize": 8.9,
        "legend": [
            {"kind": "line", "label": "traslación t", "color": "#E07A1F"},
            {"kind": "point", "label": "punto transformado", "color": "#7B2CBF"},
        ],
        "legend_fontsize": 8.0,
    }


def crear_estados_demostracion():
    estados = []
    for _ in range(24):
        estados.append(crear_estado_2d(0.0, "1/4 · Identidad 2D", "Una matriz homogénea 2D separa conceptualmente orientación, traslación y la fila homogénea."))
    for p in np.linspace(0.0, 1.0, 85):
        estados.append(crear_estado_2d(p, "2/4 · Construir H 3x3", "El bloque R orienta el frame {B}; la columna t sitúa su origen. La misma H transforma puntos homogéneos."))
    for _ in range(28):
        estados.append(crear_estado_2d(1.0, "2/4 · H 2D completa", "En 2D, una pose rígida se representa con una matriz homogénea 3x3."))
    for p in np.linspace(0.0, 1.0, 105):
        estados.append(crear_estado_3d(p, "3/4 · Extensión a 3D", "En 3D la misma idea produce una matriz 4x4: R ocupa 3x3, t ocupa la cuarta columna y la última fila es [0,0,0,1]."))
    for _ in range(55):
        estados.append(crear_estado_3d(1.0, "4/4 · T 4x4", "La ecuación central queda p_A = R p_B + t, contenida por completo en una sola matriz homogénea."))
    return {"states": estados}


def main():
    resultado = crear_estados_demostracion()
    animador = TransformAnimator(figsize=(15.5, 8.8), interval=50)
    image_path = MATRICES_DIR / "assets" / "05_frames_pose" / "01_estructura_matriz_homogenea.png"
    video_path = MATRICES_DIR / "assets" / "05_frames_pose" / "01_estructura_matriz_homogenea.webm"
    animador.animate_3d_states(
        states=resultado["states"],
        title="5.1. Estructura de una matriz homogénea 2D y 3D",
        limits=(-2.5, 5.0, -3.5, 3.2, -2.0, 4.0),
        view=(24.0, -58.0),
        final_image_path=image_path,
        video_path=video_path,
        repeat=False,
        fps=20,
        dpi=125,
        show=True,
    )


if __name__ == "__main__":
    main()
