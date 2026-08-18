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


def formatear(v):
    v = np.asarray(v, dtype=float)
    return "[" + ", ".join(f"{x:5.2f}" for x in v) + "]"


def matriz_homogenea(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def crear_estado(R, t_actual, t_final, p, v, fase, mensaje):
    T = matriz_homogenea(R, t_actual)
    p_h = np.r_[p, 1.0]
    v_h = np.r_[v, 0.0]

    p_out_h = T @ p_h
    v_out_h = T @ v_h

    p_out = p_out_h[:3]
    v_out = v_out_h[:3]

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
                "origin": t_actual,
                "rotation": R,
                "length": 1.6,
                "alpha": 0.85,
            },
        ],
        "points3d": [
            {"name": "P", "position": p, "color": "#6B7280", "alpha": 0.30, "size": 42},
            {"name": "P'", "position": p_out, "color": "#7B2CBF", "size": 75},
        ],
        "vectors3d": [
            {
                "name": "v",
                "origin": np.zeros(3),
                "value": v,
                "color": "#6B7280",
                "alpha": 0.30,
                "linewidth": 1.8,
            },
            {
                "name": "v' = Rv",
                "origin": np.zeros(3),
                "value": v_out,
                "color": "#2D7F5E",
                "linewidth": 3.0,
            },
            {
                "name": "t",
                "origin": np.zeros(3),
                "value": t_actual,
                "color": "#E07A1F",
                "linewidth": 2.8,
            },
        ],
        "message": mensaje,
        "info_title": "Puntos y direcciones: la coordenada w",
        "info_lines": [
            {"text": "PUNTO: w = 1", "bold": True},
            f"P_h  = {formatear(p_h)}",
            f"T P_h= {formatear(p_out_h)}",
            "",
            {"text": "DIRECCIÓN: w = 0", "bold": True},
            f"v_h  = {formatear(v_h)}",
            f"T v_h= {formatear(v_out_h)}",
            "",
            {"text": "TRASLACIÓN", "bold": True},
            f"t(t) = {formatear(t_actual)}",
            "",
            "t se multiplica por w:",
            "punto -> t·1 aparece",
            "dirección -> t·0 desaparece",
        ],
        "phase": fase,
        "info_line_height": 0.041,
        "info_fontsize": 8.9,
        "legend": [
            {"kind": "point", "label": "punto P'", "color": "#7B2CBF"},
            {"kind": "line", "label": "dirección Rv", "color": "#2D7F5E"},
            {"kind": "line", "label": "traslación t", "color": "#E07A1F"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.0,
    }


def crear_estado_conclusion(R, t, p, v):
    estado = crear_estado(
        R=R,
        t_actual=t,
        t_final=t,
        p=p,
        v=v,
        fase="Conclusión",
        mensaje=(
            "La misma matriz T distingue automáticamente puntos y direcciones: "
            "w=1 activa la traslación y w=0 la elimina."
        ),
    )

    T = matriz_homogenea(R, t)
    p_out = T @ np.r_[p, 1.0]
    v_out = T @ np.r_[v, 0.0]

    estado["info_lines"] = [
        {"text": "MISMA MATRIZ T", "bold": True},
        "T = [ R  t ]",
        "    [ 0  1 ]",
        "",
        {"text": "PUNTO", "bold": True},
        f"[p,1] -> {formatear(p_out)}",
        "resultado: Rp + t",
        "",
        {"text": "DIRECCIÓN", "bold": True},
        f"[v,0] -> {formatear(v_out)}",
        "resultado: Rv",
        "",
        "La posición depende del",
        "origen; la dirección no.",
    ]
    estado["info_line_height"] = 0.045
    return estado


def crear_estados_demostracion():
    """Construye la animación del apartado 4.5."""

    p = np.array([1.2, 1.5, 0.8])
    v = np.array([1.0, 0.5, 1.0])
    t = np.array([2.5, -1.2, 1.0])
    theta_final = np.radians(55.0)

    estados = []

    for _ in range(28):
        estados.append(
            crear_estado(np.eye(3), np.zeros(3), t, p, v, "1/4 · w=1 frente a w=0", "Partimos del mismo T conceptual, pero codificamos el punto con w=1 y la dirección con w=0.")
        )

    for progreso in np.linspace(0.0, 1.0, 95):
        t_actual = suavizar(progreso) * t
        estados.append(
            crear_estado(np.eye(3), t_actual, t, p, v, "2/4 · Solo traslación", "Al crecer t, el punto se desplaza. La dirección no cambia porque sus términos de traslación se multiplican por w=0.")
        )

    for progreso in np.linspace(0.0, 1.0, 100):
        theta = suavizar(progreso) * theta_final
        R = rz(theta)
        estados.append(
            crear_estado(R, t, t, p, v, "3/4 · Añadir rotación", "Ahora R cambia progresivamente: tanto el punto como la dirección rotan, pero solo el punto mantiene además la suma t.")
        )

    R_final = rz(theta_final)
    for _ in range(35):
        estados.append(
            crear_estado(R_final, t, t, p, v, "4/4 · Misma T, distinto w", "Con la T completa, [p,1] produce Rp+t y [v,0] produce únicamente Rv.")
        )

    for _ in range(55):
        estados.append(crear_estado_conclusion(R_final, t, p, v))

    return {"states": estados, "R": R_final, "t": t, "p": p, "v": v}


def imprimir_resultado(resultado):
    R, t, p, v = resultado["R"], resultado["t"], resultado["p"], resultado["v"]
    T = matriz_homogenea(R, t)
    print("\n=== 4.5. Puntos, direcciones y la coordenada w ===")
    print("\nT[p,1] =", T @ np.r_[p, 1.0])
    print("T[v,0] =", T @ np.r_[v, 0.0])
    print("Rp+t    =", np.r_[R @ p + t, 1.0])
    print("Rv      =", np.r_[R @ v, 0.0])


def main():
    resultado = crear_estados_demostracion()
    imprimir_resultado(resultado)

    animador = TransformAnimator(figsize=(15.5, 8.8), interval=50)
    image_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "05_puntos_direcciones_w.png"
    video_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "05_puntos_direcciones_w.webm"

    animacion = animador.animate_3d_states(
        states=resultado["states"],
        title="4.5. Puntos, direcciones y la coordenada w",
        limits=(-2.5, 5.5, -3.5, 4.5, -2.0, 4.8),
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
