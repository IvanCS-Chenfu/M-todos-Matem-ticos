from pathlib import Path
import sys

import numpy as np


# Permite importar módulos desde la carpeta Matrices de Transformación/
CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rotacion_2d(theta):
    """Matriz de rotación 2D activa."""

    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def suavizar(progreso):
    """Interpolación cosenoidal para movimientos suaves."""

    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def formatear_vector(vector):
    vector = np.asarray(vector, dtype=float)
    return f"[{vector[0]:5.2f}, {vector[1]:5.2f}]"


def crear_triangulo():
    """Triángulo asimétrico para hacer visible la transformación."""

    return np.array([
        [0.0, 0.0],
        [1.7, 0.2],
        [0.4, 1.3],
    ], dtype=float)


def crear_estado_lineal(theta, theta_final, fase, mensaje):
    """Muestra que p -> A p mantiene fijo el origen."""

    A = rotacion_2d(theta)
    triangulo = crear_triangulo()
    triangulo_A = (A @ triangulo.T).T

    return {
        "polygons": [
            {
                "points": triangulo,
                "facecolor": "#CBD5E1",
                "edgecolor": "#64748B",
                "alpha": 0.10,
                "linewidth": 1.0,
            },
            {
                "points": triangulo_A,
                "facecolor": "#BFDBFE",
                "edgecolor": "#2563EB",
                "alpha": 0.32,
                "linewidth": 1.5,
            },
        ],
        "points": [
            {
                "name": "0 = A0",
                "position": np.zeros(2),
                "color": "#111827",
                "size": 95,
                "label_offset": (0.12, -0.28),
            },
        ],
        "vectors": [
            {
                "name": "Ae1",
                "origin": np.zeros(2),
                "value": A[:, 0],
                "color": "#B23A48",
                "linewidth": 2.8,
            },
            {
                "name": "Ae2",
                "origin": np.zeros(2),
                "value": A[:, 1],
                "color": "#2D7F5E",
                "linewidth": 2.8,
            },
        ],
        "message": mensaje,
        "info_title": "Transformación lineal",
        "info_lines": [
            {"text": "REGLA", "bold": True},
            "f(p) = A p",
            "",
            {"text": "PRUEBA CON EL ORIGEN", "bold": True},
            "A · [0, 0]^T = [0, 0]^T",
            "",
            f"theta = {np.degrees(theta):6.1f}°",
            f"theta final = {np.degrees(theta_final):6.1f}°",
            "",
            "Aunque A rote o deforme",
            "el plano, el origen no",
            "puede abandonar [0, 0].",
        ],
        "phase": fase,
        "info_line_height": 0.052,
        "legend": [
            {"kind": "line", "label": "figura original", "color": "#64748B"},
            {"kind": "line", "label": "figura por A", "color": "#2563EB"},
            {"kind": "point", "label": "origen fijo", "color": "#111827"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.3,
    }


def crear_estado_traslacion(t_actual, t_final, fase, mensaje):
    """Muestra que p -> p+t desplaza también el origen."""

    triangulo = crear_triangulo()
    triangulo_t = triangulo + t_actual

    return {
        "polygons": [
            {
                "points": triangulo,
                "facecolor": "#CBD5E1",
                "edgecolor": "#64748B",
                "alpha": 0.10,
                "linewidth": 1.0,
            },
            {
                "points": triangulo_t,
                "facecolor": "#FDE68A",
                "edgecolor": "#D97706",
                "alpha": 0.36,
                "linewidth": 1.5,
            },
        ],
        "points": [
            {
                "name": "0",
                "position": np.zeros(2),
                "color": "#111827",
                "alpha": 0.45,
                "size": 70,
                "label_offset": (0.12, -0.25),
            },
            {
                "name": "f(0)=t",
                "position": t_actual,
                "color": "#7B2CBF",
                "size": 100,
                "label_offset": (0.14, 0.12),
            },
        ],
        "vectors": [
            {
                "name": "t",
                "origin": np.zeros(2),
                "value": t_actual,
                "color": "#E07A1F",
                "linewidth": 3.0,
            },
        ],
        "segments": [
            {
                "start": np.zeros(2),
                "end": t_final,
                "color": "#7B2CBF",
                "alpha": 0.22,
                "linestyle": "--",
                "linewidth": 1.4,
            },
        ],
        "message": mensaje,
        "info_title": "Traslación: no es lineal",
        "info_lines": [
            {"text": "REGLA", "bold": True},
            "f(p) = p + t",
            "",
            {"text": "EJEMPLO DE LA WIKI", "bold": True},
            f"t = {formatear_vector(t_final)}",
            "",
            {"text": "ORIGEN", "bold": True},
            f"f(0) = {formatear_vector(t_actual)}",
            "",
            "Si t != 0, entonces",
            "f(0) != 0.",
            "",
            "Por eso ninguna matriz",
            "2x2 puede representar",
            "una traslación pura.",
        ],
        "phase": fase,
        "info_line_height": 0.047,
        "legend": [
            {"kind": "line", "label": "figura original", "color": "#64748B"},
            {"kind": "line", "label": "figura trasladada", "color": "#D97706"},
            {"kind": "line", "label": "vector t", "color": "#E07A1F"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.3,
    }


def crear_estado_conclusion(t):
    """Compara en una sola escena A0=0 con 0+t=t."""

    A = rotacion_2d(np.radians(55.0))
    triangulo = crear_triangulo()
    triangulo_A = (A @ triangulo.T).T
    triangulo_t = triangulo + t

    return {
        "polygons": [
            {
                "points": triangulo_A,
                "facecolor": "#BFDBFE",
                "edgecolor": "#2563EB",
                "alpha": 0.30,
                "linewidth": 1.4,
            },
            {
                "points": triangulo_t,
                "facecolor": "#FDE68A",
                "edgecolor": "#D97706",
                "alpha": 0.34,
                "linewidth": 1.4,
            },
        ],
        "points": [
            {
                "name": "A0 = 0",
                "position": np.zeros(2),
                "color": "#2563EB",
                "size": 95,
                "label_offset": (0.12, -0.28),
            },
            {
                "name": "0+t = t",
                "position": t,
                "color": "#D97706",
                "size": 95,
                "label_offset": (0.12, 0.15),
            },
        ],
        "vectors": [
            {
                "name": "t=[2,3]",
                "origin": np.zeros(2),
                "value": t,
                "color": "#7B2CBF",
                "linewidth": 3.0,
            },
        ],
        "texts": [
            {
                "position": (-1.1, 1.8),
                "text": "lineal: origen fijo",
                "color": "#2563EB",
                "fontweight": "bold",
            },
            {
                "position": (2.8, 4.4),
                "text": "traslación: origen desplazado",
                "color": "#D97706",
                "fontweight": "bold",
            },
        ],
        "message": (
            "La diferencia decisiva se ve en el origen: toda transformación "
            "lineal cumple A0=0, mientras una traslación lleva 0 hasta t."
        ),
        "info_title": "Por qué la traslación no es lineal",
        "info_lines": [
            {"text": "LINEAL", "bold": True},
            "f(p)=Ap",
            "f(0)=0",
            "",
            {"text": "TRASLACIÓN", "bold": True},
            "g(p)=p+t",
            f"g(0)={formatear_vector(t)}",
            "",
            {"text": "CONCLUSIÓN", "bold": True},
            "t != 0  =>  g(0) != 0",
            "",
            "La traslación necesita",
            "una suma externa o una",
            "representación homogénea.",
        ],
        "phase": "Conclusión",
        "info_line_height": 0.047,
        "legend": [
            {"kind": "line", "label": "transformación lineal", "color": "#2563EB"},
            {"kind": "line", "label": "traslación", "color": "#D97706"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.3,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 4.1."""

    t = np.array([2.0, 3.0])
    theta_final = np.radians(55.0)
    estados = []

    for _ in range(28):
        estados.append(
            crear_estado_lineal(
                theta=0.0,
                theta_final=theta_final,
                fase="1/4 · Transformación lineal",
                mensaje=(
                    "Antes de estudiar la traslación, comprobamos la propiedad "
                    "f(0)=0 de una transformación lineal."
                ),
            )
        )

    for progreso in np.linspace(0.0, 1.0, 75):
        theta = suavizar(progreso) * theta_final
        estados.append(
            crear_estado_lineal(
                theta=theta,
                theta_final=theta_final,
                fase="2/4 · A puede mover todo salvo el origen",
                mensaje=(
                    "La matriz rota la figura y los vectores de la base, pero "
                    "el origen continúa exactamente en [0,0]."
                ),
            )
        )

    for _ in range(22):
        estados.append(
            crear_estado_traslacion(
                t_actual=np.zeros(2),
                t_final=t,
                fase="3/4 · Traslación",
                mensaje=(
                    "Ahora aplicamos g(p)=p+t con t=[2,3]. Si la traslación "
                    "fuera lineal, también debería dejar fijo el origen."
                ),
            )
        )

    for progreso in np.linspace(0.0, 1.0, 85):
        t_actual = suavizar(progreso) * t
        estados.append(
            crear_estado_traslacion(
                t_actual=t_actual,
                t_final=t,
                fase="4/4 · El origen se desplaza",
                mensaje=(
                    "El origen sigue la misma suma que cualquier otro punto y "
                    "termina en t. Esto viola f(0)=0."
                ),
            )
        )

    for _ in range(50):
        estados.append(crear_estado_conclusion(t))

    return {"states": estados, "t": t}


def imprimir_resultado(resultado):
    t = resultado["t"]
    print("\n=== 4.1. Por qué la traslación no es lineal ===")
    print("\nPara toda matriz A: A·0 = 0")
    print(f"Para g(p)=p+t con t={t}: g(0)=t={t}")
    print("Como t != 0, la traslación no es lineal.")


def main():
    resultado = crear_estados_demostracion()
    imprimir_resultado(resultado)

    animador = TransformAnimator(figsize=(15.5, 8.8), interval=50)

    image_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "01_traslacion_no_lineal.png"
    video_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "01_traslacion_no_lineal.webm"

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="4.1. Por qué la traslación no es lineal",
        limits=(-2.3, 5.8, -2.0, 5.8),
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
