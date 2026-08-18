from pathlib import Path
import sys

import numpy as np


# Permite importar módulos desde la carpeta Matrices de Transformación/
CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


CUBE_FACES = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
]


def formatear_vector(vector):
    """Devuelve un vector 3D con formato compacto."""

    vector = np.asarray(vector, dtype=float)
    return f"[{vector[0]:5.2f}, {vector[1]:5.2f}, {vector[2]:5.2f}]"


def crear_cubo(centro, size=(1.4, 1.0, 0.9)):
    """Crea los ocho vértices de un cuboide centrado en `centro`."""

    centro = np.asarray(centro, dtype=float)
    sx, sy, sz = np.asarray(size, dtype=float) / 2.0

    vertices_locales = np.array([
        [-sx, -sy, -sz],
        [ sx, -sy, -sz],
        [ sx,  sy, -sz],
        [-sx,  sy, -sz],
        [-sx, -sy,  sz],
        [ sx, -sy,  sz],
        [ sx,  sy,  sz],
        [-sx,  sy,  sz],
    ])

    return vertices_locales + centro


def suavizar(progreso):
    """Interpolación cosenoidal para un movimiento suave."""

    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def crear_estado(punto_inicial, traslacion, progreso, fase, mensaje):
    """
    Construye un estado de la traslación 3D.

    Se aplica exactamente la regla:
        p2 = p1 + t

    El frame {A} permanece fijo en el origen. El frame {B} conserva su
    orientación y únicamente cambia de posición.
    """

    punto_inicial = np.asarray(punto_inicial, dtype=float)
    traslacion = np.asarray(traslacion, dtype=float)

    desplazamiento_actual = progreso * traslacion
    punto_actual = punto_inicial + desplazamiento_actual

    cubo_inicial = crear_cubo(punto_inicial)
    cubo_actual = crear_cubo(punto_actual)

    return {
        "frames3d": [
            {
                "name": "A",
                "origin": np.zeros(3),
                "rotation": np.eye(3),
                "length": 1.35,
                "alpha": 1.0,
            },
            {
                "name": "B",
                "origin": desplazamiento_actual,
                "rotation": np.eye(3),
                "length": 1.15,
                "alpha": 1.0,
                "colors": ("#D97706", "#0F766E", "#2563EB"),
            },
        ],
        "meshes3d": [
            {
                "vertices": cubo_inicial,
                "faces": CUBE_FACES,
                "facecolor": "#CBD5E1",
                "edgecolor": "#64748B",
                "alpha": 0.10,
                "linewidth": 0.9,
            },
            {
                "vertices": cubo_actual,
                "faces": CUBE_FACES,
                "facecolor": "#93C5FD",
                "edgecolor": "#1D4ED8",
                "alpha": 0.30,
                "linewidth": 1.2,
            },
        ],
        "points3d": [
            {
                "name": "p1",
                "position": punto_inicial,
                "color": "#6B7280",
                "alpha": 0.60,
                "size": 45,
            },
            {
                "name": "p(t)",
                "position": punto_actual,
                "color": "#7B2CBF",
                "size": 70,
            },
        ],
        "vectors3d": [
            {
                "name": "t(t)",
                "origin": np.zeros(3),
                "value": desplazamiento_actual,
                "color": "#E07A1F",
                "linewidth": 3.0,
            },
        ],
        "segments3d": [
            {
                "start": punto_inicial,
                "end": punto_inicial + traslacion,
                "color": "#7B2CBF",
                "alpha": 0.30,
                "linestyle": "--",
            },
        ],
        "message": mensaje,
        "info_title": "Traslación en 3D",
        "info_lines": [
            {"text": "DATOS", "bold": True},
            f"p1 = {formatear_vector(punto_inicial)}",
            f"t  = {formatear_vector(traslacion)}",
            "",
            {"text": "ESTADO ACTUAL", "bold": True},
            f"t(t) = {formatear_vector(desplazamiento_actual)}",
            f"p(t) = {formatear_vector(punto_actual)}",
            "",
            {"text": "POSE DEL FRAME {B}", "bold": True},
            f"origen B = {formatear_vector(desplazamiento_actual)}",
            "R_B = I",
            "",
            "Solo cambia la posición.",
            "La orientación se conserva.",
        ],
        "phase": fase,
        "info_line_height": 0.046,
        "legend": [
            {"kind": "point", "label": "punto inicial", "color": "#6B7280"},
            {"kind": "point", "label": "punto trasladado", "color": "#7B2CBF"},
            {"kind": "line", "label": "vector t", "color": "#E07A1F"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.2,
    }


def crear_estado_conclusion(punto_inicial, traslacion):
    """Estado final que destaca la regla p2 = p1 + t."""

    estado = crear_estado(
        punto_inicial=punto_inicial,
        traslacion=traslacion,
        progreso=1.0,
        fase="Conclusión",
        mensaje=(
            "Una traslación suma el mismo vector t a todos los puntos. "
            "El frame {B} se desplaza, pero sus ejes mantienen la orientación."
        ),
    )

    punto_final = punto_inicial + traslacion

    estado["info_lines"] = [
        {"text": "RESULTADO", "bold": True},
        f"p1 = {formatear_vector(punto_inicial)}",
        f"t  = {formatear_vector(traslacion)}",
        f"p2 = {formatear_vector(punto_final)}",
        "",
        "p2 = p1 + t",
        "",
        {"text": "IDEA GEOMÉTRICA", "bold": True},
        "Todos los puntos reciben",
        "el mismo desplazamiento.",
        "",
        "No cambia la orientación",
        "del objeto ni del frame.",
    ]

    return estado


def crear_estados_demostracion():
    """Construye la animación del apartado 3.1."""

    punto_inicial = np.array([0.9, 1.2, 0.8])
    traslacion = np.array([2.6, -1.4, 1.7])

    estados = []

    for _ in range(30):
        estados.append(
            crear_estado(
                punto_inicial=punto_inicial,
                traslacion=traslacion,
                progreso=0.0,
                fase="1/3 · Estado inicial",
                mensaje=(
                    "Partimos de p1 y de dos frames con la misma orientación. "
                    "La traslación se describirá mediante el vector t."
                ),
            )
        )

    for progreso in np.linspace(0.0, 1.0, 95):
        estados.append(
            crear_estado(
                punto_inicial=punto_inicial,
                traslacion=traslacion,
                progreso=suavizar(progreso),
                fase="2/3 · Aplicar p2 = p1 + t",
                mensaje=(
                    "El punto y el cuboide reciben el mismo desplazamiento. "
                    "El origen de {B} avanza siguiendo t sin girar sus ejes."
                ),
            )
        )

    for _ in range(35):
        estados.append(
            crear_estado(
                punto_inicial=punto_inicial,
                traslacion=traslacion,
                progreso=1.0,
                fase="3/3 · Traslación completada",
                mensaje=(
                    "La traslación ha cambiado la posición, pero no la forma ni "
                    "la orientación del objeto o del frame {B}."
                ),
            )
        )

    for _ in range(45):
        estados.append(
            crear_estado_conclusion(
                punto_inicial=punto_inicial,
                traslacion=traslacion,
            )
        )

    return {
        "states": estados,
        "p1": punto_inicial,
        "t": traslacion,
        "p2": punto_inicial + traslacion,
    }


def imprimir_resultado(resultado):
    """Muestra el cálculo principal de la traslación."""

    print("\n=== 3.1. Traslación en 3D ===")
    print(f"\np1 = {formatear_vector(resultado['p1'])}")
    print(f"t  = {formatear_vector(resultado['t'])}")
    print(f"p2 = {formatear_vector(resultado['p2'])}")
    print("\nComprobación:")
    print(f"p1 + t = {formatear_vector(resultado['p1'] + resultado['t'])}")


def main():
    resultado = crear_estados_demostracion()
    imprimir_resultado(resultado)

    animador = TransformAnimator(
        figsize=(15.5, 8.8),
        interval=50,
    )

    image_path = (
        MATRICES_DIR
        / "assets"
        / "03_transformaciones_3d"
        / "01_traslacion_3d.png"
    )
    video_path = (
        MATRICES_DIR
        / "assets"
        / "03_transformaciones_3d"
        / "01_traslacion_3d.webm"
    )

    animacion = animador.animate_3d_states(
        states=resultado["states"],
        title="3.1. Traslación en 3D",
        limits=(-2.0, 5.2, -3.2, 3.5, -1.5, 4.2),
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
