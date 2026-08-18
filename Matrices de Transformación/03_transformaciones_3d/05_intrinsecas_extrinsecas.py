from pathlib import Path
import sys

import numpy as np


# Permite importar módulos desde la carpeta Matrices de Transformación/
CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


TETRA_FACES = [
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3],
]


def rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def ry(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def suavizar(progreso):
    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def crear_objeto_local():
    """
    Tetraedro asimétrico usado para hacer visible la orientación del frame.
    """

    return np.array([
        [0.0, 0.0, 0.0],
        [1.25, 0.15, 0.05],
        [0.15, 0.85, 0.10],
        [0.10, 0.20, 1.05],
    ])


def transformar_objeto(vertices_locales, R, origen):
    """Aplica la orientación R y coloca el objeto en `origen`."""

    return (R @ vertices_locales.T).T + origen


def formatear_vector(vector):
    vector = np.asarray(vector, dtype=float)
    return f"[{vector[0]:5.2f}, {vector[1]:5.2f}, {vector[2]:5.2f}]"


def formatear_matriz_compacta(R):
    """Devuelve tres filas compactas de una matriz 3x3."""

    return [
        f"[{R[0,0]:5.2f},{R[0,1]:5.2f},{R[0,2]:5.2f}]",
        f"[{R[1,0]:5.2f},{R[1,1]:5.2f},{R[1,2]:5.2f}]",
        f"[{R[2,0]:5.2f},{R[2,1]:5.2f},{R[2,2]:5.2f}]",
    ]


def orientaciones_etapa(etapa, fraccion, alpha, beta, gamma):
    """
    Calcula simultáneamente las orientaciones extrínseca e intrínseca.

    Convención:
    - vectores columna,
    - rotaciones activas,
    - orden conceptual X -> Y -> Z.

    Extrínseca:
        cada giro se hace alrededor de ejes globales fijos,
        por lo que las nuevas rotaciones multiplican por la izquierda.

    Intrínseca:
        cada giro se hace alrededor de ejes locales del frame móvil,
        por lo que las nuevas rotaciones multiplican por la derecha.
    """

    if etapa == 1:
        a = fraccion * alpha
        R_ext = rx(a)
        R_int = rx(a)

        eje_ext = np.array([1.0, 0.0, 0.0])
        eje_int = np.array([1.0, 0.0, 0.0])
        nombre_eje = "x"

    elif etapa == 2:
        b = fraccion * beta
        R_pre_ext = rx(alpha)
        R_pre_int = rx(alpha)

        R_ext = ry(b) @ R_pre_ext
        R_int = R_pre_int @ ry(b)

        # Extrínseca: eje y global fijo.
        eje_ext = np.array([0.0, 1.0, 0.0])

        # Intrínseca: eje y local tras la primera rotación.
        eje_int = R_pre_int @ np.array([0.0, 1.0, 0.0])
        nombre_eje = "y"

    elif etapa == 3:
        g = fraccion * gamma

        R_pre_ext = ry(beta) @ rx(alpha)
        R_pre_int = rx(alpha) @ ry(beta)

        R_ext = rz(g) @ R_pre_ext
        R_int = R_pre_int @ rz(g)

        # Extrínseca: eje z global fijo.
        eje_ext = np.array([0.0, 0.0, 1.0])

        # Intrínseca: eje z local tras las dos rotaciones previas.
        eje_int = R_pre_int @ np.array([0.0, 0.0, 1.0])
        nombre_eje = "z"

    else:
        raise ValueError("etapa debe ser 1, 2 o 3")

    return R_ext, R_int, eje_ext, eje_int, nombre_eje


def crear_estado(
    R_ext,
    R_int,
    eje_ext,
    eje_int,
    nombre_eje,
    origen_ext,
    origen_int,
    angulos,
    fase,
    mensaje,
):
    """Construye un estado que compara rotaciones extrínsecas e intrínsecas."""

    objeto = crear_objeto_local()
    objeto_ext = transformar_objeto(objeto, R_ext, origen_ext)
    objeto_int = transformar_objeto(objeto, R_int, origen_int)

    alpha, beta, gamma = angulos

    diferencia = np.linalg.norm(R_ext - R_int)

    filas_ext = formatear_matriz_compacta(R_ext)
    filas_int = formatear_matriz_compacta(R_int)

    return {
        "frames3d": [
            # Ejes globales tenues de referencia en ambas posiciones.
            {
                "name": "G",
                "origin": origen_ext,
                "rotation": np.eye(3),
                "length": 1.20,
                "alpha": 0.16,
                "colors": ("#9CA3AF", "#9CA3AF", "#9CA3AF"),
            },
            {
                "name": "G",
                "origin": origen_int,
                "rotation": np.eye(3),
                "length": 1.20,
                "alpha": 0.16,
                "colors": ("#9CA3AF", "#9CA3AF", "#9CA3AF"),
            },
            {
                "name": "E",
                "origin": origen_ext,
                "rotation": R_ext,
                "length": 1.45,
                "alpha": 1.0,
            },
            {
                "name": "I",
                "origin": origen_int,
                "rotation": R_int,
                "length": 1.45,
                "alpha": 1.0,
                "colors": ("#D97706", "#0F766E", "#2563EB"),
            },
        ],
        "meshes3d": [
            {
                "vertices": objeto_ext,
                "faces": TETRA_FACES,
                "facecolor": "#93C5FD",
                "edgecolor": "#1D4ED8",
                "alpha": 0.28,
                "linewidth": 1.1,
            },
            {
                "vertices": objeto_int,
                "faces": TETRA_FACES,
                "facecolor": "#F8CFA7",
                "edgecolor": "#C2410C",
                "alpha": 0.28,
                "linewidth": 1.1,
            },
        ],
        "vectors3d": [
            {
                "name": f"eje {nombre_eje} global",
                "origin": origen_ext - 1.55 * eje_ext,
                "value": 3.10 * eje_ext,
                "color": "#7B2CBF",
                "linewidth": 2.8,
                "show_origin": False,
            },
            {
                "name": f"eje {nombre_eje} local",
                "origin": origen_int - 1.55 * eje_int,
                "value": 3.10 * eje_int,
                "color": "#E07A1F",
                "linewidth": 2.8,
                "show_origin": False,
            },
        ],
        "texts3d": [
            {
                "position": origen_ext + np.array([0.0, 0.0, 2.1]),
                "text": "EXTRÍNSECA",
                "fontweight": "bold",
                "color": "#1D4ED8",
            },
            {
                "position": origen_int + np.array([0.0, 0.0, 2.1]),
                "text": "INTRÍNSECA",
                "fontweight": "bold",
                "color": "#C2410C",
            },
        ],
        "message": mensaje,
        "info_title": "Intrínsecas frente a extrínsecas",
        "info_lines": [
            {"text": "ÁNGULOS OBJETIVO", "bold": True},
            f"alpha(X) = {np.degrees(alpha):5.1f}°",
            f"beta (Y) = {np.degrees(beta):5.1f}°",
            f"gamma(Z) = {np.degrees(gamma):5.1f}°",
            "",
            {"text": "R EXTRÍNSECA", "bold": True},
            filas_ext[0],
            filas_ext[1],
            filas_ext[2],
            "",
            {"text": "R INTRÍNSECA", "bold": True},
            filas_int[0],
            filas_int[1],
            filas_int[2],
            "",
            f"||Rext-Rint|| = {diferencia:.3f}",
        ],
        "phase": fase,
        "info_line_height": 0.0385,
        "info_fontsize": 8.6,
        "legend": [
            {"kind": "line", "label": "eje global activo", "color": "#7B2CBF"},
            {"kind": "line", "label": "eje local activo", "color": "#E07A1F"},
            {"kind": "line", "label": "resultado extrínseco", "color": "#1D4ED8"},
            {"kind": "line", "label": "resultado intrínseco", "color": "#C2410C"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 7.7,
    }


def crear_estado_final(alpha, beta, gamma, origen_ext, origen_int):
    """Fotograma final con las dos orientaciones completas."""

    R_ext = rz(gamma) @ ry(beta) @ rx(alpha)
    R_int = rx(alpha) @ ry(beta) @ rz(gamma)

    estado = crear_estado(
        R_ext=R_ext,
        R_int=R_int,
        eje_ext=np.array([0.0, 0.0, 1.0]),
        eje_int=(rx(alpha) @ ry(beta)) @ np.array([0.0, 0.0, 1.0]),
        nombre_eje="z",
        origen_ext=origen_ext,
        origen_int=origen_int,
        angulos=(alpha, beta, gamma),
        fase="Conclusión",
        mensaje=(
            "Los mismos tres ángulos producen orientaciones diferentes porque "
            "las rotaciones extrínsecas usan ejes globales fijos y las "
            "intrínsecas usan ejes locales que se mueven con el frame."
        ),
    )

    estado["info_lines"] = [
        {"text": "ORDEN X -> Y -> Z", "bold": True},
        "",
        {"text": "EXTRÍNSECA", "bold": True},
        "Rext = Rz Ry Rx",
        "ejes: globales fijos",
        "",
        {"text": "INTRÍNSECA", "bold": True},
        "Rint = Rx Ry Rz",
        "ejes: locales móviles",
        "",
        {"text": "RESULTADO", "bold": True},
        f"||Rext-Rint|| = {np.linalg.norm(R_ext-R_int):.3f}",
        "",
        "Mismos ángulos.",
        "Distinta interpretación.",
        "Distinta orientación final.",
    ]
    estado["info_line_height"] = 0.044

    return estado


def crear_estados_demostracion():
    """Construye la animación del apartado 3.5."""

    alpha = np.radians(40.0)
    beta = np.radians(30.0)
    gamma = np.radians(55.0)

    origen_ext = np.array([-2.8, 0.0, 0.0])
    origen_int = np.array([2.8, 0.0, 0.0])

    estados = []

    # Introducción: ambos frames coinciden.
    for _ in range(32):
        estados.append(
            crear_estado(
                R_ext=np.eye(3),
                R_int=np.eye(3),
                eje_ext=np.array([1.0, 0.0, 0.0]),
                eje_int=np.array([1.0, 0.0, 0.0]),
                nombre_eje="x",
                origen_ext=origen_ext,
                origen_int=origen_int,
                angulos=(alpha, beta, gamma),
                fase="Introducción",
                mensaje=(
                    "Ambos frames parten de la misma orientación y usarán los "
                    "mismos ángulos X, Y y Z. Solo cambiará qué ejes se consideran."
                ),
            )
        )

    # Etapa X: todavía coinciden.
    for progreso in np.linspace(0.0, 1.0, 80):
        R_ext, R_int, eje_ext, eje_int, nombre = orientaciones_etapa(
            1,
            suavizar(progreso),
            alpha,
            beta,
            gamma,
        )
        estados.append(
            crear_estado(
                R_ext,
                R_int,
                eje_ext,
                eje_int,
                nombre,
                origen_ext,
                origen_int,
                (alpha, beta, gamma),
                fase="1/3 · Giro X",
                mensaje=(
                    "En el primer giro los ejes x global y local coinciden, por "
                    "lo que ambas interpretaciones producen la misma orientación."
                ),
            )
        )

    for _ in range(20):
        R_ext, R_int, eje_ext, eje_int, nombre = orientaciones_etapa(
            1, 1.0, alpha, beta, gamma
        )
        estados.append(
            crear_estado(
                R_ext,
                R_int,
                eje_ext,
                eje_int,
                nombre,
                origen_ext,
                origen_int,
                (alpha, beta, gamma),
                fase="1/3 · Giro X completado",
                mensaje="Tras el primer giro, ambos frames todavía coinciden.",
            )
        )

    # Etapa Y: comienza la divergencia.
    for progreso in np.linspace(0.0, 1.0, 90):
        R_ext, R_int, eje_ext, eje_int, nombre = orientaciones_etapa(
            2,
            suavizar(progreso),
            alpha,
            beta,
            gamma,
        )
        estados.append(
            crear_estado(
                R_ext,
                R_int,
                eje_ext,
                eje_int,
                nombre,
                origen_ext,
                origen_int,
                (alpha, beta, gamma),
                fase="2/3 · Giro Y",
                mensaje=(
                    "La extrínseca gira alrededor del eje y global. La intrínseca "
                    "gira alrededor del eje y local, que ya se movió con el frame."
                ),
            )
        )

    for _ in range(20):
        R_ext, R_int, eje_ext, eje_int, nombre = orientaciones_etapa(
            2, 1.0, alpha, beta, gamma
        )
        estados.append(
            crear_estado(
                R_ext,
                R_int,
                eje_ext,
                eje_int,
                nombre,
                origen_ext,
                origen_int,
                (alpha, beta, gamma),
                fase="2/3 · Los resultados divergen",
                mensaje=(
                    "Después del segundo giro las orientaciones ya no coinciden, "
                    "aunque se haya usado exactamente el mismo ángulo beta."
                ),
            )
        )

    # Etapa Z.
    for progreso in np.linspace(0.0, 1.0, 95):
        R_ext, R_int, eje_ext, eje_int, nombre = orientaciones_etapa(
            3,
            suavizar(progreso),
            alpha,
            beta,
            gamma,
        )
        estados.append(
            crear_estado(
                R_ext,
                R_int,
                eje_ext,
                eje_int,
                nombre,
                origen_ext,
                origen_int,
                (alpha, beta, gamma),
                fase="3/3 · Giro Z",
                mensaje=(
                    "El último giro vuelve a usar un eje z global fijo en la "
                    "extrínseca y el eje z local ya transformado en la intrínseca."
                ),
            )
        )

    for _ in range(60):
        estados.append(
            crear_estado_final(
                alpha,
                beta,
                gamma,
                origen_ext,
                origen_int,
            )
        )

    return {
        "states": estados,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "R_ext": rz(gamma) @ ry(beta) @ rx(alpha),
        "R_int": rx(alpha) @ ry(beta) @ rz(gamma),
    }


def imprimir_resultado(resultado):
    """Muestra las dos composiciones finales."""

    print("\n=== 3.5. Composición: rotaciones intrínsecas y extrínsecas ===")
    print("\nR extrínseca = Rz Ry Rx:")
    print(resultado["R_ext"])
    print("\nR intrínseca = Rx Ry Rz:")
    print(resultado["R_int"])
    print(
        "\n||R_ext - R_int|| = "
        f"{np.linalg.norm(resultado['R_ext'] - resultado['R_int']):.6f}"
    )


def main():
    resultado = crear_estados_demostracion()
    imprimir_resultado(resultado)

    animador = TransformAnimator(
        figsize=(16.0, 8.8),
        interval=50,
    )

    image_path = (
        MATRICES_DIR
        / "assets"
        / "03_transformaciones_3d"
        / "05_intrinsecas_extrinsecas.png"
    )
    video_path = (
        MATRICES_DIR
        / "assets"
        / "03_transformaciones_3d"
        / "05_intrinsecas_extrinsecas.webm"
    )

    animacion = animador.animate_3d_states(
        states=resultado["states"],
        title="3.5. Rotaciones intrínsecas y extrínsecas",
        limits=(-5.0, 5.0, -3.4, 3.4, -2.8, 3.8),
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
