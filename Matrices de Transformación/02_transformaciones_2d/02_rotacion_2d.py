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
    """Devuelve las dos filas de una matriz 2x2 como texto."""

    matriz = np.asarray(matriz, dtype=float)
    return (
        f"[[{matriz[0, 0]:6.3f}, {matriz[0, 1]:6.3f}],",
        f" [{matriz[1, 0]:6.3f}, {matriz[1, 1]:6.3f}]]",
    )


def interpolar(inicio, fin, cantidad):
    """Genera valores entre inicio y fin, incluyendo ambos extremos."""

    return np.linspace(inicio, fin, cantidad)


def suavizar(progreso):
    """Interpolación cosenoidal para movimientos suaves."""

    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def matriz_rotacion_activa(angulo):
    """Construye la matriz activa 2D con ángulo positivo antihorario."""

    c = np.cos(angulo)
    s = np.sin(angulo)

    return np.array([
        [c, -s],
        [s, c],
    ])


def rotar_puntos(puntos, angulo):
    """Rota una colección de puntos mediante R(theta)."""

    puntos = np.asarray(puntos, dtype=float)
    rotacion = matriz_rotacion_activa(angulo)

    return (rotacion @ puntos.T).T


def crear_estado_rotacion_activa(
    punto,
    figura,
    angulo_actual,
    angulo_final,
    fase,
    mensaje,
):
    """Muestra la rotación activa del punto, la figura y los vectores base."""

    punto = np.asarray(punto, dtype=float)
    figura = np.asarray(figura, dtype=float)

    rotacion = matriz_rotacion_activa(angulo_actual)
    punto_rotado = rotacion @ punto
    figura_rotada = rotar_puntos(figura, angulo_actual)

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    re1 = rotacion @ e1
    re2 = rotacion @ e2

    m1, m2 = formatear_matriz(rotacion)

    theta_deg = np.degrees(angulo_actual)
    arc_start = min(0.0, theta_deg)
    arc_end = max(0.0, theta_deg)

    return {
        "polygons": [
            {
                "points": figura,
                "facecolor": "#D1D5DB",
                "edgecolor": "#6B7280",
                "alpha": 0.16,
                "linewidth": 1.4,
            },
            {
                "points": figura_rotada,
                "facecolor": "#E9DDF4",
                "edgecolor": "#7B2CBF",
                "alpha": 0.30,
                "linewidth": 2.0,
            },
        ],
        "points": [
            {
                "name": "p",
                "position": punto,
                "color": "#6B7280",
                "alpha": 0.70,
                "size": 75,
            },
            {
                "name": "R p",
                "position": punto_rotado,
                "color": "#7B2CBF",
                "size": 95,
            },
        ],
        "vectors": [
            {
                "name": "e1",
                "origin": np.zeros(2),
                "value": e1,
                "color": "#9CA3AF",
                "alpha": 0.45,
                "linewidth": 1.8,
            },
            {
                "name": "e2",
                "origin": np.zeros(2),
                "value": e2,
                "color": "#9CA3AF",
                "alpha": 0.45,
                "linewidth": 1.8,
            },
            {
                "name": "R e1",
                "origin": np.zeros(2),
                "value": re1,
                "color": "#B23A48",
                "linewidth": 2.8,
            },
            {
                "name": "R e2",
                "origin": np.zeros(2),
                "value": re2,
                "color": "#2D7F5E",
                "linewidth": 2.8,
            },
        ],
        "arcs": [
            {
                "center": np.zeros(2),
                "radius": 0.75,
                "theta1": arc_start,
                "theta2": arc_end,
                "color": "#D97706",
                "linewidth": 2.2,
                "alpha": 0.95 if abs(theta_deg) > 1.0 else 0.0,
            },
        ],
        "texts": [
            {
                "position": np.array([0.62, 0.62]),
                "text": f"θ = {theta_deg:.1f}°",
                "color": "#D97706",
                "fontweight": "bold",
                "fontsize": 10,
            },
        ],
        "message": mensaje,
        "info_title": "Rotación activa 2D",
        "info_lines": [
            {"text": "MATRIZ R(θ)", "bold": True},
            m1,
            m2,
            "",
            {"text": "PUNTO", "bold": True},
            f"p        = {formatear_vector(punto)}",
            f"R p      = {formatear_vector(punto_rotado)}",
            "",
            {"text": "COLUMNAS DE R", "bold": True},
            f"R e1     = {formatear_vector(re1)}",
            f"R e2     = {formatear_vector(re2)}",
            "",
            f"||p||    = {np.linalg.norm(punto):.3f}",
            f"||Rp||   = {np.linalg.norm(punto_rotado):.3f}",
            f"det(R)   = {np.linalg.det(rotacion):.3f}",
        ],
        "phase": fase,
        "info_line_height": 0.044,
        "legend": [
            {"kind": "line", "label": "figura original", "color": "#6B7280"},
            {"kind": "line", "label": "figura rotada", "color": "#7B2CBF"},
            {"kind": "line", "label": "R e1", "color": "#B23A48"},
            {"kind": "line", "label": "R e2", "color": "#2D7F5E"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.2,
    }


def crear_estado_convencion_pasiva(punto, angulo_frame):
    """
    Contrasta rotación activa con cambio pasivo de coordenadas.

    El punto físico permanece fijo. El frame {B} se gira +theta respecto a {A}
    y sus coordenadas se calculan con R(theta)^T p.
    """

    punto = np.asarray(punto, dtype=float)
    rotacion = matriz_rotacion_activa(angulo_frame)

    punto_activo = rotacion @ punto
    coordenadas_pasivas = rotacion.T @ punto

    rt1, rt2 = formatear_matriz(rotacion.T)

    return {
        "frames": [
            {
                "name": "A",
                "origin": np.zeros(2),
                "angle": 0.0,
                "length": 1.65,
                "x_color": "#B23A48",
                "y_color": "#2D7F5E",
                "alpha": 0.55,
            },
            {
                "name": "B",
                "origin": np.zeros(2),
                "angle": angulo_frame,
                "length": 1.65,
                "x_color": "#D97706",
                "y_color": "#1F77B4",
                "alpha": 1.0,
            },
        ],
        "points": [
            {
                "name": "p físico",
                "position": punto,
                "color": "#7B2CBF",
                "size": 95,
            },
            {
                "name": "R p",
                "position": punto_activo,
                "color": "#D97706",
                "size": 80,
            },
        ],
        "vectors": [
            {
                "name": "[p]_B",
                "origin": np.zeros(2),
                "value": coordenadas_pasivas,
                "color": "#1F77B4",
                "linewidth": 2.7,
                "linestyle": "--",
            },
        ],
        "arcs": [
            {
                "center": np.zeros(2),
                "radius": 0.72,
                "theta1": 0.0,
                "theta2": np.degrees(angulo_frame),
                "color": "#D97706",
                "linewidth": 2.2,
            },
        ],
        "message": (
            "Activa: R(θ)p mueve el vector. Pasiva: giramos el frame +θ, "
            "mantenemos p fijo y calculamos [p]_B = R(θ)^T p."
        ),
        "info_title": "Activa frente a pasiva",
        "info_lines": [
            {"text": "ÁNGULO", "bold": True},
            f"θ = {np.degrees(angulo_frame):.1f}°",
            "",
            {"text": "ROTACIÓN ACTIVA", "bold": True},
            f"p        = {formatear_vector(punto)}",
            f"R p      = {formatear_vector(punto_activo)}",
            "",
            {"text": "CAMBIO PASIVO", "bold": True},
            "R_pasiva = R_activa^T",
            rt1,
            rt2,
            f"[p]_B    = {formatear_vector(coordenadas_pasivas)}",
        ],
        "phase": "Conclusión · Dos preguntas distintas",
        "info_line_height": 0.050,
        "legend": [
            {"kind": "point", "label": "p físico", "color": "#7B2CBF"},
            {"kind": "point", "label": "R p activa", "color": "#D97706"},
            {"kind": "line", "label": "coordenadas pasivas", "color": "#1F77B4", "linestyle": "--"},
        ],
        "legend_ncol": 3,
        "legend_fontsize": 8.0,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 2.2."""

    punto = np.array([2.0, 1.0])
    angulo_final = np.radians(90.0)

    figura = np.array([
        [0.7, 0.4],
        [2.8, 0.6],
        [2.1, 2.4],
        [0.8, 1.8],
    ])

    estados = []

    for _ in range(28):
        estados.append(
            crear_estado_rotacion_activa(
                punto=punto,
                figura=figura,
                angulo_actual=0.0,
                angulo_final=angulo_final,
                fase="1/4 · Matriz de rotación activa",
                mensaje=(
                    "Con vectores columna, un ángulo positivo rota activamente "
                    "en sentido antihorario."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 110):
        angulo = suavizar(progreso) * angulo_final

        estados.append(
            crear_estado_rotacion_activa(
                punto=punto,
                figura=figura,
                angulo_actual=angulo,
                angulo_final=angulo_final,
                fase="2/4 · Variar θ",
                mensaje=(
                    "Al variar θ cambian simultáneamente el punto, la figura y "
                    "las columnas R e1 y R e2."
                ),
            )
        )

    for _ in range(32):
        estados.append(
            crear_estado_rotacion_activa(
                punto=punto,
                figura=figura,
                angulo_actual=angulo_final,
                angulo_final=angulo_final,
                fase="3/4 · Propiedades geométricas",
                mensaje=(
                    "La rotación de 90° conserva la norma y la forma. "
                    "Además R^T R = I y det(R)=1."
                ),
            )
        )

    for _ in range(48):
        estados.append(
            crear_estado_convencion_pasiva(
                punto=punto,
                angulo_frame=angulo_final,
            )
        )

    return {
        "states": estados,
        "point": punto,
        "angle": angulo_final,
        "rotation": matriz_rotacion_activa(angulo_final),
    }


def imprimir_resultado(resultado):
    """Muestra por terminal el ejemplo numérico de 90 grados."""

    punto = resultado["point"]
    rotacion = resultado["rotation"]
    activo = rotacion @ punto
    pasivo = rotacion.T @ punto

    print("\n=== 2.2. Rotación en 2D ===")
    print("\nR_activa(90°) =")
    print(rotacion)
    print(f"\np       = {formatear_vector(punto)}")
    print(f"R p     = {formatear_vector(activo)}")
    print(f"R^T p   = {formatear_vector(pasivo)}")
    print(f"\ndet(R)  = {np.linalg.det(rotacion):.3f}")
    print(f"||p||    = {np.linalg.norm(punto):.3f}")
    print(f"||Rp||   = {np.linalg.norm(activo):.3f}")


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
        / "02_rotacion_2d.png"
    )

    video_path = (
        MATRICES_DIR
        / "assets"
        / "02_transformaciones_2d"
        / "02_rotacion_2d.webm"
    )

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="2.2. Rotación en 2D",
        limits=(-3.6, 3.8, -3.3, 3.8),
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
