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


def crear_cubo_unitario():
    """Cubo de lado 1 centrado en el origen."""

    h = 0.5
    return np.array([
        [-h, -h, -h],
        [ h, -h, -h],
        [ h,  h, -h],
        [-h,  h, -h],
        [-h, -h,  h],
        [ h, -h,  h],
        [ h,  h,  h],
        [-h,  h,  h],
    ])


def formatear_vector(vector):
    vector = np.asarray(vector, dtype=float)
    return f"[{vector[0]:5.2f}, {vector[1]:5.2f}, {vector[2]:5.2f}]"


def suavizar(progreso):
    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def crear_estado_escalado(escalas, punto, fase, mensaje):
    """Construye un estado con un escalado diagonal 3D."""

    escalas = np.asarray(escalas, dtype=float)
    S = np.diag(escalas)

    cubo = crear_cubo_unitario()
    cubo_escalado = (S @ cubo.T).T

    punto_transformado = S @ punto
    volumen = np.linalg.det(S)

    return {
        "frames3d": [
            {
                "name": "0",
                "origin": np.zeros(3),
                "rotation": np.eye(3),
                "length": 1.5,
                "alpha": 0.80,
            },
        ],
        "meshes3d": [
            {
                "vertices": cubo,
                "faces": CUBE_FACES,
                "facecolor": "#CBD5E1",
                "edgecolor": "#64748B",
                "alpha": 0.08,
                "linewidth": 0.8,
            },
            {
                "vertices": cubo_escalado,
                "faces": CUBE_FACES,
                "facecolor": "#93C5FD",
                "edgecolor": "#1D4ED8",
                "alpha": 0.32,
                "linewidth": 1.2,
            },
        ],
        "vectors3d": [
            {
                "name": "p",
                "origin": np.zeros(3),
                "value": punto,
                "color": "#6B7280",
                "alpha": 0.35,
                "linewidth": 1.8,
            },
            {
                "name": "Sp",
                "origin": np.zeros(3),
                "value": punto_transformado,
                "color": "#7B2CBF",
                "linewidth": 3.0,
            },
        ],
        "message": mensaje,
        "info_title": "Escalado en 3D",
        "info_lines": [
            {"text": "FACTORES", "bold": True},
            f"sx = {escalas[0]:6.3f}",
            f"sy = {escalas[1]:6.3f}",
            f"sz = {escalas[2]:6.3f}",
            "",
            {"text": "MATRIZ S", "bold": True},
            f"[{escalas[0]:5.2f}, 0.00, 0.00]",
            f"[0.00, {escalas[1]:5.2f}, 0.00]",
            f"[0.00, 0.00, {escalas[2]:5.2f}]",
            "",
            f"p  = {formatear_vector(punto)}",
            f"Sp = {formatear_vector(punto_transformado)}",
            "",
            {"text": "VOLUMEN", "bold": True},
            f"det(S) = {volumen:7.3f}",
        ],
        "phase": fase,
        "info_line_height": 0.041,
        "info_fontsize": 9.1,
        "legend": [
            {"kind": "line", "label": "cubo original", "color": "#64748B"},
            {"kind": "line", "label": "cubo escalado", "color": "#1D4ED8"},
            {"kind": "line", "label": "Sp", "color": "#7B2CBF"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.0,
    }


def crear_estado_comparacion(S, R):
    """
    Compara visualmente escalado y rotación usando dos copias del mismo cubo.

    Las traslaciones laterales son únicamente de maquetación para poder comparar
    ambos resultados en la misma escena.
    """

    cubo = crear_cubo_unitario()
    origen_escalado = np.array([-2.2, 0.0, 0.0])
    origen_rotado = np.array([2.2, 0.0, 0.0])

    escalado = (S @ cubo.T).T + origen_escalado
    rotado = (R @ cubo.T).T + origen_rotado

    return {
        "frames3d": [
            {
                "name": "S",
                "origin": origen_escalado,
                "rotation": np.eye(3),
                "length": 1.1,
                "alpha": 0.85,
            },
            {
                "name": "R",
                "origin": origen_rotado,
                "rotation": R,
                "length": 1.1,
                "alpha": 0.85,
            },
        ],
        "meshes3d": [
            {
                "vertices": escalado,
                "faces": CUBE_FACES,
                "facecolor": "#93C5FD",
                "edgecolor": "#1D4ED8",
                "alpha": 0.34,
                "linewidth": 1.2,
            },
            {
                "vertices": rotado,
                "faces": CUBE_FACES,
                "facecolor": "#F8CFA7",
                "edgecolor": "#C2410C",
                "alpha": 0.34,
                "linewidth": 1.2,
            },
        ],
        "texts3d": [
            {
                "position": origen_escalado + np.array([0.0, 0.0, 2.0]),
                "text": "Escalado",
                "fontweight": "bold",
                "color": "#1D4ED8",
            },
            {
                "position": origen_rotado + np.array([0.0, 0.0, 1.4]),
                "text": "Rotación",
                "fontweight": "bold",
                "color": "#C2410C",
            },
        ],
        "message": (
            "La rotación cambia orientación sin deformar. El escalado modifica "
            "las dimensiones y, en general, también el volumen."
        ),
        "info_title": "Escalado frente a rotación",
        "info_lines": [
            {"text": "ESCALADO NO UNIFORME", "bold": True},
            "S = diag(2, 0.5, 3)",
            f"det(S) = {np.linalg.det(S):.3f}",
            "",
            "Lados transformados:",
            "x: 1 -> 2",
            "y: 1 -> 0.5",
            "z: 1 -> 3",
            "",
            {"text": "ROTACIÓN RÍGIDA", "bold": True},
            f"det(R) = {np.linalg.det(R):.3f}",
            "Los tres lados siguen",
            "midiendo 1.",
            "",
            "Rotar no es escalar.",
        ],
        "phase": "Conclusión",
        "info_line_height": 0.043,
        "info_fontsize": 9.0,
        "legend": [
            {"kind": "line", "label": "escalado", "color": "#1D4ED8"},
            {"kind": "line", "label": "rotación", "color": "#C2410C"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.1,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 3.4."""

    punto = np.array([0.65, 0.45, 0.55])
    escalas_uniformes = np.array([1.7, 1.7, 1.7])
    escalas_no_uniformes = np.array([2.0, 0.5, 3.0])

    estados = []

    for _ in range(28):
        estados.append(
            crear_estado_escalado(
                escalas=np.ones(3),
                punto=punto,
                fase="1/4 · Cubo original",
                mensaje=(
                    "El cubo de referencia tiene el mismo tamaño en x, y y z. "
                    "Ahora modificaremos sus tres coordenadas con una matriz diagonal."
                ),
            )
        )

    for progreso in np.linspace(0.0, 1.0, 80):
        suave = suavizar(progreso)
        escalas = (1.0 - suave) * np.ones(3) + suave * escalas_uniformes
        estados.append(
            crear_estado_escalado(
                escalas=escalas,
                punto=punto,
                fase="2/4 · Escalado uniforme",
                mensaje=(
                    "Con sx=sy=sz el cubo aumenta de tamaño, pero conserva sus "
                    "proporciones porque todos los ejes se escalan igual."
                ),
            )
        )

    for _ in range(22):
        estados.append(
            crear_estado_escalado(
                escalas=escalas_uniformes,
                punto=punto,
                fase="2/4 · Escalado uniforme",
                mensaje=(
                    "Un escalado uniforme conserva la forma relativa: todas las "
                    "dimensiones cambian por el mismo factor."
                ),
            )
        )

    # Volvemos suavemente a la identidad antes del caso no uniforme.
    for progreso in np.linspace(0.0, 1.0, 45):
        suave = suavizar(progreso)
        escalas = (1.0 - suave) * escalas_uniformes + suave * np.ones(3)
        estados.append(
            crear_estado_escalado(
                escalas=escalas,
                punto=punto,
                fase="Transición · volver a I",
                mensaje="Regresamos al cubo original para comparar con un escalado no uniforme.",
            )
        )

    for progreso in np.linspace(0.0, 1.0, 95):
        suave = suavizar(progreso)
        escalas = (1.0 - suave) * np.ones(3) + suave * escalas_no_uniformes
        estados.append(
            crear_estado_escalado(
                escalas=escalas,
                punto=punto,
                fase="3/4 · Escalado no uniforme",
                mensaje=(
                    "Ahora sx, sy y sz son distintos. Cada dimensión cambia de "
                    "forma independiente y el volumen se multiplica por det(S)."
                ),
            )
        )

    for _ in range(30):
        estados.append(
            crear_estado_escalado(
                escalas=escalas_no_uniformes,
                punto=punto,
                fase="3/4 · S = diag(2, 0.5, 3)",
                mensaje=(
                    "El cubo se convierte en un ortoedro: x se duplica, y se "
                    "reduce a la mitad y z se triplica."
                ),
            )
        )

    S = np.diag(escalas_no_uniformes)
    R = rz(np.radians(38.0)) @ ry(np.radians(30.0)) @ rx(np.radians(-18.0))

    for _ in range(55):
        estados.append(crear_estado_comparacion(S, R))

    return {
        "states": estados,
        "S": S,
        "R": R,
        "point": punto,
    }


def imprimir_resultado(resultado):
    """Muestra los datos principales del escalado no uniforme."""

    S = resultado["S"]
    punto = resultado["point"]

    print("\n=== 3.4. Escalado en 3D ===")
    print("\nS =")
    print(S)
    print(f"\np = {formatear_vector(punto)}")
    print(f"Sp = {formatear_vector(S @ punto)}")
    print(f"det(S) = {np.linalg.det(S):.6f}")
    print("\nComparación:")
    print(f"det(R) = {np.linalg.det(resultado['R']):.6f}")


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
        / "04_escalado_3d.png"
    )
    video_path = (
        MATRICES_DIR
        / "assets"
        / "03_transformaciones_3d"
        / "04_escalado_3d.webm"
    )

    animacion = animador.animate_3d_states(
        states=resultado["states"],
        title="3.4. Escalado en 3D",
        limits=(-4.2, 4.2, -3.3, 3.3, -2.6, 3.6),
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
