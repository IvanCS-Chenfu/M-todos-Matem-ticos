from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def suavizar(progreso):
    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def formatear(v):
    v = np.asarray(v, dtype=float)
    return "[" + ", ".join(f"{x:6.2f}" for x in v) + "]"


def crear_estado_equivalencia(p, lambda_actual, fase, mensaje):
    p_h = np.r_[lambda_actual * p, lambda_actual]
    p_cart = p_h[:2] / p_h[2]

    return {
        "points": [
            {"name": "p", "position": p_cart, "color": "#7B2CBF", "size": 100},
        ],
        "vectors": [
            {"name": "dirección desde 0", "origin": np.zeros(2), "value": p_cart, "color": "#94A3B8", "alpha": 0.45, "linewidth": 1.8},
        ],
        "message": mensaje,
        "info_title": "Normalización homogénea",
        "info_lines": [
            {"text": "MISMO PUNTO", "bold": True},
            f"lambda = {lambda_actual:5.2f}",
            f"p_h = {formatear(p_h)}",
            "",
            {"text": "DIVIDIR ENTRE w", "bold": True},
            f"w = {p_h[2]:5.2f}",
            f"p = {formatear(p_cart)}",
            "",
            "Ejemplos equivalentes:",
            "[2,3,1]",
            "[4,6,2]",
            "[10,15,5]",
        ],
        "phase": fase,
        "info_line_height": 0.048,
        "legend": [
            {"kind": "point", "label": "punto cartesiano", "color": "#7B2CBF"},
        ],
        "legend_ncol": 1,
    }


def crear_estado_hacia_infinito(direccion, w, fase, mensaje):
    p_h = np.array([direccion[0], direccion[1], w], dtype=float)

    if w > 1e-9:
        p_cart = p_h[:2] / w
        points = [
            {"name": "p(w)", "position": p_cart, "color": "#7B2CBF", "size": 90},
        ]
    else:
        p_cart = None
        points = []

    # El rayo visible tiene longitud finita solo para representar la dirección.
    ray_unit = direccion / np.linalg.norm(direccion)
    ray_end = 14.0 * ray_unit

    info_lines = [
        {"text": "COORDENADA HOMOGÉNEA", "bold": True},
        f"p_h = {formatear(p_h)}",
        f"w   = {w:6.3f}",
        "",
    ]

    if p_cart is not None:
        info_lines.extend([
            {"text": "NORMALIZACIÓN", "bold": True},
            f"a/w = {p_cart[0]:7.2f}",
            f"b/w = {p_cart[1]:7.2f}",
            "",
            "Al disminuir w, el punto",
            "se aleja por la misma",
            "dirección proyectiva.",
        ])
    else:
        info_lines.extend([
            {"text": "w = 0", "bold": True},
            "No existe división finita.",
            "",
            "[a,b,0] representa una",
            "dirección / punto en el",
            "infinito en geometría",
            "proyectiva.",
        ])

    return {
        "points": points,
        "vectors": [
            {
                "name": "dirección [2,3]",
                "origin": np.zeros(2),
                "value": ray_end,
                "color": "#2D7F5E",
                "linewidth": 2.8,
            },
        ],
        "polylines": [
            {
                "points": np.array([[0.0, 0.0], ray_end]),
                "color": "#2D7F5E",
                "alpha": 0.35,
                "linewidth": 1.3,
                "linestyle": "--",
            }
        ],
        "message": mensaje,
        "info_title": "Cuando w se acerca a cero",
        "info_lines": info_lines,
        "phase": fase,
        "info_line_height": 0.049,
        "legend": [
            {"kind": "line", "label": "dirección proyectiva", "color": "#2D7F5E"},
            {"kind": "point", "label": "punto finito mientras w≠0", "color": "#7B2CBF"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.0,
    }


def crear_estado_conclusion(direccion):
    ray_unit = direccion / np.linalg.norm(direccion)
    ray_end = 14.0 * ray_unit

    return {
        "vectors": [
            {
                "name": "[2,3,0]",
                "origin": np.zeros(2),
                "value": ray_end,
                "color": "#2D7F5E",
                "linewidth": 3.0,
            },
        ],
        "texts": [
            {
                "position": (6.6, 10.5),
                "text": "w -> 0: punto hacia el infinito",
                "color": "#7B2CBF",
                "fontweight": "bold",
            },
        ],
        "message": (
            "Las coordenadas homogéneas no solo introducen traslaciones: w=0 "
            "incorpora direcciones y prepara la normalización proyectiva usada en homografías."
        ),
        "info_title": "Puente hacia geometría proyectiva",
        "info_lines": [
            {"text": "PUNTO FINITO", "bold": True},
            "[a,b,w], w != 0",
            "-> [a/w, b/w]",
            "",
            {"text": "PUNTO EN EL INFINITO", "bold": True},
            "[a,b,0]",
            "-> no normalizable",
            "-> asociado a dirección",
            "",
            {"text": "HOMOGRAFÍAS", "bold": True},
            "En una transformación",
            "proyectiva, w' puede",
            "depender de x e y.",
            "La división final crea",
            "el efecto de perspectiva.",
        ],
        "phase": "Conclusión",
        "info_line_height": 0.042,
        "info_fontsize": 9.0,
        "legend": [
            {"kind": "line", "label": "dirección / punto en infinito", "color": "#2D7F5E"},
        ],
        "legend_ncol": 1,
    }


def crear_estados_demostracion():
    """Construye la animación del apartado 4.6."""

    p = np.array([2.0, 3.0])
    direccion = np.array([2.0, 3.0])
    estados = []

    for _ in range(25):
        estados.append(
            crear_estado_equivalencia(p, 1.0, "1/4 · Punto finito", "Para w distinto de cero, las coordenadas cartesianas se recuperan dividiendo todas las componentes por w.")
        )

    for lambda_actual in np.linspace(1.0, 5.0, 85):
        estados.append(
            crear_estado_equivalencia(p, lambda_actual, "2/4 · Equivalencia por escala", "El triplete cambia con lambda, pero tras normalizar siempre recuperamos el mismo punto [2,3].")
        )

    # Empezamos con w=1 y lo reducimos solo hasta 0.25 para poder representar
    # un punto cartesiano finito dentro de los límites de la figura.
    for progreso in np.linspace(0.0, 1.0, 100):
        s = suavizar(progreso)
        w = (1.0 - s) * 1.0 + s * 0.25
        estados.append(
            crear_estado_hacia_infinito(direccion, w, "3/4 · w disminuye", "Manteniendo [a,b]=[2,3] y reduciendo w, el punto cartesiano [a/w,b/w] se aleja cada vez más por la misma dirección.")
        )

    for _ in range(30):
        estados.append(
            crear_estado_hacia_infinito(direccion, 0.25, "3/4 · Muy lejos", "Con w=0.25 el punto ya está en [8,12]. Si w siguiera acercándose a cero, su distancia crecería sin límite.")
        )

    for _ in range(40):
        estados.append(
            crear_estado_hacia_infinito(direccion, 0.0, "4/4 · w=0", "En w=0 ya no se puede dividir. La representación [2,3,0] se interpreta como una dirección o punto en el infinito.")
        )

    for _ in range(55):
        estados.append(crear_estado_conclusion(direccion))

    return {"states": estados, "p": p, "direction": direccion}


def imprimir_resultado(resultado):
    p = resultado["p"]
    print("\n=== 4.6. Normalización homogénea y puntos en el infinito ===")
    for lam in (1.0, 2.0, 5.0):
        ph = np.r_[lam * p, lam]
        print(f"{ph} -> {ph[:2] / ph[2]}")
    print("[2,3,0] no puede normalizarse a un punto cartesiano finito.")


def main():
    resultado = crear_estados_demostracion()
    imprimir_resultado(resultado)

    animador = TransformAnimator(figsize=(15.5, 8.8), interval=50)
    image_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "06_normalizacion_infinito.png"
    video_path = MATRICES_DIR / "assets" / "04_coordenadas_homogeneas" / "06_normalizacion_infinito.webm"

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="4.6. Normalización homogénea y puntos en el infinito",
        limits=(-1.5, 12.5, -1.5, 18.5),
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
