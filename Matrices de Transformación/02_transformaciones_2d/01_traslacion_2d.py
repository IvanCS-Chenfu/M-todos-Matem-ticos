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


def interpolar(inicio, fin, cantidad):
    """Genera valores entre inicio y fin, incluyendo ambos extremos."""

    return np.linspace(inicio, fin, cantidad)


def suavizar(progreso):
    """Interpolación cosenoidal para movimientos suaves."""

    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def trasladar_puntos(puntos, traslacion):
    """Aplica p' = p + t a una colección de puntos 2D."""

    puntos = np.asarray(puntos, dtype=float)
    traslacion = np.asarray(traslacion, dtype=float)

    return puntos + traslacion


def crear_estado_traslacion(
    punto_inicial,
    traslacion_final,
    triangulo_inicial,
    progreso,
    fase,
    mensaje,
):
    """
    Crea un estado donde un punto y un triángulo se trasladan sin deformarse.

    La traslación se calcula directamente mediante suma vectorial. No se
    utiliza todavía una matriz homogénea porque ese formalismo pertenece a
    apartados posteriores de la wiki.
    """

    punto_inicial = np.asarray(punto_inicial, dtype=float)
    traslacion_final = np.asarray(traslacion_final, dtype=float)
    triangulo_inicial = np.asarray(triangulo_inicial, dtype=float)

    traslacion_actual = progreso * traslacion_final
    punto_actual = punto_inicial + traslacion_actual
    triangulo_actual = trasladar_puntos(
        triangulo_inicial,
        traslacion_actual,
    )

    origen = np.zeros(2)
    origen_trasladado = origen + traslacion_actual

    return {
        "polygons": [
            {
                "points": triangulo_inicial,
                "facecolor": "#D1D5DB",
                "edgecolor": "#6B7280",
                "alpha": 0.18,
                "linewidth": 1.4,
            },
            {
                "points": triangulo_actual,
                "facecolor": "#DCEAF7",
                "edgecolor": "#1F77B4",
                "alpha": 0.34,
                "linewidth": 2.0,
            },
        ],
        "points": [
            {
                "name": "p1",
                "position": punto_inicial,
                "color": "#6B7280",
                "alpha": 0.75,
                "size": 75,
            },
            {
                "name": "p(t)",
                "position": punto_actual,
                "color": "#7B2CBF",
                "size": 95,
                "label_offset": (0.18, -0.28),
            },
            {
                "name": "0",
                "position": origen,
                "color": "#111827",
                "size": 60,
                "label_offset": (0.12, -0.28),
            },
            {
                "name": "f(0)",
                "position": origen_trasladado,
                "color": "#D97706",
                "alpha": 0.95 if progreso > 0.02 else 0.0,
                "size": 75,
                "label_offset": (0.18, -0.36),
            },
        ],
        "vectors": [
            {
                "name": "t(t)",
                "origin": punto_inicial,
                "value": traslacion_actual,
                "color": "#E07A1F",
                "alpha": 0.95 if progreso > 0.02 else 0.0,
                "linewidth": 3.0,
                "label_offset": (-0.78, 0.28),
            },
            {
                "name": "t",
                "origin": origen,
                "value": traslacion_actual,
                "color": "#D97706",
                "alpha": 0.75 if progreso > 0.02 else 0.0,
                "linewidth": 2.4,
                "linestyle": "--",
                "label_offset": (0.22, 0.22),
            },
        ],
        "message": mensaje,
        "info_title": "Traslación 2D",
        "info_lines": [
            {"text": "DATOS", "bold": True},
            f"p1       = {formatear_vector(punto_inicial)}",
            f"t final  = {formatear_vector(traslacion_final)}",
            "",
            {"text": "ESTADO ACTUAL", "bold": True},
            f"t(t)     = {formatear_vector(traslacion_actual)}",
            f"p(t)     = {formatear_vector(punto_actual)}",
            "",
            {"text": "REGLA", "bold": True},
            "p(t) = p1 + t(t)",
            "",
            {"text": "ORIGEN", "bold": True},
            f"f(0)     = {formatear_vector(origen_trasladado)}",
            "Si t != 0, el origen se mueve.",
        ],
        "phase": fase,
        "info_line_height": 0.047,
        "legend": [
            {"kind": "line", "label": "figura original", "color": "#6B7280"},
            {"kind": "line", "label": "figura trasladada", "color": "#1F77B4"},
            {"kind": "line", "label": "vector t", "color": "#E07A1F"},
            {"kind": "point", "label": "p(t)", "color": "#7B2CBF"},
        ],
        "legend_ncol": 2,
        "legend_fontsize": 8.2,
    }


def crear_estado_final(
    punto_inicial,
    traslacion,
    triangulo_inicial,
):
    """Crea el fotograma final con la conclusión del apartado 2.1."""

    estado = crear_estado_traslacion(
        punto_inicial=punto_inicial,
        traslacion_final=traslacion,
        triangulo_inicial=triangulo_inicial,
        progreso=1.0,
        fase="Conclusión",
        mensaje=(
            "Todos los puntos reciben el mismo vector t: la figura cambia de "
            "posición, pero no de forma ni de orientación. Como f(0)=t, una "
            "traslación no puede escribirse como Ap con una matriz 2x2."
        ),
    )

    estado["info_lines"] = [
        {"text": "EJEMPLO NUMÉRICO", "bold": True},
        f"p1 = {formatear_vector(punto_inicial)}",
        f"t  = {formatear_vector(traslacion)}",
        f"p2 = {formatear_vector(punto_inicial + traslacion)}",
        "",
        {"text": "PROPIEDADES", "bold": True},
        "Misma forma.",
        "Misma orientación.",
        "Mismo tamaño.",
        "",
        {"text": "LIMITACIÓN 2x2", "bold": True},
        f"f(0) = t = {formatear_vector(traslacion)}",
        "pero A·0 = 0 para toda A 2x2.",
    ]

    return estado


def crear_estados_demostracion():
    """Construye la animación del apartado 2.1."""

    punto_inicial = np.array([3.0, 2.0])
    traslacion = np.array([5.0, -1.0])

    # El triángulo se coloca alrededor del punto del ejemplo numérico.
    triangulo_inicial = np.array([
        [2.0, 1.2],
        [4.2, 1.4],
        [3.0, 3.5],
    ])

    estados = []

    for _ in range(30):
        estados.append(
            crear_estado_traslacion(
                punto_inicial=punto_inicial,
                traslacion_final=traslacion,
                triangulo_inicial=triangulo_inicial,
                progreso=0.0,
                fase="1/3 · Punto, figura y vector de traslación",
                mensaje=(
                    "Partimos del punto p1=[3,2] y de una figura fija. "
                    "La traslación se definirá mediante el vector t=[5,-1]."
                ),
            )
        )

    for progreso in interpolar(0.0, 1.0, 95):
        estados.append(
            crear_estado_traslacion(
                punto_inicial=punto_inicial,
                traslacion_final=traslacion,
                triangulo_inicial=triangulo_inicial,
                progreso=suavizar(progreso),
                fase="2/3 · Aplicar p2 = p1 + t",
                mensaje=(
                    "A cada punto de la figura se le suma exactamente el mismo "
                    "vector. La forma y la orientación no cambian."
                ),
            )
        )

    for _ in range(35):
        estados.append(
            crear_estado_final(
                punto_inicial=punto_inicial,
                traslacion=traslacion,
                triangulo_inicial=triangulo_inicial,
            )
        )

    return {
        "states": estados,
        "point": punto_inicial,
        "translation": traslacion,
        "result": punto_inicial + traslacion,
    }


def imprimir_resultado(resultado):
    """Muestra por terminal el ejemplo principal del apartado 2.1."""

    print("\n=== 2.1. Traslación en 2D ===")
    print(f"\np1 = {formatear_vector(resultado['point'])}")
    print(f"t  = {formatear_vector(resultado['translation'])}")
    print(f"p2 = p1 + t = {formatear_vector(resultado['result'])}")

    print("\nComprobación del origen:")
    print(f"  f(0) = t = {formatear_vector(resultado['translation'])}")
    print("  Una matriz 2x2 siempre cumple A·0 = 0.")


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
        / "01_traslacion_2d.png"
    )

    video_path = (
        MATRICES_DIR
        / "assets"
        / "02_transformaciones_2d"
        / "01_traslacion_2d.webm"
    )

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="2.1. Traslación en 2D",
        limits=(-1.5, 10.8, -2.0, 5.5),
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
