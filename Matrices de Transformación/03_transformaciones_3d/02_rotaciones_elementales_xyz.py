from pathlib import Path
import sys

import numpy as np


# Permite importar módulos desde la carpeta Matrices de Transformación/
CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rx(theta):
    """Matriz de rotación elemental alrededor del eje x."""

    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def ry(theta):
    """Matriz de rotación elemental alrededor del eje y."""

    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rz(theta):
    """Matriz de rotación elemental alrededor del eje z."""

    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


ROTACIONES = {
    "x": rx,
    "y": ry,
    "z": rz,
}


def formatear_vector(vector):
    """Devuelve un vector 3D con formato compacto."""

    vector = np.asarray(vector, dtype=float)
    return f"[{vector[0]:5.2f}, {vector[1]:5.2f}, {vector[2]:5.2f}]"


def formatear_matriz(matriz):
    """Devuelve una matriz 3x3 como tres cadenas de texto."""

    matriz = np.asarray(matriz, dtype=float)
    return [
        f"[{matriz[0, 0]:6.3f}, {matriz[0, 1]:6.3f}, {matriz[0, 2]:6.3f}]",
        f"[{matriz[1, 0]:6.3f}, {matriz[1, 1]:6.3f}, {matriz[1, 2]:6.3f}]",
        f"[{matriz[2, 0]:6.3f}, {matriz[2, 1]:6.3f}, {matriz[2, 2]:6.3f}]",
    ]


def suavizar(progreso):
    """Interpolación cosenoidal para variar el ángulo suavemente."""

    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def crear_arco_eje(eje, theta, radio=1.25, muestras=60):
    """
    Genera un arco circular en el plano perpendicular al eje de giro.

    El arco se usa únicamente para visualizar el ángulo positivo según la
    regla de la mano derecha.
    """

    if abs(theta) < 1e-9:
        valores = np.array([0.0, 1e-6])
    else:
        valores = np.linspace(0.0, theta, muestras)

    if eje == "x":
        return np.column_stack((
            np.zeros_like(valores),
            radio * np.cos(valores),
            radio * np.sin(valores),
        ))

    if eje == "y":
        return np.column_stack((
            radio * np.sin(valores),
            np.zeros_like(valores),
            radio * np.cos(valores),
        ))

    return np.column_stack((
        radio * np.cos(valores),
        radio * np.sin(valores),
        np.zeros_like(valores),
    ))


def crear_estado(eje, theta, punto, fase, mensaje):
    """Construye un estado para una rotación elemental 3D."""

    rotacion = ROTACIONES[eje](theta)
    punto_rotado = rotacion @ punto
    matriz_texto = formatear_matriz(rotacion)

    colores_eje = {
        "x": "#C63C3C",
        "y": "#2A8F5B",
        "z": "#1F77B4",
    }

    eje_unitario = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }[eje]

    return {
        "frames3d": [
            {
                "name": "0",
                "origin": np.zeros(3),
                "rotation": np.eye(3),
                "length": 1.65,
                "alpha": 0.22,
                "colors": ("#9CA3AF", "#9CA3AF", "#9CA3AF"),
                "origin_color": "#6B7280",
            },
            {
                "name": "R",
                "origin": np.zeros(3),
                "rotation": rotacion,
                "length": 1.85,
                "alpha": 1.0,
            },
        ],
        "points3d": [
            {
                "name": "p",
                "position": punto,
                "color": "#6B7280",
                "alpha": 0.35,
                "size": 42,
            },
            {
                "name": "Rp",
                "position": punto_rotado,
                "color": "#7B2CBF",
                "size": 72,
            },
        ],
        "vectors3d": [
            {
                "name": f"eje {eje}",
                "origin": -2.0 * eje_unitario,
                "value": 4.0 * eje_unitario,
                "color": colores_eje[eje],
                "alpha": 0.60,
                "linewidth": 2.4,
                "show_origin": False,
            },
            {
                "name": "p",
                "origin": np.zeros(3),
                "value": punto,
                "color": "#6B7280",
                "alpha": 0.35,
                "linewidth": 1.8,
            },
            {
                "name": "Rp",
                "origin": np.zeros(3),
                "value": punto_rotado,
                "color": "#7B2CBF",
                "linewidth": 3.0,
            },
        ],
        "polylines3d": [
            {
                "points": crear_arco_eje(eje, theta),
                "color": colores_eje[eje],
                "linewidth": 2.5,
                "alpha": 0.90,
            },
        ],
        "message": mensaje,
        "info_title": f"Rotación elemental alrededor de {eje.upper()}",
        "info_lines": [
            {"text": "ÁNGULO", "bold": True},
            f"theta = {np.degrees(theta):6.1f}°",
            "",
            {"text": f"R{eje.upper()}(theta)", "bold": True},
            matriz_texto[0],
            matriz_texto[1],
            matriz_texto[2],
            "",
            {"text": "PUNTO", "bold": True},
            f"p  = {formatear_vector(punto)}",
            f"Rp = {formatear_vector(punto_rotado)}",
            "",
            "Sentido positivo:",
            "regla de la mano derecha.",
        ],
        "phase": fase,
        "info_line_height": 0.044,
        "info_fontsize": 9.3,
        "legend": [
            {"kind": "line", "label": f"eje {eje}", "color": colores_eje[eje]},
            {"kind": "line", "label": "p original", "color": "#6B7280"},
            {"kind": "line", "label": "Rp", "color": "#7B2CBF"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.0,
    }


def añadir_bloque_rotacion(estados, eje, angulo_final, punto, nombre_fase):
    """Añade avance, pausa y retorno a identidad para una rotación elemental."""

    for progreso in np.linspace(0.0, 1.0, 80):
        theta = suavizar(progreso) * angulo_final
        estados.append(
            crear_estado(
                eje=eje,
                theta=theta,
                punto=punto,
                fase=f"{nombre_fase} · giro positivo",
                mensaje=(
                    f"R{eje} modifica únicamente las dos coordenadas del plano "
                    f"perpendicular al eje {eje}. El eje de giro permanece fijo."
                ),
            )
        )

    for _ in range(22):
        estados.append(
            crear_estado(
                eje=eje,
                theta=angulo_final,
                punto=punto,
                fase=f"{nombre_fase} · resultado",
                mensaje=(
                    f"La rotación alrededor de {eje} conserva distancias y gira "
                    "según la regla de la mano derecha."
                ),
            )
        )

    for progreso in np.linspace(0.0, 1.0, 45):
        theta = (1.0 - suavizar(progreso)) * angulo_final
        estados.append(
            crear_estado(
                eje=eje,
                theta=theta,
                punto=punto,
                fase=f"{nombre_fase} · volver a I",
                mensaje=(
                    "Volvemos a la identidad para estudiar la siguiente "
                    "rotación elemental de forma independiente."
                ),
            )
        )


def crear_estado_rz_90(punto):
    """Estado final con el ejemplo explícito Rz(90°)."""

    theta = np.radians(90.0)
    estado = crear_estado(
        eje="z",
        theta=theta,
        punto=punto,
        fase="Conclusión · Rz(90°)",
        mensaje=(
            "Con Rz(90°), el eje x pasa a la dirección +y y el eje y pasa a "
            "la dirección -x. Este es el ejemplo elemental de referencia."
        ),
    )

    estado["info_lines"] = [
        {"text": "EJEMPLO Rz(90°)", "bold": True},
        "[ 0, -1,  0]",
        "[ 1,  0,  0]",
        "[ 0,  0,  1]",
        "",
        f"p       = {formatear_vector(punto)}",
        f"Rz(90)p = {formatear_vector(rz(theta) @ punto)}",
        "",
        {"text": "INTERPRETACIÓN", "bold": True},
        "z permanece fijo.",
        "x gira hacia +y.",
        "Sentido positivo según",
        "la mano derecha.",
    ]

    return estado


def crear_estados_demostracion():
    """Construye la animación del apartado 3.2."""

    punto = np.array([2.0, 1.0, 0.8])
    estados = []

    for _ in range(25):
        estados.append(
            crear_estado(
                eje="x",
                theta=0.0,
                punto=punto,
                fase="Introducción",
                mensaje=(
                    "En 3D existen tres rotaciones elementales: alrededor de "
                    "los ejes x, y y z."
                ),
            )
        )

    añadir_bloque_rotacion(
        estados,
        eje="x",
        angulo_final=np.radians(65.0),
        punto=punto,
        nombre_fase="1/3 · Rx",
    )
    añadir_bloque_rotacion(
        estados,
        eje="y",
        angulo_final=np.radians(55.0),
        punto=punto,
        nombre_fase="2/3 · Ry",
    )

    # El bloque z termina directamente en 90° para conservar el ejemplo final.
    for progreso in np.linspace(0.0, 1.0, 95):
        theta = suavizar(progreso) * np.radians(90.0)
        estados.append(
            crear_estado(
                eje="z",
                theta=theta,
                punto=punto,
                fase="3/3 · Rz",
                mensaje=(
                    "Rz gira en el plano xy. El eje z permanece fijo y el "
                    "sentido positivo es antihorario visto desde +z."
                ),
            )
        )

    for _ in range(55):
        estados.append(crear_estado_rz_90(punto))

    return {
        "states": estados,
        "point": punto,
        "rz90": rz(np.radians(90.0)),
    }


def imprimir_resultado(resultado):
    """Imprime el ejemplo final Rz(90°)."""

    print("\n=== 3.2. Rotaciones elementales X, Y y Z ===")
    print("\nRz(90°) =")
    print(resultado["rz90"])
    print(f"\np = {formatear_vector(resultado['point'])}")
    print(
        "Rz(90°)p = "
        f"{formatear_vector(resultado['rz90'] @ resultado['point'])}"
    )


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
        / "02_rotaciones_elementales_xyz.png"
    )
    video_path = (
        MATRICES_DIR
        / "assets"
        / "03_transformaciones_3d"
        / "02_rotaciones_elementales_xyz.webm"
    )

    animacion = animador.animate_3d_states(
        states=resultado["states"],
        title="3.2. Rotaciones elementales alrededor de X, Y y Z",
        limits=(-3.2, 3.2, -3.2, 3.2, -3.0, 3.2),
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
