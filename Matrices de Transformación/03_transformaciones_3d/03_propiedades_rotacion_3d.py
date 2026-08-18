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


def rotacion_interpolada(progreso, angulos_finales):
    """
    Genera siempre una matriz de rotación válida.

    No se interpolan directamente los nueve elementos de R. Se hacen variar
    los tres ángulos y se compone Rz Ry Rx en cada instante.
    """

    alpha, beta, gamma = progreso * np.asarray(angulos_finales, dtype=float)
    return rz(gamma) @ ry(beta) @ rx(alpha)


def formatear_vector(vector):
    vector = np.asarray(vector, dtype=float)
    return f"[{vector[0]:5.2f}, {vector[1]:5.2f}, {vector[2]:5.2f}]"


def angulo_entre(u, v):
    """Ángulo entre dos vectores en radianes."""

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    coseno = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.arccos(np.clip(coseno, -1.0, 1.0))


def suavizar(progreso):
    return 0.5 - 0.5 * np.cos(np.pi * progreso)


def crear_tetraedro():
    """
    Objeto asimétrico sencillo para comprobar visualmente que una rotación
    cambia orientación sin deformar su geometría.
    """

    return np.array([
        [0.0, 0.0, 0.0],
        [1.50, 0.10, 0.00],
        [0.25, 1.05, 0.10],
        [0.20, 0.25, 0.95],
    ])


def crear_estado(rotacion, rotacion_final, u, v, fase, mensaje):
    """Construye un estado que comprueba propiedades de R."""

    u_r = rotacion @ u
    v_r = rotacion @ v

    tetra = crear_tetraedro()
    tetra_r = (rotacion @ tetra.T).T

    identidad_error = np.linalg.norm(rotacion.T @ rotacion - np.eye(3))
    inversa_error = np.linalg.norm(np.linalg.inv(rotacion) - rotacion.T)
    determinante = np.linalg.det(rotacion)

    angulo_original = angulo_entre(u, v)
    angulo_rotado = angulo_entre(u_r, v_r)

    return {
        "frames3d": [
            {
                "name": "0",
                "origin": np.zeros(3),
                "rotation": np.eye(3),
                "length": 1.45,
                "alpha": 0.20,
                "colors": ("#9CA3AF", "#9CA3AF", "#9CA3AF"),
            },
            {
                "name": "R",
                "origin": np.zeros(3),
                "rotation": rotacion,
                "length": 1.75,
                "alpha": 1.0,
            },
        ],
        "meshes3d": [
            {
                "vertices": tetra,
                "faces": TETRA_FACES,
                "facecolor": "#CBD5E1",
                "edgecolor": "#64748B",
                "alpha": 0.08,
                "linewidth": 0.8,
            },
            {
                "vertices": tetra_r,
                "faces": TETRA_FACES,
                "facecolor": "#93C5FD",
                "edgecolor": "#1D4ED8",
                "alpha": 0.28,
                "linewidth": 1.1,
            },
        ],
        "vectors3d": [
            {
                "name": "u",
                "origin": np.zeros(3),
                "value": u_r,
                "color": "#B23A48",
                "linewidth": 3.0,
            },
            {
                "name": "v",
                "origin": np.zeros(3),
                "value": v_r,
                "color": "#2D7F5E",
                "linewidth": 3.0,
            },
        ],
        "message": mensaje,
        "info_title": "Propiedades de una rotación 3D",
        "info_lines": [
            {"text": "ORTOGONALIDAD", "bold": True},
            f"||R^T R - I|| = {identidad_error:.2e}",
            f"||R^-1-R^T||  = {inversa_error:.2e}",
            "",
            {"text": "DETERMINANTE", "bold": True},
            f"det(R) = {determinante: .6f}",
            "",
            {"text": "NORMAS", "bold": True},
            f"||u||  = {np.linalg.norm(u):.4f}",
            f"||Ru|| = {np.linalg.norm(u_r):.4f}",
            f"||v||  = {np.linalg.norm(v):.4f}",
            f"||Rv|| = {np.linalg.norm(v_r):.4f}",
            "",
            {"text": "ÁNGULO ENTRE u Y v", "bold": True},
            f"antes = {np.degrees(angulo_original):6.2f}°",
            f"después= {np.degrees(angulo_rotado):6.2f}°",
        ],
        "phase": fase,
        "info_line_height": 0.0405,
        "info_fontsize": 8.9,
        "legend": [
            {"kind": "line", "label": "Ru", "color": "#B23A48"},
            {"kind": "line", "label": "Rv", "color": "#2D7F5E"},
            {"kind": "line", "label": "objeto rotado", "color": "#1D4ED8"},
        ],
        "legend_ncol": 1,
        "legend_fontsize": 8.0,
    }


def crear_estado_conclusion(rotacion, u, v):
    """Fotograma final con las propiedades fundamentales resumidas."""

    estado = crear_estado(
        rotacion=rotacion,
        rotacion_final=rotacion,
        u=u,
        v=v,
        fase="Conclusión",
        mensaje=(
            "Una matriz de rotación cambia orientación sin deformar: conserva "
            "normas, ángulos y orientación espacial, y cumple R^-1 = R^T."
        ),
    )

    estado["info_lines"] = [
        {"text": "MATRIZ DE ROTACIÓN", "bold": True},
        "R^T R = I",
        "R^-1  = R^T",
        "det(R)= +1",
        "",
        {"text": "GEOMETRÍA", "bold": True},
        f"||u||  = {np.linalg.norm(u):.4f}",
        f"||Ru|| = {np.linalg.norm(rotacion @ u):.4f}",
        f"||v||  = {np.linalg.norm(v):.4f}",
        f"||Rv|| = {np.linalg.norm(rotacion @ v):.4f}",
        "",
        "Se conservan distancias",
        "y ángulos.",
        "El objeto solo cambia",
        "su orientación.",
    ]
    estado["info_line_height"] = 0.045

    return estado


def crear_estados_demostracion():
    """Construye la animación del apartado 3.3."""

    u = np.array([1.8, 0.4, 1.0])
    v = np.array([-0.6, 1.7, 0.5])

    angulos_finales = np.radians([30.0, -25.0, 55.0])
    rotacion_final = rotacion_interpolada(1.0, angulos_finales)

    estados = []

    for _ in range(30):
        estados.append(
            crear_estado(
                rotacion=np.eye(3),
                rotacion_final=rotacion_final,
                u=u,
                v=v,
                fase="1/3 · Estado inicial",
                mensaje=(
                    "Partimos de dos vectores y un objeto asimétrico. Mediremos "
                    "normas, ángulo, ortogonalidad y determinante durante el giro."
                ),
            )
        )

    for progreso in np.linspace(0.0, 1.0, 115):
        suave = suavizar(progreso)
        rotacion = rotacion_interpolada(suave, angulos_finales)

        estados.append(
            crear_estado(
                rotacion=rotacion,
                rotacion_final=rotacion_final,
                u=u,
                v=v,
                fase="2/3 · Rotación rígida",
                mensaje=(
                    "R cambia continuamente la orientación. El panel comprueba "
                    "que las magnitudes y el ángulo entre los vectores no cambian."
                ),
            )
        )

    for _ in range(35):
        estados.append(
            crear_estado(
                rotacion=rotacion_final,
                rotacion_final=rotacion_final,
                u=u,
                v=v,
                fase="3/3 · Comprobaciones",
                mensaje=(
                    "Con la orientación final se verifica R^T R=I, det(R)=1 y "
                    "R^-1=R^T, además de la conservación de normas y ángulos."
                ),
            )
        )

    for _ in range(50):
        estados.append(
            crear_estado_conclusion(
                rotacion=rotacion_final,
                u=u,
                v=v,
            )
        )

    return {
        "states": estados,
        "R": rotacion_final,
        "u": u,
        "v": v,
    }


def imprimir_resultado(resultado):
    """Muestra las comprobaciones numéricas principales."""

    R = resultado["R"]
    u = resultado["u"]
    v = resultado["v"]

    print("\n=== 3.3. Propiedades de una matriz de rotación 3D ===")
    print("\nR =")
    print(R)
    print("\nR^T R =")
    print(R.T @ R)
    print(f"\ndet(R) = {np.linalg.det(R):.8f}")
    print(f"||R^-1 - R^T|| = {np.linalg.norm(np.linalg.inv(R)-R.T):.3e}")
    print(f"||u|| = {np.linalg.norm(u):.6f}")
    print(f"||Ru|| = {np.linalg.norm(R @ u):.6f}")
    print(f"ángulo(u,v) = {np.degrees(angulo_entre(u,v)):.6f}°")
    print(f"ángulo(Ru,Rv) = {np.degrees(angulo_entre(R@u,R@v)):.6f}°")


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
        / "03_propiedades_rotacion_3d.png"
    )
    video_path = (
        MATRICES_DIR
        / "assets"
        / "03_transformaciones_3d"
        / "03_propiedades_rotacion_3d.webm"
    )

    animacion = animador.animate_3d_states(
        states=resultado["states"],
        title="3.3. Propiedades de una matriz de rotación 3D",
        limits=(-2.8, 3.0, -2.8, 3.0, -2.3, 3.0),
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
