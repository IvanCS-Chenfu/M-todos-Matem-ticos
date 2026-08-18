from pathlib import Path
import sys

import numpy as np


# Permite importar módulos desde la carpeta Matrices_de_Transformacion/
CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def obtener_ejes_frame(angulo):
    """
    Devuelve los ejes unitarios x e y de un frame 2D.

    No se utiliza todavía una matriz homogénea porque el objetivo del
    apartado 1.1 es distinguir:
    - objeto geométrico,
    - coordenadas,
    - punto,
    - vector,
    - sistema de referencia.
    """

    eje_x = np.array([
        np.cos(angulo),
        np.sin(angulo),
    ])

    eje_y = np.array([
        -np.sin(angulo),
        np.cos(angulo),
    ])

    return eje_x, eje_y


def coordenadas_punto_en_frame(punto, origen_frame, angulo_frame):
    """
    Calcula las coordenadas de un mismo punto físico respecto a un frame.

    Primero se mide el desplazamiento desde el origen del frame hasta el
    punto. Después se proyecta ese desplazamiento sobre los dos ejes del
    sistema de referencia.
    """

    punto = np.asarray(punto, dtype=float)
    origen_frame = np.asarray(origen_frame, dtype=float)

    eje_x, eje_y = obtener_ejes_frame(angulo_frame)
    desplazamiento = punto - origen_frame

    return np.array([
        np.dot(desplazamiento, eje_x),
        np.dot(desplazamiento, eje_y),
    ])


def coordenadas_vector_en_frame(vector, angulo_frame):
    """
    Calcula las componentes de un vector respecto a los ejes de un frame.

    Un vector representa dirección y magnitud, no una posición. Por eso el
    origen del frame no interviene en el cálculo. Una traslación del sistema
    de referencia no cambia las componentes del vector; una rotación de los
    ejes sí puede cambiarlas.
    """

    vector = np.asarray(vector, dtype=float)
    eje_x, eje_y = obtener_ejes_frame(angulo_frame)

    return np.array([
        np.dot(vector, eje_x),
        np.dot(vector, eje_y),
    ])


def formatear_vector(vector):
    """
    Devuelve un vector 2D con un formato compacto para mostrarlo en pantalla.
    """

    return f"[{vector[0]:6.2f}, {vector[1]:6.2f}]"


def crear_estado(
    punto,
    vector,
    origen_vector,
    origen_b,
    angulo_b,
    alpha_b,
    fase,
    mensaje,
):
    """
    Crea un estado completo de la demostración del apartado 1.1.

    El frame {A} permanece fijo. El frame {B} se traslada y después rota.
    El punto P y el vector v permanecen físicamente inmóviles durante toda
    la animación: únicamente cambia la descripción mediante coordenadas.
    """

    origen_a = np.array([0.0, 0.0])
    angulo_a = 0.0

    punto_a = coordenadas_punto_en_frame(
        punto,
        origen_a,
        angulo_a,
    )
    punto_b = coordenadas_punto_en_frame(
        punto,
        origen_b,
        angulo_b,
    )

    vector_a = coordenadas_vector_en_frame(
        vector,
        angulo_a,
    )
    vector_b = coordenadas_vector_en_frame(
        vector,
        angulo_b,
    )

    norma_vector = np.linalg.norm(vector)

    return {
        "frames": [
            {
                "name": "A",
                "origin": origen_a,
                "angle": angulo_a,
                "length": 1.75,
                "x_color": "#B23A48",
                "y_color": "#2D7F5E",
                "alpha": 1.0,
            },
            {
                "name": "B",
                "origin": np.asarray(origen_b, dtype=float),
                "angle": float(angulo_b),
                "length": 1.75,
                "x_color": "#D97904",
                "y_color": "#1F77B4",
                "alpha": float(alpha_b),
            },
        ],
        "points": [
            {
                "name": "P",
                "position": np.asarray(punto, dtype=float),
                "color": "#7B2CBF",
            },
        ],
        "vectors": [
            {
                "name": "v",
                "origin": np.asarray(origen_vector, dtype=float),
                "value": np.asarray(vector, dtype=float),
                "color": "#E07A1F",
            },
        ],
        "segments": [
            {
                "start": origen_a,
                "end": np.asarray(punto, dtype=float),
                "color": "#7B2CBF",
                "alpha": 0.22,
                "linestyle": "--",
            },
            {
                "start": np.asarray(origen_b, dtype=float),
                "end": np.asarray(punto, dtype=float),
                "color": "#1F77B4",
                "alpha": 0.24 * float(alpha_b),
                "linestyle": "--",
            },
        ],
        "message": mensaje,
        "info_title": "Coordenadas del mismo objeto",
        "info_lines": [
            {"text": "OBJETOS GEOMÉTRICOS", "bold": True},
            f"P físico     = {formatear_vector(punto)}",
            f"v físico     = {formatear_vector(vector)}",
            f"||v||        = {norma_vector:6.2f}",
            "",
            {"text": "RESPECTO AL FRAME {A}", "bold": True},
            f"^A p         = {formatear_vector(punto_a)}",
            f"^A v         = {formatear_vector(vector_a)}",
            "",
            {"text": "RESPECTO AL FRAME {B}", "bold": True},
            f"^B p         = {formatear_vector(punto_b)}",
            f"^B v         = {formatear_vector(vector_b)}",
            "",
            {"text": "POSE DEL FRAME {B}", "bold": True},
            f"origen B     = {formatear_vector(origen_b)}",
            f"ángulo B     = {np.degrees(angulo_b):6.1f}°",
        ],
        "phase": fase,
    }


def interpolar(inicio, fin, cantidad):
    """
    Genera valores entre `inicio` y `fin`, incluyendo ambos extremos.
    """

    return np.linspace(inicio, fin, cantidad)


def crear_estados_demostracion():
    """
    Construye todos los estados de la animación del apartado 1.1.

    La demostración se divide en cuatro ideas:

    1. Un punto y un vector son objetos geométricos distintos.
    2. Aparece un segundo frame {B} inicialmente coincidente con {A}.
    3. Se traslada {B}: cambia ^B p, pero ^B v permanece igual.
    4. Se rota {B}: también cambian las componentes de v respecto a {B},
       aunque el vector físico conserva dirección y magnitud en la escena.

    Así se visualiza que las coordenadas describen un objeto respecto a un
    sistema de referencia; no son el objeto geométrico en sí.
    """

    punto = np.array([4.15, 2.65])
    vector = np.array([1.65, 0.85])
    origen_vector = np.array([-3.55, -2.35])

    origen_b_inicial = np.array([0.0, 0.0])
    origen_b_final = np.array([2.25, 0.95])
    angulo_b_final = np.radians(35.0)

    estados = []

    # --------------------------------------------------------------
    # Fase 1: presentar punto, vector y frame de referencia {A}.
    # --------------------------------------------------------------
    for _ in range(28):
        estados.append(
            crear_estado(
                punto=punto,
                vector=vector,
                origen_vector=origen_vector,
                origen_b=origen_b_inicial,
                angulo_b=0.0,
                alpha_b=0.0,
                fase="1/4 · Punto, vector y coordenadas",
                mensaje=(
                    "P es una posición. v representa dirección y magnitud. "
                    "Sus componentes están expresadas respecto a {A}."
                ),
            )
        )

    # --------------------------------------------------------------
    # Fase 2: hacer aparecer un segundo sistema coincidente con {A}.
    # --------------------------------------------------------------
    for alpha_b in interpolar(0.0, 1.0, 22):
        estados.append(
            crear_estado(
                punto=punto,
                vector=vector,
                origen_vector=origen_vector,
                origen_b=origen_b_inicial,
                angulo_b=0.0,
                alpha_b=alpha_b,
                fase="2/4 · Dos sistemas de referencia",
                mensaje=(
                    "{A} y {B} coinciden: el mismo objeto tiene las mismas "
                    "coordenadas en ambos frames."
                ),
            )
        )

    for _ in range(18):
        estados.append(
            crear_estado(
                punto=punto,
                vector=vector,
                origen_vector=origen_vector,
                origen_b=origen_b_inicial,
                angulo_b=0.0,
                alpha_b=1.0,
                fase="2/4 · Dos sistemas de referencia",
                mensaje=(
                    "Mientras origen y ejes coinciden, ^A p = ^B p y "
                    "^A v = ^B v."
                ),
            )
        )

    # --------------------------------------------------------------
    # Fase 3: trasladar {B} sin rotarlo.
    # --------------------------------------------------------------
    for progreso in interpolar(0.0, 1.0, 70):
        # Interpolación suave para que el movimiento arranque y termine
        # progresivamente.
        suave = 0.5 - 0.5 * np.cos(np.pi * progreso)
        origen_b = (
            (1.0 - suave) * origen_b_inicial
            + suave * origen_b_final
        )

        estados.append(
            crear_estado(
                punto=punto,
                vector=vector,
                origen_vector=origen_vector,
                origen_b=origen_b,
                angulo_b=0.0,
                alpha_b=1.0,
                fase="3/4 · Trasladar el sistema de referencia",
                mensaje=(
                    "P no se mueve, pero ^B p cambia porque cambia el origen "
                    "de {B}. Las componentes de v no cambian por traslación."
                ),
            )
        )

    for _ in range(28):
        estados.append(
            crear_estado(
                punto=punto,
                vector=vector,
                origen_vector=origen_vector,
                origen_b=origen_b_final,
                angulo_b=0.0,
                alpha_b=1.0,
                fase="3/4 · Trasladar el sistema de referencia",
                mensaje=(
                    "La posición depende del origen elegido. Una dirección "
                    "ideal no depende de dónde esté situado ese origen."
                ),
            )
        )

    # --------------------------------------------------------------
    # Fase 4: rotar los ejes de {B} manteniendo su origen fijo.
    # --------------------------------------------------------------
    for progreso in interpolar(0.0, 1.0, 80):
        suave = 0.5 - 0.5 * np.cos(np.pi * progreso)
        angulo_b = suave * angulo_b_final

        estados.append(
            crear_estado(
                punto=punto,
                vector=vector,
                origen_vector=origen_vector,
                origen_b=origen_b_final,
                angulo_b=angulo_b,
                alpha_b=1.0,
                fase="4/4 · Rotar los ejes del sistema",
                mensaje=(
                    "Al rotar {B}, cambian las componentes ^B p y ^B v. "
                    "P y v siguen siendo los mismos objetos físicos."
                ),
            )
        )

    # Pausa final para poder leer con claridad el resultado.
    for _ in range(45):
        estados.append(
            crear_estado(
                punto=punto,
                vector=vector,
                origen_vector=origen_vector,
                origen_b=origen_b_final,
                angulo_b=angulo_b_final,
                alpha_b=1.0,
                fase="Conclusión",
                mensaje=(
                    "Las coordenadas dependen del frame. El objeto geométrico "
                    "permanece igual aunque cambie su descripción numérica."
                ),
            )
        )

    return {
        "states": estados,
        "point": punto,
        "vector": vector,
        "frame_b_origin": origen_b_final,
        "frame_b_angle": angulo_b_final,
    }


def imprimir_resultado(resultado):
    """
    Imprime un pequeño resumen numérico de la demostración.
    """

    punto = resultado["point"]
    vector = resultado["vector"]
    origen_b = resultado["frame_b_origin"]
    angulo_b = resultado["frame_b_angle"]

    punto_a = coordenadas_punto_en_frame(
        punto,
        origen_frame=np.array([0.0, 0.0]),
        angulo_frame=0.0,
    )
    vector_a = coordenadas_vector_en_frame(
        vector,
        angulo_frame=0.0,
    )

    punto_b = coordenadas_punto_en_frame(
        punto,
        origen_frame=origen_b,
        angulo_frame=angulo_b,
    )
    vector_b = coordenadas_vector_en_frame(
        vector,
        angulo_frame=angulo_b,
    )

    print("\n=== 1.1. Puntos, vectores y transformaciones geométricas ===")

    print("\nObjetos físicos utilizados:")
    print(f"  P = {formatear_vector(punto)}")
    print(f"  v = {formatear_vector(vector)}")
    print(f"  ||v|| = {np.linalg.norm(vector):.3f}")

    print("\nCoordenadas respecto a {A}:")
    print(f"  ^A p = {formatear_vector(punto_a)}")
    print(f"  ^A v = {formatear_vector(vector_a)}")

    print("\nPose final del frame {B}:")
    print(f"  origen = {formatear_vector(origen_b)}")
    print(f"  ángulo = {np.degrees(angulo_b):.1f}°")

    print("\nCoordenadas respecto a {B}:")
    print(f"  ^B p = {formatear_vector(punto_b)}")
    print(f"  ^B v = {formatear_vector(vector_b)}")

    print("\nIdea principal:")
    print(
        "  P y v no han cambiado físicamente; lo que cambia es su "
        "descripción respecto al sistema de referencia."
    )


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
        / "01_fundamentos"
        / "01_puntos_vectores_transformaciones.png"
    )

    video_path = (
        MATRICES_DIR
        / "assets"
        / "01_fundamentos"
        / "01_puntos_vectores_transformaciones.webm"
    )

    animacion = animador.animate_2d_states(
        states=resultado["states"],
        title="1.1. Puntos, vectores y sistemas de referencia",
        limits=(-5.0, 6.2, -3.6, 5.0),
        final_image_path=image_path,
        video_path=video_path,
        repeat=False,
        fps=20,
        dpi=125,
        show=True,
    )

    # Mantener la referencia mientras la ventana está abierta.
    _ = animacion


if __name__ == "__main__":
    main()
