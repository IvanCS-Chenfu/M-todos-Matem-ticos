from pathlib import Path
import sys

import networkx as nx
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


# ---------------------------------------------------------------------------
# Parámetros deterministas del ejemplo
# ---------------------------------------------------------------------------

NUMERO_INCREMENTOS = 96

AMPLITUD_X_PRINCIPAL = 4.40
AMPLITUD_X_SECUNDARIA = 0.55
AMPLITUD_Y_PRINCIPAL = 3.20
AMPLITUD_Y_SECUNDARIA = 0.35

SESGO_ESCALA_BASE = 1.007
AMPLITUD_ESCALA = 0.003
SESGO_LATERAL = 0.0015
AMPLITUD_LATERAL = 0.0018
SESGO_ANGULAR_GRADOS = 0.140
AMPLITUD_ANGULAR_GRADOS = 0.025
AMPLITUD_ANGULAR_SECUNDARIA_GRADOS = 0.012

TOLERANCIA_CIERRE_REAL = 1e-9
TOLERANCIA_REINTEGRACION = 1e-10


# ---------------------------------------------------------------------------
# Operaciones en SE(2)
# ---------------------------------------------------------------------------


def normalizar_angulo(angulo):
    """Normaliza un ángulo al intervalo [-pi, pi)."""

    angulo = float(angulo)

    if not np.isfinite(angulo):
        raise ValueError("El ángulo debe ser finito.")

    return (angulo + np.pi) % (2.0 * np.pi) - np.pi


def validar_pose(pose, nombre="pose"):
    """Valida una pose plana (x, y, theta)."""

    pose = np.asarray(pose, dtype=float)

    if pose.shape != (3,):
        raise ValueError(f"{nombre} debe contener exactamente tres componentes.")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{nombre} debe contener valores finitos.")

    resultado = pose.copy()
    resultado[2] = normalizar_angulo(resultado[2])
    return resultado


def validar_trayectoria(trayectoria, nombre="trayectoria"):
    """Valida una secuencia de poses con forma (N, 3)."""

    trayectoria = np.asarray(trayectoria, dtype=float)

    if trayectoria.ndim != 2 or trayectoria.shape[1] != 3:
        raise ValueError(f"{nombre} debe tener forma (N, 3).")
    if trayectoria.shape[0] < 2:
        raise ValueError(f"{nombre} debe contener al menos dos poses.")
    if not np.all(np.isfinite(trayectoria)):
        raise ValueError(f"{nombre} debe contener valores finitos.")

    resultado = trayectoria.copy()
    resultado[:, 2] = np.array(
        [normalizar_angulo(angulo) for angulo in resultado[:, 2]],
        dtype=float,
    )
    return resultado


def pose_a_matriz_se2(pose):
    """Convierte una pose 2D en una matriz homogénea de SE(2)."""

    x, y, theta = validar_pose(pose)
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, -s, x],
            [s, c, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def matriz_a_pose_se2(matriz):
    """Convierte una matriz homogénea válida en una pose plana."""

    matriz = np.asarray(matriz, dtype=float)

    if matriz.shape != (3, 3):
        raise ValueError("La matriz de SE(2) debe tener forma 3x3.")
    if not np.all(np.isfinite(matriz)):
        raise ValueError("La matriz de SE(2) debe contener valores finitos.")
    if not np.allclose(matriz[2], [0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("La última fila no corresponde a una matriz homogénea.")

    return validar_pose(
        np.array(
            [
                matriz[0, 2],
                matriz[1, 2],
                np.arctan2(matriz[1, 0], matriz[0, 0]),
            ],
            dtype=float,
        )
    )


def componer_poses_se2(pose_a, pose_b):
    """Calcula la composición pose_a ⊕ pose_b."""

    return matriz_a_pose_se2(
        pose_a_matriz_se2(pose_a) @ pose_a_matriz_se2(pose_b)
    )


def invertir_pose_se2(pose):
    """Calcula la transformación inversa de una pose."""

    matriz = pose_a_matriz_se2(pose)
    rotacion = matriz[:2, :2]
    traslacion = matriz[:2, 2]

    inversa = np.eye(3, dtype=float)
    inversa[:2, :2] = rotacion.T
    inversa[:2, 2] = -rotacion.T @ traslacion

    return matriz_a_pose_se2(inversa)


def calcular_movimiento_relativo(pose_origen, pose_destino):
    """Calcula el incremento local pose_origen⁻¹ ⊕ pose_destino."""

    return componer_poses_se2(
        invertir_pose_se2(pose_origen),
        pose_destino,
    )


# ---------------------------------------------------------------------------
# Trayectoria real y odometría ruidosa
# ---------------------------------------------------------------------------


def crear_trayectoria_real(numero_incrementos=NUMERO_INCREMENTOS):
    """Crea una trayectoria cerrada, suave y determinista."""

    numero_incrementos = int(numero_incrementos)

    if numero_incrementos < 24:
        raise ValueError("Se necesitan al menos 24 incrementos.")

    parametro = np.linspace(
        0.0,
        2.0 * np.pi,
        numero_incrementos + 1,
        dtype=float,
    )

    x = (
        AMPLITUD_X_PRINCIPAL * np.cos(parametro)
        + AMPLITUD_X_SECUNDARIA * np.cos(2.0 * parametro)
    )
    y = (
        AMPLITUD_Y_PRINCIPAL * np.sin(parametro)
        + AMPLITUD_Y_SECUNDARIA * np.sin(3.0 * parametro)
    )

    derivada_x = (
        -AMPLITUD_X_PRINCIPAL * np.sin(parametro)
        - 2.0 * AMPLITUD_X_SECUNDARIA * np.sin(2.0 * parametro)
    )
    derivada_y = (
        AMPLITUD_Y_PRINCIPAL * np.cos(parametro)
        + 3.0 * AMPLITUD_Y_SECUNDARIA * np.cos(3.0 * parametro)
    )
    theta = np.arctan2(derivada_y, derivada_x)

    # La primera pose define el origen del mapa.
    x = x - x[0]
    y = y - y[0]

    trayectoria = np.column_stack((x, y, theta))
    trayectoria[:, 2] = np.array(
        [normalizar_angulo(angulo) for angulo in trayectoria[:, 2]],
        dtype=float,
    )

    # El último punto coincide exactamente con el primero.
    trayectoria[-1, :2] = trayectoria[0, :2]
    trayectoria[-1, 2] = trayectoria[0, 2]

    return validar_trayectoria(trayectoria, "trayectoria real")


def crear_incrementos_reales(trayectoria_real):
    """Convierte una trayectoria absoluta en incrementos relativos."""

    trayectoria_real = validar_trayectoria(
        trayectoria_real,
        "trayectoria real",
    )

    return np.array(
        [
            calcular_movimiento_relativo(
                trayectoria_real[indice - 1],
                trayectoria_real[indice],
            )
            for indice in range(1, len(trayectoria_real))
        ],
        dtype=float,
    )


def generar_ruido_odometrico(incrementos_reales):
    """Añade escala, deriva angular y error lateral deterministas."""

    incrementos_reales = validar_trayectoria(
        incrementos_reales,
        "incrementos reales",
    )

    incrementos_medidos = []
    escalas = []
    errores_laterales = []
    errores_angulares = []
    errores_longitudinales = []

    for indice, incremento in enumerate(incrementos_reales, start=1):
        escala = (
            SESGO_ESCALA_BASE
            + AMPLITUD_ESCALA * np.sin(0.19 * indice)
        )
        error_longitudinal = 0.0010 * np.cos(0.33 * indice)
        error_lateral = (
            SESGO_LATERAL
            + AMPLITUD_LATERAL * np.sin(0.41 * indice)
        )
        error_angular = np.deg2rad(
            SESGO_ANGULAR_GRADOS
            + AMPLITUD_ANGULAR_GRADOS * np.sin(0.23 * indice)
            + AMPLITUD_ANGULAR_SECUNDARIA_GRADOS
            * np.cos(0.17 * indice)
        )

        incremento_medido = validar_pose(
            np.array(
                [
                    incremento[0] * escala + error_longitudinal,
                    incremento[1] + error_lateral,
                    normalizar_angulo(incremento[2] + error_angular),
                ],
                dtype=float,
            ),
            nombre=f"incremento medido {indice}",
        )

        incrementos_medidos.append(incremento_medido)
        escalas.append(escala)
        errores_laterales.append(error_lateral)
        errores_angulares.append(error_angular)
        errores_longitudinales.append(error_longitudinal)

    return {
        "measured_increments": np.asarray(incrementos_medidos, dtype=float),
        "scale_factors": np.asarray(escalas, dtype=float),
        "lateral_errors": np.asarray(errores_laterales, dtype=float),
        "angular_errors": np.asarray(errores_angulares, dtype=float),
        "longitudinal_errors": np.asarray(
            errores_longitudinales,
            dtype=float,
        ),
    }


def integrar_odometria(pose_inicial, incrementos_medidos):
    """Integra secuencialmente los incrementos medidos."""

    pose_inicial = validar_pose(pose_inicial, "pose inicial")
    incrementos_medidos = validar_trayectoria(
        incrementos_medidos,
        "incrementos medidos",
    )

    trayectoria = [pose_inicial.copy()]

    for incremento in incrementos_medidos:
        trayectoria.append(
            componer_poses_se2(trayectoria[-1], incremento)
        )

    return validar_trayectoria(
        np.asarray(trayectoria, dtype=float),
        "trayectoria estimada",
    )


def crear_grafo_odometria(
    trayectoria_real,
    trayectoria_estimada,
    incrementos_reales,
    incrementos_medidos,
):
    """Crea la cadena de poses y restricciones de odometría."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    trayectoria_estimada = validar_trayectoria(
        trayectoria_estimada,
        "trayectoria estimada",
    )
    incrementos_reales = validar_trayectoria(
        incrementos_reales,
        "incrementos reales",
    )
    incrementos_medidos = validar_trayectoria(
        incrementos_medidos,
        "incrementos medidos",
    )

    if trayectoria_real.shape != trayectoria_estimada.shape:
        raise ValueError("Las trayectorias deben tener la misma forma.")
    if len(incrementos_reales) != len(trayectoria_real) - 1:
        raise ValueError("Debe existir un incremento real menos que poses.")
    if incrementos_reales.shape != incrementos_medidos.shape:
        raise ValueError("Los incrementos reales y medidos deben coincidir.")

    graph = nx.Graph()

    for indice, (pose_real, pose_estimada) in enumerate(
        zip(trayectoria_real, trayectoria_estimada)
    ):
        nombre = f"x{indice}"
        graph.add_node(
            nombre,
            index=indice,
            node_type="pose",
            true_pose=pose_real.copy(),
            estimate=pose_estimada.copy(),
            dimension=3,
        )

    for indice, (incremento_real, incremento_medido) in enumerate(
        zip(incrementos_reales, incrementos_medidos),
        start=1,
    ):
        graph.add_edge(
            f"x{indice - 1}",
            f"x{indice}",
            factor_name=f"odom_{indice - 1}_{indice}",
            factor_type="odometry",
            true_measurement=incremento_real.copy(),
            measurement=incremento_medido.copy(),
            variables=(f"x{indice - 1}", f"x{indice}"),
        )

    graph.graph.update(
        {
            "state_dimension": 3 * len(trayectoria_real),
            "factor_count": len(incrementos_medidos),
            "reference_frame": "x0",
            "has_loop_closure": False,
            "description": "Cadena de odometría antes de Graph SLAM",
        }
    )

    return graph


# ---------------------------------------------------------------------------
# Errores y métricas
# ---------------------------------------------------------------------------


def calcular_error_posicion(trayectoria_real, trayectoria_estimada):
    """Calcula el error euclídeo de posición en cada pose."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    trayectoria_estimada = validar_trayectoria(
        trayectoria_estimada,
        "trayectoria estimada",
    )

    if trayectoria_real.shape != trayectoria_estimada.shape:
        raise ValueError("Las trayectorias deben tener la misma forma.")

    return np.linalg.norm(
        trayectoria_estimada[:, :2] - trayectoria_real[:, :2],
        axis=1,
    )


def calcular_error_orientacion(trayectoria_real, trayectoria_estimada):
    """Calcula el error angular firmado en cada pose."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    trayectoria_estimada = validar_trayectoria(
        trayectoria_estimada,
        "trayectoria estimada",
    )

    if trayectoria_real.shape != trayectoria_estimada.shape:
        raise ValueError("Las trayectorias deben tener la misma forma.")

    return np.array(
        [
            normalizar_angulo(estimada - real)
            for real, estimada in zip(
                trayectoria_real[:, 2],
                trayectoria_estimada[:, 2],
            )
        ],
        dtype=float,
    )


def calcular_error_incremental(incrementos_reales, incrementos_medidos):
    """Compara cada incremento real con su medición odométrica."""

    incrementos_reales = validar_trayectoria(
        incrementos_reales,
        "incrementos reales",
    )
    incrementos_medidos = validar_trayectoria(
        incrementos_medidos,
        "incrementos medidos",
    )

    if incrementos_reales.shape != incrementos_medidos.shape:
        raise ValueError("Los incrementos deben tener la misma forma.")

    error_traslacional = np.linalg.norm(
        incrementos_medidos[:, :2] - incrementos_reales[:, :2],
        axis=1,
    )
    error_angular = np.array(
        [
            normalizar_angulo(medido - real)
            for real, medido in zip(
                incrementos_reales[:, 2],
                incrementos_medidos[:, 2],
            )
        ],
        dtype=float,
    )

    return {
        "translation": error_traslacional,
        "orientation": error_angular,
        "cumulative_translation": np.concatenate(
            ([0.0], np.cumsum(error_traslacional))
        ),
        "cumulative_orientation": np.concatenate(
            ([0.0], np.cumsum(np.abs(error_angular)))
        ),
    }


def calcular_longitud_trayectoria(trayectoria):
    """Calcula la longitud euclídea recorrida por una trayectoria."""

    trayectoria = validar_trayectoria(trayectoria)
    desplazamientos = np.diff(trayectoria[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(desplazamientos, axis=1)))


def calcular_metricas(
    trayectoria_real,
    trayectoria_estimada,
    incrementos_reales,
    incrementos_medidos,
):
    """Calcula métricas globales y locales de deriva."""

    errores_posicion = calcular_error_posicion(
        trayectoria_real,
        trayectoria_estimada,
    )
    errores_orientacion = calcular_error_orientacion(
        trayectoria_real,
        trayectoria_estimada,
    )
    errores_incrementales = calcular_error_incremental(
        incrementos_reales,
        incrementos_medidos,
    )

    longitud_real = calcular_longitud_trayectoria(trayectoria_real)
    longitud_estimada = calcular_longitud_trayectoria(trayectoria_estimada)
    cierre_real = float(
        np.linalg.norm(
            trayectoria_real[-1, :2] - trayectoria_real[0, :2]
        )
    )
    cierre_estimado = float(
        np.linalg.norm(
            trayectoria_estimada[-1, :2]
            - trayectoria_estimada[0, :2]
        )
    )
    error_final = float(errores_posicion[-1])

    return {
        "pose_count": int(len(trayectoria_real)),
        "increment_count": int(len(incrementos_reales)),
        "real_length": longitud_real,
        "estimated_length": longitud_estimada,
        "position_rmse": float(np.sqrt(np.mean(errores_posicion**2))),
        "position_mae": float(np.mean(np.abs(errores_posicion))),
        "position_max": float(np.max(errores_posicion)),
        "position_final": error_final,
        "orientation_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(errores_orientacion**2)))
        ),
        "orientation_final_deg": float(
            np.rad2deg(errores_orientacion[-1])
        ),
        "orientation_max_deg": float(
            np.rad2deg(np.max(np.abs(errores_orientacion)))
        ),
        "real_closure_error": cierre_real,
        "estimated_closure_error": cierre_estimado,
        "drift_percent": 100.0 * error_final / longitud_real,
        "local_translation_mean": float(
            np.mean(errores_incrementales["translation"])
        ),
        "local_translation_max": float(
            np.max(errores_incrementales["translation"])
        ),
        "local_orientation_mean_deg": float(
            np.rad2deg(
                np.mean(np.abs(errores_incrementales["orientation"]))
            )
        ),
        "local_orientation_max_deg": float(
            np.rad2deg(
                np.max(np.abs(errores_incrementales["orientation"]))
            )
        ),
        "accumulated_local_translation": float(
            errores_incrementales["cumulative_translation"][-1]
        ),
        "accumulated_local_orientation_deg": float(
            np.rad2deg(
                errores_incrementales["cumulative_orientation"][-1]
            )
        ),
    }


def crear_simulacion_slam():
    """Construye toda la simulación determinista del apartado 6.1."""

    trayectoria_real = crear_trayectoria_real()
    incrementos_reales = crear_incrementos_reales(trayectoria_real)
    ruido = generar_ruido_odometrico(incrementos_reales)
    incrementos_medidos = ruido["measured_increments"]
    trayectoria_estimada = integrar_odometria(
        trayectoria_real[0],
        incrementos_medidos,
    )

    errores_posicion = calcular_error_posicion(
        trayectoria_real,
        trayectoria_estimada,
    )
    errores_orientacion = calcular_error_orientacion(
        trayectoria_real,
        trayectoria_estimada,
    )
    errores_incrementales = calcular_error_incremental(
        incrementos_reales,
        incrementos_medidos,
    )
    metricas = calcular_metricas(
        trayectoria_real,
        trayectoria_estimada,
        incrementos_reales,
        incrementos_medidos,
    )
    graph = crear_grafo_odometria(
        trayectoria_real,
        trayectoria_estimada,
        incrementos_reales,
        incrementos_medidos,
    )

    return {
        "true_trajectory": trayectoria_real,
        "estimated_trajectory": trayectoria_estimada,
        "true_increments": incrementos_reales,
        "measured_increments": incrementos_medidos,
        "position_errors": errores_posicion,
        "orientation_errors": errores_orientacion,
        "increment_translation_errors": errores_incrementales[
            "translation"
        ],
        "increment_orientation_errors": errores_incrementales[
            "orientation"
        ],
        "cumulative_increment_translation": errores_incrementales[
            "cumulative_translation"
        ],
        "cumulative_increment_orientation": errores_incrementales[
            "cumulative_orientation"
        ],
        "scale_factors": ruido["scale_factors"],
        "lateral_errors": ruido["lateral_errors"],
        "angular_noise": ruido["angular_errors"],
        "longitudinal_errors": ruido["longitudinal_errors"],
        "metrics": metricas,
        "graph": graph,
    }


# ---------------------------------------------------------------------------
# Estados didácticos
# ---------------------------------------------------------------------------


def crear_estado_animacion(
    *,
    phase,
    message,
    visible_true_count=0,
    visible_estimated_count=0,
    active_index=None,
    show_true=False,
    show_estimated=False,
    show_error_vector=False,
    show_error_history=False,
    show_increment=False,
    show_metrics=False,
    show_loop_closure=False,
    show_connections=False,
):
    """Crea un estado autocontenido de la explicación."""

    return {
        "phase": str(phase),
        "message": str(message),
        "visible_true_count": int(visible_true_count),
        "visible_estimated_count": int(visible_estimated_count),
        "active_index": None if active_index is None else int(active_index),
        "show_true": bool(show_true),
        "show_estimated": bool(show_estimated),
        "show_error_vector": bool(show_error_vector),
        "show_error_history": bool(show_error_history),
        "show_increment": bool(show_increment),
        "show_metrics": bool(show_metrics),
        "show_loop_closure": bool(show_loop_closure),
        "show_connections": bool(show_connections),
    }


def crear_estados_animacion(simulacion):
    """Crea la secuencia completa de la introducción a SLAM."""

    numero_poses = len(simulacion["true_trajectory"])
    states = []

    def add(phase, message, repeat=1, **kwargs):
        for _ in range(repeat):
            states.append(
                crear_estado_animacion(
                    phase=phase,
                    message=message,
                    **kwargs,
                )
            )

    add(
        "introduction",
        "SLAM estima simultáneamente la trayectoria del robot y el mapa.",
        repeat=3,
    )

    add(
        "ground_truth",
        "Primero se presenta la trayectoria real utilizada como referencia.",
        repeat=2,
        show_true=True,
        visible_true_count=1,
    )

    paso_revelado = 4
    for count in range(1, numero_poses + 1, paso_revelado):
        add(
            "ground_truth",
            "La trayectoria real es cerrada: el robot vuelve al punto inicial.",
            show_true=True,
            visible_true_count=min(count, numero_poses),
        )

    if states[-1]["visible_true_count"] != numero_poses:
        add(
            "ground_truth",
            "La trayectoria real completa servirá como ground truth.",
            repeat=2,
            show_true=True,
            visible_true_count=numero_poses,
        )

    add(
        "odometry_model",
        "La odometría mide incrementos locales con errores muy pequeños.",
        repeat=3,
        show_true=True,
        visible_true_count=numero_poses,
        show_increment=True,
        active_index=1,
    )

    for index in range(1, numero_poses):
        add(
            "integration",
            "Cada incremento ruidoso se compone con la estimación anterior.",
            show_true=True,
            show_estimated=True,
            visible_true_count=numero_poses,
            visible_estimated_count=index + 1,
            active_index=index,
            show_error_vector=True,
            show_error_history=True,
            show_increment=True,
        )

    add(
        "drift",
        "Los errores locales se han acumulado y la trayectoria estimada ya no cierra.",
        repeat=4,
        show_true=True,
        show_estimated=True,
        visible_true_count=numero_poses,
        visible_estimated_count=numero_poses,
        active_index=numero_poses - 1,
        show_error_vector=True,
        show_error_history=True,
        show_metrics=True,
        show_loop_closure=True,
    )

    add(
        "loop_closure",
        "Reconocer el lugar inicial permitiría añadir un cierre de ciclo.",
        repeat=4,
        show_true=True,
        show_estimated=True,
        visible_true_count=numero_poses,
        visible_estimated_count=numero_poses,
        active_index=numero_poses - 1,
        show_error_vector=True,
        show_error_history=True,
        show_metrics=True,
        show_loop_closure=True,
        show_connections=True,
    )

    add(
        "summary",
        "Graph SLAM utilizará restricciones redundantes para redistribuir la deriva.",
        repeat=4,
        show_true=True,
        show_estimated=True,
        visible_true_count=numero_poses,
        visible_estimated_count=numero_poses,
        active_index=numero_poses - 1,
        show_error_vector=True,
        show_error_history=True,
        show_metrics=True,
        show_loop_closure=True,
        show_connections=True,
    )

    for step, state in enumerate(states, start=1):
        state["step"] = step
        state["total_steps"] = len(states)

    return states


# ---------------------------------------------------------------------------
# Validaciones y salida
# ---------------------------------------------------------------------------


def validar_grafo_odometria(graph, simulacion):
    """Valida la cadena de poses antes de introducir cierres de ciclo."""

    numero_poses = len(simulacion["true_trajectory"])
    numero_incrementos = numero_poses - 1

    if graph.number_of_nodes() != numero_poses:
        raise ValueError("El grafo debe contener una variable por pose.")
    if graph.number_of_edges() != numero_incrementos:
        raise ValueError("El grafo debe contener una arista por incremento.")
    if not nx.is_tree(graph):
        raise ValueError("La odometría inicial debe formar una cadena sin ciclos.")
    if graph.graph.get("has_loop_closure"):
        raise ValueError("Este ejemplo todavía no debe contener cierres de ciclo.")

    for indice in range(numero_incrementos):
        if not graph.has_edge(f"x{indice}", f"x{indice + 1}"):
            raise ValueError("Falta una arista de odometría consecutiva.")


def validar_odometria(simulacion):
    """Comprueba coherencia geométrica y reproducibilidad."""

    real = validar_trayectoria(
        simulacion["true_trajectory"],
        "trayectoria real",
    )
    estimada = validar_trayectoria(
        simulacion["estimated_trajectory"],
        "trayectoria estimada",
    )
    incrementos_reales = validar_trayectoria(
        simulacion["true_increments"],
        "incrementos reales",
    )
    incrementos_medidos = validar_trayectoria(
        simulacion["measured_increments"],
        "incrementos medidos",
    )

    if real.shape != estimada.shape:
        raise ValueError("Las trayectorias deben tener la misma forma.")
    if len(incrementos_reales) != len(real) - 1:
        raise ValueError("Debe existir un incremento menos que poses.")
    if incrementos_reales.shape != incrementos_medidos.shape:
        raise ValueError("Los incrementos deben tener la misma forma.")
    if not np.allclose(real[0], estimada[0], atol=1e-12):
        raise ValueError("Las dos trayectorias deben comenzar en la misma pose.")

    cierre_real = np.linalg.norm(real[-1, :2] - real[0, :2])
    if cierre_real > TOLERANCIA_CIERRE_REAL:
        raise ValueError("La trayectoria real debe cerrar exactamente.")

    reintegrada = integrar_odometria(real[0], incrementos_medidos)
    if not np.allclose(
        reintegrada,
        estimada,
        atol=TOLERANCIA_REINTEGRACION,
        rtol=0.0,
    ):
        raise ValueError("La trayectoria estimada no coincide con la reintegración.")

    ruido_repetido = generar_ruido_odometrico(incrementos_reales)
    if not np.array_equal(
        ruido_repetido["measured_increments"],
        incrementos_medidos,
    ):
        raise ValueError("El ruido odométrico debe ser determinista.")


def validar_resultados(simulacion, states):
    """Ejecuta todas las comprobaciones matemáticas y didácticas."""

    validar_odometria(simulacion)
    validar_grafo_odometria(simulacion["graph"], simulacion)

    metricas = simulacion["metrics"]
    errores_posicion = simulacion["position_errors"]
    errores_orientacion = simulacion["orientation_errors"]

    if not np.all(np.isfinite(errores_posicion)):
        raise ValueError("Los errores de posición deben ser finitos.")
    if not np.all(np.isfinite(errores_orientacion)):
        raise ValueError("Los errores angulares deben ser finitos.")
    if np.any(errores_posicion < 0.0):
        raise ValueError("Los errores de posición no pueden ser negativos.")
    if metricas["position_final"] < 0.75:
        raise ValueError("La deriva final debe ser visualmente apreciable.")
    if metricas["estimated_closure_error"] < 0.75:
        raise ValueError("La odometría debe terminar lejos de la pose inicial.")
    if metricas["real_closure_error"] > TOLERANCIA_CIERRE_REAL:
        raise ValueError("La trayectoria real debe cerrar.")
    if metricas["local_translation_max"] >= metricas["position_final"]:
        raise ValueError("El error local debe ser menor que la deriva global.")
    if metricas["local_orientation_max_deg"] >= abs(
        metricas["orientation_final_deg"]
    ):
        raise ValueError("El error angular local debe ser menor que el acumulado.")
    if len(states) < 120:
        raise ValueError("La demostración debe contener al menos 120 estados.")
    if states[-1].get("phase") != "summary":
        raise ValueError("El último estado debe ser el resumen.")
    if not states[-1].get("show_true"):
        raise ValueError("La imagen final debe mostrar la trayectoria real.")
    if not states[-1].get("show_estimated"):
        raise ValueError("La imagen final debe mostrar la odometría.")
    if not states[-1].get("show_error_history"):
        raise ValueError("La imagen final debe mostrar el historial de error.")
    if not states[-1].get("show_loop_closure"):
        raise ValueError("La imagen final debe señalar el error de cierre.")

    return {
        "pose_count": metricas["pose_count"],
        "increment_count": metricas["increment_count"],
        "state_count": len(states),
        "graph_nodes": simulacion["graph"].number_of_nodes(),
        "graph_edges": simulacion["graph"].number_of_edges(),
        "real_length": metricas["real_length"],
        "estimated_length": metricas["estimated_length"],
        "position_final": metricas["position_final"],
        "orientation_final_deg": metricas["orientation_final_deg"],
        "position_rmse": metricas["position_rmse"],
        "position_max": metricas["position_max"],
        "drift_percent": metricas["drift_percent"],
        "local_translation_mean": metricas["local_translation_mean"],
        "local_orientation_mean_deg": metricas[
            "local_orientation_mean_deg"
        ],
    }


def imprimir_resumen(validation, metricas):
    """Imprime las magnitudes principales de la demostración."""

    print("\n=== Introducción a SLAM: deriva de odometría ===")
    print(f"Poses: {validation['pose_count']}")
    print(f"Incrementos: {validation['increment_count']}")
    print(
        f"Grafo de odometría: {validation['graph_nodes']} nodos · "
        f"{validation['graph_edges']} aristas"
    )
    print(f"Longitud real: {validation['real_length']:.6f} m")
    print(f"Longitud estimada: {validation['estimated_length']:.6f} m")
    print(f"Error final de posición: {validation['position_final']:.6f} m")
    print(
        "Error final de orientación: "
        f"{validation['orientation_final_deg']:.6f}°"
    )
    print(f"RMSE de posición: {validation['position_rmse']:.6f} m")
    print(f"Error máximo: {validation['position_max']:.6f} m")
    print(f"Deriva relativa: {validation['drift_percent']:.3f} %")
    print(
        "Error local medio: "
        f"{1000.0 * validation['local_translation_mean']:.3f} mm · "
        f"{validation['local_orientation_mean_deg']:.4f}°"
    )
    print(
        "Cierre real/estimado: "
        f"{metricas['real_closure_error']:.9f} m / "
        f"{metricas['estimated_closure_error']:.6f} m"
    )
    print(f"Estados de animación: {validation['state_count']}")


def main():
    simulacion = crear_simulacion_slam()
    states = crear_estados_animacion(simulacion)
    validation = validar_resultados(simulacion, states)

    imprimir_resumen(validation, simulacion["metrics"])

    animator = GraphAnimator(
        figsize=(18, 10),
        interval=210,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "06_graph_slam"
        / "01_slam_deriva_odometria.png"
    )

    animator.animate_slam_odometry_drift(
        simulation=simulacion,
        states=states,
        title="Introducción a SLAM: deriva acumulada de la odometría",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
