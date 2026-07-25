from math import atan2, cos, degrees, pi, radians, sin
from pathlib import Path
import sys

import networkx as nx
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


POSE_X0 = np.array([1.0, 1.0, radians(0.0)], dtype=float)
POSE_X1_INICIAL = np.array([4.2, 2.0, radians(18.0)], dtype=float)
MEDICION_Z01 = np.array([3.0, 0.5, radians(10.0)], dtype=float)
SIGMAS_Z01 = np.array([0.20, 0.30, radians(5.0)], dtype=float)


# ---------------------------------------------------------------------------
# Operaciones básicas sobre poses SE(2)
# ---------------------------------------------------------------------------


def normalizar_angulo(angulo):
    """Normaliza un ángulo al intervalo [-pi, pi)."""

    return (float(angulo) + pi) % (2.0 * pi) - pi


def pose_a_matriz_se2(pose):
    """Convierte una pose (x, y, theta) en una matriz homogénea 3x3."""

    x, y, theta = np.asarray(pose, dtype=float)
    c = cos(theta)
    s = sin(theta)

    return np.array(
        [
            [c, -s, x],
            [s, c, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def matriz_a_pose_se2(matriz):
    """Convierte una matriz homogénea SE(2) en una pose (x, y, theta)."""

    matriz = np.asarray(matriz, dtype=float)

    if matriz.shape != (3, 3):
        raise ValueError("La matriz SE(2) debe tener dimensiones 3x3.")

    theta = atan2(matriz[1, 0], matriz[0, 0])

    return np.array(
        [
            matriz[0, 2],
            matriz[1, 2],
            normalizar_angulo(theta),
        ],
        dtype=float,
    )


def componer_poses_se2(pose_a, pose_b):
    """Calcula pose_a ⊕ pose_b mediante composición de transformaciones."""

    matriz_compuesta = pose_a_matriz_se2(pose_a) @ pose_a_matriz_se2(pose_b)
    return matriz_a_pose_se2(matriz_compuesta)


def invertir_pose_se2(pose):
    """Calcula la transformación inversa de una pose SE(2)."""

    matriz_inversa = np.linalg.inv(pose_a_matriz_se2(pose))
    return matriz_a_pose_se2(matriz_inversa)


def calcular_pose_relativa(pose_i, pose_j):
    """Calcula la pose de j expresada en el sistema local de i."""

    return componer_poses_se2(invertir_pose_se2(pose_i), pose_j)


def calcular_pose_esperada(pose_i, medicion_relativa):
    """Propaga una medición relativa desde la pose de origen."""

    return componer_poses_se2(pose_i, medicion_relativa)


def calcular_residuo_se2(medicion, prediccion):
    """Calcula z^-1 ⊕ z_hat y devuelve un vector (ex, ey, e_theta)."""

    residuo = componer_poses_se2(invertir_pose_se2(medicion), prediccion)
    residuo[2] = normalizar_angulo(residuo[2])
    return residuo


def calcular_error_visual_global(pose_esperada, pose_actual):
    """Error intuitivo dibujado entre la pose esperada y la actual."""

    return np.array(
        [
            pose_actual[0] - pose_esperada[0],
            pose_actual[1] - pose_esperada[1],
            normalizar_angulo(pose_actual[2] - pose_esperada[2]),
        ],
        dtype=float,
    )


def crear_covarianza(sigmas):
    """Crea una covarianza diagonal a partir de desviaciones estándar."""

    sigmas = np.asarray(sigmas, dtype=float)

    if sigmas.shape != (3,):
        raise ValueError("Se esperaban tres desviaciones estándar.")

    if np.any(sigmas <= 0.0):
        raise ValueError("Todas las desviaciones estándar deben ser positivas.")

    return np.diag(sigmas**2)


def calcular_matriz_informacion(covarianza):
    """Calcula Ω = Σ^-1 comprobando que la matriz sea invertible."""

    covarianza = np.asarray(covarianza, dtype=float)

    if covarianza.shape != (3, 3):
        raise ValueError("La covarianza debe tener dimensiones 3x3.")

    if not np.allclose(covarianza, covarianza.T):
        raise ValueError("La covarianza debe ser simétrica.")

    autovalores = np.linalg.eigvalsh(covarianza)

    if np.any(autovalores <= 0.0):
        raise ValueError("La covarianza debe ser definida positiva.")

    return np.linalg.inv(covarianza)


def calcular_error_cuadratico(residuo):
    """Calcula ||e||² sin ponderación."""

    residuo = np.asarray(residuo, dtype=float)
    return float(residuo.T @ residuo)


def calcular_error_ponderado(residuo, informacion):
    """Calcula el coste de Mahalanobis eᵀΩe."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)
    return float(residuo.T @ informacion @ residuo)


def interpolar_pose(pose_inicial, pose_final, alpha):
    """Interpola traslación y el camino angular más corto entre dos poses."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha debe pertenecer al intervalo [0, 1].")

    pose_inicial = np.asarray(pose_inicial, dtype=float)
    pose_final = np.asarray(pose_final, dtype=float)

    diferencia_angular = normalizar_angulo(
        pose_final[2] - pose_inicial[2]
    )

    return np.array(
        [
            (1.0 - alpha) * pose_inicial[0] + alpha * pose_final[0],
            (1.0 - alpha) * pose_inicial[1] + alpha * pose_final[1],
            normalizar_angulo(
                pose_inicial[2] + alpha * diferencia_angular
            ),
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Creación y evaluación del grafo de restricciones
# ---------------------------------------------------------------------------


def crear_grafo_restricciones():
    """Crea dos poses, un prior y una restricción relativa de odometría."""

    covarianza = crear_covarianza(SIGMAS_Z01)
    informacion = calcular_matriz_informacion(covarianza)

    graph = nx.DiGraph()
    graph.graph["name"] = "Restricción básica entre dos poses"
    graph.graph["prior"] = {
        "node": "x0",
        "mean": POSE_X0.copy(),
        "fixed": True,
        "description": "La primera pose fija el sistema de referencia.",
    }

    graph.add_node(
        "x0",
        pose=POSE_X0.copy(),
        fixed=True,
        label="x0",
        description="Pose inicial fijada por un prior.",
    )
    graph.add_node(
        "x1",
        pose=POSE_X1_INICIAL.copy(),
        fixed=False,
        label="x1",
        description="Pose actual con error respecto a la medición.",
    )

    graph.add_edge(
        "x0",
        "x1",
        relation="restriccion_relativa",
        sensor="odometria",
        measurement=MEDICION_Z01.copy(),
        sigmas=SIGMAS_Z01.copy(),
        covariance=covarianza,
        information=informacion,
        frame="x0",
    )

    return graph


def evaluar_restriccion(graph, pose_x1):
    """Calcula predicción, residuo, error visual y costes de la arista."""

    pose_x0 = np.asarray(graph.nodes["x0"]["pose"], dtype=float)
    edge = graph.edges["x0", "x1"]
    medicion = np.asarray(edge["measurement"], dtype=float)
    informacion = np.asarray(edge["information"], dtype=float)

    prediccion = calcular_pose_relativa(pose_x0, pose_x1)
    pose_esperada = calcular_pose_esperada(pose_x0, medicion)
    residuo = calcular_residuo_se2(medicion, prediccion)
    error_visual = calcular_error_visual_global(pose_esperada, pose_x1)

    return {
        "pose_x0": pose_x0.copy(),
        "pose_x1": np.asarray(pose_x1, dtype=float).copy(),
        "pose_x1_esperada": pose_esperada,
        "measurement": medicion.copy(),
        "prediction": prediccion,
        "residual": residuo,
        "visual_error": error_visual,
        "unweighted_error": calcular_error_cuadratico(residuo),
        "weighted_error": calcular_error_ponderado(residuo, informacion),
        "covariance": np.asarray(edge["covariance"], dtype=float).copy(),
        "information": informacion.copy(),
        "sigmas": np.asarray(edge["sigmas"], dtype=float).copy(),
    }


def _serializar_vector(vector):
    return [float(value) for value in np.asarray(vector, dtype=float)]


def _serializar_matriz(matrix):
    return [
        [float(value) for value in row]
        for row in np.asarray(matrix, dtype=float)
    ]


def crear_estado_animacion(
    evaluation,
    *,
    phase,
    message,
    step,
    total_steps,
    initial_evaluation,
    final_evaluation,
    correction_alpha=0.0,
    focus=None,
    show_graph=True,
    show_constraint_details=False,
    show_geometry=False,
    show_pose_x0=False,
    show_pose_x1=False,
    show_expected_pose=False,
    show_measurement=False,
    show_prediction=False,
    show_translation_error=False,
    show_angular_error=False,
    show_uncertainty=False,
    show_prior=False,
    show_cost=False,
    show_comparison=False,
    show_future_graph=False,
):
    """Crea una copia independiente de un fotograma conceptual."""

    return {
        "phase": phase,
        "message": message,
        "step": step,
        "total_steps": total_steps,
        "focus": focus,
        "correction_alpha": float(correction_alpha),
        "pose_x0": _serializar_vector(evaluation["pose_x0"]),
        "pose_x1": _serializar_vector(evaluation["pose_x1"]),
        "pose_x1_initial": _serializar_vector(initial_evaluation["pose_x1"]),
        "pose_x1_expected": _serializar_vector(evaluation["pose_x1_esperada"]),
        "measurement": _serializar_vector(evaluation["measurement"]),
        "prediction": _serializar_vector(evaluation["prediction"]),
        "residual": _serializar_vector(evaluation["residual"]),
        "visual_error": _serializar_vector(evaluation["visual_error"]),
        "covariance": _serializar_matriz(evaluation["covariance"]),
        "information": _serializar_matriz(evaluation["information"]),
        "sigmas": _serializar_vector(evaluation["sigmas"]),
        "unweighted_error": float(evaluation["unweighted_error"]),
        "weighted_error": float(evaluation["weighted_error"]),
        "initial_prediction": _serializar_vector(initial_evaluation["prediction"]),
        "initial_residual": _serializar_vector(initial_evaluation["residual"]),
        "initial_visual_error": _serializar_vector(initial_evaluation["visual_error"]),
        "initial_unweighted_error": float(initial_evaluation["unweighted_error"]),
        "initial_weighted_error": float(initial_evaluation["weighted_error"]),
        "final_prediction": _serializar_vector(final_evaluation["prediction"]),
        "final_residual": _serializar_vector(final_evaluation["residual"]),
        "final_unweighted_error": float(final_evaluation["unweighted_error"]),
        "final_weighted_error": float(final_evaluation["weighted_error"]),
        "show_graph": bool(show_graph),
        "show_constraint_details": bool(show_constraint_details),
        "show_geometry": bool(show_geometry),
        "show_pose_x0": bool(show_pose_x0),
        "show_pose_x1": bool(show_pose_x1),
        "show_expected_pose": bool(show_expected_pose),
        "show_measurement": bool(show_measurement),
        "show_prediction": bool(show_prediction),
        "show_translation_error": bool(show_translation_error),
        "show_angular_error": bool(show_angular_error),
        "show_uncertainty": bool(show_uncertainty),
        "show_prior": bool(show_prior),
        "show_cost": bool(show_cost),
        "show_comparison": bool(show_comparison),
        "show_future_graph": bool(show_future_graph),
    }


def crear_estados_restriccion(graph, correction_steps=18):
    """Construye la narración completa y los estados de corrección de x1."""

    initial_evaluation = evaluar_restriccion(graph, POSE_X1_INICIAL)
    expected_pose = initial_evaluation["pose_x1_esperada"]
    final_evaluation = evaluar_restriccion(graph, expected_pose)

    states = []
    total_steps = 35 + correction_steps

    def add(evaluation, phase, message, **kwargs):
        states.append(
            crear_estado_animacion(
                evaluation=evaluation,
                phase=phase,
                message=message,
                step=len(states) + 1,
                total_steps=total_steps,
                initial_evaluation=initial_evaluation,
                final_evaluation=final_evaluation,
                **kwargs,
            )
        )

    # 1. De arista normal a arista de restricción.
    add(
        initial_evaluation,
        "normal_graph",
        "En un grafo normal, la arista solo indica que x0 y x1 están conectadas.",
        focus="connection",
    )
    add(
        initial_evaluation,
        "normal_graph",
        "Todavía no sabemos qué relación geométrica debería cumplirse.",
        focus="connection",
    )
    add(
        initial_evaluation,
        "constraint_graph",
        "La arista se convierte en una restricción: almacena z01, Σ01 y Ω01.",
        focus="measurement",
        show_constraint_details=True,
    )
    add(
        initial_evaluation,
        "constraint_graph",
        "La medición procede de odometría y está expresada en el sistema local de x0.",
        focus="measurement",
        show_constraint_details=True,
    )

    # 2. Aparición de las poses.
    add(
        initial_evaluation,
        "pose_x0",
        "x0 es una variable de pose: posición y orientación en el plano.",
        focus="x0",
        show_constraint_details=True,
        show_geometry=True,
        show_pose_x0=True,
    )
    add(
        initial_evaluation,
        "pose_x0",
        "Sus ejes locales indican cómo se interpretan los desplazamientos relativos.",
        focus="x0",
        show_constraint_details=True,
        show_geometry=True,
        show_pose_x0=True,
    )
    add(
        initial_evaluation,
        "pose_x1",
        "x1 es la estimación actual almacenada en el grafo.",
        focus="x1",
        show_constraint_details=True,
        show_geometry=True,
        show_pose_x0=True,
        show_pose_x1=True,
    )
    add(
        initial_evaluation,
        "pose_x1",
        "La conexión existe, pero aún debemos comprobar si x1 satisface la medición.",
        focus="x1",
        show_constraint_details=True,
        show_geometry=True,
        show_pose_x0=True,
        show_pose_x1=True,
    )

    # 3. Medición y pose esperada.
    for message in (
        "z01 indica el movimiento que el sensor espera entre x0 y x1.",
        "Se compone x0 ⊕ z01 para obtener la pose esperada de x1.",
        "La flecha verde representa la transformación relativa medida.",
        "La elipse todavía no se muestra: primero comparamos geometría y medición.",
    ):
        add(
            initial_evaluation,
            "measurement",
            message,
            focus="measurement",
            show_constraint_details=True,
            show_geometry=True,
            show_pose_x0=True,
            show_pose_x1=True,
            show_expected_pose=True,
            show_measurement=True,
        )

    # 4. Predicción desde las poses actuales.
    for message in (
        "Las poses actuales predicen otra relación: z_hat01 = x0^-1 ⊕ x1.",
        "La flecha morada termina en la pose actual, no en la pose esperada.",
        "Medición y predicción no coinciden: la restricción está incumplida.",
        "La diferencia se cuantifica mediante un residuo en SE(2).",
    ):
        add(
            initial_evaluation,
            "prediction",
            message,
            focus="prediction",
            show_constraint_details=True,
            show_geometry=True,
            show_pose_x0=True,
            show_pose_x1=True,
            show_expected_pose=True,
            show_measurement=True,
            show_prediction=True,
        )

    # 5. Residuo de traslación.
    for message in (
        "La flecha roja muestra el error visual de traslación entre ambas poses.",
        "En coordenadas globales: Δx = 0.20 m y Δy = 0.50 m.",
        "El residuo SE(2) expresa esa discrepancia en el marco de la medición.",
    ):
        add(
            initial_evaluation,
            "translation_residual",
            message,
            focus="translation_error",
            show_constraint_details=True,
            show_geometry=True,
            show_pose_x0=True,
            show_pose_x1=True,
            show_expected_pose=True,
            show_measurement=True,
            show_prediction=True,
            show_translation_error=True,
        )

    # 6. Residuo angular.
    for message in (
        "La orientación esperada es 10°, pero la estimación actual es 18°.",
        "El arco rojo representa un error angular normalizado de 8°.",
        "El residuo completo combina ex, ey y e_theta.",
    ):
        add(
            initial_evaluation,
            "angular_residual",
            message,
            focus="angular_error",
            show_constraint_details=True,
            show_geometry=True,
            show_pose_x0=True,
            show_pose_x1=True,
            show_expected_pose=True,
            show_measurement=True,
            show_prediction=True,
            show_translation_error=True,
            show_angular_error=True,
        )

    # 7. Incertidumbre e información.
    for message in (
        "La elipse representa la incertidumbre traslacional de la medición.",
        "Σ01 contiene varianzas; valores pequeños indican mayor precisión.",
        "Ω01 = Σ01^-1 convierte la incertidumbre en peso de la restricción.",
    ):
        add(
            initial_evaluation,
            "uncertainty",
            message,
            focus="uncertainty",
            show_constraint_details=True,
            show_geometry=True,
            show_pose_x0=True,
            show_pose_x1=True,
            show_expected_pose=True,
            show_measurement=True,
            show_prediction=True,
            show_translation_error=True,
            show_angular_error=True,
            show_uncertainty=True,
        )

    # 8. Coste.
    for message in (
        "El error sin ponderar es ||e01||².",
        "El coste de la arista es E01 = e01ᵀ Ω01 e01.",
        "Una discrepancia en una dirección precisa recibe una penalización mayor.",
    ):
        add(
            initial_evaluation,
            "cost",
            message,
            focus="cost",
            show_constraint_details=True,
            show_geometry=True,
            show_pose_x0=True,
            show_pose_x1=True,
            show_expected_pose=True,
            show_measurement=True,
            show_prediction=True,
            show_translation_error=True,
            show_angular_error=True,
            show_uncertainty=True,
            show_cost=True,
        )

    # 9. Prior y libertad gauge.
    for message in (
        "El prior fija x0 y define el sistema de referencia del problema.",
        "Sin prior, ambas poses podrían trasladarse juntas conservando el mismo error.",
        "Con x0 fija y una sola arista, la corrección puede aplicarse sobre x1.",
    ):
        add(
            initial_evaluation,
            "prior",
            message,
            focus="prior",
            show_constraint_details=True,
            show_geometry=True,
            show_pose_x0=True,
            show_pose_x1=True,
            show_expected_pose=True,
            show_measurement=True,
            show_prediction=True,
            show_translation_error=True,
            show_angular_error=True,
            show_uncertainty=True,
            show_prior=True,
            show_cost=True,
        )

    # 10. Corrección visual de x1.
    for index in range(1, correction_steps + 1):
        alpha = index / correction_steps
        current_pose = interpolar_pose(
            POSE_X1_INICIAL,
            expected_pose,
            alpha,
        )
        evaluation = evaluar_restriccion(graph, current_pose)

        add(
            evaluation,
            "correction",
            (
                f"Corrección de x1: paso {index}/{correction_steps}. "
                "La predicción se acerca a la medición y el coste disminuye."
            ),
            correction_alpha=alpha,
            focus="correction",
            show_constraint_details=True,
            show_geometry=True,
            show_pose_x0=True,
            show_pose_x1=True,
            show_expected_pose=True,
            show_measurement=True,
            show_prediction=True,
            show_translation_error=True,
            show_angular_error=True,
            show_uncertainty=True,
            show_prior=True,
            show_cost=True,
        )

    # 11. Comparación y transición a Pose Graph SLAM.
    add(
        final_evaluation,
        "comparison",
        "Comparación final: la pose corregida coincide con la pose esperada.",
        correction_alpha=1.0,
        focus="comparison",
        show_constraint_details=True,
        show_geometry=True,
        show_pose_x0=True,
        show_pose_x1=True,
        show_expected_pose=True,
        show_measurement=True,
        show_prediction=True,
        show_uncertainty=True,
        show_prior=True,
        show_cost=True,
        show_comparison=True,
    )
    add(
        final_evaluation,
        "comparison",
        "El residuo y el coste son aproximadamente cero para esta única restricción.",
        correction_alpha=1.0,
        focus="comparison",
        show_constraint_details=True,
        show_geometry=True,
        show_pose_x0=True,
        show_pose_x1=True,
        show_expected_pose=True,
        show_measurement=True,
        show_prediction=True,
        show_uncertainty=True,
        show_prior=True,
        show_cost=True,
        show_comparison=True,
    )
    add(
        final_evaluation,
        "pose_graph_preview",
        "Con muchas poses y cierres de ciclo, Graph SLAM minimiza la suma de todos los costes.",
        correction_alpha=1.0,
        focus="future_graph",
        show_constraint_details=True,
        show_geometry=True,
        show_pose_x0=True,
        show_pose_x1=True,
        show_expected_pose=True,
        show_measurement=True,
        show_prediction=True,
        show_uncertainty=True,
        show_prior=True,
        show_cost=True,
        show_comparison=True,
        show_future_graph=True,
    )
    add(
        final_evaluation,
        "summary",
        "Una arista de optimización es una medición, una incertidumbre y una función de error.",
        correction_alpha=1.0,
        focus="summary",
        show_constraint_details=True,
        show_geometry=True,
        show_pose_x0=True,
        show_pose_x1=True,
        show_expected_pose=True,
        show_measurement=True,
        show_prediction=True,
        show_uncertainty=True,
        show_prior=True,
        show_cost=True,
        show_comparison=True,
        show_future_graph=True,
    )

    # Se actualiza el total real por si cambia el guion anterior.
    for index, state in enumerate(states, start=1):
        state["step"] = index
        state["total_steps"] = len(states)

    return {
        "states": states,
        "initial": initial_evaluation,
        "final": final_evaluation,
        "expected_pose": expected_pose,
    }


# ---------------------------------------------------------------------------
# Validaciones y salida
# ---------------------------------------------------------------------------


def validar_transformaciones():
    """Comprueba identidad, inversión, composición y normalización angular."""

    identity_pose = np.zeros(3, dtype=float)
    test_pose = np.array([2.0, -1.0, radians(37.0)], dtype=float)

    if not np.allclose(
        componer_poses_se2(test_pose, identity_pose),
        test_pose,
        atol=1e-10,
    ):
        raise ValueError("La identidad por la derecha no se cumple.")

    if not np.allclose(
        componer_poses_se2(identity_pose, test_pose),
        test_pose,
        atol=1e-10,
    ):
        raise ValueError("La identidad por la izquierda no se cumple.")

    composed_inverse = componer_poses_se2(test_pose, invertir_pose_se2(test_pose))

    if not np.allclose(composed_inverse, identity_pose, atol=1e-10):
        raise ValueError("La inversión SE(2) no produce la identidad.")

    wrapped = normalizar_angulo(radians(358.0))

    if not np.isclose(degrees(wrapped), -2.0, atol=1e-10):
        raise ValueError("La normalización angular no trata correctamente 358°.")


def validar_grafo_restricciones(graph):
    """Comprueba nodos, prior, medición e incertidumbre del ejemplo."""

    if not isinstance(graph, nx.DiGraph):
        raise TypeError("El grafo debe ser un nx.DiGraph.")

    if set(graph.nodes()) != {"x0", "x1"}:
        raise ValueError("El ejemplo debe contener exactamente x0 y x1.")

    if not graph.has_edge("x0", "x1"):
        raise ValueError("Debe existir la restricción dirigida x0→x1.")

    if not graph.nodes["x0"].get("fixed", False):
        raise ValueError("x0 debe estar fijada por un prior.")

    edge = graph.edges["x0", "x1"]

    for field in (
        "measurement",
        "covariance",
        "information",
        "sensor",
    ):
        if field not in edge:
            raise ValueError(f"Falta el atributo de arista {field!r}.")

    covariance = np.asarray(edge["covariance"], dtype=float)
    information = np.asarray(edge["information"], dtype=float)

    if not np.allclose(covariance @ information, np.eye(3), atol=1e-10):
        raise ValueError("ΣΩ debe ser la matriz identidad.")


def validar_resultados(graph, result):
    """Comprueba los valores esenciales antes y después de la corrección."""

    initial = result["initial"]
    final = result["final"]
    expected_pose = result["expected_pose"]

    if initial["weighted_error"] <= 0.0:
        raise ValueError("El coste inicial debe ser positivo.")

    if not np.allclose(
        expected_pose,
        np.array([4.0, 1.5, radians(10.0)]),
        atol=1e-10,
    ):
        raise ValueError("La pose esperada no coincide con x0 ⊕ z01.")

    if not np.allclose(final["residual"], np.zeros(3), atol=1e-10):
        raise ValueError("El residuo final debe ser aproximadamente cero.")

    if final["weighted_error"] > 1e-16:
        raise ValueError("El coste final debe ser aproximadamente cero.")

    if initial["weighted_error"] <= final["weighted_error"]:
        raise ValueError("La corrección debe reducir el coste.")

    if len(result["states"]) < 45:
        raise ValueError("La animación debe contener suficientes estados didácticos.")

    graph.nodes["x1"]["pose"] = expected_pose.copy()
    graph.edges["x0", "x1"]["initial_prediction"] = initial["prediction"].copy()
    graph.edges["x0", "x1"]["initial_residual"] = initial["residual"].copy()
    graph.edges["x0", "x1"]["initial_weighted_error"] = initial[
        "weighted_error"
    ]
    graph.edges["x0", "x1"]["final_prediction"] = final["prediction"].copy()
    graph.edges["x0", "x1"]["final_residual"] = final["residual"].copy()
    graph.edges["x0", "x1"]["final_weighted_error"] = final[
        "weighted_error"
    ]


def _format_pose(pose):
    pose = np.asarray(pose, dtype=float)
    return (
        f"({pose[0]:.3f} m, {pose[1]:.3f} m, "
        f"{degrees(pose[2]):.3f}°)"
    )


def imprimir_resumen(graph, result):
    """Imprime un resumen numérico determinista del ejemplo."""

    initial = result["initial"]
    final = result["final"]
    edge = graph.edges["x0", "x1"]

    print("\n=== Grafo básico de restricciones entre dos poses ===")
    print(f"Pose fija x0: {_format_pose(initial['pose_x0'])}")
    print(f"Pose inicial x1: {_format_pose(initial['pose_x1'])}")
    print(f"Medición z01: {_format_pose(initial['measurement'])}")
    print(f"Pose esperada x1*: {_format_pose(result['expected_pose'])}")
    print(f"Predicción inicial: {_format_pose(initial['prediction'])}")
    print(f"Residuo SE(2) inicial: {_format_pose(initial['residual'])}")
    print(
        "Error visual global inicial: "
        f"{_format_pose(initial['visual_error'])}"
    )
    print(f"Coste sin ponderar inicial: {initial['unweighted_error']:.6f}")
    print(f"Coste ponderado inicial: {initial['weighted_error']:.6f}")
    print(f"Coste ponderado final: {final['weighted_error']:.12f}")
    print("Sensor de la arista:", edge["sensor"])
    print(f"Estados de animación: {len(result['states'])}")


def main():
    validar_transformaciones()

    graph = crear_grafo_restricciones()
    validar_grafo_restricciones(graph)

    result = crear_estados_restriccion(graph, correction_steps=18)
    validar_resultados(graph, result)
    validar_grafo_restricciones(graph)

    imprimir_resumen(graph, result)

    animator = GraphAnimator(
        figsize=(18, 10),
        interval=560,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "05_optimizacion"
        / "01_grafo_restricciones_basico.png"
    )

    animator.animate_basic_pose_constraint(
        graph=graph,
        states=result["states"],
        title="De una arista normal a una restricción entre poses",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()