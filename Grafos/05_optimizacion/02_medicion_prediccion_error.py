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


# ---------------------------------------------------------------------------
# Datos deterministas del ejemplo
# ---------------------------------------------------------------------------

POSE_X0 = np.array([1.0, 1.0, radians(25.0)], dtype=float)
POSE_X1_INICIAL = np.array([4.2, 3.0, radians(48.0)], dtype=float)
MEDICION_Z01 = np.array([3.0, 0.4, radians(15.0)], dtype=float)
SIGMAS_Z01 = np.array([0.18, 0.28, radians(4.0)], dtype=float)


# ---------------------------------------------------------------------------
# Operaciones básicas sobre poses SE(2)
# ---------------------------------------------------------------------------


def normalizar_angulo(angulo):
    """Normaliza un ángulo al intervalo [-pi, pi)."""

    return (float(angulo) + pi) % (2.0 * pi) - pi


def pose_a_matriz_se2(pose):
    """Convierte una pose (x, y, theta) en una matriz homogénea 3x3."""

    pose = np.asarray(pose, dtype=float)

    if pose.shape != (3,):
        raise ValueError("Una pose SE(2) debe contener exactamente tres valores.")

    x, y, theta = pose
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

    if not np.allclose(matriz[2], np.array([0.0, 0.0, 1.0]), atol=1e-10):
        raise ValueError("La última fila no corresponde a una transformación SE(2).")

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
    """Calcula pose_a ⊕ pose_b mediante matrices homogéneas."""

    return matriz_a_pose_se2(
        pose_a_matriz_se2(pose_a) @ pose_a_matriz_se2(pose_b)
    )


def invertir_pose_se2(pose):
    """Calcula la transformación inversa de una pose plana."""

    return matriz_a_pose_se2(np.linalg.inv(pose_a_matriz_se2(pose)))


def calcular_prediccion_relativa(pose_i, pose_j):
    """Calcula la medición que predicen las poses actuales: xi^-1 ⊕ xj."""

    return componer_poses_se2(invertir_pose_se2(pose_i), pose_j)


def calcular_pose_esperada(pose_i, medicion_relativa):
    """Convierte la medición relativa en una pose global esperada."""

    return componer_poses_se2(pose_i, medicion_relativa)


def calcular_residuo_se2(medicion, prediccion):
    """Calcula el residuo geométrico z^-1 ⊕ z_hat."""

    residuo = componer_poses_se2(invertir_pose_se2(medicion), prediccion)
    residuo[2] = normalizar_angulo(residuo[2])
    return residuo


def calcular_error_visual_global(pose_esperada, pose_estimada):
    """Calcula una diferencia global intuitiva para la representación."""

    pose_esperada = np.asarray(pose_esperada, dtype=float)
    pose_estimada = np.asarray(pose_estimada, dtype=float)

    return np.array(
        [
            pose_estimada[0] - pose_esperada[0],
            pose_estimada[1] - pose_esperada[1],
            normalizar_angulo(pose_estimada[2] - pose_esperada[2]),
        ],
        dtype=float,
    )


def calcular_error_traslacional(error_visual):
    """Devuelve la norma euclídea del error visual de posición."""

    error_visual = np.asarray(error_visual, dtype=float)

    if error_visual.shape != (3,):
        raise ValueError("El error visual debe tener tres componentes.")

    return float(np.linalg.norm(error_visual[:2]))


def calcular_error_angular(error_visual):
    """Devuelve la magnitud angular normalizada del error visual."""

    error_visual = np.asarray(error_visual, dtype=float)

    if error_visual.shape != (3,):
        raise ValueError("El error visual debe tener tres componentes.")

    return abs(normalizar_angulo(error_visual[2]))


# ---------------------------------------------------------------------------
# Incertidumbre y costes
# ---------------------------------------------------------------------------


def crear_covarianza(sigmas):
    """Crea una covarianza diagonal desde desviaciones estándar positivas."""

    sigmas = np.asarray(sigmas, dtype=float)

    if sigmas.shape != (3,):
        raise ValueError("Se esperaban tres desviaciones estándar.")

    if not np.all(np.isfinite(sigmas)):
        raise ValueError("Las desviaciones estándar deben ser finitas.")

    if np.any(sigmas <= 0.0):
        raise ValueError("Todas las desviaciones estándar deben ser positivas.")

    return np.diag(sigmas**2)


def calcular_matriz_informacion(covarianza):
    """Calcula Ω = Σ^-1 tras validar la covarianza."""

    covarianza = np.asarray(covarianza, dtype=float)

    if covarianza.shape != (3, 3):
        raise ValueError("La covarianza debe tener dimensiones 3x3.")

    if not np.all(np.isfinite(covarianza)):
        raise ValueError("La covarianza debe contener valores finitos.")

    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise ValueError("La covarianza debe ser simétrica.")

    autovalores = np.linalg.eigvalsh(covarianza)

    if np.any(autovalores <= 0.0):
        raise ValueError("La covarianza debe ser definida positiva.")

    return np.linalg.inv(covarianza)


def calcular_error_cuadratico(residuo):
    """Calcula el error cuadrático sin ponderar e^T e."""

    residuo = np.asarray(residuo, dtype=float)

    if residuo.shape != (3,):
        raise ValueError("El residuo debe tener tres componentes.")

    return float(residuo.T @ residuo)


def calcular_contribuciones_ponderadas(residuo, informacion):
    """Calcula la contribución de cada componente para una Ω diagonal."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)

    if residuo.shape != (3,) or informacion.shape != (3, 3):
        raise ValueError("Dimensiones incompatibles para calcular contribuciones.")

    if not np.allclose(informacion, np.diag(np.diag(informacion)), atol=1e-12):
        raise ValueError(
            "La descomposición por componentes requiere una matriz diagonal."
        )

    return residuo**2 * np.diag(informacion)


def calcular_error_ponderado(residuo, informacion):
    """Calcula el coste de Mahalanobis e^T Ω e."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)

    if residuo.shape != (3,) or informacion.shape != (3, 3):
        raise ValueError("Dimensiones incompatibles para calcular el coste.")

    return float(residuo.T @ informacion @ residuo)


# ---------------------------------------------------------------------------
# Grafo, evaluación y estimaciones
# ---------------------------------------------------------------------------


def crear_grafo_medicion_prediccion():
    """Crea dos variables de pose y una medición relativa de odometría."""

    covarianza = crear_covarianza(SIGMAS_Z01)
    informacion = calcular_matriz_informacion(covarianza)

    graph = nx.DiGraph()
    graph.graph["name"] = "Variables, medición, predicción y error"
    graph.graph["convention"] = "residual = measurement^-1 ⊕ prediction"
    graph.graph["prior"] = {
        "node": "x0",
        "mean": POSE_X0.copy(),
        "fixed": True,
        "source": "referencia_global",
    }

    graph.add_node(
        "x0",
        variable_type="pose_se2",
        estimate=POSE_X0.copy(),
        fixed=True,
        label="x0",
        description="Variable de pose de origen fijada por un prior.",
    )
    graph.add_node(
        "x1",
        variable_type="pose_se2",
        estimate=POSE_X1_INICIAL.copy(),
        fixed=False,
        label="x1",
        description="Variable estimada que cambia durante la demostración.",
    )

    graph.add_edge(
        "x0",
        "x1",
        relation="medicion_relativa",
        sensor="odometria",
        measurement=MEDICION_Z01.copy(),
        frame="x0",
        sigmas=SIGMAS_Z01.copy(),
        covariance=covarianza,
        information=informacion,
        immutable_measurement=True,
    )

    return graph


def evaluar_modelo_medicion(graph, pose_x1):
    """Calcula predicción, residuo, incertidumbre y costes para una estimación."""

    if not graph.has_edge("x0", "x1"):
        raise ValueError("El grafo debe contener la medición dirigida x0→x1.")

    pose_x0 = np.asarray(graph.nodes["x0"]["estimate"], dtype=float)
    pose_x1 = np.asarray(pose_x1, dtype=float)
    edge = graph.edges["x0", "x1"]

    medicion = np.asarray(edge["measurement"], dtype=float)
    covarianza = np.asarray(edge["covariance"], dtype=float)
    informacion = np.asarray(edge["information"], dtype=float)
    sigmas = np.asarray(edge["sigmas"], dtype=float)

    prediccion = calcular_prediccion_relativa(pose_x0, pose_x1)
    pose_esperada = calcular_pose_esperada(pose_x0, medicion)
    residuo = calcular_residuo_se2(medicion, prediccion)
    error_visual = calcular_error_visual_global(pose_esperada, pose_x1)
    contribuciones = calcular_contribuciones_ponderadas(residuo, informacion)

    return {
        "pose_x0": pose_x0.copy(),
        "pose_x1": pose_x1.copy(),
        "pose_x1_esperada": pose_esperada,
        "measurement": medicion.copy(),
        "prediction": prediccion,
        "residual": residuo,
        "visual_error": error_visual,
        "translation_error": calcular_error_traslacional(error_visual),
        "angular_error": calcular_error_angular(error_visual),
        "covariance": covarianza.copy(),
        "information": informacion.copy(),
        "sigmas": sigmas.copy(),
        "contributions": contribuciones,
        "unweighted_error": calcular_error_cuadratico(residuo),
        "weighted_error": calcular_error_ponderado(residuo, informacion),
    }


def interpolar_pose(pose_inicial, pose_final, alpha):
    """Interpola posición y orientación por el camino angular más corto."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha debe pertenecer al intervalo [0, 1].")

    pose_inicial = np.asarray(pose_inicial, dtype=float)
    pose_final = np.asarray(pose_final, dtype=float)

    delta_theta = normalizar_angulo(pose_final[2] - pose_inicial[2])

    return np.array(
        [
            (1.0 - alpha) * pose_inicial[0] + alpha * pose_final[0],
            (1.0 - alpha) * pose_inicial[1] + alpha * pose_final[1],
            normalizar_angulo(pose_inicial[2] + alpha * delta_theta),
        ],
        dtype=float,
    )


def crear_estimaciones_comparacion(pose_inicial, pose_esperada):
    """Crea cuatro estimaciones para comparar predicción y residuo."""

    alphas = (0.0, 0.38, 0.72, 1.0)
    labels = (
        "A · estimación inicial",
        "B · estimación intermedia",
        "C · estimación cercana",
        "D · estimación compatible",
    )

    return [
        {
            "label": label,
            "alpha": alpha,
            "pose": interpolar_pose(pose_inicial, pose_esperada, alpha),
        }
        for label, alpha in zip(labels, alphas)
    ]


# ---------------------------------------------------------------------------
# Estados de la animación
# ---------------------------------------------------------------------------


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
    focus=None,
    experiment_label=None,
    correction_alpha=0.0,
    show_geometry=False,
    show_x0=False,
    show_x1=False,
    show_measurement=False,
    show_local_frame=False,
    show_expected=False,
    show_model=False,
    show_prediction=False,
    show_comparison=False,
    show_translation_error=False,
    show_angular_error=False,
    show_angle_wrap=False,
    show_residual=False,
    show_uncertainty=False,
    show_information=False,
    show_cost=False,
    show_initial_history=False,
    show_future_graph=False,
):
    """Convierte una evaluación en un fotograma completamente independiente."""

    return {
        "phase": phase,
        "message": message,
        "step": int(step),
        "total_steps": int(total_steps),
        "focus": focus,
        "experiment_label": experiment_label,
        "correction_alpha": float(correction_alpha),
        "measurement_is_fixed": True,
        "pose_x0": _serializar_vector(evaluation["pose_x0"]),
        "pose_x1": _serializar_vector(evaluation["pose_x1"]),
        "pose_x1_initial": _serializar_vector(initial_evaluation["pose_x1"]),
        "pose_x1_expected": _serializar_vector(evaluation["pose_x1_esperada"]),
        "measurement": _serializar_vector(evaluation["measurement"]),
        "prediction": _serializar_vector(evaluation["prediction"]),
        "residual": _serializar_vector(evaluation["residual"]),
        "visual_error": _serializar_vector(evaluation["visual_error"]),
        "translation_error": float(evaluation["translation_error"]),
        "angular_error": float(evaluation["angular_error"]),
        "covariance": _serializar_matriz(evaluation["covariance"]),
        "information": _serializar_matriz(evaluation["information"]),
        "sigmas": _serializar_vector(evaluation["sigmas"]),
        "contributions": _serializar_vector(evaluation["contributions"]),
        "unweighted_error": float(evaluation["unweighted_error"]),
        "weighted_error": float(evaluation["weighted_error"]),
        "initial_prediction": _serializar_vector(initial_evaluation["prediction"]),
        "initial_residual": _serializar_vector(initial_evaluation["residual"]),
        "initial_visual_error": _serializar_vector(initial_evaluation["visual_error"]),
        "initial_translation_error": float(initial_evaluation["translation_error"]),
        "initial_angular_error": float(initial_evaluation["angular_error"]),
        "initial_weighted_error": float(initial_evaluation["weighted_error"]),
        "final_prediction": _serializar_vector(final_evaluation["prediction"]),
        "final_residual": _serializar_vector(final_evaluation["residual"]),
        "final_weighted_error": float(final_evaluation["weighted_error"]),
        "show_geometry": bool(show_geometry),
        "show_x0": bool(show_x0),
        "show_x1": bool(show_x1),
        "show_measurement": bool(show_measurement),
        "show_local_frame": bool(show_local_frame),
        "show_expected": bool(show_expected),
        "show_model": bool(show_model),
        "show_prediction": bool(show_prediction),
        "show_comparison": bool(show_comparison),
        "show_translation_error": bool(show_translation_error),
        "show_angular_error": bool(show_angular_error),
        "show_angle_wrap": bool(show_angle_wrap),
        "show_residual": bool(show_residual),
        "show_uncertainty": bool(show_uncertainty),
        "show_information": bool(show_information),
        "show_cost": bool(show_cost),
        "show_initial_history": bool(show_initial_history),
        "show_future_graph": bool(show_future_graph),
    }


def crear_estados_animacion(graph, correction_steps=18):
    """Construye la narración completa del apartado 5.2."""

    if correction_steps < 8:
        raise ValueError("Se requieren al menos ocho pasos de corrección.")

    initial_evaluation = evaluar_modelo_medicion(graph, POSE_X1_INICIAL)
    expected_pose = initial_evaluation["pose_x1_esperada"]
    final_evaluation = evaluar_modelo_medicion(graph, expected_pose)

    states = []

    def add(evaluation, phase, message, **kwargs):
        states.append(
            crear_estado_animacion(
                evaluation=evaluation,
                phase=phase,
                message=message,
                step=len(states) + 1,
                total_steps=0,
                initial_evaluation=initial_evaluation,
                final_evaluation=final_evaluation,
                **kwargs,
            )
        )

    # 1. Variables estimadas.
    add(
        initial_evaluation,
        "variables",
        "x0 y x1 son variables estimadas: valores que mantiene el algoritmo.",
        focus="variables",
        show_geometry=True,
        show_x0=True,
        show_x1=True,
    )
    add(
        initial_evaluation,
        "variables",
        "El optimizador puede modificar x1; x0 permanece fijada por un prior.",
        focus="variables",
        show_geometry=True,
        show_x0=True,
        show_x1=True,
    )

    # 2. Estado verdadero frente a estimación.
    add(
        initial_evaluation,
        "estimate_vs_truth",
        "El estado físico verdadero normalmente es desconocido; trabajamos con estimaciones.",
        focus="variables",
        show_geometry=True,
        show_x0=True,
        show_x1=True,
    )

    # 3. Medición fija del sensor.
    for message in (
        "El sensor aporta z01: una medición relativa entre x0 y x1.",
        "z01 está expresada en el sistema local de x0, no en coordenadas globales.",
        "La medición queda registrada y no cambia durante la optimización.",
    ):
        add(
            initial_evaluation,
            "measurement",
            message,
            focus="measurement",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
        )

    # 4. Sistema local de x0.
    for message in (
        "Los 3.0 m y 0.4 m de z01 se interpretan sobre los ejes locales de x0.",
        "Como x0 está girada 25°, el desplazamiento relativo también debe rotarse.",
        "Componer x0 ⊕ z01 transforma la medición local al sistema global.",
    ):
        add(
            initial_evaluation,
            "local_frame",
            message,
            focus="local_frame",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_local_frame=True,
        )

    # 5. Pose esperada.
    for message in (
        "La medición produce una pose global esperada: x1* = x0 ⊕ z01.",
        "z01 es una transformación relativa; x1* es una pose global.",
    ):
        add(
            initial_evaluation,
            "expected_pose",
            message,
            focus="expected",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_local_frame=True,
            show_expected=True,
        )

    # 6. Función de predicción.
    for message in (
        "El modelo h(x0, x1) calcula qué debería haber medido el sensor.",
        "Para poses: z_hat01 = x0^-1 ⊕ x1.",
    ):
        add(
            initial_evaluation,
            "model",
            message,
            focus="model",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_model=True,
        )

    # 7. Predicción de las variables actuales.
    for message in (
        "La predicción se calcula desde las estimaciones actuales, no desde el sensor.",
        "La flecha morada representa z_hat01 y termina en la estimación x1.",
        "Si x1 cambia, la predicción cambia inmediatamente.",
    ):
        add(
            initial_evaluation,
            "prediction",
            message,
            focus="prediction",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_model=True,
            show_prediction=True,
        )

    # 8. Comparación directa.
    for message in (
        "Verde: dato fijo del sensor. Morado: valor calculado por el modelo.",
        "Como z01 y z_hat01 no coinciden, aparece un residuo.",
    ):
        add(
            initial_evaluation,
            "comparison",
            message,
            focus="comparison",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_model=True,
            show_prediction=True,
            show_comparison=True,
        )

    # 9. Error visual de traslación.
    for message in (
        "La flecha roja une la pose esperada con la estimación actual.",
        "Su longitud resume el error visual de posición en coordenadas globales.",
    ):
        add(
            initial_evaluation,
            "translation_error",
            message,
            focus="translation_error",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_model=True,
            show_prediction=True,
            show_comparison=True,
            show_translation_error=True,
        )

    # 10. Error angular.
    for message in (
        "El arco rojo compara la orientación esperada con la orientación estimada.",
        "La diferencia angular siempre debe normalizarse.",
    ):
        add(
            initial_evaluation,
            "angular_error",
            message,
            focus="angular_error",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_model=True,
            show_prediction=True,
            show_comparison=True,
            show_translation_error=True,
            show_angular_error=True,
        )

    # 11. Ejemplo breve de wrap angular.
    for message in (
        "179° y -179° están separados por 2°, no por 358°.",
        "wrap(358°) = -2° evita una penalización angular incorrecta.",
    ):
        add(
            initial_evaluation,
            "angle_wrap",
            message,
            focus="angle_wrap",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_prediction=True,
            show_translation_error=True,
            show_angular_error=True,
            show_angle_wrap=True,
        )

    # 12. Residuo matemático.
    for message in (
        "El residuo SE(2) es e01 = z01^-1 ⊕ z_hat01.",
        "Sus componentes se expresan en el marco geométrico de la restricción.",
    ):
        add(
            initial_evaluation,
            "residual",
            message,
            focus="residual",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_model=True,
            show_prediction=True,
            show_translation_error=True,
            show_angular_error=True,
            show_residual=True,
        )

    # 13. Incertidumbre.
    for message in (
        "La elipse representa la incertidumbre atribuida a la medición.",
        "La incertidumbre no es el residuo actual: describe la precisión esperada.",
    ):
        add(
            initial_evaluation,
            "uncertainty",
            message,
            focus="uncertainty",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_prediction=True,
            show_translation_error=True,
            show_angular_error=True,
            show_residual=True,
            show_uncertainty=True,
        )

    # 14. Información y coste.
    for message in (
        "Ω01 = Σ01^-1 convierte incertidumbre en peso matemático.",
        "El coste E01 = e01ᵀ Ω01 e01 penaliza el residuo según la precisión.",
    ):
        add(
            initial_evaluation,
            "cost",
            message,
            focus="cost",
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_model=True,
            show_prediction=True,
            show_translation_error=True,
            show_angular_error=True,
            show_residual=True,
            show_uncertainty=True,
            show_information=True,
            show_cost=True,
        )

    # 15. Cuatro estimaciones comparadas.
    for item in crear_estimaciones_comparacion(POSE_X1_INICIAL, expected_pose):
        evaluation = evaluar_modelo_medicion(graph, item["pose"])
        add(
            evaluation,
            "estimation_experiment",
            (
                f"{item['label']}: z01 permanece fija; "
                "z_hat01, e01 y E01 se vuelven a calcular."
            ),
            focus="experiment",
            experiment_label=item["label"],
            correction_alpha=item["alpha"],
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_model=True,
            show_prediction=True,
            show_comparison=True,
            show_translation_error=True,
            show_angular_error=True,
            show_residual=True,
            show_uncertainty=True,
            show_information=True,
            show_cost=True,
            show_initial_history=item["alpha"] > 0.0,
        )

    # 16. Corrección continua de la variable x1.
    for index in range(1, correction_steps + 1):
        alpha = index / correction_steps
        current_pose = interpolar_pose(POSE_X1_INICIAL, expected_pose, alpha)
        evaluation = evaluar_modelo_medicion(graph, current_pose)

        add(
            evaluation,
            "correction",
            (
                f"Cambio de la variable x1: paso {index}/{correction_steps}. "
                "La medición sigue fija y el coste disminuye."
            ),
            focus="correction",
            experiment_label=f"paso {index}/{correction_steps}",
            correction_alpha=alpha,
            show_geometry=True,
            show_x0=True,
            show_x1=True,
            show_measurement=True,
            show_expected=True,
            show_model=True,
            show_prediction=True,
            show_comparison=True,
            show_translation_error=True,
            show_angular_error=True,
            show_residual=True,
            show_uncertainty=True,
            show_information=True,
            show_cost=True,
            show_initial_history=True,
        )

    # 17. Estado compatible.
    add(
        final_evaluation,
        "compatible",
        "Cuando x1 = x0 ⊕ z01, medición y predicción se superponen.",
        focus="compatible",
        experiment_label="estimación compatible",
        correction_alpha=1.0,
        show_geometry=True,
        show_x0=True,
        show_x1=True,
        show_measurement=True,
        show_expected=True,
        show_model=True,
        show_prediction=True,
        show_comparison=True,
        show_residual=True,
        show_uncertainty=True,
        show_information=True,
        show_cost=True,
        show_initial_history=True,
    )
    add(
        final_evaluation,
        "compatible",
        "El residuo y el coste son aproximadamente cero para esta medición.",
        focus="compatible",
        experiment_label="estimación compatible",
        correction_alpha=1.0,
        show_geometry=True,
        show_x0=True,
        show_x1=True,
        show_measurement=True,
        show_expected=True,
        show_model=True,
        show_prediction=True,
        show_comparison=True,
        show_residual=True,
        show_uncertainty=True,
        show_information=True,
        show_cost=True,
        show_initial_history=True,
    )

    # 18. Transición a muchas mediciones.
    add(
        final_evaluation,
        "future_graph",
        "En Pose Graph SLAM, cada arista produce su propia predicción y residuo.",
        focus="future_graph",
        correction_alpha=1.0,
        show_geometry=True,
        show_x0=True,
        show_x1=True,
        show_measurement=True,
        show_expected=True,
        show_prediction=True,
        show_residual=True,
        show_uncertainty=True,
        show_information=True,
        show_cost=True,
        show_initial_history=True,
        show_future_graph=True,
    )
    add(
        final_evaluation,
        "future_graph",
        "La función de coste global suma los errores ponderados de todas las mediciones.",
        focus="future_graph",
        correction_alpha=1.0,
        show_geometry=True,
        show_x0=True,
        show_x1=True,
        show_measurement=True,
        show_expected=True,
        show_prediction=True,
        show_residual=True,
        show_uncertainty=True,
        show_information=True,
        show_cost=True,
        show_initial_history=True,
        show_future_graph=True,
    )

    # 19. Resumen final, utilizado también para la imagen estática.
    add(
        final_evaluation,
        "summary",
        "Sensor → medición; variables + modelo → predicción; comparación → residuo.",
        focus="summary",
        experiment_label="resumen inicial → final",
        correction_alpha=1.0,
        show_geometry=True,
        show_x0=True,
        show_x1=True,
        show_measurement=True,
        show_local_frame=True,
        show_expected=True,
        show_model=True,
        show_prediction=True,
        show_comparison=True,
        show_residual=True,
        show_uncertainty=True,
        show_information=True,
        show_cost=True,
        show_initial_history=True,
        show_future_graph=True,
    )

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
    """Comprueba composición, inversión, marcos locales y wrap angular."""

    identity = np.zeros(3, dtype=float)
    test_pose = np.array([2.0, -1.0, radians(37.0)], dtype=float)

    if not np.allclose(componer_poses_se2(identity, test_pose), test_pose):
        raise ValueError("La identidad por la izquierda no se cumple.")

    if not np.allclose(componer_poses_se2(test_pose, identity), test_pose):
        raise ValueError("La identidad por la derecha no se cumple.")

    inverse_composition = componer_poses_se2(
        test_pose,
        invertir_pose_se2(test_pose),
    )

    if not np.allclose(inverse_composition, identity, atol=1e-10):
        raise ValueError("La pose compuesta con su inversa no da la identidad.")

    if not np.isclose(degrees(normalizar_angulo(radians(358.0))), -2.0):
        raise ValueError("La normalización de 358° debe producir -2°.")

    recovered = matriz_a_pose_se2(pose_a_matriz_se2(test_pose))

    if not np.allclose(recovered, test_pose, atol=1e-10):
        raise ValueError("La conversión pose↔matriz no es reversible.")


def validar_grafo(graph):
    """Valida variables, medición, marco, covarianza e información."""

    if not isinstance(graph, nx.DiGraph):
        raise TypeError("El ejemplo debe utilizar un nx.DiGraph.")

    if set(graph.nodes()) != {"x0", "x1"}:
        raise ValueError("El grafo debe contener exactamente x0 y x1.")

    if not graph.has_edge("x0", "x1"):
        raise ValueError("Debe existir la medición dirigida x0→x1.")

    if not graph.nodes["x0"].get("fixed", False):
        raise ValueError("x0 debe estar fijada por un prior.")

    if graph.nodes["x1"].get("fixed", False):
        raise ValueError("x1 debe ser una variable modificable.")

    edge = graph.edges["x0", "x1"]

    required = {
        "measurement",
        "frame",
        "sensor",
        "sigmas",
        "covariance",
        "information",
        "immutable_measurement",
    }
    missing = required.difference(edge)

    if missing:
        raise ValueError(f"Faltan atributos de la medición: {sorted(missing)}")

    if edge["frame"] != "x0":
        raise ValueError("La medición debe estar expresada en el marco x0.")

    if not edge["immutable_measurement"]:
        raise ValueError("La medición debe permanecer fija durante el ejemplo.")

    covariance = np.asarray(edge["covariance"], dtype=float)
    information = np.asarray(edge["information"], dtype=float)

    if not np.allclose(covariance @ information, np.eye(3), atol=1e-10):
        raise ValueError("ΣΩ debe ser aproximadamente la identidad.")


def validar_resultados(graph, result):
    """Comprueba la diferencia conceptual y la reducción del coste."""

    initial = result["initial"]
    final = result["final"]
    expected = result["expected_pose"]
    edge = graph.edges["x0", "x1"]

    measurement_before = np.asarray(edge["measurement"], dtype=float).copy()

    if initial["weighted_error"] <= 0.0:
        raise ValueError("El coste inicial debe ser positivo.")

    if initial["translation_error"] <= 0.0:
        raise ValueError("Debe existir un error traslacional visible.")

    if initial["angular_error"] <= 0.0:
        raise ValueError("Debe existir un error angular visible.")

    if np.allclose(initial["measurement"], initial["prediction"]):
        raise ValueError("Medición y predicción iniciales deben ser distintas.")

    if not np.allclose(final["measurement"], final["prediction"], atol=1e-10):
        raise ValueError("Medición y predicción finales deben coincidir.")

    if not np.allclose(final["residual"], np.zeros(3), atol=1e-10):
        raise ValueError("El residuo final debe ser aproximadamente cero.")

    if final["weighted_error"] > 1e-16:
        raise ValueError("El coste final debe ser aproximadamente cero.")

    if initial["weighted_error"] <= final["weighted_error"]:
        raise ValueError("La corrección debe reducir estrictamente el coste.")

    if not np.isclose(
        np.sum(initial["contributions"]),
        initial["weighted_error"],
        atol=1e-10,
    ):
        raise ValueError("Las contribuciones no suman el coste ponderado.")

    if len(result["states"]) < 50:
        raise ValueError("La animación debe contener al menos 50 estados.")

    measurement_after = np.asarray(edge["measurement"], dtype=float)

    if not np.array_equal(measurement_before, measurement_after):
        raise ValueError("La medición cambió durante la generación de estados.")

    graph.nodes["x1"]["initial_estimate"] = POSE_X1_INICIAL.copy()
    graph.nodes["x1"]["estimate"] = expected.copy()
    graph.nodes["x1"]["final_estimate"] = expected.copy()

    edge["initial_prediction"] = initial["prediction"].copy()
    edge["initial_residual"] = initial["residual"].copy()
    edge["initial_weighted_error"] = initial["weighted_error"]
    edge["final_prediction"] = final["prediction"].copy()
    edge["final_residual"] = final["residual"].copy()
    edge["final_weighted_error"] = final["weighted_error"]


def _format_pose(pose):
    pose = np.asarray(pose, dtype=float)
    return (
        f"({pose[0]:.3f} m, {pose[1]:.3f} m, "
        f"{degrees(pose[2]):.3f}°)"
    )


def imprimir_resumen(graph, result):
    """Imprime los valores numéricos relevantes de forma determinista."""

    initial = result["initial"]
    final = result["final"]
    edge = graph.edges["x0", "x1"]

    print("\n=== Variables, medición, predicción y error ===")
    print(f"Variable fija x0: {_format_pose(initial['pose_x0'])}")
    print(f"Variable inicial x1: {_format_pose(initial['pose_x1'])}")
    print(f"Medición fija z01: {_format_pose(initial['measurement'])}")
    print(f"Pose esperada x1*: {_format_pose(result['expected_pose'])}")
    print(f"Predicción inicial: {_format_pose(initial['prediction'])}")
    print(f"Residuo inicial: {_format_pose(initial['residual'])}")
    print(f"Error traslacional visual: {initial['translation_error']:.6f} m")
    print(f"Error angular visual: {degrees(initial['angular_error']):.6f}°")
    print(f"Coste inicial: {initial['weighted_error']:.6f}")
    print(f"Coste final: {final['weighted_error']:.12f}")
    print("Contribuciones iniciales:", initial["contributions"])
    print("Sensor:", edge["sensor"])
    print("Marco de la medición:", edge["frame"])
    print(f"Estados de animación: {len(result['states'])}")


def main():
    validar_transformaciones()

    graph = crear_grafo_medicion_prediccion()
    validar_grafo(graph)

    result = crear_estados_animacion(graph, correction_steps=18)
    validar_resultados(graph, result)
    validar_grafo(graph)

    imprimir_resumen(graph, result)

    animator = GraphAnimator(
        figsize=(18, 10),
        interval=560,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "05_optimizacion"
        / "02_medicion_prediccion_error.png"
    )

    animator.animate_measurement_prediction_error(
        graph=graph,
        states=result["states"],
        title="Variables, medición, predicción y error",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
