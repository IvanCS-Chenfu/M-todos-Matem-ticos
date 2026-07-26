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

POSE_REAL = np.array([0.90, 1.20, np.deg2rad(25.0)], dtype=float)
POSE_ESTIMADA_INICIAL = np.array([1.18, 1.00, np.deg2rad(32.0)], dtype=float)
LANDMARK_REAL = np.array([5.00, 4.20], dtype=float)
LANDMARK_ESTIMADO_INICIAL = np.array([4.72, 4.48], dtype=float)

# Transformación rígida desde el marco del robot al marco del sensor.
EXTRINSECA_SENSOR_REAL = np.array(
    [0.28, 0.06, np.deg2rad(5.0)],
    dtype=float,
)
EXTRINSECA_SENSOR_ERRONEA = np.array(
    [0.36, -0.02, np.deg2rad(8.0)],
    dtype=float,
)

RUIDO_MEDICION = np.array(
    [-0.065, np.deg2rad(1.25)],
    dtype=float,
)
SIGMAS_OBSERVACION = np.array(
    [0.12, np.deg2rad(2.20)],
    dtype=float,
)

DELTA_HUBER = 2.50
EPSILON_JACOBIANO = 1e-7
LAMBDA_INICIAL = 1e-3
MAX_ITERACIONES = 30
TOLERANCIA_INCREMENTO = 1e-11
TOLERANCIA_COSTE = 1e-13


# ---------------------------------------------------------------------------
# Validación y geometría básica
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
        raise ValueError(f"{nombre} debe contener tres componentes.")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{nombre} debe contener valores finitos.")

    resultado = pose.copy()
    resultado[2] = normalizar_angulo(resultado[2])
    return resultado


def validar_landmark(landmark, nombre="landmark"):
    """Valida un landmark puntual 2D."""

    landmark = np.asarray(landmark, dtype=float)
    if landmark.shape != (2,):
        raise ValueError(f"{nombre} debe contener dos componentes.")
    if not np.all(np.isfinite(landmark)):
        raise ValueError(f"{nombre} debe contener valores finitos.")
    return landmark.copy()


def validar_medicion_rango_rumbo(medicion, nombre="medición"):
    """Valida una observación (rango, rumbo)."""

    medicion = np.asarray(medicion, dtype=float)
    if medicion.shape != (2,):
        raise ValueError(f"{nombre} debe contener rango y rumbo.")
    if not np.all(np.isfinite(medicion)):
        raise ValueError(f"{nombre} debe contener valores finitos.")
    if medicion[0] <= 0.0:
        raise ValueError(f"El rango de {nombre} debe ser positivo.")

    resultado = medicion.copy()
    resultado[1] = normalizar_angulo(resultado[1])
    return resultado


def rotacion_2d(theta):
    """Crea una matriz de rotación plana."""

    theta = normalizar_angulo(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


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
    """Convierte una matriz homogénea de SE(2) en una pose."""

    matriz = np.asarray(matriz, dtype=float)
    if matriz.shape != (3, 3):
        raise ValueError("La matriz de SE(2) debe tener forma 3x3.")
    if not np.all(np.isfinite(matriz)):
        raise ValueError("La matriz debe contener valores finitos.")
    if not np.allclose(matriz[2], [0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("La última fila no es homogénea.")

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
    """Calcula pose_a ⊕ pose_b."""

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


def transformar_punto_global_a_local(pose_marco, punto_global):
    """Expresa un punto global en el sistema local de una pose."""

    pose_marco = validar_pose(pose_marco, "pose del marco")
    punto_global = validar_landmark(punto_global, "punto global")
    return rotacion_2d(pose_marco[2]).T @ (punto_global - pose_marco[:2])


def transformar_punto_local_a_global(pose_marco, punto_local):
    """Expresa un punto local en el sistema global."""

    pose_marco = validar_pose(pose_marco, "pose del marco")
    punto_local = validar_landmark(punto_local, "punto local")
    return pose_marco[:2] + rotacion_2d(pose_marco[2]) @ punto_local


def calcular_pose_sensor(pose_robot, extrinseca_sensor=EXTRINSECA_SENSOR_REAL):
    """Calcula la pose global del sensor T_WS = T_WR T_RS."""

    return componer_poses_se2(
        validar_pose(pose_robot, "pose del robot"),
        validar_pose(extrinseca_sensor, "extrínseca del sensor"),
    )


def rango_rumbo_a_cartesiano(medicion):
    """Convierte (rango, rumbo) en coordenadas cartesianas locales."""

    rango, rumbo = validar_medicion_rango_rumbo(medicion)
    return np.array(
        [rango * np.cos(rumbo), rango * np.sin(rumbo)],
        dtype=float,
    )


def cartesiano_a_rango_rumbo(vector_local):
    """Convierte un vector local 2D en (rango, rumbo)."""

    vector_local = validar_landmark(vector_local, "vector local")
    rango = float(np.linalg.norm(vector_local))
    if rango <= 1e-12:
        raise ValueError("El rumbo no está definido para rango cero.")
    rumbo = normalizar_angulo(np.arctan2(vector_local[1], vector_local[0]))
    return np.array([rango, rumbo], dtype=float)


# ---------------------------------------------------------------------------
# Modelo de observación pose-landmark
# ---------------------------------------------------------------------------


def calcular_delta_global(pose_sensor, landmark):
    """Calcula la diferencia global entre el sensor y el landmark."""

    pose_sensor = validar_pose(pose_sensor, "pose del sensor")
    landmark = validar_landmark(landmark)
    return landmark - pose_sensor[:2]


def predecir_observacion_cartesiana(
    pose_robot,
    landmark,
    extrinseca_sensor=EXTRINSECA_SENSOR_REAL,
):
    """Predice la posición local cartesiana del landmark."""

    pose_sensor = calcular_pose_sensor(pose_robot, extrinseca_sensor)
    return transformar_punto_global_a_local(pose_sensor, landmark)


def predecir_observacion_rango_rumbo(
    pose_robot,
    landmark,
    extrinseca_sensor=EXTRINSECA_SENSOR_REAL,
):
    """Predice rango y rumbo del landmark desde el sensor."""

    local = predecir_observacion_cartesiana(
        pose_robot,
        landmark,
        extrinseca_sensor,
    )
    return cartesiano_a_rango_rumbo(local)


def crear_medicion_rango_rumbo():
    """Crea una medición fija desde la geometría real y ruido determinista."""

    ideal = predecir_observacion_rango_rumbo(
        POSE_REAL,
        LANDMARK_REAL,
        EXTRINSECA_SENSOR_REAL,
    )
    medida = ideal + RUIDO_MEDICION
    medida[1] = normalizar_angulo(medida[1])
    return {
        "ideal": validar_medicion_rango_rumbo(ideal, "medición ideal"),
        "measured": validar_medicion_rango_rumbo(medida, "medición medida"),
        "noise": RUIDO_MEDICION.copy(),
    }


def calcular_residuo_rango_rumbo(medicion, prediccion):
    """Calcula e = predicción - medición con rumbo normalizado."""

    medicion = validar_medicion_rango_rumbo(medicion, "medición")
    prediccion = validar_medicion_rango_rumbo(prediccion, "predicción")
    return np.array(
        [
            prediccion[0] - medicion[0],
            normalizar_angulo(prediccion[1] - medicion[1]),
        ],
        dtype=float,
    )


def calcular_residuo_cartesiano(medicion, prediccion):
    """Compara los vectores cartesianos de medida y predicción."""

    medicion_cartesiana = rango_rumbo_a_cartesiano(medicion)
    prediccion_cartesiana = rango_rumbo_a_cartesiano(prediccion)
    return prediccion_cartesiana - medicion_cartesiana


def evaluar_observacion(
    pose,
    landmark,
    medicion,
    extrinseca_sensor=EXTRINSECA_SENSOR_REAL,
):
    """Calcula geometría, predicción y residuos de una observación."""

    pose = validar_pose(pose)
    landmark = validar_landmark(landmark)
    medicion = validar_medicion_rango_rumbo(medicion)
    pose_sensor = calcular_pose_sensor(pose, extrinseca_sensor)
    delta_global = calcular_delta_global(pose_sensor, landmark)
    prediccion_cartesiana = transformar_punto_global_a_local(
        pose_sensor,
        landmark,
    )
    prediccion = cartesiano_a_rango_rumbo(prediccion_cartesiana)
    residuo = calcular_residuo_rango_rumbo(medicion, prediccion)
    residuo_cartesiano = calcular_residuo_cartesiano(medicion, prediccion)
    extremo_medido_global = transformar_punto_local_a_global(
        pose_sensor,
        rango_rumbo_a_cartesiano(medicion),
    )

    return {
        "pose": pose.copy(),
        "sensor_pose": pose_sensor,
        "landmark": landmark.copy(),
        "measurement": medicion.copy(),
        "prediction": prediccion,
        "delta_global": delta_global,
        "measurement_cartesian": rango_rumbo_a_cartesiano(medicion),
        "prediction_cartesian": prediccion_cartesiana,
        "measurement_endpoint_global": extremo_medido_global,
        "residual": residuo,
        "raw_angular_error": float(prediccion[1] - medicion[1]),
        "range_error": float(residuo[0]),
        "bearing_error": float(residuo[1]),
        "cartesian_residual": residuo_cartesiano,
        "cartesian_residual_norm": float(np.linalg.norm(residuo_cartesiano)),
    }


# ---------------------------------------------------------------------------
# Incertidumbre, información y robustez
# ---------------------------------------------------------------------------


def crear_covarianza_observacion(sigmas=SIGMAS_OBSERVACION):
    """Crea la covarianza diagonal de rango y rumbo."""

    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.shape != (2,):
        raise ValueError("Se esperan dos desviaciones estándar.")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("Los sigmas deben ser positivos y finitos.")
    return np.diag(sigmas**2)


def calcular_matriz_informacion(covarianza):
    """Valida e invierte una covarianza definida positiva."""

    covarianza = np.asarray(covarianza, dtype=float)
    if covarianza.shape != (2, 2):
        raise ValueError("La covarianza debe tener forma 2x2.")
    if not np.all(np.isfinite(covarianza)):
        raise ValueError("La covarianza debe contener valores finitos.")
    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise ValueError("La covarianza debe ser simétrica.")
    if np.min(np.linalg.eigvalsh(covarianza)) <= 0.0:
        raise ValueError("La covarianza debe ser definida positiva.")
    return np.linalg.inv(covarianza)


def calcular_contribuciones_mahalanobis(residuo, informacion):
    """Calcula la contribución de rango y rumbo para Ω diagonal."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)
    if residuo.shape != (2,) or informacion.shape != (2, 2):
        raise ValueError("Dimensiones incompatibles.")
    if not np.allclose(informacion, np.diag(np.diag(informacion)), atol=1e-12):
        raise ValueError("La descomposición requiere información diagonal.")
    return residuo**2 * np.diag(informacion)


def calcular_error_mahalanobis(residuo, informacion):
    """Calcula s = eᵀΩe."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)
    if residuo.shape != (2,) or informacion.shape != (2, 2):
        raise ValueError("Dimensiones incompatibles.")
    return float(residuo.T @ informacion @ residuo)


def calcular_coste_cuadratico(residuo, informacion):
    """Calcula F = 1/2 eᵀΩe."""

    return 0.5 * calcular_error_mahalanobis(residuo, informacion)


def calcular_peso_huber(residuo, informacion, delta=DELTA_HUBER):
    """Calcula el peso IRLS de Huber sobre la norma de Mahalanobis."""

    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("El umbral de Huber debe ser positivo.")
    raiz = np.sqrt(max(calcular_error_mahalanobis(residuo, informacion), 0.0))
    if raiz <= delta or raiz <= 1e-15:
        return 1.0
    return float(delta / raiz)


def calcular_coste_huber(residuo, informacion, delta=DELTA_HUBER):
    """Calcula la pérdida de Huber para el residuo ponderado."""

    delta = float(delta)
    raiz = np.sqrt(max(calcular_error_mahalanobis(residuo, informacion), 0.0))
    if raiz <= delta:
        return 0.5 * raiz**2
    return float(delta * raiz - 0.5 * delta**2)


def completar_costes(evaluacion, informacion):
    """Añade métricas ponderadas a una evaluación geométrica."""

    resultado = dict(evaluacion)
    residuo = resultado["residual"]
    resultado["information"] = np.asarray(informacion, dtype=float).copy()
    resultado["mahalanobis"] = calcular_error_mahalanobis(
        residuo,
        informacion,
    )
    resultado["quadratic_cost"] = calcular_coste_cuadratico(
        residuo,
        informacion,
    )
    resultado["huber_weight"] = calcular_peso_huber(
        residuo,
        informacion,
    )
    resultado["huber_cost"] = calcular_coste_huber(
        residuo,
        informacion,
    )
    resultado["contributions"] = calcular_contribuciones_mahalanobis(
        residuo,
        informacion,
    )
    return resultado


# ---------------------------------------------------------------------------
# Jacobianos, linealización y observabilidad local
# ---------------------------------------------------------------------------


def calcular_jacobiano_pose_analitico(
    pose,
    landmark,
    extrinseca_sensor=EXTRINSECA_SENSOR_REAL,
):
    """Calcula ∂h/∂pose para una observación rango-rumbo."""

    pose = validar_pose(pose)
    landmark = validar_landmark(landmark)
    extrinseca_sensor = validar_pose(extrinseca_sensor, "extrínseca")

    theta = pose[2]
    tx, ty, phi = extrinseca_sensor
    c = np.cos(theta)
    s = np.sin(theta)

    sensor_x = pose[0] + c * tx - s * ty
    sensor_y = pose[1] + s * tx + c * ty
    sensor_theta = normalizar_angulo(theta + phi)

    dx = landmark[0] - sensor_x
    dy = landmark[1] - sensor_y
    q = dx**2 + dy**2
    rango = np.sqrt(q)
    if rango <= 1e-12:
        raise ValueError("El jacobiano no está definido para rango cero.")

    dsx_dtheta = -s * tx - c * ty
    dsy_dtheta = c * tx - s * ty
    ddx_dtheta = -dsx_dtheta
    ddy_dtheta = -dsy_dtheta

    dr_dtheta = (dx * ddx_dtheta + dy * ddy_dtheta) / rango
    db_dtheta = (
        -dy * ddx_dtheta + dx * ddy_dtheta
    ) / q - 1.0

    return np.array(
        [
            [-dx / rango, -dy / rango, dr_dtheta],
            [dy / q, -dx / q, db_dtheta],
        ],
        dtype=float,
    )


def calcular_jacobiano_landmark_analitico(
    pose,
    landmark,
    extrinseca_sensor=EXTRINSECA_SENSOR_REAL,
):
    """Calcula ∂h/∂landmark para una observación rango-rumbo."""

    pose_sensor = calcular_pose_sensor(pose, extrinseca_sensor)
    dx, dy = calcular_delta_global(pose_sensor, landmark)
    q = dx**2 + dy**2
    rango = np.sqrt(q)
    if rango <= 1e-12:
        raise ValueError("El jacobiano no está definido para rango cero.")

    return np.array(
        [
            [dx / rango, dy / rango],
            [-dy / q, dx / q],
        ],
        dtype=float,
    )


def calcular_jacobianos_numericos(
    pose,
    landmark,
    medicion,
    extrinseca_sensor=EXTRINSECA_SENSOR_REAL,
    epsilon=EPSILON_JACOBIANO,
):
    """Calcula jacobianos del residuo mediante diferencias centrales."""

    pose = validar_pose(pose)
    landmark = validar_landmark(landmark)
    medicion = validar_medicion_rango_rumbo(medicion)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("El epsilon debe ser positivo.")

    jac_pose = np.zeros((2, 3), dtype=float)
    jac_landmark = np.zeros((2, 2), dtype=float)

    for columna in range(3):
        delta = np.zeros(3, dtype=float)
        delta[columna] = epsilon
        pose_mas = pose + delta
        pose_menos = pose - delta
        pose_mas[2] = normalizar_angulo(pose_mas[2])
        pose_menos[2] = normalizar_angulo(pose_menos[2])
        e_mas = calcular_residuo_rango_rumbo(
            medicion,
            predecir_observacion_rango_rumbo(
                pose_mas,
                landmark,
                extrinseca_sensor,
            ),
        )
        e_menos = calcular_residuo_rango_rumbo(
            medicion,
            predecir_observacion_rango_rumbo(
                pose_menos,
                landmark,
                extrinseca_sensor,
            ),
        )
        diferencia = e_mas - e_menos
        diferencia[1] = normalizar_angulo(diferencia[1])
        jac_pose[:, columna] = diferencia / (2.0 * epsilon)

    for columna in range(2):
        delta = np.zeros(2, dtype=float)
        delta[columna] = epsilon
        e_mas = calcular_residuo_rango_rumbo(
            medicion,
            predecir_observacion_rango_rumbo(
                pose,
                landmark + delta,
                extrinseca_sensor,
            ),
        )
        e_menos = calcular_residuo_rango_rumbo(
            medicion,
            predecir_observacion_rango_rumbo(
                pose,
                landmark - delta,
                extrinseca_sensor,
            ),
        )
        diferencia = e_mas - e_menos
        diferencia[1] = normalizar_angulo(diferencia[1])
        jac_landmark[:, columna] = diferencia / (2.0 * epsilon)

    return {
        "pose": jac_pose,
        "landmark": jac_landmark,
        "joint": np.hstack((jac_pose, jac_landmark)),
    }


def comparar_jacobianos(
    pose,
    landmark,
    medicion,
    extrinseca_sensor=EXTRINSECA_SENSOR_REAL,
):
    """Compara jacobianos analíticos y numéricos."""

    analitico_pose = calcular_jacobiano_pose_analitico(
        pose,
        landmark,
        extrinseca_sensor,
    )
    analitico_landmark = calcular_jacobiano_landmark_analitico(
        pose,
        landmark,
        extrinseca_sensor,
    )
    numericos = calcular_jacobianos_numericos(
        pose,
        landmark,
        medicion,
        extrinseca_sensor,
    )
    return {
        "analytic_pose": analitico_pose,
        "numeric_pose": numericos["pose"],
        "analytic_landmark": analitico_landmark,
        "numeric_landmark": numericos["landmark"],
        "analytic_joint": np.hstack((analitico_pose, analitico_landmark)),
        "numeric_joint": numericos["joint"],
        "pose_max_error": float(
            np.max(np.abs(analitico_pose - numericos["pose"]))
        ),
        "landmark_max_error": float(
            np.max(np.abs(analitico_landmark - numericos["landmark"]))
        ),
    }


def analizar_observabilidad_local(jacobiano_conjunto):
    """Calcula rango y nulidad para distintos conjuntos de variables."""

    jacobiano_conjunto = np.asarray(jacobiano_conjunto, dtype=float)
    if jacobiano_conjunto.ndim != 2:
        raise ValueError("El jacobiano debe ser una matriz.")
    rango = int(np.linalg.matrix_rank(jacobiano_conjunto, tol=1e-9))
    return {
        "shape": jacobiano_conjunto.shape,
        "rank": rango,
        "nullity": jacobiano_conjunto.shape[1] - rango,
    }


def comprobar_linealizacion(
    pose,
    landmark,
    medicion,
    jacobiano_conjunto,
    incremento,
):
    """Compara el residuo real perturbado con su aproximación lineal."""

    incremento = np.asarray(incremento, dtype=float)
    if incremento.shape != (5,):
        raise ValueError("El incremento conjunto debe tener cinco componentes.")

    evaluacion = evaluar_observacion(pose, landmark, medicion)
    prediccion_lineal = evaluacion["residual"] + jacobiano_conjunto @ incremento

    pose_nueva = validar_pose(pose + incremento[:3])
    landmark_nuevo = validar_landmark(landmark + incremento[3:])
    residual_real = evaluar_observacion(
        pose_nueva,
        landmark_nuevo,
        medicion,
    )["residual"]
    diferencia = residual_real - prediccion_lineal
    diferencia[1] = normalizar_angulo(diferencia[1])

    return {
        "increment": incremento.copy(),
        "linear_residual": prediccion_lineal,
        "true_residual": residual_real,
        "difference": diferencia,
        "error_norm": float(np.linalg.norm(diferencia)),
    }


# ---------------------------------------------------------------------------
# Grafo de la observación y optimización del landmark con pose fija
# ---------------------------------------------------------------------------


def crear_grafo_error_pose_landmark(medicion, covarianza, informacion):
    """Crea un grafo mínimo con una pose, un landmark y una observación."""

    medicion = validar_medicion_rango_rumbo(medicion)
    graph = nx.Graph()
    graph.graph.update(
        {
            "name": "Error de observación pose-landmark",
            "convention": "residual = prediction - measurement",
            "sensor_model": "range_bearing_2d",
            "extrinsic": EXTRINSECA_SENSOR_REAL.copy(),
        }
    )
    graph.add_node(
        "x0",
        node_type="pose",
        dimension=3,
        true_value=POSE_REAL.copy(),
        initial_estimate=POSE_ESTIMADA_INICIAL.copy(),
        estimate=POSE_ESTIMADA_INICIAL.copy(),
    )
    graph.add_node(
        "l0",
        node_type="landmark",
        dimension=2,
        true_value=LANDMARK_REAL.copy(),
        initial_estimate=LANDMARK_ESTIMADO_INICIAL.copy(),
        estimate=LANDMARK_ESTIMADO_INICIAL.copy(),
    )
    graph.add_edge(
        "x0",
        "l0",
        factor_name="obs_x0_l0",
        factor_type="range_bearing_observation",
        variables=("x0", "l0"),
        measurement=medicion.copy(),
        covariance=np.asarray(covarianza, dtype=float).copy(),
        information=np.asarray(informacion, dtype=float).copy(),
        robust_kernel={"type": "huber", "delta": DELTA_HUBER},
        sensor_extrinsic=EXTRINSECA_SENSOR_REAL.copy(),
    )
    return graph


def calcular_incremento_landmark_lm(
    jacobiano_landmark,
    residuo,
    informacion,
    damping,
):
    """Resuelve un paso LM para el landmark manteniendo fija la pose."""

    jacobiano_landmark = np.asarray(jacobiano_landmark, dtype=float)
    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)
    damping = float(damping)

    hessiana = jacobiano_landmark.T @ informacion @ jacobiano_landmark
    gradiente = jacobiano_landmark.T @ informacion @ residuo
    diagonal = np.maximum(np.diag(hessiana), 1.0)
    sistema = hessiana + damping * np.diag(diagonal)
    try:
        incremento = np.linalg.solve(sistema, -gradiente)
    except np.linalg.LinAlgError:
        incremento = np.linalg.lstsq(sistema, -gradiente, rcond=None)[0]
    return {
        "increment": incremento,
        "hessian": hessiana,
        "gradient": gradiente,
        "damped_system": sistema,
    }


def optimizar_landmark_pose_fija(
    pose,
    landmark_inicial,
    medicion,
    informacion,
    max_iteraciones=MAX_ITERACIONES,
):
    """Optimiza el landmark con una pose fija mediante Levenberg-Marquardt."""

    pose = validar_pose(pose)
    landmark = validar_landmark(landmark_inicial)
    medicion = validar_medicion_rango_rumbo(medicion)
    damping = LAMBDA_INICIAL
    historial = []
    convergencia = False

    for iteracion in range(int(max_iteraciones)):
        evaluacion = completar_costes(
            evaluar_observacion(pose, landmark, medicion),
            informacion,
        )
        coste_antes = evaluacion["quadratic_cost"]
        jacobiano = calcular_jacobiano_landmark_analitico(pose, landmark)
        aceptado = False
        mejor = None

        for intento in range(12):
            paso = calcular_incremento_landmark_lm(
                jacobiano,
                evaluacion["residual"],
                informacion,
                damping,
            )
            candidato = validar_landmark(landmark + paso["increment"])
            evaluacion_candidata = completar_costes(
                evaluar_observacion(pose, candidato, medicion),
                informacion,
            )
            coste_candidato = evaluacion_candidata["quadratic_cost"]
            mejor = {
                "iteration": iteracion,
                "attempt": intento,
                "landmark_before": landmark.copy(),
                "landmark_candidate": candidato.copy(),
                "cost_before": coste_antes,
                "cost_candidate": coste_candidato,
                "damping": damping,
                "step_norm": float(np.linalg.norm(paso["increment"])),
                "residual_before": evaluacion["residual"].copy(),
                "residual_candidate": evaluacion_candidata["residual"].copy(),
                "accepted": coste_candidato < coste_antes,
                "hessian": paso["hessian"].copy(),
                "gradient": paso["gradient"].copy(),
            }
            if coste_candidato < coste_antes:
                landmark = candidato
                damping = max(damping * 0.30, 1e-12)
                aceptado = True
                break
            damping = min(damping * 10.0, 1e12)

        if mejor is None:
            raise RuntimeError("No se evaluó ningún intento LM.")

        evaluacion_despues = completar_costes(
            evaluar_observacion(pose, landmark, medicion),
            informacion,
        )
        mejor.update(
            {
                "landmark_after": landmark.copy(),
                "cost_after": evaluacion_despues["quadratic_cost"],
                "range_error_after": evaluacion_despues["range_error"],
                "bearing_error_after": evaluacion_despues["bearing_error"],
                "accepted": aceptado,
                "damping_after": damping,
            }
        )
        historial.append(mejor)

        if not aceptado:
            break
        if mejor["step_norm"] < TOLERANCIA_INCREMENTO:
            convergencia = True
            break
        if abs(coste_antes - evaluacion_despues["quadratic_cost"]) < TOLERANCIA_COSTE:
            convergencia = True
            break

    return {
        "initial_landmark": validar_landmark(landmark_inicial),
        "optimized_landmark": landmark.copy(),
        "history": historial,
        "iterations": len(historial),
        "converged": convergencia,
        "initial_evaluation": completar_costes(
            evaluar_observacion(pose, landmark_inicial, medicion),
            informacion,
        ),
        "final_evaluation": completar_costes(
            evaluar_observacion(pose, landmark, medicion),
            informacion,
        ),
    }


# ---------------------------------------------------------------------------
# Casos didácticos adicionales
# ---------------------------------------------------------------------------


def crear_casos_sensibilidad(medicion, informacion):
    """Evalúa perturbaciones aisladas de pose y landmark."""

    casos = [
        (
            "estimación inicial",
            POSE_ESTIMADA_INICIAL,
            LANDMARK_ESTIMADO_INICIAL,
        ),
        (
            "solo error x de pose",
            POSE_REAL + np.array([0.30, 0.0, 0.0]),
            LANDMARK_REAL,
        ),
        (
            "solo error y de pose",
            POSE_REAL + np.array([0.0, -0.26, 0.0]),
            LANDMARK_REAL,
        ),
        (
            "solo error angular",
            POSE_REAL + np.array([0.0, 0.0, np.deg2rad(8.0)]),
            LANDMARK_REAL,
        ),
        (
            "solo error de landmark",
            POSE_REAL,
            LANDMARK_REAL + np.array([-0.32, 0.28]),
        ),
    ]
    resultados = []
    for nombre, pose, landmark in casos:
        pose = validar_pose(pose)
        evaluacion = completar_costes(
            evaluar_observacion(pose, landmark, medicion),
            informacion,
        )
        resultados.append(
            {
                "name": nombre,
                "pose": pose,
                "landmark": validar_landmark(landmark),
                "evaluation": evaluacion,
            }
        )
    return resultados


def crear_caso_normalizacion_angular():
    """Crea un caso donde la resta directa da 358° y el error real -2°."""

    medido = np.deg2rad(-179.0)
    predicho = np.deg2rad(179.0)
    crudo = predicho - medido
    normalizado = normalizar_angulo(crudo)
    return {
        "measured_bearing": medido,
        "predicted_bearing": predicho,
        "raw_error": crudo,
        "normalized_error": normalizado,
    }


def crear_caso_calibracion(medicion, informacion):
    """Compara la predicción con extrínseca correcta e incorrecta."""

    correcta = completar_costes(
        evaluar_observacion(
            POSE_REAL,
            LANDMARK_REAL,
            medicion,
            EXTRINSECA_SENSOR_REAL,
        ),
        informacion,
    )
    incorrecta = completar_costes(
        evaluar_observacion(
            POSE_REAL,
            LANDMARK_REAL,
            medicion,
            EXTRINSECA_SENSOR_ERRONEA,
        ),
        informacion,
    )
    return {
        "correct": correcta,
        "wrong": incorrecta,
        "true_extrinsic": EXTRINSECA_SENSOR_REAL.copy(),
        "wrong_extrinsic": EXTRINSECA_SENSOR_ERRONEA.copy(),
    }


# ---------------------------------------------------------------------------
# Resultado completo y estados de animación
# ---------------------------------------------------------------------------


def crear_resultado_error_pose_landmark():
    """Construye y resume todo el ejemplo pose-landmark."""

    mediciones = crear_medicion_rango_rumbo()
    covarianza = crear_covarianza_observacion()
    informacion = calcular_matriz_informacion(covarianza)
    graph = crear_grafo_error_pose_landmark(
        mediciones["measured"],
        covarianza,
        informacion,
    )

    evaluacion_real = completar_costes(
        evaluar_observacion(
            POSE_REAL,
            LANDMARK_REAL,
            mediciones["measured"],
        ),
        informacion,
    )
    evaluacion_inicial = completar_costes(
        evaluar_observacion(
            POSE_ESTIMADA_INICIAL,
            LANDMARK_ESTIMADO_INICIAL,
            mediciones["measured"],
        ),
        informacion,
    )

    jacobianos = comparar_jacobianos(
        POSE_ESTIMADA_INICIAL,
        LANDMARK_ESTIMADO_INICIAL,
        mediciones["measured"],
    )
    observabilidad = {
        "joint": analizar_observabilidad_local(jacobianos["analytic_joint"]),
        "pose_only": analizar_observabilidad_local(jacobianos["analytic_pose"]),
        "landmark_only": analizar_observabilidad_local(
            jacobianos["analytic_landmark"]
        ),
    }
    linealizacion = comprobar_linealizacion(
        POSE_ESTIMADA_INICIAL,
        LANDMARK_ESTIMADO_INICIAL,
        mediciones["measured"],
        jacobianos["analytic_joint"],
        np.array(
            [0.0008, -0.0006, np.deg2rad(0.025), 0.0007, -0.0005],
            dtype=float,
        ),
    )
    optimizacion = optimizar_landmark_pose_fija(
        POSE_ESTIMADA_INICIAL,
        LANDMARK_ESTIMADO_INICIAL,
        mediciones["measured"],
        informacion,
    )
    graph.nodes["l0"]["estimate"] = optimizacion["optimized_landmark"].copy()

    return {
        "graph": graph,
        "true_pose": POSE_REAL.copy(),
        "initial_pose": POSE_ESTIMADA_INICIAL.copy(),
        "true_landmark": LANDMARK_REAL.copy(),
        "initial_landmark": LANDMARK_ESTIMADO_INICIAL.copy(),
        "optimized_landmark": optimizacion["optimized_landmark"].copy(),
        "sensor_extrinsic": EXTRINSECA_SENSOR_REAL.copy(),
        "wrong_sensor_extrinsic": EXTRINSECA_SENSOR_ERRONEA.copy(),
        "measurement": mediciones["measured"].copy(),
        "ideal_measurement": mediciones["ideal"].copy(),
        "measurement_noise": mediciones["noise"].copy(),
        "covariance": covarianza,
        "information": informacion,
        "true_evaluation": evaluacion_real,
        "initial_evaluation": evaluacion_inicial,
        "optimized_evaluation": optimizacion["final_evaluation"],
        "jacobians": jacobianos,
        "observability": observabilidad,
        "linearization": linealizacion,
        "optimization": optimizacion,
        "sensitivity_cases": crear_casos_sensibilidad(
            mediciones["measured"],
            informacion,
        ),
        "angle_wrap": crear_caso_normalizacion_angular(),
        "calibration": crear_caso_calibracion(
            mediciones["measured"],
            informacion,
        ),
    }


def crear_estado_animacion(
    phase,
    message,
    result,
    pose=None,
    landmark=None,
    extrinsic=None,
    focus="geometry",
    show_true=False,
    show_measurement=False,
    show_prediction=False,
    show_range_error=False,
    show_bearing_error=False,
    show_uncertainty=False,
    show_jacobians=False,
    show_wrap=False,
    show_calibration=False,
    show_initial_history=False,
    experiment_label=None,
    optimization_iteration=None,
):
    """Crea un estado autosuficiente para la animación."""

    pose = result["initial_pose"] if pose is None else validar_pose(pose)
    landmark = (
        result["initial_landmark"]
        if landmark is None
        else validar_landmark(landmark)
    )
    extrinsic = (
        result["sensor_extrinsic"]
        if extrinsic is None
        else validar_pose(extrinsic, "extrínseca")
    )
    evaluacion = completar_costes(
        evaluar_observacion(
            pose,
            landmark,
            result["measurement"],
            extrinsic,
        ),
        result["information"],
    )
    return {
        "phase": phase,
        "message": str(message),
        "focus": focus,
        "pose": pose.copy(),
        "landmark": landmark.copy(),
        "sensor_extrinsic": extrinsic.copy(),
        "evaluation": evaluacion,
        "show_true": bool(show_true),
        "show_measurement": bool(show_measurement),
        "show_prediction": bool(show_prediction),
        "show_range_error": bool(show_range_error),
        "show_bearing_error": bool(show_bearing_error),
        "show_uncertainty": bool(show_uncertainty),
        "show_jacobians": bool(show_jacobians),
        "show_wrap": bool(show_wrap),
        "show_calibration": bool(show_calibration),
        "show_initial_history": bool(show_initial_history),
        "experiment_label": experiment_label,
        "optimization_iteration": optimization_iteration,
    }


def crear_estados_animacion(result):
    """Crea la secuencia didáctica completa del apartado."""

    states = []

    def add(*args, hold=1, **kwargs):
        for _ in range(int(hold)):
            states.append(crear_estado_animacion(*args, result=result, **kwargs))

    add(
        "introduction",
        "Una observación conecta una pose x0 con un landmark l0.",
        focus="graph",
        show_true=True,
        hold=3,
    )
    add(
        "true_geometry",
        "La geometría real genera la observación ideal del sensor.",
        pose=result["true_pose"],
        landmark=result["true_landmark"],
        focus="geometry",
        show_true=True,
        show_prediction=True,
        hold=3,
    )
    add(
        "measurement",
        "El sensor registra una medición fija z=(r, β) con ruido.",
        pose=result["true_pose"],
        landmark=result["true_landmark"],
        focus="measurement",
        show_true=True,
        show_measurement=True,
        hold=4,
    )
    add(
        "estimated_state",
        "La pose y el landmark estimados no coinciden con los valores reales.",
        focus="geometry",
        show_true=True,
        show_initial_history=True,
        hold=4,
    )
    add(
        "prediction",
        "El modelo h(x0,l0) calcula el rango y el rumbo predichos.",
        focus="prediction",
        show_true=True,
        show_measurement=True,
        show_prediction=True,
        show_initial_history=True,
        hold=4,
    )
    add(
        "range_error",
        "La diferencia radial produce el error de distancia e_r.",
        focus="range_error",
        show_true=True,
        show_measurement=True,
        show_prediction=True,
        show_range_error=True,
        show_initial_history=True,
        hold=4,
    )
    add(
        "bearing_error",
        "La diferencia entre direcciones produce el error angular e_β.",
        focus="bearing_error",
        show_true=True,
        show_measurement=True,
        show_prediction=True,
        show_range_error=True,
        show_bearing_error=True,
        show_initial_history=True,
        hold=4,
    )
    add(
        "residual",
        "El residuo reúne distancia y ángulo: e=[e_r, e_β].",
        focus="residual",
        show_true=True,
        show_measurement=True,
        show_prediction=True,
        show_range_error=True,
        show_bearing_error=True,
        show_initial_history=True,
        hold=4,
    )
    add(
        "angle_wrap",
        "179° y -179° están separados por 2°, no por 358°.",
        focus="wrap",
        show_measurement=True,
        show_prediction=True,
        show_bearing_error=True,
        show_wrap=True,
        hold=5,
    )
    add(
        "uncertainty",
        "La covarianza expresa incertidumbres distintas para metros y radianes.",
        focus="uncertainty",
        show_measurement=True,
        show_prediction=True,
        show_range_error=True,
        show_bearing_error=True,
        show_uncertainty=True,
        hold=4,
    )
    add(
        "mahalanobis",
        "Mahalanobis normaliza cada componente con su incertidumbre.",
        focus="cost",
        show_measurement=True,
        show_prediction=True,
        show_range_error=True,
        show_bearing_error=True,
        show_uncertainty=True,
        hold=4,
    )
    add(
        "jacobians",
        "Los jacobianos describen cómo cambia el residuo al mover x0 o l0.",
        focus="jacobians",
        show_measurement=True,
        show_prediction=True,
        show_range_error=True,
        show_bearing_error=True,
        show_jacobians=True,
        hold=4,
    )

    for caso in result["sensitivity_cases"][1:]:
        add(
            "sensitivity",
            "Perturbación aislada: " + caso["name"] + ".",
            pose=caso["pose"],
            landmark=caso["landmark"],
            focus="sensitivity",
            show_true=True,
            show_measurement=True,
            show_prediction=True,
            show_range_error=True,
            show_bearing_error=True,
            experiment_label=caso["name"],
            hold=3,
        )

    add(
        "observability",
        "Una sola observación aporta dos ecuaciones: el sistema conjunto tiene tres direcciones no observables.",
        focus="observability",
        show_measurement=True,
        show_prediction=True,
        show_jacobians=True,
        hold=4,
    )

    for entrada in result["optimization"]["history"]:
        if not entrada["accepted"]:
            continue
        pasos = 5
        for alpha in np.linspace(0.0, 1.0, pasos, endpoint=True)[1:]:
            landmark = (
                (1.0 - alpha) * entrada["landmark_before"]
                + alpha * entrada["landmark_after"]
            )
            add(
                "landmark_correction",
                "Con la pose fija, LM mueve el landmark para reducir el residuo.",
                landmark=landmark,
                focus="optimization",
                show_true=True,
                show_measurement=True,
                show_prediction=True,
                show_range_error=True,
                show_bearing_error=True,
                show_initial_history=True,
                experiment_label=(
                    f"LM · iteración {entrada['iteration'] + 1}"
                ),
                optimization_iteration=entrada["iteration"],
            )

    add(
        "optimized",
        "La predicción final coincide prácticamente con la medición.",
        landmark=result["optimized_landmark"],
        focus="optimization",
        show_true=True,
        show_measurement=True,
        show_prediction=True,
        show_initial_history=True,
        show_uncertainty=True,
        hold=4,
    )

    add(
        "calibration_correct",
        "Con la extrínseca correcta, la geometría real explica la medida salvo el ruido.",
        pose=result["true_pose"],
        landmark=result["true_landmark"],
        extrinsic=result["sensor_extrinsic"],
        focus="calibration",
        show_true=True,
        show_measurement=True,
        show_prediction=True,
        show_calibration=True,
        experiment_label="extrínseca correcta",
        hold=3,
    )
    add(
        "calibration_wrong",
        "Una extrínseca incorrecta introduce un residuo sistemático.",
        pose=result["true_pose"],
        landmark=result["true_landmark"],
        extrinsic=result["wrong_sensor_extrinsic"],
        focus="calibration",
        show_true=True,
        show_measurement=True,
        show_prediction=True,
        show_range_error=True,
        show_bearing_error=True,
        show_calibration=True,
        experiment_label="extrínseca incorrecta",
        hold=5,
    )

    add(
        "summary",
        "Medición fija, predicción geométrica, residuo normalizado, incertidumbre y coste.",
        focus="summary",
        show_true=True,
        show_measurement=True,
        show_prediction=True,
        show_range_error=True,
        show_bearing_error=True,
        show_uncertainty=True,
        show_jacobians=True,
        show_initial_history=True,
        hold=4,
    )

    total = len(states)
    for indice, state in enumerate(states, start=1):
        state["step"] = indice
        state["total_steps"] = total
    return states


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------


def validar_transformaciones():
    """Comprueba consistencia entre transformaciones y coordenadas."""

    pose = np.array([1.2, -0.7, np.deg2rad(37.0)], dtype=float)
    punto = np.array([3.1, 2.4], dtype=float)
    recuperado = transformar_punto_local_a_global(
        pose,
        transformar_punto_global_a_local(pose, punto),
    )
    if not np.allclose(recuperado, punto, atol=1e-10):
        raise ValueError("Las transformaciones global-local no son inversas.")
    pose_recuperada = matriz_a_pose_se2(pose_a_matriz_se2(pose))
    if not np.allclose(pose_recuperada, pose, atol=1e-10):
        raise ValueError("La conversión pose-matriz no es reversible.")


def validar_grafo(result):
    """Comprueba nodos, factor y metadatos del grafo."""

    graph = result["graph"]
    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise TypeError("El ejemplo debe utilizar un nx.Graph no dirigido.")
    if set(graph.nodes()) != {"x0", "l0"}:
        raise ValueError("El grafo debe contener x0 y l0.")
    if not graph.has_edge("x0", "l0"):
        raise ValueError("Debe existir el factor pose-landmark.")
    edge = graph.edges["x0", "l0"]
    if edge["factor_type"] != "range_bearing_observation":
        raise ValueError("El tipo de factor es incorrecto.")
    if edge["variables"] != ("x0", "l0"):
        raise ValueError("Las variables del factor son incorrectas.")
    if not np.allclose(
        edge["covariance"] @ edge["information"],
        np.eye(2),
        atol=1e-10,
    ):
        raise ValueError("ΣΩ debe ser la identidad.")


def validar_geometria(result):
    """Comprueba medición, predicción, residuos y normalización."""

    medicion = validar_medicion_rango_rumbo(result["measurement"])
    ideal = validar_medicion_rango_rumbo(result["ideal_measurement"])
    if not np.allclose(medicion - ideal, result["measurement_noise"], atol=1e-12):
        diferencia = medicion - ideal
        diferencia[1] = normalizar_angulo(diferencia[1])
        if not np.allclose(diferencia, result["measurement_noise"], atol=1e-12):
            raise ValueError("El ruido de medición no coincide.")

    inicial = result["initial_evaluation"]
    if inicial["residual"].shape != (2,):
        raise ValueError("El residuo debe tener dos componentes.")
    if not (-np.pi <= inicial["bearing_error"] < np.pi):
        raise ValueError("El error angular debe estar normalizado.")
    convertido = cartesiano_a_rango_rumbo(
        rango_rumbo_a_cartesiano(inicial["prediction"])
    )
    if not np.allclose(convertido, inicial["prediction"], atol=1e-10):
        raise ValueError("Las representaciones no son coherentes.")

    wrap = result["angle_wrap"]
    if not np.isclose(np.rad2deg(wrap["raw_error"]), 358.0, atol=1e-10):
        raise ValueError("El caso angular crudo debe producir 358°.")
    if not np.isclose(
        np.rad2deg(wrap["normalized_error"]),
        -2.0,
        atol=1e-10,
    ):
        raise ValueError("El error angular normalizado debe ser -2°.")


def validar_jacobianos(result):
    """Comprueba jacobianos y linealización local."""

    jac = result["jacobians"]
    if jac["analytic_pose"].shape != (2, 3):
        raise ValueError("El jacobiano de pose debe tener forma 2x3.")
    if jac["analytic_landmark"].shape != (2, 2):
        raise ValueError("El jacobiano de landmark debe tener forma 2x2.")
    if jac["pose_max_error"] > 2e-6:
        raise ValueError("El jacobiano de pose no coincide con el numérico.")
    if jac["landmark_max_error"] > 2e-6:
        raise ValueError("El jacobiano de landmark no coincide con el numérico.")
    if result["linearization"]["error_norm"] > 2e-6:
        raise ValueError("La linealización local es demasiado imprecisa.")

    observabilidad = result["observability"]
    if observabilidad["joint"]["rank"] != 2:
        raise ValueError("El factor conjunto debe tener rango dos.")
    if observabilidad["joint"]["nullity"] != 3:
        raise ValueError("El factor conjunto debe tener nulidad tres.")
    if observabilidad["pose_only"]["nullity"] != 1:
        raise ValueError("La pose sola debe conservar una dirección no observable.")
    if observabilidad["landmark_only"]["nullity"] != 0:
        raise ValueError("El landmark con pose fija debe ser observable localmente.")


def validar_costes(result):
    """Comprueba covarianza, información, Mahalanobis y Huber."""

    covarianza = result["covariance"]
    informacion = result["information"]
    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise ValueError("La covarianza debe ser simétrica.")
    if np.min(np.linalg.eigvalsh(covarianza)) <= 0.0:
        raise ValueError("La covarianza debe ser definida positiva.")
    if not np.allclose(covarianza @ informacion, np.eye(2), atol=1e-10):
        raise ValueError("La información debe ser la inversa.")

    evaluacion = result["initial_evaluation"]
    if evaluacion["mahalanobis"] < 0.0:
        raise ValueError("Mahalanobis no puede ser negativo.")
    if not np.isclose(
        np.sum(evaluacion["contributions"]),
        evaluacion["mahalanobis"],
        atol=1e-10,
    ):
        raise ValueError("Las contribuciones no suman Mahalanobis.")
    if not (0.0 <= evaluacion["huber_weight"] <= 1.0):
        raise ValueError("El peso robusto debe pertenecer a [0,1].")


def validar_optimizacion(result):
    """Comprueba que la corrección del landmark reduce el residuo."""

    opt = result["optimization"]
    inicial = opt["initial_evaluation"]
    final = opt["final_evaluation"]
    if final["quadratic_cost"] >= inicial["quadratic_cost"]:
        raise ValueError("La optimización debe reducir el coste.")
    if np.linalg.norm(final["residual"]) > 1e-8:
        raise ValueError("El residuo final debe ser prácticamente cero.")
    if opt["iterations"] < 2:
        raise ValueError("Se esperan varias iteraciones didácticas.")
    costes = [
        entrada["cost_after"]
        for entrada in opt["history"]
        if entrada["accepted"]
    ]
    if any(b >= a for a, b in zip(costes[:-1], costes[1:])):
        raise ValueError("Los costes aceptados deben disminuir estrictamente.")


def validar_resultados(result, states):
    """Ejecuta todas las validaciones y devuelve un resumen."""

    validar_transformaciones()
    validar_grafo(result)
    validar_geometria(result)
    validar_jacobianos(result)
    validar_costes(result)
    validar_optimizacion(result)

    if len(states) < 70:
        raise ValueError("La animación necesita al menos setenta estados.")
    if states[-1]["phase"] != "summary":
        raise ValueError("El último estado debe ser el resumen.")

    inicial = result["initial_evaluation"]
    final = result["optimized_evaluation"]
    calibracion = result["calibration"]
    return {
        "node_count": result["graph"].number_of_nodes(),
        "factor_count": result["graph"].number_of_edges(),
        "state_count": len(states),
        "initial_range_prediction": inicial["prediction"][0],
        "measured_range": inicial["measurement"][0],
        "initial_range_error": inicial["range_error"],
        "initial_bearing_prediction_deg": np.rad2deg(inicial["prediction"][1]),
        "measured_bearing_deg": np.rad2deg(inicial["measurement"][1]),
        "initial_bearing_error_deg": np.rad2deg(inicial["bearing_error"]),
        "initial_cartesian_error": inicial["cartesian_residual_norm"],
        "initial_mahalanobis": inicial["mahalanobis"],
        "initial_quadratic_cost": inicial["quadratic_cost"],
        "initial_huber_cost": inicial["huber_cost"],
        "initial_huber_weight": inicial["huber_weight"],
        "final_range_error": final["range_error"],
        "final_bearing_error_deg": np.rad2deg(final["bearing_error"]),
        "final_quadratic_cost": final["quadratic_cost"],
        "optimization_iterations": result["optimization"]["iterations"],
        "optimization_converged": result["optimization"]["converged"],
        "pose_jacobian_error": result["jacobians"]["pose_max_error"],
        "landmark_jacobian_error": result["jacobians"]["landmark_max_error"],
        "linearization_error": result["linearization"]["error_norm"],
        "joint_rank": result["observability"]["joint"]["rank"],
        "joint_nullity": result["observability"]["joint"]["nullity"],
        "pose_only_rank": result["observability"]["pose_only"]["rank"],
        "pose_only_nullity": result["observability"]["pose_only"]["nullity"],
        "landmark_only_rank": result["observability"]["landmark_only"]["rank"],
        "landmark_only_nullity": result["observability"]["landmark_only"]["nullity"],
        "wrap_raw_deg": np.rad2deg(result["angle_wrap"]["raw_error"]),
        "wrap_normalized_deg": np.rad2deg(
            result["angle_wrap"]["normalized_error"]
        ),
        "correct_calibration_cost": calibracion["correct"]["quadratic_cost"],
        "wrong_calibration_cost": calibracion["wrong"]["quadratic_cost"],
    }


def _format_pose(pose):
    """Formatea una pose para la salida de consola."""

    pose = validar_pose(pose)
    return f"({pose[0]:.4f} m, {pose[1]:.4f} m, {np.rad2deg(pose[2]):.4f}°)"


def _format_landmark(landmark):
    """Formatea un landmark para la salida de consola."""

    landmark = validar_landmark(landmark)
    return f"({landmark[0]:.4f} m, {landmark[1]:.4f} m)"


def imprimir_resumen(result, validation):
    """Imprime las magnitudes principales del ejemplo."""

    print("\n=== Error de observación pose-landmark ===")
    print("Pose real:", _format_pose(result["true_pose"]))
    print("Pose estimada:", _format_pose(result["initial_pose"]))
    print("Landmark real:", _format_landmark(result["true_landmark"]))
    print("Landmark inicial:", _format_landmark(result["initial_landmark"]))
    print("Landmark optimizado:", _format_landmark(result["optimized_landmark"]))
    print(
        "Rango medido/predicho/error: "
        f"{validation['measured_range']:.6f} / "
        f"{validation['initial_range_prediction']:.6f} / "
        f"{validation['initial_range_error']:.6f} m"
    )
    print(
        "Rumbo medido/predicho/error: "
        f"{validation['measured_bearing_deg']:.6f}° / "
        f"{validation['initial_bearing_prediction_deg']:.6f}° / "
        f"{validation['initial_bearing_error_deg']:.6f}°"
    )
    print(f"Mahalanobis inicial: {validation['initial_mahalanobis']:.6f}")
    print(f"Coste cuadrático inicial: {validation['initial_quadratic_cost']:.6f}")
    print(f"Coste Huber inicial: {validation['initial_huber_cost']:.6f}")
    print(f"Peso Huber inicial: {validation['initial_huber_weight']:.6f}")
    print(
        "Coste final: "
        f"{validation['final_quadratic_cost']:.12e} · "
        f"iteraciones: {validation['optimization_iterations']}"
    )
    print(
        "Jacobianos max |analítico-numérico|: "
        f"pose={validation['pose_jacobian_error']:.3e} · "
        f"landmark={validation['landmark_jacobian_error']:.3e}"
    )
    print(
        "Observabilidad local conjunto/pose/landmark: "
        f"nulidad {validation['joint_nullity']}/"
        f"{validation['pose_only_nullity']}/"
        f"{validation['landmark_only_nullity']}"
    )
    print(
        "Normalización angular: "
        f"{validation['wrap_raw_deg']:.1f}° → "
        f"{validation['wrap_normalized_deg']:.1f}°"
    )
    print(
        "Coste calibración correcta/incorrecta: "
        f"{validation['correct_calibration_cost']:.6f} / "
        f"{validation['wrong_calibration_cost']:.6f}"
    )
    print(f"Estados de animación: {validation['state_count']}")


def main():
    result = crear_resultado_error_pose_landmark()
    states = crear_estados_animacion(result)
    validation = validar_resultados(result, states)
    imprimir_resumen(result, validation)

    animator = GraphAnimator(figsize=(19, 10.5), interval=260)
    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "06_graph_slam"
        / "05_error_pose_landmark.png"
    )
    animator.animate_pose_landmark_error(
        result=result,
        states=states,
        title="Error de observación pose-landmark: medida, predicción y residuo",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
