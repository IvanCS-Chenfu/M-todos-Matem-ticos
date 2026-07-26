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

NUMERO_POSES = 16
EPSILON_JACOBIANO = 1e-7
MAX_ITERACIONES = 30
TOLERANCIA_INCREMENTO = 1e-9
TOLERANCIA_COSTE_RELATIVO = 1e-11
LAMBDA_INICIAL = 1e-3

SIGMAS_PRIOR = np.array([0.015, 0.015, np.deg2rad(0.35)], dtype=float)
SIGMAS_ODOMETRIA = np.array([0.085, 0.070, np.deg2rad(2.0)], dtype=float)
SIGMAS_CIERRE = np.array([0.035, 0.035, np.deg2rad(0.8)], dtype=float)

SESGO_ESCALA = 1.012
SESGO_LATERAL = 0.010
SESGO_ANGULAR_GRADOS = 0.38


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
        raise ValueError(f"{nombre} debe contener tres componentes.")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{nombre} debe contener valores finitos.")

    resultado = pose.copy()
    resultado[2] = normalizar_angulo(resultado[2])
    return resultado


def validar_trayectoria(trayectoria, nombre="trayectoria"):
    """Valida una colección de poses con forma (N, 3)."""

    trayectoria = np.asarray(trayectoria, dtype=float)
    if trayectoria.ndim != 2 or trayectoria.shape[1] != 3:
        raise ValueError(f"{nombre} debe tener forma (N, 3).")
    if len(trayectoria) < 2:
        raise ValueError(f"{nombre} debe contener al menos dos poses.")
    if not np.all(np.isfinite(trayectoria)):
        raise ValueError(f"{nombre} debe contener valores finitos.")

    resultado = trayectoria.copy()
    resultado[:, 2] = np.array(
        [normalizar_angulo(valor) for valor in resultado[:, 2]],
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


def calcular_movimiento_relativo(pose_origen, pose_destino):
    """Calcula pose_origen⁻¹ ⊕ pose_destino."""

    return componer_poses_se2(
        invertir_pose_se2(pose_origen),
        pose_destino,
    )


def aplicar_incremento_local(pose, incremento):
    """Aplica una perturbación local por la derecha."""

    return componer_poses_se2(
        validar_pose(pose),
        validar_pose(incremento, "incremento"),
    )


# ---------------------------------------------------------------------------
# Trayectoria, mediciones y pose graph
# ---------------------------------------------------------------------------


def crear_covarianza_diagonal(sigmas):
    """Crea una covarianza diagonal a partir de desviaciones estándar."""

    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.shape != (3,):
        raise ValueError("Se requieren tres desviaciones estándar.")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("Los sigmas deben ser positivos y finitos.")
    return np.diag(sigmas**2)


def calcular_matriz_informacion(covarianza):
    """Valida e invierte una covarianza 3x3."""

    covarianza = np.asarray(covarianza, dtype=float)
    if covarianza.shape != (3, 3):
        raise ValueError("La covarianza debe tener forma 3x3.")
    if not np.all(np.isfinite(covarianza)):
        raise ValueError("La covarianza debe contener valores finitos.")
    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise ValueError("La covarianza debe ser simétrica.")
    if np.min(np.linalg.eigvalsh(covarianza)) <= 0.0:
        raise ValueError("La covarianza debe ser definida positiva.")
    return np.linalg.inv(covarianza)


def crear_trayectoria_real(numero_poses=NUMERO_POSES):
    """Crea una trayectoria cerrada suave con poses en SE(2)."""

    numero_poses = int(numero_poses)
    if numero_poses < 10:
        raise ValueError("Se requieren al menos diez poses.")

    parametro = np.linspace(0.0, 2.0 * np.pi, numero_poses, dtype=float)
    x = 4.6 * np.cos(parametro) + 0.55 * np.cos(2.0 * parametro)
    y = 3.4 * np.sin(parametro) + 0.35 * np.sin(3.0 * parametro)

    derivada_x = -4.6 * np.sin(parametro) - 1.10 * np.sin(2.0 * parametro)
    derivada_y = 3.4 * np.cos(parametro) + 1.05 * np.cos(3.0 * parametro)
    theta = np.arctan2(derivada_y, derivada_x)

    x -= x[0]
    y -= y[0]

    trayectoria = np.column_stack((x, y, theta))
    trayectoria[-1] = trayectoria[0]
    return validar_trayectoria(trayectoria, "trayectoria real")


def crear_mediciones_odometria(trayectoria_real):
    """Genera odometría sesgada y determinista entre poses consecutivas."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    mediciones = []
    mediciones_ideales = []

    for indice in range(1, len(trayectoria_real)):
        ideal = calcular_movimiento_relativo(
            trayectoria_real[indice - 1],
            trayectoria_real[indice],
        )

        escala = SESGO_ESCALA + 0.004 * np.sin(0.47 * indice)
        error_longitudinal = 0.006 * np.cos(0.31 * indice)
        error_lateral = SESGO_LATERAL + 0.005 * np.sin(0.63 * indice)
        error_angular = np.deg2rad(
            SESGO_ANGULAR_GRADOS
            + 0.12 * np.sin(0.37 * indice)
            + 0.05 * np.cos(0.19 * indice)
        )

        medida = validar_pose(
            np.array(
                [
                    ideal[0] * escala + error_longitudinal,
                    ideal[1] + error_lateral,
                    ideal[2] + error_angular,
                ],
                dtype=float,
            ),
            f"odometría {indice - 1}-{indice}",
        )

        mediciones_ideales.append(ideal)
        mediciones.append(medida)

    return {
        "ideal": np.asarray(mediciones_ideales, dtype=float),
        "measured": np.asarray(mediciones, dtype=float),
    }


def integrar_odometria(pose_inicial, mediciones):
    """Integra una secuencia de mediciones relativas."""

    pose_inicial = validar_pose(pose_inicial, "pose inicial")
    mediciones = validar_trayectoria(mediciones, "mediciones")

    poses = [pose_inicial.copy()]
    for medicion in mediciones:
        poses.append(componer_poses_se2(poses[-1], medicion))
    return validar_trayectoria(np.asarray(poses), "trayectoria integrada")


def crear_pose_graph():
    """Crea un pose graph 2D con prior, odometría y cierre de ciclo."""

    trayectoria_real = crear_trayectoria_real()
    odometria = crear_mediciones_odometria(trayectoria_real)
    estimacion_inicial = integrar_odometria(
        trayectoria_real[0],
        odometria["measured"],
    )

    cov_prior = crear_covarianza_diagonal(SIGMAS_PRIOR)
    cov_odom = crear_covarianza_diagonal(SIGMAS_ODOMETRIA)
    cov_loop = crear_covarianza_diagonal(SIGMAS_CIERRE)

    graph = nx.Graph()

    for indice, (pose_real, pose_inicial) in enumerate(
        zip(trayectoria_real, estimacion_inicial)
    ):
        graph.add_node(
            f"x{indice}",
            index=indice,
            node_type="pose",
            dimension=3,
            true_pose=pose_real.copy(),
            initial_estimate=pose_inicial.copy(),
            estimate=pose_inicial.copy(),
            is_prior=indice == 0,
        )

    factor_order = []

    for indice, medicion in enumerate(odometria["measured"], start=1):
        nombre = f"odom_{indice - 1}_{indice}"
        graph.add_edge(
            f"x{indice - 1}",
            f"x{indice}",
            factor_name=nombre,
            factor_type="odometry",
            variables=(f"x{indice - 1}", f"x{indice}"),
            measurement=medicion.copy(),
            true_measurement=odometria["ideal"][indice - 1].copy(),
            covariance=cov_odom.copy(),
            information=calcular_matriz_informacion(cov_odom),
        )
        factor_order.append(nombre)

    # Las poses x15 y x0 corresponden al mismo lugar físico.
    loop_measurement = validar_pose(
        np.array([0.018, -0.012, np.deg2rad(0.18)], dtype=float),
        "medición de cierre",
    )
    graph.add_edge(
        "x15",
        "x0",
        factor_name="loop_15_0",
        factor_type="loop_closure",
        variables=("x15", "x0"),
        measurement=loop_measurement,
        true_measurement=np.zeros(3, dtype=float),
        covariance=cov_loop.copy(),
        information=calcular_matriz_informacion(cov_loop),
    )
    factor_order.append("loop_15_0")

    prior = {
        "factor_name": "prior_x0",
        "factor_type": "prior",
        "variables": ("x0",),
        "measurement": trayectoria_real[0].copy(),
        "covariance": cov_prior.copy(),
        "information": calcular_matriz_informacion(cov_prior),
    }

    graph.graph.update(
        {
            "prior": prior,
            "variable_order": [f"x{i}" for i in range(len(trayectoria_real))],
            "factor_order": factor_order,
            "state_dimension": 3 * len(trayectoria_real),
            "residual_dimension": 3 * (1 + len(factor_order)),
            "reference_frame": "x0",
            "description": "Pose Graph SLAM 2D con prior, odometría y cierre",
        }
    )

    return graph


def obtener_estimaciones(graph, atributo="estimate"):
    """Extrae las poses en el orden global del estado."""

    poses = []
    for nombre in graph.graph["variable_order"]:
        poses.append(validar_pose(graph.nodes[nombre][atributo], nombre))
    return np.asarray(poses, dtype=float)


def actualizar_estimaciones_grafo(graph, poses):
    """Copia una trayectoria al atributo estimate de cada nodo."""

    poses = validar_trayectoria(poses, "poses")
    if len(poses) != len(graph.graph["variable_order"]):
        raise ValueError("El número de poses no coincide con el grafo.")

    for nombre, pose in zip(graph.graph["variable_order"], poses):
        graph.nodes[nombre]["estimate"] = pose.copy()


def obtener_factor(graph, factor_name):
    """Recupera un factor por su nombre estable."""

    if factor_name == "prior_x0":
        return dict(graph.graph["prior"])

    for origen, destino, datos in graph.edges(data=True):
        if datos.get("factor_name") == factor_name:
            factor = dict(datos)
            factor["origin"] = origen
            factor["target"] = destino
            return factor

    raise KeyError(f"No existe el factor {factor_name!r}.")


# ---------------------------------------------------------------------------
# Residuos, jacobianos y sistema global
# ---------------------------------------------------------------------------


def calcular_residuo_prior(pose, medicion):
    """Calcula el error geométrico de un prior."""

    return calcular_movimiento_relativo(medicion, pose)


def calcular_prediccion_relativa(pose_origen, pose_destino):
    """Predice la medición relativa entre dos poses."""

    return calcular_movimiento_relativo(pose_origen, pose_destino)


def calcular_residuo_relativo(pose_origen, pose_destino, medicion):
    """Calcula z⁻¹ ⊕ (x_i⁻¹ ⊕ x_j)."""

    prediccion = calcular_prediccion_relativa(pose_origen, pose_destino)
    return calcular_movimiento_relativo(medicion, prediccion)


def calcular_residuo_factor(graph, factor_name, poses):
    """Calcula el residuo de un prior, odometría o cierre de ciclo."""

    poses = validar_trayectoria(poses, "poses")
    indices = {
        nombre: indice
        for indice, nombre in enumerate(graph.graph["variable_order"])
    }
    factor = obtener_factor(graph, factor_name)

    if factor["factor_type"] == "prior":
        nombre = factor["variables"][0]
        return calcular_residuo_prior(
            poses[indices[nombre]],
            factor["measurement"],
        )

    origen, destino = factor["variables"]
    return calcular_residuo_relativo(
        poses[indices[origen]],
        poses[indices[destino]],
        factor["measurement"],
    )


def calcular_jacobianos_locales_numericos(
    graph,
    factor_name,
    poses,
    epsilon=EPSILON_JACOBIANO,
):
    """Calcula bloques jacobianos con diferencias centrales."""

    poses = validar_trayectoria(poses, "poses")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon debe ser positivo y finito.")

    factor = obtener_factor(graph, factor_name)
    indices = {
        nombre: indice
        for indice, nombre in enumerate(graph.graph["variable_order"])
    }
    bloques = {}

    for nombre in factor["variables"]:
        indice_pose = indices[nombre]
        bloque = np.zeros((3, 3), dtype=float)

        for componente in range(3):
            delta = np.zeros(3, dtype=float)
            delta[componente] = epsilon

            poses_mas = poses.copy()
            poses_menos = poses.copy()
            poses_mas[indice_pose] = aplicar_incremento_local(
                poses[indice_pose], delta
            )
            poses_menos[indice_pose] = aplicar_incremento_local(
                poses[indice_pose], -delta
            )

            residuo_mas = calcular_residuo_factor(
                graph, factor_name, poses_mas
            )
            residuo_menos = calcular_residuo_factor(
                graph, factor_name, poses_menos
            )

            diferencia = residuo_mas - residuo_menos
            diferencia[2] = normalizar_angulo(diferencia[2])
            bloque[:, componente] = diferencia / (2.0 * epsilon)

        bloques[nombre] = bloque

    return bloques


def ensamblar_sistema(graph, poses, incluir_prior=True):
    """Ensambla e, J, Omega, H, g y el coste global."""

    poses = validar_trayectoria(poses, "poses")
    orden_variables = graph.graph["variable_order"]
    factores = list(graph.graph["factor_order"])
    if incluir_prior:
        factores = ["prior_x0"] + factores

    numero_filas = 3 * len(factores)
    numero_columnas = 3 * len(orden_variables)

    residual = np.zeros(numero_filas, dtype=float)
    jacobiano = np.zeros((numero_filas, numero_columnas), dtype=float)
    informacion = np.zeros((numero_filas, numero_filas), dtype=float)

    indices = {
        nombre: indice
        for indice, nombre in enumerate(orden_variables)
    }
    factor_slices = {}

    for indice_factor, factor_name in enumerate(factores):
        filas = slice(3 * indice_factor, 3 * indice_factor + 3)
        factor = obtener_factor(graph, factor_name)
        residual[filas] = calcular_residuo_factor(graph, factor_name, poses)
        informacion[filas, filas] = factor["information"]

        bloques = calcular_jacobianos_locales_numericos(
            graph, factor_name, poses
        )
        for variable, bloque in bloques.items():
            columna = 3 * indices[variable]
            jacobiano[filas, columna : columna + 3] = bloque

        factor_slices[factor_name] = filas

    hessiana = jacobiano.T @ informacion @ jacobiano
    gradiente = jacobiano.T @ informacion @ residual
    coste = 0.5 * float(residual.T @ informacion @ residual)

    return {
        "factor_order": factores,
        "factor_slices": factor_slices,
        "residual": residual,
        "jacobian": jacobiano,
        "information": informacion,
        "hessian": hessiana,
        "gradient": gradiente,
        "cost": coste,
    }


def calcular_coste_total(graph, poses):
    """Calcula el coste ponderado del pose graph."""

    return float(ensamblar_sistema(graph, poses)["cost"])


def aplicar_incremento_estado(poses, incremento):
    """Aplica un vector de incrementos locales a todas las poses."""

    poses = validar_trayectoria(poses, "poses")
    incremento = np.asarray(incremento, dtype=float)
    if incremento.shape != (3 * len(poses),):
        raise ValueError("La dimensión del incremento no coincide con el estado.")
    if not np.all(np.isfinite(incremento)):
        raise ValueError("El incremento debe ser finito.")

    resultado = []
    for indice, pose in enumerate(poses):
        delta = incremento[3 * indice : 3 * indice + 3]
        resultado.append(aplicar_incremento_local(pose, delta))
    return validar_trayectoria(np.asarray(resultado), "poses actualizadas")


def calcular_error_cierre(graph, poses):
    """Calcula el residuo y las magnitudes del cierre de ciclo."""

    residuo = calcular_residuo_factor(graph, "loop_15_0", poses)
    return {
        "residual": residuo,
        "translation": float(np.linalg.norm(residuo[:2])),
        "orientation_deg": float(np.rad2deg(abs(residuo[2]))),
    }


def calcular_metricas_trayectoria(trayectoria_real, trayectoria):
    """Calcula errores de posición y orientación respecto a ground truth."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    trayectoria = validar_trayectoria(trayectoria, "trayectoria")
    if trayectoria_real.shape != trayectoria.shape:
        raise ValueError("Las trayectorias deben tener la misma forma.")

    errores_posicion = np.linalg.norm(
        trayectoria[:, :2] - trayectoria_real[:, :2], axis=1
    )
    errores_angulo = np.array(
        [
            normalizar_angulo(estimada - real)
            for real, estimada in zip(
                trayectoria_real[:, 2], trayectoria[:, 2]
            )
        ],
        dtype=float,
    )

    return {
        "position_errors": errores_posicion,
        "orientation_errors": errores_angulo,
        "position_rmse": float(np.sqrt(np.mean(errores_posicion**2))),
        "position_mae": float(np.mean(errores_posicion)),
        "position_max": float(np.max(errores_posicion)),
        "orientation_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(errores_angulo**2)))
        ),
        "orientation_max_deg": float(
            np.rad2deg(np.max(np.abs(errores_angulo)))
        ),
    }


def analizar_rango_y_gauge(graph, poses):
    """Compara el rango del jacobiano con y sin prior."""

    con_prior = ensamblar_sistema(graph, poses, incluir_prior=True)["jacobian"]
    sin_prior = ensamblar_sistema(graph, poses, incluir_prior=False)["jacobian"]

    rango_con = int(np.linalg.matrix_rank(con_prior, tol=1e-7))
    rango_sin = int(np.linalg.matrix_rank(sin_prior, tol=1e-7))
    dimension = con_prior.shape[1]

    return {
        "with_prior": {
            "rank": rango_con,
            "nullity": dimension - rango_con,
            "shape": con_prior.shape,
        },
        "without_prior": {
            "rank": rango_sin,
            "nullity": dimension - rango_sin,
            "shape": sin_prior.shape,
        },
    }


# ---------------------------------------------------------------------------
# Optimización no lineal
# ---------------------------------------------------------------------------


def resolver_incremento_lm(hessiana, gradiente, damping):
    """Resuelve el sistema amortiguado de Levenberg-Marquardt."""

    hessiana = np.asarray(hessiana, dtype=float)
    gradiente = np.asarray(gradiente, dtype=float)
    damping = float(damping)

    if hessiana.shape[0] != hessiana.shape[1]:
        raise ValueError("La Hessiana debe ser cuadrada.")
    if gradiente.shape != (hessiana.shape[0],):
        raise ValueError("El gradiente tiene una dimensión incorrecta.")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("El damping debe ser positivo y finito.")

    diagonal = np.maximum(np.diag(hessiana), 1.0)
    sistema = hessiana + damping * np.diag(diagonal)

    try:
        incremento = np.linalg.solve(sistema, -gradiente)
    except np.linalg.LinAlgError:
        incremento = np.linalg.lstsq(sistema, -gradiente, rcond=None)[0]

    if not np.all(np.isfinite(incremento)):
        raise ValueError("El incremento calculado no es finito.")
    return incremento


def optimizar_pose_graph(
    graph,
    poses_iniciales,
    max_iteraciones=MAX_ITERACIONES,
):
    """Optimiza el pose graph con Levenberg-Marquardt determinista."""

    poses = validar_trayectoria(poses_iniciales, "poses iniciales")
    damping = LAMBDA_INICIAL
    history = []
    converged = False

    for iteration in range(int(max_iteraciones)):
        sistema = ensamblar_sistema(graph, poses)
        coste_antes = sistema["cost"]
        gradiente_norma = float(np.linalg.norm(sistema["gradient"]))

        aceptado = False
        mejor_intento = None

        for intento in range(10):
            incremento = resolver_incremento_lm(
                sistema["hessian"],
                sistema["gradient"],
                damping,
            )
            norma_incremento = float(np.linalg.norm(incremento))
            candidato = aplicar_incremento_estado(poses, incremento)
            coste_candidato = calcular_coste_total(graph, candidato)

            mejor_intento = {
                "iteration": iteration,
                "attempt": intento,
                "poses_before": poses.copy(),
                "poses_candidate": candidato.copy(),
                "cost_before": coste_antes,
                "cost_candidate": coste_candidato,
                "damping": damping,
                "step_norm": norma_incremento,
                "gradient_norm": gradiente_norma,
                "accepted": coste_candidato < coste_antes,
            }

            if coste_candidato < coste_antes:
                poses = candidato
                damping = max(damping * 0.32, 1e-10)
                aceptado = True
                break

            damping = min(damping * 8.0, 1e12)

        if mejor_intento is None:
            raise RuntimeError("No se llegó a evaluar ningún paso de optimización.")

        coste_despues = calcular_coste_total(graph, poses)
        metricas_actuales = calcular_metricas_trayectoria(
            obtener_estimaciones(graph, "true_pose"), poses
        )
        cierre_actual = calcular_error_cierre(graph, poses)

        mejor_intento.update(
            {
                "poses_after": poses.copy(),
                "cost_after": coste_despues,
                "accepted": aceptado,
                "damping_after": damping,
                "rmse_after": metricas_actuales["position_rmse"],
                "closure_after": cierre_actual["translation"],
                "closure_angle_after_deg": cierre_actual["orientation_deg"],
            }
        )
        history.append(mejor_intento)

        if not aceptado:
            break

        mejora_relativa = (coste_antes - coste_despues) / max(coste_antes, 1.0)
        if mejor_intento["step_norm"] < TOLERANCIA_INCREMENTO:
            converged = True
            break
        if mejora_relativa < TOLERANCIA_COSTE_RELATIVO:
            converged = True
            break

    sistema_final = ensamblar_sistema(graph, poses)
    return {
        "initial_poses": validar_trayectoria(poses_iniciales),
        "optimized_poses": poses,
        "history": history,
        "iterations": len(history),
        "converged": converged,
        "final_system": sistema_final,
    }


def crear_resultado_pose_graph_slam():
    """Construye, optimiza y analiza el ejemplo completo."""

    graph = crear_pose_graph()
    poses_reales = obtener_estimaciones(graph, "true_pose")
    poses_iniciales = obtener_estimaciones(graph, "initial_estimate")

    sistema_inicial = ensamblar_sistema(graph, poses_iniciales)
    metricas_iniciales = calcular_metricas_trayectoria(
        poses_reales, poses_iniciales
    )
    cierre_inicial = calcular_error_cierre(graph, poses_iniciales)

    optimizacion = optimizar_pose_graph(graph, poses_iniciales)
    poses_optimizadas = optimizacion["optimized_poses"]
    actualizar_estimaciones_grafo(graph, poses_optimizadas)

    sistema_final = optimizacion["final_system"]
    metricas_finales = calcular_metricas_trayectoria(
        poses_reales, poses_optimizadas
    )
    cierre_final = calcular_error_cierre(graph, poses_optimizadas)
    gauge = analizar_rango_y_gauge(graph, poses_optimizadas)

    return {
        "graph": graph,
        "true_trajectory": poses_reales,
        "initial_trajectory": poses_iniciales,
        "optimized_trajectory": poses_optimizadas,
        "initial_system": sistema_inicial,
        "final_system": sistema_final,
        "initial_metrics": metricas_iniciales,
        "final_metrics": metricas_finales,
        "initial_closure": cierre_inicial,
        "final_closure": cierre_final,
        "optimization": optimizacion,
        "gauge": gauge,
    }


# ---------------------------------------------------------------------------
# Estados didácticos de la animación
# ---------------------------------------------------------------------------


def crear_estado_animacion(
    *,
    phase,
    message,
    visible_pose_count=0,
    visible_odometry_count=0,
    show_prior=False,
    show_loop=False,
    show_true=True,
    show_initial=False,
    show_current=False,
    current_poses=None,
    iteration=None,
    cost=None,
    rmse=None,
    closure_error=None,
    damping=None,
    step_norm=None,
    accepted=None,
    show_cost_history=False,
    show_connections=False,
):
    """Crea un estado autocontenido para el visualizador."""

    return {
        "phase": str(phase),
        "message": str(message),
        "visible_pose_count": int(visible_pose_count),
        "visible_odometry_count": int(visible_odometry_count),
        "show_prior": bool(show_prior),
        "show_loop": bool(show_loop),
        "show_true": bool(show_true),
        "show_initial": bool(show_initial),
        "show_current": bool(show_current),
        "current_poses": (
            None
            if current_poses is None
            else np.asarray(current_poses, dtype=float).copy()
        ),
        "iteration": None if iteration is None else int(iteration),
        "cost": None if cost is None else float(cost),
        "rmse": None if rmse is None else float(rmse),
        "closure_error": (
            None if closure_error is None else float(closure_error)
        ),
        "damping": None if damping is None else float(damping),
        "step_norm": None if step_norm is None else float(step_norm),
        "accepted": accepted,
        "show_cost_history": bool(show_cost_history),
        "show_connections": bool(show_connections),
    }


def interpolar_trayectorias(origen, destino, alpha):
    """Interpola posiciones y orientaciones por el camino angular corto."""

    origen = validar_trayectoria(origen, "origen")
    destino = validar_trayectoria(destino, "destino")
    alpha = float(alpha)
    if origen.shape != destino.shape:
        raise ValueError("Las trayectorias deben tener la misma forma.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha debe pertenecer a [0, 1].")

    resultado = (1.0 - alpha) * origen + alpha * destino
    for indice in range(len(resultado)):
        diferencia = normalizar_angulo(destino[indice, 2] - origen[indice, 2])
        resultado[indice, 2] = normalizar_angulo(
            origen[indice, 2] + alpha * diferencia
        )
    return resultado


def crear_estados_animacion(resultado):
    """Crea una secuencia completa de Pose Graph SLAM."""

    graph = resultado["graph"]
    inicial = resultado["initial_trajectory"]
    optimizada = resultado["optimized_trajectory"]
    numero_poses = len(inicial)
    numero_odometrias = numero_poses - 1
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
        "Pose Graph SLAM representa poses como vértices y mediciones como aristas.",
        repeat=3,
        show_true=True,
    )

    for count in range(1, numero_poses + 1):
        add(
            "build_poses",
            "Se crean las poses que formarán el estado del problema.",
            visible_pose_count=count,
            show_true=True,
            show_initial=True,
        )

    add(
        "prior",
        "El prior fija x0 y elimina la libertad global de gauge.",
        repeat=4,
        visible_pose_count=numero_poses,
        show_true=True,
        show_initial=True,
        show_prior=True,
    )

    for count in range(1, numero_odometrias + 1):
        add(
            "odometry",
            "La odometría añade restricciones entre poses consecutivas.",
            visible_pose_count=numero_poses,
            visible_odometry_count=count,
            show_true=True,
            show_initial=True,
            show_prior=True,
        )

    add(
        "drift",
        "La cadena es localmente coherente, pero la trayectoria acumula deriva.",
        repeat=4,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        current_poses=inicial,
        cost=resultado["initial_system"]["cost"],
        rmse=resultado["initial_metrics"]["position_rmse"],
        closure_error=resultado["initial_closure"]["translation"],
    )

    add(
        "loop_closure",
        "El cierre de ciclo conecta la última pose con el lugar inicial.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_loop=True,
        current_poses=inicial,
        cost=resultado["initial_system"]["cost"],
        rmse=resultado["initial_metrics"]["position_rmse"],
        closure_error=resultado["initial_closure"]["translation"],
    )

    cost_history = [resultado["initial_system"]["cost"]]

    for entry in resultado["optimization"]["history"]:
        antes = entry["poses_before"]
        despues = entry["poses_after"]
        for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            poses_interpoladas = interpolar_trayectorias(antes, despues, alpha)
            coste_interp = calcular_coste_total(graph, poses_interpoladas)
            metricas_interp = calcular_metricas_trayectoria(
                resultado["true_trajectory"], poses_interpoladas
            )
            cierre_interp = calcular_error_cierre(graph, poses_interpoladas)

            add(
                "optimization",
                "Todas las poses se corrigen conjuntamente para reducir el coste.",
                visible_pose_count=numero_poses,
                visible_odometry_count=numero_odometrias,
                show_true=True,
                show_initial=True,
                show_current=True,
                show_prior=True,
                show_loop=True,
                current_poses=poses_interpoladas,
                iteration=entry["iteration"] + 1,
                cost=coste_interp,
                rmse=metricas_interp["position_rmse"],
                closure_error=cierre_interp["translation"],
                damping=entry["damping"],
                step_norm=entry["step_norm"],
                accepted=entry["accepted"],
                show_cost_history=True,
            )

        cost_history.append(entry["cost_after"])

    add(
        "comparison",
        "La optimización reduce el coste, el RMSE y el error de cierre.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_current=True,
        show_prior=True,
        show_loop=True,
        current_poses=optimizada,
        iteration=resultado["optimization"]["iterations"],
        cost=resultado["final_system"]["cost"],
        rmse=resultado["final_metrics"]["position_rmse"],
        closure_error=resultado["final_closure"]["translation"],
        show_cost_history=True,
    )

    add(
        "summary",
        "La odometría aporta continuidad local; el cierre y el prior permiten coherencia global.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_current=True,
        show_prior=True,
        show_loop=True,
        current_poses=optimizada,
        iteration=resultado["optimization"]["iterations"],
        cost=resultado["final_system"]["cost"],
        rmse=resultado["final_metrics"]["position_rmse"],
        closure_error=resultado["final_closure"]["translation"],
        show_cost_history=True,
        show_connections=True,
    )

    for step, state in enumerate(states, start=1):
        state["step"] = step
        state["total_steps"] = len(states)
        state["cost_history"] = list(cost_history)

    return states


# ---------------------------------------------------------------------------
# Validaciones y salida
# ---------------------------------------------------------------------------


def validar_grafo(graph):
    """Comprueba la estructura del pose graph."""

    if graph.number_of_nodes() != NUMERO_POSES:
        raise ValueError("Debe existir un nodo por pose.")
    if graph.number_of_edges() != NUMERO_POSES:
        raise ValueError("Deben existir 15 odometrías y un cierre de ciclo.")
    if not nx.is_connected(graph):
        raise ValueError("El pose graph debe ser conectado.")
    if "prior" not in graph.graph:
        raise ValueError("Debe existir un prior.")

    odometrias = 0
    cierres = 0
    for _, _, datos in graph.edges(data=True):
        if datos["factor_type"] == "odometry":
            odometrias += 1
        elif datos["factor_type"] == "loop_closure":
            cierres += 1

        covarianza = datos["covariance"]
        informacion = datos["information"]
        if not np.allclose(covarianza @ informacion, np.eye(3), atol=1e-10):
            raise ValueError("Covarianza e información no son inversas.")

    if odometrias != NUMERO_POSES - 1:
        raise ValueError("Falta alguna restricción de odometría.")
    if cierres != 1:
        raise ValueError("Debe existir exactamente un cierre de ciclo.")


def validar_sistema(graph, poses):
    """Valida dimensiones, simetría, ensamblaje y gauge."""

    sistema = ensamblar_sistema(graph, poses)
    dimension_estado = 3 * NUMERO_POSES
    dimension_residuo = 3 * (1 + graph.number_of_edges())

    if sistema["jacobian"].shape != (dimension_residuo, dimension_estado):
        raise ValueError("El jacobiano tiene una dimensión incorrecta.")
    if sistema["hessian"].shape != (dimension_estado, dimension_estado):
        raise ValueError("La Hessiana tiene una dimensión incorrecta.")
    if not np.allclose(sistema["hessian"], sistema["hessian"].T, atol=1e-8):
        raise ValueError("La Hessiana debe ser simétrica.")
    if not np.allclose(
        sistema["hessian"],
        sistema["jacobian"].T
        @ sistema["information"]
        @ sistema["jacobian"],
        atol=1e-8,
    ):
        raise ValueError("H no coincide con JᵀΩJ.")
    if not np.allclose(
        sistema["gradient"],
        sistema["jacobian"].T
        @ sistema["information"]
        @ sistema["residual"],
        atol=1e-8,
    ):
        raise ValueError("g no coincide con JᵀΩe.")

    gauge = analizar_rango_y_gauge(graph, poses)
    if gauge["without_prior"]["nullity"] != 3:
        raise ValueError("Sin prior deben quedar tres libertades de gauge.")
    if gauge["with_prior"]["nullity"] != 0:
        raise ValueError("Con prior el sistema debe tener rango completo.")


def validar_optimizacion(resultado):
    """Comprueba que la optimización mejora el pose graph."""

    coste_inicial = resultado["initial_system"]["cost"]
    coste_final = resultado["final_system"]["cost"]
    rmse_inicial = resultado["initial_metrics"]["position_rmse"]
    rmse_final = resultado["final_metrics"]["position_rmse"]
    cierre_inicial = resultado["initial_closure"]["translation"]
    cierre_final = resultado["final_closure"]["translation"]

    if not coste_final < coste_inicial:
        raise ValueError("El coste final debe ser menor que el inicial.")
    if not rmse_final < rmse_inicial:
        raise ValueError("El RMSE final debe ser menor que el inicial.")
    if not cierre_final < cierre_inicial:
        raise ValueError("El error de cierre debe reducirse.")
    if coste_final > 0.25 * coste_inicial:
        raise ValueError("La reducción de coste debe ser claramente visible.")
    if cierre_final > 0.25 * cierre_inicial:
        raise ValueError("El cierre final debe mejorar de forma clara.")

    history = resultado["optimization"]["history"]
    if not history:
        raise ValueError("La optimización debe realizar al menos una iteración.")
    accepted_costs = [entry["cost_after"] for entry in history if entry["accepted"]]
    if any(b >= a for a, b in zip(accepted_costs, accepted_costs[1:])):
        raise ValueError("Los costes aceptados deben disminuir estrictamente.")


def validar_resultados(resultado, states):
    """Ejecuta todas las validaciones matemáticas y didácticas."""

    validar_grafo(resultado["graph"])
    validar_sistema(resultado["graph"], resultado["optimized_trajectory"])
    validar_optimizacion(resultado)

    if len(states) < 60:
        raise ValueError("La animación debe contener al menos sesenta estados.")
    if states[-1]["phase"] != "summary":
        raise ValueError("El último estado debe ser el resumen.")
    if not states[-1]["show_initial"] or not states[-1]["show_current"]:
        raise ValueError("El estado final debe comparar antes y después.")
    if not states[-1]["show_loop"] or not states[-1]["show_prior"]:
        raise ValueError("El estado final debe mostrar prior y cierre.")

    inicial = resultado["initial_metrics"]
    final = resultado["final_metrics"]
    gauge = resultado["gauge"]

    return {
        "pose_count": resultado["graph"].number_of_nodes(),
        "odometry_count": NUMERO_POSES - 1,
        "loop_count": 1,
        "factor_count": 1 + resultado["graph"].number_of_edges(),
        "state_count": len(states),
        "iterations": resultado["optimization"]["iterations"],
        "converged": resultado["optimization"]["converged"],
        "initial_cost": resultado["initial_system"]["cost"],
        "final_cost": resultado["final_system"]["cost"],
        "initial_rmse": inicial["position_rmse"],
        "final_rmse": final["position_rmse"],
        "initial_closure": resultado["initial_closure"]["translation"],
        "final_closure": resultado["final_closure"]["translation"],
        "initial_angle_rmse_deg": inicial["orientation_rmse_deg"],
        "final_angle_rmse_deg": final["orientation_rmse_deg"],
        "rank_without_prior": gauge["without_prior"]["rank"],
        "nullity_without_prior": gauge["without_prior"]["nullity"],
        "rank_with_prior": gauge["with_prior"]["rank"],
        "nullity_with_prior": gauge["with_prior"]["nullity"],
        "jacobian_shape": resultado["final_system"]["jacobian"].shape,
        "hessian_shape": resultado["final_system"]["hessian"].shape,
    }


def imprimir_resumen(validation):
    """Imprime las magnitudes principales del ejemplo."""

    print("\n=== Pose Graph SLAM 2D básico ===")
    print(f"Poses: {validation['pose_count']}")
    print(
        f"Factores: {validation['factor_count']} "
        f"(prior 1 · odometría {validation['odometry_count']} · "
        f"cierre {validation['loop_count']})"
    )
    print(f"Iteraciones: {validation['iterations']}")
    print(f"Convergencia detectada: {validation['converged']}")
    print(
        f"Coste: {validation['initial_cost']:.6f} "
        f"→ {validation['final_cost']:.6f}"
    )
    print(
        f"RMSE de posición: {validation['initial_rmse']:.6f} m "
        f"→ {validation['final_rmse']:.6f} m"
    )
    print(
        f"Error de cierre: {validation['initial_closure']:.6f} m "
        f"→ {validation['final_closure']:.6f} m"
    )
    print(
        "RMSE angular: "
        f"{validation['initial_angle_rmse_deg']:.6f}° "
        f"→ {validation['final_angle_rmse_deg']:.6f}°"
    )
    print(
        "Gauge sin/con prior: "
        f"rango {validation['rank_without_prior']} / "
        f"{validation['rank_with_prior']} · "
        f"nulidad {validation['nullity_without_prior']} / "
        f"{validation['nullity_with_prior']}"
    )
    print(f"J: {validation['jacobian_shape']} · H: {validation['hessian_shape']}")
    print(f"Estados de animación: {validation['state_count']}")


def main():
    resultado = crear_resultado_pose_graph_slam()
    states = crear_estados_animacion(resultado)
    validation = validar_resultados(resultado, states)
    imprimir_resumen(validation)

    animator = GraphAnimator(
        figsize=(18, 10),
        interval=230,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "06_graph_slam"
        / "02_pose_graph_slam_basico.png"
    )

    animator.animate_pose_graph_slam(
        result=resultado,
        states=states,
        title="Pose Graph SLAM 2D: antes y después de optimizar",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
