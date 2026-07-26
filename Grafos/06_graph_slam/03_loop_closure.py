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

NUMERO_POSES = 25
INDICE_CANDIDATO_FALSO = 8
SEPARACION_TEMPORAL_MINIMA = 8
UMBRAL_SIMILITUD = 0.93
UMBRAL_INLIER = 0.065
MINIMO_INLIERS = 24
MINIMA_FRACCION_INLIERS = 0.55
MAXIMO_RMSE_GEOMETRICO = 0.050
EPSILON_JACOBIANO = 1e-7
MAX_ITERACIONES = 35
TOLERANCIA_INCREMENTO = 1e-9
TOLERANCIA_COSTE_RELATIVO = 1e-11
LAMBDA_INICIAL = 1e-3
DELTA_HUBER_LOOP = 15.0

SIGMAS_PRIOR = np.array([0.015, 0.015, np.deg2rad(0.35)], dtype=float)
SIGMAS_ODOMETRIA = np.array([0.085, 0.075, np.deg2rad(2.0)], dtype=float)
SIGMAS_CIERRE = np.array([0.040, 0.040, np.deg2rad(0.9)], dtype=float)

SESGO_ESCALA = 1.010
SESGO_LATERAL = 0.008
SESGO_ANGULAR_GRADOS = 0.31


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


def transformar_puntos_se2(pose, puntos):
    """Aplica una transformación de SE(2) a puntos 2D."""

    pose = validar_pose(pose)
    puntos = np.asarray(puntos, dtype=float)
    if puntos.ndim != 2 or puntos.shape[1] != 2:
        raise ValueError("Los puntos deben tener forma (N, 2).")
    if not np.all(np.isfinite(puntos)):
        raise ValueError("Los puntos deben ser finitos.")

    c = np.cos(pose[2])
    s = np.sin(pose[2])
    rotacion = np.array([[c, -s], [s, c]], dtype=float)
    return puntos @ rotacion.T + pose[:2]


# ---------------------------------------------------------------------------
# Trayectoria y odometría
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
    """Crea una trayectoria cerrada con forma de cuadrado redondeado."""

    numero_poses = int(numero_poses)
    if numero_poses < 17:
        raise ValueError("Se requieren al menos diecisiete poses.")

    parametro = np.linspace(0.0, 2.0 * np.pi, numero_poses, dtype=float)
    coseno = np.cos(parametro)
    seno = np.sin(parametro)

    # Superelipse con exponente 4: cuadrado suave sin esquinas discontinuas.
    x = 4.6 * np.sign(coseno) * np.sqrt(np.abs(coseno))
    y = 3.5 * np.sign(seno) * np.sqrt(np.abs(seno))

    x -= x[0]
    y -= y[0]

    dx = np.gradient(x)
    dy = np.gradient(y)
    theta = np.arctan2(dy, dx)

    trayectoria = np.column_stack((x, y, theta))
    trayectoria[-1] = trayectoria[0]
    return validar_trayectoria(trayectoria, "trayectoria real")


def crear_mediciones_odometria(trayectoria_real):
    """Genera odometría determinista con deriva de escala, lateral y angular."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    ideales = []
    medidas = []

    for indice in range(1, len(trayectoria_real)):
        ideal = calcular_movimiento_relativo(
            trayectoria_real[indice - 1],
            trayectoria_real[indice],
        )

        escala = SESGO_ESCALA + 0.0035 * np.sin(0.43 * indice)
        error_longitudinal = 0.0045 * np.cos(0.29 * indice)
        error_lateral = SESGO_LATERAL + 0.004 * np.sin(0.61 * indice)
        error_angular = np.deg2rad(
            SESGO_ANGULAR_GRADOS
            + 0.10 * np.sin(0.35 * indice)
            + 0.04 * np.cos(0.21 * indice)
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

        ideales.append(ideal)
        medidas.append(medida)

    return {
        "ideal": np.asarray(ideales, dtype=float),
        "measured": np.asarray(medidas, dtype=float),
    }


def integrar_odometria(pose_inicial, mediciones):
    """Integra una secuencia de mediciones relativas."""

    pose_inicial = validar_pose(pose_inicial, "pose inicial")
    mediciones = validar_trayectoria(mediciones, "mediciones")

    poses = [pose_inicial.copy()]
    for medicion in mediciones:
        poses.append(componer_poses_se2(poses[-1], medicion))
    return validar_trayectoria(np.asarray(poses), "trayectoria integrada")


# ---------------------------------------------------------------------------
# Reconocimiento de lugares y verificación geométrica
# ---------------------------------------------------------------------------


def _normalizar_vector(vector):
    """Normaliza un vector y evita divisiones por cero."""

    vector = np.asarray(vector, dtype=float)
    norma = float(np.linalg.norm(vector))
    if norma <= 1e-12:
        raise ValueError("No se puede normalizar un vector nulo.")
    return vector / norma


def crear_descriptor_lugar(pose):
    """Crea un descriptor global determinista a partir de una pose real."""

    x, y, theta = validar_pose(pose)
    descriptor = np.array(
        [
            np.cos(0.31 * x),
            np.sin(0.27 * y),
            np.cos(theta),
            np.sin(theta),
            np.cos(0.17 * x + 0.23 * y),
            np.sin(0.21 * x - 0.19 * y),
            np.cos(0.13 * x * y),
            1.0,
        ],
        dtype=float,
    )
    return _normalizar_vector(descriptor)


def crear_base_lugares(trayectoria_real):
    """Crea una base de keyframes con un alias visual controlado."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    descriptores = np.asarray(
        [crear_descriptor_lugar(pose) for pose in trayectoria_real],
        dtype=float,
    )

    # x8 se hace visualmente similar a x0 para simular aliasing perceptual.
    descriptores[INDICE_CANDIDATO_FALSO] = _normalizar_vector(
        0.90 * descriptores[0]
        + 0.10 * descriptores[INDICE_CANDIDATO_FALSO]
    )

    return [
        {
            "index": indice,
            "name": f"x{indice}",
            "descriptor": descriptor.copy(),
            "pose": trayectoria_real[indice].copy(),
        }
        for indice, descriptor in enumerate(descriptores)
    ]


def calcular_similitud_lugares(descriptor_a, descriptor_b):
    """Calcula similitud coseno entre dos descriptores."""

    descriptor_a = _normalizar_vector(descriptor_a)
    descriptor_b = _normalizar_vector(descriptor_b)
    return float(np.clip(descriptor_a @ descriptor_b, -1.0, 1.0))


def generar_candidatos_loop(
    base_lugares,
    indice_actual,
    separacion_minima=SEPARACION_TEMPORAL_MINIMA,
    maximo_candidatos=5,
):
    """Recupera candidatos visuales respetando una separación temporal."""

    indice_actual = int(indice_actual)
    if not 0 <= indice_actual < len(base_lugares):
        raise ValueError("El índice actual no pertenece a la base.")

    descriptor_actual = base_lugares[indice_actual]["descriptor"]
    candidatos = []

    for entrada in base_lugares:
        indice = entrada["index"]
        separacion = abs(indice_actual - indice)
        if indice == indice_actual or separacion < separacion_minima:
            continue

        similitud = calcular_similitud_lugares(
            descriptor_actual,
            entrada["descriptor"],
        )
        candidatos.append(
            {
                "index": indice,
                "name": entrada["name"],
                "similarity": similitud,
                "temporal_separation": separacion,
                "passes_similarity": similitud >= UMBRAL_SIMILITUD,
            }
        )

    candidatos.sort(key=lambda item: (-item["similarity"], item["index"]))
    return candidatos[: int(maximo_candidatos)]


def crear_puntos_base_correspondencias(numero=44):
    """Crea un patrón 2D no degenerado para la verificación geométrica."""

    numero = int(numero)
    if numero < 8:
        raise ValueError("Se requieren al menos ocho puntos.")

    indices = np.arange(numero, dtype=float)
    angulos = 2.0 * np.pi * indices / numero
    radios = 1.1 + 0.38 * np.sin(3.0 * angulos) + 0.17 * np.cos(5.0 * angulos)
    return np.column_stack(
        (
            radios * np.cos(angulos) + 0.18 * np.sin(2.0 * angulos),
            0.82 * radios * np.sin(angulos) + 0.12 * np.cos(4.0 * angulos),
        )
    )


def crear_correspondencias_sinteticas(indice_candidato, indice_actual):
    """Crea correspondencias con inliers para x0 y aliasing para x8."""

    indice_candidato = int(indice_candidato)
    indice_actual = int(indice_actual)
    puntos_actuales = crear_puntos_base_correspondencias(44)
    indices = np.arange(len(puntos_actuales), dtype=float)

    ruido_actual = 0.004 * np.column_stack(
        (np.sin(0.73 * indices), np.cos(0.51 * indices))
    )
    puntos_actuales = puntos_actuales + ruido_actual

    if indice_candidato == 0 and indice_actual == NUMERO_POSES - 1:
        transformacion_real = np.array(
            [0.018, -0.012, np.deg2rad(0.18)],
            dtype=float,
        )
        puntos_candidato = transformar_puntos_se2(
            transformacion_real,
            puntos_actuales,
        )
        puntos_candidato += 0.0045 * np.column_stack(
            (np.cos(0.47 * indices), np.sin(0.59 * indices))
        )

        # Ocho correspondencias se convierten en outliers deterministas.
        outlier_indices = np.array([3, 8, 13, 18, 24, 29, 35, 40], dtype=int)
        puntos_candidato[outlier_indices] += np.column_stack(
            (
                0.55 + 0.08 * np.sin(outlier_indices),
                -0.48 + 0.07 * np.cos(outlier_indices),
            )
        )
    else:
        # Un lugar visualmente parecido, pero geométricamente incompatible.
        angulo = 0.83
        puntos_candidato = np.column_stack(
            (
                1.25 * np.sin(1.71 * indices + angulo),
                0.95 * np.cos(1.29 * indices - 0.4),
            )
        )
        puntos_candidato += 0.18 * np.column_stack(
            (np.sin(0.37 * indices), np.cos(0.43 * indices))
        )

    return {
        "source_points": puntos_actuales,
        "target_points": puntos_candidato,
    }


def estimar_transformacion_rigida_2d(puntos_origen, puntos_destino):
    """Estima la transformación rígida 2D por SVD."""

    puntos_origen = np.asarray(puntos_origen, dtype=float)
    puntos_destino = np.asarray(puntos_destino, dtype=float)
    if puntos_origen.shape != puntos_destino.shape:
        raise ValueError("Los conjuntos de puntos deben tener la misma forma.")
    if puntos_origen.ndim != 2 or puntos_origen.shape[1] != 2:
        raise ValueError("Los puntos deben tener forma (N, 2).")
    if len(puntos_origen) < 2:
        raise ValueError("Se requieren al menos dos correspondencias.")

    centro_origen = np.mean(puntos_origen, axis=0)
    centro_destino = np.mean(puntos_destino, axis=0)
    origen_centrado = puntos_origen - centro_origen
    destino_centrado = puntos_destino - centro_destino

    matriz = origen_centrado.T @ destino_centrado
    u, _, vt = np.linalg.svd(matriz)
    rotacion = vt.T @ u.T
    if np.linalg.det(rotacion) < 0.0:
        vt[-1] *= -1.0
        rotacion = vt.T @ u.T

    traslacion = centro_destino - rotacion @ centro_origen
    theta = np.arctan2(rotacion[1, 0], rotacion[0, 0])
    return validar_pose(np.array([traslacion[0], traslacion[1], theta]))


def calcular_errores_correspondencias(transformacion, origen, destino):
    """Calcula errores euclídeos por correspondencia."""

    predichos = transformar_puntos_se2(transformacion, origen)
    destino = np.asarray(destino, dtype=float)
    return np.linalg.norm(predichos - destino, axis=1)


def verificar_candidato_geometricamente(
    indice_candidato,
    indice_actual,
    umbral_inlier=UMBRAL_INLIER,
):
    """Ejecuta un RANSAC determinista para aceptar o rechazar un candidato."""

    correspondencias = crear_correspondencias_sinteticas(
        indice_candidato,
        indice_actual,
    )
    origen = correspondencias["source_points"]
    destino = correspondencias["target_points"]

    mejor_mascara = None
    mejor_transformacion = None
    mejor_numero = -1
    mejor_rmse = float("inf")

    # Dos correspondencias bastan para proponer una transformación rígida 2D.
    for i in range(len(origen) - 1):
        for j in range(i + 1, len(origen)):
            if np.linalg.norm(origen[i] - origen[j]) < 0.25:
                continue
            if np.linalg.norm(destino[i] - destino[j]) < 0.25:
                continue

            propuesta = estimar_transformacion_rigida_2d(
                origen[[i, j]],
                destino[[i, j]],
            )
            errores = calcular_errores_correspondencias(
                propuesta,
                origen,
                destino,
            )
            mascara = errores <= float(umbral_inlier)
            numero = int(np.count_nonzero(mascara))
            rmse = (
                float(np.sqrt(np.mean(errores[mascara] ** 2)))
                if numero > 0
                else float("inf")
            )

            if numero > mejor_numero or (
                numero == mejor_numero and rmse < mejor_rmse
            ):
                mejor_numero = numero
                mejor_rmse = rmse
                mejor_mascara = mascara
                mejor_transformacion = propuesta

    if mejor_mascara is None or mejor_numero < 2:
        raise RuntimeError("No se pudo estimar ninguna transformación candidata.")

    refinada = estimar_transformacion_rigida_2d(
        origen[mejor_mascara],
        destino[mejor_mascara],
    )
    errores = calcular_errores_correspondencias(refinada, origen, destino)
    mascara = errores <= float(umbral_inlier)
    numero_inliers = int(np.count_nonzero(mascara))
    fraccion = numero_inliers / len(origen)
    rmse = (
        float(np.sqrt(np.mean(errores[mascara] ** 2)))
        if numero_inliers > 0
        else float("inf")
    )

    aceptado = (
        numero_inliers >= MINIMO_INLIERS
        and fraccion >= MINIMA_FRACCION_INLIERS
        and rmse <= MAXIMO_RMSE_GEOMETRICO
    )

    return {
        "candidate_index": int(indice_candidato),
        "current_index": int(indice_actual),
        "measurement": refinada,
        "errors": errores,
        "inlier_mask": mascara,
        "inliers": numero_inliers,
        "outliers": int(len(origen) - numero_inliers),
        "inlier_ratio": float(fraccion),
        "rmse": rmse,
        "accepted_geometry": bool(aceptado),
        "source_points": origen,
        "target_points": destino,
    }


def evaluar_candidatos_loop(base_lugares, indice_actual):
    """Combina similitud visual, separación temporal y geometría."""

    candidatos = generar_candidatos_loop(base_lugares, indice_actual)

    # Se garantiza que el alias visual x8 esté presente en la demostración.
    indices = {item["index"] for item in candidatos}
    if INDICE_CANDIDATO_FALSO not in indices:
        entrada = base_lugares[INDICE_CANDIDATO_FALSO]
        candidatos.append(
            {
                "index": INDICE_CANDIDATO_FALSO,
                "name": entrada["name"],
                "similarity": calcular_similitud_lugares(
                    base_lugares[indice_actual]["descriptor"],
                    entrada["descriptor"],
                ),
                "temporal_separation": abs(
                    indice_actual - INDICE_CANDIDATO_FALSO
                ),
                "passes_similarity": True,
            }
        )

    evaluaciones = []
    for candidato in candidatos:
        geometria = verificar_candidato_geometricamente(
            candidato["index"],
            indice_actual,
        )
        evaluacion = dict(candidato)
        evaluacion.update(geometria)
        evaluacion["accepted"] = bool(
            evaluacion["passes_similarity"]
            and evaluacion["temporal_separation"] >= SEPARACION_TEMPORAL_MINIMA
            and evaluacion["accepted_geometry"]
        )
        evaluaciones.append(evaluacion)

    evaluaciones.sort(
        key=lambda item: (
            not item["accepted"],
            -item["similarity"],
            -item["inliers"],
        )
    )

    aceptados = [item for item in evaluaciones if item["accepted"]]
    if len(aceptados) != 1:
        raise ValueError("Debe existir exactamente un cierre verificado.")

    return {
        "candidates": candidatos,
        "evaluations": evaluaciones,
        "accepted": aceptados[0],
        "false_candidate": next(
            item
            for item in evaluaciones
            if item["candidate_index"] == INDICE_CANDIDATO_FALSO
        ),
    }


# ---------------------------------------------------------------------------
# Construcción del pose graph
# ---------------------------------------------------------------------------


def crear_pose_graph_base(trayectoria_real, odometria, estimacion_inicial):
    """Crea un pose graph con prior y odometría, todavía sin loop closure."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    estimacion_inicial = validar_trayectoria(
        estimacion_inicial,
        "estimación inicial",
    )
    if trayectoria_real.shape != estimacion_inicial.shape:
        raise ValueError("Las trayectorias deben tener la misma forma.")

    cov_prior = crear_covarianza_diagonal(SIGMAS_PRIOR)
    cov_odom = crear_covarianza_diagonal(SIGMAS_ODOMETRIA)

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
            robust_kernel=None,
        )
        factor_order.append(nombre)

    prior = {
        "factor_name": "prior_x0",
        "factor_type": "prior",
        "variables": ("x0",),
        "measurement": trayectoria_real[0].copy(),
        "covariance": cov_prior.copy(),
        "information": calcular_matriz_informacion(cov_prior),
        "robust_kernel": None,
    }

    graph.graph.update(
        {
            "prior": prior,
            "variable_order": [f"x{i}" for i in range(len(trayectoria_real))],
            "factor_order": factor_order,
            "state_dimension": 3 * len(trayectoria_real),
            "reference_frame": "x0",
            "has_loop_closure": False,
            "description": "Pose graph con prior y odometría, sin cierre",
        }
    )
    return graph


def añadir_loop_closure(graph, deteccion):
    """Añade al grafo únicamente el candidato verificado."""

    if graph.graph.get("has_loop_closure"):
        raise ValueError("El grafo ya contiene un cierre de ciclo.")

    aceptado = deteccion["accepted"]
    indice_actual = aceptado["current_index"]
    indice_candidato = aceptado["candidate_index"]
    if graph.has_edge(f"x{indice_actual}", f"x{indice_candidato}"):
        raise ValueError("El cierre coincide con una arista ya existente.")

    cov_loop = crear_covarianza_diagonal(SIGMAS_CIERRE)
    nombre = f"loop_{indice_actual}_{indice_candidato}"
    graph.add_edge(
        f"x{indice_actual}",
        f"x{indice_candidato}",
        factor_name=nombre,
        factor_type="loop_closure",
        variables=(f"x{indice_actual}", f"x{indice_candidato}"),
        measurement=aceptado["measurement"].copy(),
        true_measurement=np.zeros(3, dtype=float),
        covariance=cov_loop.copy(),
        information=calcular_matriz_informacion(cov_loop),
        robust_kernel={"type": "huber", "delta": DELTA_HUBER_LOOP},
        similarity=aceptado["similarity"],
        inliers=aceptado["inliers"],
        outliers=aceptado["outliers"],
        geometric_rmse=aceptado["rmse"],
    )
    graph.graph["factor_order"].append(nombre)
    graph.graph["has_loop_closure"] = True
    graph.graph["loop_factor_name"] = nombre
    graph.graph["description"] = "Pose graph con cierre de ciclo verificado"
    return nombre


def obtener_estimaciones(graph, atributo="estimate"):
    """Extrae las poses en el orden global del estado."""

    return np.asarray(
        [
            validar_pose(graph.nodes[nombre][atributo], nombre)
            for nombre in graph.graph["variable_order"]
        ],
        dtype=float,
    )


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
# Residuos, robustez y sistema global
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


def calcular_peso_huber(residuo, informacion, delta=DELTA_HUBER_LOOP):
    """Calcula el peso IRLS asociado a un kernel de Huber."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)
    delta = float(delta)
    if delta <= 0.0 or not np.isfinite(delta):
        raise ValueError("delta debe ser positivo y finito.")

    norma = float(np.sqrt(max(residuo.T @ informacion @ residuo, 0.0)))
    if norma <= delta or norma <= 1e-12:
        return 1.0
    return float(delta / norma)


def calcular_coste_huber(residuo, informacion, delta=DELTA_HUBER_LOOP):
    """Calcula el coste de Huber a partir de la norma de Mahalanobis."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)
    norma = float(np.sqrt(max(residuo.T @ informacion @ residuo, 0.0)))
    if norma <= delta:
        return 0.5 * norma**2
    return float(delta * (norma - 0.5 * delta))


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


def ensamblar_sistema(graph, poses, incluir_prior=True, usar_robustez=True):
    """Ensambla e, J, Omega efectiva, H, g y el coste global."""

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
    robust_weights = {}
    cost_by_type = {"prior": 0.0, "odometry": 0.0, "loop_closure": 0.0}

    for indice_factor, factor_name in enumerate(factores):
        filas = slice(3 * indice_factor, 3 * indice_factor + 3)
        factor = obtener_factor(graph, factor_name)
        residuo = calcular_residuo_factor(graph, factor_name, poses)
        residual[filas] = residuo

        peso = 1.0
        kernel = factor.get("robust_kernel")
        if usar_robustez and kernel and kernel.get("type") == "huber":
            peso = calcular_peso_huber(
                residuo,
                factor["information"],
                kernel["delta"],
            )
            coste_factor = calcular_coste_huber(
                residuo,
                factor["information"],
                kernel["delta"],
            )
        else:
            coste_factor = 0.5 * float(
                residuo.T @ factor["information"] @ residuo
            )

        informacion[filas, filas] = peso * factor["information"]
        robust_weights[factor_name] = peso
        cost_by_type[factor["factor_type"]] += coste_factor

        bloques = calcular_jacobianos_locales_numericos(
            graph, factor_name, poses
        )
        for variable, bloque in bloques.items():
            columna = 3 * indices[variable]
            jacobiano[filas, columna : columna + 3] = bloque

        factor_slices[factor_name] = filas

    hessiana = jacobiano.T @ informacion @ jacobiano
    gradiente = jacobiano.T @ informacion @ residual
    coste = float(sum(cost_by_type.values()))

    return {
        "factor_order": factores,
        "factor_slices": factor_slices,
        "residual": residual,
        "jacobian": jacobiano,
        "information": informacion,
        "hessian": hessiana,
        "gradient": gradiente,
        "cost": coste,
        "cost_by_type": cost_by_type,
        "robust_weights": robust_weights,
    }


def calcular_coste_total(graph, poses):
    """Calcula el coste robusto ponderado del pose graph."""

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
    """Calcula el residuo y las magnitudes del cierre aceptado."""

    factor_name = graph.graph.get("loop_factor_name")
    if factor_name is None:
        pose_final = validar_trayectoria(poses)[-1]
        pose_inicial = validar_trayectoria(poses)[0]
        residuo = calcular_movimiento_relativo(pose_final, pose_inicial)
    else:
        residuo = calcular_residuo_factor(graph, factor_name, poses)

    return {
        "residual": residuo,
        "translation": float(np.linalg.norm(residuo[:2])),
        "orientation_deg": float(np.rad2deg(abs(residuo[2]))),
    }


def calcular_metricas_trayectoria(trayectoria_real, trayectoria):
    """Calcula errores de posición y orientación respecto al ground truth."""

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


def optimizar_pose_graph(graph, poses_iniciales, max_iteraciones=MAX_ITERACIONES):
    """Optimiza el pose graph con Levenberg-Marquardt y Huber en el loop."""

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

        for intento in range(12):
            incremento = resolver_incremento_lm(
                sistema["hessian"],
                sistema["gradient"],
                damping,
            )
            norma_incremento = float(np.linalg.norm(incremento))
            candidato = aplicar_incremento_estado(poses, incremento)
            sistema_candidato = ensamblar_sistema(graph, candidato)
            coste_candidato = sistema_candidato["cost"]

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
                "loop_weight_before": sistema["robust_weights"].get(
                    graph.graph.get("loop_factor_name"), 1.0
                ),
            }

            if coste_candidato < coste_antes:
                poses = candidato
                damping = max(damping * 0.32, 1e-10)
                aceptado = True
                break

            damping = min(damping * 8.0, 1e12)

        if mejor_intento is None:
            raise RuntimeError("No se evaluó ningún paso de optimización.")

        sistema_despues = ensamblar_sistema(graph, poses)
        metricas = calcular_metricas_trayectoria(
            obtener_estimaciones(graph, "true_pose"),
            poses,
        )
        cierre = calcular_error_cierre(graph, poses)
        mejor_intento.update(
            {
                "poses_after": poses.copy(),
                "cost_after": sistema_despues["cost"],
                "cost_by_type_after": dict(sistema_despues["cost_by_type"]),
                "accepted": aceptado,
                "damping_after": damping,
                "rmse_after": metricas["position_rmse"],
                "closure_after": cierre["translation"],
                "closure_angle_after_deg": cierre["orientation_deg"],
                "loop_weight_after": sistema_despues["robust_weights"].get(
                    graph.graph.get("loop_factor_name"), 1.0
                ),
            }
        )
        history.append(mejor_intento)

        if not aceptado:
            break

        mejora_relativa = (
            coste_antes - sistema_despues["cost"]
        ) / max(coste_antes, 1.0)
        if mejor_intento["step_norm"] < TOLERANCIA_INCREMENTO:
            converged = True
            break
        if mejora_relativa < TOLERANCIA_COSTE_RELATIVO:
            converged = True
            break

    return {
        "initial_poses": validar_trayectoria(poses_iniciales),
        "optimized_poses": poses,
        "history": history,
        "iterations": len(history),
        "converged": converged,
        "final_system": ensamblar_sistema(graph, poses),
    }


# ---------------------------------------------------------------------------
# Construcción del resultado completo
# ---------------------------------------------------------------------------


def crear_resultado_loop_closure():
    """Simula detección, verificación, inserción del loop y optimización."""

    trayectoria_real = crear_trayectoria_real()
    odometria = crear_mediciones_odometria(trayectoria_real)
    trayectoria_inicial = integrar_odometria(
        trayectoria_real[0],
        odometria["measured"],
    )

    base_lugares = crear_base_lugares(trayectoria_real)
    deteccion = evaluar_candidatos_loop(base_lugares, len(trayectoria_real) - 1)

    graph = crear_pose_graph_base(
        trayectoria_real,
        odometria,
        trayectoria_inicial,
    )
    graph_before_loop = graph.copy()
    graph_before_loop.graph = dict(graph.graph)

    # Antes de insertar el loop, el coste de la cadena es casi cero.
    sistema_sin_loop = ensamblar_sistema(graph, trayectoria_inicial)
    cierre_geometrico_inicial = {
        "residual": calcular_movimiento_relativo(
            trayectoria_inicial[-1], trayectoria_inicial[0]
        )
    }
    cierre_geometrico_inicial["translation"] = float(
        np.linalg.norm(cierre_geometrico_inicial["residual"][:2])
    )
    cierre_geometrico_inicial["orientation_deg"] = float(
        np.rad2deg(abs(cierre_geometrico_inicial["residual"][2]))
    )

    loop_factor_name = añadir_loop_closure(graph, deteccion)
    sistema_inicial = ensamblar_sistema(graph, trayectoria_inicial)
    metricas_iniciales = calcular_metricas_trayectoria(
        trayectoria_real,
        trayectoria_inicial,
    )
    cierre_inicial = calcular_error_cierre(graph, trayectoria_inicial)

    optimizacion = optimizar_pose_graph(graph, trayectoria_inicial)
    trayectoria_optimizada = optimizacion["optimized_poses"]
    actualizar_estimaciones_grafo(graph, trayectoria_optimizada)

    sistema_final = optimizacion["final_system"]
    metricas_finales = calcular_metricas_trayectoria(
        trayectoria_real,
        trayectoria_optimizada,
    )
    cierre_final = calcular_error_cierre(graph, trayectoria_optimizada)
    gauge = analizar_rango_y_gauge(graph, trayectoria_optimizada)

    # Peso que recibiría el candidato falso si se insertara erróneamente.
    falso = deteccion["false_candidate"]
    informacion_loop = calcular_matriz_informacion(
        crear_covarianza_diagonal(SIGMAS_CIERRE)
    )
    residuo_falso = calcular_residuo_relativo(
        trayectoria_inicial[-1],
        trayectoria_inicial[falso["candidate_index"]],
        falso["measurement"],
    )
    falso["hypothetical_robust_weight"] = calcular_peso_huber(
        residuo_falso,
        informacion_loop,
        DELTA_HUBER_LOOP,
    )

    return {
        "graph_before_loop": graph_before_loop,
        "graph": graph,
        "true_trajectory": trayectoria_real,
        "initial_trajectory": trayectoria_inicial,
        "optimized_trajectory": trayectoria_optimizada,
        "odometry": odometria,
        "place_database": base_lugares,
        "detection": deteccion,
        "loop_factor_name": loop_factor_name,
        "system_without_loop": sistema_sin_loop,
        "initial_system": sistema_inicial,
        "final_system": sistema_final,
        "initial_metrics": metricas_iniciales,
        "final_metrics": metricas_finales,
        "geometric_initial_closure": cierre_geometrico_inicial,
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
    show_true=True,
    show_initial=False,
    show_prior=False,
    show_database=False,
    show_candidates=False,
    active_candidate=None,
    show_matches=False,
    show_loop=False,
    show_current=False,
    current_poses=None,
    iteration=None,
    cost=None,
    rmse=None,
    closure_error=None,
    damping=None,
    step_norm=None,
    loop_weight=None,
    accepted=None,
    show_history=False,
    show_connections=False,
):
    """Crea un estado autocontenido para el visualizador."""

    return {
        "phase": str(phase),
        "message": str(message),
        "visible_pose_count": int(visible_pose_count),
        "visible_odometry_count": int(visible_odometry_count),
        "show_true": bool(show_true),
        "show_initial": bool(show_initial),
        "show_prior": bool(show_prior),
        "show_database": bool(show_database),
        "show_candidates": bool(show_candidates),
        "active_candidate": active_candidate,
        "show_matches": bool(show_matches),
        "show_loop": bool(show_loop),
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
        "loop_weight": None if loop_weight is None else float(loop_weight),
        "accepted": accepted,
        "show_history": bool(show_history),
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
    """Crea la secuencia completa de detección y corrección del loop."""

    inicial = resultado["initial_trajectory"]
    optimizada = resultado["optimized_trajectory"]
    numero_poses = len(inicial)
    numero_odometrias = numero_poses - 1
    aceptado = resultado["detection"]["accepted"]
    falso = resultado["detection"]["false_candidate"]
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
        "Loop closure: reconocer un lugar, verificarlo y añadir una nueva restricción.",
        repeat=3,
        show_true=True,
    )

    for count in range(1, numero_poses + 1):
        add(
            "odometry_build",
            "La odometría construye la trayectoria pose a pose y acumula deriva.",
            visible_pose_count=count,
            visible_odometry_count=max(0, count - 1),
            show_true=True,
            show_initial=True,
            show_prior=True,
        )

    add(
        "drift",
        "El robot ha vuelto físicamente al inicio, pero la odometría no cierra.",
        repeat=5,
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
        "database",
        "La observación actual se compara con la base de keyframes anteriores.",
        repeat=4,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_database=True,
    )

    add(
        "candidates",
        "La similitud visual propone candidatos, pero todavía no crea aristas.",
        repeat=4,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_database=True,
        show_candidates=True,
    )

    add(
        "false_candidate",
        "x8 es visualmente parecido, pero sus correspondencias no son coherentes.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_database=True,
        show_candidates=True,
        active_candidate=falso["candidate_index"],
        show_matches=True,
        accepted=False,
    )

    add(
        "true_candidate",
        "x0 supera similitud, separación temporal y verificación geométrica.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_database=True,
        show_candidates=True,
        active_candidate=aceptado["candidate_index"],
        show_matches=True,
        accepted=True,
    )

    add(
        "loop_added",
        "La transformación verificada se convierte en la arista x24 → x0.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_candidates=True,
        active_candidate=aceptado["candidate_index"],
        show_loop=True,
        cost=resultado["initial_system"]["cost"],
        rmse=resultado["initial_metrics"]["position_rmse"],
        closure_error=resultado["initial_closure"]["translation"],
        loop_weight=resultado["initial_system"]["robust_weights"][
            resultado["loop_factor_name"]
        ],
    )

    for entry in resultado["optimization"]["history"]:
        for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            poses_interpoladas = interpolar_trayectorias(
                entry["poses_before"],
                entry["poses_after"],
                alpha,
            )
            sistema = ensamblar_sistema(resultado["graph"], poses_interpoladas)
            metricas = calcular_metricas_trayectoria(
                resultado["true_trajectory"],
                poses_interpoladas,
            )
            cierre = calcular_error_cierre(resultado["graph"], poses_interpoladas)
            add(
                "optimization",
                "La optimización redistribuye la inconsistencia por todas las poses.",
                visible_pose_count=numero_poses,
                visible_odometry_count=numero_odometrias,
                show_true=True,
                show_initial=True,
                show_prior=True,
                show_candidates=True,
                active_candidate=aceptado["candidate_index"],
                show_loop=True,
                show_current=True,
                current_poses=poses_interpoladas,
                iteration=entry["iteration"] + 1,
                cost=sistema["cost"],
                rmse=metricas["position_rmse"],
                closure_error=cierre["translation"],
                damping=entry["damping"],
                step_norm=entry["step_norm"],
                loop_weight=sistema["robust_weights"].get(
                    resultado["loop_factor_name"], 1.0
                ),
                accepted=entry["accepted"],
                show_history=True,
            )

    add(
        "robustness",
        "Un kernel robusto limita la influencia de un cierre con residuo extremo.",
        repeat=4,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_candidates=True,
        active_candidate=falso["candidate_index"],
        show_matches=True,
        show_loop=True,
        show_current=True,
        current_poses=optimizada,
        iteration=resultado["optimization"]["iterations"],
        cost=resultado["final_system"]["cost"],
        rmse=resultado["final_metrics"]["position_rmse"],
        closure_error=resultado["final_closure"]["translation"],
        loop_weight=falso["hypothetical_robust_weight"],
        accepted=False,
        show_history=True,
    )

    add(
        "comparison",
        "Antes y después: el cierre reduce deriva, RMSE y error de cierre.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_candidates=True,
        active_candidate=aceptado["candidate_index"],
        show_loop=True,
        show_current=True,
        current_poses=optimizada,
        iteration=resultado["optimization"]["iterations"],
        cost=resultado["final_system"]["cost"],
        rmse=resultado["final_metrics"]["position_rmse"],
        closure_error=resultado["final_closure"]["translation"],
        loop_weight=resultado["final_system"]["robust_weights"][
            resultado["loop_factor_name"]
        ],
        accepted=True,
        show_history=True,
    )

    add(
        "summary",
        "Reconocimiento propone; geometría verifica; el loop añade coherencia global.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_database=True,
        show_candidates=True,
        active_candidate=aceptado["candidate_index"],
        show_matches=True,
        show_loop=True,
        show_current=True,
        current_poses=optimizada,
        iteration=resultado["optimization"]["iterations"],
        cost=resultado["final_system"]["cost"],
        rmse=resultado["final_metrics"]["position_rmse"],
        closure_error=resultado["final_closure"]["translation"],
        loop_weight=resultado["final_system"]["robust_weights"][
            resultado["loop_factor_name"]
        ],
        accepted=True,
        show_history=True,
        show_connections=True,
    )

    for step, state in enumerate(states, start=1):
        state["step"] = step
        state["total_steps"] = len(states)

    return states


# ---------------------------------------------------------------------------
# Validaciones y salida
# ---------------------------------------------------------------------------


def validar_deteccion_loop(resultado):
    """Comprueba recuperación, falso candidato y verificación geométrica."""

    deteccion = resultado["detection"]
    aceptado = deteccion["accepted"]
    falso = deteccion["false_candidate"]

    if aceptado["candidate_index"] != 0:
        raise ValueError("El lugar revisitado correcto debe ser x0.")
    if not aceptado["accepted"]:
        raise ValueError("El candidato verdadero debe aceptarse.")
    if aceptado["similarity"] < UMBRAL_SIMILITUD:
        raise ValueError("El candidato verdadero debe superar la similitud.")
    if aceptado["inliers"] < MINIMO_INLIERS:
        raise ValueError("El candidato verdadero necesita suficientes inliers.")
    if aceptado["rmse"] > MAXIMO_RMSE_GEOMETRICO:
        raise ValueError("El error geométrico del cierre es excesivo.")

    if falso["candidate_index"] != INDICE_CANDIDATO_FALSO:
        raise ValueError("No se ha conservado el candidato falso previsto.")
    if falso["accepted"]:
        raise ValueError("El candidato falso debe rechazarse.")
    if falso["similarity"] < UMBRAL_SIMILITUD:
        raise ValueError("El candidato falso debe ilustrar aliasing visual.")
    if falso["accepted_geometry"]:
        raise ValueError("La geometría debe rechazar el alias visual.")


def validar_grafo(resultado):
    """Comprueba la topología antes y después del loop closure."""

    antes = resultado["graph_before_loop"]
    despues = resultado["graph"]
    if antes.number_of_nodes() != NUMERO_POSES:
        raise ValueError("Debe existir un nodo por pose.")
    if antes.number_of_edges() != NUMERO_POSES - 1:
        raise ValueError("Antes del loop solo debe existir la cadena odométrica.")
    if not nx.is_tree(antes):
        raise ValueError("Antes del loop el grafo debe ser un árbol.")
    if despues.number_of_edges() != NUMERO_POSES:
        raise ValueError("Después del loop debe existir una arista adicional.")
    if nx.is_tree(despues):
        raise ValueError("Después del loop debe existir un ciclo.")
    if not despues.graph.get("has_loop_closure"):
        raise ValueError("El grafo debe marcar la presencia del cierre.")

    tipos = [datos["factor_type"] for _, _, datos in despues.edges(data=True)]
    if tipos.count("odometry") != NUMERO_POSES - 1:
        raise ValueError("Falta alguna odometría.")
    if tipos.count("loop_closure") != 1:
        raise ValueError("Debe existir exactamente un loop closure.")


def validar_sistema(resultado):
    """Valida dimensiones, simetría, ensamblaje y gauge."""

    graph = resultado["graph"]
    poses = resultado["optimized_trajectory"]
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

    gauge = resultado["gauge"]
    if gauge["without_prior"]["nullity"] != 3:
        raise ValueError("Sin prior deben quedar tres libertades de gauge.")
    if gauge["with_prior"]["nullity"] != 0:
        raise ValueError("Con prior el sistema debe tener rango completo.")


def validar_optimizacion(resultado):
    """Comprueba la mejora tras insertar el loop closure."""

    coste_inicial = resultado["initial_system"]["cost"]
    coste_final = resultado["final_system"]["cost"]
    rmse_inicial = resultado["initial_metrics"]["position_rmse"]
    rmse_final = resultado["final_metrics"]["position_rmse"]
    cierre_inicial = resultado["initial_closure"]["translation"]
    cierre_final = resultado["final_closure"]["translation"]

    if not coste_final < coste_inicial:
        raise ValueError("El coste final debe ser menor.")
    if not rmse_final < rmse_inicial:
        raise ValueError("El RMSE final debe ser menor.")
    if not cierre_final < cierre_inicial:
        raise ValueError("El error de cierre debe reducirse.")
    if cierre_final > 0.20 * cierre_inicial:
        raise ValueError("La mejora del cierre debe ser claramente visible.")

    history = resultado["optimization"]["history"]
    if not history:
        raise ValueError("Debe existir al menos una iteración.")
    accepted_costs = [entry["cost_after"] for entry in history if entry["accepted"]]
    if any(b >= a for a, b in zip(accepted_costs, accepted_costs[1:])):
        raise ValueError("Los costes aceptados deben disminuir estrictamente.")


def validar_resultados(resultado, states):
    """Ejecuta todas las validaciones matemáticas y didácticas."""

    validar_deteccion_loop(resultado)
    validar_grafo(resultado)
    validar_sistema(resultado)
    validar_optimizacion(resultado)

    if len(states) < 80:
        raise ValueError("La animación debe contener al menos ochenta estados.")
    if states[-1]["phase"] != "summary":
        raise ValueError("El último estado debe ser el resumen.")
    if not states[-1]["show_loop"] or not states[-1]["show_current"]:
        raise ValueError("El estado final debe mostrar loop y trayectoria corregida.")
    if not any(state["phase"] == "false_candidate" for state in states):
        raise ValueError("Debe mostrarse el candidato falso rechazado.")

    aceptado = resultado["detection"]["accepted"]
    falso = resultado["detection"]["false_candidate"]
    gauge = resultado["gauge"]
    return {
        "pose_count": resultado["graph"].number_of_nodes(),
        "odometry_count": NUMERO_POSES - 1,
        "loop_count": 1,
        "candidate_count": len(resultado["detection"]["evaluations"]),
        "accepted_candidate": aceptado["candidate_index"],
        "accepted_similarity": aceptado["similarity"],
        "accepted_inliers": aceptado["inliers"],
        "accepted_outliers": aceptado["outliers"],
        "accepted_inlier_ratio": aceptado["inlier_ratio"],
        "accepted_geometric_rmse": aceptado["rmse"],
        "false_candidate": falso["candidate_index"],
        "false_similarity": falso["similarity"],
        "false_inliers": falso["inliers"],
        "false_robust_weight": falso["hypothetical_robust_weight"],
        "state_count": len(states),
        "iterations": resultado["optimization"]["iterations"],
        "converged": resultado["optimization"]["converged"],
        "initial_cost": resultado["initial_system"]["cost"],
        "final_cost": resultado["final_system"]["cost"],
        "initial_rmse": resultado["initial_metrics"]["position_rmse"],
        "final_rmse": resultado["final_metrics"]["position_rmse"],
        "initial_closure": resultado["initial_closure"]["translation"],
        "final_closure": resultado["final_closure"]["translation"],
        "initial_angle_rmse_deg": resultado["initial_metrics"]["orientation_rmse_deg"],
        "final_angle_rmse_deg": resultado["final_metrics"]["orientation_rmse_deg"],
        "initial_loop_weight": resultado["initial_system"]["robust_weights"][resultado["loop_factor_name"]],
        "final_loop_weight": resultado["final_system"]["robust_weights"][resultado["loop_factor_name"]],
        "rank_without_prior": gauge["without_prior"]["rank"],
        "nullity_without_prior": gauge["without_prior"]["nullity"],
        "rank_with_prior": gauge["with_prior"]["rank"],
        "nullity_with_prior": gauge["with_prior"]["nullity"],
        "jacobian_shape": resultado["final_system"]["jacobian"].shape,
        "hessian_shape": resultado["final_system"]["hessian"].shape,
    }


def imprimir_resumen(validation):
    """Imprime las magnitudes principales del ejemplo."""

    print("\n=== Loop closure: detección, verificación y optimización ===")
    print(
        f"Poses: {validation['pose_count']} · "
        f"odometrías: {validation['odometry_count']} · loops: {validation['loop_count']}"
    )
    print(
        f"Candidato aceptado: x{validation['accepted_candidate']} · "
        f"similitud {validation['accepted_similarity']:.4f} · "
        f"inliers {validation['accepted_inliers']}/"
        f"{validation['accepted_inliers'] + validation['accepted_outliers']} · "
        f"RMSE geométrico {validation['accepted_geometric_rmse']:.5f} m"
    )
    print(
        f"Candidato falso: x{validation['false_candidate']} · "
        f"similitud {validation['false_similarity']:.4f} · "
        f"inliers {validation['false_inliers']} · "
        f"peso Huber hipotético {validation['false_robust_weight']:.4f}"
    )
    print(f"Iteraciones: {validation['iterations']}")
    print(
        f"Coste: {validation['initial_cost']:.6f} "
        f"→ {validation['final_cost']:.6f}"
    )
    print(
        f"RMSE: {validation['initial_rmse']:.6f} m "
        f"→ {validation['final_rmse']:.6f} m"
    )
    print(
        f"Cierre: {validation['initial_closure']:.6f} m "
        f"→ {validation['final_closure']:.6f} m"
    )
    print(
        "RMSE angular: "
        f"{validation['initial_angle_rmse_deg']:.6f}° "
        f"→ {validation['final_angle_rmse_deg']:.6f}°"
    )
    print(
        "Peso robusto del loop: "
        f"{validation['initial_loop_weight']:.4f} "
        f"→ {validation['final_loop_weight']:.4f}"
    )
    print(
        "Gauge sin/con prior: "
        f"rango {validation['rank_without_prior']}/"
        f"{validation['rank_with_prior']} · "
        f"nulidad {validation['nullity_without_prior']}/"
        f"{validation['nullity_with_prior']}"
    )
    print(f"J: {validation['jacobian_shape']} · H: {validation['hessian_shape']}")
    print(f"Estados de animación: {validation['state_count']}")


def main():
    resultado = crear_resultado_loop_closure()
    states = crear_estados_animacion(resultado)
    validation = validar_resultados(resultado, states)
    imprimir_resumen(validation)

    animator = GraphAnimator(figsize=(19, 10.5), interval=220)
    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "06_graph_slam"
        / "03_loop_closure.png"
    )
    animator.animate_loop_closure(
        result=resultado,
        states=states,
        title="Loop closure: reconocimiento, verificación y corrección global",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
