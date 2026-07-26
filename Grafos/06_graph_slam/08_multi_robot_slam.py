from itertools import combinations
from math import atan2, cos, pi, sin
from pathlib import Path
import sys

import networkx as nx
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.graph_anim import GraphAnimator


NUM_POSES_A = 10
NUM_POSES_B = 10
SIGMAS_PRIOR = np.array([0.025, 0.025, np.deg2rad(0.8)], dtype=float)
SIGMAS_ODOMETRIA = np.array([0.065, 0.065, np.deg2rad(1.6)], dtype=float)
SIGMAS_INTER_ROBOT = np.array([0.11, 0.11, np.deg2rad(2.5)], dtype=float)
SIGMAS_INTER_ROBOT_FALSA = np.array([0.10, 0.10, np.deg2rad(2.0)], dtype=float)
DELTA_HUBER = 2.5
UMBRAL_RANSAC = 0.55
MIN_INLIERS_RANSAC = 3
MAX_ITERACIONES_LM = 18
DAMPING_INICIAL = 1e-3
EPSILON_JACOBIANO = 1e-6
TOLERANCIA_PASO = 1e-8


def normalizar_angulo(angulo):
    """Normaliza un ángulo o array de ángulos al intervalo [-pi, pi)."""

    return (np.asarray(angulo) + pi) % (2.0 * pi) - pi


def validar_pose(pose, nombre="pose"):
    """Valida una pose SE(2) almacenada como [x, y, theta]."""

    pose = np.asarray(pose, dtype=float)
    if pose.shape != (3,):
        raise ValueError(f"{nombre} debe tener forma (3,), no {pose.shape}.")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{nombre} contiene valores no finitos.")
    pose = pose.copy()
    pose[2] = float(normalizar_angulo(pose[2]))
    return pose


def validar_trayectoria(trayectoria, numero_poses=None, nombre="trayectoria"):
    """Valida una trayectoria de poses SE(2)."""

    trayectoria = np.asarray(trayectoria, dtype=float)
    if trayectoria.ndim != 2 or trayectoria.shape[1] != 3:
        raise ValueError(f"{nombre} debe tener forma (N, 3).")
    if numero_poses is not None and len(trayectoria) != numero_poses:
        raise ValueError(
            f"{nombre} debe tener {numero_poses} poses, no {len(trayectoria)}."
        )
    if not np.all(np.isfinite(trayectoria)):
        raise ValueError(f"{nombre} contiene valores no finitos.")
    trayectoria = trayectoria.copy()
    trayectoria[:, 2] = normalizar_angulo(trayectoria[:, 2])
    return trayectoria


def rotacion_2d(angulo):
    """Devuelve la matriz de rotación plana asociada a un ángulo."""

    return np.array(
        [[cos(float(angulo)), -sin(float(angulo))],
         [sin(float(angulo)), cos(float(angulo))]],
        dtype=float,
    )


def pose_a_matriz_se2(pose):
    """Convierte [x, y, theta] en una matriz homogénea de SE(2)."""

    x, y, theta = validar_pose(pose)
    matriz = np.eye(3, dtype=float)
    matriz[:2, :2] = rotacion_2d(theta)
    matriz[:2, 2] = [x, y]
    return matriz


def matriz_a_pose_se2(matriz):
    """Convierte una matriz homogénea de SE(2) en [x, y, theta]."""

    matriz = np.asarray(matriz, dtype=float)
    if matriz.shape != (3, 3):
        raise ValueError("La matriz SE(2) debe tener forma (3, 3).")
    if not np.all(np.isfinite(matriz)):
        raise ValueError("La matriz SE(2) contiene valores no finitos.")
    theta = atan2(matriz[1, 0], matriz[0, 0])
    return validar_pose([matriz[0, 2], matriz[1, 2], theta])


def componer_poses_se2(pose_a, pose_b):
    """Compone dos poses SE(2): T_ac = T_ab T_bc."""

    return matriz_a_pose_se2(pose_a_matriz_se2(pose_a) @ pose_a_matriz_se2(pose_b))


def invertir_pose_se2(pose):
    """Calcula la pose inversa en SE(2)."""

    return matriz_a_pose_se2(np.linalg.inv(pose_a_matriz_se2(pose)))


def calcular_movimiento_relativo(pose_origen, pose_destino):
    """Calcula la transformación relativa desde origen hasta destino."""

    return componer_poses_se2(invertir_pose_se2(pose_origen), pose_destino)


def aplicar_incremento_local(pose, incremento):
    """Aplica un incremento local a una pose SE(2)."""

    return componer_poses_se2(pose, validar_pose(incremento, "incremento"))


def transformar_trayectoria(trayectoria, transformacion):
    """Expresa todas las poses de una trayectoria en otro marco."""

    trayectoria = validar_trayectoria(trayectoria)
    transformacion = validar_pose(transformacion, "transformacion")
    return np.vstack(
        [componer_poses_se2(transformacion, pose) for pose in trayectoria]
    )


def crear_covarianza_diagonal(sigmas):
    """Crea una matriz de covarianza diagonal a partir de desviaciones típicas."""

    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.shape != (3,) or np.any(sigmas <= 0.0):
        raise ValueError("Las sigmas deben ser tres valores positivos.")
    return np.diag(sigmas**2)


def calcular_matriz_informacion(covarianza):
    """Calcula la matriz de información como inversa de la covarianza."""

    covarianza = np.asarray(covarianza, dtype=float)
    if covarianza.shape != (3, 3):
        raise ValueError("La covarianza debe tener forma (3, 3).")
    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise ValueError("La covarianza debe ser simétrica.")
    if np.min(np.linalg.eigvalsh(covarianza)) <= 0.0:
        raise ValueError("La covarianza debe ser definida positiva.")
    return np.linalg.inv(covarianza)


def crear_trayectorias_globales_reales():
    """Crea dos trayectorias globales con tres lugares visitados por ambos robots."""

    trayectoria_a = np.array(
        [
            [0.0, 0.0, np.deg2rad(0.0)],
            [1.45, 0.18, np.deg2rad(8.0)],
            [3.00, 0.80, np.deg2rad(22.0)],
            [4.25, 1.85, np.deg2rad(42.0)],
            [4.82, 3.18, np.deg2rad(72.0)],
            [4.40, 4.60, np.deg2rad(108.0)],
            [3.18, 5.38, np.deg2rad(150.0)],
            [1.82, 5.20, np.deg2rad(-166.0)],
            [0.60, 4.20, np.deg2rad(-126.0)],
            [0.08, 2.52, np.deg2rad(-96.0)],
        ],
        dtype=float,
    )

    trayectoria_b = np.array(
        [
            [7.00, -1.00, np.deg2rad(168.0)],
            [5.25, -0.18, np.deg2rad(158.0)],
            [3.00, 0.80, np.deg2rad(142.0)],
            [3.72, 1.92, np.deg2rad(112.0)],
            [4.48, 3.05, np.deg2rad(86.0)],
            [4.40, 4.60, np.deg2rad(76.0)],
            [3.52, 5.18, np.deg2rad(161.0)],
            [1.92, 5.37, np.deg2rad(-171.0)],
            [0.60, 4.20, np.deg2rad(-108.0)],
            [-0.28, 2.82, np.deg2rad(-96.0)],
        ],
        dtype=float,
    )

    return (
        validar_trayectoria(trayectoria_a, NUM_POSES_A, "trayectoria A real"),
        validar_trayectoria(trayectoria_b, NUM_POSES_B, "trayectoria B real"),
    )


def expresar_trayectorias_en_marcos_locales(trayectoria_a_global, trayectoria_b_global):
    """Expresa cada trayectoria en el marco de su primera pose."""

    trayectoria_a_global = validar_trayectoria(trayectoria_a_global, NUM_POSES_A)
    trayectoria_b_global = validar_trayectoria(trayectoria_b_global, NUM_POSES_B)

    transformacion_global_a = trayectoria_a_global[0]
    transformacion_global_b = trayectoria_b_global[0]

    trayectoria_a_local = transformar_trayectoria(
        trayectoria_a_global,
        invertir_pose_se2(transformacion_global_a),
    )
    trayectoria_b_local = transformar_trayectoria(
        trayectoria_b_global,
        invertir_pose_se2(transformacion_global_b),
    )

    transformacion_a_b = calcular_movimiento_relativo(
        transformacion_global_a,
        transformacion_global_b,
    )

    return {
        "trajectory_a_local": trayectoria_a_local,
        "trajectory_b_local": trayectoria_b_local,
        "transform_a_b_true": transformacion_a_b,
        "transform_global_a": transformacion_global_a,
        "transform_global_b": transformacion_global_b,
    }


def crear_mediciones_odometria(trayectoria_local, robot):
    """Crea odometrías con sesgo determinista para un robot."""

    trayectoria_local = validar_trayectoria(trayectoria_local)
    robot = str(robot).upper()
    if robot not in {"A", "B"}:
        raise ValueError("El robot debe ser A o B.")

    mediciones = []
    for indice in range(len(trayectoria_local) - 1):
        verdadera = calcular_movimiento_relativo(
            trayectoria_local[indice], trayectoria_local[indice + 1]
        )
        signo = 1.0 if robot == "A" else -1.0
        ruido = np.array(
            [
                signo * (0.012 + 0.004 * (indice % 3)),
                0.010 * np.sin(0.9 * indice + (0.2 if robot == "B" else 0.0)),
                np.deg2rad(signo * (0.22 + 0.05 * (indice % 4))),
            ],
            dtype=float,
        )
        medicion = verdadera + ruido
        medicion[2] = float(normalizar_angulo(medicion[2]))
        mediciones.append(medicion)

    return np.asarray(mediciones, dtype=float)


def integrar_odometria(mediciones, pose_inicial=None):
    """Integra una secuencia de odometrías para obtener una trayectoria inicial."""

    mediciones = np.asarray(mediciones, dtype=float)
    if mediciones.ndim != 2 or mediciones.shape[1] != 3:
        raise ValueError("Las odometrías deben tener forma (N-1, 3).")
    pose = np.zeros(3, dtype=float) if pose_inicial is None else validar_pose(pose_inicial)
    trayectoria = [pose.copy()]
    for medicion in mediciones:
        pose = componer_poses_se2(pose, medicion)
        trayectoria.append(pose.copy())
    return validar_trayectoria(np.asarray(trayectoria, dtype=float))


def crear_lugares_compartidos(trayectoria_a_local, trayectoria_b_local):
    """Crea tres correspondencias correctas y un alias perceptual falso."""

    trayectoria_a_local = validar_trayectoria(trayectoria_a_local)
    trayectoria_b_local = validar_trayectoria(trayectoria_b_local)

    especificaciones = [
        ("c0", 2, 2, True, 0.13),
        ("c1", 5, 5, True, 0.16),
        ("c2", 8, 8, True, 0.18),
        ("c3", 1, 7, False, 0.11),
    ]

    lugares = []
    for nombre, indice_a, indice_b, correcto, distancia_descriptor in especificaciones:
        ruido_a = np.array(
            [0.025 * np.sin(indice_a + 0.4), 0.020 * np.cos(indice_a + 0.2)]
        )
        ruido_b = np.array(
            [0.022 * np.cos(indice_b + 0.3), -0.018 * np.sin(indice_b + 0.5)]
        )
        lugares.append(
            {
                "name": nombre,
                "pose_a": f"A_x{indice_a}",
                "pose_b": f"B_x{indice_b}",
                "index_a": indice_a,
                "index_b": indice_b,
                "point_a": trayectoria_a_local[indice_a, :2] + ruido_a,
                "point_b": trayectoria_b_local[indice_b, :2] + ruido_b,
                "descriptor_distance": float(distancia_descriptor),
                "is_true": bool(correcto),
            }
        )
    return lugares


def estimar_transformacion_rigida_2d(puntos_b, puntos_a):
    """Estima la transformación que alinea puntos de B con puntos de A mediante SVD."""

    puntos_b = np.asarray(puntos_b, dtype=float)
    puntos_a = np.asarray(puntos_a, dtype=float)
    if puntos_b.shape != puntos_a.shape or puntos_b.ndim != 2 or puntos_b.shape[1] != 2:
        raise ValueError("Los conjuntos de puntos deben tener la misma forma (N, 2).")
    if len(puntos_a) < 2:
        raise ValueError("Se necesitan al menos dos correspondencias.")

    centro_b = np.mean(puntos_b, axis=0)
    centro_a = np.mean(puntos_a, axis=0)
    qb = puntos_b - centro_b
    qa = puntos_a - centro_a
    h = qb.T @ qa
    u, _, vt = np.linalg.svd(h)
    rotacion = vt.T @ u.T
    if np.linalg.det(rotacion) < 0.0:
        vt[-1, :] *= -1.0
        rotacion = vt.T @ u.T
    traslacion = centro_a - rotacion @ centro_b
    angulo = atan2(rotacion[1, 0], rotacion[0, 0])
    return validar_pose([traslacion[0], traslacion[1], angulo])


def aplicar_transformacion_puntos(puntos, transformacion):
    """Aplica una transformación SE(2) a un conjunto de puntos 2D."""

    puntos = np.asarray(puntos, dtype=float)
    if puntos.ndim != 2 or puntos.shape[1] != 2:
        raise ValueError("Los puntos deben tener forma (N, 2).")
    transformacion = validar_pose(transformacion)
    return (rotacion_2d(transformacion[2]) @ puntos.T).T + transformacion[:2]


def verificar_transformacion_ransac(lugares, umbral=UMBRAL_RANSAC):
    """Verifica correspondencias inter-robot probando todas las muestras mínimas."""

    if len(lugares) < 2:
        raise ValueError("RANSAC necesita al menos dos correspondencias.")

    puntos_a = np.vstack([lugar["point_a"] for lugar in lugares])
    puntos_b = np.vstack([lugar["point_b"] for lugar in lugares])
    historial = []
    mejor = None

    for indice_hipotesis, muestra in enumerate(combinations(range(len(lugares)), 2)):
        transformacion = estimar_transformacion_rigida_2d(
            puntos_b[list(muestra)], puntos_a[list(muestra)]
        )
        transformados = aplicar_transformacion_puntos(puntos_b, transformacion)
        errores = np.linalg.norm(transformados - puntos_a, axis=1)
        inliers = errores <= float(umbral)
        numero_inliers = int(np.sum(inliers))
        rmse = (
            float(np.sqrt(np.mean(errores[inliers] ** 2)))
            if numero_inliers > 0
            else float("inf")
        )
        entrada = {
            "hypothesis": indice_hipotesis,
            "sample": tuple(int(v) for v in muestra),
            "transform": transformacion,
            "errors": errores,
            "inliers": inliers,
            "inlier_count": numero_inliers,
            "rmse": rmse,
        }
        historial.append(entrada)
        criterio = (-numero_inliers, rmse, muestra)
        if mejor is None or criterio < mejor[0]:
            mejor = (criterio, entrada)

    mejor_entrada = mejor[1]
    inliers_iniciales = mejor_entrada["inliers"]
    if int(np.sum(inliers_iniciales)) >= 2:
        refinada = estimar_transformacion_rigida_2d(
            puntos_b[inliers_iniciales], puntos_a[inliers_iniciales]
        )
    else:
        refinada = mejor_entrada["transform"]

    transformados = aplicar_transformacion_puntos(puntos_b, refinada)
    errores = np.linalg.norm(transformados - puntos_a, axis=1)
    inliers = errores <= float(umbral)
    numero_inliers = int(np.sum(inliers))
    rmse = (
        float(np.sqrt(np.mean(errores[inliers] ** 2)))
        if numero_inliers > 0
        else float("inf")
    )

    return {
        "transform": refinada,
        "errors": errores,
        "inliers": inliers,
        "outliers": ~inliers,
        "inlier_count": numero_inliers,
        "outlier_count": int(len(lugares) - numero_inliers),
        "inlier_ratio": float(numero_inliers / len(lugares)),
        "rmse": rmse,
        "accepted": bool(numero_inliers >= MIN_INLIERS_RANSAC),
        "history": historial,
        "threshold": float(umbral),
    }


def crear_mediciones_inter_robot(trayectoria_a_global, trayectoria_b_global, lugares, ransac):
    """Crea mediciones pose-pose para los lugares verificados como inliers."""

    trayectoria_a_global = validar_trayectoria(trayectoria_a_global)
    trayectoria_b_global = validar_trayectoria(trayectoria_b_global)
    mediciones = []

    for indice, lugar in enumerate(lugares):
        if not bool(ransac["inliers"][indice]):
            continue
        ia = lugar["index_a"]
        ib = lugar["index_b"]
        verdadera = calcular_movimiento_relativo(
            trayectoria_a_global[ia], trayectoria_b_global[ib]
        )
        ruido = np.array(
            [
                0.018 * np.cos(indice + 0.4),
                0.014 * np.sin(indice + 0.8),
                np.deg2rad(0.28 * np.cos(indice + 0.2)),
            ],
            dtype=float,
        )
        medicion = verdadera + ruido
        medicion[2] = float(normalizar_angulo(medicion[2]))
        mediciones.append(
            {
                "factor_name": f"inter_{lugar['name']}",
                "node_i": lugar["pose_a"],
                "node_j": lugar["pose_b"],
                "measurement": medicion,
                "is_false": False,
                "candidate_name": lugar["name"],
            }
        )

    return mediciones


def crear_medicion_inter_robot_falsa(trayectoria_a_global, trayectoria_b_global, lugar_falso):
    """Construye la falsa restricción que habría producido el alias rechazado."""

    ia = lugar_falso["index_a"]
    ib = lugar_falso["index_b"]
    # La medición falsa imita una coincidencia de lugar: traslación casi nula.
    medicion = np.array([0.10, -0.08, np.deg2rad(3.0)], dtype=float)
    verdadera = calcular_movimiento_relativo(
        trayectoria_a_global[ia], trayectoria_b_global[ib]
    )
    return {
        "factor_name": "inter_falsa_c3",
        "node_i": lugar_falso["pose_a"],
        "node_j": lugar_falso["pose_b"],
        "measurement": medicion,
        "true_relative": verdadera,
        "is_false": True,
        "candidate_name": lugar_falso["name"],
    }


def crear_grafos_locales(trayectoria_a_inicial, trayectoria_b_inicial, odometria_a, odometria_b):
    """Crea los dos pose graphs locales antes de la fusión."""

    grafo_a = nx.Graph(robot="A", frame="W_A")
    grafo_b = nx.Graph(robot="B", frame="W_B")

    for indice, pose in enumerate(validar_trayectoria(trayectoria_a_inicial)):
        grafo_a.add_node(f"A_x{indice}", estimate=pose.copy(), robot="A", index=indice)
    for indice, pose in enumerate(validar_trayectoria(trayectoria_b_inicial)):
        grafo_b.add_node(f"B_x{indice}", estimate=pose.copy(), robot="B", index=indice)

    for indice, medicion in enumerate(np.asarray(odometria_a, dtype=float)):
        grafo_a.add_edge(
            f"A_x{indice}", f"A_x{indice + 1}",
            measurement=medicion.copy(), factor_type="odometry"
        )
    for indice, medicion in enumerate(np.asarray(odometria_b, dtype=float)):
        grafo_b.add_edge(
            f"B_x{indice}", f"B_x{indice + 1}",
            measurement=medicion.copy(), factor_type="odometry"
        )

    return grafo_a, grafo_b


def crear_grafo_global(trayectoria_a_inicial, trayectoria_b_alineada, odometria_a, odometria_b, mediciones_inter):
    """Fusiona ambos pose graphs y añade las restricciones inter-robot aceptadas."""

    grafo = nx.Graph(name="SLAM multi-robot")
    factores = {}
    orden_factores = []

    for robot, trayectoria in (("A", trayectoria_a_inicial), ("B", trayectoria_b_alineada)):
        for indice, pose in enumerate(validar_trayectoria(trayectoria)):
            nombre = f"{robot}_x{indice}"
            grafo.add_node(nombre, estimate=pose.copy(), robot=robot, index=indice)

    cov_prior = crear_covarianza_diagonal(SIGMAS_PRIOR)
    prior = {
        "factor_name": "prior_A_x0",
        "factor_type": "prior",
        "variables": ("A_x0",),
        "measurement": np.zeros(3, dtype=float),
        "covariance": cov_prior,
        "information": calcular_matriz_informacion(cov_prior),
        "robust_kernel": None,
        "is_inter_robot": False,
        "is_false": False,
    }
    factores[prior["factor_name"]] = prior
    orden_factores.append(prior["factor_name"])

    cov_odom = crear_covarianza_diagonal(SIGMAS_ODOMETRIA)
    info_odom = calcular_matriz_informacion(cov_odom)
    for robot, odometria in (("A", odometria_a), ("B", odometria_b)):
        for indice, medicion in enumerate(np.asarray(odometria, dtype=float)):
            nombre = f"odom_{robot}_{indice}_{indice + 1}"
            factor = {
                "factor_name": nombre,
                "factor_type": "odometry",
                "variables": (f"{robot}_x{indice}", f"{robot}_x{indice + 1}"),
                "measurement": medicion.copy(),
                "covariance": cov_odom.copy(),
                "information": info_odom.copy(),
                "robust_kernel": None,
                "is_inter_robot": False,
                "is_false": False,
            }
            factores[nombre] = factor
            orden_factores.append(nombre)
            grafo.add_edge(*factor["variables"], factor_name=nombre, factor_type="odometry")

    cov_inter = crear_covarianza_diagonal(SIGMAS_INTER_ROBOT)
    info_inter = calcular_matriz_informacion(cov_inter)
    for especificacion in mediciones_inter:
        nombre = especificacion["factor_name"]
        factor = {
            "factor_name": nombre,
            "factor_type": "inter_robot",
            "variables": (especificacion["node_i"], especificacion["node_j"]),
            "measurement": especificacion["measurement"].copy(),
            "covariance": cov_inter.copy(),
            "information": info_inter.copy(),
            "robust_kernel": {"type": "huber", "delta": DELTA_HUBER},
            "is_inter_robot": True,
            "is_false": False,
            "candidate_name": especificacion["candidate_name"],
        }
        factores[nombre] = factor
        orden_factores.append(nombre)
        grafo.add_edge(*factor["variables"], factor_name=nombre, factor_type="inter_robot")

    grafo.graph["factors"] = factores
    grafo.graph["factor_order"] = orden_factores
    grafo.graph["node_order"] = [
        *(f"A_x{i}" for i in range(NUM_POSES_A)),
        *(f"B_x{i}" for i in range(NUM_POSES_B)),
    ]
    grafo.graph["state_dimension"] = 3 * (NUM_POSES_A + NUM_POSES_B)
    return grafo


def obtener_estimaciones(grafo):
    """Extrae las poses del grafo en el orden global de nodos."""

    return np.vstack(
        [grafo.nodes[nombre]["estimate"] for nombre in grafo.graph["node_order"]]
    )


def actualizar_estimaciones_grafo(grafo, estimaciones):
    """Actualiza las estimaciones almacenadas en los nodos."""

    estimaciones = validar_trayectoria(
        estimaciones, len(grafo.graph["node_order"]), "estimaciones"
    )
    for nombre, pose in zip(grafo.graph["node_order"], estimaciones):
        grafo.nodes[nombre]["estimate"] = pose.copy()


def calcular_residuo_prior(pose, medicion):
    """Calcula el residuo de un prior absoluto."""

    residuo = validar_pose(pose) - validar_pose(medicion)
    residuo[2] = float(normalizar_angulo(residuo[2]))
    return residuo


def calcular_residuo_relativo(pose_i, pose_j, medicion):
    """Calcula el residuo de un factor pose-pose."""

    prediccion = calcular_movimiento_relativo(pose_i, pose_j)
    residuo = prediccion - validar_pose(medicion)
    residuo[2] = float(normalizar_angulo(residuo[2]))
    return residuo


def calcular_residuo_factor(factor, poses):
    """Evalúa el residuo de cualquier factor del grafo multi-robot."""

    if factor["factor_type"] == "prior":
        return calcular_residuo_prior(poses[factor["variables"][0]], factor["measurement"])
    i, j = factor["variables"]
    return calcular_residuo_relativo(poses[i], poses[j], factor["measurement"])


def calcular_jacobianos_numericos_factor(factor, poses, epsilon=EPSILON_JACOBIANO):
    """Calcula jacobianos centrales respecto a las poses del factor."""

    jacobianos = {}
    for nombre in factor["variables"]:
        jacobiano = np.zeros((3, 3), dtype=float)
        for columna in range(3):
            incremento = np.zeros(3, dtype=float)
            incremento[columna] = epsilon
            poses_mas = {clave: valor.copy() for clave, valor in poses.items()}
            poses_menos = {clave: valor.copy() for clave, valor in poses.items()}
            poses_mas[nombre] = aplicar_incremento_local(poses_mas[nombre], incremento)
            poses_menos[nombre] = aplicar_incremento_local(poses_menos[nombre], -incremento)
            r_mas = calcular_residuo_factor(factor, poses_mas)
            r_menos = calcular_residuo_factor(factor, poses_menos)
            diferencia = r_mas - r_menos
            diferencia[2] = float(normalizar_angulo(diferencia[2]))
            jacobiano[:, columna] = diferencia / (2.0 * epsilon)
        jacobianos[nombre] = jacobiano
    return jacobianos


def calcular_norma_mahalanobis(residuo, informacion):
    """Calcula la norma de Mahalanobis de un residuo."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)
    valor = float(residuo.T @ informacion @ residuo)
    return float(np.sqrt(max(valor, 0.0)))


def calcular_peso_huber(norma, delta=DELTA_HUBER):
    """Calcula el peso IRLS de Huber."""

    norma = float(abs(norma))
    if norma <= float(delta) or norma <= 1e-15:
        return 1.0
    return float(delta / norma)


def calcular_coste_huber(norma, delta=DELTA_HUBER):
    """Calcula la pérdida de Huber para una norma no negativa."""

    norma = float(abs(norma))
    delta = float(delta)
    if norma <= delta:
        return 0.5 * norma**2
    return delta * (norma - 0.5 * delta)


def ensamblar_sistema_global(grafo, estimaciones, robusto=True):
    """Ensambla residuo, Jacobiano, Hessiana, gradiente y costes del grafo."""

    estimaciones = validar_trayectoria(
        estimaciones, len(grafo.graph["node_order"]), "estimaciones"
    )
    nombres = grafo.graph["node_order"]
    indices = {nombre: 3 * indice for indice, nombre in enumerate(nombres)}
    poses = {nombre: estimaciones[indice].copy() for indice, nombre in enumerate(nombres)}
    filas = 3 * len(grafo.graph["factor_order"])
    columnas = 3 * len(nombres)
    residuo_global = np.zeros(filas, dtype=float)
    jacobiano_global = np.zeros((filas, columnas), dtype=float)
    informacion_global = np.zeros((filas, filas), dtype=float)
    pesos = {}
    mahalanobis = {}
    coste_por_tipo = {"prior": 0.0, "odometry": 0.0, "inter_robot": 0.0}
    coste_total = 0.0

    for indice_factor, nombre_factor in enumerate(grafo.graph["factor_order"]):
        factor = grafo.graph["factors"][nombre_factor]
        residuo = calcular_residuo_factor(factor, poses)
        jacobianos = calcular_jacobianos_numericos_factor(factor, poses)
        norma = calcular_norma_mahalanobis(residuo, factor["information"])
        kernel = factor.get("robust_kernel")
        peso = 1.0
        if robusto and kernel is not None and kernel.get("type") == "huber":
            peso = calcular_peso_huber(norma, kernel.get("delta", DELTA_HUBER))
            coste_factor = calcular_coste_huber(norma, kernel.get("delta", DELTA_HUBER))
        else:
            coste_factor = 0.5 * norma**2

        fila = slice(3 * indice_factor, 3 * indice_factor + 3)
        residuo_global[fila] = residuo
        informacion_global[fila, fila] = peso * factor["information"]
        for nombre_variable, jacobiano in jacobianos.items():
            columna = slice(indices[nombre_variable], indices[nombre_variable] + 3)
            jacobiano_global[fila, columna] = jacobiano

        pesos[nombre_factor] = float(peso)
        mahalanobis[nombre_factor] = float(norma)
        coste_por_tipo[factor["factor_type"]] += float(coste_factor)
        coste_total += float(coste_factor)

    hessiana = jacobiano_global.T @ informacion_global @ jacobiano_global
    gradiente = jacobiano_global.T @ informacion_global @ residuo_global

    return {
        "residual": residuo_global,
        "jacobian": jacobiano_global,
        "information": informacion_global,
        "hessian": hessiana,
        "gradient": gradiente,
        "weights": pesos,
        "mahalanobis": mahalanobis,
        "cost": float(coste_total),
        "cost_by_type": coste_por_tipo,
    }


def aplicar_incremento_global(estimaciones, incremento):
    """Aplica un vector de incrementos locales a todas las poses."""

    estimaciones = validar_trayectoria(estimaciones)
    incremento = np.asarray(incremento, dtype=float)
    if incremento.shape != (3 * len(estimaciones),):
        raise ValueError("El incremento global tiene una dimensión incorrecta.")
    actualizadas = []
    for indice, pose in enumerate(estimaciones):
        actualizadas.append(
            aplicar_incremento_local(pose, incremento[3 * indice:3 * indice + 3])
        )
    return validar_trayectoria(np.asarray(actualizadas, dtype=float))


def calcular_metricas_multi_robot(estimaciones, trayectoria_a_real, trayectoria_b_real):
    """Calcula RMSE global y métricas separadas para ambos robots."""

    estimaciones = validar_trayectoria(
        estimaciones, NUM_POSES_A + NUM_POSES_B, "estimaciones"
    )
    real = np.vstack(
        [validar_trayectoria(trayectoria_a_real), validar_trayectoria(trayectoria_b_real)]
    )
    error_posicion = np.linalg.norm(estimaciones[:, :2] - real[:, :2], axis=1)
    error_angular = normalizar_angulo(estimaciones[:, 2] - real[:, 2])
    error_a = error_posicion[:NUM_POSES_A]
    error_b = error_posicion[NUM_POSES_A:]
    ang_a = error_angular[:NUM_POSES_A]
    ang_b = error_angular[NUM_POSES_A:]
    return {
        "position_rmse": float(np.sqrt(np.mean(error_posicion**2))),
        "position_rmse_a": float(np.sqrt(np.mean(error_a**2))),
        "position_rmse_b": float(np.sqrt(np.mean(error_b**2))),
        "position_max": float(np.max(error_posicion)),
        "orientation_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error_angular**2)))),
        "orientation_rmse_a_deg": float(np.rad2deg(np.sqrt(np.mean(ang_a**2)))),
        "orientation_rmse_b_deg": float(np.rad2deg(np.sqrt(np.mean(ang_b**2)))),
    }


def calcular_error_transformacion(transformacion_estimada, transformacion_real):
    """Compara una transformación estimada con la transformación real."""

    error = calcular_movimiento_relativo(transformacion_real, transformacion_estimada)
    return {
        "translation": float(np.linalg.norm(error[:2])),
        "angle_deg": float(abs(np.rad2deg(normalizar_angulo(error[2])))),
        "relative_error": error,
    }


def optimizar_grafo_multi_robot(grafo, estimaciones_iniciales, trayectoria_a_real, trayectoria_b_real):
    """Optimiza el pose graph global mediante Levenberg-Marquardt robusto."""

    estimaciones = validar_trayectoria(
        estimaciones_iniciales, NUM_POSES_A + NUM_POSES_B, "estimaciones iniciales"
    )
    damping = DAMPING_INICIAL
    historial = []
    convergio = False

    for iteracion in range(MAX_ITERACIONES_LM):
        sistema_antes = ensamblar_sistema_global(grafo, estimaciones, robusto=True)
        h = sistema_antes["hessian"]
        g = sistema_antes["gradient"]
        diagonal = np.maximum(np.diag(h), 1e-9)
        aceptado = False
        estimaciones_candidatas = estimaciones.copy()
        sistema_despues = sistema_antes
        incremento = np.zeros(3 * len(estimaciones), dtype=float)

        for _ in range(12):
            h_lm = h + damping * np.diag(diagonal)
            try:
                incremento = np.linalg.solve(h_lm, -g)
            except np.linalg.LinAlgError:
                incremento = np.linalg.lstsq(h_lm, -g, rcond=None)[0]
            estimaciones_candidatas = aplicar_incremento_global(estimaciones, incremento)
            sistema_despues = ensamblar_sistema_global(
                grafo, estimaciones_candidatas, robusto=True
            )
            if sistema_despues["cost"] < sistema_antes["cost"]:
                aceptado = True
                break
            damping *= 8.0

        metricas_antes = calcular_metricas_multi_robot(
            estimaciones, trayectoria_a_real, trayectoria_b_real
        )
        metricas_despues = calcular_metricas_multi_robot(
            estimaciones_candidatas, trayectoria_a_real, trayectoria_b_real
        )
        transformacion_b_antes = estimaciones[NUM_POSES_A]
        transformacion_b_despues = estimaciones_candidatas[NUM_POSES_A]

        historial.append(
            {
                "iteration": iteracion,
                "accepted": aceptado,
                "cost_before": sistema_antes["cost"],
                "cost_after": sistema_despues["cost"],
                "damping_before": damping,
                "damping_after": damping * 0.35 if aceptado else damping,
                "step_norm": float(np.linalg.norm(incremento)),
                "estimates_before": estimaciones.copy(),
                "estimates_after": estimaciones_candidatas.copy(),
                "metrics_before": metricas_antes,
                "metrics_after": metricas_despues,
                "transform_b_before": transformacion_b_antes.copy(),
                "transform_b_after": transformacion_b_despues.copy(),
                "weights_after": dict(sistema_despues["weights"]),
                "mahalanobis_after": dict(sistema_despues["mahalanobis"]),
            }
        )

        if not aceptado:
            break

        estimaciones = estimaciones_candidatas
        damping = max(damping * 0.35, 1e-9)
        if np.linalg.norm(incremento) < TOLERANCIA_PASO:
            convergio = True
            break

    sistema_final = ensamblar_sistema_global(grafo, estimaciones, robusto=True)
    if historial and historial[-1]["accepted"]:
        convergio = convergio or historial[-1]["step_norm"] < 1e-6

    return {
        "estimates": estimaciones,
        "history": historial,
        "iterations": len(historial),
        "converged": bool(convergio),
        "final_system": sistema_final,
    }


def analizar_restriccion_falsa(grafo, estimaciones, especificacion_falsa):
    """Evalúa el outlier que habría creado el alias perceptual rechazado."""

    cov = crear_covarianza_diagonal(SIGMAS_INTER_ROBOT_FALSA)
    factor = {
        "factor_name": especificacion_falsa["factor_name"],
        "factor_type": "inter_robot",
        "variables": (especificacion_falsa["node_i"], especificacion_falsa["node_j"]),
        "measurement": especificacion_falsa["measurement"],
        "covariance": cov,
        "information": calcular_matriz_informacion(cov),
        "robust_kernel": {"type": "huber", "delta": DELTA_HUBER},
        "is_inter_robot": True,
        "is_false": True,
    }
    poses = {
        nombre: estimaciones[indice].copy()
        for indice, nombre in enumerate(grafo.graph["node_order"])
    }
    residuo = calcular_residuo_factor(factor, poses)
    norma = calcular_norma_mahalanobis(residuo, factor["information"])
    return {
        "factor": factor,
        "residual": residuo,
        "mahalanobis": norma,
        "huber_weight": calcular_peso_huber(norma, DELTA_HUBER),
        "quadratic_cost": 0.5 * norma**2,
        "robust_cost": calcular_coste_huber(norma, DELTA_HUBER),
    }


def crear_resultado_multi_robot_slam():
    """Construye el escenario, verifica encuentros, fusiona grafos y optimiza."""

    trayectoria_a_global, trayectoria_b_global = crear_trayectorias_globales_reales()
    locales = expresar_trayectorias_en_marcos_locales(
        trayectoria_a_global, trayectoria_b_global
    )
    trayectoria_a_local_real = locales["trajectory_a_local"]
    trayectoria_b_local_real = locales["trajectory_b_local"]
    transformacion_real = locales["transform_a_b_true"]

    odometria_a = crear_mediciones_odometria(trayectoria_a_local_real, "A")
    odometria_b = crear_mediciones_odometria(trayectoria_b_local_real, "B")
    trayectoria_a_local_inicial = integrar_odometria(odometria_a)
    trayectoria_b_local_inicial = integrar_odometria(odometria_b)

    grafo_a, grafo_b = crear_grafos_locales(
        trayectoria_a_local_inicial,
        trayectoria_b_local_inicial,
        odometria_a,
        odometria_b,
    )

    lugares = crear_lugares_compartidos(
        trayectoria_a_local_inicial, trayectoria_b_local_inicial
    )
    ransac = verificar_transformacion_ransac(lugares)
    transformacion_inicial = ransac["transform"]
    trayectoria_b_alineada_inicial = transformar_trayectoria(
        trayectoria_b_local_inicial, transformacion_inicial
    )
    trayectoria_a_global_inicial = transformar_trayectoria(
        trayectoria_a_local_inicial, locales["transform_global_a"]
    )

    mediciones_inter = crear_mediciones_inter_robot(
        trayectoria_a_global, trayectoria_b_global, lugares, ransac
    )
    lugar_falso = next(lugar for lugar in lugares if not lugar["is_true"])
    medicion_falsa = crear_medicion_inter_robot_falsa(
        trayectoria_a_global, trayectoria_b_global, lugar_falso
    )

    grafo_global = crear_grafo_global(
        trayectoria_a_global_inicial,
        trayectoria_b_alineada_inicial,
        odometria_a,
        odometria_b,
        mediciones_inter,
    )
    estimaciones_iniciales = obtener_estimaciones(grafo_global)
    metricas_iniciales = calcular_metricas_multi_robot(
        estimaciones_iniciales, trayectoria_a_global, trayectoria_b_global
    )
    error_transformacion_inicial = calcular_error_transformacion(
        transformacion_inicial, transformacion_real
    )

    optimizacion = optimizar_grafo_multi_robot(
        grafo_global,
        estimaciones_iniciales,
        trayectoria_a_global,
        trayectoria_b_global,
    )
    estimaciones_finales = optimizacion["estimates"]
    actualizar_estimaciones_grafo(grafo_global, estimaciones_finales)
    trayectoria_a_final = estimaciones_finales[:NUM_POSES_A]
    trayectoria_b_final = estimaciones_finales[NUM_POSES_A:]
    transformacion_final = trayectoria_b_final[0]
    metricas_finales = calcular_metricas_multi_robot(
        estimaciones_finales, trayectoria_a_global, trayectoria_b_global
    )
    error_transformacion_final = calcular_error_transformacion(
        transformacion_final, transformacion_real
    )
    falsa = analizar_restriccion_falsa(
        grafo_global, estimaciones_finales, medicion_falsa
    )

    grafo_separado = nx.disjoint_union(grafo_a, grafo_b)
    componentes_antes = nx.number_connected_components(grafo_separado)
    componentes_despues = nx.number_connected_components(grafo_global)

    return {
        "true_trajectory_a_global": trayectoria_a_global,
        "true_trajectory_b_global": trayectoria_b_global,
        "true_trajectory_a_local": trayectoria_a_local_real,
        "true_trajectory_b_local": trayectoria_b_local_real,
        "initial_trajectory_a_local": trayectoria_a_local_inicial,
        "initial_trajectory_b_local": trayectoria_b_local_inicial,
        "initial_trajectory_a_global": trayectoria_a_global_inicial,
        "initial_trajectory_b_aligned": trayectoria_b_alineada_inicial,
        "optimized_trajectory_a": trayectoria_a_final,
        "optimized_trajectory_b": trayectoria_b_final,
        "transform_a_b_true": transformacion_real,
        "transform_a_b_initial": transformacion_inicial,
        "transform_a_b_final": transformacion_final,
        "transform_error_initial": error_transformacion_inicial,
        "transform_error_final": error_transformacion_final,
        "odometry_a": odometria_a,
        "odometry_b": odometria_b,
        "local_graph_a": grafo_a,
        "local_graph_b": grafo_b,
        "global_graph": grafo_global,
        "shared_places": lugares,
        "ransac": ransac,
        "inter_robot_measurements": mediciones_inter,
        "false_inter_robot_measurement": medicion_falsa,
        "false_constraint_analysis": falsa,
        "initial_estimates": estimaciones_iniciales,
        "optimized_estimates": estimaciones_finales,
        "initial_metrics": metricas_iniciales,
        "final_metrics": metricas_finales,
        "optimization": optimizacion,
        "components_before": componentes_antes,
        "components_after": componentes_despues,
        "parameters": {
            "ransac_threshold": UMBRAL_RANSAC,
            "min_ransac_inliers": MIN_INLIERS_RANSAC,
            "huber_delta": DELTA_HUBER,
            "sigmas_prior": SIGMAS_PRIOR.copy(),
            "sigmas_odometry": SIGMAS_ODOMETRIA.copy(),
            "sigmas_inter_robot": SIGMAS_INTER_ROBOT.copy(),
        },
    }


def interpolar_trayectorias(trayectoria_a, trayectoria_b, alpha):
    """Interpola dos trayectorias y trata los ángulos por el camino corto."""

    trayectoria_a = validar_trayectoria(trayectoria_a)
    trayectoria_b = validar_trayectoria(trayectoria_b, len(trayectoria_a))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    salida = (1.0 - alpha) * trayectoria_a + alpha * trayectoria_b
    diferencia_angular = normalizar_angulo(trayectoria_b[:, 2] - trayectoria_a[:, 2])
    salida[:, 2] = normalizar_angulo(trayectoria_a[:, 2] + alpha * diferencia_angular)
    return salida


def crear_estado_animacion(phase, message, **kwargs):
    """Crea un estado homogéneo para la demostración visual."""

    estado = {
        "phase": str(phase),
        "message": str(message),
        "show_a_count": 0,
        "show_b_count": 0,
        "candidate_count": 0,
        "ransac_hypothesis": None,
        "show_ransac_result": False,
        "alignment_alpha": 0.0,
        "inter_count": 0,
        "trajectory_a": None,
        "trajectory_b": None,
        "iteration": None,
        "cost": None,
        "rmse": None,
        "transform_error": None,
        "show_summary": False,
    }
    estado.update(kwargs)
    return estado


def crear_estados_animacion(resultado):
    """Crea la secuencia completa de mapas separados, encuentro y fusión."""

    estados = []

    def añadir(estado, repeticiones=1):
        for _ in range(repeticiones):
            estados.append(dict(estado))

    añadir(
        crear_estado_animacion(
            "introduccion",
            "Dos robots construyen mapas locales en sistemas de coordenadas distintos.",
            show_a_count=1,
            show_b_count=1,
        ),
        3,
    )

    for cantidad in range(1, NUM_POSES_A + 1):
        añadir(
            crear_estado_animacion(
                "mapa_A",
                f"Robot A incorpora A_x{cantidad - 1} y su odometría local.",
                show_a_count=cantidad,
                show_b_count=1,
            )
        )

    for cantidad in range(1, NUM_POSES_B + 1):
        añadir(
            crear_estado_animacion(
                "mapa_B",
                f"Robot B incorpora B_x{cantidad - 1} en el marco W_B.",
                show_a_count=NUM_POSES_A,
                show_b_count=cantidad,
            )
        )

    añadir(
        crear_estado_animacion(
            "mapas_separados",
            "Los dos pose graphs son conexos internamente, pero el conjunto tiene dos componentes.",
            show_a_count=NUM_POSES_A,
            show_b_count=NUM_POSES_B,
        ),
        4,
    )

    for cantidad in range(1, len(resultado["shared_places"]) + 1):
        añadir(
            crear_estado_animacion(
                "candidatos",
                "Los descriptores proponen lugares comunes; todavía son hipótesis.",
                show_a_count=NUM_POSES_A,
                show_b_count=NUM_POSES_B,
                candidate_count=cantidad,
            ),
            2,
        )

    for indice in range(len(resultado["ransac"]["history"])):
        añadir(
            crear_estado_animacion(
                "ransac",
                f"RANSAC prueba la hipótesis mínima {indice + 1}.",
                show_a_count=NUM_POSES_A,
                show_b_count=NUM_POSES_B,
                candidate_count=len(resultado["shared_places"]),
                ransac_hypothesis=indice,
            )
        )

    añadir(
        crear_estado_animacion(
            "ransac_resultado",
            "Tres correspondencias son coherentes y el alias perceptual queda rechazado.",
            show_a_count=NUM_POSES_A,
            show_b_count=NUM_POSES_B,
            candidate_count=len(resultado["shared_places"]),
            show_ransac_result=True,
        ),
        4,
    )

    for alpha in np.linspace(0.0, 1.0, 13):
        añadir(
            crear_estado_animacion(
                "alineacion_inicial",
                "La transformación estimada expresa el mapa B en el marco W_A.",
                show_a_count=NUM_POSES_A,
                show_b_count=NUM_POSES_B,
                candidate_count=len(resultado["shared_places"]),
                show_ransac_result=True,
                alignment_alpha=float(alpha),
            )
        )

    for cantidad in range(1, len(resultado["inter_robot_measurements"]) + 1):
        añadir(
            crear_estado_animacion(
                "restricciones_inter_robot",
                f"Se añade la restricción inter-robot verificada {cantidad}.",
                show_a_count=NUM_POSES_A,
                show_b_count=NUM_POSES_B,
                candidate_count=len(resultado["shared_places"]),
                show_ransac_result=True,
                alignment_alpha=1.0,
                inter_count=cantidad,
            ),
            3,
        )

    añadir(
        crear_estado_animacion(
            "grafo_conectado",
            "Las restricciones inter-robot convierten dos componentes en un único grafo global.",
            show_a_count=NUM_POSES_A,
            show_b_count=NUM_POSES_B,
            candidate_count=len(resultado["shared_places"]),
            show_ransac_result=True,
            alignment_alpha=1.0,
            inter_count=len(resultado["inter_robot_measurements"]),
            trajectory_a=resultado["initial_trajectory_a_global"],
            trajectory_b=resultado["initial_trajectory_b_aligned"],
            cost=resultado["optimization"]["history"][0]["cost_before"],
            rmse=resultado["initial_metrics"]["position_rmse"],
            transform_error=resultado["transform_error_initial"]["translation"],
        ),
        4,
    )

    for entrada in resultado["optimization"]["history"]:
        for alpha in np.linspace(0.0, 1.0, 4, endpoint=False):
            estimaciones = interpolar_trayectorias(
                entrada["estimates_before"], entrada["estimates_after"], alpha
            )
            metricas = calcular_metricas_multi_robot(
                estimaciones,
                resultado["true_trajectory_a_global"],
                resultado["true_trajectory_b_global"],
            )
            transform_error = calcular_error_transformacion(
                estimaciones[NUM_POSES_A], resultado["transform_a_b_true"]
            )
            añadir(
                crear_estado_animacion(
                    "optimizacion",
                    f"LM ajusta simultáneamente las poses de A y B · iteración {entrada['iteration'] + 1}.",
                    show_a_count=NUM_POSES_A,
                    show_b_count=NUM_POSES_B,
                    candidate_count=len(resultado["shared_places"]),
                    show_ransac_result=True,
                    alignment_alpha=1.0,
                    inter_count=len(resultado["inter_robot_measurements"]),
                    trajectory_a=estimaciones[:NUM_POSES_A],
                    trajectory_b=estimaciones[NUM_POSES_A:],
                    iteration=entrada["iteration"] + 1,
                    cost=(1.0 - alpha) * entrada["cost_before"] + alpha * entrada["cost_after"],
                    rmse=metricas["position_rmse"],
                    transform_error=transform_error["translation"],
                )
            )

    añadir(
        crear_estado_animacion(
            "resultado_final",
            "Mapa fusionado: ambos robots quedan expresados y optimizados en W_A.",
            show_a_count=NUM_POSES_A,
            show_b_count=NUM_POSES_B,
            candidate_count=len(resultado["shared_places"]),
            show_ransac_result=True,
            alignment_alpha=1.0,
            inter_count=len(resultado["inter_robot_measurements"]),
            trajectory_a=resultado["optimized_trajectory_a"],
            trajectory_b=resultado["optimized_trajectory_b"],
            iteration=resultado["optimization"]["iterations"],
            cost=resultado["optimization"]["final_system"]["cost"],
            rmse=resultado["final_metrics"]["position_rmse"],
            transform_error=resultado["transform_error_final"]["translation"],
            show_summary=True,
        ),
        8,
    )

    total = len(estados)
    for indice, estado in enumerate(estados, start=1):
        estado["step"] = indice
        estado["total_steps"] = total

    return estados


def validar_transformaciones(resultado):
    """Valida marcos locales, transformación real y alineación estimada."""

    identidad = componer_poses_se2(
        resultado["transform_a_b_true"], invertir_pose_se2(resultado["transform_a_b_true"])
    )
    if not np.allclose(identidad, np.zeros(3), atol=1e-9):
        raise AssertionError("La transformación y su inversa no producen identidad.")
    b_recuperada = transformar_trayectoria(
        resultado["true_trajectory_b_local"], resultado["transform_a_b_true"]
    )
    if not np.allclose(b_recuperada, resultado["true_trajectory_b_global"], atol=1e-9):
        raise AssertionError("La transformación real no recupera el mapa global B.")
    if resultado["transform_error_initial"]["translation"] >= 0.50:
        raise AssertionError("La alineación inicial es demasiado imprecisa.")


def validar_ransac(resultado):
    """Valida que RANSAC acepte las tres correspondencias correctas y rechace el alias."""

    ransac = resultado["ransac"]
    if ransac["inlier_count"] != 3 or ransac["outlier_count"] != 1:
        raise AssertionError("RANSAC debe producir tres inliers y un outlier.")
    for lugar, inlier in zip(resultado["shared_places"], ransac["inliers"]):
        if bool(inlier) != bool(lugar["is_true"]):
            raise AssertionError("La clasificación RANSAC no coincide con la verdad.")
    if not ransac["accepted"]:
        raise AssertionError("La transformación inter-robot debe aceptarse.")


def validar_grafos(resultado):
    """Valida identificadores, conectividad y factores del grafo global."""

    if resultado["components_before"] != 2:
        raise AssertionError("Antes de fusionar deben existir dos componentes.")
    if resultado["components_after"] != 1:
        raise AssertionError("Después de fusionar debe existir una componente.")
    grafo = resultado["global_graph"]
    if grafo.number_of_nodes() != NUM_POSES_A + NUM_POSES_B:
        raise AssertionError("Número incorrecto de nodos globales.")
    if len(grafo.graph["factor_order"]) != 1 + 18 + 3:
        raise AssertionError("Número incorrecto de factores globales.")
    if len(set(grafo.graph["node_order"])) != len(grafo.graph["node_order"]):
        raise AssertionError("Los identificadores globales deben ser únicos.")
    if not all(nombre.startswith(("A_x", "B_x")) for nombre in grafo.nodes):
        raise AssertionError("Cada pose debe conservar el prefijo de robot.")


def validar_sistema_global(resultado):
    """Valida dimensiones, simetría y valores finitos del sistema optimizado."""

    sistema = resultado["optimization"]["final_system"]
    dimension = 3 * (NUM_POSES_A + NUM_POSES_B)
    filas = 3 * (1 + 18 + 3)
    if sistema["jacobian"].shape != (filas, dimension):
        raise AssertionError("Forma incorrecta del Jacobiano global.")
    if sistema["hessian"].shape != (dimension, dimension):
        raise AssertionError("Forma incorrecta de la Hessiana global.")
    if not np.allclose(sistema["hessian"], sistema["hessian"].T, atol=1e-8):
        raise AssertionError("La Hessiana global debe ser simétrica.")
    for clave in ("residual", "jacobian", "information", "hessian", "gradient"):
        if not np.all(np.isfinite(sistema[clave])):
            raise AssertionError(f"El bloque {clave} contiene valores no finitos.")


def validar_optimizacion(resultado):
    """Valida convergencia y mejora de la fusión global."""

    historial = resultado["optimization"]["history"]
    if not historial:
        raise AssertionError("La optimización debe registrar iteraciones.")
    if not all(entrada["cost_after"] <= entrada["cost_before"] + 1e-10 for entrada in historial if entrada["accepted"]):
        raise AssertionError("Cada paso aceptado debe reducir el coste.")
    if resultado["final_metrics"]["position_rmse"] >= resultado["initial_metrics"]["position_rmse"]:
        raise AssertionError("La optimización debe reducir el RMSE global.")
    if resultado["transform_error_final"]["translation"] >= resultado["transform_error_initial"]["translation"]:
        raise AssertionError("La transformación final debe mejorar la inicial.")
    if resultado["final_metrics"]["position_rmse"] >= 0.18:
        raise AssertionError("El RMSE final debe ser inferior a 0.18 m.")


def validar_restriccion_falsa(resultado):
    """Valida que el alias rechazado produciría un factor claramente atípico."""

    falsa = resultado["false_constraint_analysis"]
    if falsa["mahalanobis"] <= 20.0:
        raise AssertionError("La falsa restricción debe tener Mahalanobis elevado.")
    if falsa["huber_weight"] >= 0.15:
        raise AssertionError("Huber debería reducir fuertemente la falsa restricción.")


def validar_resultados(resultado, estados):
    """Ejecuta todas las validaciones y devuelve un resumen numérico."""

    validar_transformaciones(resultado)
    validar_ransac(resultado)
    validar_grafos(resultado)
    validar_sistema_global(resultado)
    validar_optimizacion(resultado)
    validar_restriccion_falsa(resultado)
    if len(estados) < 70:
        raise AssertionError("La demostración debe contener al menos 70 estados.")

    return {
        "robot_count": 2,
        "pose_count_a": NUM_POSES_A,
        "pose_count_b": NUM_POSES_B,
        "local_factor_count_a": resultado["local_graph_a"].number_of_edges(),
        "local_factor_count_b": resultado["local_graph_b"].number_of_edges(),
        "inter_candidate_count": len(resultado["shared_places"]),
        "inter_accepted_count": len(resultado["inter_robot_measurements"]),
        "ransac_inliers": resultado["ransac"]["inlier_count"],
        "ransac_outliers": resultado["ransac"]["outlier_count"],
        "ransac_rmse": resultado["ransac"]["rmse"],
        "components_before": resultado["components_before"],
        "components_after": resultado["components_after"],
        "initial_transform_translation_error": resultado["transform_error_initial"]["translation"],
        "initial_transform_angle_error_deg": resultado["transform_error_initial"]["angle_deg"],
        "final_transform_translation_error": resultado["transform_error_final"]["translation"],
        "final_transform_angle_error_deg": resultado["transform_error_final"]["angle_deg"],
        "initial_rmse": resultado["initial_metrics"]["position_rmse"],
        "final_rmse": resultado["final_metrics"]["position_rmse"],
        "initial_angle_rmse_deg": resultado["initial_metrics"]["orientation_rmse_deg"],
        "final_angle_rmse_deg": resultado["final_metrics"]["orientation_rmse_deg"],
        "initial_cost": resultado["optimization"]["history"][0]["cost_before"],
        "final_cost": resultado["optimization"]["final_system"]["cost"],
        "optimization_iterations": resultado["optimization"]["iterations"],
        "optimization_converged": resultado["optimization"]["converged"],
        "false_constraint_mahalanobis": resultado["false_constraint_analysis"]["mahalanobis"],
        "false_constraint_huber_weight": resultado["false_constraint_analysis"]["huber_weight"],
        "jacobian_shape": resultado["optimization"]["final_system"]["jacobian"].shape,
        "hessian_shape": resultado["optimization"]["final_system"]["hessian"].shape,
        "state_count": len(estados),
    }


def main():
    """Ejecuta el ejemplo determinista y lanza la representación."""

    resultado = crear_resultado_multi_robot_slam()
    estados = crear_estados_animacion(resultado)
    resumen = validar_resultados(resultado, estados)

    print("\nSLAM multi-robot")
    print("----------------")
    for clave, valor in resumen.items():
        print(f"{clave}: {valor}")

    animator = GraphAnimator(figsize=(18, 9.5), interval=320)
    image_path = PROJECT_ROOT / "assets" / "06_graph_slam" / "08_multi_robot_slam.png"
    animator.animate_multi_robot_slam(
        result=resultado,
        states=estados,
        title="SLAM multi-robot: mapas separados, encuentros y fusión",
        final_image_path=image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
