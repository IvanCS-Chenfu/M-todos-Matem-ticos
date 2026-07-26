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

NUMERO_POSES = 14
LANDMARKS_CONOCIDOS = ("l0", "l1")
LANDMARKS_DESCONOCIDOS = ("l2", "l3", "l4", "l5")
CAMPO_VISION_GRADOS = 250.0
ALCANCE_MINIMO = 0.35
ALCANCE_MAXIMO = 5.8

EPSILON_JACOBIANO = 1e-7
MAX_ITERACIONES = 40
TOLERANCIA_INCREMENTO = 1e-9
TOLERANCIA_COSTE_RELATIVO = 1e-11
LAMBDA_INICIAL = 1e-3
DELTA_HUBER_OBSERVACION = 6.0

SIGMAS_PRIOR_POSE = np.array(
    [0.020, 0.020, np.deg2rad(0.45)], dtype=float
)
SIGMAS_ODOMETRIA = np.array(
    [0.085, 0.075, np.deg2rad(1.8)], dtype=float
)
SIGMAS_OBSERVACION = np.array([0.075, 0.075], dtype=float)

SESGO_ESCALA = 1.012
SESGO_LATERAL = 0.008
SESGO_ANGULAR_GRADOS = 0.24


# ---------------------------------------------------------------------------
# Operaciones geométricas y validación
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


def validar_landmarks(landmarks, nombre="landmarks"):
    """Valida un diccionario nombre -> posición 2D."""

    if not isinstance(landmarks, dict) or not landmarks:
        raise ValueError(f"{nombre} debe ser un diccionario no vacío.")
    return {
        str(clave): validar_landmark(valor, f"{nombre}[{clave!r}]")
        for clave, valor in landmarks.items()
    }


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


def aplicar_incremento_pose(pose, incremento):
    """Aplica una perturbación local por la derecha a una pose."""

    return componer_poses_se2(
        validar_pose(pose),
        validar_pose(incremento, "incremento de pose"),
    )


def aplicar_incremento_landmark(landmark, incremento):
    """Aplica un incremento euclídeo a un landmark 2D."""

    landmark = validar_landmark(landmark)
    incremento = validar_landmark(incremento, "incremento de landmark")
    return validar_landmark(landmark + incremento)


def rotacion_2d(theta):
    """Devuelve una matriz de rotación plana."""

    theta = normalizar_angulo(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def crear_covarianza_diagonal(sigmas):
    """Crea una covarianza diagonal desde desviaciones estándar."""

    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.ndim != 1 or len(sigmas) == 0:
        raise ValueError("Los sigmas deben formar un vector no vacío.")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("Los sigmas deben ser positivos y finitos.")
    return np.diag(sigmas**2)


def calcular_matriz_informacion(covarianza):
    """Valida e invierte una matriz de covarianza."""

    covarianza = np.asarray(covarianza, dtype=float)
    if covarianza.ndim != 2 or covarianza.shape[0] != covarianza.shape[1]:
        raise ValueError("La covarianza debe ser cuadrada.")
    if not np.all(np.isfinite(covarianza)):
        raise ValueError("La covarianza debe contener valores finitos.")
    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise ValueError("La covarianza debe ser simétrica.")
    if np.min(np.linalg.eigvalsh(covarianza)) <= 0.0:
        raise ValueError("La covarianza debe ser definida positiva.")
    return np.linalg.inv(covarianza)


# ---------------------------------------------------------------------------
# Trayectoria, landmarks y observaciones
# ---------------------------------------------------------------------------


def crear_trayectoria_real(numero_poses=NUMERO_POSES):
    """Crea una trayectoria suave y abierta para el robot."""

    numero_poses = int(numero_poses)
    if numero_poses < 10:
        raise ValueError("Se requieren al menos diez poses.")

    t = np.linspace(0.0, 1.0, numero_poses, dtype=float)
    x = -4.4 + 8.8 * t
    y = 0.95 * np.sin(2.0 * np.pi * t) + 0.28 * np.sin(4.0 * np.pi * t)

    dx = np.gradient(x)
    dy = np.gradient(y)
    theta = np.arctan2(dy, dx)

    return validar_trayectoria(
        np.column_stack((x, y, theta)),
        "trayectoria real",
    )


def crear_mediciones_odometria(trayectoria_real):
    """Genera odometría determinista con deriva moderada."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    ideales = []
    medidas = []

    for indice in range(1, len(trayectoria_real)):
        ideal = calcular_movimiento_relativo(
            trayectoria_real[indice - 1],
            trayectoria_real[indice],
        )

        escala = SESGO_ESCALA + 0.003 * np.sin(0.41 * indice)
        error_longitudinal = 0.005 * np.cos(0.33 * indice)
        error_lateral = SESGO_LATERAL + 0.004 * np.sin(0.57 * indice)
        error_angular = np.deg2rad(
            SESGO_ANGULAR_GRADOS
            + 0.09 * np.sin(0.37 * indice)
            + 0.04 * np.cos(0.23 * indice)
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


def crear_landmarks_reales():
    """Crea dos referencias conocidas y cuatro landmarks desconocidos."""

    return validar_landmarks(
        {
            "l0": np.array([-3.25, 2.15], dtype=float),
            "l1": np.array([3.05, -2.05], dtype=float),
            "l2": np.array([-3.55, -1.75], dtype=float),
            "l3": np.array([-0.85, 2.35], dtype=float),
            "l4": np.array([1.35, -2.20], dtype=float),
            "l5": np.array([3.75, 1.95], dtype=float),
        },
        "landmarks reales",
    )


def predecir_observacion_cartesiana(pose, landmark):
    """Predice la posición local 2D de un landmark desde una pose."""

    pose = validar_pose(pose)
    landmark = validar_landmark(landmark)
    return rotacion_2d(pose[2]).T @ (landmark - pose[:2])


def calcular_rango_rumbo(pose, landmark):
    """Calcula rango y rumbo de un landmark respecto a una pose."""

    local = predecir_observacion_cartesiana(pose, landmark)
    rango = float(np.linalg.norm(local))
    rumbo = normalizar_angulo(np.arctan2(local[1], local[0]))
    return rango, rumbo


def landmark_visible(
    pose,
    landmark,
    campo_vision_grados=CAMPO_VISION_GRADOS,
    alcance_minimo=ALCANCE_MINIMO,
    alcance_maximo=ALCANCE_MAXIMO,
):
    """Comprueba alcance y campo de visión para una observación."""

    rango, rumbo = calcular_rango_rumbo(pose, landmark)
    semiangulo = np.deg2rad(float(campo_vision_grados)) / 2.0
    visible = (
        float(alcance_minimo) <= rango <= float(alcance_maximo)
        and abs(rumbo) <= semiangulo
    )
    return {
        "visible": bool(visible),
        "range": rango,
        "bearing": rumbo,
        "bearing_deg": float(np.rad2deg(rumbo)),
    }


def crear_observaciones_landmarks(trayectoria_real, landmarks_reales):
    """Genera observaciones cartesianas locales deterministas."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    landmarks_reales = validar_landmarks(landmarks_reales, "landmarks reales")
    covarianza = crear_covarianza_diagonal(SIGMAS_OBSERVACION)
    informacion = calcular_matriz_informacion(covarianza)

    observaciones = []
    for indice_pose, pose in enumerate(trayectoria_real):
        for nombre_landmark, landmark in landmarks_reales.items():
            visibilidad = landmark_visible(pose, landmark)
            if not visibilidad["visible"]:
                continue

            ideal = predecir_observacion_cartesiana(pose, landmark)
            indice_landmark = int(nombre_landmark[1:])
            ruido = np.array(
                [
                    0.018 * np.sin(0.47 * (indice_pose + 1) * (indice_landmark + 1))
                    + 0.006 * np.cos(0.19 * (indice_pose + 2)),
                    0.016 * np.cos(0.39 * (indice_pose + 1) * (indice_landmark + 2))
                    - 0.005 * np.sin(0.23 * (indice_pose + 1)),
                ],
                dtype=float,
            )
            medida = validar_landmark(
                ideal + ruido,
                f"observación x{indice_pose}-{nombre_landmark}",
            )

            observaciones.append(
                {
                    "factor_name": f"obs_{indice_pose}_{nombre_landmark}",
                    "factor_type": "landmark_observation",
                    "pose_name": f"x{indice_pose}",
                    "landmark_name": nombre_landmark,
                    "variables": (f"x{indice_pose}", nombre_landmark),
                    "measurement": medida,
                    "true_measurement": ideal,
                    "covariance": covarianza.copy(),
                    "information": informacion.copy(),
                    "range": visibilidad["range"],
                    "bearing": visibilidad["bearing"],
                    "robust_kernel": {
                        "type": "huber",
                        "delta": DELTA_HUBER_OBSERVACION,
                    },
                }
            )

    observaciones.sort(
        key=lambda item: (
            int(item["pose_name"][1:]),
            int(item["landmark_name"][1:]),
        )
    )
    return observaciones


def contar_observaciones_por_landmark(observaciones):
    """Cuenta cuántas poses observan cada landmark."""

    conteo = {}
    for observacion in observaciones:
        nombre = observacion["landmark_name"]
        conteo[nombre] = conteo.get(nombre, 0) + 1
    return conteo


def inicializar_landmark_desde_observaciones(
    nombre_landmark,
    observaciones,
    trayectoria_inicial,
):
    """Inicializa un landmark promediando observaciones transformadas al mundo."""

    trayectoria_inicial = validar_trayectoria(
        trayectoria_inicial,
        "trayectoria inicial",
    )
    candidatas = []
    for observacion in observaciones:
        if observacion["landmark_name"] != nombre_landmark:
            continue
        indice_pose = int(observacion["pose_name"][1:])
        pose = trayectoria_inicial[indice_pose]
        global_estimada = (
            pose[:2]
            + rotacion_2d(pose[2]) @ observacion["measurement"]
        )
        candidatas.append(global_estimada)

    if len(candidatas) < 2:
        raise ValueError(
            f"{nombre_landmark} necesita al menos dos observaciones para inicializarse."
        )

    estimacion = np.mean(np.asarray(candidatas, dtype=float), axis=0)
    indice = int(nombre_landmark[1:])
    sesgo = np.array(
        [
            0.18 * np.sin(0.73 * (indice + 1)),
            -0.16 * np.cos(0.51 * (indice + 1)),
        ],
        dtype=float,
    )
    return validar_landmark(estimacion + sesgo, f"inicial {nombre_landmark}")


def crear_estimaciones_iniciales_landmarks(
    landmarks_reales,
    observaciones,
    trayectoria_inicial,
):
    """Mantiene fijos los conocidos e inicializa los desconocidos."""

    landmarks_reales = validar_landmarks(landmarks_reales)
    resultado = {}
    for nombre, posicion in landmarks_reales.items():
        if nombre in LANDMARKS_CONOCIDOS:
            resultado[nombre] = posicion.copy()
        else:
            resultado[nombre] = inicializar_landmark_desde_observaciones(
                nombre,
                observaciones,
                trayectoria_inicial,
            )
    return validar_landmarks(resultado, "landmarks iniciales")


# ---------------------------------------------------------------------------
# Construcción del grafo pose-landmark
# ---------------------------------------------------------------------------


def crear_pose_landmark_graph(
    trayectoria_real,
    trayectoria_inicial,
    landmarks_reales,
    landmarks_iniciales,
    odometria,
    observaciones,
):
    """Construye un grafo con poses, landmarks, odometría y observaciones."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    trayectoria_inicial = validar_trayectoria(
        trayectoria_inicial,
        "trayectoria inicial",
    )
    landmarks_reales = validar_landmarks(landmarks_reales, "landmarks reales")
    landmarks_iniciales = validar_landmarks(
        landmarks_iniciales,
        "landmarks iniciales",
    )
    if trayectoria_real.shape != trayectoria_inicial.shape:
        raise ValueError("Las trayectorias deben tener la misma forma.")
    if set(landmarks_reales) != set(landmarks_iniciales):
        raise ValueError("Los diccionarios de landmarks deben coincidir.")

    graph = nx.Graph()

    for indice, (pose_real, pose_inicial) in enumerate(
        zip(trayectoria_real, trayectoria_inicial)
    ):
        graph.add_node(
            f"x{indice}",
            index=indice,
            node_type="pose",
            dimension=3,
            fixed=False,
            true_pose=pose_real.copy(),
            initial_estimate=pose_inicial.copy(),
            estimate=pose_inicial.copy(),
            is_prior=indice == 0,
        )

    for nombre, posicion_real in landmarks_reales.items():
        fijo = nombre in LANDMARKS_CONOCIDOS
        graph.add_node(
            nombre,
            index=int(nombre[1:]),
            node_type="landmark",
            dimension=2,
            fixed=fijo,
            landmark_kind="known" if fijo else "unknown",
            true_position=posicion_real.copy(),
            initial_estimate=landmarks_iniciales[nombre].copy(),
            estimate=landmarks_iniciales[nombre].copy(),
            observation_count=0,
        )

    cov_prior = crear_covarianza_diagonal(SIGMAS_PRIOR_POSE)
    cov_odom = crear_covarianza_diagonal(SIGMAS_ODOMETRIA)

    prior = {
        "factor_name": "prior_x0",
        "factor_type": "pose_prior",
        "variables": ("x0",),
        "measurement": trayectoria_real[0].copy(),
        "covariance": cov_prior.copy(),
        "information": calcular_matriz_informacion(cov_prior),
        "robust_kernel": None,
    }

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

    observation_order = []
    for observacion in observaciones:
        datos = dict(observacion)
        nombre = datos.pop("factor_name")
        pose_name = datos["pose_name"]
        landmark_name = datos["landmark_name"]
        graph.add_edge(
            pose_name,
            landmark_name,
            factor_name=nombre,
            **datos,
        )
        graph.nodes[landmark_name]["observation_count"] += 1
        factor_order.append(nombre)
        observation_order.append(nombre)

    variable_order = [f"x{i}" for i in range(len(trayectoria_real))]
    variable_order.extend(LANDMARKS_DESCONOCIDOS)

    variable_slices = {}
    cursor = 0
    for nombre in variable_order:
        dimension = int(graph.nodes[nombre]["dimension"])
        variable_slices[nombre] = slice(cursor, cursor + dimension)
        cursor += dimension

    graph.graph.update(
        {
            "prior": prior,
            "variable_order": variable_order,
            "variable_slices": variable_slices,
            "factor_order": factor_order,
            "observation_order": observation_order,
            "state_dimension": cursor,
            "pose_dimension": 3 * len(trayectoria_real),
            "known_landmarks": list(LANDMARKS_CONOCIDOS),
            "unknown_landmarks": list(LANDMARKS_DESCONOCIDOS),
            "reference_frame": "x0",
            "description": "Grafo SLAM 2D con poses y landmarks",
        }
    )
    return graph


def obtener_factor(graph, factor_name):
    """Recupera un factor por su nombre estable."""

    if factor_name == "prior_x0":
        return dict(graph.graph["prior"])

    for _, _, data in graph.edges(data=True):
        if data.get("factor_name") == factor_name:
            return dict(data)
    raise KeyError(f"No existe el factor {factor_name!r}.")


def obtener_estado(graph, atributo_pose="estimate", atributo_landmark="estimate"):
    """Extrae poses y landmarks del grafo."""

    poses = np.asarray(
        [
            validar_pose(graph.nodes[f"x{i}"][atributo_pose], f"x{i}")
            for i in range(NUMERO_POSES)
        ],
        dtype=float,
    )
    landmarks = {
        nombre: validar_landmark(graph.nodes[nombre][atributo_landmark], nombre)
        for nombre in sorted(
            [n for n, d in graph.nodes(data=True) if d["node_type"] == "landmark"],
            key=lambda item: int(item[1:]),
        )
    }
    return {"poses": poses, "landmarks": landmarks}


def copiar_estado(estado):
    """Crea una copia profunda y validada del estado geométrico."""

    return {
        "poses": validar_trayectoria(estado["poses"], "poses del estado"),
        "landmarks": validar_landmarks(
            estado["landmarks"],
            "landmarks del estado",
        ),
    }


def actualizar_estimaciones_grafo(graph, estado):
    """Copia un estado optimizado a los atributos estimate del grafo."""

    estado = copiar_estado(estado)
    for indice, pose in enumerate(estado["poses"]):
        graph.nodes[f"x{indice}"]["estimate"] = pose.copy()
    for nombre, landmark in estado["landmarks"].items():
        if graph.nodes[nombre]["fixed"]:
            if not np.allclose(
                landmark,
                graph.nodes[nombre]["true_position"],
                atol=1e-12,
            ):
                raise ValueError("Un landmark conocido no puede desplazarse.")
        graph.nodes[nombre]["estimate"] = landmark.copy()


# ---------------------------------------------------------------------------
# Residuos, robustez y sistema lineal
# ---------------------------------------------------------------------------


def calcular_residuo_prior_pose(medicion, pose):
    """Calcula el residuo de un prior de pose."""

    return calcular_movimiento_relativo(
        validar_pose(medicion, "medición prior"),
        validar_pose(pose, "pose prior"),
    )


def calcular_prediccion_odometria(pose_origen, pose_destino):
    """Predice una medición odométrica relativa."""

    return calcular_movimiento_relativo(pose_origen, pose_destino)


def calcular_residuo_odometria(medicion, pose_origen, pose_destino):
    """Compara una odometría medida con su predicción."""

    prediccion = calcular_prediccion_odometria(pose_origen, pose_destino)
    return calcular_movimiento_relativo(
        validar_pose(medicion, "medición odométrica"),
        prediccion,
    )


def calcular_residuo_observacion(medicion, pose, landmark):
    """Calcula el error cartesiano local pose-landmark."""

    medicion = validar_landmark(medicion, "medición de landmark")
    prediccion = predecir_observacion_cartesiana(pose, landmark)
    return validar_landmark(
        prediccion - medicion,
        "residuo pose-landmark",
    )


def calcular_residuo_factor(graph, factor_name, estado):
    """Calcula el residuo de prior, odometría u observación."""

    estado = copiar_estado(estado)
    factor = obtener_factor(graph, factor_name)
    tipo = factor["factor_type"]

    if tipo == "pose_prior":
        indice = int(factor["variables"][0][1:])
        return calcular_residuo_prior_pose(
            factor["measurement"],
            estado["poses"][indice],
        )

    if tipo == "odometry":
        origen, destino = factor["variables"]
        return calcular_residuo_odometria(
            factor["measurement"],
            estado["poses"][int(origen[1:])],
            estado["poses"][int(destino[1:])],
        )

    if tipo == "landmark_observation":
        pose_name, landmark_name = factor["variables"]
        return calcular_residuo_observacion(
            factor["measurement"],
            estado["poses"][int(pose_name[1:])],
            estado["landmarks"][landmark_name],
        )

    raise ValueError(f"Tipo de factor desconocido: {tipo!r}")


def calcular_peso_huber(residuo, informacion, delta):
    """Calcula el peso IRLS de Huber desde una norma de Mahalanobis."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)
    delta = float(delta)
    norma = float(np.sqrt(max(residuo.T @ informacion @ residuo, 0.0)))
    if norma <= delta or norma <= 1e-12:
        return 1.0
    return float(delta / norma)


def calcular_coste_huber(residuo, informacion, delta):
    """Calcula el coste de Huber asociado a un factor."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)
    delta = float(delta)
    norma = float(np.sqrt(max(residuo.T @ informacion @ residuo, 0.0)))
    if norma <= delta:
        return 0.5 * norma**2
    return float(delta * (norma - 0.5 * delta))


def aplicar_perturbacion_variable(graph, estado, nombre_variable, incremento):
    """Perturba una pose o landmark variable dentro de una copia del estado."""

    resultado = copiar_estado(estado)
    datos = graph.nodes[nombre_variable]
    if datos["node_type"] == "pose":
        indice = int(nombre_variable[1:])
        resultado["poses"][indice] = aplicar_incremento_pose(
            resultado["poses"][indice],
            incremento,
        )
    elif datos["node_type"] == "landmark":
        if datos["fixed"]:
            raise ValueError("No se puede perturbar un landmark conocido.")
        resultado["landmarks"][nombre_variable] = aplicar_incremento_landmark(
            resultado["landmarks"][nombre_variable],
            incremento,
        )
    else:
        raise ValueError("Tipo de variable desconocido.")
    return resultado


def calcular_jacobianos_locales_numericos(
    graph,
    factor_name,
    estado,
    epsilon=EPSILON_JACOBIANO,
):
    """Calcula los bloques jacobianos mediante diferencias centrales."""

    estado = copiar_estado(estado)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon debe ser positivo y finito.")

    factor = obtener_factor(graph, factor_name)
    dimension_residuo = len(calcular_residuo_factor(graph, factor_name, estado))
    bloques = {}

    for nombre in factor["variables"]:
        datos = graph.nodes[nombre]
        if datos.get("fixed", False):
            continue
        dimension = int(datos["dimension"])
        bloque = np.zeros((dimension_residuo, dimension), dtype=float)

        for componente in range(dimension):
            delta = np.zeros(dimension, dtype=float)
            delta[componente] = epsilon
            estado_mas = aplicar_perturbacion_variable(
                graph,
                estado,
                nombre,
                delta,
            )
            estado_menos = aplicar_perturbacion_variable(
                graph,
                estado,
                nombre,
                -delta,
            )
            residuo_mas = calcular_residuo_factor(
                graph,
                factor_name,
                estado_mas,
            )
            residuo_menos = calcular_residuo_factor(
                graph,
                factor_name,
                estado_menos,
            )
            diferencia = residuo_mas - residuo_menos
            if dimension_residuo == 3:
                diferencia[2] = normalizar_angulo(diferencia[2])
            bloque[:, componente] = diferencia / (2.0 * epsilon)

        bloques[nombre] = bloque

    return bloques


def seleccionar_factores(
    graph,
    incluir_prior=True,
    incluir_observaciones_conocidas=True,
):
    """Selecciona factores para estudiar anclaje y libertad de gauge."""

    factores = list(graph.graph["factor_order"])
    if not incluir_observaciones_conocidas:
        filtrados = []
        for nombre in factores:
            factor = obtener_factor(graph, nombre)
            if (
                factor["factor_type"] == "landmark_observation"
                and factor["landmark_name"] in graph.graph["known_landmarks"]
            ):
                continue
            filtrados.append(nombre)
        factores = filtrados
    if incluir_prior:
        factores = ["prior_x0"] + factores
    return factores


def ensamblar_sistema(
    graph,
    estado,
    incluir_prior=True,
    incluir_observaciones_conocidas=True,
    usar_robustez=True,
):
    """Ensambla residuos, J, información efectiva, H, g y costes."""

    estado = copiar_estado(estado)
    factores = seleccionar_factores(
        graph,
        incluir_prior=incluir_prior,
        incluir_observaciones_conocidas=incluir_observaciones_conocidas,
    )
    dimensiones = [
        len(calcular_residuo_factor(graph, nombre, estado))
        for nombre in factores
    ]
    numero_filas = int(sum(dimensiones))
    numero_columnas = int(graph.graph["state_dimension"])

    residual = np.zeros(numero_filas, dtype=float)
    jacobiano = np.zeros((numero_filas, numero_columnas), dtype=float)
    informacion = np.zeros((numero_filas, numero_filas), dtype=float)
    factor_slices = {}
    robust_weights = {}
    cost_by_type = {
        "pose_prior": 0.0,
        "odometry": 0.0,
        "landmark_observation_known": 0.0,
        "landmark_observation_unknown": 0.0,
    }

    fila = 0
    for factor_name, dimension in zip(factores, dimensiones):
        filas = slice(fila, fila + dimension)
        fila += dimension
        factor = obtener_factor(graph, factor_name)
        residuo = calcular_residuo_factor(graph, factor_name, estado)
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

        tipo_coste = factor["factor_type"]
        if tipo_coste == "landmark_observation":
            tipo_coste += (
                "_known"
                if factor["landmark_name"] in graph.graph["known_landmarks"]
                else "_unknown"
            )
        cost_by_type[tipo_coste] += coste_factor

        bloques = calcular_jacobianos_locales_numericos(
            graph,
            factor_name,
            estado,
        )
        for variable, bloque in bloques.items():
            columnas = graph.graph["variable_slices"][variable]
            jacobiano[filas, columnas] = bloque

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


def calcular_coste_total(graph, estado):
    """Calcula el coste robusto del grafo completo."""

    return float(ensamblar_sistema(graph, estado)["cost"])


def aplicar_incremento_estado(graph, estado, incremento):
    """Aplica un vector global de incrementos locales."""

    estado = copiar_estado(estado)
    incremento = np.asarray(incremento, dtype=float)
    if incremento.shape != (graph.graph["state_dimension"],):
        raise ValueError("La dimensión del incremento no coincide con el estado.")
    if not np.all(np.isfinite(incremento)):
        raise ValueError("El incremento debe ser finito.")

    resultado = copiar_estado(estado)
    for nombre in graph.graph["variable_order"]:
        delta = incremento[graph.graph["variable_slices"][nombre]]
        resultado = aplicar_perturbacion_variable(
            graph,
            resultado,
            nombre,
            delta,
        )
    return resultado


# ---------------------------------------------------------------------------
# Métricas, gauge y complemento de Schur
# ---------------------------------------------------------------------------


def calcular_metricas_poses(trayectoria_real, trayectoria):
    """Calcula errores de posición y orientación de las poses."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    trayectoria = validar_trayectoria(trayectoria, "trayectoria")
    if trayectoria_real.shape != trayectoria.shape:
        raise ValueError("Las trayectorias deben tener la misma forma.")

    errores_posicion = np.linalg.norm(
        trayectoria[:, :2] - trayectoria_real[:, :2],
        axis=1,
    )
    errores_angulo = np.array(
        [
            normalizar_angulo(estimada - real)
            for real, estimada in zip(
                trayectoria_real[:, 2],
                trayectoria[:, 2],
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


def calcular_metricas_landmarks(
    landmarks_reales,
    landmarks_estimados,
    nombres=LANDMARKS_DESCONOCIDOS,
):
    """Calcula errores de posición para landmarks seleccionados."""

    reales = validar_landmarks(landmarks_reales, "landmarks reales")
    estimados = validar_landmarks(landmarks_estimados, "landmarks estimados")
    errores = {
        nombre: float(np.linalg.norm(estimados[nombre] - reales[nombre]))
        for nombre in nombres
    }
    valores = np.asarray(list(errores.values()), dtype=float)
    return {
        "errors": errores,
        "rmse": float(np.sqrt(np.mean(valores**2))),
        "mae": float(np.mean(valores)),
        "max": float(np.max(valores)),
    }


def calcular_metricas_observaciones(graph, estado):
    """Calcula RMSE cartesiano de todos los factores pose-landmark."""

    residuos = []
    conocidos = []
    desconocidos = []
    for nombre in graph.graph["observation_order"]:
        factor = obtener_factor(graph, nombre)
        residuo = calcular_residuo_factor(graph, nombre, estado)
        norma = float(np.linalg.norm(residuo))
        residuos.append(norma)
        if factor["landmark_name"] in graph.graph["known_landmarks"]:
            conocidos.append(norma)
        else:
            desconocidos.append(norma)

    def rmse(valores):
        valores = np.asarray(valores, dtype=float)
        return float(np.sqrt(np.mean(valores**2))) if len(valores) else 0.0

    return {
        "errors": np.asarray(residuos, dtype=float),
        "rmse": rmse(residuos),
        "known_rmse": rmse(conocidos),
        "unknown_rmse": rmse(desconocidos),
        "max": float(np.max(residuos)) if residuos else 0.0,
    }


def analizar_anclaje(graph, estado):
    """Compara gauge relativo, anclaje por landmarks conocidos y prior."""

    relativo = ensamblar_sistema(
        graph,
        estado,
        incluir_prior=False,
        incluir_observaciones_conocidas=False,
    )["jacobian"]
    referencias = ensamblar_sistema(
        graph,
        estado,
        incluir_prior=False,
        incluir_observaciones_conocidas=True,
    )["jacobian"]
    completo = ensamblar_sistema(
        graph,
        estado,
        incluir_prior=True,
        incluir_observaciones_conocidas=True,
    )["jacobian"]

    def info(matriz):
        rango = int(np.linalg.matrix_rank(matriz, tol=1e-7))
        dimension = matriz.shape[1]
        return {
            "shape": matriz.shape,
            "rank": rango,
            "nullity": dimension - rango,
        }

    return {
        "relative_only": info(relativo),
        "known_landmarks": info(referencias),
        "full": info(completo),
    }


def calcular_complemento_schur(graph, sistema):
    """Elimina los incrementos de landmarks desconocidos del sistema."""

    hessiana = np.asarray(sistema["hessian"], dtype=float)
    gradiente = np.asarray(sistema["gradient"], dtype=float)
    dimension_poses = int(graph.graph["pose_dimension"])

    h_pp = hessiana[:dimension_poses, :dimension_poses]
    h_pl = hessiana[:dimension_poses, dimension_poses:]
    h_lp = hessiana[dimension_poses:, :dimension_poses]
    h_ll = hessiana[dimension_poses:, dimension_poses:]
    g_p = gradiente[:dimension_poses]
    g_l = gradiente[dimension_poses:]

    if h_ll.size == 0:
        raise ValueError("No existen landmarks variables para eliminar.")

    h_ll_regularizada = h_ll + 1e-10 * np.eye(h_ll.shape[0])
    solucion_hlp = np.linalg.solve(h_ll_regularizada, h_lp)
    solucion_gl = np.linalg.solve(h_ll_regularizada, g_l)

    h_reducida = h_pp - h_pl @ solucion_hlp
    g_reducido = g_p - h_pl @ solucion_gl
    return {
        "H_pp": h_pp,
        "H_pl": h_pl,
        "H_lp": h_lp,
        "H_ll": h_ll,
        "g_p": g_p,
        "g_l": g_l,
        "reduced_hessian": h_reducida,
        "reduced_gradient": g_reducido,
        "pose_dimension": dimension_poses,
        "landmark_dimension": h_ll.shape[0],
        "full_dimension": hessiana.shape[0],
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


def optimizar_grafo_landmarks(graph, estado_inicial, max_iteraciones=MAX_ITERACIONES):
    """Optimiza poses y landmarks desconocidos con Levenberg-Marquardt."""

    estado = copiar_estado(estado_inicial)
    damping = LAMBDA_INICIAL
    history = []
    converged = False
    estado_real = obtener_estado(graph, "true_pose", "true_position")

    for iteration in range(int(max_iteraciones)):
        sistema = ensamblar_sistema(graph, estado)
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
            candidato = aplicar_incremento_estado(graph, estado, incremento)
            sistema_candidato = ensamblar_sistema(graph, candidato)
            coste_candidato = sistema_candidato["cost"]

            mejor_intento = {
                "iteration": iteration,
                "attempt": intento,
                "state_before": copiar_estado(estado),
                "state_candidate": copiar_estado(candidato),
                "cost_before": coste_antes,
                "cost_candidate": coste_candidato,
                "damping": damping,
                "step_norm": norma_incremento,
                "gradient_norm": gradiente_norma,
                "accepted": coste_candidato < coste_antes,
            }

            if coste_candidato < coste_antes:
                estado = candidato
                damping = max(damping * 0.32, 1e-10)
                aceptado = True
                break

            damping = min(damping * 8.0, 1e12)

        if mejor_intento is None:
            raise RuntimeError("No se evaluó ningún paso de optimización.")

        sistema_despues = ensamblar_sistema(graph, estado)
        metricas_poses = calcular_metricas_poses(
            estado_real["poses"],
            estado["poses"],
        )
        metricas_landmarks = calcular_metricas_landmarks(
            estado_real["landmarks"],
            estado["landmarks"],
        )
        metricas_observaciones = calcular_metricas_observaciones(graph, estado)

        mejor_intento.update(
            {
                "state_after": copiar_estado(estado),
                "cost_after": sistema_despues["cost"],
                "cost_by_type_after": dict(sistema_despues["cost_by_type"]),
                "accepted": aceptado,
                "damping_after": damping,
                "pose_rmse_after": metricas_poses["position_rmse"],
                "angle_rmse_after_deg": metricas_poses["orientation_rmse_deg"],
                "landmark_rmse_after": metricas_landmarks["rmse"],
                "observation_rmse_after": metricas_observaciones["rmse"],
                "minimum_observation_weight": min(
                    sistema_despues["robust_weights"].values()
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
        "initial_state": copiar_estado(estado_inicial),
        "optimized_state": copiar_estado(estado),
        "history": history,
        "iterations": len(history),
        "converged": converged,
        "final_system": ensamblar_sistema(graph, estado),
    }


# ---------------------------------------------------------------------------
# Resultado completo y estados didácticos
# ---------------------------------------------------------------------------


def crear_resultado_landmarks_slam():
    """Construye, optimiza y resume el ejemplo completo."""

    trayectoria_real = crear_trayectoria_real()
    odometria = crear_mediciones_odometria(trayectoria_real)
    trayectoria_inicial = integrar_odometria(
        trayectoria_real[0],
        odometria["measured"],
    )
    landmarks_reales = crear_landmarks_reales()
    observaciones = crear_observaciones_landmarks(
        trayectoria_real,
        landmarks_reales,
    )
    landmarks_iniciales = crear_estimaciones_iniciales_landmarks(
        landmarks_reales,
        observaciones,
        trayectoria_inicial,
    )
    graph = crear_pose_landmark_graph(
        trayectoria_real,
        trayectoria_inicial,
        landmarks_reales,
        landmarks_iniciales,
        odometria,
        observaciones,
    )

    estado_real = obtener_estado(graph, "true_pose", "true_position")
    estado_inicial = obtener_estado(graph, "initial_estimate", "initial_estimate")
    sistema_inicial = ensamblar_sistema(graph, estado_inicial)
    metricas_poses_iniciales = calcular_metricas_poses(
        estado_real["poses"],
        estado_inicial["poses"],
    )
    metricas_landmarks_iniciales = calcular_metricas_landmarks(
        estado_real["landmarks"],
        estado_inicial["landmarks"],
    )
    metricas_observaciones_iniciales = calcular_metricas_observaciones(
        graph,
        estado_inicial,
    )

    optimization = optimizar_grafo_landmarks(graph, estado_inicial)
    estado_optimizado = optimization["optimized_state"]
    actualizar_estimaciones_grafo(graph, estado_optimizado)
    sistema_final = ensamblar_sistema(graph, estado_optimizado)
    metricas_poses_finales = calcular_metricas_poses(
        estado_real["poses"],
        estado_optimizado["poses"],
    )
    metricas_landmarks_finales = calcular_metricas_landmarks(
        estado_real["landmarks"],
        estado_optimizado["landmarks"],
    )
    metricas_observaciones_finales = calcular_metricas_observaciones(
        graph,
        estado_optimizado,
    )
    anclaje = analizar_anclaje(graph, estado_optimizado)
    schur = calcular_complemento_schur(graph, sistema_final)

    return {
        "graph": graph,
        "true_state": estado_real,
        "initial_state": estado_inicial,
        "optimized_state": estado_optimizado,
        "odometry": odometria,
        "observations": observaciones,
        "observation_counts": contar_observaciones_por_landmark(observaciones),
        "initial_system": sistema_inicial,
        "final_system": sistema_final,
        "initial_pose_metrics": metricas_poses_iniciales,
        "final_pose_metrics": metricas_poses_finales,
        "initial_landmark_metrics": metricas_landmarks_iniciales,
        "final_landmark_metrics": metricas_landmarks_finales,
        "initial_observation_metrics": metricas_observaciones_iniciales,
        "final_observation_metrics": metricas_observaciones_finales,
        "optimization": optimization,
        "anchoring": anclaje,
        "schur": schur,
    }


def crear_estado_animacion(
    *,
    phase,
    message,
    visible_pose_count=0,
    visible_landmark_count=0,
    visible_observation_count=0,
    show_true=True,
    show_initial=False,
    show_current=False,
    current_state=None,
    show_known=True,
    show_unknown=True,
    show_fov=False,
    active_pose=None,
    active_landmark=None,
    iteration=None,
    cost=None,
    pose_rmse=None,
    landmark_rmse=None,
    observation_rmse=None,
    damping=None,
    step_norm=None,
    accepted=None,
    show_history=False,
    show_graph_connections=False,
):
    """Crea un estado autocontenido para la demostración visual."""

    return {
        "phase": str(phase),
        "message": str(message),
        "visible_pose_count": int(visible_pose_count),
        "visible_landmark_count": int(visible_landmark_count),
        "visible_observation_count": int(visible_observation_count),
        "show_true": bool(show_true),
        "show_initial": bool(show_initial),
        "show_current": bool(show_current),
        "current_state": None if current_state is None else copiar_estado(current_state),
        "show_known": bool(show_known),
        "show_unknown": bool(show_unknown),
        "show_fov": bool(show_fov),
        "active_pose": None if active_pose is None else int(active_pose),
        "active_landmark": active_landmark,
        "iteration": None if iteration is None else int(iteration),
        "cost": None if cost is None else float(cost),
        "pose_rmse": None if pose_rmse is None else float(pose_rmse),
        "landmark_rmse": None if landmark_rmse is None else float(landmark_rmse),
        "observation_rmse": (
            None if observation_rmse is None else float(observation_rmse)
        ),
        "damping": None if damping is None else float(damping),
        "step_norm": None if step_norm is None else float(step_norm),
        "accepted": accepted,
        "show_history": bool(show_history),
        "show_graph_connections": bool(show_graph_connections),
    }


def interpolar_estados(estado_origen, estado_destino, alpha):
    """Interpola poses y landmarks desconocidos entre dos iteraciones."""

    origen = copiar_estado(estado_origen)
    destino = copiar_estado(estado_destino)
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha debe pertenecer a [0, 1].")

    poses = (1.0 - alpha) * origen["poses"] + alpha * destino["poses"]
    for indice in range(len(poses)):
        diferencia = normalizar_angulo(
            destino["poses"][indice, 2] - origen["poses"][indice, 2]
        )
        poses[indice, 2] = normalizar_angulo(
            origen["poses"][indice, 2] + alpha * diferencia
        )

    landmarks = {}
    for nombre in origen["landmarks"]:
        if nombre in LANDMARKS_CONOCIDOS:
            landmarks[nombre] = origen["landmarks"][nombre].copy()
        else:
            landmarks[nombre] = (
                (1.0 - alpha) * origen["landmarks"][nombre]
                + alpha * destino["landmarks"][nombre]
            )

    return {
        "poses": validar_trayectoria(poses),
        "landmarks": validar_landmarks(landmarks),
    }


def crear_estados_animacion(resultado):
    """Crea la secuencia completa de poses, observaciones y optimización."""

    graph = resultado["graph"]
    estado_real = resultado["true_state"]
    estado_inicial = resultado["initial_state"]
    estado_final = resultado["optimized_state"]
    observaciones = resultado["observations"]
    numero_poses = len(estado_real["poses"])
    numero_landmarks = len(estado_real["landmarks"])
    numero_observaciones = len(observaciones)
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
        "Landmarks en SLAM: las observaciones conectan poses y referencias del entorno.",
        repeat=3,
        show_true=True,
        visible_pose_count=0,
        visible_landmark_count=0,
    )

    for count in range(1, numero_landmarks + 1):
        add(
            "landmarks",
            "Los landmarks conocidos están fijos; los desconocidos deberán estimarse.",
            visible_pose_count=0,
            visible_landmark_count=count,
            show_true=True,
            show_known=True,
            show_unknown=True,
            active_landmark=f"l{count - 1}",
        )

    observaciones_por_pose = {}
    for indice, observacion in enumerate(observaciones, start=1):
        pose_index = int(observacion["pose_name"][1:])
        observaciones_por_pose.setdefault(pose_index, []).append(indice)

    visibles = 0
    for count in range(1, numero_poses + 1):
        pose_index = count - 1
        if pose_index in observaciones_por_pose:
            visibles = max(visibles, max(observaciones_por_pose[pose_index]))
        add(
            "sensing",
            "El robot avanza y crea factores hacia los landmarks visibles.",
            visible_pose_count=count,
            visible_landmark_count=numero_landmarks,
            visible_observation_count=visibles,
            show_true=True,
            show_known=True,
            show_unknown=True,
            show_fov=True,
            active_pose=pose_index,
        )

    add(
        "known_landmarks",
        "Las referencias conocidas aportan información global y no se optimizan.",
        repeat=4,
        visible_pose_count=numero_poses,
        visible_landmark_count=numero_landmarks,
        visible_observation_count=numero_observaciones,
        show_true=True,
        show_initial=True,
        show_known=True,
        show_unknown=True,
        active_landmark=LANDMARKS_CONOCIDOS[0],
    )

    add(
        "unknown_landmarks",
        "Los landmarks desconocidos se inicializan desde varias observaciones.",
        repeat=4,
        visible_pose_count=numero_poses,
        visible_landmark_count=numero_landmarks,
        visible_observation_count=numero_observaciones,
        show_true=True,
        show_initial=True,
        show_known=True,
        show_unknown=True,
        current_state=estado_inicial,
        active_landmark=LANDMARKS_DESCONOCIDOS[0],
        cost=resultado["initial_system"]["cost"],
        pose_rmse=resultado["initial_pose_metrics"]["position_rmse"],
        landmark_rmse=resultado["initial_landmark_metrics"]["rmse"],
        observation_rmse=resultado["initial_observation_metrics"]["rmse"],
    )

    add(
        "factor_graph",
        "El grafo combina odometría, prior y observaciones pose-landmark.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_landmark_count=numero_landmarks,
        visible_observation_count=numero_observaciones,
        show_true=True,
        show_initial=True,
        show_known=True,
        show_unknown=True,
        current_state=estado_inicial,
        show_graph_connections=True,
        cost=resultado["initial_system"]["cost"],
        pose_rmse=resultado["initial_pose_metrics"]["position_rmse"],
        landmark_rmse=resultado["initial_landmark_metrics"]["rmse"],
        observation_rmse=resultado["initial_observation_metrics"]["rmse"],
    )

    for entry in resultado["optimization"]["history"]:
        for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            estado_interpolado = interpolar_estados(
                entry["state_before"],
                entry["state_after"],
                alpha,
            )
            sistema = ensamblar_sistema(graph, estado_interpolado)
            metricas_poses = calcular_metricas_poses(
                estado_real["poses"],
                estado_interpolado["poses"],
            )
            metricas_landmarks = calcular_metricas_landmarks(
                estado_real["landmarks"],
                estado_interpolado["landmarks"],
            )
            metricas_obs = calcular_metricas_observaciones(
                graph,
                estado_interpolado,
            )
            add(
                "optimization",
                "La optimización corrige poses y landmarks desconocidos conjuntamente.",
                visible_pose_count=numero_poses,
                visible_landmark_count=numero_landmarks,
                visible_observation_count=numero_observaciones,
                show_true=True,
                show_initial=True,
                show_current=True,
                current_state=estado_interpolado,
                show_known=True,
                show_unknown=True,
                show_graph_connections=True,
                iteration=entry["iteration"] + 1,
                cost=sistema["cost"],
                pose_rmse=metricas_poses["position_rmse"],
                landmark_rmse=metricas_landmarks["rmse"],
                observation_rmse=metricas_obs["rmse"],
                damping=entry["damping"],
                step_norm=entry["step_norm"],
                accepted=entry["accepted"],
                show_history=True,
            )

    add(
        "comparison",
        "Antes y después: las referencias conocidas anclan y las desconocidas convergen.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_landmark_count=numero_landmarks,
        visible_observation_count=numero_observaciones,
        show_true=True,
        show_initial=True,
        show_current=True,
        current_state=estado_final,
        show_known=True,
        show_unknown=True,
        show_graph_connections=True,
        iteration=resultado["optimization"]["iterations"],
        cost=resultado["final_system"]["cost"],
        pose_rmse=resultado["final_pose_metrics"]["position_rmse"],
        landmark_rmse=resultado["final_landmark_metrics"]["rmse"],
        observation_rmse=resultado["final_observation_metrics"]["rmse"],
        show_history=True,
    )

    add(
        "summary",
        "Landmarks compartidos conectan la trayectoria con un mapa consistente.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_landmark_count=numero_landmarks,
        visible_observation_count=numero_observaciones,
        show_true=True,
        show_initial=True,
        show_current=True,
        current_state=estado_final,
        show_known=True,
        show_unknown=True,
        show_graph_connections=True,
        iteration=resultado["optimization"]["iterations"],
        cost=resultado["final_system"]["cost"],
        pose_rmse=resultado["final_pose_metrics"]["position_rmse"],
        landmark_rmse=resultado["final_landmark_metrics"]["rmse"],
        observation_rmse=resultado["final_observation_metrics"]["rmse"],
        show_history=True,
    )

    for step, state in enumerate(states, start=1):
        state["step"] = step
        state["total_steps"] = len(states)
    return states


# ---------------------------------------------------------------------------
# Validaciones y salida
# ---------------------------------------------------------------------------


def validar_visibilidad(resultado):
    """Comprueba visibilidad y redundancia de las observaciones."""

    conteo = resultado["observation_counts"]
    for nombre in resultado["true_state"]["landmarks"]:
        if conteo.get(nombre, 0) < 2:
            raise ValueError(f"{nombre} debe observarse al menos dos veces.")

    for observacion in resultado["observations"]:
        indice_pose = int(observacion["pose_name"][1:])
        visibilidad = landmark_visible(
            resultado["true_state"]["poses"][indice_pose],
            resultado["true_state"]["landmarks"][observacion["landmark_name"]],
        )
        if not visibilidad["visible"]:
            raise ValueError("Se ha almacenado una observación fuera de visibilidad.")


def validar_grafo(resultado):
    """Comprueba nodos, factores, tipos y conectividad."""

    graph = resultado["graph"]
    if graph.number_of_nodes() != NUMERO_POSES + 6:
        raise ValueError("Debe existir un nodo por pose y landmark.")
    if len(graph.graph["known_landmarks"]) != 2:
        raise ValueError("Deben existir dos landmarks conocidos.")
    if len(graph.graph["unknown_landmarks"]) != 4:
        raise ValueError("Deben existir cuatro landmarks desconocidos.")
    if graph.graph["state_dimension"] != 3 * NUMERO_POSES + 2 * 4:
        raise ValueError("La dimensión global del estado es incorrecta.")
    if not nx.is_connected(graph):
        raise ValueError("El grafo pose-landmark debe ser conectado.")

    for nombre in graph.graph["known_landmarks"]:
        datos = graph.nodes[nombre]
        if not datos["fixed"] or nombre in graph.graph["variable_order"]:
            raise ValueError("Un landmark conocido debe permanecer fuera del estado.")
    for nombre in graph.graph["unknown_landmarks"]:
        datos = graph.nodes[nombre]
        if datos["fixed"] or nombre not in graph.graph["variable_order"]:
            raise ValueError("Un landmark desconocido debe ser variable.")

    for factor_name in ["prior_x0"] + graph.graph["factor_order"]:
        factor = obtener_factor(graph, factor_name)
        cov = factor["covariance"]
        info = factor["information"]
        if not np.allclose(cov @ info, np.eye(len(cov)), atol=1e-9):
            raise ValueError("Covarianza e información no son inversas.")
        for variable in factor["variables"]:
            if variable not in graph:
                raise ValueError("Un factor referencia una variable inexistente.")


def validar_sistema(resultado):
    """Valida dimensiones, ensamblaje, gauge y complemento de Schur."""

    graph = resultado["graph"]
    sistema = resultado["final_system"]
    j = sistema["jacobian"]
    h = sistema["hessian"]
    omega = sistema["information"]
    e = sistema["residual"]
    g = sistema["gradient"]

    expected_columns = graph.graph["state_dimension"]
    expected_rows = 3 + 3 * (NUMERO_POSES - 1) + 2 * len(resultado["observations"])
    if j.shape != (expected_rows, expected_columns):
        raise ValueError("La forma del jacobiano global es incorrecta.")
    if h.shape != (expected_columns, expected_columns):
        raise ValueError("La forma de la Hessiana es incorrecta.")
    if not np.allclose(h, h.T, atol=1e-8):
        raise ValueError("La Hessiana debe ser simétrica.")
    if not np.allclose(h, j.T @ omega @ j, atol=1e-8):
        raise ValueError("H no coincide con JᵀΩJ.")
    if not np.allclose(g, j.T @ omega @ e, atol=1e-8):
        raise ValueError("g no coincide con JᵀΩe.")

    anclaje = resultado["anchoring"]
    if anclaje["relative_only"]["nullity"] != 3:
        raise ValueError("El sistema relativo debe conservar tres grados de gauge.")
    if anclaje["known_landmarks"]["nullity"] != 0:
        raise ValueError("Los landmarks conocidos deben eliminar el gauge.")
    if anclaje["full"]["nullity"] != 0:
        raise ValueError("El sistema completo debe tener rango total.")

    schur = resultado["schur"]
    if schur["pose_dimension"] != 3 * NUMERO_POSES:
        raise ValueError("La dimensión de poses en Schur es incorrecta.")
    if schur["landmark_dimension"] != 2 * len(LANDMARKS_DESCONOCIDOS):
        raise ValueError("La dimensión de landmarks en Schur es incorrecta.")
    if schur["reduced_hessian"].shape != (
        3 * NUMERO_POSES,
        3 * NUMERO_POSES,
    ):
        raise ValueError("La Hessiana reducida tiene una forma incorrecta.")
    if not np.allclose(
        schur["reduced_hessian"],
        schur["reduced_hessian"].T,
        atol=1e-7,
    ):
        raise ValueError("La Hessiana reducida debe ser simétrica.")


def validar_optimizacion(resultado):
    """Comprueba que el ajuste mejora trayectoria, landmarks y observaciones."""

    if resultado["final_system"]["cost"] >= resultado["initial_system"]["cost"]:
        raise ValueError("El coste final debe ser menor que el inicial.")
    if (
        resultado["final_pose_metrics"]["position_rmse"]
        >= resultado["initial_pose_metrics"]["position_rmse"]
    ):
        raise ValueError("El RMSE de poses debe disminuir.")
    if (
        resultado["final_landmark_metrics"]["rmse"]
        >= resultado["initial_landmark_metrics"]["rmse"]
    ):
        raise ValueError("El RMSE de landmarks debe disminuir.")
    if (
        resultado["final_observation_metrics"]["rmse"]
        >= resultado["initial_observation_metrics"]["rmse"]
    ):
        raise ValueError("El error de observación debe disminuir.")

    for nombre in LANDMARKS_CONOCIDOS:
        if not np.allclose(
            resultado["initial_state"]["landmarks"][nombre],
            resultado["optimized_state"]["landmarks"][nombre],
            atol=1e-12,
        ):
            raise ValueError("Los landmarks conocidos no deben moverse.")

    costes = [
        entrada["cost_after"]
        for entrada in resultado["optimization"]["history"]
        if entrada["accepted"]
    ]
    if any(b >= a for a, b in zip(costes[:-1], costes[1:])):
        raise ValueError("Los costes aceptados deben disminuir estrictamente.")


def validar_resultados(resultado, states):
    """Ejecuta todas las validaciones y devuelve un resumen."""

    validar_visibilidad(resultado)
    validar_grafo(resultado)
    validar_sistema(resultado)
    validar_optimizacion(resultado)

    if len(states) < 60:
        raise ValueError("La demostración necesita suficientes estados didácticos.")
    if states[-1]["phase"] != "summary":
        raise ValueError("El último estado debe ser el resumen final.")
    if states[-1]["current_state"] is None:
        raise ValueError("El resumen debe contener el estado optimizado.")

    graph = resultado["graph"]
    return {
        "pose_count": NUMERO_POSES,
        "landmark_count": 6,
        "known_landmark_count": len(LANDMARKS_CONOCIDOS),
        "unknown_landmark_count": len(LANDMARKS_DESCONOCIDOS),
        "odometry_count": NUMERO_POSES - 1,
        "observation_count": len(resultado["observations"]),
        "observation_counts": dict(resultado["observation_counts"]),
        "state_dimension": graph.graph["state_dimension"],
        "factor_count": 1 + len(graph.graph["factor_order"]),
        "state_count": len(states),
        "iterations": resultado["optimization"]["iterations"],
        "converged": resultado["optimization"]["converged"],
        "initial_cost": resultado["initial_system"]["cost"],
        "final_cost": resultado["final_system"]["cost"],
        "initial_pose_rmse": resultado["initial_pose_metrics"]["position_rmse"],
        "final_pose_rmse": resultado["final_pose_metrics"]["position_rmse"],
        "initial_angle_rmse_deg": resultado["initial_pose_metrics"]["orientation_rmse_deg"],
        "final_angle_rmse_deg": resultado["final_pose_metrics"]["orientation_rmse_deg"],
        "initial_landmark_rmse": resultado["initial_landmark_metrics"]["rmse"],
        "final_landmark_rmse": resultado["final_landmark_metrics"]["rmse"],
        "initial_observation_rmse": resultado["initial_observation_metrics"]["rmse"],
        "final_observation_rmse": resultado["final_observation_metrics"]["rmse"],
        "relative_rank": resultado["anchoring"]["relative_only"]["rank"],
        "relative_nullity": resultado["anchoring"]["relative_only"]["nullity"],
        "known_rank": resultado["anchoring"]["known_landmarks"]["rank"],
        "known_nullity": resultado["anchoring"]["known_landmarks"]["nullity"],
        "full_rank": resultado["anchoring"]["full"]["rank"],
        "full_nullity": resultado["anchoring"]["full"]["nullity"],
        "jacobian_shape": resultado["final_system"]["jacobian"].shape,
        "hessian_shape": resultado["final_system"]["hessian"].shape,
        "schur_shape": resultado["schur"]["reduced_hessian"].shape,
    }


def imprimir_resumen(validation):
    """Imprime las magnitudes principales del ejemplo."""

    print("\n=== Landmarks en SLAM: poses, referencias y optimización ===")
    print(
        f"Poses: {validation['pose_count']} · "
        f"landmarks: {validation['landmark_count']} "
        f"({validation['known_landmark_count']} conocidos, "
        f"{validation['unknown_landmark_count']} desconocidos)"
    )
    print(
        f"Odometrías: {validation['odometry_count']} · "
        f"observaciones: {validation['observation_count']} · "
        f"factores totales: {validation['factor_count']}"
    )
    print("Observaciones por landmark:", validation["observation_counts"])
    print(f"Iteraciones: {validation['iterations']}")
    print(
        f"Coste: {validation['initial_cost']:.6f} "
        f"→ {validation['final_cost']:.6f}"
    )
    print(
        f"RMSE poses: {validation['initial_pose_rmse']:.6f} m "
        f"→ {validation['final_pose_rmse']:.6f} m"
    )
    print(
        "RMSE angular: "
        f"{validation['initial_angle_rmse_deg']:.6f}° "
        f"→ {validation['final_angle_rmse_deg']:.6f}°"
    )
    print(
        f"RMSE landmarks: {validation['initial_landmark_rmse']:.6f} m "
        f"→ {validation['final_landmark_rmse']:.6f} m"
    )
    print(
        f"RMSE observaciones: {validation['initial_observation_rmse']:.6f} m "
        f"→ {validation['final_observation_rmse']:.6f} m"
    )
    print(
        "Gauge relativo / referencias conocidas / completo: "
        f"nulidad {validation['relative_nullity']}/"
        f"{validation['known_nullity']}/"
        f"{validation['full_nullity']}"
    )
    print(
        f"J: {validation['jacobian_shape']} · "
        f"H: {validation['hessian_shape']} · "
        f"Schur: {validation['schur_shape']}"
    )
    print(f"Estados de animación: {validation['state_count']}")


def main():
    resultado = crear_resultado_landmarks_slam()
    states = crear_estados_animacion(resultado)
    validation = validar_resultados(resultado, states)
    imprimir_resumen(validation)

    animator = GraphAnimator(figsize=(19, 10.5), interval=220)
    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "06_graph_slam"
        / "04_landmarks_slam.png"
    )
    animator.animate_landmarks_slam(
        result=resultado,
        states=states,
        title="Landmarks en SLAM: observaciones, referencias y optimización conjunta",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
