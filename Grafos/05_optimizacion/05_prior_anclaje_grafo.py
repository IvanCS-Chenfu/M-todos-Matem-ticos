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

POSES_BASE = {
    "x0": np.array([0.0, 0.0, np.deg2rad(0.0)], dtype=float),
    "x1": np.array([2.0, 0.35, np.deg2rad(12.0)], dtype=float),
    "x2": np.array([3.75, 1.55, np.deg2rad(36.0)], dtype=float),
    "x3": np.array([2.25, 3.10, np.deg2rad(82.0)], dtype=float),
}

ARISTAS_RELATIVAS = [
    ("x0", "x1", "odometría"),
    ("x1", "x2", "odometría"),
    ("x2", "x3", "odometría"),
    ("x3", "x0", "cierre de ciclo"),
]

SIGMAS_RELATIVAS = np.array(
    [0.18, 0.18, np.deg2rad(4.0)],
    dtype=float,
)

RUIDO_CIERRE = np.array(
    [0.08, -0.05, np.deg2rad(2.0)],
    dtype=float,
)

POSE_PRIOR = np.array([0.0, 0.0, 0.0], dtype=float)
SIGMAS_PRIOR = np.array(
    [0.10, 0.10, np.deg2rad(2.0)],
    dtype=float,
)

TRANSFORMACION_TRASLADADA = np.array(
    [3.40, 1.45, 0.0],
    dtype=float,
)

TRANSFORMACION_ROTADA = np.array(
    [-1.20, 2.35, np.deg2rad(35.0)],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Operaciones de SE(2)
# ---------------------------------------------------------------------------


def normalizar_angulo(angulo):
    """Normaliza un ángulo al intervalo [-pi, pi)."""

    angulo = float(angulo)
    if not np.isfinite(angulo):
        raise ValueError("El ángulo debe ser finito.")

    return float((angulo + np.pi) % (2.0 * np.pi) - np.pi)


def validar_pose(pose, nombre="pose"):
    """Valida una pose SE(2) representada por (x, y, theta)."""

    pose = np.asarray(pose, dtype=float)

    if pose.shape != (3,):
        raise ValueError(f"{nombre} debe contener (x, y, theta).")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{nombre} debe contener valores finitos.")

    pose = pose.copy()
    pose[2] = normalizar_angulo(pose[2])
    return pose


def pose_a_matriz_se2(pose):
    """Convierte una pose (x, y, theta) en una matriz homogénea 3x3."""

    x, y, theta = validar_pose(pose)
    coseno = np.cos(theta)
    seno = np.sin(theta)

    return np.array(
        [
            [coseno, -seno, x],
            [seno, coseno, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def matriz_a_pose_se2(matriz):
    """Convierte una matriz homogénea 3x3 en una pose SE(2)."""

    matriz = np.asarray(matriz, dtype=float)

    if matriz.shape != (3, 3):
        raise ValueError("La transformación debe ser una matriz 3x3.")
    if not np.all(np.isfinite(matriz)):
        raise ValueError("La transformación debe contener valores finitos.")
    if not np.allclose(matriz[2], [0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("La última fila de una transformación SE(2) no es válida.")

    theta = np.arctan2(matriz[1, 0], matriz[0, 0])
    return np.array(
        [matriz[0, 2], matriz[1, 2], normalizar_angulo(theta)],
        dtype=float,
    )


def componer_poses_se2(pose_a, pose_b):
    """Calcula pose_a ⊕ pose_b mediante matrices homogéneas."""

    return matriz_a_pose_se2(
        pose_a_matriz_se2(pose_a) @ pose_a_matriz_se2(pose_b)
    )


def invertir_pose_se2(pose):
    """Calcula la transformación inversa de una pose SE(2)."""

    matriz = pose_a_matriz_se2(pose)
    rotacion = matriz[:2, :2]
    traslacion = matriz[:2, 2]

    inversa = np.eye(3, dtype=float)
    inversa[:2, :2] = rotacion.T
    inversa[:2, 2] = -rotacion.T @ traslacion
    return matriz_a_pose_se2(inversa)


def calcular_prediccion_relativa(pose_i, pose_j):
    """Calcula la pose de x_j expresada en el sistema de x_i."""

    return componer_poses_se2(invertir_pose_se2(pose_i), pose_j)


def calcular_residuo_relativo(medicion, prediccion):
    """Calcula el error SE(2): z_ij^-1 ⊕ (x_i^-1 ⊕ x_j)."""

    error = componer_poses_se2(invertir_pose_se2(medicion), prediccion)
    error[2] = normalizar_angulo(error[2])
    return error


def aplicar_transformacion_global(poses, transformacion_global):
    """Aplica por la izquierda la misma transformación global a todas las poses."""

    transformacion_global = validar_pose(
        transformacion_global,
        "transformación global",
    )

    return {
        nombre: componer_poses_se2(transformacion_global, pose)
        for nombre, pose in poses.items()
    }


def interpolar_transformacion(transformacion_final, progreso):
    """Interpola desde la identidad hasta una transformación global."""

    transformacion_final = validar_pose(
        transformacion_final,
        "transformación final",
    )
    progreso = float(progreso)

    if not 0.0 <= progreso <= 1.0:
        raise ValueError("El progreso debe pertenecer al intervalo [0, 1].")

    return np.array(
        [
            progreso * transformacion_final[0],
            progreso * transformacion_final[1],
            normalizar_angulo(progreso * transformacion_final[2]),
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Covarianzas, mediciones, residuos y costes
# ---------------------------------------------------------------------------


def crear_covarianza_diagonal(sigmas):
    """Construye una covarianza diagonal a partir de desviaciones estándar."""

    sigmas = np.asarray(sigmas, dtype=float)

    if sigmas.shape != (3,):
        raise ValueError("Se esperaban tres desviaciones estándar.")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("Las desviaciones estándar deben ser positivas y finitas.")

    return np.diag(sigmas**2)


def calcular_matriz_informacion(covarianza):
    """Calcula la inversa de una covarianza definida positiva."""

    covarianza = np.asarray(covarianza, dtype=float)

    if covarianza.shape != (3, 3):
        raise ValueError("La covarianza debe ser una matriz 3x3.")
    if not np.all(np.isfinite(covarianza)):
        raise ValueError("La covarianza debe contener valores finitos.")
    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise ValueError("La covarianza debe ser simétrica.")
    if np.any(np.linalg.eigvalsh(covarianza) <= 0.0):
        raise ValueError("La covarianza debe ser definida positiva.")

    informacion = np.linalg.inv(covarianza)

    if not np.allclose(
        covarianza @ informacion,
        np.eye(3),
        atol=1e-10,
    ):
        raise ValueError("La información no es la inversa de la covarianza.")

    return informacion


def calcular_coste_ponderado(residuo, informacion):
    """Calcula e^T Omega e y evita negativos debidos a redondeo."""

    residuo = validar_pose(residuo, "residuo")
    informacion = np.asarray(informacion, dtype=float)

    if informacion.shape != (3, 3):
        raise ValueError("La información debe ser una matriz 3x3.")
    if not np.all(np.isfinite(informacion)):
        raise ValueError("La información debe contener valores finitos.")

    coste = float(residuo.T @ informacion @ residuo)

    if coste < -1e-10:
        raise ValueError("Un coste cuadrático no puede ser negativo.")

    return max(coste, 0.0)


def crear_mediciones_relativas(poses):
    """Crea mediciones relativas deterministas a partir de las poses base."""

    mediciones = {}

    for origen, destino, sensor in ARISTAS_RELATIVAS:
        medicion = calcular_prediccion_relativa(
            poses[origen],
            poses[destino],
        )

        if sensor == "cierre de ciclo":
            medicion = componer_poses_se2(medicion, RUIDO_CIERRE)

        mediciones[(origen, destino)] = medicion

    return mediciones


def calcular_residuos_relativos(graph, poses):
    """Evalúa todos los residuos de las aristas relativas del pose graph."""

    residuos = {}

    for _, _, datos in graph.edges(data=True):
        if datos.get("edge_type") != "relative_pose":
            continue

        origen = datos["origin"]
        destino = datos["target"]
        prediccion = calcular_prediccion_relativa(
            poses[origen],
            poses[destino],
        )
        residuos[(origen, destino)] = calcular_residuo_relativo(
            datos["measurement"],
            prediccion,
        )

    return residuos


def calcular_coste_relativo(graph, poses):
    """Suma los costes de todas las restricciones relativas."""

    residuos = calcular_residuos_relativos(graph, poses)
    contribuciones = {}

    for arista, residuo in residuos.items():
        informacion = graph.edges[arista]["information"]
        contribuciones[arista] = calcular_coste_ponderado(
            residuo,
            informacion,
        )

    return {
        "residuals": residuos,
        "contributions": contribuciones,
        "cost": float(sum(contribuciones.values())),
    }


def calcular_residuo_prior(pose, pose_prior):
    """Calcula el residuo absoluto del prior en SE(2)."""

    residuo = componer_poses_se2(
        invertir_pose_se2(pose_prior),
        pose,
    )
    residuo[2] = normalizar_angulo(residuo[2])
    return residuo


def calcular_coste_prior(pose, pose_prior, informacion_prior):
    """Calcula el término absoluto e_0^T Omega_0 e_0."""

    residuo = calcular_residuo_prior(pose, pose_prior)
    return {
        "residual": residuo,
        "cost": calcular_coste_ponderado(residuo, informacion_prior),
    }


def calcular_coste_total(
    graph,
    poses,
    pose_prior=None,
    informacion_prior=None,
):
    """Calcula coste relativo, coste del prior y coste total."""

    relativo = calcular_coste_relativo(graph, poses)

    if pose_prior is None or informacion_prior is None:
        prior = {
            "residual": np.zeros(3, dtype=float),
            "cost": 0.0,
        }
    else:
        prior = calcular_coste_prior(
            poses["x0"],
            pose_prior,
            informacion_prior,
        )

    return {
        "relative": relativo,
        "prior": prior,
        "total_cost": relativo["cost"] + prior["cost"],
    }


# ---------------------------------------------------------------------------
# Construcción del grafo sin prior y del grafo anclado
# ---------------------------------------------------------------------------


def crear_grafo_sin_prior(poses_base):
    """Crea un pose graph conectado formado solo por restricciones relativas."""

    poses_base = {
        nombre: validar_pose(pose, nombre)
        for nombre, pose in poses_base.items()
    }
    mediciones = crear_mediciones_relativas(poses_base)
    covarianza = crear_covarianza_diagonal(SIGMAS_RELATIVAS)
    informacion = calcular_matriz_informacion(covarianza)

    graph = nx.Graph()
    graph.graph["name"] = "Pose graph relativo sin prior"
    graph.graph["gauge_dof_expected"] = 3
    graph.graph["reference_frame"] = "no fijado"

    for nombre, pose in poses_base.items():
        graph.add_node(
            nombre,
            node_type="pose",
            dimension=3,
            estimate=pose.copy(),
            label=nombre,
        )

    for origen, destino, sensor in ARISTAS_RELATIVAS:
        graph.add_edge(
            origen,
            destino,
            edge_type="relative_pose",
            origin=origen,
            target=destino,
            sensor=sensor,
            measurement=mediciones[(origen, destino)].copy(),
            covariance=covarianza.copy(),
            information=informacion.copy(),
            residual_model="z_ij^-1 ⊕ (x_i^-1 ⊕ x_j)",
            cost_model="e_ij.T @ Omega_ij @ e_ij",
        )

    return graph


def agregar_prior_al_grafo(
    graph,
    pose_prior,
    sigmas_prior,
):
    """Devuelve una copia del pose graph con un factor prior sobre x0."""

    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise TypeError("Se esperaba un nx.Graph no dirigido.")
    if "x0" not in graph:
        raise ValueError("El grafo debe contener la pose x0.")

    pose_prior = validar_pose(pose_prior, "pose prior")
    covarianza = crear_covarianza_diagonal(sigmas_prior)
    informacion = calcular_matriz_informacion(covarianza)

    anchored = graph.copy()
    anchored.graph["name"] = "Pose graph con prior sobre x0"
    anchored.graph["reference_frame"] = "definido por el prior"
    anchored.graph["prior_node"] = "prior_x0"

    anchored.add_node(
        "prior_x0",
        node_type="prior_factor",
        dimension=3,
        label="prior x0",
        measurement=pose_prior.copy(),
        covariance=covarianza.copy(),
        information=informacion.copy(),
        factor_type="unary_prior",
    )
    anchored.add_edge(
        "prior_x0",
        "x0",
        edge_type="prior",
        measurement=pose_prior.copy(),
        covariance=covarianza.copy(),
        information=informacion.copy(),
        residual_model="prior^-1 ⊕ x0",
        cost_model="e_0.T @ Omega_0 @ e_0",
    )

    return anchored


def obtener_datos_prior(graph_with_prior):
    """Extrae la medición y la información del factor prior."""

    if "prior_x0" not in graph_with_prior:
        raise ValueError("El grafo anclado no contiene el factor prior_x0.")

    datos = graph_with_prior.nodes["prior_x0"]
    return datos["measurement"], datos["information"]


# ---------------------------------------------------------------------------
# Jacobiano numérico, rango y espacio nulo
# ---------------------------------------------------------------------------


def obtener_nombres_poses(graph):
    """Devuelve los nodos pose ordenados por su índice numérico."""

    nombres = [
        nodo
        for nodo, datos in graph.nodes(data=True)
        if datos.get("node_type") == "pose"
    ]
    return sorted(nombres, key=lambda nombre: int(nombre[1:]))


def poses_a_vector(poses, nombres):
    """Apila las poses en un único vector de optimización."""

    return np.concatenate(
        [validar_pose(poses[nombre], nombre) for nombre in nombres]
    )


def vector_a_poses(vector, nombres):
    """Reconstruye el diccionario de poses desde el vector apilado."""

    vector = np.asarray(vector, dtype=float)

    if vector.shape != (3 * len(nombres),):
        raise ValueError("El vector no coincide con el número de poses.")

    poses = {}
    for indice, nombre in enumerate(nombres):
        pose = vector[3 * indice : 3 * indice + 3].copy()
        pose[2] = normalizar_angulo(pose[2])
        poses[nombre] = pose

    return poses


def raiz_informacion(informacion):
    """Devuelve L^T para que ||L^T e||² = e^T Omega e."""

    informacion = np.asarray(informacion, dtype=float)
    return np.linalg.cholesky(informacion).T


def construir_vector_residuos_ponderados(
    graph,
    poses,
    incluir_prior=False,
):
    """Apila residuos relativos y, opcionalmente, el residuo del prior."""

    valores = []

    for _, _, datos in graph.edges(data=True):
        if datos.get("edge_type") != "relative_pose":
            continue

        origen = datos["origin"]
        destino = datos["target"]
        prediccion = calcular_prediccion_relativa(
            poses[origen],
            poses[destino],
        )
        residuo = calcular_residuo_relativo(
            datos["measurement"],
            prediccion,
        )
        valores.append(raiz_informacion(datos["information"]) @ residuo)

    if incluir_prior:
        pose_prior, informacion_prior = obtener_datos_prior(graph)
        residuo_prior = calcular_residuo_prior(poses["x0"], pose_prior)
        valores.append(raiz_informacion(informacion_prior) @ residuo_prior)

    if not valores:
        return np.empty(0, dtype=float)

    return np.concatenate(valores)


def construir_jacobiano_numerico(
    graph,
    poses,
    incluir_prior=False,
    epsilon=1e-7,
):
    """Calcula por diferencias centrales el jacobiano de los residuos."""

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon debe ser positivo y finito.")

    nombres = obtener_nombres_poses(graph)
    vector = poses_a_vector(poses, nombres)
    referencia = construir_vector_residuos_ponderados(
        graph,
        poses,
        incluir_prior=incluir_prior,
    )
    jacobiano = np.zeros((referencia.size, vector.size), dtype=float)

    for columna in range(vector.size):
        positivo = vector.copy()
        negativo = vector.copy()
        positivo[columna] += epsilon
        negativo[columna] -= epsilon

        poses_positivas = vector_a_poses(positivo, nombres)
        poses_negativas = vector_a_poses(negativo, nombres)

        residual_positivo = construir_vector_residuos_ponderados(
            graph,
            poses_positivas,
            incluir_prior=incluir_prior,
        )
        residual_negativo = construir_vector_residuos_ponderados(
            graph,
            poses_negativas,
            incluir_prior=incluir_prior,
        )

        jacobiano[:, columna] = (
            residual_positivo - residual_negativo
        ) / (2.0 * epsilon)

    return jacobiano


def construir_direcciones_gauge(poses, nombres):
    """Construye traslación x, traslación y y rotación global infinitesimal."""

    numero_variables = 3 * len(nombres)
    traslacion_x = np.zeros(numero_variables, dtype=float)
    traslacion_y = np.zeros(numero_variables, dtype=float)
    rotacion = np.zeros(numero_variables, dtype=float)

    for indice, nombre in enumerate(nombres):
        x, y, _ = poses[nombre]
        traslacion_x[3 * indice] = 1.0
        traslacion_y[3 * indice + 1] = 1.0
        rotacion[3 * indice] = -y
        rotacion[3 * indice + 1] = x
        rotacion[3 * indice + 2] = 1.0

    return {
        "translation_x": traslacion_x,
        "translation_y": traslacion_y,
        "rotation": rotacion,
    }


def analizar_rango_y_nulidad(jacobiano, tolerancia=1e-5):
    """Calcula rango, nulidad, Hessiana aproximada y valores singulares."""

    jacobiano = np.asarray(jacobiano, dtype=float)

    if jacobiano.ndim != 2:
        raise ValueError("El jacobiano debe ser bidimensional.")

    singulares = np.linalg.svd(jacobiano, compute_uv=False)
    rango = int(np.sum(singulares > tolerancia))
    nulidad = int(jacobiano.shape[1] - rango)
    hessiana = jacobiano.T @ jacobiano
    autovalores_hessiana = np.linalg.eigvalsh(hessiana)

    return {
        "shape": jacobiano.shape,
        "rank": rango,
        "nullity": nulidad,
        "singular_values": singulares,
        "hessian": hessiana,
        "hessian_eigenvalues": autovalores_hessiana,
        "condition_nonzero": float(
            singulares[0] / singulares[rango - 1]
        ) if rango > 0 else float("inf"),
    }


def analizar_observabilidad(graph_without_prior, graph_with_prior, poses):
    """Compara rango y espacio nulo antes y después de añadir el prior."""

    jacobiano_sin = construir_jacobiano_numerico(
        graph_without_prior,
        poses,
        incluir_prior=False,
    )
    jacobiano_con = construir_jacobiano_numerico(
        graph_with_prior,
        poses,
        incluir_prior=True,
    )

    analisis_sin = analizar_rango_y_nulidad(jacobiano_sin)
    analisis_con = analizar_rango_y_nulidad(jacobiano_con)

    nombres = obtener_nombres_poses(graph_without_prior)
    direcciones = construir_direcciones_gauge(poses, nombres)
    normas = {
        nombre: float(np.linalg.norm(jacobiano_sin @ direccion))
        for nombre, direccion in direcciones.items()
    }

    return {
        "jacobian_without_prior": jacobiano_sin,
        "jacobian_with_prior": jacobiano_con,
        "without_prior": analisis_sin,
        "with_prior": analisis_con,
        "gauge_directions": direcciones,
        "gauge_projection_norms": normas,
    }


# ---------------------------------------------------------------------------
# Configuraciones equivalentes y evaluación completa
# ---------------------------------------------------------------------------


def crear_configuraciones_equivalentes(
    graph_without_prior,
    graph_with_prior,
    poses_base,
):
    """Crea tres copias globales del mismo pose graph y evalúa sus costes."""

    pose_prior, informacion_prior = obtener_datos_prior(graph_with_prior)
    especificaciones = [
        (
            "A",
            "Original",
            np.array([0.0, 0.0, 0.0], dtype=float),
        ),
        (
            "B",
            "Trasladada",
            TRANSFORMACION_TRASLADADA,
        ),
        (
            "C",
            "Rotada y trasladada",
            TRANSFORMACION_ROTADA,
        ),
    ]

    configuraciones = []

    for identificador, nombre, transformacion in especificaciones:
        poses = aplicar_transformacion_global(poses_base, transformacion)
        sin_prior = calcular_coste_total(graph_without_prior, poses)
        con_prior = calcular_coste_total(
            graph_without_prior,
            poses,
            pose_prior=pose_prior,
            informacion_prior=informacion_prior,
        )

        configuraciones.append(
            {
                "id": identificador,
                "name": nombre,
                "transform": transformacion.copy(),
                "poses": poses,
                "relative_cost": sin_prior["relative"]["cost"],
                "prior_residual": con_prior["prior"]["residual"],
                "prior_cost": con_prior["prior"]["cost"],
                "total_cost": con_prior["total_cost"],
            }
        )

    return configuraciones


def evaluar_transformacion_dinamica(
    graph_without_prior,
    graph_with_prior,
    poses_base,
    transformacion,
):
    """Evalúa una transformación global intermedia para la animación."""

    poses = aplicar_transformacion_global(poses_base, transformacion)
    pose_prior, informacion_prior = obtener_datos_prior(graph_with_prior)
    evaluacion = calcular_coste_total(
        graph_without_prior,
        poses,
        pose_prior=pose_prior,
        informacion_prior=informacion_prior,
    )

    return {
        "transform": transformacion.copy(),
        "poses": poses,
        "relative_cost": evaluacion["relative"]["cost"],
        "prior_residual": evaluacion["prior"]["residual"],
        "prior_cost": evaluacion["prior"]["cost"],
        "total_cost": evaluacion["total_cost"],
    }


# ---------------------------------------------------------------------------
# Estados de la animación
# ---------------------------------------------------------------------------


def _serializar_vector(vector):
    return [float(valor) for valor in np.asarray(vector, dtype=float)]


def _serializar_poses(poses):
    return {
        nombre: _serializar_vector(pose)
        for nombre, pose in poses.items()
    }


def _serializar_configuracion(configuracion):
    return {
        "id": configuracion["id"],
        "name": configuracion["name"],
        "transform": _serializar_vector(configuracion["transform"]),
        "poses": _serializar_poses(configuracion["poses"]),
        "relative_cost": float(configuracion["relative_cost"]),
        "prior_residual": _serializar_vector(
            configuracion["prior_residual"]
        ),
        "prior_cost": float(configuracion["prior_cost"]),
        "total_cost": float(configuracion["total_cost"]),
    }


def crear_estado_animacion(
    *,
    phase,
    message,
    graph_without_prior,
    graph_with_prior,
    configurations,
    observability,
    dynamic=None,
    visible_pose_count=4,
    visible_edge_count=4,
    show_original=False,
    show_translated=False,
    show_rotated=False,
    show_dynamic=False,
    show_equal_cost=False,
    show_gauge=False,
    show_prior=False,
    show_prior_costs=False,
    show_cost_comparison=False,
    show_rank_without=False,
    show_rank_with=False,
    show_connections=False,
):
    """Convierte todos los resultados en un fotograma independiente."""

    pose_prior, informacion_prior = obtener_datos_prior(graph_with_prior)

    if dynamic is None:
        dynamic = configurations[0]

    relative_edges = []
    for _, _, datos in graph_without_prior.edges(data=True):
        relative_edges.append(
            {
                "origin": datos["origin"],
                "target": datos["target"],
                "sensor": datos["sensor"],
                "measurement": _serializar_vector(datos["measurement"]),
            }
        )

    return {
        "phase": phase,
        "message": message,
        "base_poses": _serializar_poses(POSES_BASE),
        "relative_edges": relative_edges,
        "configurations": [
            _serializar_configuracion(configuracion)
            for configuracion in configurations
        ],
        "dynamic": {
            "transform": _serializar_vector(dynamic["transform"]),
            "poses": _serializar_poses(dynamic["poses"]),
            "relative_cost": float(dynamic["relative_cost"]),
            "prior_residual": _serializar_vector(dynamic["prior_residual"]),
            "prior_cost": float(dynamic["prior_cost"]),
            "total_cost": float(dynamic["total_cost"]),
        },
        "prior_pose": _serializar_vector(pose_prior),
        "prior_information_diag": _serializar_vector(
            np.diag(informacion_prior)
        ),
        "rank_without_prior": int(
            observability["without_prior"]["rank"]
        ),
        "nullity_without_prior": int(
            observability["without_prior"]["nullity"]
        ),
        "shape_without_prior": list(
            observability["without_prior"]["shape"]
        ),
        "rank_with_prior": int(observability["with_prior"]["rank"]),
        "nullity_with_prior": int(
            observability["with_prior"]["nullity"]
        ),
        "shape_with_prior": list(observability["with_prior"]["shape"]),
        "singular_values_without": _serializar_vector(
            observability["without_prior"]["singular_values"]
        ),
        "singular_values_with": _serializar_vector(
            observability["with_prior"]["singular_values"]
        ),
        "gauge_projection_norms": {
            nombre: float(valor)
            for nombre, valor in observability[
                "gauge_projection_norms"
            ].items()
        },
        "visible_pose_count": int(visible_pose_count),
        "visible_edge_count": int(visible_edge_count),
        "show_original": bool(show_original),
        "show_translated": bool(show_translated),
        "show_rotated": bool(show_rotated),
        "show_dynamic": bool(show_dynamic),
        "show_equal_cost": bool(show_equal_cost),
        "show_gauge": bool(show_gauge),
        "show_prior": bool(show_prior),
        "show_prior_costs": bool(show_prior_costs),
        "show_cost_comparison": bool(show_cost_comparison),
        "show_rank_without": bool(show_rank_without),
        "show_rank_with": bool(show_rank_with),
        "show_connections": bool(show_connections),
    }


def crear_estados_animacion(
    graph_without_prior,
    graph_with_prior,
    configurations,
    observability,
):
    """Crea una secuencia didáctica completa para el apartado 5.5."""

    states = []

    def add(phase, message, repeat=1, **flags):
        for _ in range(repeat):
            states.append(
                crear_estado_animacion(
                    phase=phase,
                    message=message,
                    graph_without_prior=graph_without_prior,
                    graph_with_prior=graph_with_prior,
                    configurations=configurations,
                    observability=observability,
                    **flags,
                )
            )

    add(
        "introduction",
        "Las restricciones relativas fijan la forma, pero no el marco global.",
        repeat=3,
    )

    for count in range(1, 5):
        add(
            "build_poses",
            f"Se añade la pose x{count - 1} al conjunto de variables.",
            repeat=2,
            visible_pose_count=count,
            visible_edge_count=0,
            show_original=True,
        )

    for count in range(1, 5):
        add(
            "build_edges",
            "Las aristas solo miden transformaciones relativas entre poses.",
            repeat=2,
            visible_pose_count=4,
            visible_edge_count=count,
            show_original=True,
        )

    add(
        "relative_cost",
        "La geometría interna ya tiene un coste relativo bien definido.",
        repeat=3,
        show_original=True,
        show_cost_comparison=True,
    )

    for progress in np.linspace(0.0, 1.0, 7):
        transform = interpolar_transformacion(
            TRANSFORMACION_TRASLADADA,
            progress,
        )
        dynamic = evaluar_transformacion_dinamica(
            graph_without_prior,
            graph_with_prior,
            POSES_BASE,
            transform,
        )
        states.append(
            crear_estado_animacion(
                phase="translate_gauge",
                message=(
                    "Todas las poses se trasladan juntas y el coste relativo "
                    "permanece constante."
                ),
                graph_without_prior=graph_without_prior,
                graph_with_prior=graph_with_prior,
                configurations=configurations,
                observability=observability,
                dynamic=dynamic,
                show_original=True,
                show_dynamic=True,
                show_equal_cost=True,
            )
        )

    for progress in np.linspace(0.0, 1.0, 8):
        transform = interpolar_transformacion(
            TRANSFORMACION_ROTADA,
            progress,
        )
        dynamic = evaluar_transformacion_dinamica(
            graph_without_prior,
            graph_with_prior,
            POSES_BASE,
            transform,
        )
        states.append(
            crear_estado_animacion(
                phase="rotate_gauge",
                message=(
                    "Una rotación y una traslación global tampoco cambian "
                    "las predicciones relativas."
                ),
                graph_without_prior=graph_without_prior,
                graph_with_prior=graph_with_prior,
                configurations=configurations,
                observability=observability,
                dynamic=dynamic,
                show_original=True,
                show_dynamic=True,
                show_equal_cost=True,
                show_gauge=True,
            )
        )

    add(
        "equivalent_solutions",
        "A, B y C son representaciones globales distintas de la misma forma relativa.",
        repeat=4,
        show_original=True,
        show_translated=True,
        show_rotated=True,
        show_equal_cost=True,
        show_gauge=True,
        show_cost_comparison=True,
    )

    add(
        "nullspace",
        "Sin prior aparecen tres direcciones no observables: tx, ty y rotación.",
        repeat=4,
        show_original=True,
        show_translated=True,
        show_rotated=True,
        show_equal_cost=True,
        show_gauge=True,
        show_cost_comparison=True,
        show_rank_without=True,
    )

    add(
        "add_prior",
        "Se añade un factor unario que relaciona x0 con el sistema mundial.",
        repeat=4,
        show_original=True,
        show_translated=True,
        show_rotated=True,
        show_equal_cost=True,
        show_gauge=True,
        show_prior=True,
        show_cost_comparison=True,
        show_rank_without=True,
    )

    add(
        "prior_penalty",
        "El coste relativo sigue igual, pero las copias desplazadas incumplen el prior.",
        repeat=5,
        show_original=True,
        show_translated=True,
        show_rotated=True,
        show_equal_cost=True,
        show_gauge=True,
        show_prior=True,
        show_prior_costs=True,
        show_cost_comparison=True,
        show_rank_without=True,
    )

    add(
        "anchored_solution",
        "La configuración A define el origen y la orientación del mapa.",
        repeat=4,
        show_original=True,
        show_translated=True,
        show_rotated=True,
        show_prior=True,
        show_prior_costs=True,
        show_cost_comparison=True,
        show_rank_without=True,
        show_rank_with=True,
    )

    add(
        "physical_priors",
        "GPS, landmarks conocidos o rumbo absoluto pueden aportar priors físicos.",
        repeat=3,
        show_original=True,
        show_prior=True,
        show_prior_costs=True,
        show_cost_comparison=True,
        show_rank_without=True,
        show_rank_with=True,
        show_connections=True,
    )

    add(
        "summary",
        "Sin prior el grafo flota; con prior se expresa en coordenadas concretas.",
        repeat=4,
        show_original=True,
        show_translated=True,
        show_rotated=True,
        show_equal_cost=True,
        show_gauge=True,
        show_prior=True,
        show_prior_costs=True,
        show_cost_comparison=True,
        show_rank_without=True,
        show_rank_with=True,
        show_connections=True,
    )

    for index, state in enumerate(states, start=1):
        state["step"] = index
        state["total_steps"] = len(states)

    return states


# ---------------------------------------------------------------------------
# Validaciones y salida
# ---------------------------------------------------------------------------


def validar_grafo_sin_prior(graph):
    """Valida la estructura y las matrices del pose graph relativo."""

    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise TypeError("El pose graph debe ser un nx.Graph no dirigido.")

    expected_nodes = set(POSES_BASE)
    if set(graph.nodes()) != expected_nodes:
        raise ValueError("El grafo sin prior debe contener x0, x1, x2 y x3.")
    if not nx.is_connected(graph):
        raise ValueError("El pose graph relativo debe ser conectado.")
    if graph.number_of_edges() != len(ARISTAS_RELATIVAS):
        raise ValueError("El número de restricciones relativas es incorrecto.")

    for node in expected_nodes:
        if graph.nodes[node].get("node_type") != "pose":
            raise ValueError(f"{node} debe representar una pose.")
        validar_pose(graph.nodes[node]["estimate"], node)

    for origen, destino, datos in graph.edges(data=True):
        if datos.get("edge_type") != "relative_pose":
            raise ValueError("El grafo sin prior solo puede contener aristas relativas.")
        if datos.get("origin") not in expected_nodes or datos.get("target") not in expected_nodes:
            raise ValueError("La orientación almacenada de una arista no es válida.")
        validar_pose(datos["measurement"], "medición relativa")
        covarianza = np.asarray(datos["covariance"], dtype=float)
        informacion = np.asarray(datos["information"], dtype=float)
        if not np.allclose(
            covarianza @ informacion,
            np.eye(3),
            atol=1e-10,
        ):
            raise ValueError("Sigma Omega debe ser aproximadamente la identidad.")


def validar_grafo_con_prior(graph):
    """Valida el factor unario y su conexión con x0."""

    if "prior_x0" not in graph:
        raise ValueError("Falta el factor prior_x0.")
    if graph.nodes["prior_x0"].get("node_type") != "prior_factor":
        raise ValueError("prior_x0 debe ser un factor prior.")
    if not graph.has_edge("prior_x0", "x0"):
        raise ValueError("El factor prior debe estar conectado a x0.")
    if graph.degree("prior_x0") != 1:
        raise ValueError("Un factor prior debe ser unario.")

    datos = graph.edges["prior_x0", "x0"]
    if datos.get("edge_type") != "prior":
        raise ValueError("La arista del factor debe tener tipo prior.")

    pose_prior, informacion = obtener_datos_prior(graph)
    validar_pose(pose_prior, "pose prior")
    if np.any(np.linalg.eigvalsh(informacion) <= 0.0):
        raise ValueError("La información del prior debe ser definida positiva.")


def validar_invariancia_global(graph, configurations, tolerancia=1e-8):
    """Comprueba que todas las copias tengan el mismo coste relativo."""

    costes = [configuracion["relative_cost"] for configuracion in configurations]

    if not np.allclose(costes, costes[0], atol=tolerancia, rtol=0.0):
        raise ValueError("El coste relativo cambió bajo una transformación global.")

    reference = configurations[0]["poses"]

    for configuration in configurations[1:]:
        for origen, destino, _ in ARISTAS_RELATIVAS:
            prediccion_referencia = calcular_prediccion_relativa(
                reference[origen],
                reference[destino],
            )
            prediccion_transformada = calcular_prediccion_relativa(
                configuration["poses"][origen],
                configuration["poses"][destino],
            )

            if not np.allclose(
                prediccion_referencia,
                prediccion_transformada,
                atol=1e-9,
            ):
                raise ValueError(
                    "Una predicción relativa cambió al mover todo el grafo."
                )


def validar_anclaje(configurations, observability):
    """Comprueba que el prior seleccione la configuración de referencia."""

    original, translated, rotated = configurations

    if original["prior_cost"] > 1e-10:
        raise ValueError("La configuración original debe satisfacer el prior.")
    if translated["prior_cost"] <= original["prior_cost"]:
        raise ValueError("La configuración trasladada debe incumplir el prior.")
    if rotated["prior_cost"] <= original["prior_cost"]:
        raise ValueError("La configuración rotada debe incumplir el prior.")

    if observability["without_prior"]["nullity"] != 3:
        raise ValueError("El grafo SE(2) sin prior debe tener nulidad tres.")
    if observability["with_prior"]["nullity"] != 0:
        raise ValueError("El prior completo debe eliminar la nulidad global.")

    for nombre, norma in observability["gauge_projection_norms"].items():
        if norma > 2e-5:
            raise ValueError(
                f"La dirección de gauge {nombre} no pertenece al espacio nulo."
            )


def validar_resultados(
    graph_without_prior,
    graph_with_prior,
    configurations,
    observability,
    states,
):
    """Ejecuta todas las comprobaciones matemáticas y didácticas."""

    validar_grafo_sin_prior(graph_without_prior)
    validar_grafo_con_prior(graph_with_prior)
    validar_invariancia_global(graph_without_prior, configurations)
    validar_anclaje(configurations, observability)

    if len(states) < 50:
        raise ValueError("La demostración debe contener al menos cincuenta estados.")
    if states[-1].get("phase") != "summary":
        raise ValueError("El último estado debe ser el resumen final.")
    if not states[-1].get("show_prior"):
        raise ValueError("La imagen final debe mostrar el factor prior.")
    if not states[-1].get("show_rank_with"):
        raise ValueError("La imagen final debe mostrar el rango con prior.")


def _formatear_pose(pose):
    pose = validar_pose(pose)
    return (
        f"({pose[0]:.3f} m, {pose[1]:.3f} m, "
        f"{np.rad2deg(pose[2]):.3f} deg)"
    )


def imprimir_resumen(
    graph_without_prior,
    graph_with_prior,
    configurations,
    observability,
    states,
):
    """Imprime las magnitudes principales del ejemplo."""

    print("\n=== Priors y anclaje del grafo ===")
    print(f"Poses: {graph_without_prior.number_of_nodes()}")
    print(f"Restricciones relativas: {graph_without_prior.number_of_edges()}")
    print("Pose prior:", _formatear_pose(POSE_PRIOR))

    print("\nConfiguraciones globales:")
    for configuration in configurations:
        transform = configuration["transform"]
        print(
            f"  {configuration['id']} · {configuration['name']}: "
            f"g={_formatear_pose(transform)}, "
            f"F_rel={configuration['relative_cost']:.9f}, "
            f"F_prior={configuration['prior_cost']:.6f}, "
            f"F_total={configuration['total_cost']:.6f}"
        )

    without_prior = observability["without_prior"]
    with_prior = observability["with_prior"]

    print("\nObservabilidad:")
    print(
        "  Sin prior: "
        f"J={without_prior['shape']}, "
        f"rango={without_prior['rank']}, "
        f"nulidad={without_prior['nullity']}"
    )
    print(
        "  Con prior: "
        f"J={with_prior['shape']}, "
        f"rango={with_prior['rank']}, "
        f"nulidad={with_prior['nullity']}"
    )
    print("  Normas Jv de las direcciones de gauge:")
    for nombre, norma in observability["gauge_projection_norms"].items():
        print(f"    {nombre}: {norma:.12e}")

    print(f"\nNodos del grafo anclado: {graph_with_prior.number_of_nodes()}")
    print(f"Aristas del grafo anclado: {graph_with_prior.number_of_edges()}")
    print(f"Estados de animación: {len(states)}")


def main():
    graph_without_prior = crear_grafo_sin_prior(POSES_BASE)
    graph_with_prior = agregar_prior_al_grafo(
        graph_without_prior,
        POSE_PRIOR,
        SIGMAS_PRIOR,
    )

    validar_grafo_sin_prior(graph_without_prior)
    validar_grafo_con_prior(graph_with_prior)

    configurations = crear_configuraciones_equivalentes(
        graph_without_prior,
        graph_with_prior,
        POSES_BASE,
    )
    observability = analizar_observabilidad(
        graph_without_prior,
        graph_with_prior,
        POSES_BASE,
    )
    states = crear_estados_animacion(
        graph_without_prior,
        graph_with_prior,
        configurations,
        observability,
    )

    validar_resultados(
        graph_without_prior,
        graph_with_prior,
        configurations,
        observability,
        states,
    )

    imprimir_resumen(
        graph_without_prior,
        graph_with_prior,
        configurations,
        observability,
        states,
    )

    animator = GraphAnimator(
        figsize=(18, 10),
        interval=560,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "05_optimizacion"
        / "05_prior_anclaje_grafo.png"
    )

    animator.animate_prior_graph_anchoring(
        graph_without_prior=graph_without_prior,
        graph_with_prior=graph_with_prior,
        states=states,
        title="Priors y anclaje del grafo",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
