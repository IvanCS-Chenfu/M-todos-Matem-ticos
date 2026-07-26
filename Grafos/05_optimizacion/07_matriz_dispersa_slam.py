from pathlib import Path
import sys

import networkx as nx
import numpy as np
from scipy import sparse


CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


# ---------------------------------------------------------------------------
# Datos deterministas del ejemplo
# ---------------------------------------------------------------------------

POSES_VERDADERAS = {
    "x0": np.array([0.00, 0.00, np.deg2rad(0.0)], dtype=float),
    "x1": np.array([1.35, 0.18, np.deg2rad(7.0)], dtype=float),
    "x2": np.array([2.62, 0.58, np.deg2rad(15.0)], dtype=float),
    "x3": np.array([3.70, 1.36, np.deg2rad(27.0)], dtype=float),
    "x4": np.array([4.25, 2.48, np.deg2rad(43.0)], dtype=float),
    "x5": np.array([3.55, 3.52, np.deg2rad(68.0)], dtype=float),
}

PERTURBACIONES_INICIALES = {
    "x0": np.array([0.00, 0.00, np.deg2rad(0.0)], dtype=float),
    "x1": np.array([0.06, -0.04, np.deg2rad(1.2)], dtype=float),
    "x2": np.array([0.10, -0.03, np.deg2rad(1.8)], dtype=float),
    "x3": np.array([0.12, 0.05, np.deg2rad(2.4)], dtype=float),
    "x4": np.array([0.05, 0.11, np.deg2rad(2.0)], dtype=float),
    "x5": np.array([-0.04, 0.12, np.deg2rad(1.5)], dtype=float),
}

POSE_PRIOR = np.array([0.0, 0.0, 0.0], dtype=float)
SIGMAS_PRIOR = np.array([0.08, 0.08, np.deg2rad(1.5)], dtype=float)
SIGMAS_ODOMETRIA = np.array([0.12, 0.12, np.deg2rad(3.0)], dtype=float)
SIGMAS_CIERRE = np.array([0.09, 0.09, np.deg2rad(2.0)], dtype=float)

ORDEN_FACTORES = [
    "prior_x0",
    "odom_01",
    "odom_12",
    "odom_23",
    "odom_34",
    "odom_45",
    "loop_50",
    "loop_14",
]

ORDEN_VARIABLES = ["x0", "x1", "x2", "x3", "x4", "x5"]

RUIDO_FACTORES = {
    "odom_01": np.array([0.015, -0.010, np.deg2rad(0.20)], dtype=float),
    "odom_12": np.array([-0.010, 0.008, np.deg2rad(-0.15)], dtype=float),
    "odom_23": np.array([0.012, 0.006, np.deg2rad(0.18)], dtype=float),
    "odom_34": np.array([-0.008, -0.012, np.deg2rad(-0.12)], dtype=float),
    "odom_45": np.array([0.006, 0.010, np.deg2rad(0.10)], dtype=float),
    "loop_50": np.array([0.045, -0.030, np.deg2rad(0.65)], dtype=float),
    "loop_14": np.array([-0.035, 0.025, np.deg2rad(-0.55)], dtype=float),
}

UMBRAL_NO_NULO = 1e-10
EPSILON_JACOBIANO = 1e-7


# ---------------------------------------------------------------------------
# Operaciones en SE(2)
# ---------------------------------------------------------------------------


def normalizar_angulo(angulo):
    """Normaliza un ángulo al intervalo [-pi, pi)."""

    angulo = float(angulo)
    return (angulo + np.pi) % (2.0 * np.pi) - np.pi


def validar_pose(pose, nombre="pose"):
    """Valida una pose (x, y, theta) y normaliza su orientación."""

    pose = np.asarray(pose, dtype=float)

    if pose.shape != (3,):
        raise ValueError(f"{nombre} debe contener exactamente tres componentes.")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{nombre} debe contener valores finitos.")

    resultado = pose.copy()
    resultado[2] = normalizar_angulo(resultado[2])
    return resultado


def pose_a_matriz_se2(pose):
    """Convierte una pose 2D en una matriz homogénea 3x3."""

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
    """Convierte una matriz homogénea válida en una pose 2D."""

    matriz = np.asarray(matriz, dtype=float)

    if matriz.shape != (3, 3):
        raise ValueError("La matriz de SE(2) debe tener forma 3x3.")
    if not np.all(np.isfinite(matriz)):
        raise ValueError("La matriz de SE(2) debe contener valores finitos.")
    if not np.allclose(matriz[2], [0.0, 0.0, 1.0], atol=1e-10):
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


def calcular_prediccion_relativa(pose_origen, pose_destino):
    """Calcula x_i^{-1} ⊕ x_j."""

    return componer_poses_se2(
        invertir_pose_se2(pose_origen),
        pose_destino,
    )


def calcular_residuo_relativo(pose_origen, pose_destino, medicion):
    """Calcula z_ij^{-1} ⊕ (x_i^{-1} ⊕ x_j)."""

    prediccion = calcular_prediccion_relativa(
        pose_origen,
        pose_destino,
    )
    error = componer_poses_se2(
        invertir_pose_se2(medicion),
        prediccion,
    )
    error[2] = normalizar_angulo(error[2])
    return error


def calcular_residuo_prior(pose, medicion_prior):
    """Calcula prior^{-1} ⊕ x."""

    error = componer_poses_se2(
        invertir_pose_se2(medicion_prior),
        pose,
    )
    error[2] = normalizar_angulo(error[2])
    return error


# ---------------------------------------------------------------------------
# Construcción del pose graph
# ---------------------------------------------------------------------------


def crear_covarianza_diagonal(sigmas):
    """Construye una covarianza diagonal a partir de desviaciones estándar."""

    sigmas = np.asarray(sigmas, dtype=float)

    if sigmas.shape != (3,):
        raise ValueError("Se necesitan tres desviaciones estándar.")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("Las desviaciones estándar deben ser positivas y finitas.")

    return np.diag(sigmas**2)


def calcular_matriz_informacion(covarianza):
    """Invierte una covarianza simétrica definida positiva."""

    covarianza = np.asarray(covarianza, dtype=float)

    if covarianza.shape != (3, 3):
        raise ValueError("La covarianza debe tener forma 3x3.")
    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise ValueError("La covarianza debe ser simétrica.")
    if np.any(np.linalg.eigvalsh(covarianza) <= 0.0):
        raise ValueError("La covarianza debe ser definida positiva.")

    return np.linalg.inv(covarianza)


def crear_estimaciones_iniciales():
    """Crea estimaciones ligeramente perturbadas respecto a la trayectoria real."""

    estimaciones = {}

    for nombre in ORDEN_VARIABLES:
        estimaciones[nombre] = validar_pose(
            POSES_VERDADERAS[nombre] + PERTURBACIONES_INICIALES[nombre],
            nombre=f"estimación {nombre}",
        )

    return estimaciones


def crear_medicion_relativa(origen, destino, ruido):
    """Genera una medición relativa determinista a partir de poses verdaderas."""

    medicion_ideal = calcular_prediccion_relativa(
        POSES_VERDADERAS[origen],
        POSES_VERDADERAS[destino],
    )
    return componer_poses_se2(medicion_ideal, ruido)


def crear_pose_graph():
    """Construye seis poses, un prior, odometría y dos cierres de ciclo."""

    graph = nx.Graph()
    estimaciones = crear_estimaciones_iniciales()

    for nombre in ORDEN_VARIABLES:
        graph.add_node(
            nombre,
            node_type="pose",
            dimension=3,
            estimate=estimaciones[nombre].copy(),
            true_pose=POSES_VERDADERAS[nombre].copy(),
            label=nombre,
        )

    cov_prior = crear_covarianza_diagonal(SIGMAS_PRIOR)
    info_prior = calcular_matriz_informacion(cov_prior)

    graph.add_node(
        "prior_x0",
        node_type="prior_factor",
        factor_type="unary_prior",
        dimension=3,
        measurement=POSE_PRIOR.copy(),
        covariance=cov_prior,
        information=info_prior,
        label="prior",
    )
    graph.add_edge(
        "prior_x0",
        "x0",
        factor_name="prior_x0",
        factor_type="prior",
        variables=("x0",),
        measurement=POSE_PRIOR.copy(),
        covariance=cov_prior,
        information=info_prior,
    )

    factores_binarios = [
        ("odom_01", "x0", "x1", "odometry"),
        ("odom_12", "x1", "x2", "odometry"),
        ("odom_23", "x2", "x3", "odometry"),
        ("odom_34", "x3", "x4", "odometry"),
        ("odom_45", "x4", "x5", "odometry"),
        ("loop_50", "x5", "x0", "loop_closure"),
        ("loop_14", "x1", "x4", "loop_closure"),
    ]

    for factor_name, origen, destino, factor_type in factores_binarios:
        sigmas = (
            SIGMAS_ODOMETRIA
            if factor_type == "odometry"
            else SIGMAS_CIERRE
        )
        covarianza = crear_covarianza_diagonal(sigmas)
        informacion = calcular_matriz_informacion(covarianza)
        medicion = crear_medicion_relativa(
            origen,
            destino,
            RUIDO_FACTORES[factor_name],
        )

        graph.add_edge(
            origen,
            destino,
            factor_name=factor_name,
            factor_type=factor_type,
            variables=(origen, destino),
            measurement=medicion,
            covariance=covarianza,
            information=informacion,
        )

    graph.graph.update(
        {
            "factor_order": list(ORDEN_FACTORES),
            "variable_order": list(ORDEN_VARIABLES),
            "state_dimension": 3 * len(ORDEN_VARIABLES),
            "residual_dimension": 3 * len(ORDEN_FACTORES),
            "reference_frame": "x0 anclada mediante prior",
        }
    )

    return graph


def obtener_estimaciones(graph):
    """Extrae las poses optimizables en un diccionario independiente."""

    return {
        nombre: validar_pose(graph.nodes[nombre]["estimate"], nombre)
        for nombre in ORDEN_VARIABLES
    }


def obtener_factor(graph, factor_name):
    """Recupera los datos de un factor usando su nombre estable."""

    if factor_name == "prior_x0":
        datos = graph.get_edge_data("prior_x0", "x0")
        return dict(datos)

    for origen, destino, datos in graph.edges(data=True):
        if datos.get("factor_name") == factor_name:
            resultado = dict(datos)
            resultado["origin"] = origen
            resultado["target"] = destino
            return resultado

    raise KeyError(f"No existe el factor {factor_name!r}.")


# ---------------------------------------------------------------------------
# Residuos y jacobianos locales
# ---------------------------------------------------------------------------


def calcular_residuo_factor(graph, factor_name, estimaciones):
    """Evalúa el residuo de un factor con las estimaciones indicadas."""

    factor = obtener_factor(graph, factor_name)

    if factor_name == "prior_x0":
        return calcular_residuo_prior(
            estimaciones["x0"],
            factor["measurement"],
        )

    origen, destino = factor["variables"]
    return calcular_residuo_relativo(
        estimaciones[origen],
        estimaciones[destino],
        factor["measurement"],
    )


def calcular_jacobiano_local_numerico(
    graph,
    factor_name,
    estimaciones,
    epsilon=EPSILON_JACOBIANO,
):
    """Calcula los bloques locales del factor mediante diferencias centrales."""

    epsilon = float(epsilon)

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon debe ser positivo y finito.")

    factor = obtener_factor(graph, factor_name)
    variables = tuple(factor["variables"])
    bloques = {}

    for variable in variables:
        bloque = np.zeros((3, 3), dtype=float)

        for componente in range(3):
            positivas = {
                nombre: pose.copy()
                for nombre, pose in estimaciones.items()
            }
            negativas = {
                nombre: pose.copy()
                for nombre, pose in estimaciones.items()
            }

            positivas[variable][componente] += epsilon
            negativas[variable][componente] -= epsilon
            positivas[variable][2] = normalizar_angulo(
                positivas[variable][2]
            )
            negativas[variable][2] = normalizar_angulo(
                negativas[variable][2]
            )

            residuo_positivo = calcular_residuo_factor(
                graph,
                factor_name,
                positivas,
            )
            residuo_negativo = calcular_residuo_factor(
                graph,
                factor_name,
                negativas,
            )

            diferencia = residuo_positivo - residuo_negativo
            diferencia[2] = normalizar_angulo(diferencia[2])
            bloque[:, componente] = diferencia / (2.0 * epsilon)

        bloques[variable] = bloque

    return bloques


def ensamblar_sistema_disperso(graph, estimaciones):
    """Ensambla e, J, Omega, H=JᵀOmegaJ y g=JᵀOmegae."""

    numero_factores = len(ORDEN_FACTORES)
    numero_variables = len(ORDEN_VARIABLES)
    dimension_residuo = 3 * numero_factores
    dimension_estado = 3 * numero_variables

    residuos = np.zeros(dimension_residuo, dtype=float)
    jacobiano = sparse.lil_matrix(
        (dimension_residuo, dimension_estado),
        dtype=float,
    )
    bloques_informacion = []
    jacobianos_locales = {}
    dependencias = {}

    indices_variables = {
        nombre: 3 * indice
        for indice, nombre in enumerate(ORDEN_VARIABLES)
    }

    for indice_factor, factor_name in enumerate(ORDEN_FACTORES):
        fila = 3 * indice_factor
        factor = obtener_factor(graph, factor_name)
        residuo = calcular_residuo_factor(
            graph,
            factor_name,
            estimaciones,
        )
        bloques = calcular_jacobiano_local_numerico(
            graph,
            factor_name,
            estimaciones,
        )

        residuos[fila : fila + 3] = residuo
        jacobianos_locales[factor_name] = {
            nombre: bloque.copy()
            for nombre, bloque in bloques.items()
        }
        dependencias[factor_name] = tuple(factor["variables"])

        for variable, bloque in bloques.items():
            columna = indices_variables[variable]
            jacobiano[
                fila : fila + 3,
                columna : columna + 3,
            ] = bloque

        bloques_informacion.append(
            sparse.csr_matrix(factor["information"])
        )

    jacobiano = jacobiano.tocsr()
    informacion = sparse.block_diag(
        bloques_informacion,
        format="csr",
    )
    hessiana = (jacobiano.T @ informacion @ jacobiano).tocsr()
    gradiente = np.asarray(
        jacobiano.T @ (informacion @ residuos),
        dtype=float,
    ).reshape(-1)

    coste = 0.5 * float(residuos @ (informacion @ residuos))

    return {
        "residuals": residuos,
        "jacobian": jacobiano,
        "information": informacion,
        "hessian": hessiana,
        "gradient": gradiente,
        "cost": coste,
        "local_jacobians": jacobianos_locales,
        "dependencies": dependencias,
        "variable_indices": indices_variables,
    }


def construir_jacobiano_global_numerico(
    graph,
    estimaciones,
    incluir_prior=True,
    epsilon=EPSILON_JACOBIANO,
):
    """Calcula un jacobiano global independiente para validar el ensamblaje."""

    factores = (
        list(ORDEN_FACTORES)
        if incluir_prior
        else list(ORDEN_FACTORES[1:])
    )
    vector_base = np.concatenate(
        [estimaciones[nombre] for nombre in ORDEN_VARIABLES]
    )

    def vector_residuos(vector_estado):
        poses = {}
        for indice, nombre in enumerate(ORDEN_VARIABLES):
            inicio = 3 * indice
            poses[nombre] = validar_pose(
                vector_estado[inicio : inicio + 3],
                nombre,
            )
        return np.concatenate(
            [
                calcular_residuo_factor(graph, factor, poses)
                for factor in factores
            ]
        )

    jacobiano = np.zeros(
        (3 * len(factores), vector_base.size),
        dtype=float,
    )

    for columna in range(vector_base.size):
        positivo = vector_base.copy()
        negativo = vector_base.copy()
        positivo[columna] += epsilon
        negativo[columna] -= epsilon

        residuo_positivo = vector_residuos(positivo)
        residuo_negativo = vector_residuos(negativo)
        diferencia = residuo_positivo - residuo_negativo

        for inicio in range(2, diferencia.size, 3):
            diferencia[inicio] = normalizar_angulo(diferencia[inicio])

        jacobiano[:, columna] = diferencia / (2.0 * epsilon)

    return jacobiano


# ---------------------------------------------------------------------------
# Patrones de dispersión y eliminación simbólica
# ---------------------------------------------------------------------------


def calcular_patron_dispersion(matriz, umbral=UMBRAL_NO_NULO):
    """Devuelve una matriz binaria que marca los elementos no nulos."""

    if sparse.issparse(matriz):
        densa = matriz.toarray()
    else:
        densa = np.asarray(matriz, dtype=float)

    if densa.ndim != 2:
        raise ValueError("La matriz debe ser bidimensional.")

    return np.abs(densa) > float(umbral)


def calcular_metricas_dispersion(matriz, umbral=UMBRAL_NO_NULO):
    """Calcula forma, nnz, densidad y dispersión."""

    patron = calcular_patron_dispersion(matriz, umbral)
    total = int(patron.size)
    nnz = int(np.count_nonzero(patron))
    densidad = nnz / total if total else 0.0

    return {
        "shape": tuple(int(valor) for valor in patron.shape),
        "nnz": nnz,
        "total": total,
        "density": float(densidad),
        "sparsity": float(1.0 - densidad),
    }


def calcular_patron_bloques_jacobiano(dependencias):
    """Marca qué factores dependen de qué variables."""

    patron = np.zeros(
        (len(ORDEN_FACTORES), len(ORDEN_VARIABLES)),
        dtype=bool,
    )

    for fila, factor_name in enumerate(ORDEN_FACTORES):
        for variable in dependencias[factor_name]:
            patron[fila, ORDEN_VARIABLES.index(variable)] = True

    return patron


def calcular_patron_bloques_hessiana(graph):
    """Construye el patrón pose-pose inducido por los factores."""

    patron = np.eye(len(ORDEN_VARIABLES), dtype=bool)

    for factor_name in ORDEN_FACTORES[1:]:
        factor = obtener_factor(graph, factor_name)
        origen, destino = factor["variables"]
        i = ORDEN_VARIABLES.index(origen)
        j = ORDEN_VARIABLES.index(destino)
        patron[i, j] = True
        patron[j, i] = True

    return patron


def construir_grafo_hessiana(graph):
    """Crea el grafo de adyacencia por bloques de la Hessiana."""

    h_graph = nx.Graph()
    h_graph.add_nodes_from(ORDEN_VARIABLES)

    for factor_name in ORDEN_FACTORES[1:]:
        factor = obtener_factor(graph, factor_name)
        origen, destino = factor["variables"]
        h_graph.add_edge(origen, destino, factor_name=factor_name)

    return h_graph


def simular_eliminacion_simbolica(h_graph, orden):
    """Simula eliminación por bloques y cuenta las aristas de fill-in."""

    orden = list(orden)

    if set(orden) != set(h_graph.nodes()) or len(orden) != h_graph.number_of_nodes():
        raise ValueError("El orden debe contener todas las variables una vez.")

    trabajo = nx.Graph(h_graph)
    fill_edges = []
    history = []

    for paso, variable in enumerate(orden):
        vecinos = sorted(trabajo.neighbors(variable))
        nuevos = []

        for indice, origen in enumerate(vecinos):
            for destino in vecinos[indice + 1 :]:
                if not trabajo.has_edge(origen, destino):
                    trabajo.add_edge(origen, destino, fill_in=True)
                    arista = tuple(sorted((origen, destino)))
                    fill_edges.append(arista)
                    nuevos.append(arista)

        history.append(
            {
                "step": paso,
                "variable": variable,
                "neighbors": vecinos,
                "new_fill_edges": nuevos,
                "cumulative_fill_edges": list(fill_edges),
            }
        )
        trabajo.remove_node(variable)

    return {
        "order": orden,
        "fill_edges": fill_edges,
        "fill_count": len(fill_edges),
        "history": history,
    }


def calcular_orden_minimo_grado(h_graph):
    """Calcula una heurística sencilla de mínimo grado dinámico."""

    trabajo = nx.Graph(h_graph)
    orden = []

    while trabajo.nodes:
        variable = min(
            trabajo.nodes,
            key=lambda nodo: (trabajo.degree[nodo], nodo),
        )
        vecinos = list(trabajo.neighbors(variable))

        for indice, origen in enumerate(vecinos):
            for destino in vecinos[indice + 1 :]:
                trabajo.add_edge(origen, destino)

        orden.append(variable)
        trabajo.remove_node(variable)

    return orden


def analizar_rango_y_gauge(graph, estimaciones, sistema):
    """Compara el rango del jacobiano con y sin prior."""

    jacobiano_con_prior = sistema["jacobian"].toarray()
    jacobiano_sin_prior = construir_jacobiano_global_numerico(
        graph,
        estimaciones,
        incluir_prior=False,
    )

    rango_con = int(np.linalg.matrix_rank(jacobiano_con_prior, tol=1e-7))
    rango_sin = int(np.linalg.matrix_rank(jacobiano_sin_prior, tol=1e-7))
    dimension_estado = jacobiano_con_prior.shape[1]

    return {
        "with_prior": {
            "rank": rango_con,
            "nullity": dimension_estado - rango_con,
            "shape": jacobiano_con_prior.shape,
        },
        "without_prior": {
            "rank": rango_sin,
            "nullity": dimension_estado - rango_sin,
            "shape": jacobiano_sin_prior.shape,
        },
    }


def analizar_estructura_dispersa(graph, sistema, estimaciones):
    """Calcula métricas escalares, por bloques, rango y fill-in."""

    j_metrics = calcular_metricas_dispersion(sistema["jacobian"])
    h_metrics = calcular_metricas_dispersion(sistema["hessian"])
    j_block_pattern = calcular_patron_bloques_jacobiano(
        sistema["dependencies"]
    )
    h_block_pattern = calcular_patron_bloques_hessiana(graph)

    h_graph = construir_grafo_hessiana(graph)
    orden_bueno = calcular_orden_minimo_grado(h_graph)
    orden_malo = ["x1", "x4", "x0", "x2", "x3", "x5"]
    eliminacion_buena = simular_eliminacion_simbolica(
        h_graph,
        orden_bueno,
    )
    eliminacion_mala = simular_eliminacion_simbolica(
        h_graph,
        orden_malo,
    )

    h_densa = sistema["hessian"].toarray()
    valores_singulares = np.linalg.svd(h_densa, compute_uv=False)
    positivos = valores_singulares[valores_singulares > 1e-10]
    condicion = (
        float(positivos[0] / positivos[-1])
        if positivos.size
        else float("inf")
    )

    return {
        "jacobian_metrics": j_metrics,
        "hessian_metrics": h_metrics,
        "jacobian_pattern": calcular_patron_dispersion(
            sistema["jacobian"]
        ),
        "hessian_pattern": calcular_patron_dispersion(
            sistema["hessian"]
        ),
        "jacobian_block_pattern": j_block_pattern,
        "hessian_block_pattern": h_block_pattern,
        "block_jacobian_nnz": int(np.count_nonzero(j_block_pattern)),
        "block_jacobian_total": int(j_block_pattern.size),
        "block_hessian_nnz": int(np.count_nonzero(h_block_pattern)),
        "block_hessian_total": int(h_block_pattern.size),
        "condition_number": condicion,
        "gauge": analizar_rango_y_gauge(graph, estimaciones, sistema),
        "good_elimination": eliminacion_buena,
        "bad_elimination": eliminacion_mala,
    }


# ---------------------------------------------------------------------------
# Estados didácticos
# ---------------------------------------------------------------------------


def _serializar_matriz(matriz):
    if sparse.issparse(matriz):
        matriz = matriz.toarray()
    return np.asarray(matriz).tolist()


def crear_estado_animacion(
    *,
    phase,
    message,
    graph,
    sistema,
    analysis,
    visible_poses=0,
    visible_factors=0,
    active_factor=None,
    show_jacobian=False,
    show_hessian=False,
    show_block_grid=False,
    show_fill_in=False,
    elimination=None,
    elimination_step=None,
    show_connections=False,
):
    """Crea un fotograma autocontenido sin dibujarlo."""

    factor_names = list(ORDEN_FACTORES)
    variable_names = list(ORDEN_VARIABLES)

    active_j_blocks = []
    active_h_blocks = []

    if active_factor in factor_names:
        fila = factor_names.index(active_factor)
        for variable in sistema["dependencies"][active_factor]:
            columna = variable_names.index(variable)
            active_j_blocks.append((fila, columna))
            active_h_blocks.append((columna, columna))

        variables = sistema["dependencies"][active_factor]
        if len(variables) == 2:
            i = variable_names.index(variables[0])
            j = variable_names.index(variables[1])
            active_h_blocks.extend([(i, j), (j, i)])

    fill_edges = []
    eliminated_variables = []
    active_elimination_variable = None

    if elimination is not None and elimination_step is not None:
        step = min(
            int(elimination_step),
            len(elimination["history"]) - 1,
        )
        record = elimination["history"][step]
        fill_edges = list(record["cumulative_fill_edges"])
        eliminated_variables = elimination["order"][:step]
        active_elimination_variable = record["variable"]

    poses = {
        nombre: _serializar_matriz(graph.nodes[nombre]["estimate"])
        for nombre in variable_names
    }

    factors = []
    for factor_name in factor_names:
        factor = obtener_factor(graph, factor_name)
        factors.append(
            {
                "name": factor_name,
                "type": factor["factor_type"],
                "variables": list(factor["variables"]),
            }
        )

    return {
        "phase": phase,
        "message": message,
        "poses": poses,
        "factors": factors,
        "factor_names": factor_names,
        "variable_names": variable_names,
        "visible_poses": int(visible_poses),
        "visible_factors": int(visible_factors),
        "active_factor": active_factor,
        "show_jacobian": bool(show_jacobian),
        "show_hessian": bool(show_hessian),
        "show_block_grid": bool(show_block_grid),
        "show_fill_in": bool(show_fill_in),
        "show_connections": bool(show_connections),
        "jacobian_pattern": _serializar_matriz(
            analysis["jacobian_pattern"]
        ),
        "hessian_pattern": _serializar_matriz(
            analysis["hessian_pattern"]
        ),
        "jacobian_block_pattern": _serializar_matriz(
            analysis["jacobian_block_pattern"]
        ),
        "hessian_block_pattern": _serializar_matriz(
            analysis["hessian_block_pattern"]
        ),
        "active_j_blocks": [list(item) for item in active_j_blocks],
        "active_h_blocks": [list(item) for item in active_h_blocks],
        "fill_edges": [list(item) for item in fill_edges],
        "eliminated_variables": list(eliminated_variables),
        "active_elimination_variable": active_elimination_variable,
        "metrics": {
            "state_dimension": int(sistema["jacobian"].shape[1]),
            "residual_dimension": int(sistema["jacobian"].shape[0]),
            "factor_count": len(factor_names),
            "pose_count": len(variable_names),
            "cost": float(sistema["cost"]),
            "gradient_norm": float(np.linalg.norm(sistema["gradient"])),
            "jacobian_shape": list(analysis["jacobian_metrics"]["shape"]),
            "jacobian_nnz": int(analysis["jacobian_metrics"]["nnz"]),
            "jacobian_density": float(
                analysis["jacobian_metrics"]["density"]
            ),
            "hessian_shape": list(analysis["hessian_metrics"]["shape"]),
            "hessian_nnz": int(analysis["hessian_metrics"]["nnz"]),
            "hessian_density": float(
                analysis["hessian_metrics"]["density"]
            ),
            "block_jacobian_nnz": int(
                analysis["block_jacobian_nnz"]
            ),
            "block_jacobian_total": int(
                analysis["block_jacobian_total"]
            ),
            "block_hessian_nnz": int(
                analysis["block_hessian_nnz"]
            ),
            "block_hessian_total": int(
                analysis["block_hessian_total"]
            ),
            "rank_without_prior": int(
                analysis["gauge"]["without_prior"]["rank"]
            ),
            "nullity_without_prior": int(
                analysis["gauge"]["without_prior"]["nullity"]
            ),
            "rank_with_prior": int(
                analysis["gauge"]["with_prior"]["rank"]
            ),
            "nullity_with_prior": int(
                analysis["gauge"]["with_prior"]["nullity"]
            ),
            "condition_number": float(analysis["condition_number"]),
            "good_fill_count": int(
                analysis["good_elimination"]["fill_count"]
            ),
            "bad_fill_count": int(
                analysis["bad_elimination"]["fill_count"]
            ),
            "good_order": list(
                analysis["good_elimination"]["order"]
            ),
            "bad_order": list(
                analysis["bad_elimination"]["order"]
            ),
        },
    }


def crear_estados_animacion(graph, sistema, analysis):
    """Crea una secuencia didáctica completa para el apartado 5.7."""

    states = []

    def add(phase, message, repeat=1, **kwargs):
        for _ in range(repeat):
            states.append(
                crear_estado_animacion(
                    phase=phase,
                    message=message,
                    graph=graph,
                    sistema=sistema,
                    analysis=analysis,
                    **kwargs,
                )
            )

    add(
        "introduction",
        "Cada factor depende de pocas variables: esa localidad crea matrices dispersas.",
        repeat=3,
    )

    for count in range(1, len(ORDEN_VARIABLES) + 1):
        add(
            "poses",
            "Se incorporan las poses que forman el vector global de estado.",
            visible_poses=count,
        )

    add(
        "prior",
        "El prior solo depende de x0 y ocupa un único bloque del jacobiano.",
        repeat=3,
        visible_poses=len(ORDEN_VARIABLES),
        visible_factors=1,
        active_factor="prior_x0",
        show_jacobian=True,
        show_block_grid=True,
    )

    for factor_index, factor_name in enumerate(ORDEN_FACTORES[1:], start=2):
        factor = obtener_factor(graph, factor_name)
        variables = factor["variables"]
        factor_type = factor["factor_type"]
        message = (
            f"{factor_name} conecta {variables[0]} y {variables[1]}: "
            "solo aparecen dos bloques en su fila."
        )
        if factor_type == "loop_closure":
            message = (
                f"{factor_name} es un cierre de ciclo: añade bloques "
                "alejados en la estructura."
            )

        add(
            "factor",
            message,
            repeat=2,
            visible_poses=len(ORDEN_VARIABLES),
            visible_factors=factor_index,
            active_factor=factor_name,
            show_jacobian=True,
            show_block_grid=True,
        )

    add(
        "jacobian",
        "El jacobiano global es grande, pero la mayoría de sus bloques son cero.",
        repeat=4,
        visible_poses=len(ORDEN_VARIABLES),
        visible_factors=len(ORDEN_FACTORES),
        show_jacobian=True,
        show_block_grid=True,
    )

    add(
        "normal_equations",
        "Gauss-Newton forma H=JᵀΩJ y g=JᵀΩe mediante contribuciones locales.",
        repeat=4,
        visible_poses=len(ORDEN_VARIABLES),
        visible_factors=len(ORDEN_FACTORES),
        show_jacobian=True,
        show_hessian=True,
        show_block_grid=True,
    )

    for factor_index, factor_name in enumerate(ORDEN_FACTORES):
        add(
            "assembly",
            "Cada factor añade pequeños bloques simétricos a la Hessiana global.",
            visible_poses=len(ORDEN_VARIABLES),
            visible_factors=len(ORDEN_FACTORES),
            active_factor=factor_name,
            show_jacobian=True,
            show_hessian=True,
            show_block_grid=True,
        )

    add(
        "gauge",
        "Sin prior quedan tres direcciones de gauge; con prior el sistema recupera rango completo.",
        repeat=4,
        visible_poses=len(ORDEN_VARIABLES),
        visible_factors=len(ORDEN_FACTORES),
        active_factor="prior_x0",
        show_jacobian=True,
        show_hessian=True,
        show_block_grid=True,
    )

    good_elimination = analysis["good_elimination"]
    for step in range(len(good_elimination["history"])):
        add(
            "good_elimination",
            "Un orden de mínimo grado reduce las conexiones nuevas durante la eliminación.",
            visible_poses=len(ORDEN_VARIABLES),
            visible_factors=len(ORDEN_FACTORES),
            show_jacobian=True,
            show_hessian=True,
            show_block_grid=True,
            show_fill_in=True,
            elimination=good_elimination,
            elimination_step=step,
        )

    bad_elimination = analysis["bad_elimination"]
    for step in range(len(bad_elimination["history"])):
        add(
            "bad_elimination",
            "Un orden desfavorable crea más fill-in y aumenta el coste de la factorización.",
            visible_poses=len(ORDEN_VARIABLES),
            visible_factors=len(ORDEN_FACTORES),
            show_jacobian=True,
            show_hessian=True,
            show_block_grid=True,
            show_fill_in=True,
            elimination=bad_elimination,
            elimination_step=step,
        )

    add(
        "connections",
        "GTSAM, g2o y Ceres explotan esta estructura local y dispersa.",
        repeat=4,
        visible_poses=len(ORDEN_VARIABLES),
        visible_factors=len(ORDEN_FACTORES),
        show_jacobian=True,
        show_hessian=True,
        show_block_grid=True,
        show_connections=True,
    )

    add(
        "summary",
        "La topología del grafo se convierte directamente en el patrón de J y H.",
        repeat=4,
        visible_poses=len(ORDEN_VARIABLES),
        visible_factors=len(ORDEN_FACTORES),
        show_jacobian=True,
        show_hessian=True,
        show_block_grid=True,
        show_connections=True,
    )

    for index, state in enumerate(states, start=1):
        state["step"] = index
        state["total_steps"] = len(states)

    return states


# ---------------------------------------------------------------------------
# Validaciones y salida
# ---------------------------------------------------------------------------


def validar_grafo(graph):
    """Valida la estructura del pose graph y sus factores."""

    poses = [
        nodo
        for nodo, datos in graph.nodes(data=True)
        if datos.get("node_type") == "pose"
    ]

    if sorted(poses) != sorted(ORDEN_VARIABLES):
        raise ValueError("El conjunto de poses no coincide con el esperado.")
    if not nx.is_connected(graph):
        raise ValueError("El grafo completo debe ser conectado.")
    if graph.degree["prior_x0"] != 1:
        raise ValueError("El factor prior debe ser unario.")

    nombres_factores = {
        datos.get("factor_name")
        for _, _, datos in graph.edges(data=True)
    }
    if nombres_factores != set(ORDEN_FACTORES):
        raise ValueError("Los factores del grafo no coinciden con el orden esperado.")

    for factor_name in ORDEN_FACTORES:
        factor = obtener_factor(graph, factor_name)
        informacion = factor["information"]
        covarianza = factor["covariance"]

        if not np.allclose(
            covarianza @ informacion,
            np.eye(3),
            atol=1e-9,
        ):
            raise ValueError(f"Covarianza e información inconsistentes en {factor_name}.")


def validar_estructura_dispersa(graph, sistema, analysis):
    """Comprueba dimensiones, patrones, simetría, rango y fill-in."""

    j = sistema["jacobian"]
    h = sistema["hessian"]
    omega = sistema["information"]
    e = sistema["residuals"]
    g = sistema["gradient"]

    if j.shape != (24, 18):
        raise ValueError("El jacobiano debe tener forma 24x18.")
    if h.shape != (18, 18):
        raise ValueError("La Hessiana debe tener forma 18x18.")
    if omega.shape != (24, 24):
        raise ValueError("La información global debe tener forma 24x24.")
    if e.shape != (24,) or g.shape != (18,):
        raise ValueError("Las dimensiones de e o g no son correctas.")

    if not np.allclose(h.toarray(), h.toarray().T, atol=1e-9):
        raise ValueError("La Hessiana debe ser simétrica.")

    h_recalculada = j.T @ omega @ j
    g_recalculado = np.asarray(j.T @ (omega @ e)).reshape(-1)

    if not np.allclose(h.toarray(), h_recalculada.toarray(), atol=1e-9):
        raise ValueError("H no coincide con JᵀΩJ.")
    if not np.allclose(g, g_recalculado, atol=1e-9):
        raise ValueError("g no coincide con JᵀΩe.")

    estimaciones = obtener_estimaciones(graph)
    jacobiano_global = construir_jacobiano_global_numerico(
        graph,
        estimaciones,
        incluir_prior=True,
    )

    if not np.allclose(j.toarray(), jacobiano_global, atol=3e-6, rtol=3e-6):
        raise ValueError("El jacobiano ensamblado no coincide con el global numérico.")

    patron_bloques = analysis["jacobian_block_pattern"]
    for fila, factor_name in enumerate(ORDEN_FACTORES):
        variables = set(sistema["dependencies"][factor_name])
        for columna, variable in enumerate(ORDEN_VARIABLES):
            esperado = variable in variables
            if bool(patron_bloques[fila, columna]) != esperado:
                raise ValueError("El patrón por bloques no refleja las dependencias.")

    if analysis["gauge"]["without_prior"]["nullity"] != 3:
        raise ValueError("Sin prior deben existir tres grados de gauge en SE(2).")
    if analysis["gauge"]["with_prior"]["nullity"] != 0:
        raise ValueError("El prior debe eliminar la nulidad global.")

    if analysis["good_elimination"]["fill_count"] >= analysis["bad_elimination"]["fill_count"]:
        raise ValueError("El orden bueno debe producir menos fill-in que el malo.")

    if not (0.0 < analysis["jacobian_metrics"]["density"] < 1.0):
        raise ValueError("La densidad del jacobiano debe estar entre cero y uno.")
    if not (0.0 < analysis["hessian_metrics"]["density"] < 1.0):
        raise ValueError("La densidad de la Hessiana debe estar entre cero y uno.")


def validar_resultados(graph, sistema, analysis, states):
    """Ejecuta todas las validaciones matemáticas y didácticas."""

    validar_grafo(graph)
    validar_estructura_dispersa(graph, sistema, analysis)

    if not np.isfinite(sistema["cost"]) or sistema["cost"] < 0.0:
        raise ValueError("El coste debe ser finito y no negativo.")
    if not np.all(np.isfinite(sistema["gradient"])):
        raise ValueError("El gradiente debe contener valores finitos.")
    if len(states) < 60:
        raise ValueError("La demostración debe contener al menos sesenta estados.")
    if states[-1].get("phase") != "summary":
        raise ValueError("El último estado debe ser el resumen.")
    if not states[-1].get("show_jacobian"):
        raise ValueError("La imagen final debe mostrar el jacobiano.")
    if not states[-1].get("show_hessian"):
        raise ValueError("La imagen final debe mostrar la Hessiana.")
    if not states[-1].get("show_connections"):
        raise ValueError("La imagen final debe mostrar las conexiones.")

    return {
        "factor_count": len(ORDEN_FACTORES),
        "pose_count": len(ORDEN_VARIABLES),
        "state_count": len(states),
        "cost": float(sistema["cost"]),
        "jacobian_nnz": int(analysis["jacobian_metrics"]["nnz"]),
        "hessian_nnz": int(analysis["hessian_metrics"]["nnz"]),
        "good_fill": int(analysis["good_elimination"]["fill_count"]),
        "bad_fill": int(analysis["bad_elimination"]["fill_count"]),
    }


def imprimir_resumen(sistema, analysis, validation):
    """Imprime las magnitudes principales de la demostración."""

    j_metrics = analysis["jacobian_metrics"]
    h_metrics = analysis["hessian_metrics"]

    print("\n=== Jacobianos y estructura dispersa ===")
    print(f"Poses: {validation['pose_count']}")
    print(f"Factores: {validation['factor_count']}")
    print(f"J: {j_metrics['shape']} · nnz={j_metrics['nnz']} · densidad={j_metrics['density']:.4f}")
    print(f"H: {h_metrics['shape']} · nnz={h_metrics['nnz']} · densidad={h_metrics['density']:.4f}")
    print(f"Coste: {sistema['cost']:.9f}")
    print(
        "Rango/nulidad sin prior: "
        f"{analysis['gauge']['without_prior']['rank']} / "
        f"{analysis['gauge']['without_prior']['nullity']}"
    )
    print(
        "Rango/nulidad con prior: "
        f"{analysis['gauge']['with_prior']['rank']} / "
        f"{analysis['gauge']['with_prior']['nullity']}"
    )
    print(
        "Fill-in orden bueno/malo: "
        f"{validation['good_fill']} / {validation['bad_fill']}"
    )
    print(f"Estados de animación: {validation['state_count']}")


def main():
    graph = crear_pose_graph()
    estimaciones = obtener_estimaciones(graph)
    sistema = ensamblar_sistema_disperso(graph, estimaciones)
    analysis = analizar_estructura_dispersa(
        graph,
        sistema,
        estimaciones,
    )
    states = crear_estados_animacion(
        graph,
        sistema,
        analysis,
    )
    validation = validar_resultados(
        graph,
        sistema,
        analysis,
        states,
    )

    imprimir_resumen(
        sistema,
        analysis,
        validation,
    )

    animator = GraphAnimator(
        figsize=(18, 10),
        interval=520,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "05_optimizacion"
        / "07_matriz_dispersa_slam.png"
    )

    animator.animate_sparse_slam_matrices(
        graph=graph,
        states=states,
        title="Jacobianos y estructura dispersa en SLAM",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
