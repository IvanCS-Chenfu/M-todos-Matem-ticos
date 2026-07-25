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

MEDICION_FIABLE = np.array([3.0, 2.0], dtype=float)
SIGMAS_FIABLE = np.array([0.18, 0.30], dtype=float)
CORRELACION_FIABLE = 0.35

MEDICION_POCO_FIABLE = np.array([4.2, 3.0], dtype=float)
SIGMAS_POCO_FIABLE = np.array([0.85, 1.20], dtype=float)
CORRELACION_POCO_FIABLE = -0.25

RESIDUO_COMUN = np.array([0.45, -0.30], dtype=float)
NIVEL_ELIPSE = 2.0


# ---------------------------------------------------------------------------
# Validación y operaciones estadísticas
# ---------------------------------------------------------------------------


def validar_vector_2d(vector, nombre="vector"):
    """Valida un vector bidimensional y devuelve una copia float."""

    vector = np.asarray(vector, dtype=float)

    if vector.shape != (2,):
        raise ValueError(f"{nombre} debe contener exactamente dos componentes.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{nombre} debe contener valores finitos.")

    return vector.copy()


def validar_correlacion(correlacion):
    """Valida un coeficiente de correlación estrictamente interior a [-1, 1]."""

    correlacion = float(correlacion)

    if not np.isfinite(correlacion):
        raise ValueError("La correlación debe ser finita.")

    if not -1.0 < correlacion < 1.0:
        raise ValueError("La correlación debe pertenecer al intervalo (-1, 1).")

    return correlacion


def crear_covarianza_correlacionada(sigmas, correlacion=0.0):
    """Construye una covarianza 2D a partir de sigmas y correlación."""

    sigmas = validar_vector_2d(sigmas, "sigmas")
    correlacion = validar_correlacion(correlacion)

    if np.any(sigmas <= 0.0):
        raise ValueError("Las desviaciones estándar deben ser positivas.")

    sigma_x, sigma_y = sigmas
    covarianza_xy = correlacion * sigma_x * sigma_y

    covarianza = np.array(
        [
            [sigma_x**2, covarianza_xy],
            [covarianza_xy, sigma_y**2],
        ],
        dtype=float,
    )

    validar_covarianza(covarianza)
    return covarianza


def validar_covarianza(covarianza):
    """Comprueba dimensiones, simetría y definición positiva."""

    covarianza = np.asarray(covarianza, dtype=float)

    if covarianza.shape != (2, 2):
        raise ValueError("La covarianza debe ser una matriz 2x2.")

    if not np.all(np.isfinite(covarianza)):
        raise ValueError("La covarianza debe contener valores finitos.")

    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise ValueError("La covarianza debe ser simétrica.")

    autovalores = np.linalg.eigvalsh(covarianza)

    if np.any(autovalores <= 0.0):
        raise ValueError("La covarianza debe ser definida positiva.")

    return covarianza.copy()


def calcular_matriz_informacion(covarianza):
    """Calcula Ω = Σ⁻¹ después de validar la covarianza."""

    covarianza = validar_covarianza(covarianza)
    informacion = np.linalg.inv(covarianza)

    if not np.all(np.isfinite(informacion)):
        raise ValueError("La matriz de información contiene valores no finitos.")

    if not np.allclose(informacion, informacion.T, atol=1e-12):
        raise ValueError("La matriz de información debe ser simétrica.")

    return informacion


def calcular_desviaciones_estandar(covarianza):
    """Recupera las desviaciones estándar desde la diagonal de Σ."""

    covarianza = validar_covarianza(covarianza)
    return np.sqrt(np.diag(covarianza))


def calcular_correlacion(covarianza):
    """Calcula ρxy = σxy / (σx σy)."""

    covarianza = validar_covarianza(covarianza)
    sigmas = calcular_desviaciones_estandar(covarianza)
    return float(covarianza[0, 1] / (sigmas[0] * sigmas[1]))


def descomponer_covarianza(covarianza):
    """Devuelve autovalores y autovectores ordenados de mayor a menor."""

    covarianza = validar_covarianza(covarianza)
    autovalores, autovectores = np.linalg.eigh(covarianza)
    orden = np.argsort(autovalores)[::-1]

    return autovalores[orden], autovectores[:, orden]


def calcular_parametros_elipse(covarianza, nivel=NIVEL_ELIPSE):
    """Calcula anchura, altura, ángulo y área de una elipse de confianza."""

    nivel = float(nivel)

    if not np.isfinite(nivel) or nivel <= 0.0:
        raise ValueError("El nivel de la elipse debe ser positivo y finito.")

    autovalores, autovectores = descomponer_covarianza(covarianza)
    eje_mayor = nivel * np.sqrt(autovalores[0])
    eje_menor = nivel * np.sqrt(autovalores[1])
    vector_principal = autovectores[:, 0]
    angulo = np.degrees(np.arctan2(vector_principal[1], vector_principal[0]))

    return {
        "width": float(2.0 * eje_mayor),
        "height": float(2.0 * eje_menor),
        "angle_deg": float(angulo),
        "semi_major": float(eje_mayor),
        "semi_minor": float(eje_menor),
        "area": float(np.pi * eje_mayor * eje_menor),
        "eigenvalues": autovalores.copy(),
        "eigenvectors": autovectores.copy(),
    }


def calcular_distancia_euclidea(residuo):
    """Calcula ||e||₂."""

    residuo = validar_vector_2d(residuo, "residuo")
    return float(np.linalg.norm(residuo))


def calcular_distancia_mahalanobis_cuadrada(residuo, informacion):
    """Calcula d² = eᵀ Ω e."""

    residuo = validar_vector_2d(residuo, "residuo")
    informacion = np.asarray(informacion, dtype=float)

    if informacion.shape != (2, 2):
        raise ValueError("La información debe ser una matriz 2x2.")

    if not np.all(np.isfinite(informacion)):
        raise ValueError("La información debe contener valores finitos.")

    if not np.allclose(informacion, informacion.T, atol=1e-12):
        raise ValueError("La información debe ser simétrica.")

    coste = float(residuo.T @ informacion @ residuo)

    if coste < -1e-12:
        raise ValueError("La distancia de Mahalanobis no puede ser negativa.")

    return max(coste, 0.0)


def calcular_distancia_mahalanobis(residuo, informacion):
    """Calcula la raíz de la distancia de Mahalanobis cuadrada."""

    return float(
        np.sqrt(
            calcular_distancia_mahalanobis_cuadrada(
                residuo,
                informacion,
            )
        )
    )


def calcular_media_no_ponderada(mediciones):
    """Calcula la media geométrica de varias mediciones 2D."""

    mediciones = np.asarray(mediciones, dtype=float)

    if mediciones.ndim != 2 or mediciones.shape[1] != 2:
        raise ValueError("Las mediciones deben formar una matriz n x 2.")

    if mediciones.shape[0] < 1 or not np.all(np.isfinite(mediciones)):
        raise ValueError("Debe existir al menos una medición 2D finita.")

    return np.mean(mediciones, axis=0)


def fusionar_mediciones_gaussianas(mediciones, covarianzas):
    """Fusiona gaussianas independientes mediante suma de informaciones."""

    mediciones = np.asarray(mediciones, dtype=float)

    if mediciones.ndim != 2 or mediciones.shape[1] != 2:
        raise ValueError("Las mediciones deben formar una matriz n x 2.")

    if len(covarianzas) != mediciones.shape[0]:
        raise ValueError("Debe existir una covarianza por medición.")

    informaciones = [
        calcular_matriz_informacion(covarianza)
        for covarianza in covarianzas
    ]

    informacion_fusionada = np.sum(informaciones, axis=0)
    covarianza_fusionada = np.linalg.inv(informacion_fusionada)

    termino = np.zeros(2, dtype=float)

    for medicion, informacion in zip(mediciones, informaciones):
        termino += informacion @ medicion

    media_fusionada = covarianza_fusionada @ termino

    return {
        "mean": media_fusionada,
        "covariance": covarianza_fusionada,
        "information": informacion_fusionada,
        "individual_information": informaciones,
    }


def escalar_covarianza(covarianza, factor_sigma):
    """Escala las desviaciones estándar por un factor y Σ por su cuadrado."""

    covarianza = validar_covarianza(covarianza)
    factor_sigma = float(factor_sigma)

    if not np.isfinite(factor_sigma) or factor_sigma <= 0.0:
        raise ValueError("El factor de escala debe ser positivo y finito.")

    return covarianza * factor_sigma**2


# ---------------------------------------------------------------------------
# Grafo y evaluación completa
# ---------------------------------------------------------------------------


def crear_grafo_incertidumbre(
    medicion_fiable,
    covarianza_fiable,
    medicion_poco_fiable,
    covarianza_poco_fiable,
):
    """Crea un grafo de factores con una variable y dos mediciones 2D."""

    medicion_fiable = validar_vector_2d(medicion_fiable, "medición fiable")
    medicion_poco_fiable = validar_vector_2d(
        medicion_poco_fiable,
        "medición poco fiable",
    )
    covarianza_fiable = validar_covarianza(covarianza_fiable)
    covarianza_poco_fiable = validar_covarianza(covarianza_poco_fiable)

    informacion_fiable = calcular_matriz_informacion(covarianza_fiable)
    informacion_poco_fiable = calcular_matriz_informacion(
        covarianza_poco_fiable
    )

    graph = nx.Graph()
    graph.graph["name"] = "Incertidumbre, covarianza e información"
    graph.graph["objective"] = "sum_k e_k.T @ Omega_k @ e_k"
    graph.graph["variable"] = "p"

    graph.add_node(
        "p",
        node_type="variable",
        dimension=2,
        label="p",
        description="Posición 2D que se desea estimar.",
    )

    factors = [
        (
            "f_fiable",
            "Medición fiable",
            medicion_fiable,
            covarianza_fiable,
            informacion_fiable,
            "Sensor A",
        ),
        (
            "f_poco_fiable",
            "Medición poco fiable",
            medicion_poco_fiable,
            covarianza_poco_fiable,
            informacion_poco_fiable,
            "Sensor B",
        ),
    ]

    for factor, label, measurement, covariance, information, sensor in factors:
        graph.add_node(
            factor,
            node_type="measurement_factor",
            bipartite=1,
            label=label,
            sensor=sensor,
            measurement=measurement.copy(),
            covariance=covariance.copy(),
            information=information.copy(),
            information_trace=float(np.trace(information)),
        )
        graph.add_edge(
            "p",
            factor,
            relation="gaussian_measurement_2d",
            measurement=measurement.copy(),
            covariance=covariance.copy(),
            information=information.copy(),
            residual_model="p - z_k",
            cost_model="e_k.T @ Omega_k @ e_k",
        )

    return graph


def comparar_mediciones(
    medicion_fiable,
    covarianza_fiable,
    medicion_poco_fiable,
    covarianza_poco_fiable,
    residuo_comun,
):
    """Calcula el resumen numérico utilizado por la demostración."""

    mediciones = np.vstack(
        [
            validar_vector_2d(medicion_fiable, "medición fiable"),
            validar_vector_2d(
                medicion_poco_fiable,
                "medición poco fiable",
            ),
        ]
    )
    covarianzas = [
        validar_covarianza(covarianza_fiable),
        validar_covarianza(covarianza_poco_fiable),
    ]
    residuo_comun = validar_vector_2d(residuo_comun, "residuo común")
    informaciones = [
        calcular_matriz_informacion(covarianza)
        for covarianza in covarianzas
    ]
    elipses = [
        calcular_parametros_elipse(covarianza)
        for covarianza in covarianzas
    ]
    fusion = fusionar_mediciones_gaussianas(mediciones, covarianzas)

    return {
        "measurements": mediciones,
        "covariances": covarianzas,
        "informations": informaciones,
        "sigmas": [
            calcular_desviaciones_estandar(covarianza)
            for covarianza in covarianzas
        ],
        "correlations": [
            calcular_correlacion(covarianza)
            for covarianza in covarianzas
        ],
        "ellipses": elipses,
        "residual": residuo_comun,
        "euclidean_distance": calcular_distancia_euclidea(residuo_comun),
        "mahalanobis_squared": [
            calcular_distancia_mahalanobis_cuadrada(
                residuo_comun,
                informacion,
            )
            for informacion in informaciones
        ],
        "mahalanobis": [
            calcular_distancia_mahalanobis(
                residuo_comun,
                informacion,
            )
            for informacion in informaciones
        ],
        "unweighted_mean": calcular_media_no_ponderada(mediciones),
        "fusion": fusion,
        "fused_ellipse": calcular_parametros_elipse(
            fusion["covariance"],
        ),
    }


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


def _serializar_elipse(ellipse):
    return {
        "width": float(ellipse["width"]),
        "height": float(ellipse["height"]),
        "angle_deg": float(ellipse["angle_deg"]),
        "semi_major": float(ellipse["semi_major"]),
        "semi_minor": float(ellipse["semi_minor"]),
        "area": float(ellipse["area"]),
        "eigenvalues": _serializar_vector(ellipse["eigenvalues"]),
        "eigenvectors": _serializar_matriz(ellipse["eigenvectors"]),
    }


def crear_estado_animacion(
    comparison,
    *,
    phase,
    message,
    uncertainty_scale=1.0,
    show_measurement_reliable=False,
    show_measurement_uncertain=False,
    show_ellipse_reliable=False,
    show_ellipse_uncertain=False,
    show_covariances=False,
    show_information=False,
    show_same_residual=False,
    show_mahalanobis=False,
    show_unweighted_fusion=False,
    show_weighted_fusion=False,
    show_factor_graph=False,
    show_sensor_connection=False,
    show_scale_experiment=False,
    scale_history=None,
):
    """Convierte los resultados en un fotograma independiente."""

    base_covariance_uncertain = comparison["covariances"][1]
    dynamic_covariance = escalar_covarianza(
        base_covariance_uncertain,
        uncertainty_scale,
    )
    dynamic_information = calcular_matriz_informacion(dynamic_covariance)
    dynamic_ellipse = calcular_parametros_elipse(dynamic_covariance)
    dynamic_cost = calcular_distancia_mahalanobis_cuadrada(
        comparison["residual"],
        dynamic_information,
    )

    history = list(scale_history or [])

    return {
        "phase": phase,
        "message": message,
        "measurements": [
            _serializar_vector(value)
            for value in comparison["measurements"]
        ],
        "covariances": [
            _serializar_matriz(comparison["covariances"][0]),
            _serializar_matriz(dynamic_covariance),
        ],
        "informations": [
            _serializar_matriz(comparison["informations"][0]),
            _serializar_matriz(dynamic_information),
        ],
        "sigmas": [
            _serializar_vector(comparison["sigmas"][0]),
            _serializar_vector(
                calcular_desviaciones_estandar(dynamic_covariance)
            ),
        ],
        "correlations": [
            float(comparison["correlations"][0]),
            float(calcular_correlacion(dynamic_covariance)),
        ],
        "ellipses": [
            _serializar_elipse(comparison["ellipses"][0]),
            _serializar_elipse(dynamic_ellipse),
        ],
        "residual": _serializar_vector(comparison["residual"]),
        "same_residual_points": [
            _serializar_vector(
                comparison["measurements"][0] + comparison["residual"]
            ),
            _serializar_vector(
                comparison["measurements"][1] + comparison["residual"]
            ),
        ],
        "euclidean_distance": float(comparison["euclidean_distance"]),
        "mahalanobis_squared": [
            float(comparison["mahalanobis_squared"][0]),
            float(dynamic_cost),
        ],
        "information_traces": [
            float(np.trace(comparison["informations"][0])),
            float(np.trace(dynamic_information)),
        ],
        "ellipse_areas": [
            float(comparison["ellipses"][0]["area"]),
            float(dynamic_ellipse["area"]),
        ],
        "unweighted_mean": _serializar_vector(
            comparison["unweighted_mean"]
        ),
        "weighted_mean": _serializar_vector(
            comparison["fusion"]["mean"]
        ),
        "fused_covariance": _serializar_matriz(
            comparison["fusion"]["covariance"]
        ),
        "fused_information": _serializar_matriz(
            comparison["fusion"]["information"]
        ),
        "fused_ellipse": _serializar_elipse(
            comparison["fused_ellipse"]
        ),
        "uncertainty_scale": float(uncertainty_scale),
        "scale_history": [
            {
                "scale": float(item["scale"]),
                "area": float(item["area"]),
                "information_trace": float(item["information_trace"]),
                "cost": float(item["cost"]),
            }
            for item in history
        ],
        "show_measurement_reliable": bool(show_measurement_reliable),
        "show_measurement_uncertain": bool(show_measurement_uncertain),
        "show_ellipse_reliable": bool(show_ellipse_reliable),
        "show_ellipse_uncertain": bool(show_ellipse_uncertain),
        "show_covariances": bool(show_covariances),
        "show_information": bool(show_information),
        "show_same_residual": bool(show_same_residual),
        "show_mahalanobis": bool(show_mahalanobis),
        "show_unweighted_fusion": bool(show_unweighted_fusion),
        "show_weighted_fusion": bool(show_weighted_fusion),
        "show_factor_graph": bool(show_factor_graph),
        "show_sensor_connection": bool(show_sensor_connection),
        "show_scale_experiment": bool(show_scale_experiment),
    }


def crear_estados_animacion(graph):
    """Crea la secuencia didáctica completa del apartado 5.4."""

    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise TypeError("Se esperaba un grafo de factores no dirigido.")

    covariance_reliable = graph.nodes["f_fiable"]["covariance"]
    covariance_uncertain = graph.nodes["f_poco_fiable"]["covariance"]

    comparison = comparar_mediciones(
        MEDICION_FIABLE,
        covariance_reliable,
        MEDICION_POCO_FIABLE,
        covariance_uncertain,
        RESIDUO_COMUN,
    )

    states = []
    scale_history = []

    def add(phase, message, repeat=1, **flags):
        for _ in range(repeat):
            states.append(
                crear_estado_animacion(
                    comparison,
                    phase=phase,
                    message=message,
                    scale_history=scale_history,
                    **flags,
                )
            )

    add(
        "introduction",
        "Una medición no es solo un punto: también necesita una incertidumbre.",
        repeat=3,
    )
    add(
        "reliable_measurement",
        "La primera medición tiene una dispersión pequeña y será la más fiable.",
        repeat=3,
        show_measurement_reliable=True,
    )
    add(
        "reliable_sigmas",
        "Sus desviaciones estándar describen la dispersión en x e y.",
        repeat=3,
        show_measurement_reliable=True,
        show_covariances=True,
    )
    add(
        "reliable_ellipse",
        "La covarianza se representa mediante una elipse de incertidumbre.",
        repeat=4,
        show_measurement_reliable=True,
        show_ellipse_reliable=True,
        show_covariances=True,
    )
    add(
        "uncertain_measurement",
        "La segunda medición tiene una dispersión mucho mayor.",
        repeat=3,
        show_measurement_reliable=True,
        show_measurement_uncertain=True,
        show_ellipse_reliable=True,
    )
    add(
        "compare_ellipses",
        "Una elipse pequeña indica mayor precisión; una grande, menor confianza.",
        repeat=4,
        show_measurement_reliable=True,
        show_measurement_uncertain=True,
        show_ellipse_reliable=True,
        show_ellipse_uncertain=True,
        show_covariances=True,
    )
    add(
        "information_inverse",
        "La matriz de información es la inversa de la covarianza: Ω = Σ⁻¹.",
        repeat=4,
        show_measurement_reliable=True,
        show_measurement_uncertain=True,
        show_ellipse_reliable=True,
        show_ellipse_uncertain=True,
        show_covariances=True,
        show_information=True,
    )
    add(
        "same_residual",
        "Se aplica el mismo residuo geométrico a las dos mediciones.",
        repeat=4,
        show_measurement_reliable=True,
        show_measurement_uncertain=True,
        show_ellipse_reliable=True,
        show_ellipse_uncertain=True,
        show_information=True,
        show_same_residual=True,
    )
    add(
        "mahalanobis",
        "La distancia euclídea es igual, pero el coste de Mahalanobis es distinto.",
        repeat=4,
        show_measurement_reliable=True,
        show_measurement_uncertain=True,
        show_ellipse_reliable=True,
        show_ellipse_uncertain=True,
        show_information=True,
        show_same_residual=True,
        show_mahalanobis=True,
    )

    for scale in [0.65, 0.80, 1.00, 1.25, 1.50, 1.80, 2.20]:
        dynamic_covariance = escalar_covarianza(
            covariance_uncertain,
            scale,
        )
        dynamic_information = calcular_matriz_informacion(dynamic_covariance)
        dynamic_ellipse = calcular_parametros_elipse(dynamic_covariance)
        dynamic_cost = calcular_distancia_mahalanobis_cuadrada(
            RESIDUO_COMUN,
            dynamic_information,
        )
        scale_history.append(
            {
                "scale": scale,
                "area": dynamic_ellipse["area"],
                "information_trace": np.trace(dynamic_information),
                "cost": dynamic_cost,
            }
        )
        states.append(
            crear_estado_animacion(
                comparison,
                phase="scale_experiment",
                message=(
                    f"Escala σ = {scale:.2f}: al crecer la elipse, "
                    "disminuyen la información y el coste del mismo residuo."
                ),
                uncertainty_scale=scale,
                scale_history=scale_history,
                show_measurement_reliable=True,
                show_measurement_uncertain=True,
                show_ellipse_reliable=True,
                show_ellipse_uncertain=True,
                show_information=True,
                show_same_residual=True,
                show_mahalanobis=True,
                show_scale_experiment=True,
            )
        )

    add(
        "unweighted_fusion",
        "Sin incertidumbre, la fusión es el punto medio de las dos mediciones.",
        repeat=4,
        show_measurement_reliable=True,
        show_measurement_uncertain=True,
        show_ellipse_reliable=True,
        show_ellipse_uncertain=True,
        show_unweighted_fusion=True,
    )
    add(
        "weighted_fusion",
        "Con información, la solución se desplaza hacia la medición más fiable.",
        repeat=5,
        show_measurement_reliable=True,
        show_measurement_uncertain=True,
        show_ellipse_reliable=True,
        show_ellipse_uncertain=True,
        show_covariances=True,
        show_information=True,
        show_same_residual=True,
        show_mahalanobis=True,
        show_unweighted_fusion=True,
        show_weighted_fusion=True,
    )
    add(
        "factor_graph",
        "Cada factor almacena una medición, su covarianza y su información.",
        repeat=4,
        show_measurement_reliable=True,
        show_measurement_uncertain=True,
        show_ellipse_reliable=True,
        show_ellipse_uncertain=True,
        show_information=True,
        show_unweighted_fusion=True,
        show_weighted_fusion=True,
        show_factor_graph=True,
    )
    add(
        "sensor_connection",
        "Odometría, LiDAR, cámara e IMU necesitan modelos de incertidumbre propios.",
        repeat=3,
        show_measurement_reliable=True,
        show_measurement_uncertain=True,
        show_ellipse_reliable=True,
        show_ellipse_uncertain=True,
        show_information=True,
        show_weighted_fusion=True,
        show_factor_graph=True,
        show_sensor_connection=True,
    )
    add(
        "summary",
        "Menor Σ → mayor Ω → mayor peso. Cada arista indica cuánto confiar en ella.",
        repeat=4,
        show_measurement_reliable=True,
        show_measurement_uncertain=True,
        show_ellipse_reliable=True,
        show_ellipse_uncertain=True,
        show_covariances=True,
        show_information=True,
        show_same_residual=True,
        show_mahalanobis=True,
        show_unweighted_fusion=True,
        show_weighted_fusion=True,
        show_factor_graph=True,
        show_sensor_connection=True,
        show_scale_experiment=True,
    )

    for index, state in enumerate(states, start=1):
        state["step"] = index
        state["total_steps"] = len(states)

    return {
        "states": states,
        "comparison": comparison,
        "scale_history": scale_history,
    }


# ---------------------------------------------------------------------------
# Validaciones y salida
# ---------------------------------------------------------------------------


def validar_grafo_incertidumbre(graph):
    """Valida la variable, los factores y sus matrices estadísticas."""

    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise TypeError("El ejemplo debe utilizar un nx.Graph no dirigido.")

    expected_nodes = {"p", "f_fiable", "f_poco_fiable"}

    if set(graph.nodes()) != expected_nodes:
        raise ValueError("El grafo debe contener p y dos factores de medición.")

    if graph.nodes["p"].get("node_type") != "variable":
        raise ValueError("El nodo p debe representar una variable 2D.")

    for factor in ("f_fiable", "f_poco_fiable"):
        if graph.nodes[factor].get("node_type") != "measurement_factor":
            raise ValueError(f"{factor} debe representar un factor de medición.")

        if not graph.has_edge("p", factor):
            raise ValueError(f"Falta la arista entre p y {factor}.")

        edge = graph.edges["p", factor]
        covariance = validar_covarianza(edge["covariance"])
        information = np.asarray(edge["information"], dtype=float)

        if not np.allclose(
            covariance @ information,
            np.eye(2),
            atol=1e-10,
        ):
            raise ValueError("ΣΩ debe ser aproximadamente la identidad.")

        if edge.get("residual_model") != "p - z_k":
            raise ValueError("El modelo de residuo debe ser p - z_k.")

        if edge.get("cost_model") != "e_k.T @ Omega_k @ e_k":
            raise ValueError("El modelo de coste ponderado es incorrecto.")


def validar_resultados(graph, result):
    """Comprueba incertidumbres, costes, fusión y secuencia didáctica."""

    validar_grafo_incertidumbre(graph)
    comparison = result["comparison"]

    covariance_reliable, covariance_uncertain = comparison["covariances"]
    information_reliable, information_uncertain = comparison["informations"]
    ellipse_reliable, ellipse_uncertain = comparison["ellipses"]
    cost_reliable, cost_uncertain = comparison["mahalanobis_squared"]

    if np.linalg.det(covariance_reliable) >= np.linalg.det(covariance_uncertain):
        raise ValueError("La medición fiable debe tener menor covarianza.")

    if np.trace(information_reliable) <= np.trace(information_uncertain):
        raise ValueError("La medición fiable debe tener mayor información.")

    if ellipse_reliable["area"] >= ellipse_uncertain["area"]:
        raise ValueError("La elipse fiable debe tener menor área.")

    if cost_reliable <= cost_uncertain:
        raise ValueError(
            "El mismo residuo debe costar más en la medición fiable."
        )

    if not np.isclose(
        comparison["euclidean_distance"],
        np.linalg.norm(RESIDUO_COMUN),
    ):
        raise ValueError("La distancia euclídea del residuo es incorrecta.")

    unweighted = comparison["unweighted_mean"]
    weighted = comparison["fusion"]["mean"]

    distance_unweighted_to_reliable = np.linalg.norm(
        unweighted - MEDICION_FIABLE
    )
    distance_weighted_to_reliable = np.linalg.norm(
        weighted - MEDICION_FIABLE
    )

    if distance_weighted_to_reliable >= distance_unweighted_to_reliable:
        raise ValueError(
            "La fusión ponderada debe quedar más cerca de la medición fiable."
        )

    fused_covariance = comparison["fusion"]["covariance"]

    if np.linalg.det(fused_covariance) >= np.linalg.det(covariance_reliable):
        raise ValueError(
            "La fusión independiente debe reducir la incertidumbre resultante."
        )

    for covariance, information in zip(
        comparison["covariances"],
        comparison["informations"],
    ):
        if not np.allclose(covariance @ information, np.eye(2), atol=1e-10):
            raise ValueError("Una matriz de información no es la inversa de Σ.")

    scale_history = result["scale_history"]
    areas = [item["area"] for item in scale_history]
    traces = [item["information_trace"] for item in scale_history]
    costs = [item["cost"] for item in scale_history]

    if any(next_area <= area for area, next_area in zip(areas, areas[1:])):
        raise ValueError("El área debe crecer al aumentar la escala de sigma.")

    if any(next_trace >= trace for trace, next_trace in zip(traces, traces[1:])):
        raise ValueError("La información debe disminuir al crecer la covarianza.")

    if any(next_cost >= cost for cost, next_cost in zip(costs, costs[1:])):
        raise ValueError("El coste debe disminuir para el mismo residuo.")

    if len(result["states"]) < 50:
        raise ValueError("La animación debe contener al menos cincuenta estados.")

    graph.nodes["p"]["unweighted_estimate"] = unweighted.copy()
    graph.nodes["p"]["weighted_estimate"] = weighted.copy()
    graph.nodes["p"]["fused_covariance"] = fused_covariance.copy()


def imprimir_resumen(graph, result):
    """Imprime las magnitudes principales del ejemplo."""

    comparison = result["comparison"]
    covariance_reliable, covariance_uncertain = comparison["covariances"]
    information_reliable, information_uncertain = comparison["informations"]

    print("\n=== Incertidumbre, covarianza y matriz de información ===")
    print("Medición fiable:", MEDICION_FIABLE)
    print("Covarianza fiable:\n", covariance_reliable)
    print("Información fiable:\n", information_reliable)
    print("Medición poco fiable:", MEDICION_POCO_FIABLE)
    print("Covarianza poco fiable:\n", covariance_uncertain)
    print("Información poco fiable:\n", information_uncertain)
    print("Residuo común:", RESIDUO_COMUN)
    print(f"Distancia euclídea común: {comparison['euclidean_distance']:.6f}")
    print(
        "Costes de Mahalanobis:",
        np.asarray(comparison["mahalanobis_squared"]),
    )
    print("Media no ponderada:", comparison["unweighted_mean"])
    print("Media fusionada:", comparison["fusion"]["mean"])
    print("Covarianza fusionada:\n", comparison["fusion"]["covariance"])
    print(f"Nodos del grafo: {graph.number_of_nodes()}")
    print(f"Factores de medición: {graph.number_of_edges()}")
    print(f"Estados de animación: {len(result['states'])}")


def main():
    covariance_reliable = crear_covarianza_correlacionada(
        SIGMAS_FIABLE,
        CORRELACION_FIABLE,
    )
    covariance_uncertain = crear_covarianza_correlacionada(
        SIGMAS_POCO_FIABLE,
        CORRELACION_POCO_FIABLE,
    )

    graph = crear_grafo_incertidumbre(
        MEDICION_FIABLE,
        covariance_reliable,
        MEDICION_POCO_FIABLE,
        covariance_uncertain,
    )
    validar_grafo_incertidumbre(graph)

    result = crear_estados_animacion(graph)
    validar_resultados(graph, result)
    validar_grafo_incertidumbre(graph)

    imprimir_resumen(graph, result)

    animator = GraphAnimator(
        figsize=(18, 10),
        interval=560,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "05_optimizacion"
        / "04_incertidumbre_pesos.png"
    )

    animator.animate_uncertainty_information(
        graph=graph,
        states=result["states"],
        title="Incertidumbre, covarianza y matriz de información",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
