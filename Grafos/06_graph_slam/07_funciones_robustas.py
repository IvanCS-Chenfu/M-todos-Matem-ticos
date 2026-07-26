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
INDICE_FALSO_ORIGEN = 3
INDICE_FALSO_DESTINO = 11
EPSILON_JACOBIANO = 1e-7
MAX_ITERACIONES = 35
TOLERANCIA_INCREMENTO = 1e-9
TOLERANCIA_COSTE_RELATIVO = 1e-11
LAMBDA_INICIAL = 1e-3
DELTA_HUBER = 1.0
ESCALA_KERNEL = 1.0

SIGMAS_PRIOR = np.array([0.015, 0.015, np.deg2rad(0.30)], dtype=float)
SIGMAS_ODOMETRIA = np.array([0.085, 0.075, np.deg2rad(1.8)], dtype=float)
SIGMAS_LOOP_CORRECTO = np.array([0.045, 0.045, np.deg2rad(0.75)], dtype=float)
SIGMAS_LOOP_FALSO = np.array([0.040, 0.040, np.deg2rad(0.65)], dtype=float)

SESGO_ESCALA = 1.012
SESGO_LATERAL = 0.007
SESGO_ANGULAR_GRADOS = 0.28


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
# Covarianza, trayectoria y mediciones
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
    """Crea una trayectoria cerrada suave para comparar los optimizadores."""

    numero_poses = int(numero_poses)
    if numero_poses < 10:
        raise ValueError("Se requieren al menos diez poses.")

    parametro = np.linspace(
        0.0,
        2.0 * np.pi,
        numero_poses,
        endpoint=False,
        dtype=float,
    )
    x = 4.7 * np.cos(parametro) + 0.42 * np.cos(2.0 * parametro)
    y = 3.25 * np.sin(parametro) - 0.22 * np.sin(2.0 * parametro)

    x -= x[0]
    y -= y[0]

    dx = -4.7 * np.sin(parametro) - 0.84 * np.sin(2.0 * parametro)
    dy = 3.25 * np.cos(parametro) - 0.44 * np.cos(2.0 * parametro)
    theta = np.arctan2(dy, dx)

    trayectoria = np.column_stack((x, y, theta))
    return validar_trayectoria(trayectoria, "trayectoria real")


def crear_mediciones_odometria(trayectoria_real):
    """Genera odometría determinista con deriva suave."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    ideales = []
    medidas = []

    for indice in range(1, len(trayectoria_real)):
        ideal = calcular_movimiento_relativo(
            trayectoria_real[indice - 1],
            trayectoria_real[indice],
        )

        escala = SESGO_ESCALA + 0.003 * np.sin(0.47 * indice)
        error_longitudinal = 0.005 * np.cos(0.31 * indice)
        error_lateral = SESGO_LATERAL + 0.004 * np.sin(0.73 * indice)
        error_angular = np.deg2rad(
            SESGO_ANGULAR_GRADOS + 0.10 * np.sin(0.57 * indice)
        )

        medida = ideal.copy()
        medida[0] = escala * ideal[0] + error_longitudinal
        medida[1] = escala * ideal[1] + error_lateral
        medida[2] = normalizar_angulo(ideal[2] + error_angular)

        ideales.append(ideal)
        medidas.append(medida)

    return {
        "ideal": validar_trayectoria(np.asarray(ideales), "odometría ideal"),
        "measured": validar_trayectoria(np.asarray(medidas), "odometría medida"),
    }


def integrar_odometria(pose_inicial, mediciones):
    """Integra una secuencia de movimientos relativos."""

    pose_actual = validar_pose(pose_inicial, "pose inicial")
    mediciones = validar_trayectoria(mediciones, "mediciones")
    poses = [pose_actual.copy()]

    for medicion in mediciones:
        pose_actual = componer_poses_se2(pose_actual, medicion)
        poses.append(pose_actual.copy())

    return validar_trayectoria(np.asarray(poses), "trayectoria integrada")


def crear_medicion_cierre_correcto(trayectoria_real):
    """Crea el cierre correcto entre la última y la primera pose."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    medicion = calcular_movimiento_relativo(
        trayectoria_real[-1],
        trayectoria_real[0],
    )
    ruido = np.array([0.010, -0.008, np.deg2rad(0.18)], dtype=float)
    return componer_poses_se2(medicion, ruido)


def crear_medicion_cierre_falso():
    """Crea una medición falsa que afirma que dos poses lejanas coinciden."""

    return np.array([0.18, -0.12, np.deg2rad(1.5)], dtype=float)


# ---------------------------------------------------------------------------
# Grafo de factores
# ---------------------------------------------------------------------------


def crear_pose_graph_robusto(
    trayectoria_real,
    odometria,
    estimacion_inicial,
    incluir_factor_falso=True,
):
    """Crea el pose graph con prior, odometría y dos cierres."""

    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    estimacion_inicial = validar_trayectoria(
        estimacion_inicial,
        "estimación inicial",
    )
    odometria = validar_trayectoria(odometria, "odometría")

    if len(odometria) != len(trayectoria_real) - 1:
        raise ValueError("La odometría debe conectar poses consecutivas.")

    graph = nx.Graph()
    variable_order = [f"x{indice}" for indice in range(len(trayectoria_real))]

    for indice, nombre in enumerate(variable_order):
        graph.add_node(
            nombre,
            node_type="pose",
            index=indice,
            true_pose=trayectoria_real[indice].copy(),
            initial_estimate=estimacion_inicial[indice].copy(),
            estimate=estimacion_inicial[indice].copy(),
        )

    cov_prior = crear_covarianza_diagonal(SIGMAS_PRIOR)
    graph.graph["prior"] = {
        "factor_name": "prior_x0",
        "factor_type": "prior",
        "variables": ("x0",),
        "measurement": trayectoria_real[0].copy(),
        "covariance": cov_prior,
        "information": calcular_matriz_informacion(cov_prior),
        "robust_kernel": None,
    }

    factor_order = []
    cov_odom = crear_covarianza_diagonal(SIGMAS_ODOMETRIA)
    info_odom = calcular_matriz_informacion(cov_odom)

    for indice, medicion in enumerate(odometria, start=1):
        origen = f"x{indice - 1}"
        destino = f"x{indice}"
        nombre = f"odom_{indice - 1}_{indice}"
        graph.add_edge(
            origen,
            destino,
            factor_name=nombre,
            factor_type="odometry",
            variables=(origen, destino),
            measurement=medicion.copy(),
            covariance=cov_odom.copy(),
            information=info_odom.copy(),
            robust_kernel=None,
            is_false=False,
        )
        factor_order.append(nombre)

    cov_correcto = crear_covarianza_diagonal(SIGMAS_LOOP_CORRECTO)
    nombre_correcto = f"loop_{len(trayectoria_real) - 1}_0_correcto"
    graph.add_edge(
        f"x{len(trayectoria_real) - 1}",
        "x0",
        factor_name=nombre_correcto,
        factor_type="loop_correct",
        variables=(f"x{len(trayectoria_real) - 1}", "x0"),
        measurement=crear_medicion_cierre_correcto(trayectoria_real),
        covariance=cov_correcto.copy(),
        information=calcular_matriz_informacion(cov_correcto),
        robust_kernel={"type": "huber", "delta": DELTA_HUBER},
        is_false=False,
    )
    factor_order.append(nombre_correcto)

    nombre_falso = None
    if incluir_factor_falso:
        cov_falso = crear_covarianza_diagonal(SIGMAS_LOOP_FALSO)
        nombre_falso = (
            f"loop_{INDICE_FALSO_ORIGEN}_{INDICE_FALSO_DESTINO}_falso"
        )
        graph.add_edge(
            f"x{INDICE_FALSO_ORIGEN}",
            f"x{INDICE_FALSO_DESTINO}",
            factor_name=nombre_falso,
            factor_type="loop_false",
            variables=(
                f"x{INDICE_FALSO_ORIGEN}",
                f"x{INDICE_FALSO_DESTINO}",
            ),
            measurement=crear_medicion_cierre_falso(),
            covariance=cov_falso.copy(),
            information=calcular_matriz_informacion(cov_falso),
            robust_kernel={"type": "huber", "delta": DELTA_HUBER},
            is_false=True,
        )
        factor_order.append(nombre_falso)

    graph.graph.update(
        {
            "name": "pose_graph_funciones_robustas",
            "variable_order": variable_order,
            "factor_order": factor_order,
            "correct_loop_factor_name": nombre_correcto,
            "false_loop_factor_name": nombre_falso,
            "state_dimension": 3 * len(variable_order),
            "description": "Pose graph con cierre correcto y cierre falso",
        }
    )
    return graph


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
# Funciones de pérdida robusta
# ---------------------------------------------------------------------------


def calcular_norma_mahalanobis(residuo, informacion):
    """Calcula sqrt(eᵀ Ω e)."""

    residuo = np.asarray(residuo, dtype=float)
    informacion = np.asarray(informacion, dtype=float)
    if residuo.shape != (3,):
        raise ValueError("El residuo debe tener tres componentes.")
    if informacion.shape != (3, 3):
        raise ValueError("La información debe tener forma 3x3.")
    valor = float(residuo.T @ informacion @ residuo)
    return float(np.sqrt(max(valor, 0.0)))


def calcular_perdida_cuadratica(r):
    """Calcula ρ(r)=r²/2."""

    r = np.asarray(r, dtype=float)
    return 0.5 * r**2


def calcular_influencia_cuadratica(r):
    """Calcula ψ(r)=r para mínimos cuadrados."""

    return np.asarray(r, dtype=float)


def calcular_peso_cuadratico(r):
    """Devuelve peso unitario para mínimos cuadrados."""

    return np.ones_like(np.asarray(r, dtype=float), dtype=float)


def calcular_perdida_huber(r, delta=DELTA_HUBER):
    """Calcula la pérdida de Huber sobre una norma no negativa."""

    r = np.asarray(r, dtype=float)
    delta = float(delta)
    if delta <= 0.0 or not np.isfinite(delta):
        raise ValueError("delta debe ser positivo y finito.")
    absoluto = np.abs(r)
    return np.where(
        absoluto <= delta,
        0.5 * absoluto**2,
        delta * (absoluto - 0.5 * delta),
    )


def calcular_influencia_huber(r, delta=DELTA_HUBER):
    """Calcula la influencia limitada de Huber."""

    r = np.asarray(r, dtype=float)
    delta = float(delta)
    absoluto = np.abs(r)
    return np.where(absoluto <= delta, r, delta * np.sign(r))


def calcular_peso_huber(r, delta=DELTA_HUBER):
    """Calcula el peso IRLS de Huber."""

    r = np.asarray(r, dtype=float)
    delta = float(delta)
    absoluto = np.abs(r)
    peso = np.ones_like(absoluto, dtype=float)
    mascara = absoluto > delta
    peso[mascara] = delta / absoluto[mascara]
    return peso


def calcular_perdida_cauchy(r, escala=ESCALA_KERNEL):
    """Calcula la pérdida de Cauchy."""

    r = np.asarray(r, dtype=float)
    escala = float(escala)
    return 0.5 * escala**2 * np.log1p((r / escala) ** 2)


def calcular_influencia_cauchy(r, escala=ESCALA_KERNEL):
    """Calcula la influencia de Cauchy."""

    r = np.asarray(r, dtype=float)
    escala = float(escala)
    return r / (1.0 + (r / escala) ** 2)


def calcular_peso_cauchy(r, escala=ESCALA_KERNEL):
    """Calcula el peso de Cauchy."""

    r = np.asarray(r, dtype=float)
    escala = float(escala)
    return 1.0 / (1.0 + (r / escala) ** 2)


def calcular_perdida_tukey(r, escala=ESCALA_KERNEL):
    """Calcula la pérdida biweight de Tukey."""

    r = np.asarray(r, dtype=float)
    escala = float(escala)
    u = r / escala
    dentro = np.abs(u) <= 1.0
    resultado = np.full_like(r, escala**2 / 6.0, dtype=float)
    resultado[dentro] = (escala**2 / 6.0) * (
        1.0 - (1.0 - u[dentro] ** 2) ** 3
    )
    return resultado


def calcular_influencia_tukey(r, escala=ESCALA_KERNEL):
    """Calcula la influencia redescendente de Tukey."""

    r = np.asarray(r, dtype=float)
    escala = float(escala)
    u = r / escala
    dentro = np.abs(u) <= 1.0
    resultado = np.zeros_like(r, dtype=float)
    resultado[dentro] = r[dentro] * (1.0 - u[dentro] ** 2) ** 2
    return resultado


def calcular_peso_tukey(r, escala=ESCALA_KERNEL):
    """Calcula el peso de Tukey."""

    r = np.asarray(r, dtype=float)
    escala = float(escala)
    u = r / escala
    dentro = np.abs(u) <= 1.0
    resultado = np.zeros_like(r, dtype=float)
    resultado[dentro] = (1.0 - u[dentro] ** 2) ** 2
    return resultado


def calcular_perdida_geman_mcclure(r, escala=ESCALA_KERNEL):
    """Calcula la pérdida de Geman-McClure."""

    r = np.asarray(r, dtype=float)
    escala = float(escala)
    u2 = (r / escala) ** 2
    return 0.5 * r**2 / (1.0 + u2)


def calcular_influencia_geman_mcclure(r, escala=ESCALA_KERNEL):
    """Calcula la influencia de Geman-McClure."""

    r = np.asarray(r, dtype=float)
    escala = float(escala)
    u2 = (r / escala) ** 2
    return r / (1.0 + u2) ** 2


def calcular_peso_geman_mcclure(r, escala=ESCALA_KERNEL):
    """Calcula el peso de Geman-McClure."""

    r = np.asarray(r, dtype=float)
    escala = float(escala)
    u2 = (r / escala) ** 2
    return 1.0 / (1.0 + u2) ** 2


def crear_curvas_kernels(maximo=18.0, numero=361):
    """Evalúa pérdida, influencia y peso para cinco kernels."""

    r = np.linspace(0.0, float(maximo), int(numero), dtype=float)
    curvas = {
        "least_squares": {
            "loss": calcular_perdida_cuadratica(r),
            "influence": calcular_influencia_cuadratica(r),
            "weight": calcular_peso_cuadratico(r),
        },
        "huber": {
            "loss": calcular_perdida_huber(r),
            "influence": calcular_influencia_huber(r),
            "weight": calcular_peso_huber(r),
        },
        "cauchy": {
            "loss": calcular_perdida_cauchy(r),
            "influence": calcular_influencia_cauchy(r),
            "weight": calcular_peso_cauchy(r),
        },
        "tukey": {
            "loss": calcular_perdida_tukey(r),
            "influence": calcular_influencia_tukey(r),
            "weight": calcular_peso_tukey(r),
        },
        "geman_mcclure": {
            "loss": calcular_perdida_geman_mcclure(r),
            "influence": calcular_influencia_geman_mcclure(r),
            "weight": calcular_peso_geman_mcclure(r),
        },
    }
    return {
        "r": r,
        "curves": curvas,
        "delta_huber": DELTA_HUBER,
        "scale": ESCALA_KERNEL,
    }


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
    """Calcula z⁻¹ ⊕ (xᵢ⁻¹ ⊕ xⱼ)."""

    prediccion = calcular_prediccion_relativa(pose_origen, pose_destino)
    return calcular_movimiento_relativo(medicion, prediccion)


def calcular_residuo_factor(graph, factor_name, poses):
    """Calcula el residuo de un prior, odometría o cierre."""

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
    """Calcula bloques jacobianos mediante diferencias centrales."""

    poses = validar_trayectoria(poses, "poses")
    epsilon = float(epsilon)
    if epsilon <= 0.0 or not np.isfinite(epsilon):
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
                poses[indice_pose],
                delta,
            )
            poses_menos[indice_pose] = aplicar_incremento_local(
                poses[indice_pose],
                -delta,
            )

            residuo_mas = calcular_residuo_factor(
                graph,
                factor_name,
                poses_mas,
            )
            residuo_menos = calcular_residuo_factor(
                graph,
                factor_name,
                poses_menos,
            )
            diferencia = residuo_mas - residuo_menos
            diferencia[2] = normalizar_angulo(diferencia[2])
            bloque[:, componente] = diferencia / (2.0 * epsilon)

        bloques[nombre] = bloque

    return bloques


def calcular_peso_factor(factor, residuo, modo):
    """Calcula el peso efectivo de un factor para el modo solicitado."""

    if modo == "quadratic":
        return 1.0
    if modo != "huber":
        raise ValueError(f"Modo de optimización desconocido: {modo!r}")

    kernel = factor.get("robust_kernel")
    if not kernel or kernel.get("type") != "huber":
        return 1.0

    norma = calcular_norma_mahalanobis(residuo, factor["information"])
    return float(calcular_peso_huber(np.array([norma]), kernel["delta"])[0])


def calcular_coste_factor(factor, residuo, modo):
    """Calcula el coste cuadrático o robusto de un factor."""

    norma = calcular_norma_mahalanobis(residuo, factor["information"])
    if modo == "quadratic":
        return float(calcular_perdida_cuadratica(np.array([norma]))[0])
    if modo == "huber":
        kernel = factor.get("robust_kernel")
        if kernel and kernel.get("type") == "huber":
            return float(
                calcular_perdida_huber(
                    np.array([norma]),
                    kernel["delta"],
                )[0]
            )
        return float(calcular_perdida_cuadratica(np.array([norma]))[0])
    raise ValueError(f"Modo de optimización desconocido: {modo!r}")


def ensamblar_sistema(
    graph,
    poses,
    modo="quadratic",
    incluir_prior=True,
):
    """Ensambla residuo, J, Ω efectiva, Hessiana, gradiente y costes."""

    poses = validar_trayectoria(poses, "poses")
    factores = list(graph.graph["factor_order"])
    if incluir_prior:
        factores = ["prior_x0"] + factores

    numero_filas = 3 * len(factores)
    numero_columnas = 3 * len(graph.graph["variable_order"])
    residual = np.zeros(numero_filas, dtype=float)
    jacobiano = np.zeros((numero_filas, numero_columnas), dtype=float)
    informacion_efectiva = np.zeros((numero_filas, numero_filas), dtype=float)
    indices = {
        nombre: indice
        for indice, nombre in enumerate(graph.graph["variable_order"])
    }

    factor_slices = {}
    weights = {}
    mahalanobis = {}
    factor_costs = {}
    cost_by_type = {
        "prior": 0.0,
        "odometry": 0.0,
        "loop_correct": 0.0,
        "loop_false": 0.0,
    }

    for indice_factor, factor_name in enumerate(factores):
        filas = slice(3 * indice_factor, 3 * indice_factor + 3)
        factor = obtener_factor(graph, factor_name)
        residuo = calcular_residuo_factor(graph, factor_name, poses)
        residual[filas] = residuo

        peso = calcular_peso_factor(factor, residuo, modo)
        coste = calcular_coste_factor(factor, residuo, modo)
        norma = calcular_norma_mahalanobis(residuo, factor["information"])

        informacion_efectiva[filas, filas] = peso * factor["information"]
        weights[factor_name] = peso
        mahalanobis[factor_name] = norma
        factor_costs[factor_name] = coste
        cost_by_type[factor["factor_type"]] += coste

        bloques = calcular_jacobianos_locales_numericos(
            graph,
            factor_name,
            poses,
        )
        for variable, bloque in bloques.items():
            columna = 3 * indices[variable]
            jacobiano[filas, columna : columna + 3] = bloque

        factor_slices[factor_name] = filas

    hessiana = jacobiano.T @ informacion_efectiva @ jacobiano
    gradiente = jacobiano.T @ informacion_efectiva @ residual
    coste_total = float(sum(factor_costs.values()))

    return {
        "mode": modo,
        "factor_order": factores,
        "factor_slices": factor_slices,
        "residual": residual,
        "jacobian": jacobiano,
        "information": informacion_efectiva,
        "hessian": hessiana,
        "gradient": gradiente,
        "cost": coste_total,
        "cost_by_type": cost_by_type,
        "factor_costs": factor_costs,
        "weights": weights,
        "mahalanobis": mahalanobis,
    }


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


def calcular_metricas_trayectoria(trayectoria_real, trayectoria):
    """Calcula errores de posición y orientación respecto a la referencia."""

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


def analizar_factor(graph, poses, factor_name, modo):
    """Resume residuo, Mahalanobis, peso y coste de un factor."""

    factor = obtener_factor(graph, factor_name)
    residuo = calcular_residuo_factor(graph, factor_name, poses)
    return {
        "residual": residuo,
        "translation": float(np.linalg.norm(residuo[:2])),
        "orientation_deg": float(np.rad2deg(abs(residuo[2]))),
        "mahalanobis": calcular_norma_mahalanobis(
            residuo,
            factor["information"],
        ),
        "weight": calcular_peso_factor(factor, residuo, modo),
        "cost": calcular_coste_factor(factor, residuo, modo),
    }


# ---------------------------------------------------------------------------
# Optimización con Levenberg-Marquardt e IRLS
# ---------------------------------------------------------------------------


def resolver_incremento_lm(hessiana, gradiente, damping):
    """Resuelve el sistema amortiguado de Levenberg-Marquardt."""

    hessiana = np.asarray(hessiana, dtype=float)
    gradiente = np.asarray(gradiente, dtype=float)
    damping = float(damping)

    if hessiana.shape[0] != hessiana.shape[1]:
        raise ValueError("La Hessiana debe ser cuadrada.")
    if gradiente.shape != (hessiana.shape[0],):
        raise ValueError("El gradiente tiene dimensión incorrecta.")
    if damping <= 0.0 or not np.isfinite(damping):
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
    trayectoria_real,
    modo="quadratic",
    max_iteraciones=MAX_ITERACIONES,
):
    """Optimiza el mismo grafo con coste cuadrático o Huber mediante IRLS."""

    poses = validar_trayectoria(poses_iniciales, "poses iniciales")
    trayectoria_real = validar_trayectoria(trayectoria_real, "trayectoria real")
    damping = LAMBDA_INICIAL
    history = []
    converged = False
    false_name = graph.graph.get("false_loop_factor_name")
    correct_name = graph.graph.get("correct_loop_factor_name")

    for iteration in range(int(max_iteraciones)):
        sistema = ensamblar_sistema(graph, poses, modo=modo)
        coste_antes = sistema["cost"]
        gradiente_norma = float(np.linalg.norm(sistema["gradient"]))
        aceptado = False
        mejor_intento = None

        for intento in range(14):
            incremento = resolver_incremento_lm(
                sistema["hessian"],
                sistema["gradient"],
                damping,
            )
            norma_incremento = float(np.linalg.norm(incremento))
            candidato = aplicar_incremento_estado(poses, incremento)
            sistema_candidato = ensamblar_sistema(
                graph,
                candidato,
                modo=modo,
            )

            mejor_intento = {
                "iteration": iteration,
                "attempt": intento,
                "poses_before": poses.copy(),
                "poses_candidate": candidato.copy(),
                "cost_before": coste_antes,
                "cost_candidate": sistema_candidato["cost"],
                "damping": damping,
                "step_norm": norma_incremento,
                "gradient_norm": gradiente_norma,
                "accepted": sistema_candidato["cost"] < coste_antes,
                "weights_before": dict(sistema["weights"]),
                "mahalanobis_before": dict(sistema["mahalanobis"]),
            }

            if sistema_candidato["cost"] < coste_antes:
                poses = candidato
                damping = max(damping * 0.32, 1e-10)
                aceptado = True
                break

            damping = min(damping * 8.0, 1e12)

        if mejor_intento is None:
            raise RuntimeError("No se evaluó ningún paso de optimización.")

        sistema_despues = ensamblar_sistema(graph, poses, modo=modo)
        metricas = calcular_metricas_trayectoria(trayectoria_real, poses)
        pesos_correctos = [
            peso
            for nombre, peso in sistema_despues["weights"].items()
            if nombre != false_name
        ]
        reducidos = sum(
            peso < 0.999999
            for peso in sistema_despues["weights"].values()
        )

        mejor_intento.update(
            {
                "poses_after": poses.copy(),
                "cost_after": sistema_despues["cost"],
                "cost_by_type_after": dict(sistema_despues["cost_by_type"]),
                "accepted": aceptado,
                "damping_after": damping,
                "rmse_after": metricas["position_rmse"],
                "angle_rmse_after_deg": metricas["orientation_rmse_deg"],
                "weights_after": dict(sistema_despues["weights"]),
                "mahalanobis_after": dict(sistema_despues["mahalanobis"]),
                "false_weight_after": (
                    sistema_despues["weights"].get(false_name)
                    if false_name is not None
                    else None
                ),
                "false_mahalanobis_after": (
                    sistema_despues["mahalanobis"].get(false_name)
                    if false_name is not None
                    else None
                ),
                "correct_loop_weight_after": sistema_despues["weights"].get(
                    correct_name,
                    1.0,
                ),
                "minimum_weight_after": float(
                    min(sistema_despues["weights"].values())
                ),
                "mean_inlier_weight_after": float(np.mean(pesos_correctos)),
                "reduced_factor_count_after": int(reducidos),
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
        "mode": modo,
        "initial_poses": validar_trayectoria(poses_iniciales),
        "optimized_poses": poses,
        "history": history,
        "iterations": len(history),
        "converged": converged,
        "final_system": ensamblar_sistema(graph, poses, modo=modo),
    }


# ---------------------------------------------------------------------------
# Resultado completo
# ---------------------------------------------------------------------------


def crear_resultado_funciones_robustas():
    """Compara referencia limpia, mínimos cuadrados y Huber."""

    trayectoria_real = crear_trayectoria_real()
    odometria = crear_mediciones_odometria(trayectoria_real)
    trayectoria_inicial = integrar_odometria(
        trayectoria_real[0],
        odometria["measured"],
    )

    graph_clean = crear_pose_graph_robusto(
        trayectoria_real,
        odometria["measured"],
        trayectoria_inicial,
        incluir_factor_falso=False,
    )
    graph_full = crear_pose_graph_robusto(
        trayectoria_real,
        odometria["measured"],
        trayectoria_inicial,
        incluir_factor_falso=True,
    )

    optimizacion_limpia = optimizar_pose_graph(
        graph_clean,
        trayectoria_inicial,
        trayectoria_real,
        modo="quadratic",
    )
    optimizacion_cuadratica = optimizar_pose_graph(
        graph_full,
        trayectoria_inicial,
        trayectoria_real,
        modo="quadratic",
    )
    optimizacion_robusta = optimizar_pose_graph(
        graph_full,
        trayectoria_inicial,
        trayectoria_real,
        modo="huber",
    )

    trayectoria_limpia = optimizacion_limpia["optimized_poses"]
    trayectoria_cuadratica = optimizacion_cuadratica["optimized_poses"]
    trayectoria_robusta = optimizacion_robusta["optimized_poses"]

    false_name = graph_full.graph["false_loop_factor_name"]
    correct_name = graph_full.graph["correct_loop_factor_name"]

    resultado = {
        "true_trajectory": trayectoria_real,
        "initial_trajectory": trayectoria_inicial,
        "clean_trajectory": trayectoria_limpia,
        "quadratic_trajectory": trayectoria_cuadratica,
        "robust_trajectory": trayectoria_robusta,
        "odometry": odometria,
        "graph_clean": graph_clean,
        "graph": graph_full,
        "correct_loop_factor_name": correct_name,
        "false_loop_factor_name": false_name,
        "clean_optimization": optimizacion_limpia,
        "quadratic_optimization": optimizacion_cuadratica,
        "robust_optimization": optimizacion_robusta,
        "initial_quadratic_system": ensamblar_sistema(
            graph_full,
            trayectoria_inicial,
            modo="quadratic",
        ),
        "initial_robust_system": ensamblar_sistema(
            graph_full,
            trayectoria_inicial,
            modo="huber",
        ),
        "final_quadratic_system": optimizacion_cuadratica["final_system"],
        "final_robust_system": optimizacion_robusta["final_system"],
        "initial_metrics": calcular_metricas_trayectoria(
            trayectoria_real,
            trayectoria_inicial,
        ),
        "clean_metrics": calcular_metricas_trayectoria(
            trayectoria_real,
            trayectoria_limpia,
        ),
        "quadratic_metrics": calcular_metricas_trayectoria(
            trayectoria_real,
            trayectoria_cuadratica,
        ),
        "robust_metrics": calcular_metricas_trayectoria(
            trayectoria_real,
            trayectoria_robusta,
        ),
        "false_factor": {
            "initial_quadratic": analizar_factor(
                graph_full,
                trayectoria_inicial,
                false_name,
                "quadratic",
            ),
            "initial_robust": analizar_factor(
                graph_full,
                trayectoria_inicial,
                false_name,
                "huber",
            ),
            "final_quadratic": analizar_factor(
                graph_full,
                trayectoria_cuadratica,
                false_name,
                "quadratic",
            ),
            "final_robust": analizar_factor(
                graph_full,
                trayectoria_robusta,
                false_name,
                "huber",
            ),
        },
        "correct_factor": {
            "final_quadratic": analizar_factor(
                graph_full,
                trayectoria_cuadratica,
                correct_name,
                "quadratic",
            ),
            "final_robust": analizar_factor(
                graph_full,
                trayectoria_robusta,
                correct_name,
                "huber",
            ),
        },
        "kernel_curves": crear_curvas_kernels(),
        "parameters": {
            "delta_huber": DELTA_HUBER,
            "kernel_scale": ESCALA_KERNEL,
            "false_origin": INDICE_FALSO_ORIGEN,
            "false_target": INDICE_FALSO_DESTINO,
        },
    }
    return resultado


# ---------------------------------------------------------------------------
# Estados didácticos
# ---------------------------------------------------------------------------


def interpolar_trayectorias(origen, destino, alpha):
    """Interpola dos trayectorias respetando el camino angular corto."""

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


def crear_estado_animacion(phase, message, **kwargs):
    """Crea un estado visual independiente de Matplotlib."""

    estado = {
        "phase": phase,
        "message": str(message),
        "visible_pose_count": kwargs.get("visible_pose_count", 0),
        "visible_odometry_count": kwargs.get("visible_odometry_count", 0),
        "show_true": bool(kwargs.get("show_true", False)),
        "show_initial": bool(kwargs.get("show_initial", False)),
        "show_prior": bool(kwargs.get("show_prior", False)),
        "show_correct_loop": bool(kwargs.get("show_correct_loop", False)),
        "show_false_loop": bool(kwargs.get("show_false_loop", False)),
        "show_quadratic": bool(kwargs.get("show_quadratic", False)),
        "show_robust": bool(kwargs.get("show_robust", False)),
        "show_clean": bool(kwargs.get("show_clean", False)),
        "show_curves": bool(kwargs.get("show_curves", False)),
        "show_history": bool(kwargs.get("show_history", False)),
        "show_summary": bool(kwargs.get("show_summary", False)),
        "quadratic_poses": kwargs.get("quadratic_poses"),
        "robust_poses": kwargs.get("robust_poses"),
        "active_method": kwargs.get("active_method"),
        "iteration": kwargs.get("iteration"),
        "cost": kwargs.get("cost"),
        "rmse": kwargs.get("rmse"),
        "angle_rmse_deg": kwargs.get("angle_rmse_deg"),
        "false_weight": kwargs.get("false_weight"),
        "false_mahalanobis": kwargs.get("false_mahalanobis"),
        "correct_weight": kwargs.get("correct_weight"),
        "damping": kwargs.get("damping"),
        "step_norm": kwargs.get("step_norm"),
        "accepted": kwargs.get("accepted"),
        "curve_marker": kwargs.get("curve_marker"),
    }
    return estado


def crear_estados_animacion(resultado):
    """Crea la comparación animada entre mínimos cuadrados y Huber."""

    true = resultado["true_trajectory"]
    initial = resultado["initial_trajectory"]
    quadratic = resultado["quadratic_trajectory"]
    robust = resultado["robust_trajectory"]
    numero_poses = len(true)
    numero_odometrias = numero_poses - 1
    states = []

    def add(phase, message, repeat=1, **kwargs):
        for _ in range(repeat):
            states.append(crear_estado_animacion(phase, message, **kwargs))

    add(
        "introduction",
        "Funciones robustas: el mismo grafo, una arista falsa y dos objetivos.",
        repeat=3,
        show_true=True,
    )

    for count in range(1, numero_poses + 1):
        add(
            "odometry_build",
            "La odometría construye una estimación inicial con deriva.",
            visible_pose_count=count,
            visible_odometry_count=max(0, count - 1),
            show_true=True,
            show_initial=True,
            show_prior=True,
        )

    add(
        "correct_loop",
        "El cierre correcto conecta la última pose con el inicio.",
        repeat=4,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_correct_loop=True,
    )

    add(
        "false_loop",
        "La arista roja afirma falsamente que x3 y x11 representan el mismo lugar.",
        repeat=5,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_correct_loop=True,
        show_false_loop=True,
        false_weight=1.0,
        false_mahalanobis=resultado["false_factor"]["initial_quadratic"]["mahalanobis"],
    )

    for marker in (0.0, 1.5, 3.0, 5.0, 8.0, 12.0, 16.0):
        add(
            "kernel_curves",
            "La pérdida robusta coincide cerca de cero y crece más lentamente para outliers.",
            visible_pose_count=numero_poses,
            visible_odometry_count=numero_odometrias,
            show_true=True,
            show_initial=True,
            show_prior=True,
            show_correct_loop=True,
            show_false_loop=True,
            show_curves=True,
            curve_marker=marker,
        )

    historial_cuadratico = resultado["quadratic_optimization"]["history"]
    indices_cuadraticos = np.unique(
        np.linspace(
            0,
            len(historial_cuadratico) - 1,
            min(12, len(historial_cuadratico)),
            dtype=int,
        )
    )
    for indice_historial in indices_cuadraticos:
        entry = historial_cuadratico[int(indice_historial)]
        for alpha in (0.0, 0.50, 1.0):
            poses = interpolar_trayectorias(
                entry["poses_before"],
                entry["poses_after"],
                alpha,
            )
            sistema = ensamblar_sistema(
                resultado["graph"],
                poses,
                modo="quadratic",
            )
            metricas = calcular_metricas_trayectoria(true, poses)
            add(
                "quadratic_optimization",
                "Mínimos cuadrados mantiene peso uno y deforma la trayectoria para satisfacer el falso cierre.",
                visible_pose_count=numero_poses,
                visible_odometry_count=numero_odometrias,
                show_true=True,
                show_initial=True,
                show_prior=True,
                show_correct_loop=True,
                show_false_loop=True,
                show_quadratic=True,
                quadratic_poses=poses,
                active_method="quadratic",
                iteration=entry["iteration"] + 1,
                cost=sistema["cost"],
                rmse=metricas["position_rmse"],
                angle_rmse_deg=metricas["orientation_rmse_deg"],
                false_weight=1.0,
                false_mahalanobis=sistema["mahalanobis"][resultado["false_loop_factor_name"]],
                correct_weight=1.0,
                damping=entry["damping"],
                step_norm=entry["step_norm"],
                accepted=entry["accepted"],
                show_curves=True,
                show_history=True,
            )

    add(
        "reset",
        "Se reinicia exactamente desde la misma estimación inicial.",
        repeat=4,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_correct_loop=True,
        show_false_loop=True,
        show_quadratic=True,
        quadratic_poses=quadratic,
        show_curves=True,
    )

    historial_robusto = resultado["robust_optimization"]["history"]
    for entry in historial_robusto:
        for alpha in (0.0, 0.50, 1.0):
            poses = interpolar_trayectorias(
                entry["poses_before"],
                entry["poses_after"],
                alpha,
            )
            sistema = ensamblar_sistema(
                resultado["graph"],
                poses,
                modo="huber",
            )
            metricas = calcular_metricas_trayectoria(true, poses)
            add(
                "robust_optimization",
                "IRLS recalcula el peso de Huber: el falso factor pierde influencia.",
                visible_pose_count=numero_poses,
                visible_odometry_count=numero_odometrias,
                show_true=True,
                show_initial=True,
                show_prior=True,
                show_correct_loop=True,
                show_false_loop=True,
                show_quadratic=True,
                quadratic_poses=quadratic,
                show_robust=True,
                robust_poses=poses,
                active_method="huber",
                iteration=entry["iteration"] + 1,
                cost=sistema["cost"],
                rmse=metricas["position_rmse"],
                angle_rmse_deg=metricas["orientation_rmse_deg"],
                false_weight=sistema["weights"][resultado["false_loop_factor_name"]],
                false_mahalanobis=sistema["mahalanobis"][resultado["false_loop_factor_name"]],
                correct_weight=sistema["weights"][resultado["correct_loop_factor_name"]],
                damping=entry["damping"],
                step_norm=entry["step_norm"],
                accepted=entry["accepted"],
                show_curves=True,
                show_history=True,
            )

    add(
        "comparison",
        "Comparación final: la solución robusta permanece próxima a la referencia limpia.",
        repeat=6,
        visible_pose_count=numero_poses,
        visible_odometry_count=numero_odometrias,
        show_true=True,
        show_initial=True,
        show_prior=True,
        show_correct_loop=True,
        show_false_loop=True,
        show_clean=True,
        show_quadratic=True,
        quadratic_poses=quadratic,
        show_robust=True,
        robust_poses=robust,
        show_curves=True,
        show_history=True,
        show_summary=True,
        active_method="comparison",
        iteration=resultado["robust_optimization"]["iterations"],
        cost=resultado["final_robust_system"]["cost"],
        rmse=resultado["robust_metrics"]["position_rmse"],
        angle_rmse_deg=resultado["robust_metrics"]["orientation_rmse_deg"],
        false_weight=resultado["false_factor"]["final_robust"]["weight"],
        false_mahalanobis=resultado["false_factor"]["final_robust"]["mahalanobis"],
        correct_weight=resultado["correct_factor"]["final_robust"]["weight"],
    )

    for step, state in enumerate(states, start=1):
        state["step"] = step
        state["total_steps"] = len(states)

    return states


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------


def validar_kernels(resultado):
    """Comprueba continuidad, positividad y comportamiento de los kernels."""

    curvas = resultado["kernel_curves"]
    r = curvas["r"]
    if r.ndim != 1 or len(r) < 100:
        raise AssertionError("El dominio de las curvas es insuficiente.")

    for nombre, datos in curvas["curves"].items():
        for clave in ("loss", "influence", "weight"):
            valores = np.asarray(datos[clave], dtype=float)
            if valores.shape != r.shape or not np.all(np.isfinite(valores)):
                raise AssertionError(f"Curva inválida: {nombre}/{clave}.")
        if np.any(datos["loss"] < -1e-12):
            raise AssertionError("Las pérdidas no pueden ser negativas.")
        if np.any(datos["weight"] < -1e-12):
            raise AssertionError("Los pesos no pueden ser negativos.")

    delta = DELTA_HUBER
    epsilon = 1e-8
    izquierda = float(calcular_perdida_huber(np.array([delta - epsilon]))[0])
    derecha = float(calcular_perdida_huber(np.array([delta + epsilon]))[0])
    if abs(izquierda - derecha) > 1e-6:
        raise AssertionError("La pérdida de Huber no es continua.")

    influencia_izquierda = float(
        calcular_influencia_huber(np.array([delta - epsilon]))[0]
    )
    influencia_derecha = float(
        calcular_influencia_huber(np.array([delta + epsilon]))[0]
    )
    if abs(influencia_izquierda - influencia_derecha) > 1e-6:
        raise AssertionError("La influencia de Huber no es continua.")

    if not np.isclose(
        calcular_peso_huber(np.array([0.5 * delta]))[0],
        1.0,
        atol=1e-12,
    ):
        raise AssertionError("Huber debe conservar peso uno bajo el umbral.")
    if not calcular_peso_huber(np.array([4.0 * delta]))[0] < 1.0:
        raise AssertionError("Huber debe reducir pesos sobre el umbral.")


def validar_grafo(resultado):
    """Comprueba nodos, factores, conectividad y falso cierre."""

    graph = resultado["graph"]
    if graph.number_of_nodes() != NUMERO_POSES:
        raise AssertionError("Número de poses incorrecto.")
    if not nx.is_connected(graph):
        raise AssertionError("El pose graph debe ser conexo.")
    if graph.graph["false_loop_factor_name"] is None:
        raise AssertionError("Debe existir una arista falsa.")
    false_factor = obtener_factor(graph, graph.graph["false_loop_factor_name"])
    if not false_factor.get("is_false", False):
        raise AssertionError("La arista falsa debe estar identificada.")
    if "prior" not in graph.graph:
        raise AssertionError("El grafo debe contener un prior.")


def validar_sistemas(resultado):
    """Comprueba dimensiones, simetría y construcción de H y g."""

    dimension = 3 * NUMERO_POSES
    for clave in (
        "initial_quadratic_system",
        "initial_robust_system",
        "final_quadratic_system",
        "final_robust_system",
    ):
        sistema = resultado[clave]
        h = sistema["hessian"]
        j = sistema["jacobian"]
        omega = sistema["information"]
        e = sistema["residual"]
        if h.shape != (dimension, dimension):
            raise AssertionError("Dimensión de Hessiana incorrecta.")
        if j.shape[1] != dimension:
            raise AssertionError("Dimensión de Jacobiano incorrecta.")
        if not np.allclose(h, h.T, atol=1e-8):
            raise AssertionError("La Hessiana debe ser simétrica.")
        if not np.allclose(h, j.T @ omega @ j, atol=1e-7):
            raise AssertionError("Debe cumplirse H=JᵀΩJ.")
        if not np.allclose(
            sistema["gradient"],
            j.T @ omega @ e,
            atol=1e-7,
        ):
            raise AssertionError("Debe cumplirse g=JᵀΩe.")
        if sistema["cost"] < 0.0 or not np.isfinite(sistema["cost"]):
            raise AssertionError("El coste debe ser no negativo y finito.")


def validar_optimizaciones(resultado):
    """Comprueba convergencia y ventaja de la solución robusta."""

    quadratic = resultado["quadratic_optimization"]
    robust = resultado["robust_optimization"]
    clean = resultado["clean_optimization"]

    for optimizacion in (clean, quadratic, robust):
        if not optimizacion["history"]:
            raise AssertionError("Cada optimización debe producir historial.")
        costes = [entrada["cost_after"] for entrada in optimizacion["history"]]
        if any(
            siguiente > anterior + 1e-9
            for anterior, siguiente in zip(costes[:-1], costes[1:])
        ):
            raise AssertionError("Los costes aceptados no deben aumentar.")

    rmse_q = resultado["quadratic_metrics"]["position_rmse"]
    rmse_r = resultado["robust_metrics"]["position_rmse"]
    rmse_clean = resultado["clean_metrics"]["position_rmse"]
    if not rmse_r < rmse_q:
        raise AssertionError("La solución robusta debe mejorar el RMSE.")
    if not abs(rmse_r - rmse_clean) < abs(rmse_q - rmse_clean):
        raise AssertionError("La solución robusta debe acercarse a la referencia limpia.")

    peso_falso = resultado["false_factor"]["final_robust"]["weight"]
    peso_correcto = resultado["correct_factor"]["final_robust"]["weight"]
    if not 0.0 < peso_falso < 0.5:
        raise AssertionError("El falso factor debe quedar fuertemente atenuado.")
    if not peso_correcto > 0.9:
        raise AssertionError("El cierre correcto debe conservar peso alto.")


def validar_resultados(resultado, estados):
    """Ejecuta todas las comprobaciones y devuelve un resumen."""

    validar_kernels(resultado)
    validar_grafo(resultado)
    validar_sistemas(resultado)
    validar_optimizaciones(resultado)

    if len(estados) < 70:
        raise AssertionError("Se requieren suficientes estados didácticos.")

    false_initial = resultado["false_factor"]["initial_robust"]
    false_final = resultado["false_factor"]["final_robust"]

    return {
        "pose_count": NUMERO_POSES,
        "odometry_count": NUMERO_POSES - 1,
        "factor_count": 1 + len(resultado["graph"].graph["factor_order"]),
        "state_dimension": 3 * NUMERO_POSES,
        "state_count": len(estados),
        "clean_iterations": resultado["clean_optimization"]["iterations"],
        "quadratic_iterations": resultado["quadratic_optimization"]["iterations"],
        "robust_iterations": resultado["robust_optimization"]["iterations"],
        "initial_rmse": resultado["initial_metrics"]["position_rmse"],
        "clean_rmse": resultado["clean_metrics"]["position_rmse"],
        "quadratic_rmse": resultado["quadratic_metrics"]["position_rmse"],
        "robust_rmse": resultado["robust_metrics"]["position_rmse"],
        "initial_angle_rmse_deg": resultado["initial_metrics"]["orientation_rmse_deg"],
        "quadratic_angle_rmse_deg": resultado["quadratic_metrics"]["orientation_rmse_deg"],
        "robust_angle_rmse_deg": resultado["robust_metrics"]["orientation_rmse_deg"],
        "initial_false_mahalanobis": false_initial["mahalanobis"],
        "final_false_mahalanobis": false_final["mahalanobis"],
        "initial_false_weight": false_initial["weight"],
        "final_false_weight": false_final["weight"],
        "final_correct_weight": resultado["correct_factor"]["final_robust"]["weight"],
        "initial_quadratic_cost": resultado["initial_quadratic_system"]["cost"],
        "final_quadratic_cost": resultado["final_quadratic_system"]["cost"],
        "initial_robust_cost": resultado["initial_robust_system"]["cost"],
        "final_robust_cost": resultado["final_robust_system"]["cost"],
        "jacobian_shape": resultado["final_robust_system"]["jacobian"].shape,
        "hessian_shape": resultado["final_robust_system"]["hessian"].shape,
        "quadratic_converged": resultado["quadratic_optimization"]["converged"],
        "robust_converged": resultado["robust_optimization"]["converged"],
    }


def main():
    """Ejecuta el ejemplo, valida resultados y lanza la representación."""

    resultado = crear_resultado_funciones_robustas()
    estados = crear_estados_animacion(resultado)
    resumen = validar_resultados(resultado, estados)

    print("\n=== FUNCIONES ROBUSTAS ===")
    print(f"Poses: {resumen['pose_count']}")
    print(f"Factores totales: {resumen['factor_count']}")
    print(f"Dimensión del estado: {resumen['state_dimension']}")
    print(f"Estados de la demostración: {resumen['state_count']}")
    print()
    print(
        "RMSE posición: "
        f"inicial={resumen['initial_rmse']:.6f} m · "
        f"cuadrático={resumen['quadratic_rmse']:.6f} m · "
        f"robusto={resumen['robust_rmse']:.6f} m · "
        f"limpio={resumen['clean_rmse']:.6f} m"
    )
    print(
        "RMSE angular: "
        f"inicial={resumen['initial_angle_rmse_deg']:.6f}° · "
        f"cuadrático={resumen['quadratic_angle_rmse_deg']:.6f}° · "
        f"robusto={resumen['robust_angle_rmse_deg']:.6f}°"
    )
    print(
        "Falso factor: "
        f"Mahalanobis {resumen['initial_false_mahalanobis']:.3f} → "
        f"{resumen['final_false_mahalanobis']:.3f} · "
        f"peso Huber {resumen['initial_false_weight']:.5f} → "
        f"{resumen['final_false_weight']:.5f}"
    )
    print(
        "Peso final del cierre correcto: "
        f"{resumen['final_correct_weight']:.6f}"
    )
    print(
        "Coste cuadrático: "
        f"{resumen['initial_quadratic_cost']:.6f} → "
        f"{resumen['final_quadratic_cost']:.6f}"
    )
    print(
        "Coste robusto: "
        f"{resumen['initial_robust_cost']:.6f} → "
        f"{resumen['final_robust_cost']:.6f}"
    )
    print(f"J: {resumen['jacobian_shape']} · H: {resumen['hessian_shape']}")

    ruta_imagen = (
        GRAFOS_DIR
        / "assets"
        / "06_graph_slam"
        / "07_funciones_robustas.png"
    )

    animator = GraphAnimator(figsize=(19, 10.5), interval=520)
    animator.animate_robust_functions(
        result=resultado,
        states=estados,
        title="Funciones robustas: mínimos cuadrados frente a Huber",
        final_image_path=ruta_imagen,
        repeat=False,
    )


if __name__ == "__main__":
    main()
