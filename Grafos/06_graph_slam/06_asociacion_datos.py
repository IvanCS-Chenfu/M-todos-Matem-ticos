from itertools import combinations
from pathlib import Path
import sys

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment


CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


# ---------------------------------------------------------------------------
# Parámetros deterministas del ejemplo
# ---------------------------------------------------------------------------

POSE_REAL = np.array([-4.10, 0.05, np.deg2rad(6.0)], dtype=float)
POSE_ESTIMADA = np.array([-3.96, -0.08, np.deg2rad(7.8)], dtype=float)

LANDMARKS_REALES = {
    "l0": np.array([-1.20, 2.40], dtype=float),
    "l1": np.array([1.00, 4.20], dtype=float),
    "l2": np.array([3.50, 2.70], dtype=float),
    "l3": np.array([4.30, -0.40], dtype=float),
    "l4": np.array([1.60, -2.60], dtype=float),
    "l5": np.array([2.25, -2.25], dtype=float),
}

ASOCIACIONES_REALES = {
    "z0": "l0",
    "z1": "l1",
    "z2": "l2",
    "z3": "l3",
    "z4": "l4",
    "z5": "l5",
    "z6": None,
}

RUIDOS_RANGO_RUMBO = {
    "z0": np.array([0.050, np.deg2rad(0.60)], dtype=float),
    "z1": np.array([-0.060, np.deg2rad(-0.80)], dtype=float),
    "z2": np.array([0.080, np.deg2rad(0.50)], dtype=float),
    "z3": np.array([-0.040, np.deg2rad(1.00)], dtype=float),
    "z4": np.array([0.030, np.deg2rad(-0.70)], dtype=float),
    "z5": np.array([-0.050, np.deg2rad(0.90)], dtype=float),
}

MEDICION_NUEVA = np.array([5.10, np.deg2rad(-44.0)], dtype=float)

SIGMAS_MEDICION = np.array([0.12, np.deg2rad(1.60)], dtype=float)
SIGMAS_POSE = np.array([0.12, 0.12, np.deg2rad(2.00)], dtype=float)
SIGMAS_LANDMARK = np.array([0.10, 0.10], dtype=float)

UMBRAL_DESCRIPTOR = 0.72
NUMERO_CANDIDATOS_DESCRIPTOR = 3
UMBRAL_MAHALANOBIS = 9.21
COSTE_ASIGNACION_NULA = 1.00
UMBRAL_MARGEN_DUDOSO = 0.10
UMBRAL_RATIO_DUDOSO = 0.78

PESO_DESCRIPTOR = 0.48
PESO_GEOMETRIA = 0.52

UMBRAL_RANSAC = 0.26
MIN_INLIERS_RANSAC = 4
EPSILON_JACOBIANO = 1e-7
DELTA_HUBER = 2.5

LAMBDA_POSE = 2e-3
MAX_ITERACIONES_POSE = 15
TOLERANCIA_INCREMENTO = 1e-10


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


def validar_medicion(medicion, nombre="medición"):
    """Valida una observación rango-rumbo."""

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


def validar_descriptor(descriptor, dimension=8, nombre="descriptor"):
    """Valida y normaliza un descriptor de apariencia."""

    descriptor = np.asarray(descriptor, dtype=float)
    if descriptor.shape != (dimension,):
        raise ValueError(
            f"{nombre} debe tener dimensión {dimension}."
        )
    if not np.all(np.isfinite(descriptor)):
        raise ValueError(f"{nombre} debe contener valores finitos.")
    norma = float(np.linalg.norm(descriptor))
    if norma <= 1e-12:
        raise ValueError(f"{nombre} no puede tener norma cero.")
    return descriptor / norma


def rotacion_2d(theta):
    """Devuelve la matriz de rotación plana."""

    theta = normalizar_angulo(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def rango_rumbo_a_cartesiano(medicion):
    """Convierte rango-rumbo a coordenadas cartesianas locales."""

    rango, rumbo = validar_medicion(medicion)
    return np.array(
        [rango * np.cos(rumbo), rango * np.sin(rumbo)],
        dtype=float,
    )


def transformar_local_a_global(pose, punto_local):
    """Transforma un punto del marco del robot al marco global."""

    pose = validar_pose(pose)
    punto_local = validar_landmark(punto_local, "punto local")
    return pose[:2] + rotacion_2d(pose[2]) @ punto_local


def transformar_global_a_local(pose, punto_global):
    """Transforma un punto global al marco del robot."""

    pose = validar_pose(pose)
    punto_global = validar_landmark(punto_global, "punto global")
    return rotacion_2d(pose[2]).T @ (punto_global - pose[:2])


def predecir_observacion(pose, landmark):
    """Predice rango y rumbo de un landmark desde una pose."""

    local = transformar_global_a_local(pose, landmark)
    rango = float(np.linalg.norm(local))
    if rango <= 1e-12:
        raise ValueError("El rumbo no está definido para rango cero.")
    rumbo = normalizar_angulo(np.arctan2(local[1], local[0]))
    return np.array([rango, rumbo], dtype=float)


def calcular_innovacion(medicion, prediccion):
    """Calcula medición menos predicción con ángulo normalizado."""

    medicion = validar_medicion(medicion)
    prediccion = validar_medicion(prediccion, "predicción")
    innovacion = medicion - prediccion
    innovacion[1] = normalizar_angulo(innovacion[1])
    return innovacion


def crear_covarianza_diagonal(sigmas):
    """Construye una covarianza diagonal a partir de desviaciones típicas."""

    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.ndim != 1 or not np.all(np.isfinite(sigmas)):
        raise ValueError("Los sigmas deben formar un vector finito.")
    if np.any(sigmas <= 0.0):
        raise ValueError("Todos los sigmas deben ser positivos.")
    return np.diag(sigmas**2)


def calcular_matriz_informacion(covarianza):
    """Calcula la inversa de una covarianza definida positiva."""

    covarianza = np.asarray(covarianza, dtype=float)
    if covarianza.ndim != 2 or covarianza.shape[0] != covarianza.shape[1]:
        raise ValueError("La covarianza debe ser cuadrada.")
    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise ValueError("La covarianza debe ser simétrica.")
    if np.min(np.linalg.eigvalsh(covarianza)) <= 0.0:
        raise ValueError("La covarianza debe ser definida positiva.")
    return np.linalg.inv(covarianza)


# ---------------------------------------------------------------------------
# Escenario, observaciones y descriptores
# ---------------------------------------------------------------------------


def crear_landmarks_reales():
    """Devuelve una copia de los landmarks conocidos del mapa."""

    return {
        nombre: validar_landmark(posicion, nombre)
        for nombre, posicion in LANDMARKS_REALES.items()
    }


def crear_descriptores_landmarks():
    """Crea descriptores deterministas para los seis landmarks."""

    bases = {
        "l0": [1.00, 0.18, 0.08, 0.02, 0.04, 0.00, 0.10, 0.03],
        "l1": [0.08, 1.00, 0.12, 0.05, 0.04, 0.08, 0.02, 0.12],
        "l2": [0.05, 0.10, 1.00, 0.16, 0.05, 0.08, 0.11, 0.02],
        "l3": [0.10, 0.05, 0.12, 1.00, 0.09, 0.03, 0.04, 0.13],
        "l4": [0.12, 0.18, 0.04, 0.06, 1.00, 0.32, 0.10, 0.03],
        "l5": [0.08, 0.05, 0.18, 0.10, 0.48, 1.00, 0.20, 0.07],
    }
    return {
        nombre: validar_descriptor(vector, nombre=f"descriptor {nombre}")
        for nombre, vector in bases.items()
    }


def crear_observaciones_rango_rumbo(pose_real, landmarks):
    """Genera seis observaciones reales con ruido y una observación nueva."""

    pose_real = validar_pose(pose_real, "pose real")
    observaciones = {}

    for nombre_obs, nombre_landmark in ASOCIACIONES_REALES.items():
        if nombre_landmark is None:
            observaciones[nombre_obs] = validar_medicion(
                MEDICION_NUEVA,
                nombre_obs,
            )
            continue

        ideal = predecir_observacion(
            pose_real,
            landmarks[nombre_landmark],
        )
        medida = ideal + RUIDOS_RANGO_RUMBO[nombre_obs]
        medida[1] = normalizar_angulo(medida[1])
        observaciones[nombre_obs] = validar_medicion(medida, nombre_obs)

    return observaciones


def crear_descriptores_observaciones(descriptores_landmarks):
    """Crea observaciones visuales con ambigüedad y aliasing perceptual."""

    perturbaciones = {
        "z0": np.array([0.00, 0.02, -0.01, 0.01, 0.00, 0.01, -0.01, 0.00]),
        "z1": np.array([0.01, -0.01, 0.02, 0.00, 0.01, -0.01, 0.00, 0.01]),
        "z2": np.array([-0.01, 0.01, 0.00, 0.02, -0.01, 0.00, 0.01, -0.01]),
        "z3": np.array([0.02, 0.00, -0.01, 0.00, 0.01, 0.00, -0.01, 0.01]),
    }

    resultado = {}
    for nombre_obs, nombre_landmark in ASOCIACIONES_REALES.items():
        if nombre_obs in perturbaciones:
            resultado[nombre_obs] = validar_descriptor(
                descriptores_landmarks[nombre_landmark]
                + perturbaciones[nombre_obs],
                nombre=f"descriptor {nombre_obs}",
            )

    # z4 es deliberadamente ambiguo entre l4 y l5.
    resultado["z4"] = validar_descriptor(
        0.20 * descriptores_landmarks["l4"]
        + 0.80 * descriptores_landmarks["l5"],
        nombre="descriptor z4",
    )

    # z5 procede de l5, pero visualmente se parece mucho a l2.
    resultado["z5"] = validar_descriptor(
        0.86 * descriptores_landmarks["l2"]
        + 0.14 * descriptores_landmarks["l5"],
        nombre="descriptor z5",
    )

    # z6 no pertenece al mapa, aunque su apariencia recuerda a l3.
    resultado["z6"] = validar_descriptor(
        0.72 * descriptores_landmarks["l3"]
        + 0.20 * descriptores_landmarks["l1"]
        + np.array([0.05, 0.00, 0.03, 0.00, 0.04, 0.02, 0.00, 0.06]),
        nombre="descriptor z6",
    )

    return resultado


def calcular_distancia_descriptor(descriptor_a, descriptor_b):
    """Calcula distancia euclídea entre descriptores normalizados."""

    descriptor_a = validar_descriptor(descriptor_a)
    descriptor_b = validar_descriptor(descriptor_b)
    return float(np.linalg.norm(descriptor_a - descriptor_b))


def obtener_candidatos_descriptor(
    descriptor_observacion,
    descriptores_landmarks,
    top_k=NUMERO_CANDIDATOS_DESCRIPTOR,
    umbral=UMBRAL_DESCRIPTOR,
):
    """Recupera candidatos por distancia descriptiva y conserva el top-k."""

    if top_k <= 0:
        raise ValueError("top_k debe ser positivo.")
    distancias = []
    for nombre, descriptor in descriptores_landmarks.items():
        distancias.append(
            (
                calcular_distancia_descriptor(
                    descriptor_observacion,
                    descriptor,
                ),
                nombre,
            )
        )
    distancias.sort(key=lambda item: (item[0], item[1]))

    seleccionados = []
    for indice, (distancia, nombre) in enumerate(distancias):
        if indice < top_k or distancia <= umbral:
            seleccionados.append(
                {
                    "landmark": nombre,
                    "descriptor_distance": distancia,
                    "descriptor_rank": indice + 1,
                    "inside_descriptor_gate": distancia <= umbral,
                }
            )
    return seleccionados


# ---------------------------------------------------------------------------
# Compatibilidad geométrica y matriz de costes
# ---------------------------------------------------------------------------


def calcular_jacobianos_observacion_numericos(
    pose,
    landmark,
    epsilon=EPSILON_JACOBIANO,
):
    """Calcula jacobianos de la predicción por diferencias centrales."""

    pose = validar_pose(pose)
    landmark = validar_landmark(landmark)
    if epsilon <= 0.0:
        raise ValueError("epsilon debe ser positivo.")

    jac_pose = np.zeros((2, 3), dtype=float)
    jac_landmark = np.zeros((2, 2), dtype=float)

    for columna in range(3):
        delta = np.zeros(3, dtype=float)
        delta[columna] = epsilon
        pose_mas = pose + delta
        pose_menos = pose - delta
        pose_mas[2] = normalizar_angulo(pose_mas[2])
        pose_menos[2] = normalizar_angulo(pose_menos[2])
        mas = predecir_observacion(pose_mas, landmark)
        menos = predecir_observacion(pose_menos, landmark)
        diferencia = mas - menos
        diferencia[1] = normalizar_angulo(diferencia[1])
        jac_pose[:, columna] = diferencia / (2.0 * epsilon)

    for columna in range(2):
        delta = np.zeros(2, dtype=float)
        delta[columna] = epsilon
        mas = predecir_observacion(pose, landmark + delta)
        menos = predecir_observacion(pose, landmark - delta)
        diferencia = mas - menos
        diferencia[1] = normalizar_angulo(diferencia[1])
        jac_landmark[:, columna] = diferencia / (2.0 * epsilon)

    return jac_pose, jac_landmark


def calcular_covarianza_innovacion(
    pose,
    landmark,
    covarianza_pose,
    covarianza_landmark,
    covarianza_medicion,
):
    """Propaga la incertidumbre de pose, landmark y sensor."""

    jac_pose, jac_landmark = calcular_jacobianos_observacion_numericos(
        pose,
        landmark,
    )
    covarianza = (
        jac_pose @ covarianza_pose @ jac_pose.T
        + jac_landmark @ covarianza_landmark @ jac_landmark.T
        + covarianza_medicion
    )
    covarianza = 0.5 * (covarianza + covarianza.T)
    if np.min(np.linalg.eigvalsh(covarianza)) <= 0.0:
        raise ValueError("La covarianza de innovación no es positiva.")
    return covarianza, jac_pose, jac_landmark


def calcular_distancia_mahalanobis(innovacion, covarianza_innovacion):
    """Calcula la distancia de Mahalanobis cuadrática."""

    innovacion = np.asarray(innovacion, dtype=float)
    if innovacion.shape != (2,) or not np.all(np.isfinite(innovacion)):
        raise ValueError("La innovación debe ser un vector finito de dimensión dos.")
    informacion = calcular_matriz_informacion(covarianza_innovacion)
    return float(innovacion.T @ informacion @ innovacion)


def evaluar_par_observacion_landmark(
    nombre_observacion,
    nombre_landmark,
    medicion,
    pose_estimada,
    landmark,
    descriptor_observacion,
    descriptor_landmark,
    covarianza_pose,
    covarianza_landmark,
    covarianza_medicion,
):
    """Evalúa apariencia, innovación, Mahalanobis y coste combinado."""

    prediccion = predecir_observacion(pose_estimada, landmark)
    innovacion = calcular_innovacion(medicion, prediccion)
    cov_innovacion, jac_pose, jac_landmark = calcular_covarianza_innovacion(
        pose_estimada,
        landmark,
        covarianza_pose,
        covarianza_landmark,
        covarianza_medicion,
    )
    mahalanobis = calcular_distancia_mahalanobis(
        innovacion,
        cov_innovacion,
    )
    distancia_descriptor = calcular_distancia_descriptor(
        descriptor_observacion,
        descriptor_landmark,
    )

    dentro_descriptor = distancia_descriptor <= UMBRAL_DESCRIPTOR
    dentro_geometria = mahalanobis <= UMBRAL_MAHALANOBIS

    coste_descriptor = min(distancia_descriptor / UMBRAL_DESCRIPTOR, 2.5)
    coste_geometria = min(mahalanobis / UMBRAL_MAHALANOBIS, 3.0)
    coste_combinado = (
        PESO_DESCRIPTOR * coste_descriptor
        + PESO_GEOMETRIA * coste_geometria
    )

    return {
        "observation": nombre_observacion,
        "landmark": nombre_landmark,
        "measurement": validar_medicion(medicion),
        "prediction": prediccion,
        "innovation": innovacion,
        "innovation_covariance": cov_innovacion,
        "pose_jacobian": jac_pose,
        "landmark_jacobian": jac_landmark,
        "descriptor_distance": distancia_descriptor,
        "mahalanobis": mahalanobis,
        "inside_descriptor_gate": dentro_descriptor,
        "inside_geometry_gate": dentro_geometria,
        "eligible": dentro_geometria,
        "combined_cost": coste_combinado,
    }


def evaluar_todos_los_pares(
    observaciones,
    landmarks,
    pose_estimada,
    descriptores_observaciones,
    descriptores_landmarks,
):
    """Evalúa todas las combinaciones observación-landmark."""

    cov_pose = crear_covarianza_diagonal(SIGMAS_POSE)
    cov_landmark = crear_covarianza_diagonal(SIGMAS_LANDMARK)
    cov_medicion = crear_covarianza_diagonal(SIGMAS_MEDICION)

    evaluaciones = {}
    for nombre_obs, medicion in observaciones.items():
        evaluaciones[nombre_obs] = {}
        for nombre_lm, landmark in landmarks.items():
            evaluaciones[nombre_obs][nombre_lm] = (
                evaluar_par_observacion_landmark(
                    nombre_obs,
                    nombre_lm,
                    medicion,
                    pose_estimada,
                    landmark,
                    descriptores_observaciones[nombre_obs],
                    descriptores_landmarks[nombre_lm],
                    cov_pose,
                    cov_landmark,
                    cov_medicion,
                )
            )
    return evaluaciones


def crear_lista_candidatos(
    evaluaciones,
    descriptores_observaciones,
    descriptores_landmarks,
):
    """Combina recuperación visual y evaluación geométrica."""

    candidatos = []
    for nombre_obs in sorted(evaluaciones, key=lambda x: int(x[1:])):
        recuperados = obtener_candidatos_descriptor(
            descriptores_observaciones[nombre_obs],
            descriptores_landmarks,
        )
        recuperados_por_nombre = {
            item["landmark"]: item
            for item in recuperados
        }
        for nombre_lm, evaluacion in evaluaciones[nombre_obs].items():
            if nombre_lm not in recuperados_por_nombre:
                continue
            registro = dict(evaluacion)
            registro.update(recuperados_por_nombre[nombre_lm])
            registro["false_visual_alias"] = (
                nombre_obs == "z5" and nombre_lm == "l2"
            )
            candidatos.append(registro)

    candidatos.sort(
        key=lambda item: (
            int(item["observation"][1:]),
            item["descriptor_rank"],
            item["landmark"],
        )
    )
    return candidatos


def crear_matriz_costes(evaluaciones):
    """Crea la matriz de costes geométrico-descriptivos."""

    nombres_obs = sorted(evaluaciones, key=lambda x: int(x[1:]))
    nombres_lm = sorted(
        next(iter(evaluaciones.values())),
        key=lambda x: int(x[1:]),
    )
    matriz = np.full(
        (len(nombres_obs), len(nombres_lm)),
        np.inf,
        dtype=float,
    )

    for fila, nombre_obs in enumerate(nombres_obs):
        for columna, nombre_lm in enumerate(nombres_lm):
            evaluacion = evaluaciones[nombre_obs][nombre_lm]
            if evaluacion["inside_geometry_gate"]:
                matriz[fila, columna] = evaluacion["combined_cost"]

    return nombres_obs, nombres_lm, matriz


def seleccionar_vecino_mas_cercano_independiente(
    nombres_obs,
    nombres_lm,
    matriz_costes,
):
    """Selecciona el mejor candidato de cada fila sin evitar duplicidades."""

    asociaciones = {}
    for fila, nombre_obs in enumerate(nombres_obs):
        costes = matriz_costes[fila]
        indices_validos = np.flatnonzero(np.isfinite(costes))
        if indices_validos.size == 0:
            asociaciones[nombre_obs] = None
            continue
        indice = int(indices_validos[np.argmin(costes[indices_validos])])
        if costes[indice] >= COSTE_ASIGNACION_NULA:
            asociaciones[nombre_obs] = None
        else:
            asociaciones[nombre_obs] = nombres_lm[indice]
    return asociaciones


def contar_duplicidades(asociaciones):
    """Cuenta landmarks asignados a más de una observación."""

    usados = [valor for valor in asociaciones.values() if valor is not None]
    return len(usados) - len(set(usados))


def resolver_asignacion_global(
    nombres_obs,
    nombres_lm,
    matriz_costes,
):
    """Resuelve el matching uno-a-uno con asignaciones nulas."""

    numero_obs = len(nombres_obs)
    numero_lm = len(nombres_lm)
    coste_grande = 1e6

    real = np.where(np.isfinite(matriz_costes), matriz_costes, coste_grande)
    dummies = np.full(
        (numero_obs, numero_obs),
        COSTE_ASIGNACION_NULA,
        dtype=float,
    )
    for fila in range(numero_obs):
        dummies[fila] += np.linspace(0.0, 1e-5, numero_obs)

    ampliada = np.hstack([real, dummies])
    filas, columnas = linear_sum_assignment(ampliada)

    asociaciones = {nombre: None for nombre in nombres_obs}
    coste_total = 0.0
    asignaciones = []

    for fila, columna in zip(filas, columnas):
        nombre_obs = nombres_obs[fila]
        coste = float(ampliada[fila, columna])
        if columna < numero_lm and coste < COSTE_ASIGNACION_NULA:
            nombre_lm = nombres_lm[columna]
            asociaciones[nombre_obs] = nombre_lm
            tipo = "landmark"
        else:
            nombre_lm = None
            tipo = "null"
        coste_total += coste
        asignaciones.append(
            {
                "observation": nombre_obs,
                "landmark": nombre_lm,
                "column": int(columna),
                "cost": coste,
                "type": tipo,
            }
        )

    return {
        "associations": asociaciones,
        "assignments": asignaciones,
        "augmented_cost_matrix": ampliada,
        "total_cost": coste_total,
    }


def calcular_confianza_asociacion(
    nombre_obs,
    nombre_lm,
    nombres_lm,
    matriz_costes,
):
    """Calcula mejor coste, segundo coste, margen y ratio."""

    fila = int(nombre_obs[1:])
    costes = matriz_costes[fila]
    validos = sorted(
        (
            float(coste),
            nombres_lm[indice],
        )
        for indice, coste in enumerate(costes)
        if np.isfinite(coste)
    )
    coste_elegido = float(costes[nombres_lm.index(nombre_lm)])
    otros = [item for item in validos if item[1] != nombre_lm]
    if otros:
        segundo_coste, segundo_landmark = otros[0]
        margen = segundo_coste - coste_elegido
        ratio = coste_elegido / max(segundo_coste, 1e-12)
    else:
        segundo_coste = np.inf
        segundo_landmark = None
        margen = np.inf
        ratio = 0.0

    return {
        "best_cost": coste_elegido,
        "second_cost": segundo_coste,
        "second_landmark": segundo_landmark,
        "margin": margen,
        "ratio": ratio,
    }


def clasificar_asignaciones_finales(
    resultado_global,
    nombres_lm,
    matriz_costes,
):
    """Clasifica asignaciones en correctas, dudosas, falsas o nuevas."""

    decisiones = {}
    for nombre_obs, nombre_lm in resultado_global["associations"].items():
        verdadero = ASOCIACIONES_REALES[nombre_obs]
        if nombre_lm is None:
            decisiones[nombre_obs] = {
                "observation": nombre_obs,
                "landmark": None,
                "truth": verdadero,
                "status": "new" if verdadero is None else "rejected",
                "accepted": False,
                "correct": verdadero is None,
                "confidence": None,
            }
            continue

        confianza = calcular_confianza_asociacion(
            nombre_obs,
            nombre_lm,
            nombres_lm,
            matriz_costes,
        )
        es_correcta = nombre_lm == verdadero
        es_dudosa = (
            es_correcta
            and (
                confianza["margin"] < UMBRAL_MARGEN_DUDOSO
                or confianza["ratio"] > UMBRAL_RATIO_DUDOSO
            )
        )

        if not es_correcta:
            estado = "false"
            aceptada = False
        elif es_dudosa:
            estado = "doubtful"
            aceptada = False
        else:
            estado = "correct"
            aceptada = True

        decisiones[nombre_obs] = {
            "observation": nombre_obs,
            "landmark": nombre_lm,
            "truth": verdadero,
            "status": estado,
            "accepted": aceptada,
            "correct": es_correcta,
            "confidence": confianza,
        }

    return decisiones


# ---------------------------------------------------------------------------
# Verificación geométrica conjunta mediante RANSAC
# ---------------------------------------------------------------------------


def estimar_transformacion_rigida_2d(puntos_origen, puntos_destino):
    """Estima una transformación rígida 2D mediante SVD."""

    origen = np.asarray(puntos_origen, dtype=float)
    destino = np.asarray(puntos_destino, dtype=float)
    if origen.shape != destino.shape or origen.ndim != 2 or origen.shape[1] != 2:
        raise ValueError("Los conjuntos deben tener forma Nx2 y coincidir.")
    if origen.shape[0] < 2:
        raise ValueError("Se necesitan al menos dos correspondencias.")

    centro_origen = np.mean(origen, axis=0)
    centro_destino = np.mean(destino, axis=0)
    origen_centrado = origen - centro_origen
    destino_centrado = destino - centro_destino
    matriz = origen_centrado.T @ destino_centrado
    u, _, vt = np.linalg.svd(matriz)
    rotacion = vt.T @ u.T
    if np.linalg.det(rotacion) < 0.0:
        vt[-1, :] *= -1.0
        rotacion = vt.T @ u.T
    traslacion = centro_destino - rotacion @ centro_origen
    theta = normalizar_angulo(np.arctan2(rotacion[1, 0], rotacion[0, 0]))
    return np.array([traslacion[0], traslacion[1], theta], dtype=float)


def aplicar_transformacion_rigida_2d(pose, puntos):
    """Aplica una pose de SE(2) a un conjunto de puntos 2D."""

    pose = validar_pose(pose)
    puntos = np.asarray(puntos, dtype=float)
    if puntos.ndim != 2 or puntos.shape[1] != 2:
        raise ValueError("Los puntos deben tener forma Nx2.")
    return (rotacion_2d(pose[2]) @ puntos.T).T + pose[:2]


def crear_correspondencias_ransac(observaciones, landmarks):
    """Crea cinco correspondencias correctas y dos outliers deliberados."""

    hipotesis = [
        ("z0", "l0"),
        ("z1", "l1"),
        ("z2", "l2"),
        ("z3", "l3"),
        ("z4", "l4"),
        ("z5", "l2"),
        ("z6", "l5"),
    ]
    origen = np.vstack(
        [rango_rumbo_a_cartesiano(observaciones[z]) for z, _ in hipotesis]
    )
    destino = np.vstack([landmarks[l] for _, l in hipotesis])
    return {
        "pairs": hipotesis,
        "source_points": origen,
        "target_points": destino,
    }


def verificar_correspondencias_ransac(correspondencias):
    """Ejecuta RANSAC determinista probando todos los pares mínimos."""

    origen = np.asarray(correspondencias["source_points"], dtype=float)
    destino = np.asarray(correspondencias["target_points"], dtype=float)
    mejor = None
    historial = []

    for indice_hipotesis, indices in enumerate(combinations(range(len(origen)), 2)):
        indices = np.asarray(indices, dtype=int)
        pose = estimar_transformacion_rigida_2d(origen[indices], destino[indices])
        transformados = aplicar_transformacion_rigida_2d(pose, origen)
        errores = np.linalg.norm(transformados - destino, axis=1)
        inliers = errores <= UMBRAL_RANSAC
        numero_inliers = int(np.sum(inliers))
        rmse = (
            float(np.sqrt(np.mean(errores[inliers] ** 2)))
            if numero_inliers > 0
            else np.inf
        )
        registro = {
            "hypothesis": indice_hipotesis,
            "sample": indices.tolist(),
            "pose": pose,
            "errors": errores,
            "inliers": inliers,
            "inlier_count": numero_inliers,
            "rmse": rmse,
        }
        historial.append(registro)
        clave = (numero_inliers, -rmse)
        if mejor is None or clave > mejor[0]:
            mejor = (clave, registro)

    if mejor is None:
        raise RuntimeError("RANSAC no generó hipótesis.")

    inliers = mejor[1]["inliers"]
    pose_refinada = estimar_transformacion_rigida_2d(
        origen[inliers],
        destino[inliers],
    )
    transformados = aplicar_transformacion_rigida_2d(pose_refinada, origen)
    errores = np.linalg.norm(transformados - destino, axis=1)
    inliers = errores <= UMBRAL_RANSAC
    numero_inliers = int(np.sum(inliers))
    rmse = float(np.sqrt(np.mean(errores[inliers] ** 2)))

    return {
        "pose": pose_refinada,
        "errors": errores,
        "inliers": inliers,
        "outliers": ~inliers,
        "inlier_count": numero_inliers,
        "outlier_count": int(len(inliers) - numero_inliers),
        "inlier_ratio": numero_inliers / len(inliers),
        "rmse": rmse,
        "accepted": numero_inliers >= MIN_INLIERS_RANSAC,
        "history": historial,
    }


# ---------------------------------------------------------------------------
# Efecto hipotético de aceptar una asociación falsa
# ---------------------------------------------------------------------------


def calcular_peso_huber(mahalanobis, delta=DELTA_HUBER):
    """Calcula el peso IRLS del kernel de Huber."""

    mahalanobis = float(mahalanobis)
    if mahalanobis < 0.0 or not np.isfinite(mahalanobis):
        raise ValueError("Mahalanobis debe ser finito y no negativo.")
    norma = np.sqrt(mahalanobis)
    if norma <= delta or norma <= 1e-15:
        return 1.0
    return float(delta / norma)


def aplicar_incremento_pose(pose, incremento):
    """Aplica un incremento aditivo local a una pose 2D."""

    pose = validar_pose(pose)
    incremento = np.asarray(incremento, dtype=float)
    if incremento.shape != (3,) or not np.all(np.isfinite(incremento)):
        raise ValueError("El incremento debe tener tres componentes finitas.")
    resultado = pose + incremento
    resultado[2] = normalizar_angulo(resultado[2])
    return resultado


def ensamblar_sistema_pose(
    pose,
    pares,
    observaciones,
    landmarks,
    robusto=False,
):
    """Ensambla Hessiana, gradiente y coste para una pose y mapa fijo."""

    pose = validar_pose(pose)
    informacion = calcular_matriz_informacion(
        crear_covarianza_diagonal(SIGMAS_MEDICION)
    )
    prior_info = calcular_matriz_informacion(
        crear_covarianza_diagonal([0.45, 0.45, np.deg2rad(10.0)])
    )

    hessiana = prior_info.copy()
    gradiente = prior_info @ (pose - POSE_ESTIMADA)
    coste = 0.5 * float((pose - POSE_ESTIMADA).T @ prior_info @ (pose - POSE_ESTIMADA))
    factores = []

    for nombre_obs, nombre_lm in pares:
        prediccion = predecir_observacion(pose, landmarks[nombre_lm])
        residual = prediccion - observaciones[nombre_obs]
        residual[1] = normalizar_angulo(residual[1])
        jac_pose, _ = calcular_jacobianos_observacion_numericos(
            pose,
            landmarks[nombre_lm],
        )
        mahalanobis = float(residual.T @ informacion @ residual)
        peso = calcular_peso_huber(mahalanobis) if robusto else 1.0
        omega = peso * informacion
        hessiana += jac_pose.T @ omega @ jac_pose
        gradiente += jac_pose.T @ omega @ residual
        coste += 0.5 * peso * mahalanobis
        factores.append(
            {
                "observation": nombre_obs,
                "landmark": nombre_lm,
                "residual": residual,
                "mahalanobis": mahalanobis,
                "weight": peso,
            }
        )

    return {
        "hessian": 0.5 * (hessiana + hessiana.T),
        "gradient": gradiente,
        "cost": coste,
        "factors": factores,
    }


def optimizar_pose_con_asociaciones(
    pose_inicial,
    pares,
    observaciones,
    landmarks,
    robusto=False,
):
    """Optimiza la pose con landmarks fijos para mostrar el efecto de un falso match."""

    pose = validar_pose(pose_inicial)
    damping = LAMBDA_POSE
    historial = []

    for iteracion in range(MAX_ITERACIONES_POSE):
        sistema = ensamblar_sistema_pose(
            pose,
            pares,
            observaciones,
            landmarks,
            robusto=robusto,
        )
        diagonal = np.maximum(np.diag(sistema["hessian"]), 1e-9)
        matriz = sistema["hessian"] + damping * np.diag(diagonal)
        incremento = np.linalg.solve(matriz, -sistema["gradient"])
        candidata = aplicar_incremento_pose(pose, incremento)
        sistema_candidato = ensamblar_sistema_pose(
            candidata,
            pares,
            observaciones,
            landmarks,
            robusto=robusto,
        )
        aceptada = sistema_candidato["cost"] < sistema["cost"]
        if aceptada:
            pose = candidata
            damping = max(damping * 0.35, 1e-9)
        else:
            damping = min(damping * 8.0, 1e8)

        historial.append(
            {
                "iteration": iteracion,
                "cost_before": sistema["cost"],
                "cost_after": sistema_candidato["cost"] if aceptada else sistema["cost"],
                "step_norm": float(np.linalg.norm(incremento)),
                "damping": damping,
                "accepted": aceptada,
                "pose": pose.copy(),
            }
        )
        if aceptada and np.linalg.norm(incremento) < TOLERANCIA_INCREMENTO:
            break

    final = ensamblar_sistema_pose(
        pose,
        pares,
        observaciones,
        landmarks,
        robusto=robusto,
    )
    return {
        "pose": pose,
        "cost": final["cost"],
        "history": historial,
        "iterations": len(historial),
        "factors": final["factors"],
    }


def analizar_efecto_asociacion_falsa(
    decisiones,
    observaciones,
    landmarks,
):
    """Compara la pose usando asociaciones correctas y añadiendo z5-l2."""

    pares_correctos = [
        (nombre_obs, decision["landmark"])
        for nombre_obs, decision in decisiones.items()
        if decision["accepted"]
    ]
    par_falso = ("z5", "l2")

    solo_correctas = optimizar_pose_con_asociaciones(
        POSE_ESTIMADA,
        pares_correctos,
        observaciones,
        landmarks,
        robusto=False,
    )
    con_falsa = optimizar_pose_con_asociaciones(
        POSE_ESTIMADA,
        pares_correctos + [par_falso],
        observaciones,
        landmarks,
        robusto=False,
    )
    con_falsa_robusta = optimizar_pose_con_asociaciones(
        POSE_ESTIMADA,
        pares_correctos + [par_falso],
        observaciones,
        landmarks,
        robusto=True,
    )

    evaluacion_falsa = ensamblar_sistema_pose(
        POSE_ESTIMADA,
        [par_falso],
        observaciones,
        landmarks,
        robusto=False,
    )["factors"][0]

    return {
        "correct_only": solo_correctas,
        "with_false": con_falsa,
        "with_false_robust": con_falsa_robusta,
        "false_factor": evaluacion_falsa,
        "false_pair": par_falso,
        "correct_pairs": pares_correctos,
        "translation_shift_false": float(
            np.linalg.norm(con_falsa["pose"][:2] - solo_correctas["pose"][:2])
        ),
        "angle_shift_false_deg": float(
            np.rad2deg(
                normalizar_angulo(con_falsa["pose"][2] - solo_correctas["pose"][2])
            )
        ),
        "translation_shift_robust": float(
            np.linalg.norm(
                con_falsa_robusta["pose"][:2] - solo_correctas["pose"][:2]
            )
        ),
        "angle_shift_robust_deg": float(
            np.rad2deg(
                normalizar_angulo(
                    con_falsa_robusta["pose"][2] - solo_correctas["pose"][2]
                )
            )
        ),
    }


# ---------------------------------------------------------------------------
# Grafos y métricas
# ---------------------------------------------------------------------------


def crear_grafo_candidatos(candidatos, observaciones, landmarks):
    """Construye un grafo bipartito observación-landmark."""

    grafo = nx.Graph()
    for nombre_obs in observaciones:
        grafo.add_node(nombre_obs, bipartite="observation", node_type="observation")
    for nombre_lm, posicion in landmarks.items():
        grafo.add_node(
            nombre_lm,
            bipartite="landmark",
            node_type="landmark",
            position=posicion.copy(),
        )
    for candidato in candidatos:
        grafo.add_edge(
            candidato["observation"],
            candidato["landmark"],
            descriptor_distance=candidato["descriptor_distance"],
            mahalanobis=candidato["mahalanobis"],
            combined_cost=candidato["combined_cost"],
            inside_geometry_gate=candidato["inside_geometry_gate"],
            false_visual_alias=candidato["false_visual_alias"],
        )
    return grafo


def crear_grafo_factores(decisiones, landmarks):
    """Crea factores pose-landmark solo para asociaciones aceptadas."""

    grafo = nx.Graph()
    grafo.add_node("x0", node_type="pose", estimate=POSE_ESTIMADA.copy())
    for nombre_lm, posicion in landmarks.items():
        grafo.add_node(
            nombre_lm,
            node_type="landmark",
            fixed=True,
            estimate=posicion.copy(),
        )
    for nombre_obs, decision in decisiones.items():
        if not decision["accepted"]:
            continue
        grafo.add_edge(
            "x0",
            decision["landmark"],
            factor_type="landmark_observation",
            observation=nombre_obs,
        )
    return grafo


def calcular_metricas_asociacion(decisiones, numero_landmarks):
    """Calcula matriz de confusión por pares y precision, recall y F1."""

    verdaderos_positivos = 0
    falsos_positivos = 0
    falsos_negativos = 0

    for nombre_obs, verdadero in ASOCIACIONES_REALES.items():
        decision = decisiones[nombre_obs]
        if verdadero is None:
            if decision["accepted"]:
                falsos_positivos += 1
            continue
        if decision["accepted"] and decision["landmark"] == verdadero:
            verdaderos_positivos += 1
        elif decision["accepted"]:
            falsos_positivos += 1
            falsos_negativos += 1
        else:
            falsos_negativos += 1

    total_pares = len(ASOCIACIONES_REALES) * numero_landmarks
    verdaderos_negativos = (
        total_pares
        - verdaderos_positivos
        - falsos_positivos
        - falsos_negativos
    )
    precision = (
        verdaderos_positivos / (verdaderos_positivos + falsos_positivos)
        if verdaderos_positivos + falsos_positivos > 0
        else 0.0
    )
    recall = (
        verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
        if verdaderos_positivos + falsos_negativos > 0
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )

    return {
        "true_positives": verdaderos_positivos,
        "false_positives": falsos_positivos,
        "false_negatives": falsos_negativos,
        "true_negatives": verdaderos_negativos,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calcular_resultados_metodos_simples(
    observaciones,
    descriptores_observaciones,
    descriptores_landmarks,
    asociaciones_independientes,
):
    """Evalúa descriptor puro y vecino geométrico independiente."""

    descriptor_only = {}
    for nombre_obs in observaciones:
        candidatos = obtener_candidatos_descriptor(
            descriptores_observaciones[nombre_obs],
            descriptores_landmarks,
            top_k=1,
            umbral=2.0,
        )
        descriptor_only[nombre_obs] = candidatos[0]["landmark"]

    def exactitud(mapeo):
        correctas = sum(
            mapeo.get(nombre_obs) == verdadero
            for nombre_obs, verdadero in ASOCIACIONES_REALES.items()
            if verdadero is not None
        )
        return correctas / sum(valor is not None for valor in ASOCIACIONES_REALES.values())

    return {
        "descriptor_only": descriptor_only,
        "descriptor_accuracy": exactitud(descriptor_only),
        "descriptor_duplicates": contar_duplicidades(descriptor_only),
        "independent": asociaciones_independientes,
        "independent_accuracy": exactitud(asociaciones_independientes),
        "independent_duplicates": contar_duplicidades(asociaciones_independientes),
    }


# ---------------------------------------------------------------------------
# Construcción del resultado completo
# ---------------------------------------------------------------------------


def crear_resultado_asociacion_datos():
    """Ejecuta toda la simulación de asociación de datos."""

    landmarks = crear_landmarks_reales()
    descriptores_lm = crear_descriptores_landmarks()
    observaciones = crear_observaciones_rango_rumbo(POSE_REAL, landmarks)
    descriptores_obs = crear_descriptores_observaciones(descriptores_lm)
    evaluaciones = evaluar_todos_los_pares(
        observaciones,
        landmarks,
        POSE_ESTIMADA,
        descriptores_obs,
        descriptores_lm,
    )
    candidatos = crear_lista_candidatos(
        evaluaciones,
        descriptores_obs,
        descriptores_lm,
    )
    nombres_obs, nombres_lm, matriz_costes = crear_matriz_costes(evaluaciones)
    independientes = seleccionar_vecino_mas_cercano_independiente(
        nombres_obs,
        nombres_lm,
        matriz_costes,
    )
    global_result = resolver_asignacion_global(
        nombres_obs,
        nombres_lm,
        matriz_costes,
    )
    decisiones = clasificar_asignaciones_finales(
        global_result,
        nombres_lm,
        matriz_costes,
    )

    correspondencias_ransac = crear_correspondencias_ransac(
        observaciones,
        landmarks,
    )
    ransac = verificar_correspondencias_ransac(correspondencias_ransac)
    efecto_falso = analizar_efecto_asociacion_falsa(
        decisiones,
        observaciones,
        landmarks,
    )
    metricas = calcular_metricas_asociacion(decisiones, len(landmarks))
    metodos = calcular_resultados_metodos_simples(
        observaciones,
        descriptores_obs,
        descriptores_lm,
        independientes,
    )

    puntos_observados_globales_reales = {
        nombre_obs: transformar_local_a_global(
            POSE_REAL,
            rango_rumbo_a_cartesiano(medicion),
        )
        for nombre_obs, medicion in observaciones.items()
    }
    puntos_observados_globales_estimados = {
        nombre_obs: transformar_local_a_global(
            POSE_ESTIMADA,
            rango_rumbo_a_cartesiano(medicion),
        )
        for nombre_obs, medicion in observaciones.items()
    }

    alias_falso = next(
        item
        for item in candidatos
        if item["false_visual_alias"]
    )

    resultado = {
        "true_pose": POSE_REAL.copy(),
        "estimated_pose": POSE_ESTIMADA.copy(),
        "landmarks": landmarks,
        "observations": observaciones,
        "true_associations": dict(ASOCIACIONES_REALES),
        "landmark_descriptors": descriptores_lm,
        "observation_descriptors": descriptores_obs,
        "evaluations": evaluaciones,
        "candidates": candidatos,
        "observation_names": nombres_obs,
        "landmark_names": nombres_lm,
        "cost_matrix": matriz_costes,
        "independent_associations": independientes,
        "global_assignment": global_result,
        "decisions": decisiones,
        "candidate_graph": crear_grafo_candidatos(
            candidatos,
            observaciones,
            landmarks,
        ),
        "factor_graph": crear_grafo_factores(decisiones, landmarks),
        "ransac_correspondences": correspondencias_ransac,
        "ransac": ransac,
        "false_effect": efecto_falso,
        "metrics": metricas,
        "method_comparison": metodos,
        "false_visual_alias": alias_falso,
        "observed_global_true": puntos_observados_globales_reales,
        "observed_global_estimated": puntos_observados_globales_estimados,
        "measurement_covariance": crear_covarianza_diagonal(SIGMAS_MEDICION),
        "pose_covariance": crear_covarianza_diagonal(SIGMAS_POSE),
        "landmark_covariance": crear_covarianza_diagonal(SIGMAS_LANDMARK),
    }
    return resultado


# ---------------------------------------------------------------------------
# Estados didácticos de la animación
# ---------------------------------------------------------------------------


def crear_estado_animacion(phase, message, **kwargs):
    """Crea un estado visual con valores por defecto estables."""

    estado = {
        "phase": phase,
        "message": message,
        "visible_landmarks": 0,
        "visible_observations": 0,
        "visible_candidates": 0,
        "selected_observation": None,
        "show_gates": False,
        "show_cost_matrix": False,
        "matrix_rows": 0,
        "show_independent": False,
        "show_global": False,
        "show_decisions": False,
        "show_false_alias": False,
        "show_ransac": False,
        "ransac_hypotheses": 0,
        "show_false_effect": False,
        "show_metrics": False,
        "focus": phase,
    }
    estado.update(kwargs)
    return estado


def crear_estados_animacion(result):
    """Crea una secuencia completa de propuesta, matching y verificación."""

    estados = [
        crear_estado_animacion(
            "intro",
            "Asociar significa decidir qué landmark produjo cada observación.",
        )
    ]

    for cantidad in range(1, len(result["landmarks"]) + 1):
        estados.append(
            crear_estado_animacion(
                "landmarks",
                f"Se incorpora el landmark l{cantidad - 1} al mapa conocido.",
                visible_landmarks=cantidad,
            )
        )
        estados.append(
            crear_estado_animacion(
                "landmarks",
                "El mapa almacena posición, descriptor e incertidumbre.",
                visible_landmarks=cantidad,
            )
        )

    for cantidad in range(1, len(result["observations"]) + 1):
        nombre_obs = f"z{cantidad - 1}"
        verdadero = result["true_associations"][nombre_obs]
        texto = (
            f"{nombre_obs} procede de {verdadero} y contiene ruido."
            if verdadero is not None
            else f"{nombre_obs} no pertenece a ningún landmark del mapa."
        )
        estados.append(
            crear_estado_animacion(
                "observations",
                texto,
                visible_landmarks=len(result["landmarks"]),
                visible_observations=cantidad,
                selected_observation=nombre_obs,
            )
        )
        estados.append(
            crear_estado_animacion(
                "observations",
                "Detección no implica identidad: aún falta asociar.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=cantidad,
                selected_observation=nombre_obs,
            )
        )

    candidatos = result["candidates"]
    acumulados = 0
    for nombre_obs in result["observation_names"]:
        candidatos_obs = [
            item for item in candidatos if item["observation"] == nombre_obs
        ]
        for _ in candidatos_obs:
            acumulados += 1
            estados.append(
                crear_estado_animacion(
                    "descriptor_candidates",
                    f"El descriptor de {nombre_obs} propone candidatos por apariencia.",
                    visible_landmarks=len(result["landmarks"]),
                    visible_observations=len(result["observations"]),
                    visible_candidates=acumulados,
                    selected_observation=nombre_obs,
                )
            )

    for nombre_obs in result["observation_names"]:
        estados.append(
            crear_estado_animacion(
                "gating",
                f"Se compara la innovación de {nombre_obs} con su covarianza.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=len(result["observations"]),
                visible_candidates=len(candidatos),
                selected_observation=nombre_obs,
                show_gates=True,
            )
        )
        estados.append(
            crear_estado_animacion(
                "gating",
                "Los candidatos fuera del gate de Mahalanobis se rechazan.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=len(result["observations"]),
                visible_candidates=len(candidatos),
                selected_observation=nombre_obs,
                show_gates=True,
            )
        )

    for filas in range(1, len(result["observation_names"]) + 1):
        estados.append(
            crear_estado_animacion(
                "cost_matrix",
                "La matriz reúne costes de apariencia y geometría.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=len(result["observations"]),
                visible_candidates=len(candidatos),
                show_gates=True,
                show_cost_matrix=True,
                matrix_rows=filas,
            )
        )

    for nombre_obs in result["observation_names"]:
        estados.append(
            crear_estado_animacion(
                "independent_matching",
                "El vecino independiente puede reutilizar el mismo landmark.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=len(result["observations"]),
                visible_candidates=len(candidatos),
                show_gates=True,
                show_cost_matrix=True,
                matrix_rows=len(result["observation_names"]),
                show_independent=True,
                selected_observation=nombre_obs,
            )
        )

    for nombre_obs in result["observation_names"]:
        estados.append(
            crear_estado_animacion(
                "global_matching",
                "El matching global impone una asignación uno-a-uno.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=len(result["observations"]),
                visible_candidates=len(candidatos),
                show_gates=True,
                show_cost_matrix=True,
                matrix_rows=len(result["observation_names"]),
                show_global=True,
                selected_observation=nombre_obs,
            )
        )

    for _ in range(6):
        estados.append(
            crear_estado_animacion(
                "false_alias",
                "z5 se parece a l2, pero la geometría demuestra que es un falso match.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=len(result["observations"]),
                visible_candidates=len(candidatos),
                show_gates=True,
                show_cost_matrix=True,
                matrix_rows=len(result["observation_names"]),
                show_global=True,
                show_false_alias=True,
                selected_observation="z5",
            )
        )

    numero_hipotesis = len(result["ransac"]["history"])
    pasos_ransac = np.linspace(1, numero_hipotesis, 10, dtype=int)
    for numero in pasos_ransac:
        estados.append(
            crear_estado_animacion(
                "ransac",
                "RANSAC busca una transformación respaldada por varias correspondencias.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=len(result["observations"]),
                visible_candidates=len(candidatos),
                show_gates=True,
                show_global=True,
                show_false_alias=True,
                show_ransac=True,
                ransac_hypotheses=int(numero),
            )
        )

    for nombre_obs in result["observation_names"]:
        estados.append(
            crear_estado_animacion(
                "decisions",
                "Se clasifican asociaciones correctas, dudosas, rechazadas y nuevas.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=len(result["observations"]),
                visible_candidates=len(candidatos),
                show_gates=True,
                show_cost_matrix=True,
                matrix_rows=len(result["observation_names"]),
                show_global=True,
                show_decisions=True,
                show_ransac=True,
                ransac_hypotheses=numero_hipotesis,
                selected_observation=nombre_obs,
            )
        )

    for _ in range(6):
        estados.append(
            crear_estado_animacion(
                "false_effect",
                "Aceptar z5-l2 desplazaría la pose y deformaría los factores correctos.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=len(result["observations"]),
                visible_candidates=len(candidatos),
                show_global=True,
                show_decisions=True,
                show_ransac=True,
                ransac_hypotheses=numero_hipotesis,
                show_false_alias=True,
                show_false_effect=True,
            )
        )

    for _ in range(5):
        estados.append(
            crear_estado_animacion(
                "metrics",
                "Precision y recall resumen la calidad de las asociaciones aceptadas.",
                visible_landmarks=len(result["landmarks"]),
                visible_observations=len(result["observations"]),
                visible_candidates=len(candidatos),
                show_global=True,
                show_decisions=True,
                show_ransac=True,
                ransac_hypotheses=numero_hipotesis,
                show_metrics=True,
            )
        )

    estados.append(
        crear_estado_animacion(
            "summary",
            "El descriptor propone, el gate filtra, el matching resuelve y la geometría verifica.",
            visible_landmarks=len(result["landmarks"]),
            visible_observations=len(result["observations"]),
            visible_candidates=len(candidatos),
            show_gates=True,
            show_cost_matrix=True,
            matrix_rows=len(result["observation_names"]),
            show_global=True,
            show_decisions=True,
            show_false_alias=True,
            show_ransac=True,
            ransac_hypotheses=numero_hipotesis,
            show_false_effect=True,
            show_metrics=True,
        )
    )
    return estados


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------


def validar_escenario(result):
    """Comprueba dimensiones, identidades y número de observaciones."""

    validar_pose(result["true_pose"], "pose real")
    validar_pose(result["estimated_pose"], "pose estimada")
    if len(result["landmarks"]) != 6:
        raise ValueError("El ejemplo debe contener seis landmarks.")
    if len(result["observations"]) != 7:
        raise ValueError("El ejemplo debe contener siete observaciones.")
    for nombre, landmark in result["landmarks"].items():
        validar_landmark(landmark, nombre)
    for nombre, medicion in result["observations"].items():
        validar_medicion(medicion, nombre)


def validar_covarianzas(result):
    """Comprueba covarianzas y gates geométricos."""

    for clave in (
        "measurement_covariance",
        "pose_covariance",
        "landmark_covariance",
    ):
        matriz = result[clave]
        if not np.allclose(matriz, matriz.T, atol=1e-12):
            raise ValueError(f"{clave} debe ser simétrica.")
        if np.min(np.linalg.eigvalsh(matriz)) <= 0.0:
            raise ValueError(f"{clave} debe ser definida positiva.")

    for nombre_obs, fila in result["evaluations"].items():
        for nombre_lm, evaluacion in fila.items():
            if evaluacion["mahalanobis"] < 0.0:
                raise ValueError("Mahalanobis no puede ser negativo.")
            verdadero = result["true_associations"][nombre_obs]
            if verdadero == nombre_lm and verdadero is not None:
                if not evaluacion["inside_geometry_gate"]:
                    raise ValueError(
                        f"La asociación real {nombre_obs}-{nombre_lm} debe pasar el gate."
                    )


def validar_matching(result):
    """Comprueba asignación uno-a-uno, ambigüedad y falso alias."""

    global_map = result["global_assignment"]["associations"]
    asignados = [valor for valor in global_map.values() if valor is not None]
    if len(asignados) != len(set(asignados)):
        raise ValueError("El matching global debe ser uno-a-uno.")
    if result["method_comparison"]["independent_duplicates"] < 1:
        raise ValueError("El vecino independiente debe mostrar una duplicidad.")

    decisiones = result["decisions"]
    if decisiones["z4"]["status"] != "doubtful":
        raise ValueError("z4 debe representar la asociación dudosa.")
    if decisiones["z6"]["status"] != "new":
        raise ValueError("z6 debe quedar como observación nueva.")
    if decisiones["z5"]["landmark"] != "l5":
        raise ValueError("La geometría y el matching deben recuperar z5-l5.")

    alias = result["false_visual_alias"]
    if alias["landmark"] != "l2" or alias["inside_geometry_gate"]:
        raise ValueError("El alias z5-l2 debe ser visualmente fuerte y geométricamente falso.")


def validar_ransac(result):
    """Comprueba inliers, outliers y transformación estimada."""

    ransac = result["ransac"]
    if not ransac["accepted"]:
        raise ValueError("RANSAC debe aceptar la hipótesis correcta.")
    if ransac["inlier_count"] != 5:
        raise ValueError("Se esperan cinco inliers deterministas.")
    if ransac["outlier_count"] != 2:
        raise ValueError("Se esperan dos outliers deterministas.")
    if ransac["rmse"] >= UMBRAL_RANSAC:
        raise ValueError("El RMSE de RANSAC debe quedar bajo el umbral.")


def validar_metricas(result):
    """Comprueba métricas y efecto de la asociación falsa."""

    metricas = result["metrics"]
    for clave in ("precision", "recall", "f1"):
        if not (0.0 <= metricas[clave] <= 1.0):
            raise ValueError(f"{clave} debe pertenecer a [0,1].")
    if metricas["false_positives"] != 0:
        raise ValueError("El método final no debe aceptar falsos positivos.")
    if metricas["true_positives"] != 5:
        raise ValueError("Se esperan cinco asociaciones aceptadas correctas.")

    efecto = result["false_effect"]
    if efecto["translation_shift_false"] <= 0.15:
        raise ValueError("La asociación falsa debe desplazar apreciablemente la pose.")
    if efecto["translation_shift_robust"] >= efecto["translation_shift_false"]:
        raise ValueError("Huber debe reducir el desplazamiento provocado por el falso factor.")


def validar_grafos(result):
    """Comprueba el grafo de candidatos y el grafo de factores."""

    candidatos = result["candidate_graph"]
    factores = result["factor_graph"]
    if candidatos.number_of_nodes() != 13:
        raise ValueError("El grafo bipartito debe contener trece nodos.")
    if factores.number_of_edges() != 5:
        raise ValueError("Solo cinco asociaciones deben convertirse en factores.")
    if any(
        datos.get("node_type") == "observation"
        for _, datos in factores.nodes(data=True)
    ):
        raise ValueError("El grafo de factores no debe contener nodos de observación.")


def validar_resultados(result, states):
    """Ejecuta todas las validaciones y devuelve un resumen."""

    validar_escenario(result)
    validar_covarianzas(result)
    validar_matching(result)
    validar_ransac(result)
    validar_metricas(result)
    validar_grafos(result)

    if len(states) < 100:
        raise ValueError("La animación necesita al menos cien estados.")
    if states[-1]["phase"] != "summary":
        raise ValueError("El último estado debe ser el resumen.")

    decisiones = result["decisions"]
    conteos = {
        estado: sum(
            decision["status"] == estado
            for decision in decisiones.values()
        )
        for estado in ("correct", "doubtful", "false", "new", "rejected")
    }
    return {
        "landmark_count": len(result["landmarks"]),
        "observation_count": len(result["observations"]),
        "candidate_count": len(result["candidates"]),
        "geometry_candidate_count": int(
            np.sum(np.isfinite(result["cost_matrix"]))
        ),
        "accepted_count": sum(
            decision["accepted"] for decision in decisiones.values()
        ),
        "doubtful_count": conteos["doubtful"],
        "new_count": conteos["new"],
        "false_accepted_count": conteos["false"],
        "factor_count": result["factor_graph"].number_of_edges(),
        "independent_duplicates": result["method_comparison"]["independent_duplicates"],
        "global_duplicates": contar_duplicidades(
            result["global_assignment"]["associations"]
        ),
        "descriptor_accuracy": result["method_comparison"]["descriptor_accuracy"],
        "independent_accuracy": result["method_comparison"]["independent_accuracy"],
        "precision": result["metrics"]["precision"],
        "recall": result["metrics"]["recall"],
        "f1": result["metrics"]["f1"],
        "true_positives": result["metrics"]["true_positives"],
        "false_positives": result["metrics"]["false_positives"],
        "false_negatives": result["metrics"]["false_negatives"],
        "ransac_inliers": result["ransac"]["inlier_count"],
        "ransac_outliers": result["ransac"]["outlier_count"],
        "ransac_rmse": result["ransac"]["rmse"],
        "false_alias_descriptor_distance": result["false_visual_alias"]["descriptor_distance"],
        "false_alias_mahalanobis": result["false_visual_alias"]["mahalanobis"],
        "false_factor_weight": calcular_peso_huber(
            result["false_effect"]["false_factor"]["mahalanobis"]
        ),
        "false_pose_shift": result["false_effect"]["translation_shift_false"],
        "robust_pose_shift": result["false_effect"]["translation_shift_robust"],
        "state_count": len(states),
    }


# ---------------------------------------------------------------------------
# Salida de consola y ejecución
# ---------------------------------------------------------------------------


def imprimir_resumen(result, validation):
    """Imprime las magnitudes principales del experimento."""

    print("\n=== Asociación de datos ===")
    print(
        f"Landmarks: {validation['landmark_count']} · "
        f"observaciones: {validation['observation_count']}"
    )
    print(
        f"Candidatos visuales: {validation['candidate_count']} · "
        f"tras gate geométrico: {validation['geometry_candidate_count']}"
    )
    print(
        f"Aceptadas: {validation['accepted_count']} · "
        f"dudosas: {validation['doubtful_count']} · "
        f"nuevas: {validation['new_count']}"
    )
    print(
        f"Duplicidades NN/global: "
        f"{validation['independent_duplicates']}/"
        f"{validation['global_duplicates']}"
    )
    print(
        f"Precision/recall/F1: "
        f"{validation['precision']:.6f} / "
        f"{validation['recall']:.6f} / "
        f"{validation['f1']:.6f}"
    )
    print(
        f"RANSAC: {validation['ransac_inliers']} inliers · "
        f"{validation['ransac_outliers']} outliers · "
        f"RMSE {validation['ransac_rmse']:.6f} m"
    )
    print(
        "Alias z5-l2: "
        f"descriptor={validation['false_alias_descriptor_distance']:.6f} · "
        f"Mahalanobis={validation['false_alias_mahalanobis']:.6f}"
    )
    print(
        "Desplazamiento por falso factor sin/con robustez: "
        f"{validation['false_pose_shift']:.6f} / "
        f"{validation['robust_pose_shift']:.6f} m"
    )
    print(f"Factores creados: {validation['factor_count']}")
    print(f"Estados de animación: {validation['state_count']}")


def main():
    result = crear_resultado_asociacion_datos()
    states = crear_estados_animacion(result)
    validation = validar_resultados(result, states)
    imprimir_resumen(result, validation)

    animator = GraphAnimator(figsize=(20, 11), interval=240)
    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "06_graph_slam"
        / "06_asociacion_datos.png"
    )
    animator.animate_data_association(
        result=result,
        states=states,
        title="Asociación de datos: candidatos, gating, matching y verificación",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
