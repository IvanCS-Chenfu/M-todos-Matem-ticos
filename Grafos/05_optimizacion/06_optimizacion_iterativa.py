from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


# ---------------------------------------------------------------------------
# Datos deterministas del ejemplo
# ---------------------------------------------------------------------------

PARAMETROS_VERDADEROS = np.array(
    [1.80, 1.25, 0.45, 0.30],
    dtype=float,
)

PARAMETROS_INICIALES = np.array(
    [1.00, 1.00, 0.00, 0.70],
    dtype=float,
)

NUMERO_PUNTOS = 32
DOMINIO_X = (0.0, 6.0)
SIGMA_MEDICION = 0.10

LAMBDA_INICIAL = 1e-4
FACTOR_AUMENTO_LAMBDA = 10.0
LAMBDA_MINIMA = 1e-12
LAMBDA_MAXIMA = 1e12

MAXIMO_INTENTOS = 40
TOLERANCIA_PASO = 1e-7
TOLERANCIA_COSTE = 1e-10
TOLERANCIA_GRADIENTE = 1e-7


# ---------------------------------------------------------------------------
# Validación y modelo no lineal
# ---------------------------------------------------------------------------


def validar_vector(vector, nombre="vector", longitud=None):
    """Convierte un vector a float y valida dimensiones y valores finitos."""

    vector = np.asarray(vector, dtype=float)

    if vector.ndim != 1:
        raise ValueError(f"{nombre} debe ser un vector unidimensional.")
    if longitud is not None and vector.shape != (longitud,):
        raise ValueError(
            f"{nombre} debe contener exactamente {longitud} componentes."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{nombre} debe contener valores finitos.")

    return vector.copy()


def validar_parametros(parametros):
    """Valida los parámetros (amplitud, frecuencia, fase, desplazamiento)."""

    parametros = validar_vector(
        parametros,
        "parámetros",
        longitud=4,
    )

    amplitud, frecuencia, fase, desplazamiento = parametros

    if amplitud <= 0.0:
        raise ValueError("La amplitud debe ser positiva.")
    if frecuencia <= 0.0:
        raise ValueError("La frecuencia debe ser positiva.")
    if not np.isfinite(fase) or not np.isfinite(desplazamiento):
        raise ValueError("La fase y el desplazamiento deben ser finitos.")

    return parametros


def evaluar_modelo(x, parametros):
    """Evalúa y = a sin(bx + c) + d."""

    x = validar_vector(x, "x")
    amplitud, frecuencia, fase, desplazamiento = validar_parametros(parametros)

    return (
        amplitud * np.sin(frecuencia * x + fase)
        + desplazamiento
    )


def generar_datos_ruidosos(
    numero_puntos=NUMERO_PUNTOS,
    dominio=DOMINIO_X,
    parametros_verdaderos=PARAMETROS_VERDADEROS,
):
    """Genera puntos reproducibles mediante una combinación determinista de ondas."""

    numero_puntos = int(numero_puntos)

    if numero_puntos < 8:
        raise ValueError("Se necesitan al menos ocho puntos.")
    if len(dominio) != 2:
        raise ValueError("El dominio debe contener un límite inferior y otro superior.")

    x_min, x_max = map(float, dominio)

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        raise ValueError("El dominio debe ser finito y creciente.")

    x = np.linspace(x_min, x_max, numero_puntos, dtype=float)
    curva_verdadera = evaluar_modelo(x, parametros_verdaderos)

    ruido = (
        0.090 * np.sin(2.70 * x)
        + 0.045 * np.cos(5.10 * x)
        + 0.020 * np.sin(8.30 * x)
    )

    y = curva_verdadera + ruido
    sigmas = np.full(numero_puntos, SIGMA_MEDICION, dtype=float)

    return {
        "x": x,
        "y": y,
        "true_curve": curva_verdadera,
        "noise": ruido,
        "sigmas": sigmas,
    }


# ---------------------------------------------------------------------------
# Residuos, coste y jacobianos
# ---------------------------------------------------------------------------


def calcular_residuos(x, y, parametros):
    """Calcula e = y_medido - y_predicho."""

    x = validar_vector(x, "x")
    y = validar_vector(y, "y")

    if x.shape != y.shape:
        raise ValueError("x e y deben tener la misma longitud.")

    return y - evaluar_modelo(x, parametros)


def calcular_pesos(sigmas):
    """Calcula pesos escalares inversamente proporcionales a la varianza."""

    sigmas = validar_vector(sigmas, "sigmas")

    if np.any(sigmas <= 0.0):
        raise ValueError("Todas las desviaciones estándar deben ser positivas.")

    return 1.0 / (sigmas**2)


def calcular_coste(residuos, pesos):
    """Calcula 1/2 sum_k w_k e_k²."""

    residuos = validar_vector(residuos, "residuos")
    pesos = validar_vector(pesos, "pesos")

    if residuos.shape != pesos.shape:
        raise ValueError("Los residuos y los pesos deben tener la misma longitud.")
    if np.any(pesos <= 0.0):
        raise ValueError("Los pesos deben ser positivos.")

    coste = 0.5 * float(np.sum(pesos * residuos**2))

    if coste < -1e-12:
        raise ValueError("El coste no puede ser negativo.")

    return max(coste, 0.0)


def calcular_jacobiano_analitico(x, parametros):
    """Calcula el jacobiano de e = y - [a sin(bx+c) + d]."""

    x = validar_vector(x, "x")
    amplitud, frecuencia, fase, _ = validar_parametros(parametros)

    argumento = frecuencia * x + fase
    seno = np.sin(argumento)
    coseno = np.cos(argumento)

    return np.column_stack(
        [
            -seno,
            -amplitud * x * coseno,
            -amplitud * coseno,
            -np.ones_like(x),
        ]
    )


def calcular_jacobiano_numerico(x, y, parametros, epsilon=1e-7):
    """Aproxima el jacobiano de los residuos mediante diferencias centrales."""

    x = validar_vector(x, "x")
    y = validar_vector(y, "y")
    parametros = validar_parametros(parametros)
    epsilon = float(epsilon)

    if x.shape != y.shape:
        raise ValueError("x e y deben tener la misma longitud.")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon debe ser positivo y finito.")

    jacobiano = np.zeros((x.size, parametros.size), dtype=float)

    for columna in range(parametros.size):
        positivos = parametros.copy()
        negativos = parametros.copy()
        positivos[columna] += epsilon
        negativos[columna] -= epsilon

        if columna in {0, 1} and negativos[columna] <= 0.0:
            raise ValueError(
                "epsilon es demasiado grande para amplitud o frecuencia."
            )

        residuos_positivos = calcular_residuos(x, y, positivos)
        residuos_negativos = calcular_residuos(x, y, negativos)

        jacobiano[:, columna] = (
            residuos_positivos - residuos_negativos
        ) / (2.0 * epsilon)

    return jacobiano


def validar_jacobiano(
    x,
    y,
    parametros,
    tolerancia_absoluta=2e-6,
    tolerancia_relativa=2e-6,
):
    """Compara el jacobiano analítico con uno calculado numéricamente."""

    analitico = calcular_jacobiano_analitico(x, parametros)
    numerico = calcular_jacobiano_numerico(x, y, parametros)

    error_maximo = float(np.max(np.abs(analitico - numerico)))

    if not np.allclose(
        analitico,
        numerico,
        atol=tolerancia_absoluta,
        rtol=tolerancia_relativa,
    ):
        raise ValueError(
            "El jacobiano analítico no coincide con el jacobiano numérico."
        )

    return error_maximo


def construir_sistema_gauss_newton(
    x,
    y,
    parametros,
    pesos,
):
    """Construye H = JᵀWJ y g = JᵀWe."""

    residuos = calcular_residuos(x, y, parametros)
    pesos = validar_vector(pesos, "pesos")

    if pesos.shape != residuos.shape:
        raise ValueError("Los pesos no coinciden con el número de residuos.")

    jacobiano = calcular_jacobiano_analitico(x, parametros)
    raiz_pesos = np.sqrt(pesos)

    jacobiano_ponderado = jacobiano * raiz_pesos[:, None]
    residuos_ponderados = residuos * raiz_pesos

    hessiana = jacobiano_ponderado.T @ jacobiano_ponderado
    gradiente = jacobiano_ponderado.T @ residuos_ponderados

    if not np.allclose(hessiana, hessiana.T, atol=1e-10):
        raise ValueError("La Hessiana aproximada debe ser simétrica.")

    singulares = np.linalg.svd(hessiana, compute_uv=False)
    positivos = singulares[singulares > 1e-14]
    condicion = (
        float(positivos[0] / positivos[-1])
        if positivos.size
        else float("inf")
    )

    return {
        "residuals": residuos,
        "jacobian": jacobiano,
        "weighted_jacobian": jacobiano_ponderado,
        "weighted_residuals": residuos_ponderados,
        "hessian": hessiana,
        "gradient": gradiente,
        "cost": calcular_coste(residuos, pesos),
        "gradient_norm": float(np.linalg.norm(gradiente)),
        "condition_number": condicion,
    }


# ---------------------------------------------------------------------------
# Levenberg-Marquardt
# ---------------------------------------------------------------------------


def resolver_actualizacion(
    hessiana,
    gradiente,
    amortiguacion,
):
    """Resuelve (H + lambda diag(H)) delta = -g sin invertir matrices."""

    hessiana = np.asarray(hessiana, dtype=float)
    gradiente = validar_vector(gradiente, "gradiente")
    amortiguacion = float(amortiguacion)

    if hessiana.shape != (gradiente.size, gradiente.size):
        raise ValueError("H y g tienen dimensiones incompatibles.")
    if not np.all(np.isfinite(hessiana)):
        raise ValueError("La Hessiana debe contener valores finitos.")
    if not np.isfinite(amortiguacion) or amortiguacion <= 0.0:
        raise ValueError("La amortiguación debe ser positiva y finita.")

    diagonal = np.maximum(np.diag(hessiana), 1e-12)
    sistema = hessiana + amortiguacion * np.diag(diagonal)

    try:
        incremento = np.linalg.solve(sistema, -gradiente)
    except np.linalg.LinAlgError as exc:
        raise ValueError("No se pudo resolver el sistema amortiguado.") from exc

    if not np.all(np.isfinite(incremento)):
        raise ValueError("La actualización contiene valores no finitos.")

    return {
        "delta": incremento,
        "damped_hessian": sistema,
        "diagonal": diagonal,
    }


def aplicar_actualizacion(parametros, incremento):
    """Suma un incremento y valida el nuevo vector de parámetros."""

    parametros = validar_parametros(parametros)
    incremento = validar_vector(
        incremento,
        "incremento",
        longitud=parametros.size,
    )

    candidato = parametros + incremento

    if candidato[0] <= 0.0 or candidato[1] <= 0.0:
        return None

    return validar_parametros(candidato)


def calcular_reduccion_predicha(
    gradiente,
    hessiana,
    incremento,
):
    """Calcula la reducción del modelo cuadrático de Gauss-Newton."""

    gradiente = validar_vector(gradiente, "gradiente")
    incremento = validar_vector(
        incremento,
        "incremento",
        longitud=gradiente.size,
    )
    hessiana = np.asarray(hessiana, dtype=float)

    reduccion = -float(
        gradiente @ incremento
        + 0.5 * incremento @ hessiana @ incremento
    )

    return reduccion


def actualizar_amortiguacion(
    amortiguacion,
    rho,
    aceptado,
):
    """Actualiza lambda: disminuye tras un buen paso y aumenta tras un rechazo."""

    amortiguacion = float(amortiguacion)
    rho = float(rho)

    if aceptado:
        factor = min(
            1.0,
            max(
                1.0 / 3.0,
                1.0 - (2.0 * rho - 1.0) ** 3,
            ),
        )
        nueva = amortiguacion * factor
    else:
        nueva = amortiguacion * FACTOR_AUMENTO_LAMBDA

    return float(np.clip(nueva, LAMBDA_MINIMA, LAMBDA_MAXIMA))


def comprobar_convergencia(
    coste_anterior,
    coste_nuevo,
    norma_incremento,
    norma_gradiente,
):
    """Evalúa tres criterios de parada independientes."""

    cambio_coste = abs(float(coste_anterior) - float(coste_nuevo))

    return {
        "step": norma_incremento < TOLERANCIA_PASO,
        "cost": cambio_coste < TOLERANCIA_COSTE,
        "gradient": norma_gradiente < TOLERANCIA_GRADIENTE,
        "converged": (
            norma_incremento < TOLERANCIA_PASO
            or cambio_coste < TOLERANCIA_COSTE
            or norma_gradiente < TOLERANCIA_GRADIENTE
        ),
        "cost_change": cambio_coste,
    }


def ejecutar_levenberg_marquardt(
    x,
    y,
    sigmas,
    parametros_iniciales=PARAMETROS_INICIALES,
    lambda_inicial=LAMBDA_INICIAL,
    maximo_intentos=MAXIMO_INTENTOS,
):
    """Ajusta la curva y conserva cada propuesta aceptada o rechazada."""

    x = validar_vector(x, "x")
    y = validar_vector(y, "y")
    sigmas = validar_vector(sigmas, "sigmas")
    parametros = validar_parametros(parametros_iniciales)

    if x.shape != y.shape or x.shape != sigmas.shape:
        raise ValueError("x, y y sigmas deben tener la misma longitud.")

    pesos = calcular_pesos(sigmas)
    amortiguacion = float(lambda_inicial)
    historial = []
    costes_aceptados = []

    sistema_inicial = construir_sistema_gauss_newton(
        x,
        y,
        parametros,
        pesos,
    )
    costes_aceptados.append(sistema_inicial["cost"])

    convergencia = {
        "converged": False,
        "step": False,
        "cost": False,
        "gradient": False,
        "cost_change": float("inf"),
    }

    for intento in range(int(maximo_intentos)):
        sistema = construir_sistema_gauss_newton(
            x,
            y,
            parametros,
            pesos,
        )
        solucion = resolver_actualizacion(
            sistema["hessian"],
            sistema["gradient"],
            amortiguacion,
        )

        incremento = solucion["delta"]
        norma_incremento = float(np.linalg.norm(incremento))
        candidato = aplicar_actualizacion(parametros, incremento)

        coste_actual = sistema["cost"]
        coste_candidato = float("inf")

        if candidato is not None:
            residuos_candidato = calcular_residuos(x, y, candidato)
            coste_candidato = calcular_coste(
                residuos_candidato,
                pesos,
            )

        reduccion_real = coste_actual - coste_candidato
        reduccion_predicha = calcular_reduccion_predicha(
            sistema["gradient"],
            sistema["hessian"],
            incremento,
        )

        rho = (
            reduccion_real / reduccion_predicha
            if reduccion_predicha > 0.0
            else float("-inf")
        )

        aceptado = bool(
            candidato is not None
            and np.isfinite(coste_candidato)
            and reduccion_real > 0.0
            and rho > 1e-4
        )

        lambda_siguiente = actualizar_amortiguacion(
            amortiguacion,
            rho,
            aceptado,
        )

        registro = {
            "trial": intento,
            "accepted": aceptado,
            "parameters": parametros.copy(),
            "candidate_parameters": (
                candidato.copy()
                if candidato is not None
                else None
            ),
            "cost": coste_actual,
            "candidate_cost": coste_candidato,
            "lambda": amortiguacion,
            "next_lambda": lambda_siguiente,
            "delta": incremento.copy(),
            "step_norm": norma_incremento,
            "gradient_norm": sistema["gradient_norm"],
            "condition_number": sistema["condition_number"],
            "predicted_reduction": reduccion_predicha,
            "actual_reduction": reduccion_real,
            "rho": rho,
        }

        if aceptado:
            convergencia = comprobar_convergencia(
                coste_actual,
                coste_candidato,
                norma_incremento,
                sistema["gradient_norm"],
            )
            parametros = candidato
            costes_aceptados.append(coste_candidato)
        else:
            convergencia = {
                "converged": False,
                "step": False,
                "cost": False,
                "gradient": False,
                "cost_change": 0.0,
            }

        registro["convergence"] = dict(convergencia)
        historial.append(registro)
        amortiguacion = lambda_siguiente

        if aceptado and convergencia["converged"]:
            break

    sistema_final = construir_sistema_gauss_newton(
        x,
        y,
        parametros,
        pesos,
    )

    return {
        "initial_parameters": validar_parametros(parametros_iniciales),
        "final_parameters": parametros,
        "weights": pesos,
        "history": historial,
        "accepted_costs": np.asarray(costes_aceptados, dtype=float),
        "initial_cost": float(costes_aceptados[0]),
        "final_cost": sistema_final["cost"],
        "final_residuals": sistema_final["residuals"],
        "final_jacobian": sistema_final["jacobian"],
        "final_hessian": sistema_final["hessian"],
        "final_gradient": sistema_final["gradient"],
        "final_gradient_norm": sistema_final["gradient_norm"],
        "final_condition_number": sistema_final["condition_number"],
        "final_lambda": amortiguacion,
        "convergence": convergencia,
    }


# ---------------------------------------------------------------------------
# Métricas y estados didácticos
# ---------------------------------------------------------------------------


def calcular_metricas_ajuste(
    x,
    y,
    parametros,
    parametros_verdaderos,
):
    """Calcula RMSE, error máximo y distancia a los parámetros verdaderos."""

    residuos = calcular_residuos(x, y, parametros)
    parametros = validar_parametros(parametros)
    parametros_verdaderos = validar_parametros(parametros_verdaderos)

    return {
        "rmse": float(np.sqrt(np.mean(residuos**2))),
        "mae": float(np.mean(np.abs(residuos))),
        "max_abs_error": float(np.max(np.abs(residuos))),
        "parameter_error": float(
            np.linalg.norm(parametros - parametros_verdaderos)
        ),
    }


def _serializar_vector(vector):
    return [float(valor) for valor in np.asarray(vector, dtype=float)]


def _serializar_historial(historial):
    serializado = []

    for registro in historial:
        serializado.append(
            {
                "trial": int(registro["trial"]),
                "accepted": bool(registro["accepted"]),
                "parameters": _serializar_vector(registro["parameters"]),
                "candidate_parameters": (
                    _serializar_vector(registro["candidate_parameters"])
                    if registro["candidate_parameters"] is not None
                    else None
                ),
                "cost": float(registro["cost"]),
                "candidate_cost": float(registro["candidate_cost"]),
                "lambda": float(registro["lambda"]),
                "next_lambda": float(registro["next_lambda"]),
                "delta": _serializar_vector(registro["delta"]),
                "step_norm": float(registro["step_norm"]),
                "gradient_norm": float(registro["gradient_norm"]),
                "condition_number": float(registro["condition_number"]),
                "predicted_reduction": float(
                    registro["predicted_reduction"]
                ),
                "actual_reduction": float(registro["actual_reduction"]),
                "rho": float(registro["rho"]),
                "convergence": dict(registro["convergence"]),
            }
        )

    return serializado


def crear_estado_animacion(
    *,
    phase,
    message,
    data,
    optimization,
    visible_points=None,
    current_parameters=None,
    candidate_parameters=None,
    visible_history_count=0,
    show_true_curve=False,
    show_initial_curve=False,
    show_current_curve=False,
    show_candidate_curve=False,
    show_final_curve=False,
    show_residuals=False,
    show_cost_history=False,
    show_linearization=False,
    show_damping=False,
    show_connections=False,
    accepted=None,
    trial=None,
):
    """Crea un fotograma independiente con todos los datos necesarios."""

    if visible_points is None:
        visible_points = data["x"].size
    if current_parameters is None:
        current_parameters = optimization["initial_parameters"]

    current_parameters = validar_parametros(current_parameters)
    candidate_curve = None

    if candidate_parameters is not None:
        candidate_parameters = validar_parametros(candidate_parameters)
        candidate_curve = evaluar_modelo(
            data["x"],
            candidate_parameters,
        )

    current_curve = evaluar_modelo(
        data["x"],
        current_parameters,
    )

    current_residuals = data["y"] - current_curve
    current_cost = calcular_coste(
        current_residuals,
        optimization["weights"],
    )

    record = None
    if trial is not None and 0 <= trial < len(optimization["history"]):
        record = optimization["history"][trial]

    return {
        "phase": phase,
        "message": message,
        "x_values": _serializar_vector(data["x"]),
        "y_values": _serializar_vector(data["y"]),
        "true_curve": _serializar_vector(data["true_curve"]),
        "initial_curve": _serializar_vector(
            evaluar_modelo(
                data["x"],
                optimization["initial_parameters"],
            )
        ),
        "current_curve": _serializar_vector(current_curve),
        "candidate_curve": (
            _serializar_vector(candidate_curve)
            if candidate_curve is not None
            else None
        ),
        "final_curve": _serializar_vector(
            evaluar_modelo(
                data["x"],
                optimization["final_parameters"],
            )
        ),
        "true_parameters": _serializar_vector(PARAMETROS_VERDADEROS),
        "initial_parameters": _serializar_vector(
            optimization["initial_parameters"]
        ),
        "current_parameters": _serializar_vector(current_parameters),
        "candidate_parameters": (
            _serializar_vector(candidate_parameters)
            if candidate_parameters is not None
            else None
        ),
        "final_parameters": _serializar_vector(
            optimization["final_parameters"]
        ),
        "current_residuals": _serializar_vector(current_residuals),
        "initial_cost": float(optimization["initial_cost"]),
        "current_cost": float(current_cost),
        "final_cost": float(optimization["final_cost"]),
        "accepted_costs": _serializar_vector(
            optimization["accepted_costs"]
        ),
        "history": _serializar_historial(optimization["history"]),
        "visible_history_count": int(visible_history_count),
        "visible_points": int(visible_points),
        "show_true_curve": bool(show_true_curve),
        "show_initial_curve": bool(show_initial_curve),
        "show_current_curve": bool(show_current_curve),
        "show_candidate_curve": bool(show_candidate_curve),
        "show_final_curve": bool(show_final_curve),
        "show_residuals": bool(show_residuals),
        "show_cost_history": bool(show_cost_history),
        "show_linearization": bool(show_linearization),
        "show_damping": bool(show_damping),
        "show_connections": bool(show_connections),
        "accepted": accepted,
        "trial": trial,
        "lambda": (
            float(record["lambda"])
            if record is not None
            else float(LAMBDA_INICIAL)
        ),
        "next_lambda": (
            float(record["next_lambda"])
            if record is not None
            else float(LAMBDA_INICIAL)
        ),
        "step_norm": (
            float(record["step_norm"])
            if record is not None
            else 0.0
        ),
        "gradient_norm": (
            float(record["gradient_norm"])
            if record is not None
            else 0.0
        ),
        "condition_number": (
            float(record["condition_number"])
            if record is not None
            else 0.0
        ),
        "rho": (
            float(record["rho"])
            if record is not None
            else 0.0
        ),
        "candidate_cost": (
            float(record["candidate_cost"])
            if record is not None
            else float(current_cost)
        ),
        "actual_reduction": (
            float(record["actual_reduction"])
            if record is not None
            else 0.0
        ),
        "predicted_reduction": (
            float(record["predicted_reduction"])
            if record is not None
            else 0.0
        ),
        "final_metrics": calcular_metricas_ajuste(
            data["x"],
            data["y"],
            optimization["final_parameters"],
            PARAMETROS_VERDADEROS,
        ),
        "initial_metrics": calcular_metricas_ajuste(
            data["x"],
            data["y"],
            optimization["initial_parameters"],
            PARAMETROS_VERDADEROS,
        ),
    }


def crear_estados_animacion(data, optimization):
    """Crea una secuencia didáctica completa para el apartado 5.6."""

    states = []

    def add(phase, message, repeat=1, **kwargs):
        for _ in range(repeat):
            states.append(
                crear_estado_animacion(
                    phase=phase,
                    message=message,
                    data=data,
                    optimization=optimization,
                    **kwargs,
                )
            )

    add(
        "introduction",
        "Los modelos no lineales se ajustan mediante aproximaciones locales sucesivas.",
        repeat=3,
        visible_points=0,
    )

    add(
        "true_model",
        "La curva verdadera solo se utiliza para generar y validar el ejemplo.",
        repeat=3,
        visible_points=0,
        show_true_curve=True,
    )

    for count in [4, 8, 12, 18, 24, data["x"].size]:
        add(
            "measurements",
            "Se incorporan mediciones con ruido determinista.",
            visible_points=count,
            show_true_curve=True,
        )

    add(
        "initial_curve",
        "La estimación inicial no explica correctamente los puntos medidos.",
        repeat=3,
        show_true_curve=True,
        show_initial_curve=True,
        show_current_curve=True,
    )

    add(
        "initial_residuals",
        "Los segmentos verticales representan los residuos de la estimación actual.",
        repeat=3,
        show_true_curve=True,
        show_initial_curve=True,
        show_current_curve=True,
        show_residuals=True,
        show_cost_history=True,
    )

    add(
        "linearization",
        "Se linealizan los residuos y se construyen J, H=JᵀWJ y g=JᵀWe.",
        repeat=3,
        show_true_curve=True,
        show_initial_curve=True,
        show_current_curve=True,
        show_residuals=True,
        show_cost_history=True,
        show_linearization=True,
    )

    current_parameters = optimization["initial_parameters"].copy()
    visible_history = 0

    for trial, record in enumerate(optimization["history"]):
        candidate = record["candidate_parameters"]

        add(
            "proposal",
            "Levenberg-Marquardt propone una actualización amortiguada.",
            current_parameters=current_parameters,
            candidate_parameters=candidate,
            visible_history_count=visible_history,
            show_true_curve=True,
            show_initial_curve=True,
            show_current_curve=True,
            show_candidate_curve=candidate is not None,
            show_residuals=True,
            show_cost_history=True,
            show_linearization=True,
            show_damping=True,
            accepted=None,
            trial=trial,
        )

        visible_history = trial + 1

        if record["accepted"]:
            add(
                "accepted",
                "El coste disminuye: el paso se acepta y lambda se reduce.",
                current_parameters=current_parameters,
                candidate_parameters=candidate,
                visible_history_count=visible_history,
                show_true_curve=True,
                show_initial_curve=True,
                show_current_curve=True,
                show_candidate_curve=True,
                show_residuals=True,
                show_cost_history=True,
                show_linearization=True,
                show_damping=True,
                accepted=True,
                trial=trial,
            )

            current_parameters = candidate.copy()

            add(
                "updated",
                "La curva aceptada se convierte en la nueva estimación.",
                current_parameters=current_parameters,
                visible_history_count=visible_history,
                show_true_curve=True,
                show_initial_curve=True,
                show_current_curve=True,
                show_residuals=True,
                show_cost_history=True,
                show_linearization=True,
                show_damping=True,
                accepted=True,
                trial=trial,
            )
        else:
            add(
                "rejected",
                "El coste aumentaría: el paso se rechaza y lambda se incrementa.",
                repeat=2,
                current_parameters=current_parameters,
                candidate_parameters=candidate,
                visible_history_count=visible_history,
                show_true_curve=True,
                show_initial_curve=True,
                show_current_curve=True,
                show_candidate_curve=candidate is not None,
                show_residuals=True,
                show_cost_history=True,
                show_linearization=True,
                show_damping=True,
                accepted=False,
                trial=trial,
            )

    add(
        "convergence",
        "Las actualizaciones ya son pequeñas y el coste se ha estabilizado.",
        repeat=4,
        current_parameters=optimization["final_parameters"],
        visible_history_count=len(optimization["history"]),
        show_true_curve=True,
        show_initial_curve=True,
        show_current_curve=True,
        show_final_curve=True,
        show_residuals=True,
        show_cost_history=True,
        show_linearization=True,
        show_damping=True,
        accepted=True,
        trial=len(optimization["history"]) - 1,
    )

    add(
        "connections",
        "La misma estructura iterativa aparece en Gauss-Newton, LM y Graph SLAM.",
        repeat=4,
        current_parameters=optimization["final_parameters"],
        visible_history_count=len(optimization["history"]),
        show_true_curve=True,
        show_initial_curve=True,
        show_current_curve=True,
        show_final_curve=True,
        show_residuals=True,
        show_cost_history=True,
        show_linearization=True,
        show_damping=True,
        show_connections=True,
        accepted=True,
        trial=len(optimization["history"]) - 1,
    )

    add(
        "summary",
        "Linealizar, resolver, actualizar y repetir transforma el ajuste no lineal.",
        repeat=4,
        current_parameters=optimization["final_parameters"],
        visible_history_count=len(optimization["history"]),
        show_true_curve=True,
        show_initial_curve=True,
        show_current_curve=True,
        show_final_curve=True,
        show_residuals=True,
        show_cost_history=True,
        show_linearization=True,
        show_damping=True,
        show_connections=True,
        accepted=True,
        trial=len(optimization["history"]) - 1,
    )

    for index, state in enumerate(states, start=1):
        state["step"] = index
        state["total_steps"] = len(states)

    return states


# ---------------------------------------------------------------------------
# Validaciones y salida
# ---------------------------------------------------------------------------


def validar_descenso_coste(optimization):
    """Comprueba que los costes aceptados sean monótonos no crecientes."""

    accepted_costs = validar_vector(
        optimization["accepted_costs"],
        "costes aceptados",
    )

    if accepted_costs.size < 2:
        raise ValueError("Debe existir al menos una actualización aceptada.")

    differences = np.diff(accepted_costs)

    if np.any(differences > 1e-9):
        raise ValueError("Un paso aceptado aumentó el coste.")

    return differences


def validar_resultados(data, optimization, states):
    """Ejecuta comprobaciones matemáticas, numéricas y didácticas."""

    x = validar_vector(data["x"], "x")
    y = validar_vector(data["y"], "y")
    sigmas = validar_vector(data["sigmas"], "sigmas")

    if x.shape != y.shape or x.shape != sigmas.shape:
        raise ValueError("Los datos del ejemplo no tienen dimensiones coherentes.")

    validar_jacobiano(
        x,
        y,
        optimization["initial_parameters"],
    )
    validar_jacobiano(
        x,
        y,
        optimization["final_parameters"],
    )

    validar_descenso_coste(optimization)

    if optimization["final_cost"] >= optimization["initial_cost"]:
        raise ValueError("El coste final debe ser menor que el coste inicial.")

    if not np.all(np.isfinite(optimization["final_parameters"])):
        raise ValueError("Los parámetros finales deben ser finitos.")

    initial_metrics = calcular_metricas_ajuste(
        x,
        y,
        optimization["initial_parameters"],
        PARAMETROS_VERDADEROS,
    )
    final_metrics = calcular_metricas_ajuste(
        x,
        y,
        optimization["final_parameters"],
        PARAMETROS_VERDADEROS,
    )

    if final_metrics["rmse"] >= initial_metrics["rmse"]:
        raise ValueError("El RMSE final debe ser menor que el inicial.")
    if final_metrics["parameter_error"] >= initial_metrics["parameter_error"]:
        raise ValueError(
            "Los parámetros finales deben acercarse a los parámetros verdaderos."
        )

    rejected = [
        record
        for record in optimization["history"]
        if not record["accepted"]
    ]
    accepted = [
        record
        for record in optimization["history"]
        if record["accepted"]
    ]

    if not rejected:
        raise ValueError(
            "La demostración debe contener al menos un paso rechazado."
        )
    if not accepted:
        raise ValueError(
            "La demostración debe contener al menos un paso aceptado."
        )

    for record in rejected:
        if record["next_lambda"] <= record["lambda"]:
            raise ValueError("lambda debe aumentar tras un rechazo.")

    for record in accepted:
        if record["next_lambda"] > record["lambda"] * 1.000001:
            raise ValueError("lambda no debe aumentar tras un paso aceptado.")

    if len(states) < 60:
        raise ValueError("La demostración debe contener al menos sesenta estados.")
    if states[-1].get("phase") != "summary":
        raise ValueError("El último estado debe ser el resumen final.")
    if not states[-1].get("show_final_curve"):
        raise ValueError("La imagen final debe mostrar la curva ajustada.")
    if not states[-1].get("show_cost_history"):
        raise ValueError("La imagen final debe mostrar el historial de coste.")

    return {
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "accepted_steps": len(accepted),
        "rejected_steps": len(rejected),
    }


def _formatear_parametros(parametros):
    parametros = validar_parametros(parametros)
    return (
        f"a={parametros[0]:.6f}, "
        f"b={parametros[1]:.6f}, "
        f"c={parametros[2]:.6f}, "
        f"d={parametros[3]:.6f}"
    )


def imprimir_resumen(data, optimization, validation, states):
    """Imprime las magnitudes principales del ajuste."""

    print("\n=== Optimización no lineal iterativa ===")
    print(f"Puntos: {data['x'].size}")
    print("Parámetros verdaderos:", _formatear_parametros(PARAMETROS_VERDADEROS))
    print(
        "Parámetros iniciales:",
        _formatear_parametros(optimization["initial_parameters"]),
    )
    print(
        "Parámetros finales:",
        _formatear_parametros(optimization["final_parameters"]),
    )
    print(f"Coste inicial: {optimization['initial_cost']:.9f}")
    print(f"Coste final: {optimization['final_cost']:.9f}")
    print(
        "RMSE inicial/final: "
        f"{validation['initial_metrics']['rmse']:.9f} / "
        f"{validation['final_metrics']['rmse']:.9f}"
    )
    print(
        "Pasos aceptados/rechazados: "
        f"{validation['accepted_steps']} / {validation['rejected_steps']}"
    )
    print(f"Intentos totales: {len(optimization['history'])}")
    print(f"Estados de animación: {len(states)}")
    print(f"Lambda final: {optimization['final_lambda']:.12g}")


def main():
    data = generar_datos_ruidosos()

    optimization = ejecutar_levenberg_marquardt(
        x=data["x"],
        y=data["y"],
        sigmas=data["sigmas"],
        parametros_iniciales=PARAMETROS_INICIALES,
    )

    states = crear_estados_animacion(
        data,
        optimization,
    )

    validation = validar_resultados(
        data,
        optimization,
        states,
    )

    imprimir_resumen(
        data,
        optimization,
        validation,
        states,
    )

    animator = GraphAnimator(
        figsize=(18, 10),
        interval=520,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "05_optimizacion"
        / "06_optimizacion_iterativa.png"
    )

    animator.animate_nonlinear_optimization(
        x_values=data["x"],
        y_values=data["y"],
        states=states,
        title="Optimización no lineal iterativa",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
