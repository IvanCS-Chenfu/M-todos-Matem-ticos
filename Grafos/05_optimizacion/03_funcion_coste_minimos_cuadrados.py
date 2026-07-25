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

MEDICIONES = np.array([2.0, 4.0, 7.0], dtype=float)
PESOS = np.array([9.0, 1.0, 4.0], dtype=float)
ESTIMACION_INICIAL = 8.2
DOMINIO_X = (0.5, 8.7)


# ---------------------------------------------------------------------------
# Validación y operaciones de mínimos cuadrados
# ---------------------------------------------------------------------------


def validar_datos(mediciones, pesos=None):
    """Valida mediciones escalares y, opcionalmente, sus pesos."""

    mediciones = np.asarray(mediciones, dtype=float)

    if mediciones.ndim != 1 or mediciones.size == 0:
        raise ValueError("Debe existir al menos una medición escalar.")

    if not np.all(np.isfinite(mediciones)):
        raise ValueError("Todas las mediciones deben ser finitas.")

    if pesos is None:
        return mediciones

    pesos = np.asarray(pesos, dtype=float)

    if pesos.shape != mediciones.shape:
        raise ValueError("Debe existir un peso por cada medición.")

    if not np.all(np.isfinite(pesos)):
        raise ValueError("Todos los pesos deben ser finitos.")

    if np.any(pesos <= 0.0):
        raise ValueError("Todos los pesos deben ser estrictamente positivos.")

    if float(np.sum(pesos)) <= 0.0:
        raise ValueError("La suma de los pesos debe ser positiva.")

    return mediciones, pesos


def calcular_residuos(estimacion, mediciones):
    """Calcula e_k(x) = x - z_k para todas las mediciones."""

    mediciones = validar_datos(mediciones)
    estimacion = float(estimacion)

    if not np.isfinite(estimacion):
        raise ValueError("La estimación debe ser finita.")

    return estimacion - mediciones


def calcular_costes_individuales(residuos):
    """Calcula las contribuciones no ponderadas e_k²."""

    residuos = np.asarray(residuos, dtype=float)

    if residuos.ndim != 1 or residuos.size == 0:
        raise ValueError("Se esperaba un vector no vacío de residuos.")

    if not np.all(np.isfinite(residuos)):
        raise ValueError("Los residuos deben ser finitos.")

    return residuos**2


def calcular_coste_total(estimacion, mediciones):
    """Calcula F(x) = sum_k (x - z_k)²."""

    residuos = calcular_residuos(estimacion, mediciones)
    return float(np.sum(calcular_costes_individuales(residuos)))


def calcular_costes_ponderados_individuales(estimacion, mediciones, pesos):
    """Calcula w_k (x - z_k)² para cada medición."""

    mediciones, pesos = validar_datos(mediciones, pesos)
    residuos = calcular_residuos(estimacion, mediciones)
    return pesos * residuos**2


def calcular_coste_ponderado(estimacion, mediciones, pesos):
    """Calcula F_w(x) = sum_k w_k (x - z_k)²."""

    return float(
        np.sum(
            calcular_costes_ponderados_individuales(
                estimacion,
                mediciones,
                pesos,
            )
        )
    )


def calcular_media(mediciones):
    """Devuelve el mínimo analítico del coste no ponderado."""

    mediciones = validar_datos(mediciones)
    return float(np.mean(mediciones))


def calcular_media_ponderada(mediciones, pesos):
    """Devuelve el mínimo analítico del coste ponderado."""

    mediciones, pesos = validar_datos(mediciones, pesos)
    return float(np.dot(pesos, mediciones) / np.sum(pesos))


def calcular_derivada_coste(estimacion, mediciones, pesos=None):
    """Calcula la primera derivada del coste en una estimación."""

    mediciones = validar_datos(mediciones)
    residuos = calcular_residuos(estimacion, mediciones)

    if pesos is None:
        return float(2.0 * np.sum(residuos))

    _, pesos = validar_datos(mediciones, pesos)
    return float(2.0 * np.dot(pesos, residuos))


def calcular_segunda_derivada(mediciones, pesos=None):
    """Calcula la curvatura constante de la función cuadrática."""

    mediciones = validar_datos(mediciones)

    if pesos is None:
        return float(2.0 * mediciones.size)

    _, pesos = validar_datos(mediciones, pesos)
    return float(2.0 * np.sum(pesos))


def evaluar_estimacion(estimacion, mediciones, pesos):
    """Reúne residuos, costes, derivadas y mínimos para un valor de x."""

    mediciones, pesos = validar_datos(mediciones, pesos)
    estimacion = float(estimacion)
    residuos = calcular_residuos(estimacion, mediciones)
    costes = calcular_costes_individuales(residuos)
    costes_ponderados = pesos * costes

    return {
        "estimate": estimacion,
        "measurements": mediciones.copy(),
        "weights": pesos.copy(),
        "residuals": residuos,
        "individual_costs": costes,
        "weighted_individual_costs": costes_ponderados,
        "unweighted_cost": float(np.sum(costes)),
        "weighted_cost": float(np.sum(costes_ponderados)),
        "unweighted_derivative": calcular_derivada_coste(
            estimacion,
            mediciones,
        ),
        "weighted_derivative": calcular_derivada_coste(
            estimacion,
            mediciones,
            pesos,
        ),
        "unweighted_second_derivative": calcular_segunda_derivada(
            mediciones,
        ),
        "weighted_second_derivative": calcular_segunda_derivada(
            mediciones,
            pesos,
        ),
        "unweighted_minimum": calcular_media(mediciones),
        "weighted_minimum": calcular_media_ponderada(mediciones, pesos),
    }


# ---------------------------------------------------------------------------
# Curvas, grafo de factores y descenso
# ---------------------------------------------------------------------------


def crear_dominio_coste(limites=DOMINIO_X, numero_puntos=420):
    """Crea el dominio usado para representar las funciones de coste."""

    if len(limites) != 2:
        raise ValueError("Los límites deben contener mínimo y máximo.")

    minimo, maximo = map(float, limites)

    if not np.isfinite(minimo) or not np.isfinite(maximo):
        raise ValueError("Los límites del dominio deben ser finitos.")

    if minimo >= maximo:
        raise ValueError("El límite inferior debe ser menor que el superior.")

    if numero_puntos < 100:
        raise ValueError("Se requieren al menos cien puntos para la curva.")

    return np.linspace(minimo, maximo, int(numero_puntos), dtype=float)


def evaluar_curvas_coste(dominio, mediciones, pesos):
    """Evalúa costes individuales, total y ponderado sobre un dominio."""

    mediciones, pesos = validar_datos(mediciones, pesos)
    dominio = np.asarray(dominio, dtype=float)

    if dominio.ndim != 1 or dominio.size == 0:
        raise ValueError("El dominio debe ser un vector no vacío.")

    if not np.all(np.isfinite(dominio)):
        raise ValueError("El dominio debe contener valores finitos.")

    residuos = dominio[np.newaxis, :] - mediciones[:, np.newaxis]
    curvas_individuales = residuos**2
    curvas_ponderadas = pesos[:, np.newaxis] * curvas_individuales

    return {
        "domain": dominio.copy(),
        "individual_curves": curvas_individuales,
        "weighted_individual_curves": curvas_ponderadas,
        "unweighted_curve": np.sum(curvas_individuales, axis=0),
        "weighted_curve": np.sum(curvas_ponderadas, axis=0),
    }


def crear_grafo_funcion_coste(mediciones, pesos):
    """Crea un grafo bipartito con una variable y un factor por medición."""

    mediciones, pesos = validar_datos(mediciones, pesos)
    graph = nx.Graph()
    graph.graph["name"] = "Función de coste y mínimos cuadrados"
    graph.graph["objective"] = "sum_k w_k * (x - z_k)^2"
    graph.graph["variable"] = "x"

    graph.add_node(
        "x",
        node_type="variable",
        bipartite=0,
        estimate=float(ESTIMACION_INICIAL),
        label="x",
        description="Variable escalar que se desea estimar.",
    )

    for index, (measurement, weight) in enumerate(
        zip(mediciones, pesos),
        start=1,
    ):
        factor = f"f{index}"
        graph.add_node(
            factor,
            node_type="measurement_factor",
            bipartite=1,
            measurement=float(measurement),
            weight=float(weight),
            sigma=float(1.0 / np.sqrt(weight)),
            label=factor,
        )
        graph.add_edge(
            "x",
            factor,
            relation="scalar_residual",
            measurement=float(measurement),
            weight=float(weight),
            residual_model="x - z_k",
            cost_model="w_k * residual^2",
        )

    return graph


def realizar_descenso_gradiente(
    estimacion_inicial,
    mediciones,
    *,
    pesos=None,
    tasa_aprendizaje,
    iteraciones,
):
    """Ejecuta descenso por gradiente sobre el coste cuadrático elegido."""

    mediciones = validar_datos(mediciones)

    if pesos is not None:
        _, pesos = validar_datos(mediciones, pesos)

    tasa_aprendizaje = float(tasa_aprendizaje)

    if not np.isfinite(tasa_aprendizaje) or tasa_aprendizaje <= 0.0:
        raise ValueError("La tasa de aprendizaje debe ser positiva y finita.")

    if iteraciones < 1:
        raise ValueError("Se requiere al menos una iteración.")

    curvatura = calcular_segunda_derivada(mediciones, pesos)

    if tasa_aprendizaje >= 2.0 / curvatura:
        raise ValueError(
            "La tasa elegida no garantiza convergencia para esta cuadrática."
        )

    estimacion = float(estimacion_inicial)
    history = []

    for iteration in range(int(iteraciones) + 1):
        evaluation = evaluar_estimacion(estimacion, mediciones, PESOS)
        derivative = calcular_derivada_coste(
            estimacion,
            mediciones,
            pesos,
        )
        active_cost = (
            evaluation["unweighted_cost"]
            if pesos is None
            else evaluation["weighted_cost"]
        )

        history.append(
            {
                "iteration": iteration,
                "estimate": estimacion,
                "derivative": derivative,
                "active_cost": active_cost,
                "evaluation": evaluation,
            }
        )

        estimacion = estimacion - tasa_aprendizaje * derivative

    return history


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


def crear_estado_animacion(
    evaluation,
    curves,
    *,
    phase,
    message,
    step,
    total_steps,
    mode="unweighted",
    iteration=None,
    learning_rate=None,
    history=None,
    show_measurements=False,
    show_residuals=False,
    show_individual_costs=False,
    show_total_cost=False,
    show_estimate=False,
    show_gradient=False,
    show_unweighted_minimum=False,
    show_weights=False,
    show_weighted_curve=False,
    show_weighted_minimum=False,
    show_factor_graph=False,
    show_pose_graph_connection=False,
):
    """Convierte una evaluación en un fotograma independiente."""

    history = history or []
    active_cost = (
        evaluation["unweighted_cost"]
        if mode == "unweighted"
        else evaluation["weighted_cost"]
    )
    active_derivative = (
        evaluation["unweighted_derivative"]
        if mode == "unweighted"
        else evaluation["weighted_derivative"]
    )
    active_second_derivative = (
        evaluation["unweighted_second_derivative"]
        if mode == "unweighted"
        else evaluation["weighted_second_derivative"]
    )

    return {
        "phase": phase,
        "message": message,
        "step": int(step),
        "total_steps": int(total_steps),
        "mode": mode,
        "iteration": None if iteration is None else int(iteration),
        "learning_rate": (
            None if learning_rate is None else float(learning_rate)
        ),
        "estimate": float(evaluation["estimate"]),
        "initial_estimate": float(ESTIMACION_INICIAL),
        "measurements": _serializar_vector(evaluation["measurements"]),
        "weights": _serializar_vector(evaluation["weights"]),
        "residuals": _serializar_vector(evaluation["residuals"]),
        "individual_costs": _serializar_vector(
            evaluation["individual_costs"]
        ),
        "weighted_individual_costs": _serializar_vector(
            evaluation["weighted_individual_costs"]
        ),
        "unweighted_cost": float(evaluation["unweighted_cost"]),
        "weighted_cost": float(evaluation["weighted_cost"]),
        "active_cost": float(active_cost),
        "unweighted_derivative": float(
            evaluation["unweighted_derivative"]
        ),
        "weighted_derivative": float(evaluation["weighted_derivative"]),
        "active_derivative": float(active_derivative),
        "active_second_derivative": float(active_second_derivative),
        "unweighted_minimum": float(evaluation["unweighted_minimum"]),
        "weighted_minimum": float(evaluation["weighted_minimum"]),
        "unweighted_minimum_cost": calcular_coste_total(
            evaluation["unweighted_minimum"],
            evaluation["measurements"],
        ),
        "weighted_minimum_cost": calcular_coste_ponderado(
            evaluation["weighted_minimum"],
            evaluation["measurements"],
            evaluation["weights"],
        ),
        "domain": _serializar_vector(curves["domain"]),
        "individual_curves": _serializar_matriz(
            curves["individual_curves"]
        ),
        "weighted_individual_curves": _serializar_matriz(
            curves["weighted_individual_curves"]
        ),
        "unweighted_curve": _serializar_vector(
            curves["unweighted_curve"]
        ),
        "weighted_curve": _serializar_vector(curves["weighted_curve"]),
        "history_estimates": [float(item["estimate"]) for item in history],
        "history_costs": [float(item["active_cost"]) for item in history],
        "show_measurements": bool(show_measurements),
        "show_residuals": bool(show_residuals),
        "show_individual_costs": bool(show_individual_costs),
        "show_total_cost": bool(show_total_cost),
        "show_estimate": bool(show_estimate),
        "show_gradient": bool(show_gradient),
        "show_unweighted_minimum": bool(show_unweighted_minimum),
        "show_weights": bool(show_weights),
        "show_weighted_curve": bool(show_weighted_curve),
        "show_weighted_minimum": bool(show_weighted_minimum),
        "show_factor_graph": bool(show_factor_graph),
        "show_pose_graph_connection": bool(show_pose_graph_connection),
    }


def crear_estados_animacion(graph):
    """Construye la narración completa del apartado 5.3."""

    validar_grafo_funcion_coste(graph)
    domain = crear_dominio_coste()
    curves = evaluar_curvas_coste(domain, MEDICIONES, PESOS)
    initial = evaluar_estimacion(ESTIMACION_INICIAL, MEDICIONES, PESOS)
    unweighted_minimum = calcular_media(MEDICIONES)
    weighted_minimum = calcular_media_ponderada(MEDICIONES, PESOS)
    unweighted_final = evaluar_estimacion(
        unweighted_minimum,
        MEDICIONES,
        PESOS,
    )
    weighted_final = evaluar_estimacion(
        weighted_minimum,
        MEDICIONES,
        PESOS,
    )

    unweighted_descent = realizar_descenso_gradiente(
        ESTIMACION_INICIAL,
        MEDICIONES,
        pesos=None,
        tasa_aprendizaje=0.08,
        iteraciones=11,
    )
    weighted_descent = realizar_descenso_gradiente(
        unweighted_minimum,
        MEDICIONES,
        pesos=PESOS,
        tasa_aprendizaje=0.025,
        iteraciones=9,
    )

    states = []

    def add(evaluation, phase, message, **kwargs):
        states.append(
            crear_estado_animacion(
                evaluation=evaluation,
                curves=curves,
                phase=phase,
                message=message,
                step=len(states) + 1,
                total_steps=0,
                **kwargs,
            )
        )

    # 1. Variable desconocida.
    add(
        initial,
        "variable",
        "x es la variable escalar que deseamos estimar.",
        show_estimate=True,
    )
    add(
        initial,
        "variable",
        "Una estimación concreta de x se representa como un punto sobre el eje.",
        show_estimate=True,
    )

    # 2. Mediciones.
    for message in (
        "El sensor proporciona tres mediciones: z1, z2 y z3.",
        "Las mediciones permanecen fijas; la variable x es la que puede cambiar.",
        "Cada medición propone un valor diferente para la misma variable.",
    ):
        add(
            initial,
            "measurements",
            message,
            show_measurements=True,
            show_estimate=True,
        )

    # 3. Residuos.
    for message in (
        "Cada medición produce un residuo e_k(x) = x - z_k.",
        "El signo indica a qué lado de la medición se encuentra la estimación.",
        "Los residuos cambian cada vez que cambia x.",
    ):
        add(
            initial,
            "residuals",
            message,
            show_measurements=True,
            show_residuals=True,
            show_estimate=True,
        )

    # 4. Cuadrados de los residuos.
    for message in (
        "Elevar al cuadrado evita que residuos positivos y negativos se cancelen.",
        "Cada medición aporta un coste local F_k(x) = e_k(x)^2.",
        "Los errores grandes reciben una penalización cuadrática mayor.",
    ):
        add(
            initial,
            "squared_residuals",
            message,
            show_measurements=True,
            show_residuals=True,
            show_individual_costs=True,
            show_estimate=True,
        )

    # 5. Curvas individuales.
    for message in (
        "Cada coste local es una parábola centrada en su medición.",
        "El mínimo de F_k aparece cuando x coincide con z_k.",
        "Las tres curvas individuales expresan tres preferencias diferentes.",
    ):
        add(
            initial,
            "individual_curves",
            message,
            show_measurements=True,
            show_residuals=True,
            show_individual_costs=True,
            show_estimate=True,
        )

    # 6. Suma total.
    for message in (
        "La función total suma verticalmente todos los costes individuales.",
        "F(x) = Σ_k (x - z_k)^2 convierte todos los residuos en un escalar.",
        "Optimizar significa buscar el punto más bajo de esta curva.",
    ):
        add(
            initial,
            "total_cost",
            message,
            show_measurements=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
        )

    # 7. Estimación inicial.
    for message in (
        "La estimación inicial está lejos del mínimo y tiene un coste alto.",
        "El valor de F permite comparar objetivamente dos estimaciones.",
    ):
        add(
            initial,
            "initial_estimate",
            message,
            show_measurements=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
        )

    # 8. Derivada y dirección de descenso.
    for message in (
        "La derivada indica la pendiente local de la función de coste.",
        "Como la pendiente es positiva, reducir x disminuye el coste.",
    ):
        add(
            initial,
            "gradient",
            message,
            show_measurements=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
            show_gradient=True,
        )

    # 9. Descenso no ponderado.
    visible_history = []
    for item in unweighted_descent[1:]:
        visible_history.append(item)
        add(
            item["evaluation"],
            "unweighted_descent",
            (
                f"Descenso no ponderado: iteración {item['iteration']}. "
                "La estimación avanza en sentido opuesto al gradiente."
            ),
            mode="unweighted",
            iteration=item["iteration"],
            learning_rate=0.08,
            history=visible_history,
            show_measurements=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
            show_gradient=True,
        )

    # 10. Mínimo no ponderado exacto.
    for message in (
        "El mínimo analítico no ponderado coincide con la media aritmética.",
        "En el mínimo, la derivada es cero y la segunda derivada es positiva.",
    ):
        add(
            unweighted_final,
            "unweighted_minimum",
            message,
            mode="unweighted",
            history=unweighted_descent,
            show_measurements=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
            show_unweighted_minimum=True,
            show_factor_graph=True,
        )

    # 11. Pesos e información.
    for message in (
        "Ahora cada medición recibe un peso w_k relacionado con su confianza.",
        "Una medición con peso grande deforma más la función de coste total.",
        "El coste ponderado es F_w(x) = Σ_k w_k (x - z_k)^2.",
    ):
        add(
            unweighted_final,
            "weights",
            message,
            mode="weighted",
            show_measurements=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
            show_unweighted_minimum=True,
            show_weights=True,
            show_factor_graph=True,
        )

    # 12. Curva ponderada.
    for message in (
        "Las parábolas ponderadas tienen curvaturas diferentes.",
        "La medición z1 posee el peso mayor y atrae con más fuerza la solución.",
        "La suma ponderada produce una curva y un mínimo distintos.",
    ):
        add(
            unweighted_final,
            "weighted_curve",
            message,
            mode="weighted",
            show_measurements=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
            show_unweighted_minimum=True,
            show_weights=True,
            show_weighted_curve=True,
            show_factor_graph=True,
        )

    # 13. Desplazamiento del mínimo.
    for message in (
        "El mínimo se desplaza desde la media hasta la media ponderada.",
        "La solución ponderada queda más cerca de las mediciones más fiables.",
    ):
        add(
            unweighted_final,
            "minimum_shift",
            message,
            mode="weighted",
            show_measurements=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
            show_unweighted_minimum=True,
            show_weights=True,
            show_weighted_curve=True,
            show_weighted_minimum=True,
            show_factor_graph=True,
        )

    # 14. Descenso ponderado.
    visible_history = []
    for item in weighted_descent[1:]:
        visible_history.append(item)
        add(
            item["evaluation"],
            "weighted_descent",
            (
                f"Descenso ponderado: iteración {item['iteration']}. "
                "La estimación se acerca a la media ponderada."
            ),
            mode="weighted",
            iteration=item["iteration"],
            learning_rate=0.025,
            history=visible_history,
            show_measurements=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
            show_gradient=True,
            show_unweighted_minimum=True,
            show_weights=True,
            show_weighted_curve=True,
            show_weighted_minimum=True,
            show_factor_graph=True,
        )

    # 15. Mínimo ponderado exacto.
    for message in (
        "La media ponderada es el mínimo analítico del nuevo coste.",
        "En este punto, las contribuciones siguen siendo positivas pero su suma es mínima.",
    ):
        add(
            weighted_final,
            "weighted_minimum",
            message,
            mode="weighted",
            history=weighted_descent,
            show_measurements=True,
            show_residuals=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
            show_unweighted_minimum=True,
            show_weights=True,
            show_weighted_curve=True,
            show_weighted_minimum=True,
            show_factor_graph=True,
        )

    # 16. Conexión con grafos de restricciones.
    for message in (
        "En el grafo, cada factor aporta un residuo y un coste local.",
        "Graph SLAM repite esta suma para muchas variables y muchas aristas.",
    ):
        add(
            weighted_final,
            "graph_connection",
            message,
            mode="weighted",
            history=weighted_descent,
            show_measurements=True,
            show_residuals=True,
            show_individual_costs=True,
            show_total_cost=True,
            show_estimate=True,
            show_unweighted_minimum=True,
            show_weights=True,
            show_weighted_curve=True,
            show_weighted_minimum=True,
            show_factor_graph=True,
            show_pose_graph_connection=True,
        )

    # 17. Resumen final e imagen estática.
    add(
        weighted_final,
        "summary",
        "Residuos → costes locales → suma global → mínimo de la función.",
        mode="weighted",
        history=weighted_descent,
        show_measurements=True,
        show_residuals=True,
        show_individual_costs=True,
        show_total_cost=True,
        show_estimate=True,
        show_unweighted_minimum=True,
        show_weights=True,
        show_weighted_curve=True,
        show_weighted_minimum=True,
        show_factor_graph=True,
        show_pose_graph_connection=True,
    )

    for index, state in enumerate(states, start=1):
        state["step"] = index
        state["total_steps"] = len(states)

    return {
        "states": states,
        "initial": initial,
        "unweighted_final": unweighted_final,
        "weighted_final": weighted_final,
        "curves": curves,
        "unweighted_descent": unweighted_descent,
        "weighted_descent": weighted_descent,
    }


# ---------------------------------------------------------------------------
# Validaciones y salida
# ---------------------------------------------------------------------------


def validar_grafo_funcion_coste(graph):
    """Valida la variable, los factores y los atributos de cada medición."""

    if not isinstance(graph, nx.Graph) or graph.is_directed():
        raise TypeError("El ejemplo debe utilizar un nx.Graph no dirigido.")

    expected_nodes = {"x", "f1", "f2", "f3"}

    if set(graph.nodes()) != expected_nodes:
        raise ValueError("El grafo debe contener x y tres factores de medición.")

    if graph.nodes["x"].get("node_type") != "variable":
        raise ValueError("El nodo x debe representar una variable.")

    for index, (measurement, weight) in enumerate(
        zip(MEDICIONES, PESOS),
        start=1,
    ):
        factor = f"f{index}"

        if graph.nodes[factor].get("node_type") != "measurement_factor":
            raise ValueError(f"{factor} debe ser un factor de medición.")

        if not graph.has_edge("x", factor):
            raise ValueError(f"Falta la conexión entre x y {factor}.")

        edge = graph.edges["x", factor]

        if not np.isclose(edge.get("measurement"), measurement):
            raise ValueError(f"La medición almacenada en {factor} es incorrecta.")

        if not np.isclose(edge.get("weight"), weight):
            raise ValueError(f"El peso almacenado en {factor} es incorrecto.")

        if edge.get("residual_model") != "x - z_k":
            raise ValueError("El modelo de residuo debe ser x - z_k.")


def validar_resultados(graph, result):
    """Comprueba mínimos, derivadas, curvas y descenso de coste."""

    validar_grafo_funcion_coste(graph)
    initial = result["initial"]
    unweighted_final = result["unweighted_final"]
    weighted_final = result["weighted_final"]
    curves = result["curves"]

    expected_mean = float(np.mean(MEDICIONES))
    expected_weighted_mean = float(np.dot(PESOS, MEDICIONES) / np.sum(PESOS))

    if not np.isclose(unweighted_final["estimate"], expected_mean):
        raise ValueError("El mínimo no ponderado debe coincidir con la media.")

    if not np.isclose(weighted_final["estimate"], expected_weighted_mean):
        raise ValueError(
            "El mínimo ponderado debe coincidir con la media ponderada."
        )

    if abs(unweighted_final["unweighted_derivative"]) > 1e-10:
        raise ValueError("La derivada no ponderada debe anularse en el mínimo.")

    if abs(weighted_final["weighted_derivative"]) > 1e-10:
        raise ValueError("La derivada ponderada debe anularse en el mínimo.")

    if unweighted_final["unweighted_second_derivative"] <= 0.0:
        raise ValueError("La segunda derivada no ponderada debe ser positiva.")

    if weighted_final["weighted_second_derivative"] <= 0.0:
        raise ValueError("La segunda derivada ponderada debe ser positiva.")

    if initial["unweighted_cost"] <= unweighted_final["unweighted_cost"]:
        raise ValueError("El descenso no ponderado debe reducir el coste.")

    if unweighted_final["weighted_cost"] <= weighted_final["weighted_cost"]:
        raise ValueError("La ponderación debe desplazar el mínimo del coste.")

    for history_key in ("unweighted_descent", "weighted_descent"):
        costs = [item["active_cost"] for item in result[history_key]]

        if any(next_cost >= cost for cost, next_cost in zip(costs, costs[1:])):
            raise ValueError(
                f"El coste de {history_key} debe disminuir estrictamente."
            )

    if np.any(curves["unweighted_curve"] < -1e-12):
        raise ValueError("La curva no ponderada no puede ser negativa.")

    if np.any(curves["weighted_curve"] < -1e-12):
        raise ValueError("La curva ponderada no puede ser negativa.")

    domain = curves["domain"]
    numerical_unweighted = float(domain[np.argmin(curves["unweighted_curve"])])
    numerical_weighted = float(domain[np.argmin(curves["weighted_curve"])])
    tolerance = float(domain[1] - domain[0]) * 1.1

    if abs(numerical_unweighted - expected_mean) > tolerance:
        raise ValueError("El mínimo numérico no ponderado no coincide con el analítico.")

    if abs(numerical_weighted - expected_weighted_mean) > tolerance:
        raise ValueError("El mínimo numérico ponderado no coincide con el analítico.")

    if len(result["states"]) < 50:
        raise ValueError("La animación debe contener al menos cincuenta estados.")

    graph.nodes["x"]["initial_estimate"] = float(ESTIMACION_INICIAL)
    graph.nodes["x"]["unweighted_estimate"] = expected_mean
    graph.nodes["x"]["weighted_estimate"] = expected_weighted_mean
    graph.nodes["x"]["estimate"] = expected_weighted_mean

    for index, factor in enumerate(("f1", "f2", "f3")):
        edge = graph.edges["x", factor]
        edge["initial_residual"] = float(initial["residuals"][index])
        edge["final_residual"] = float(weighted_final["residuals"][index])
        edge["initial_cost"] = float(
            initial["weighted_individual_costs"][index]
        )
        edge["final_cost"] = float(
            weighted_final["weighted_individual_costs"][index]
        )


def imprimir_resumen(graph, result):
    """Imprime los resultados numéricos principales."""

    initial = result["initial"]
    unweighted = result["unweighted_final"]
    weighted = result["weighted_final"]

    print("\n=== Funciones de coste y mínimos cuadrados ===")
    print("Mediciones:", MEDICIONES)
    print("Pesos:", PESOS)
    print(f"Estimación inicial: {ESTIMACION_INICIAL:.6f}")
    print(f"Media aritmética: {unweighted['estimate']:.6f}")
    print(f"Media ponderada: {weighted['estimate']:.6f}")
    print(f"Coste inicial no ponderado: {initial['unweighted_cost']:.6f}")
    print(f"Coste mínimo no ponderado: {unweighted['unweighted_cost']:.6f}")
    print(f"Coste inicial ponderado: {initial['weighted_cost']:.6f}")
    print(f"Coste mínimo ponderado: {weighted['weighted_cost']:.6f}")
    print("Residuos en el mínimo ponderado:", weighted["residuals"])
    print(
        "Contribuciones ponderadas finales:",
        weighted["weighted_individual_costs"],
    )
    print(f"Nodos del grafo de factores: {graph.number_of_nodes()}")
    print(f"Factores de medición: {graph.number_of_edges()}")
    print(f"Estados de animación: {len(result['states'])}")


def main():
    validar_datos(MEDICIONES, PESOS)

    graph = crear_grafo_funcion_coste(MEDICIONES, PESOS)
    validar_grafo_funcion_coste(graph)

    result = crear_estados_animacion(graph)
    validar_resultados(graph, result)
    validar_grafo_funcion_coste(graph)

    imprimir_resumen(graph, result)

    animator = GraphAnimator(
        figsize=(18, 10),
        interval=560,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "05_optimizacion"
        / "03_funcion_coste_minimos_cuadrados.png"
    )

    animator.animate_cost_function_least_squares(
        graph=graph,
        states=result["states"],
        title="Funciones de coste y mínimos cuadrados",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
