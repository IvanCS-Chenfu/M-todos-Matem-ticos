from pathlib import Path
import sys

import networkx as nx


CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


PENDING = "pendiente"
AVAILABLE = "disponible"
RUNNING = "en_ejecucion"
COMPLETED = "completada"
FAILED = "fallida"

SUCCESS = "exito"
FAILURE = "fallo"
ALWAYS = "siempre"


TASKS = {
    "INI": {
        "name": "Inicio de la misión",
        "short_name": "Inicio",
        "category": "sistema",
        "duration": 0,
        "resource": "coordinador",
    },
    "SEN": {
        "name": "Activar sensores",
        "short_name": "Sensores",
        "category": "sistema",
        "duration": 2,
        "resource": "sistema",
    },
    "MAP": {
        "name": "Cargar mapa",
        "short_name": "Mapa",
        "category": "planificacion",
        "duration": 3,
        "resource": "computo",
    },
    "AP0": {
        "name": "Abrir pinza",
        "short_name": "Abrir pinza",
        "category": "manipulacion",
        "duration": 2,
        "resource": "brazo",
    },
    "DET": {
        "name": "Detectar objeto",
        "short_name": "Detectar",
        "category": "percepcion",
        "duration": 3,
        "resource": "camara",
    },
    "EST": {
        "name": "Estimar pose del objeto",
        "short_name": "Estimar pose",
        "category": "percepcion",
        "duration": 2,
        "resource": "computo",
    },
    "PLA": {
        "name": "Planificar acercamiento",
        "short_name": "Plan acerc.",
        "category": "planificacion",
        "duration": 2,
        "resource": "computo",
    },
    "ACE": {
        "name": "Acercarse al objeto",
        "short_name": "Acercarse",
        "category": "navegacion",
        "duration": 4,
        "resource": "base_movil",
    },
    "VAL": {
        "name": "Verificar alcance",
        "short_name": "Verif. alcance",
        "category": "verificacion",
        "duration": 1,
        "resource": "verificacion",
    },
    "AGR1": {
        "name": "Agarrar objeto · intento 1",
        "short_name": "Agarre 1",
        "category": "manipulacion",
        "duration": 2,
        "resource": "brazo",
    },
    "VAG1": {
        "name": "Verificar agarre · intento 1",
        "short_name": "Verif. 1",
        "category": "verificacion",
        "duration": 1,
        "resource": "verificacion",
        "programmed_result": FAILURE,
    },
    "REC1": {
        "name": "Abrir pinza de recuperación",
        "short_name": "Abrir recup.",
        "category": "recuperacion",
        "duration": 1,
        "resource": "brazo",
    },
    "REC2": {
        "name": "Reposicionar brazo",
        "short_name": "Reposicionar",
        "category": "recuperacion",
        "duration": 2,
        "resource": "brazo",
    },
    "AGR2": {
        "name": "Agarrar objeto · intento 2",
        "short_name": "Agarre 2",
        "category": "manipulacion",
        "duration": 2,
        "resource": "brazo",
    },
    "VAG2": {
        "name": "Verificar agarre · intento 2",
        "short_name": "Verif. 2",
        "category": "verificacion",
        "duration": 1,
        "resource": "verificacion",
    },
    "PLT": {
        "name": "Planificar transporte",
        "short_name": "Plan transp.",
        "category": "planificacion",
        "duration": 2,
        "resource": "computo",
    },
    "TRA": {
        "name": "Transportar objeto",
        "short_name": "Transportar",
        "category": "navegacion",
        "duration": 5,
        "resource": "base_movil",
    },
    "POS": {
        "name": "Posicionarse para entregar",
        "short_name": "Posicionar",
        "category": "navegacion",
        "duration": 2,
        "resource": "base_movil",
    },
    "SOL": {
        "name": "Soltar objeto",
        "short_name": "Soltar",
        "category": "manipulacion",
        "duration": 1,
        "resource": "brazo",
    },
    "VEN": {
        "name": "Verificar entrega",
        "short_name": "Verif. entrega",
        "category": "verificacion",
        "duration": 1,
        "resource": "verificacion",
    },
    "FIN": {
        "name": "Fin de la misión",
        "short_name": "Fin",
        "category": "sistema",
        "duration": 0,
        "resource": "coordinador",
    },
}


DEPENDENCIES = (
    ("INI", "SEN", ALWAYS),
    ("INI", "MAP", ALWAYS),
    ("INI", "AP0", ALWAYS),
    ("SEN", "DET", ALWAYS),
    ("DET", "EST", ALWAYS),
    ("EST", "PLA", ALWAYS),
    ("MAP", "PLA", ALWAYS),
    ("PLA", "ACE", ALWAYS),
    ("ACE", "VAL", ALWAYS),
    ("VAL", "AGR1", ALWAYS),
    ("AP0", "AGR1", ALWAYS),
    ("AGR1", "VAG1", ALWAYS),
    ("VAG1", "PLT", SUCCESS),
    ("VAG1", "REC1", FAILURE),
    ("REC1", "REC2", ALWAYS),
    ("REC2", "AGR2", ALWAYS),
    ("AGR2", "VAG2", ALWAYS),
    ("VAG2", "PLT", SUCCESS),
    ("PLT", "TRA", ALWAYS),
    ("TRA", "POS", ALWAYS),
    ("POS", "SOL", ALWAYS),
    ("SOL", "VEN", ALWAYS),
    ("VEN", "FIN", ALWAYS),
)


POSITIONS = {
    "INI": (0.0, 0.0),
    "SEN": (1.2, 1.65),
    "MAP": (1.2, 0.0),
    "AP0": (1.2, -1.65),
    "DET": (2.5, 1.65),
    "EST": (3.8, 1.65),
    "PLA": (5.0, 0.85),
    "ACE": (6.3, 0.85),
    "VAL": (7.6, 0.85),
    "AGR1": (8.9, 0.0),
    "VAG1": (10.2, 0.0),
    "REC1": (10.8, -1.55),
    "REC2": (12.1, -1.55),
    "AGR2": (13.4, -1.55),
    "VAG2": (14.7, -1.55),
    "PLT": (15.4, 0.0),
    "TRA": (16.8, 0.0),
    "POS": (18.2, 0.0),
    "SOL": (19.6, 0.0),
    "VEN": (21.0, 0.0),
    "FIN": (22.4, 0.0),
}


RESOURCE_ORDER = (
    "sistema",
    "computo",
    "camara",
    "base_movil",
    "brazo",
    "verificacion",
)


RESOURCE_LABELS = {
    "sistema": "Sistema",
    "computo": "Cómputo",
    "camara": "Cámara",
    "base_movil": "Base móvil",
    "brazo": "Brazo",
    "verificacion": "Verificación",
}


def crear_grafo_mision():
    """Crea el DAG de una misión robótica de recogida y entrega."""

    graph = nx.DiGraph()

    for node, attributes in TASKS.items():
        node_attributes = dict(attributes)
        node_attributes.setdefault("programmed_result", SUCCESS)
        graph.add_node(node, **node_attributes)

    for origin, destination, condition in DEPENDENCIES:
        graph.add_edge(
            origin,
            destination,
            condition=condition,
            label=(
                ""
                if condition == ALWAYS
                else "éxito" if condition == SUCCESS else "fallo"
            ),
        )

    graph.graph["resource_order"] = list(RESOURCE_ORDER)
    graph.graph["resource_labels"] = dict(RESOURCE_LABELS)
    graph.graph["start_node"] = "INI"
    graph.graph["end_node"] = "FIN"

    return graph


def validar_grafo_tareas(graph):
    """Valida la estructura estática del grafo antes de ejecutarlo."""

    if not graph.is_directed():
        raise ValueError("La misión debe representarse con un grafo dirigido.")

    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph, orientation="original")
        raise ValueError(f"El grafo de tareas contiene un ciclo: {cycle}")

    start_node = graph.graph["start_node"]
    end_node = graph.graph["end_node"]

    if graph.in_degree(start_node) != 0:
        raise ValueError("La tarea INI no puede tener predecesores.")

    if graph.out_degree(end_node) != 0:
        raise ValueError("La tarea FIN no puede tener sucesores.")

    unreachable = [
        node
        for node in graph.nodes()
        if node != start_node and not nx.has_path(graph, start_node, node)
    ]

    if unreachable:
        raise ValueError(
            "Todas las tareas deben ser alcanzables desde INI: "
            + ", ".join(unreachable)
        )

    dead_ends = [
        node
        for node in graph.nodes()
        if node != end_node and not nx.has_path(graph, node, end_node)
    ]

    if dead_ends:
        raise ValueError(
            "Todas las tareas deben poder conducir a FIN: "
            + ", ".join(dead_ends)
        )

    for node, data in graph.nodes(data=True):
        if data.get("duration", -1) < 0:
            raise ValueError(f"La duración de {node} no puede ser negativa.")

        if not data.get("resource"):
            raise ValueError(f"La tarea {node} debe declarar un recurso.")

    valid_conditions = {ALWAYS, SUCCESS, FAILURE}

    for origin, destination, data in graph.edges(data=True):
        condition = data.get("condition", ALWAYS)

        if condition not in valid_conditions:
            raise ValueError(
                f"Condición desconocida en {origin}→{destination}: {condition}"
            )

    return {
        "topological_order": list(nx.topological_sort(graph)),
        "number_of_tasks": graph.number_of_nodes(),
        "number_of_dependencies": graph.number_of_edges(),
    }


def dependencias_satisfechas(
    graph,
    node,
    statuses,
    triggered_condition_edges,
):
    """Comprueba dependencias AND y activaciones condicionales OR."""

    incoming = list(graph.in_edges(node, data=True))

    if not incoming:
        return True

    always_edges = [
        (origin, destination)
        for origin, destination, data in incoming
        if data.get("condition", ALWAYS) == ALWAYS
    ]

    conditional_edges = [
        (origin, destination)
        for origin, destination, data in incoming
        if data.get("condition", ALWAYS) != ALWAYS
    ]

    always_ready = all(
        statuses[origin] == COMPLETED
        for origin, _ in always_edges
    )

    if not always_ready:
        return False

    if not conditional_edges:
        return True

    return any(
        edge in triggered_condition_edges
        for edge in conditional_edges
    )


def obtener_tareas_disponibles(
    graph,
    statuses,
    triggered_condition_edges,
):
    """Devuelve las tareas pendientes cuyas dependencias están satisfechas."""

    return [
        node
        for node in graph.nodes()
        if statuses[node] == PENDING
        and dependencias_satisfechas(
            graph=graph,
            node=node,
            statuses=statuses,
            triggered_condition_edges=triggered_condition_edges,
        )
    ]


def activar_dependencias_por_resultado(
    graph,
    node,
    result,
    satisfied_edges,
    triggered_condition_edges,
    inactive_condition_edges,
):
    """Actualiza aristas satisfechas, activadas e inactivas tras una tarea."""

    recent_edges = set()

    for _, successor, data in graph.out_edges(node, data=True):
        edge = (node, successor)
        condition = data.get("condition", ALWAYS)

        if condition == ALWAYS:
            if result == SUCCESS:
                satisfied_edges.add(edge)
                recent_edges.add(edge)
            continue

        if condition == result:
            triggered_condition_edges.add(edge)
            satisfied_edges.add(edge)
            recent_edges.add(edge)
        else:
            inactive_condition_edges.add(edge)

    return recent_edges


def crear_estado_mision(
    graph,
    time,
    statuses,
    remaining,
    start_times,
    end_times,
    results,
    satisfied_edges,
    triggered_condition_edges,
    inactive_condition_edges,
    execution_order,
    start_order,
    message,
    phase,
    focus_task=None,
    recent_edges=None,
    critical_nodes=None,
    critical_edges=None,
    critical_path=None,
    total_duration=None,
):
    """Crea una copia independiente de un estado de la misión."""

    available = {
        node
        for node, status in statuses.items()
        if status == AVAILABLE
    }
    running = {
        node
        for node, status in statuses.items()
        if status == RUNNING
    }
    completed = {
        node
        for node, status in statuses.items()
        if status == COMPLETED
    }
    failed = {
        node
        for node, status in statuses.items()
        if status == FAILED
    }

    return {
        "time": time,
        "phase": phase,
        "statuses": dict(statuses),
        "remaining": dict(remaining),
        "start_times": dict(start_times),
        "end_times": dict(end_times),
        "results": dict(results),
        "available": set(available),
        "running": set(running),
        "completed": set(completed),
        "failed": set(failed),
        "satisfied_edges": set(satisfied_edges),
        "triggered_condition_edges": set(triggered_condition_edges),
        "inactive_condition_edges": set(inactive_condition_edges),
        "recent_edges": set(recent_edges or set()),
        "execution_order": list(execution_order),
        "start_order": list(start_order),
        "focus_task": focus_task,
        "message": message,
        "critical_nodes": set(critical_nodes or set()),
        "critical_edges": set(critical_edges or set()),
        "critical_path": list(critical_path or []),
        "total_duration": total_duration,
        "task_count": graph.number_of_nodes(),
        "dependency_count": graph.number_of_edges(),
    }


def calcular_camino_critico_activo(
    graph,
    statuses,
    triggered_condition_edges,
):
    """Calcula el camino crítico del subgrafo realmente ejecutado."""

    executed_nodes = {
        node
        for node, status in statuses.items()
        if status in {COMPLETED, FAILED}
    }

    active_graph = nx.DiGraph()

    for node in executed_nodes:
        active_graph.add_node(node, **graph.nodes[node])

    for origin, destination, data in graph.edges(data=True):
        if origin not in executed_nodes or destination not in executed_nodes:
            continue

        condition = data.get("condition", ALWAYS)

        if condition == ALWAYS or (origin, destination) in triggered_condition_edges:
            active_graph.add_edge(origin, destination)

    order = list(nx.topological_sort(active_graph))
    earliest_finish = {}
    predecessor = {}

    for node in order:
        duration = active_graph.nodes[node].get("duration", 0)
        incoming = list(active_graph.predecessors(node))

        if not incoming:
            earliest_finish[node] = duration
            predecessor[node] = None
            continue

        best_predecessor = max(
            incoming,
            key=lambda candidate: earliest_finish[candidate],
        )

        earliest_finish[node] = (
            earliest_finish[best_predecessor] + duration
        )
        predecessor[node] = best_predecessor

    end_node = graph.graph["end_node"]

    if end_node not in predecessor:
        raise ValueError("FIN no pertenece al subgrafo ejecutado.")

    critical_path = [end_node]
    current = end_node

    while predecessor[current] is not None:
        current = predecessor[current]
        critical_path.append(current)

    critical_path.reverse()

    critical_edges = {
        (origin, destination)
        for origin, destination in zip(
            critical_path[:-1],
            critical_path[1:],
        )
    }

    return {
        "path": critical_path,
        "nodes": set(critical_path),
        "edges": critical_edges,
        "duration": earliest_finish[end_node],
        "active_graph": active_graph,
    }


def simular_ejecucion_mision(graph):
    """Simula la ejecución paralela, el fallo y la recuperación."""

    statuses = {
        node: PENDING
        for node in graph.nodes()
    }
    remaining = {
        node: graph.nodes[node].get("duration", 0)
        for node in graph.nodes()
    }
    start_times = {}
    end_times = {}
    results = {}

    satisfied_edges = set()
    triggered_condition_edges = set()
    inactive_condition_edges = set()

    execution_order = []
    start_order = []
    states = []
    time = 0

    states.append(
        crear_estado_mision(
            graph=graph,
            time=time,
            statuses=statuses,
            remaining=remaining,
            start_times=start_times,
            end_times=end_times,
            results=results,
            satisfied_edges=satisfied_edges,
            triggered_condition_edges=triggered_condition_edges,
            inactive_condition_edges=inactive_condition_edges,
            execution_order=execution_order,
            start_order=start_order,
            phase="validation",
            message=(
                "El plan es un DAG válido. Las flechas representan "
                "dependencias y condiciones de ejecución."
            ),
        )
    )

    start_node = graph.graph["start_node"]
    statuses[start_node] = AVAILABLE

    states.append(
        crear_estado_mision(
            graph=graph,
            time=time,
            statuses=statuses,
            remaining=remaining,
            start_times=start_times,
            end_times=end_times,
            results=results,
            satisfied_edges=satisfied_edges,
            triggered_condition_edges=triggered_condition_edges,
            inactive_condition_edges=inactive_condition_edges,
            execution_order=execution_order,
            start_order=start_order,
            phase="ready",
            focus_task=start_node,
            message="La misión comienza: INI es la única tarea disponible.",
        )
    )

    def complete_instantaneous(node, phase, message):
        statuses[node] = COMPLETED
        start_times[node] = time
        end_times[node] = time
        results[node] = SUCCESS
        start_order.append(node)
        execution_order.append(node)

        recent_edges = activar_dependencias_por_resultado(
            graph=graph,
            node=node,
            result=SUCCESS,
            satisfied_edges=satisfied_edges,
            triggered_condition_edges=triggered_condition_edges,
            inactive_condition_edges=inactive_condition_edges,
        )

        states.append(
            crear_estado_mision(
                graph=graph,
                time=time,
                statuses=statuses,
                remaining=remaining,
                start_times=start_times,
                end_times=end_times,
                results=results,
                satisfied_edges=satisfied_edges,
                triggered_condition_edges=triggered_condition_edges,
                inactive_condition_edges=inactive_condition_edges,
                execution_order=execution_order,
                start_order=start_order,
                phase=phase,
                focus_task=node,
                recent_edges=recent_edges,
                message=message,
            )
        )

    complete_instantaneous(
        start_node,
        phase="start",
        message=(
            "INI termina de forma instantánea y desbloquea tres tareas "
            "independientes."
        ),
    )

    end_node = graph.graph["end_node"]
    running_end_times = {}
    max_iterations = 200

    for _ in range(max_iterations):
        newly_available = obtener_tareas_disponibles(
            graph=graph,
            statuses=statuses,
            triggered_condition_edges=triggered_condition_edges,
        )

        for node in newly_available:
            statuses[node] = AVAILABLE

        if newly_available:
            states.append(
                crear_estado_mision(
                    graph=graph,
                    time=time,
                    statuses=statuses,
                    remaining=remaining,
                    start_times=start_times,
                    end_times=end_times,
                    results=results,
                    satisfied_edges=satisfied_edges,
                    triggered_condition_edges=triggered_condition_edges,
                    inactive_condition_edges=inactive_condition_edges,
                    execution_order=execution_order,
                    start_order=start_order,
                    phase="ready",
                    focus_task=newly_available[0],
                    message=(
                        "Nuevas tareas disponibles: "
                        + ", ".join(newly_available)
                        + "."
                    ),
                )
            )

        if statuses[end_node] == AVAILABLE:
            complete_instantaneous(
                end_node,
                phase="finished",
                message="FIN se activa: la misión ha terminado correctamente.",
            )
            break

        occupied_resources = {
            graph.nodes[node]["resource"]
            for node, status in statuses.items()
            if status == RUNNING
        }

        tasks_started_now = []

        for node in graph.nodes():
            if statuses[node] != AVAILABLE:
                continue

            resource = graph.nodes[node]["resource"]

            if resource in occupied_resources:
                continue

            duration = graph.nodes[node]["duration"]

            if duration == 0:
                continue

            statuses[node] = RUNNING
            start_times[node] = time
            running_end_times[node] = time + duration
            remaining[node] = duration
            start_order.append(node)
            tasks_started_now.append(node)
            occupied_resources.add(resource)

        if tasks_started_now:
            states.append(
                crear_estado_mision(
                    graph=graph,
                    time=time,
                    statuses=statuses,
                    remaining=remaining,
                    start_times=start_times,
                    end_times=end_times,
                    results=results,
                    satisfied_edges=satisfied_edges,
                    triggered_condition_edges=triggered_condition_edges,
                    inactive_condition_edges=inactive_condition_edges,
                    execution_order=execution_order,
                    start_order=start_order,
                    phase="running",
                    focus_task=tasks_started_now[0],
                    message=(
                        "Comienzan en paralelo: "
                        + ", ".join(tasks_started_now)
                        + "."
                    ),
                )
            )

        if not running_end_times:
            pending_nodes = [
                node
                for node, status in statuses.items()
                if status in {PENDING, AVAILABLE}
            ]

            raise RuntimeError(
                "La misión ha quedado bloqueada. Pendientes: "
                + ", ".join(pending_nodes)
            )

        next_time = min(running_end_times.values())

        for tick in range(time + 1, next_time + 1):
            for node in list(running_end_times):
                if statuses[node] == RUNNING:
                    remaining[node] = max(
                        running_end_times[node] - tick,
                        0,
                    )

            states.append(
                crear_estado_mision(
                    graph=graph,
                    time=tick,
                    statuses=statuses,
                    remaining=remaining,
                    start_times=start_times,
                    end_times=end_times,
                    results=results,
                    satisfied_edges=satisfied_edges,
                    triggered_condition_edges=triggered_condition_edges,
                    inactive_condition_edges=inactive_condition_edges,
                    execution_order=execution_order,
                    start_order=start_order,
                    phase="running",
                    message=(
                        f"Avanza el reloj de la misión hasta t={tick}. "
                        "Las tareas activas consumen tiempo y recursos."
                    ),
                )
            )

        time = next_time
        completed_now = sorted(
            [
                node
                for node, finish_time in running_end_times.items()
                if finish_time == time
            ],
            key=list(graph.nodes()).index,
        )

        for node in completed_now:
            running_end_times.pop(node)
            end_times[node] = time
            remaining[node] = 0

            result = graph.nodes[node].get(
                "programmed_result",
                SUCCESS,
            )
            results[node] = result

            if result == SUCCESS:
                statuses[node] = COMPLETED
                phase = "completed"
                result_text = "termina con éxito"
            else:
                statuses[node] = FAILED
                phase = "failure"
                result_text = "falla"

            execution_order.append(node)

            recent_edges = activar_dependencias_por_resultado(
                graph=graph,
                node=node,
                result=result,
                satisfied_edges=satisfied_edges,
                triggered_condition_edges=triggered_condition_edges,
                inactive_condition_edges=inactive_condition_edges,
            )

            if node == "VAG1":
                message = (
                    "VAG1 detecta un agarre incorrecto. La rama de éxito "
                    "queda inactiva y se activa la recuperación."
                )
            elif node == "VAG2":
                message = (
                    "VAG2 confirma el segundo agarre. Se desbloquea la "
                    "planificación del transporte."
                )
            else:
                message = (
                    f"{node} ({graph.nodes[node]['name']}) {result_text} "
                    f"en t={time}."
                )

            states.append(
                crear_estado_mision(
                    graph=graph,
                    time=time,
                    statuses=statuses,
                    remaining=remaining,
                    start_times=start_times,
                    end_times=end_times,
                    results=results,
                    satisfied_edges=satisfied_edges,
                    triggered_condition_edges=triggered_condition_edges,
                    inactive_condition_edges=inactive_condition_edges,
                    execution_order=execution_order,
                    start_order=start_order,
                    phase=phase,
                    focus_task=node,
                    recent_edges=recent_edges,
                    message=message,
                )
            )
    else:
        raise RuntimeError("La simulación superó el límite de iteraciones.")

    critical = calcular_camino_critico_activo(
        graph=graph,
        statuses=statuses,
        triggered_condition_edges=triggered_condition_edges,
    )

    total_duration = end_times[end_node]

    states.append(
        crear_estado_mision(
            graph=graph,
            time=total_duration,
            statuses=statuses,
            remaining=remaining,
            start_times=start_times,
            end_times=end_times,
            results=results,
            satisfied_edges=satisfied_edges,
            triggered_condition_edges=triggered_condition_edges,
            inactive_condition_edges=inactive_condition_edges,
            execution_order=execution_order,
            start_order=start_order,
            phase="summary",
            focus_task=end_node,
            critical_nodes=critical["nodes"],
            critical_edges=critical["edges"],
            critical_path=critical["path"],
            total_duration=total_duration,
            message=(
                "Misión completada. Se resalta el camino crítico activo, "
                "incluida la recuperación tras el primer agarre fallido."
            ),
        )
    )

    return {
        "states": states,
        "statuses": statuses,
        "results": results,
        "start_times": start_times,
        "end_times": end_times,
        "execution_order": execution_order,
        "start_order": start_order,
        "satisfied_edges": satisfied_edges,
        "triggered_condition_edges": triggered_condition_edges,
        "inactive_condition_edges": inactive_condition_edges,
        "critical_path": critical["path"],
        "critical_edges": critical["edges"],
        "total_duration": total_duration,
    }


def validar_ejecucion(graph, result):
    """Comprueba los resultados esenciales de la simulación."""

    statuses = result["statuses"]

    if statuses["FIN"] != COMPLETED:
        raise ValueError("La misión debe alcanzar FIN.")

    if statuses["VAG1"] != FAILED:
        raise ValueError("El primer control de agarre debe fallar.")

    if statuses["VAG2"] != COMPLETED:
        raise ValueError("El segundo control de agarre debe tener éxito.")

    if ("VAG1", "REC1") not in result["triggered_condition_edges"]:
        raise ValueError("El fallo debe activar la rama de recuperación.")

    if ("VAG1", "PLT") not in result["inactive_condition_edges"]:
        raise ValueError("La rama de éxito del primer intento debe desactivarse.")

    if ("VAG2", "PLT") not in result["triggered_condition_edges"]:
        raise ValueError("El segundo agarre debe desbloquear el transporte.")

    if result["total_duration"] <= 0:
        raise ValueError("La duración total debe ser positiva.")

    if result["critical_path"][0] != "INI":
        raise ValueError("El camino crítico debe comenzar en INI.")

    if result["critical_path"][-1] != "FIN":
        raise ValueError("El camino crítico debe terminar en FIN.")

    intervals_by_resource = {}

    for node, start_time in result["start_times"].items():
        duration = graph.nodes[node]["duration"]

        if duration == 0:
            continue

        resource = graph.nodes[node]["resource"]
        interval = (
            start_time,
            result["end_times"][node],
            node,
        )
        intervals_by_resource.setdefault(resource, []).append(interval)

    for resource, intervals in intervals_by_resource.items():
        intervals.sort()

        for first, second in zip(intervals[:-1], intervals[1:]):
            if first[1] > second[0]:
                raise ValueError(
                    f"El recurso {resource} se usa simultáneamente por "
                    f"{first[2]} y {second[2]}."
                )


def imprimir_resumen(graph, validation, result):
    """Imprime un resumen determinista de la misión."""

    print("\n=== Planificación de tareas para una misión robótica ===")
    print(f"Tareas del DAG: {validation['number_of_tasks']}")
    print(f"Dependencias: {validation['number_of_dependencies']}")
    print("Grafo acíclico: sí")
    print(
        "Tareas iniciales paralelas: "
        "SEN, MAP y AP0"
    )
    print("Fallo programado: VAG1")
    print("Rama de recuperación: REC1 → REC2 → AGR2 → VAG2")
    print(f"Duración total: {result['total_duration']} unidades")
    print("Orden de finalización:")
    print("  " + " → ".join(result["execution_order"]))
    print("Camino crítico activo:")
    print("  " + " → ".join(result["critical_path"]))
    print(f"Estados de animación: {len(result['states'])}")


def main():
    graph = crear_grafo_mision()
    validation = validar_grafo_tareas(graph)
    result = simular_ejecucion_mision(graph)
    validar_ejecucion(graph, result)

    imprimir_resumen(
        graph=graph,
        validation=validation,
        result=result,
    )

    animator = GraphAnimator(
        figsize=(18, 10),
        interval=430,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "04_robotica"
        / "04_planificacion_tareas_robot.png"
    )

    animator.animate_robot_task_planning(
        graph=graph,
        pos=POSITIONS,
        states=result["states"],
        title="Planificación de tareas de una misión robótica",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
