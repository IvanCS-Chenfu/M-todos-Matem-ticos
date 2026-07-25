from heapq import heappop, heappush
from pathlib import Path
import sys

import networkx as nx


CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


FREE = 0
OBSTACLE = 1
MOVEMENTS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


def crear_mapa_ocupacion():
    """
    Crea un mapa de ocupación con aspecto de pequeño almacén.

    Convención:
    - 0: casilla libre y transitable;
    - 1: obstáculo estático.

    Las paredes verticales representan estanterías o divisiones. Sus huecos
    funcionan como pasos entre pasillos y permiten varias rutas alternativas.
    """

    rows = 15
    columns = 25

    occupancy_grid = [
        [FREE for _ in range(columns)]
        for _ in range(rows)
    ]

    # Borde exterior no transitable.
    for row in range(rows):
        occupancy_grid[row][0] = OBSTACLE
        occupancy_grid[row][columns - 1] = OBSTACLE

    for column in range(columns):
        occupancy_grid[0][column] = OBSTACLE
        occupancy_grid[rows - 1][column] = OBSTACLE

    # Estanterías verticales con pasos alternos.
    for row in range(1, 13):
        if row not in {2, 7, 11}:
            occupancy_grid[row][5] = OBSTACLE

    for row in range(2, 14):
        if row not in {4, 9, 12}:
            occupancy_grid[row][10] = OBSTACLE

    for row in range(1, 13):
        if row not in {3, 8, 11}:
            occupancy_grid[row][15] = OBSTACLE

    for row in range(2, 14):
        if row not in {5, 10, 13}:
            occupancy_grid[row][20] = OBSTACLE

    # Obstáculos horizontales cortos que obligan a rodear algunas zonas.
    for column in range(6, 10):
        if column != 8:
            occupancy_grid[5][column] = OBSTACLE

    for column in range(11, 15):
        if column != 13:
            occupancy_grid[10][column] = OBSTACLE

    # Cajas y zonas ocupadas adicionales.
    occupied_cells = {
        (2, 2),
        (2, 3),
        (3, 3),
        (12, 2),
        (11, 3),
        (6, 18),
        (7, 18),
        (8, 18),
        (3, 22),
        (4, 22),
    }

    for row, column in occupied_cells:
        occupancy_grid[row][column] = OBSTACLE

    start = (1, 1)
    goal = (13, 23)

    occupancy_grid[start[0]][start[1]] = FREE
    occupancy_grid[goal[0]][goal[1]] = FREE

    return occupancy_grid, start, goal


def crear_grafo_desde_grid(occupancy_grid):
    """
    Convierte el mapa de ocupación en un grafo no dirigido.

    - Cada casilla libre se convierte en un vértice ``(fila, columna)``.
    - Dos vértices se conectan si sus casillas son adyacentes en vertical u
      horizontal.
    - Cada movimiento tiene coste unitario.
    """

    graph = nx.Graph()
    rows = len(occupancy_grid)
    columns = len(occupancy_grid[0])

    for row in range(rows):
        for column in range(columns):
            if occupancy_grid[row][column] == FREE:
                graph.add_node((row, column))

    for row, column in list(graph.nodes()):
        for row_delta, column_delta in MOVEMENTS:
            neighbor = (
                row + row_delta,
                column + column_delta,
            )

            if neighbor in graph:
                graph.add_edge(
                    (row, column),
                    neighbor,
                    weight=1,
                )

    return graph


def heuristica_manhattan(cell, goal):
    """Calcula la distancia Manhattan entre dos casillas."""

    return (
        abs(cell[0] - goal[0])
        + abs(cell[1] - goal[1])
    )


def reconstruir_camino(predecessors, goal):
    """Reconstruye un camino siguiendo los predecesores de A*."""

    path = [goal]
    current = goal

    while current in predecessors:
        current = predecessors[current]
        path.append(current)

    path.reverse()
    return path


def obtener_vecinos_ordenados(graph, node):
    """
    Devuelve los vecinos en orden estable para que la demo sea reproducible.
    """

    return sorted(graph.neighbors(node))


def calcular_ruta_astar(graph, start, goal):
    """
    Ejecuta A* sin registrar estados.

    Se utiliza para comprobar candidatos a obstáculo dinámico antes de crear
    la secuencia visual definitiva.
    """

    if start not in graph or goal not in graph:
        return None

    priority_queue = []
    insertion_order = 0

    g_scores = {start: 0}
    predecessors = {}
    closed_nodes = set()

    heappush(
        priority_queue,
        (
            heuristica_manhattan(start, goal),
            insertion_order,
            start,
        ),
    )

    while priority_queue:
        f_value, _, current = heappop(priority_queue)

        if current in closed_nodes:
            continue

        expected_f = (
            g_scores[current]
            + heuristica_manhattan(current, goal)
        )

        if f_value != expected_f:
            continue

        if current == goal:
            return reconstruir_camino(predecessors, goal)

        closed_nodes.add(current)

        for neighbor in obtener_vecinos_ordenados(graph, current):
            if neighbor in closed_nodes:
                continue

            candidate_g = g_scores[current] + graph[current][neighbor]["weight"]

            if candidate_g < g_scores.get(neighbor, float("inf")):
                g_scores[neighbor] = candidate_g
                predecessors[neighbor] = current
                insertion_order += 1

                candidate_f = (
                    candidate_g
                    + heuristica_manhattan(neighbor, goal)
                )

                heappush(
                    priority_queue,
                    (
                        candidate_f,
                        insertion_order,
                        neighbor,
                    ),
                )

    return None


def crear_estado_busqueda(
    phase,
    current,
    open_nodes,
    closed_nodes,
    g_scores,
    goal,
    expanded_count,
    message,
):
    """Crea una copia independiente de un estado visual de A*."""

    if current is None:
        current_g = "—"
        current_h = "—"
        current_f = "—"
    else:
        current_g = g_scores.get(current, "—")
        current_h = heuristica_manhattan(current, goal)
        current_f = current_g + current_h

    return {
        "phase": phase,
        "current": current,
        "open_nodes": set(open_nodes),
        "closed_nodes": set(closed_nodes),
        "current_g": current_g,
        "current_h": current_h,
        "current_f": current_f,
        "expanded_count": expanded_count,
        "active_path": [],
        "previous_path": [],
        "traversed_path": [],
        "dynamic_obstacle": None,
        "robot": None,
        "message": message,
    }


def ejecutar_astar_animado(
    graph,
    start,
    goal,
    phase,
    state_stride=2,
):
    """
    Ejecuta A* y registra estados adecuados para la animación.

    Se conserva un estado cada ``state_stride`` expansiones para que la
    animación sea fluida sin generar cientos de imágenes casi idénticas.
    """

    if start not in graph:
        raise ValueError("La casilla inicial no pertenece al grafo.")

    if goal not in graph:
        raise ValueError("La casilla objetivo no pertenece al grafo.")

    priority_queue = []
    insertion_order = 0

    g_scores = {start: 0}
    predecessors = {}
    open_nodes = {start}
    closed_nodes = set()
    states = []
    expanded_count = 0

    heappush(
        priority_queue,
        (
            heuristica_manhattan(start, goal),
            insertion_order,
            start,
        ),
    )

    states.append(
        crear_estado_busqueda(
            phase=phase,
            current=None,
            open_nodes=open_nodes,
            closed_nodes=closed_nodes,
            g_scores=g_scores,
            goal=goal,
            expanded_count=expanded_count,
            message=(
                f"A* comienza en {start}. La prioridad es f(n)=g(n)+h(n)."
            ),
        )
    )

    while priority_queue:
        f_value, _, current = heappop(priority_queue)

        if current in closed_nodes:
            continue

        expected_f = (
            g_scores[current]
            + heuristica_manhattan(current, goal)
        )

        if f_value != expected_f:
            continue

        open_nodes.discard(current)

        if current == goal:
            path = reconstruir_camino(predecessors, goal)

            final_state = crear_estado_busqueda(
                phase=phase,
                current=current,
                open_nodes=open_nodes,
                closed_nodes=closed_nodes,
                g_scores=g_scores,
                goal=goal,
                expanded_count=expanded_count,
                message=(
                    f"A* alcanza el objetivo y reconstruye una ruta de "
                    f"{len(path) - 1} movimientos."
                ),
            )
            final_state["active_path"] = list(path)
            states.append(final_state)

            return {
                "path": path,
                "states": states,
                "expanded_count": expanded_count,
                "closed_nodes": set(closed_nodes),
                "open_nodes": set(open_nodes),
                "g_scores": dict(g_scores),
            }

        closed_nodes.add(current)
        expanded_count += 1

        for neighbor in obtener_vecinos_ordenados(graph, current):
            if neighbor in closed_nodes:
                continue

            edge_cost = graph[current][neighbor].get("weight", 1)
            candidate_g = g_scores[current] + edge_cost

            if candidate_g < g_scores.get(neighbor, float("inf")):
                g_scores[neighbor] = candidate_g
                predecessors[neighbor] = current
                open_nodes.add(neighbor)
                insertion_order += 1

                candidate_f = (
                    candidate_g
                    + heuristica_manhattan(neighbor, goal)
                )

                heappush(
                    priority_queue,
                    (
                        candidate_f,
                        insertion_order,
                        neighbor,
                    ),
                )

        if (
            expanded_count == 1
            or expanded_count % state_stride == 0
        ):
            states.append(
                crear_estado_busqueda(
                    phase=phase,
                    current=current,
                    open_nodes=open_nodes,
                    closed_nodes=closed_nodes,
                    g_scores=g_scores,
                    goal=goal,
                    expanded_count=expanded_count,
                    message=(
                        f"Se expande {current}: g={g_scores[current]}, "
                        f"h={heuristica_manhattan(current, goal)} y "
                        f"f={expected_f}."
                    ),
                )
            )

    raise nx.NetworkXNoPath(
        f"No existe una ruta entre {start} y {goal}."
    )


def seleccionar_obstaculo_dinamico(
    graph,
    initial_path,
    travelled_steps,
    goal,
):
    """
    Selecciona una casilla futura de la ruta cuya ocupación obligue a desviar
    al robot, pero sin dejar el objetivo inaccesible.
    """

    robot_position = initial_path[travelled_steps]
    initial_cost = len(initial_path) - 1

    candidate_cells = initial_path[
        travelled_steps + 3:
        -3
    ]

    for candidate in candidate_cells:
        blocked_graph = graph.copy()
        blocked_graph.remove_node(candidate)

        alternative_path = calcular_ruta_astar(
            blocked_graph,
            robot_position,
            goal,
        )

        if alternative_path is None:
            continue

        total_cost = travelled_steps + len(alternative_path) - 1

        if total_cost > initial_cost:
            return candidate, robot_position

    raise RuntimeError(
        "No se encontró un obstáculo dinámico que produjera una "
        "replanificación válida."
    )


def copiar_estado(state):
    """Copia las estructuras mutables de un estado de animación."""

    copied = dict(state)

    for key in (
        "open_nodes",
        "closed_nodes",
    ):
        copied[key] = set(state.get(key, set()))

    for key in (
        "active_path",
        "previous_path",
        "traversed_path",
    ):
        copied[key] = list(state.get(key, []))

    return copied


def repetir_estado(states, state, repetitions):
    """Añade varias copias de un estado para crear una pausa visual."""

    for _ in range(repetitions):
        states.append(copiar_estado(state))


def completar_metricas_estado(
    state,
    graph,
    initial_path,
    replanned_path,
    travelled_cost,
    total_travel_cost,
):
    """Añade métricas comunes a un estado ya construido."""

    state["graph_nodes"] = graph.number_of_nodes()
    state["graph_edges"] = graph.number_of_edges()
    state["initial_path_cost"] = len(initial_path) - 1
    state["replanned_path_cost"] = len(replanned_path) - 1
    state["travelled_cost"] = travelled_cost
    state["total_travel_cost"] = total_travel_cost

    return state


def crear_estados_navegacion(
    graph,
    start,
    goal,
    travelled_steps=9,
):
    """
    Construye toda la demostración:

    1. planificación inicial;
    2. movimiento parcial;
    3. aparición de un obstáculo;
    4. replanificación;
    5. llegada al objetivo.
    """

    initial_result = ejecutar_astar_animado(
        graph=graph,
        start=start,
        goal=goal,
        phase="initial_search",
        state_stride=2,
    )
    initial_path = initial_result["path"]

    dynamic_obstacle, robot_position = seleccionar_obstaculo_dinamico(
        graph=graph,
        initial_path=initial_path,
        travelled_steps=travelled_steps,
        goal=goal,
    )

    replanning_graph = graph.copy()
    replanning_graph.remove_node(dynamic_obstacle)

    replanning_result = ejecutar_astar_animado(
        graph=replanning_graph,
        start=robot_position,
        goal=goal,
        phase="replanning",
        state_stride=2,
    )
    replanned_path = replanning_result["path"]

    total_travel_cost = (
        travelled_steps
        + len(replanned_path)
        - 1
    )

    states = []

    map_state = {
        "phase": "map",
        "current": None,
        "open_nodes": set(),
        "closed_nodes": set(),
        "current_g": "—",
        "current_h": "—",
        "current_f": "—",
        "expanded_count": 0,
        "active_path": [],
        "previous_path": [],
        "traversed_path": [],
        "dynamic_obstacle": None,
        "robot": start,
        "message": (
            "El mapa de ocupación se convierte en un grafo de casillas "
            "libres conectadas en cuatro direcciones."
        ),
    }
    completar_metricas_estado(
        map_state,
        graph,
        initial_path,
        replanned_path,
        travelled_cost=0,
        total_travel_cost=total_travel_cost,
    )
    repetir_estado(states, map_state, 8)

    for state in initial_result["states"]:
        state["robot"] = start
        completar_metricas_estado(
            state,
            graph,
            initial_path,
            replanned_path,
            travelled_cost=0,
            total_travel_cost=total_travel_cost,
        )
        states.append(state)

    initial_path_state = copiar_estado(initial_result["states"][-1])
    initial_path_state["phase"] = "initial_path"
    initial_path_state["current"] = None
    initial_path_state["current_g"] = "—"
    initial_path_state["current_h"] = "—"
    initial_path_state["current_f"] = "—"
    initial_path_state["active_path"] = list(initial_path)
    initial_path_state["robot"] = start
    initial_path_state["message"] = (
        f"Ruta inicial encontrada: {len(initial_path) - 1} movimientos. "
        "El robot comienza a seguirla."
    )
    completar_metricas_estado(
        initial_path_state,
        graph,
        initial_path,
        replanned_path,
        travelled_cost=0,
        total_travel_cost=total_travel_cost,
    )
    repetir_estado(states, initial_path_state, 8)

    for step in range(1, travelled_steps + 1):
        movement_state = copiar_estado(initial_path_state)
        movement_state["phase"] = "movement"
        movement_state["robot"] = initial_path[step]
        movement_state["traversed_path"] = initial_path[: step + 1]
        movement_state["message"] = (
            f"El robot avanza por la ruta inicial: paso "
            f"{step} de {travelled_steps}."
        )
        completar_metricas_estado(
            movement_state,
            graph,
            initial_path,
            replanned_path,
            travelled_cost=step,
            total_travel_cost=total_travel_cost,
        )
        repetir_estado(states, movement_state, 2)

    obstacle_state = copiar_estado(states[-1])
    obstacle_state["phase"] = "obstacle"
    obstacle_state["dynamic_obstacle"] = dynamic_obstacle
    obstacle_state["previous_path"] = list(initial_path)
    obstacle_state["active_path"] = []
    obstacle_state["message"] = (
        f"Los sensores detectan un obstáculo nuevo en {dynamic_obstacle}. "
        "La ruta inicial deja de ser válida."
    )
    completar_metricas_estado(
        obstacle_state,
        graph,
        initial_path,
        replanned_path,
        travelled_cost=travelled_steps,
        total_travel_cost=total_travel_cost,
    )
    repetir_estado(states, obstacle_state, 10)

    traversed_before_replanning = initial_path[: travelled_steps + 1]

    for state in replanning_result["states"]:
        state["phase"] = "replanning"
        state["robot"] = robot_position
        state["dynamic_obstacle"] = dynamic_obstacle
        state["previous_path"] = list(initial_path)
        state["traversed_path"] = list(traversed_before_replanning)
        state["message"] = (
            "A* vuelve a ejecutarse desde la posición actual del robot. "
            + state["message"]
        )
        completar_metricas_estado(
            state,
            graph,
            initial_path,
            replanned_path,
            travelled_cost=travelled_steps,
            total_travel_cost=total_travel_cost,
        )
        states.append(state)

    replanned_state = copiar_estado(replanning_result["states"][-1])
    replanned_state["phase"] = "replanned_path"
    replanned_state["current"] = None
    replanned_state["current_g"] = "—"
    replanned_state["current_h"] = "—"
    replanned_state["current_f"] = "—"
    replanned_state["robot"] = robot_position
    replanned_state["dynamic_obstacle"] = dynamic_obstacle
    replanned_state["previous_path"] = list(initial_path)
    replanned_state["active_path"] = list(replanned_path)
    replanned_state["traversed_path"] = list(traversed_before_replanning)
    replanned_state["message"] = (
        f"Ruta alternativa encontrada: {len(replanned_path) - 1} "
        "movimientos desde la posición actual."
    )
    completar_metricas_estado(
        replanned_state,
        graph,
        initial_path,
        replanned_path,
        travelled_cost=travelled_steps,
        total_travel_cost=total_travel_cost,
    )
    repetir_estado(states, replanned_state, 10)

    for step in range(1, len(replanned_path)):
        final_movement_state = copiar_estado(replanned_state)
        final_movement_state["phase"] = "final_movement"
        final_movement_state["robot"] = replanned_path[step]
        final_movement_state["traversed_path"] = (
            list(traversed_before_replanning)
            + replanned_path[1: step + 1]
        )
        final_movement_state["message"] = (
            f"El robot sigue la ruta replanificada: "
            f"{step} de {len(replanned_path) - 1} movimientos."
        )
        completar_metricas_estado(
            final_movement_state,
            graph,
            initial_path,
            replanned_path,
            travelled_cost=travelled_steps + step,
            total_travel_cost=total_travel_cost,
        )
        states.append(final_movement_state)

    finished_state = copiar_estado(states[-1])
    finished_state["phase"] = "finished"
    finished_state["robot"] = goal
    finished_state["message"] = (
        f"Objetivo alcanzado. Coste inicial previsto: "
        f"{len(initial_path) - 1}; coste real tras replanificar: "
        f"{total_travel_cost}."
    )
    completar_metricas_estado(
        finished_state,
        graph,
        initial_path,
        replanned_path,
        travelled_cost=total_travel_cost,
        total_travel_cost=total_travel_cost,
    )
    repetir_estado(states, finished_state, 10)

    return {
        "states": states,
        "initial_path": initial_path,
        "replanned_path": replanned_path,
        "dynamic_obstacle": dynamic_obstacle,
        "robot_position_when_blocked": robot_position,
        "travelled_steps": travelled_steps,
        "total_travel_cost": total_travel_cost,
        "initial_expanded": initial_result["expanded_count"],
        "replanned_expanded": replanning_result["expanded_count"],
    }


def validar_camino(graph, path):
    """Comprueba que un camino esté formado por aristas válidas."""

    if not path:
        raise ValueError("El camino no puede estar vacío.")

    for origin, destination in zip(path[:-1], path[1:]):
        if not graph.has_edge(origin, destination):
            raise ValueError(
                f"El movimiento {origin}→{destination} no es válido."
            )


def validar_demostracion(
    occupancy_grid,
    graph,
    start,
    goal,
    result,
):
    """Valida las propiedades esenciales de la demostración."""

    if occupancy_grid[start[0]][start[1]] != FREE:
        raise ValueError("La casilla inicial debe ser transitable.")

    if occupancy_grid[goal[0]][goal[1]] != FREE:
        raise ValueError("La casilla objetivo debe ser transitable.")

    if not nx.is_connected(graph):
        raise ValueError("El grafo de casillas libres debe ser conexo.")

    initial_path = result["initial_path"]
    replanned_path = result["replanned_path"]
    dynamic_obstacle = result["dynamic_obstacle"]
    robot_position = result["robot_position_when_blocked"]

    validar_camino(graph, initial_path)

    if initial_path[0] != start or initial_path[-1] != goal:
        raise ValueError("La ruta inicial no conecta inicio y objetivo.")

    if dynamic_obstacle not in initial_path:
        raise ValueError(
            "El obstáculo dinámico debe bloquear la ruta inicial."
        )

    if replanned_path[0] != robot_position:
        raise ValueError(
            "La ruta replanificada debe comenzar en la posición actual."
        )

    if replanned_path[-1] != goal:
        raise ValueError(
            "La ruta replanificada debe terminar en el objetivo."
        )

    if dynamic_obstacle in replanned_path:
        raise ValueError(
            "La ruta replanificada no puede atravesar el obstáculo."
        )

    replanning_graph = graph.copy()
    replanning_graph.remove_node(dynamic_obstacle)
    validar_camino(replanning_graph, replanned_path)


def imprimir_resumen(graph, result, start, goal):
    """Imprime los resultados principales de la demostración."""

    initial_cost = len(result["initial_path"]) - 1
    replanned_cost = len(result["replanned_path"]) - 1

    print("\n=== Navegación en grid con A* y replanificación ===")
    print(f"Inicio: {start}")
    print(f"Objetivo: {goal}")
    print(f"Vértices transitables: {graph.number_of_nodes()}")
    print(f"Aristas de movimiento: {graph.number_of_edges()}")
    print(f"Ruta inicial: {initial_cost} movimientos")
    print(f"Casillas expandidas inicialmente: {result['initial_expanded']}")
    print(
        "Posición del robot al detectar el obstáculo: "
        f"{result['robot_position_when_blocked']}"
    )
    print(f"Obstáculo dinámico: {result['dynamic_obstacle']}")
    print(
        "Ruta replanificada desde la posición actual: "
        f"{replanned_cost} movimientos"
    )
    print(f"Casillas expandidas al replanificar: {result['replanned_expanded']}")
    print(f"Coste real total: {result['total_travel_cost']} movimientos")
    print(f"Estados de animación: {len(result['states'])}")


def main():
    occupancy_grid, start, goal = crear_mapa_ocupacion()
    graph = crear_grafo_desde_grid(occupancy_grid)

    result = crear_estados_navegacion(
        graph=graph,
        start=start,
        goal=goal,
        travelled_steps=9,
    )

    validar_demostracion(
        occupancy_grid=occupancy_grid,
        graph=graph,
        start=start,
        goal=goal,
        result=result,
    )

    imprimir_resumen(
        graph=graph,
        result=result,
        start=start,
        goal=goal,
    )

    animator = GraphAnimator(
        figsize=(17, 10),
        interval=160,
    )

    final_image_path = (
        GRAFOS_DIR
        / "assets"
        / "04_robotica"
        / "02_navegacion_astar_replanificacion.png"
    )

    animator.animate_grid_astar_replanning(
        occupancy_grid=occupancy_grid,
        states=result["states"],
        start=start,
        goal=goal,
        title="Navegación robótica con A* y replanificación dinámica",
        final_image_path=final_image_path,
        repeat=False,
    )


if __name__ == "__main__":
    main()
