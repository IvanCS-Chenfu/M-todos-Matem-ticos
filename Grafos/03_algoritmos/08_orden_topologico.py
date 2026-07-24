from pathlib import Path
import heapq
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


def obtener_tareas_orden_topologico():
    """
    Devuelve las tareas representadas por los vértices.
    """

    return {
        "A": "Encender sistemas",
        "B": "Comprobar batería",
        "C": "Cargar mapa",
        "D": "Inicializar sensores",
        "E": "Verificar comunicaciones",
        "F": "Localizar robot",
        "G": "Planificar ruta",
        "H": "Activar navegación",
        "I": "Detectar objeto",
        "J": "Planificar agarre",
        "K": "Transportar objeto",
        "L": "Depositar objeto",
    }


def obtener_nombres_cortos_orden_topologico():
    """
    Devuelve etiquetas compactas para el dibujo del grafo.
    """

    return {
        "A": "Encender",
        "B": "Batería",
        "C": "Mapa",
        "D": "Sensores",
        "E": "Comunic.",
        "F": "Localizar",
        "G": "Planificar",
        "H": "Navegación",
        "I": "Detectar",
        "J": "Agarre",
        "K": "Transportar",
        "L": "Depositar",
    }


def obtener_dependencias_orden_topologico(
    incluir_ciclo=False,
):
    """
    Devuelve las dependencias dirigidas del ejemplo.

    La arista u→v significa que la tarea u debe terminar antes de v.

    El modo opcional añade L→F y crea el ciclo:

        F→G→H→I→J→K→L→F
    """

    dependencias = [
        ("A", "D"),
        ("A", "E"),

        ("B", "E"),
        ("B", "H"),

        ("C", "F"),
        ("C", "G"),

        ("D", "F"),
        ("D", "I"),

        ("E", "H"),

        ("F", "G"),

        ("G", "H"),

        ("H", "I"),
        ("H", "K"),

        ("I", "J"),

        ("J", "K"),

        ("K", "L"),
    ]

    if incluir_ciclo:
        dependencias.append(("L", "F"))

    return dependencias


def crear_grafo_orden_topologico(
    incluir_ciclo=False,
):
    """
    Crea el grafo dirigido de tareas.

    Por defecto es un DAG con varias fuentes iniciales:

        A, B y C

    En el modo opcional se introduce una dependencia circular.
    """

    grafo = nx.DiGraph()

    tareas = obtener_tareas_orden_topologico()
    nombres_cortos = obtener_nombres_cortos_orden_topologico()

    for node, task in tareas.items():
        grafo.add_node(
            node,
            task=task,
            short_name=nombres_cortos[node],
        )

    grafo.add_edges_from(
        obtener_dependencias_orden_topologico(
            incluir_ciclo=incluir_ciclo,
        )
    )

    return grafo


def obtener_posiciones():
    """
    Define posiciones manuales para mostrar las ramas y los niveles.

    La cadena final se pliega hacia la parte inferior para evitar una
    figura excesivamente ancha.
    """

    return {
        "A": (-6.0, 3.0),
        "B": (-6.0, 0.0),
        "C": (-6.0, -3.0),

        "D": (-3.2, 2.0),
        "E": (-3.2, -0.7),

        "F": (-0.3, 2.2),
        "G": (2.2, 2.2),
        "H": (4.8, 1.1),

        "I": (4.8, -1.5),
        "J": (2.2, -2.4),
        "K": (-0.3, -2.4),
        "L": (-3.0, -2.4),
    }


def validar_grafo_dirigido(grafo):
    """
    Comprueba que el ejemplo sea un grafo dirigido no vacío.
    """

    if not grafo.is_directed():
        raise ValueError(
            "El ordenamiento topológico requiere un grafo dirigido."
        )

    if grafo.number_of_nodes() == 0:
        raise ValueError(
            "El grafo no puede estar vacío."
        )


def crear_cola_visible(disponibles):
    """
    Devuelve la cola de prioridad ordenada para mostrarla.

    El montículo interno también usa el identificador del vértice como
    criterio de desempate, por lo que el resultado es determinista.
    """

    return sorted(disponibles)


def detectar_ciclo_dirigido(grafo):
    """
    Devuelve una lista de aristas de un ciclo dirigido, si existe.
    """

    try:
        cycle = nx.find_cycle(
            grafo,
            orientation="original",
        )
    except nx.NetworkXNoCycle:
        return []

    return [
        (origin, destination)
        for origin, destination, _ in cycle
    ]


def crear_estado_orden_topologico(
    grafo,
    initial_in_degrees,
    in_degrees,
    available_nodes,
    processed_nodes,
    order,
    levels,
    processed_edges,
    message,
    phase,
    is_unique,
    multiple_choice_count,
    current_node=None,
    active_edge=None,
    active_successor=None,
    action=None,
    old_in_degree=None,
    new_in_degree=None,
    blocked_nodes=None,
    cycle_edges=None,
):
    """
    Crea una copia independiente de un estado del algoritmo de Kahn.
    """

    return {
        "task_names": nx.get_node_attributes(
            grafo,
            "task",
        ),
        "short_names": nx.get_node_attributes(
            grafo,
            "short_name",
        ),
        "initial_in_degrees": dict(initial_in_degrees),
        "in_degrees": dict(in_degrees),
        "available_nodes": list(
            crear_cola_visible(available_nodes)
        ),
        "processed_nodes": set(processed_nodes),
        "order": list(order),
        "levels": dict(levels),
        "processed_edges": set(processed_edges),
        "current_node": current_node,
        "active_edge": active_edge,
        "active_successor": active_successor,
        "action": action,
        "old_in_degree": old_in_degree,
        "new_in_degree": new_in_degree,
        "blocked_nodes": set(blocked_nodes or set()),
        "cycle_edges": set(cycle_edges or set()),
        "phase": phase,
        "is_unique": is_unique,
        "multiple_choice_count": multiple_choice_count,
        "message": message,
    }


def ejecutar_kahn_por_pasos(grafo):
    """
    Ejecuta el algoritmo de Kahn y registra todos sus pasos.

    Se utiliza un montículo para seleccionar alfabéticamente la siguiente
    fuente. Además de producir el orden, se calculan niveles:

        nivel[v] =
            0 para las fuentes iniciales
            1 + máximo nivel de sus predecesores para el resto
    """

    validar_grafo_dirigido(grafo)

    nodes = sorted(grafo.nodes())

    initial_in_degrees = {
        node: grafo.in_degree(node)
        for node in nodes
    }
    in_degrees = dict(initial_in_degrees)

    available_heap = [
        node
        for node in nodes
        if in_degrees[node] == 0
    ]
    heapq.heapify(available_heap)

    levels = {
        node: 0
        for node in available_heap
    }

    processed_nodes = set()
    processed_edges = set()
    order = []
    states = []

    is_unique = True
    multiple_choice_count = 0

    states.append(
        crear_estado_orden_topologico(
            grafo=grafo,
            initial_in_degrees=initial_in_degrees,
            in_degrees=in_degrees,
            available_nodes=available_heap,
            processed_nodes=processed_nodes,
            order=order,
            levels=levels,
            processed_edges=processed_edges,
            message=(
                "Inicialización: A, B y C tienen grado de entrada cero "
                "y pueden comenzar. La cola usa prioridad alfabética."
            ),
            phase="initial",
            is_unique=is_unique,
            multiple_choice_count=multiple_choice_count,
        )
    )

    while available_heap:
        choices_before_selection = len(available_heap)

        if choices_before_selection > 1:
            is_unique = False
            multiple_choice_count += 1

        current = heapq.heappop(available_heap)
        processed_nodes.add(current)
        order.append(current)

        selection_message = (
            f"Se selecciona {current}: "
            f"{grafo.nodes[current]['task']}."
        )

        if choices_before_selection > 1:
            selection_message += (
                f" Había {choices_before_selection} tareas disponibles; "
                "por tanto, el orden topológico no es único."
            )

        states.append(
            crear_estado_orden_topologico(
                grafo=grafo,
                initial_in_degrees=initial_in_degrees,
                in_degrees=in_degrees,
                available_nodes=available_heap,
                processed_nodes=processed_nodes,
                order=order,
                levels=levels,
                processed_edges=processed_edges,
                current_node=current,
                action="select",
                message=selection_message,
                phase="processing",
                is_unique=is_unique,
                multiple_choice_count=multiple_choice_count,
            )
        )

        for successor in sorted(grafo.successors(current)):
            old_in_degree = in_degrees[successor]
            new_in_degree = old_in_degree - 1
            in_degrees[successor] = new_in_degree

            processed_edges.add(
                (current, successor)
            )

            candidate_level = levels.get(current, 0) + 1
            levels[successor] = max(
                levels.get(successor, 0),
                candidate_level,
            )

            if new_in_degree == 0:
                heapq.heappush(
                    available_heap,
                    successor,
                )
                action = "unlock"
                message = (
                    f"{current} satisface la última dependencia de "
                    f"{successor}. {successor} entra en la cola."
                )
            else:
                action = "decrement"
                message = (
                    f"Se satisface {current}→{successor}, pero "
                    f"{successor} conserva {new_in_degree} "
                    "dependencia(s) pendiente(s)."
                )

            states.append(
                crear_estado_orden_topologico(
                    grafo=grafo,
                    initial_in_degrees=initial_in_degrees,
                    in_degrees=in_degrees,
                    available_nodes=available_heap,
                    processed_nodes=processed_nodes,
                    order=order,
                    levels=levels,
                    processed_edges=processed_edges,
                    current_node=current,
                    active_edge=(current, successor),
                    active_successor=successor,
                    action=action,
                    old_in_degree=old_in_degree,
                    new_in_degree=new_in_degree,
                    message=message,
                    phase="processing",
                    is_unique=is_unique,
                    multiple_choice_count=multiple_choice_count,
                )
            )

        states.append(
            crear_estado_orden_topologico(
                grafo=grafo,
                initial_in_degrees=initial_in_degrees,
                in_degrees=in_degrees,
                available_nodes=available_heap,
                processed_nodes=processed_nodes,
                order=order,
                levels=levels,
                processed_edges=processed_edges,
                current_node=current,
                action="finish_node",
                message=(
                    f"Finaliza {current}. El orden parcial contiene "
                    f"{len(order)} tarea(s)."
                ),
                phase="processing",
                is_unique=is_unique,
                multiple_choice_count=multiple_choice_count,
            )
        )

    has_cycle = len(order) != grafo.number_of_nodes()

    if has_cycle:
        blocked_nodes = set(grafo.nodes()) - processed_nodes
        cycle_edges = detectar_ciclo_dirigido(
            grafo.subgraph(blocked_nodes).copy()
        )

        states.append(
            crear_estado_orden_topologico(
                grafo=grafo,
                initial_in_degrees=initial_in_degrees,
                in_degrees=in_degrees,
                available_nodes=[],
                processed_nodes=processed_nodes,
                order=order,
                levels=levels,
                processed_edges=processed_edges,
                blocked_nodes=blocked_nodes,
                cycle_edges=cycle_edges,
                action="cycle",
                message=(
                    "La cola está vacía antes de procesar todas las tareas. "
                    "Las tareas moradas están bloqueadas por un ciclo "
                    "dirigido y no existe un orden topológico completo."
                ),
                phase="cycle",
                is_unique=False,
                multiple_choice_count=multiple_choice_count,
            )
        )
    else:
        states.append(
            crear_estado_orden_topologico(
                grafo=grafo,
                initial_in_degrees=initial_in_degrees,
                in_degrees=in_degrees,
                available_nodes=[],
                processed_nodes=processed_nodes,
                order=order,
                levels=levels,
                processed_edges=processed_edges,
                action="finished",
                message=(
                    "Kahn ha procesado todas las tareas. El orden es válido "
                    "y los niveles muestran las ramas potencialmente "
                    "paralelas."
                ),
                phase="finished",
                is_unique=is_unique,
                multiple_choice_count=multiple_choice_count,
            )
        )

    return {
        "states": states,
        "order": order,
        "levels": levels,
        "initial_in_degrees": initial_in_degrees,
        "final_in_degrees": in_degrees,
        "processed_edges": processed_edges,
        "has_cycle": has_cycle,
        "cycle_edges": (
            detectar_ciclo_dirigido(grafo)
            if has_cycle
            else []
        ),
        "blocked_nodes": (
            sorted(set(grafo.nodes()) - processed_nodes)
            if has_cycle
            else []
        ),
        "is_unique": is_unique and not has_cycle,
        "multiple_choice_count": multiple_choice_count,
    }


def ejecutar_orden_topologico_dfs(grafo):
    """
    Obtiene un segundo orden topológico mediante DFS.

    Estados:
    - blanco: no visitado;
    - gris: activo en la pila;
    - negro: finalizado.

    Una arista hacia un vértice gris detecta un ciclo dirigido.
    """

    color = {
        node: "blanco"
        for node in grafo.nodes()
    }
    postorder = []
    cycle_edge = None

    def visitar(node):
        nonlocal cycle_edge

        color[node] = "gris"

        for successor in sorted(grafo.successors(node)):
            if color[successor] == "gris":
                cycle_edge = (node, successor)
                return False

            if (
                color[successor] == "blanco"
                and not visitar(successor)
            ):
                return False

        color[node] = "negro"
        postorder.append(node)

        return True

    for node in sorted(grafo.nodes()):
        if color[node] == "blanco":
            if not visitar(node):
                return {
                    "has_cycle": True,
                    "cycle_edge": cycle_edge,
                    "order": [],
                    "postorder": postorder,
                }

    order = list(reversed(postorder))

    return {
        "has_cycle": False,
        "cycle_edge": None,
        "order": order,
        "postorder": postorder,
    }


def validar_orden_topologico(grafo, order):
    """
    Comprueba que el orden incluya todos los vértices y respete cada arista.
    """

    if len(order) != grafo.number_of_nodes():
        return False

    if set(order) != set(grafo.nodes()):
        return False

    position = {
        node: index
        for index, node in enumerate(order)
    }

    return all(
        position[origin] < position[destination]
        for origin, destination in grafo.edges()
    )


def calcular_niveles_con_networkx(grafo):
    """
    Calcula niveles mediante las generaciones topológicas de NetworkX.
    """

    levels = {}

    for level, generation in enumerate(
        nx.topological_generations(grafo)
    ):
        for node in generation:
            levels[node] = level

    return levels


def comprobar_con_networkx(
    grafo,
    resultado_kahn,
    resultado_dfs,
):
    """
    Compara la implementación con NetworkX.

    En el DAG normal también se compara el orden lexicográfico producido
    por Kahn y los niveles topológicos.
    """

    is_dag_networkx = nx.is_directed_acyclic_graph(grafo)

    checks = {
        "cycle_detection_matches": (
            resultado_kahn["has_cycle"]
            == (not is_dag_networkx)
        ),
        "dfs_cycle_detection_matches": (
            resultado_dfs["has_cycle"]
            == (not is_dag_networkx)
        ),
    }

    if is_dag_networkx:
        networkx_order = list(
            nx.lexicographical_topological_sort(
                grafo,
                key=lambda node: node,
            )
        )
        networkx_levels = calcular_niveles_con_networkx(
            grafo
        )

        checks.update(
            {
                "kahn_order_is_valid": validar_orden_topologico(
                    grafo,
                    resultado_kahn["order"],
                ),
                "dfs_order_is_valid": validar_orden_topologico(
                    grafo,
                    resultado_dfs["order"],
                ),
                "lexicographical_order_matches": (
                    resultado_kahn["order"]
                    == networkx_order
                ),
                "levels_match": (
                    resultado_kahn["levels"]
                    == networkx_levels
                ),
                "all_vertices_processed": (
                    len(resultado_kahn["order"])
                    == grafo.number_of_nodes()
                ),
            }
        )
    else:
        checks.update(
            {
                "kahn_order_is_partial": (
                    len(resultado_kahn["order"])
                    < grafo.number_of_nodes()
                ),
                "blocked_nodes_exist": bool(
                    resultado_kahn["blocked_nodes"]
                ),
                "cycle_edges_exist": bool(
                    resultado_kahn["cycle_edges"]
                ),
            }
        )

    return checks


def agrupar_por_nivel(levels):
    """
    Agrupa los vértices que pueden habilitarse en el mismo nivel.
    """

    groups = {}

    for node, level in levels.items():
        groups.setdefault(level, []).append(node)

    return {
        level: sorted(nodes)
        for level, nodes in sorted(groups.items())
    }


def imprimir_resultado_orden_topologico(
    grafo,
    resultado_kahn,
    resultado_dfs,
    comprobacion,
):
    """
    Muestra los resultados principales por terminal.
    """

    print("\n=== DAG y ordenamiento topológico ===")

    print("\nGrados de entrada iniciales:")

    for node in sorted(grafo.nodes()):
        print(
            f"  {node}: "
            f"{resultado_kahn['initial_in_degrees'][node]}"
        )

    print("\nResultado de Kahn:")

    if resultado_kahn["has_cycle"]:
        print("  Existe un ciclo dirigido.")
        print(
            "  Orden parcial: "
            + " -> ".join(resultado_kahn["order"])
        )
        print(
            "  Tareas bloqueadas: "
            + ", ".join(resultado_kahn["blocked_nodes"])
        )
        print(
            "  Ciclo detectado: "
            + " -> ".join(
                origin
                for origin, _ in resultado_kahn["cycle_edges"]
            )
        )
    else:
        print(
            "  Orden: "
            + " -> ".join(resultado_kahn["order"])
        )
        print(
            "  ¿Orden único?: "
            f"{resultado_kahn['is_unique']}"
        )
        print(
            "  Pasos con varias opciones: "
            f"{resultado_kahn['multiple_choice_count']}"
        )

        print("\nNiveles:")

        for level, nodes in agrupar_por_nivel(
            resultado_kahn["levels"]
        ).items():
            print(
                f"  Nivel {level}: "
                + ", ".join(nodes)
            )

    print("\nResultado de DFS:")

    if resultado_dfs["has_cycle"]:
        print(
            "  Ciclo detectado mediante arista activa: "
            f"{resultado_dfs['cycle_edge']}"
        )
    else:
        print(
            "  Orden válido: "
            + " -> ".join(resultado_dfs["order"])
        )

    print("\nComprobación con NetworkX:")

    for name, value in comprobacion.items():
        print(f"  {name}: {value}")


def main():
    # Cambiar a True añade L→F y crea una dependencia circular.
    incluir_ciclo = False

    grafo = crear_grafo_orden_topologico(
        incluir_ciclo=incluir_ciclo,
    )
    posiciones = obtener_posiciones()

    resultado_kahn = ejecutar_kahn_por_pasos(grafo)
    resultado_dfs = ejecutar_orden_topologico_dfs(grafo)

    comprobacion = comprobar_con_networkx(
        grafo=grafo,
        resultado_kahn=resultado_kahn,
        resultado_dfs=resultado_dfs,
    )

    imprimir_resultado_orden_topologico(
        grafo=grafo,
        resultado_kahn=resultado_kahn,
        resultado_dfs=resultado_dfs,
        comprobacion=comprobacion,
    )

    animador = GraphAnimator(
        figsize=(16, 10),
        interval=750,
    )

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "03_algoritmos"
        / "08_orden_topologico.png"
    )

    animacion = animador.animate_topological_sort(
        graph=grafo,
        pos=posiciones,
        states=resultado_kahn["states"],
        title="DAG y ordenamiento topológico con Kahn",
        final_image_path=output_path,
        repeat=False,
    )

    _ = animacion


if __name__ == "__main__":
    main()
