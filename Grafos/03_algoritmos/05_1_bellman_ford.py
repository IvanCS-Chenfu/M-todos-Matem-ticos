from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


def obtener_aristas_bellman_ford():
    """
    Devuelve las aristas en el orden usado por Bellman-Ford.

    El orden es deliberadamente poco favorable: las aristas cercanas al
    destino aparecen antes que las cercanas al origen. Esto obliga a que
    las mejoras se propaguen durante varias pasadas y permite observar
    claramente la diferencia con Dijkstra y A*.
    """

    return [
        ("G", "I", 4),
        ("H", "I", 2),
        ("G", "H", -1),

        ("F", "H", 4),
        ("F", "G", 2),

        ("E", "G", 5),
        ("E", "F", -1),

        ("D", "F", 3),

        ("C", "E", 5),

        ("B", "F", 7),
        ("B", "E", 4),

        ("C", "B", -2),

        ("A", "D", 5),
        ("A", "C", 4),
        ("A", "B", 6),
    ]


def crear_grafo_bellman_ford(
    incluir_ciclo_negativo=False,
):
    """
    Crea un grafo dirigido ponderado para estudiar Bellman-Ford.

    El caso predeterminado contiene pesos negativos, pero no un ciclo
    negativo. Si ``incluir_ciclo_negativo`` es verdadero, se añade la
    arista I→C con peso -5. Esa arista forma un ciclo negativo alcanzable
    desde A y permite probar la detección del algoritmo.
    """

    grafo = nx.DiGraph()

    aristas = obtener_aristas_bellman_ford()

    if incluir_ciclo_negativo:
        aristas = list(aristas) + [
            ("I", "C", -5),
        ]

    grafo.add_weighted_edges_from(aristas)

    # Se conserva el orden explícito porque Bellman-Ford depende de él
    # para la evolución intermedia, aunque no para el resultado final.
    grafo.graph["relaxation_edges"] = list(aristas)

    return grafo


def obtener_posiciones():
    """
    Define posiciones manuales de izquierda a derecha.

    La distribución conserva el estilo de los ejemplos de Dijkstra y A*,
    pero el grafo es dirigido porque contiene pesos negativos.
    """

    return {
        "A": (-6.0, 0.0),

        "B": (-3.0, 3.0),
        "C": (-3.0, 0.0),
        "D": (-3.0, -3.0),

        "E": (0.0, 2.5),
        "F": (0.0, -0.5),

        "G": (3.0, 2.5),
        "H": (3.0, -1.5),

        "I": (6.0, 0.0),
    }


def obtener_orden_relajacion(grafo):
    """
    Recupera el orden explícito de relajación del grafo.
    """

    aristas = grafo.graph.get("relaxation_edges")

    if aristas is None:
        return [
            (
                origen,
                destino,
                datos.get("weight", 1),
            )
            for origen, destino, datos
            in grafo.edges(data=True)
        ]

    return list(aristas)


def validar_origen_y_destino(
    grafo,
    origen,
    destino,
):
    """
    Comprueba que origen y destino existan en el grafo.
    """

    if origen not in grafo:
        raise ValueError(
            f"El origen {origen!r} no existe en el grafo."
        )

    if destino not in grafo:
        raise ValueError(
            f"El destino {destino!r} no existe en el grafo."
        )


def crear_estado_bellman_ford(
    distances,
    predecessors,
    edge_order,
    message,
    iteration,
    max_iterations,
    pass_changes,
    processed_edges,
    phase,
    active_edge=None,
    active_edge_index=None,
    current_source=None,
    current_target=None,
    action=None,
    candidate=None,
    old_distance=None,
    edge_weight=None,
    final_path=None,
    final_distances=False,
    negative_cycle=False,
):
    """
    Crea una copia independiente del estado actual de Bellman-Ford.
    """

    return {
        "distances": dict(distances),
        "predecessors": dict(predecessors),
        "edge_order": list(edge_order),
        "active_edge": active_edge,
        "active_edge_index": active_edge_index,
        "current_source": current_source,
        "current_target": current_target,
        "action": action,
        "candidate": candidate,
        "old_distance": old_distance,
        "edge_weight": edge_weight,
        "final_path": list(final_path or []),
        "iteration": iteration,
        "max_iterations": max_iterations,
        "pass_changes": pass_changes,
        "processed_edges": processed_edges,
        "phase": phase,
        "final_distances": final_distances,
        "negative_cycle": negative_cycle,
        "message": message,
    }


def reconstruir_camino(
    predecesores,
    origen,
    destino,
):
    """
    Reconstruye un camino siguiendo los predecesores hacia atrás.

    El conjunto ``visitados`` evita un bucle infinito si se llama por
    error después de detectar un ciclo en la cadena de predecesores.
    """

    if origen == destino:
        return [origen]

    if predecesores.get(destino) is None:
        return []

    camino_inverso = []
    visitados = set()
    actual = destino

    while actual is not None:
        if actual in visitados:
            return []

        visitados.add(actual)
        camino_inverso.append(actual)

        if actual == origen:
            break

        actual = predecesores.get(actual)

    if not camino_inverso or camino_inverso[-1] != origen:
        return []

    return list(reversed(camino_inverso))


def ejecutar_bellman_ford_por_pasos(
    grafo,
    origen,
    destino,
):
    """
    Ejecuta Bellman-Ford y registra todas las relajaciones.

    Se realizan como máximo |V|-1 pasadas. Si una pasada completa no
    produce cambios, el algoritmo termina anticipadamente. Después se
    examinan nuevamente las aristas para detectar un ciclo negativo
    alcanzable desde el origen.
    """

    validar_origen_y_destino(
        grafo=grafo,
        origen=origen,
        destino=destino,
    )

    edge_order = obtener_orden_relajacion(grafo)
    max_iterations = grafo.number_of_nodes() - 1

    distances = {
        node: float("inf")
        for node in grafo.nodes()
    }

    predecessors = {
        node: None
        for node in grafo.nodes()
    }

    distances[origen] = 0

    states = []
    passes_completed = 0
    early_stop = False

    states.append(
        crear_estado_bellman_ford(
            distances=distances,
            predecessors=predecessors,
            edge_order=edge_order,
            message=(
                f"Inicialización: d({origen}) = 0 y el resto "
                "de distancias se establece a infinito."
            ),
            iteration=0,
            max_iterations=max_iterations,
            pass_changes=0,
            processed_edges=0,
            phase="initial",
        )
    )

    for iteration in range(1, max_iterations + 1):
        pass_changes = 0

        states.append(
            crear_estado_bellman_ford(
                distances=distances,
                predecessors=predecessors,
                edge_order=edge_order,
                message=(
                    f"Comienza la pasada {iteration}. "
                    "Todas las aristas se examinarán en orden."
                ),
                iteration=iteration,
                max_iterations=max_iterations,
                pass_changes=pass_changes,
                processed_edges=0,
                phase="relaxation",
            )
        )

        for edge_index, (origin, target, weight) in enumerate(edge_order):
            old_distance = distances[target]

            if distances[origin] == float("inf"):
                candidate = float("inf")
                action = "unreachable_source"
                message = (
                    f"No se relaja {origin}→{target}: "
                    f"{origin} todavía no es alcanzable."
                )
            else:
                candidate = distances[origin] + weight

                if candidate < old_distance:
                    distances[target] = candidate
                    predecessors[target] = origin
                    pass_changes += 1
                    action = "improvement"
                    message = (
                        f"Mejora para {target}: su distancia pasa de "
                        f"{old_distance} a {candidate} y su "
                        f"predecesor pasa a ser {origin}."
                    )
                else:
                    action = "no_improvement"
                    message = (
                        f"La arista {origin}→{target} propone "
                        f"{candidate}; no mejora la distancia actual "
                        f"de {target} ({old_distance})."
                    )

            states.append(
                crear_estado_bellman_ford(
                    distances=distances,
                    predecessors=predecessors,
                    edge_order=edge_order,
                    message=message,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    pass_changes=pass_changes,
                    processed_edges=edge_index + 1,
                    phase="relaxation",
                    active_edge=(origin, target),
                    active_edge_index=edge_index,
                    current_source=origin,
                    current_target=target,
                    action=action,
                    candidate=candidate,
                    old_distance=old_distance,
                    edge_weight=weight,
                )
            )

        passes_completed = iteration

        states.append(
            crear_estado_bellman_ford(
                distances=distances,
                predecessors=predecessors,
                edge_order=edge_order,
                message=(
                    f"Finaliza la pasada {iteration} con "
                    f"{pass_changes} mejora(s)."
                ),
                iteration=iteration,
                max_iterations=max_iterations,
                pass_changes=pass_changes,
                processed_edges=len(edge_order),
                phase="relaxation",
            )
        )

        if pass_changes == 0:
            early_stop = True

            states.append(
                crear_estado_bellman_ford(
                    distances=distances,
                    predecessors=predecessors,
                    edge_order=edge_order,
                    message=(
                        "No se produjo ninguna mejora. Las distancias "
                        "han convergido y no son necesarias más pasadas."
                    ),
                    iteration=iteration,
                    max_iterations=max_iterations,
                    pass_changes=0,
                    processed_edges=len(edge_order),
                    phase="converged",
                )
            )
            break

    # Pasada adicional: no actualiza distancias, solo comprueba si alguna
    # arista todavía podría mejorar.
    negative_cycle = False
    negative_cycle_edge = None

    states.append(
        crear_estado_bellman_ford(
            distances=distances,
            predecessors=predecessors,
            edge_order=edge_order,
            message=(
                "Comienza la pasada adicional para comprobar si existe "
                "un ciclo negativo alcanzable desde el origen."
            ),
            iteration=passes_completed,
            max_iterations=max_iterations,
            pass_changes=0,
            processed_edges=0,
            phase="negative_cycle_check",
        )
    )

    for edge_index, (origin, target, weight) in enumerate(edge_order):
        old_distance = distances[target]

        if distances[origin] == float("inf"):
            candidate = float("inf")
            action = "unreachable_source"
            message = (
                f"{origin} no es alcanzable; la arista "
                f"{origin}→{target} no puede revelar una mejora."
            )
        else:
            candidate = distances[origin] + weight

            if candidate < old_distance:
                negative_cycle = True
                negative_cycle_edge = (origin, target)
                action = "negative_cycle"
                message = (
                    f"La arista {origin}→{target} todavía podría reducir "
                    f"d({target}) de {old_distance} a {candidate}. "
                    "Existe un ciclo negativo alcanzable."
                )
            else:
                action = "no_improvement"
                message = (
                    f"{origin}→{target} no produce una mejora adicional."
                )

        states.append(
            crear_estado_bellman_ford(
                distances=distances,
                predecessors=predecessors,
                edge_order=edge_order,
                message=message,
                iteration=passes_completed,
                max_iterations=max_iterations,
                pass_changes=0,
                processed_edges=edge_index + 1,
                phase="negative_cycle_check",
                active_edge=(origin, target),
                active_edge_index=edge_index,
                current_source=origin,
                current_target=target,
                action=action,
                candidate=candidate,
                old_distance=old_distance,
                edge_weight=weight,
                negative_cycle=negative_cycle,
            )
        )

        if negative_cycle:
            break

    if negative_cycle:
        path = []
        path_cost = None
        final_message = (
            "Bellman-Ford ha detectado un ciclo negativo alcanzable. "
            "Las distancias afectadas no tienen un mínimo finito."
        )
    else:
        path = reconstruir_camino(
            predecesores=predecessors,
            origen=origen,
            destino=destino,
        )
        path_cost = distances[destino]
        final_message = (
            "Bellman-Ford ha terminado sin detectar ciclos negativos. "
            f"Se resalta el camino mínimo desde {origen} hasta {destino}."
        )

    states.append(
        crear_estado_bellman_ford(
            distances=distances,
            predecessors=predecessors,
            edge_order=edge_order,
            message=final_message,
            iteration=passes_completed,
            max_iterations=max_iterations,
            pass_changes=0,
            processed_edges=len(edge_order),
            phase="finished",
            final_path=path,
            final_distances=not negative_cycle,
            negative_cycle=negative_cycle,
            active_edge=negative_cycle_edge,
            current_source=(
                negative_cycle_edge[0]
                if negative_cycle_edge
                else None
            ),
            current_target=(
                negative_cycle_edge[1]
                if negative_cycle_edge
                else None
            ),
        )
    )

    return {
        "states": states,
        "distances": distances,
        "predecessors": predecessors,
        "passes_completed": passes_completed,
        "early_stop": early_stop,
        "negative_cycle": negative_cycle,
        "negative_cycle_edge": negative_cycle_edge,
        "path": path,
        "path_cost": path_cost,
    }


def comprobar_con_networkx(
    grafo,
    origen,
    destino,
    resultado,
):
    """
    Compara la implementación manual con NetworkX.
    """

    try:
        distances_networkx, paths_networkx = (
            nx.single_source_bellman_ford(
                grafo,
                source=origen,
                weight="weight",
            )
        )

        networkx_negative_cycle = False
        networkx_path = paths_networkx.get(destino, [])
        networkx_cost = distances_networkx.get(
            destino,
            float("inf"),
        )

    except nx.NetworkXUnbounded:
        networkx_negative_cycle = True
        networkx_path = []
        networkx_cost = None
        distances_networkx = {}

    if resultado["negative_cycle"]:
        distances_match = networkx_negative_cycle
        path_match = networkx_negative_cycle
        cost_match = networkx_negative_cycle
    else:
        distances_match = all(
            resultado["distances"][node]
            == distances_networkx.get(node, float("inf"))
            for node in grafo.nodes()
        )
        path_match = resultado["path"] == networkx_path
        cost_match = resultado["path_cost"] == networkx_cost

    return {
        "negative_cycle_match": (
            resultado["negative_cycle"]
            == networkx_negative_cycle
        ),
        "distances_match": distances_match,
        "path_match": path_match,
        "cost_match": cost_match,
        "networkx_path": networkx_path,
        "networkx_cost": networkx_cost,
    }


def formatear_valor(value):
    """
    Formatea infinito y números enteros para la salida por terminal.
    """

    if value == float("inf"):
        return "∞"

    if value is None:
        return "no definido"

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def imprimir_resultado_bellman_ford(
    resultado,
    comprobacion,
):
    """
    Muestra por terminal los resultados principales.
    """

    print("\n=== Caminos mínimos con Bellman-Ford ===")

    print(
        "\nPasadas de relajación realizadas: "
        f"{resultado['passes_completed']}"
    )
    print(
        "Terminación anticipada: "
        f"{resultado['early_stop']}"
    )

    print("\nDistancias y predecesores:")

    for node in sorted(resultado["distances"]):
        distance = formatear_valor(
            resultado["distances"][node]
        )
        predecessor = resultado["predecessors"][node]
        predecessor_text = (
            "—"
            if predecessor is None
            else predecessor
        )

        print(
            f"  {node}: distancia={distance}, "
            f"predecesor={predecessor_text}"
        )

    print("\nDetección de ciclo negativo:")
    print(f"  {resultado['negative_cycle']}")

    if resultado["negative_cycle_edge"]:
        origin, target = resultado["negative_cycle_edge"]
        print(f"  Arista que lo revela: {origin} -> {target}")

    print("\nCamino mínimo hacia el destino:")

    if resultado["path"]:
        print("  " + " -> ".join(resultado["path"]))
        print(
            "  Coste total: "
            + formatear_valor(resultado["path_cost"])
        )
    elif resultado["negative_cycle"]:
        print(
            "  No existe un coste mínimo finito por el "
            "ciclo negativo alcanzable."
        )
    else:
        print("  El destino no es alcanzable desde el origen.")

    print("\nComprobación con NetworkX:")
    print(
        "  ¿Coincide la detección de ciclo negativo?: "
        f"{comprobacion['negative_cycle_match']}"
    )
    print(
        "  ¿Coinciden las distancias?: "
        f"{comprobacion['distances_match']}"
    )
    print(
        "  ¿Coincide el camino?: "
        f"{comprobacion['path_match']}"
    )
    print(
        "  ¿Coincide el coste?: "
        f"{comprobacion['cost_match']}"
    )


def main():
    # Cambiar a True permite observar la detección de un ciclo negativo.
    incluir_ciclo_negativo = False

    grafo = crear_grafo_bellman_ford(
        incluir_ciclo_negativo=incluir_ciclo_negativo,
    )
    posiciones = obtener_posiciones()

    origen = "A"
    destino = "I"

    resultado = ejecutar_bellman_ford_por_pasos(
        grafo=grafo,
        origen=origen,
        destino=destino,
    )

    comprobacion = comprobar_con_networkx(
        grafo=grafo,
        origen=origen,
        destino=destino,
        resultado=resultado,
    )

    imprimir_resultado_bellman_ford(
        resultado=resultado,
        comprobacion=comprobacion,
    )

    animador = GraphAnimator(
        figsize=(16, 10),
        interval=650,
    )

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "03_algoritmos"
        / "05_1_bellman_ford.png"
    )

    animacion = animador.animate_bellman_ford(
        graph=grafo,
        pos=posiciones,
        states=resultado["states"],
        source_node=origen,
        target_node=destino,
        title="Caminos mínimos con Bellman-Ford",
        final_image_path=output_path,
        repeat=False,
    )

    _ = animacion


if __name__ == "__main__":
    main()
