from collections import deque
from itertools import combinations
from math import sqrt
from pathlib import Path
import sys

import networkx as nx


CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


DAMPING_FACTOR = 0.85
PAGERANK_TOLERANCE = 1e-10
EIGENVECTOR_TOLERANCE = 1e-10
TARGET_COMMUNITIES = 3


def obtener_aristas_red_estructural():
    """
    Devuelve una red no dirigida con tres grupos densos.

    Cada grupo contiene cinco vértices y forma un K5. Los enlaces E—F y
    J—K conectan las comunidades y hacen que algunos vértices actúen como
    puentes estructurales.
    """

    community_1 = list("ABCDE")
    community_2 = list("FGHIJ")
    community_3 = list("KLMNO")

    edges = []
    for community in (community_1, community_2, community_3):
        edges.extend(combinations(community, 2))

    edges.extend([
        ("E", "F"),
        ("J", "K"),
    ])

    return list(edges)


def crear_grafo_estructural():
    """Crea el grafo base usado por centralidades y comunidades."""

    graph = nx.Graph()
    graph.add_nodes_from(list("ABCDEFGHIJKLMNO"))
    graph.add_edges_from(obtener_aristas_red_estructural())

    graph.graph["expected_communities"] = [
        frozenset("ABCDE"),
        frozenset("FGHIJ"),
        frozenset("KLMNO"),
    ]
    graph.graph["bridge_edges"] = [
        ("E", "F"),
        ("J", "K"),
    ]

    return graph


def obtener_posiciones():
    """Define posiciones manuales y estables para los tres grupos."""

    return {
        "A": (-6.2, 0.0),
        "B": (-5.1, 2.0),
        "C": (-3.2, 1.2),
        "D": (-5.1, -2.0),
        "E": (-3.2, -1.2),
        "F": (-1.2, -1.2),
        "G": (-1.2, 1.2),
        "H": (0.8, 2.0),
        "I": (2.0, 0.0),
        "J": (0.8, -2.0),
        "K": (3.2, -1.2),
        "L": (3.2, 1.2),
        "M": (5.2, 2.0),
        "N": (6.4, 0.0),
        "O": (5.2, -2.0),
    }


def validar_grafo_estructural(graph):
    """Comprueba las propiedades básicas del ejemplo."""

    if graph.is_directed():
        raise ValueError("El grafo base debe ser no dirigido.")
    if graph.number_of_nodes() == 0:
        raise ValueError("El grafo no puede estar vacío.")
    if not nx.is_connected(graph):
        raise ValueError("El ejemplo debe ser conexo.")
    if nx.number_of_selfloops(graph) != 0:
        raise ValueError("El ejemplo no debe contener bucles propios.")


def ordenar_puntuaciones(scores):
    """Ordena de mayor a menor y usa el nombre como desempate."""

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], str(item[0])),
    )


def calcular_centralidad_grado(graph):
    """Calcula grado normalizado: deg(v)/(n-1)."""

    n = graph.number_of_nodes()
    denominator = max(n - 1, 1)
    return {
        node: graph.degree(node) / denominator
        for node in graph.nodes()
    }


def calcular_centralidad_cercania(graph):
    """
    Calcula cercanía con la corrección habitual para desconexión.

    En este ejemplo el grafo es conexo, pero la fórmula también funciona
    por componentes.
    """

    n = graph.number_of_nodes()
    closeness = {}

    for source in graph.nodes():
        distances = dict(nx.single_source_shortest_path_length(graph, source))
        reachable = len(distances)
        total_distance = sum(distances.values())

        if total_distance == 0 or n <= 1:
            closeness[source] = 0.0
            continue

        local = (reachable - 1) / total_distance
        correction = (reachable - 1) / (n - 1)
        closeness[source] = local * correction

    return closeness


def calcular_cercania_armonica(graph):
    """Calcula cercanía armónica normalizada."""

    n = graph.number_of_nodes()
    denominator = max(n - 1, 1)
    harmonic = {}

    for source in graph.nodes():
        distances = nx.single_source_shortest_path_length(graph, source)
        harmonic[source] = sum(
            1.0 / distance
            for node, distance in distances.items()
            if node != source and distance > 0
        ) / denominator

    return harmonic


def calcular_intermediacion_brandes(graph):
    """Implementa Brandes para un grafo no ponderado y no dirigido."""

    nodes = list(graph.nodes())
    centrality = dict.fromkeys(nodes, 0.0)

    for source in nodes:
        stack = []
        predecessors = {node: [] for node in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        sigma[source] = 1.0
        distance = dict.fromkeys(nodes, -1)
        distance[source] = 0
        queue = deque([source])

        while queue:
            vertex = queue.popleft()
            stack.append(vertex)

            for neighbor in sorted(graph.neighbors(vertex)):
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[vertex] + 1

                if distance[neighbor] == distance[vertex] + 1:
                    sigma[neighbor] += sigma[vertex]
                    predecessors[neighbor].append(vertex)

        dependency = dict.fromkeys(nodes, 0.0)

        while stack:
            vertex = stack.pop()
            for predecessor in predecessors[vertex]:
                if sigma[vertex] > 0:
                    dependency[predecessor] += (
                        sigma[predecessor] / sigma[vertex]
                    ) * (1.0 + dependency[vertex])

            if vertex != source:
                centrality[vertex] += dependency[vertex]

    for node in centrality:
        centrality[node] /= 2.0

    n = len(nodes)
    if n > 2:
        normalization = 2.0 / ((n - 1) * (n - 2))
        for node in centrality:
            centrality[node] *= normalization

    return centrality


def calcular_autovector_iterativo(
    graph,
    tolerance=EIGENVECTOR_TOLERANCE,
    max_iterations=500,
):
    """Calcula centralidad de autovector mediante el método de potencias."""

    nodes = sorted(graph.nodes())
    initial = 1.0 / sqrt(len(nodes))
    scores = {node: initial for node in nodes}
    trace = [(0, dict(scores), float("inf"))]

    for iteration in range(1, max_iterations + 1):
        new_scores = {
            node: sum(scores[neighbor] for neighbor in graph.neighbors(node))
            for node in nodes
        }
        norm = sqrt(sum(value * value for value in new_scores.values()))
        if norm == 0:
            raise ValueError("No puede normalizarse el autovector.")

        new_scores = {
            node: value / norm
            for node, value in new_scores.items()
        }
        delta = max(
            abs(new_scores[node] - scores[node])
            for node in nodes
        )
        scores = new_scores
        trace.append((iteration, dict(scores), delta))

        if delta < tolerance:
            break
    else:
        raise RuntimeError("La centralidad de autovector no convergió.")

    return scores, trace


def crear_grafo_dirigido_pagerank(graph):
    """
    Crea una versión dirigida y ponderada para PageRank.

    Cada arista base se añade en ambos sentidos. Después se refuerzan
    enlaces entrantes hacia E, H y L. El vértice O queda sin salidas para
    mostrar el tratamiento de vértices colgantes.
    """

    directed = nx.DiGraph()
    directed.add_nodes_from(graph.nodes())

    for origin, destination in graph.edges():
        directed.add_edge(origin, destination, weight=1.0)
        directed.add_edge(destination, origin, weight=1.0)

    for target, boosted_weight in (("E", 1.8), ("H", 3.0), ("L", 2.2)):
        for origin in graph.neighbors(target):
            if directed.has_edge(origin, target):
                directed[origin][target]["weight"] = boosted_weight

    for destination in list(directed.successors("O")):
        directed.remove_edge("O", destination)

    return directed


def ejecutar_pagerank_iterativo(
    graph,
    damping=DAMPING_FACTOR,
    tolerance=PAGERANK_TOLERANCE,
    max_iterations=500,
):
    """Calcula PageRank ponderado, incluyendo la masa de nodos colgantes."""

    if not graph.is_directed():
        raise ValueError("PageRank se ejecuta sobre el grafo dirigido.")

    nodes = sorted(graph.nodes())
    n = len(nodes)
    scores = {node: 1.0 / n for node in nodes}
    trace = [(0, dict(scores), float("inf"))]

    out_weight = {
        node: sum(
            float(data.get("weight", 1.0))
            for _, _, data in graph.out_edges(node, data=True)
        )
        for node in nodes
    }

    for iteration in range(1, max_iterations + 1):
        dangling_mass = sum(
            scores[node]
            for node in nodes
            if out_weight[node] == 0
        )

        new_scores = {}
        for node in nodes:
            incoming = 0.0
            for predecessor, _, data in graph.in_edges(node, data=True):
                if out_weight[predecessor] > 0:
                    incoming += (
                        scores[predecessor]
                        * float(data.get("weight", 1.0))
                        / out_weight[predecessor]
                    )

            new_scores[node] = (
                (1.0 - damping) / n
                + damping * dangling_mass / n
                + damping * incoming
            )

        total = sum(new_scores.values())
        new_scores = {
            node: value / total
            for node, value in new_scores.items()
        }
        delta = sum(
            abs(new_scores[node] - scores[node])
            for node in nodes
        )
        scores = new_scores
        trace.append((iteration, dict(scores), delta))

        if delta < tolerance:
            break
    else:
        raise RuntimeError("PageRank no convergió.")

    return scores, trace


def normalizar_particion(communities):
    """Convierte una partición en un conjunto comparable de frozenset."""

    return frozenset(frozenset(community) for community in communities)


def detectar_comunidades_voraz(
    graph,
    target_communities=TARGET_COMMUNITIES,
):
    """
    Fusiona comunidades eligiendo en cada paso la partición de mayor Q.

    Se detiene al alcanzar el número objetivo de comunidades. El ejemplo
    se ha construido con tres grupos densos, por lo que el objetivo es 3.
    """

    communities = [frozenset([node]) for node in sorted(graph.nodes())]
    initial_modularity = nx.community.modularity(graph, communities)
    trace = [
        {
            "communities": list(communities),
            "modularity": initial_modularity,
            "merged": None,
        }
    ]

    while len(communities) > target_communities:
        best = None

        for first_index, second_index in combinations(range(len(communities)), 2):
            first = communities[first_index]
            second = communities[second_index]

            if not any(
                graph.has_edge(origin, destination)
                for origin in first
                for destination in second
            ):
                continue

            merged = first | second
            candidate = [
                community
                for index, community in enumerate(communities)
                if index not in {first_index, second_index}
            ] + [merged]
            candidate = sorted(candidate, key=lambda group: tuple(sorted(group)))
            modularity = nx.community.modularity(graph, candidate)
            tie_key = (
                tuple(sorted(first)),
                tuple(sorted(second)),
            )

            if (
                best is None
                or modularity > best[0] + 1e-15
                or (
                    abs(modularity - best[0]) <= 1e-15
                    and tie_key < best[1]
                )
            ):
                best = (
                    modularity,
                    tie_key,
                    candidate,
                    (first, second),
                )

        if best is None:
            raise RuntimeError("No se encontró una fusión válida.")

        modularity, _, communities, merged_pair = best
        trace.append(
            {
                "communities": list(communities),
                "modularity": modularity,
                "merged": merged_pair,
            }
        )

    return communities, trace


def calcular_mapa_comunidades(communities):
    """Asigna una etiqueta canónica C1, C2, ... a cada vértice."""

    ordered = sorted(communities, key=lambda group: tuple(sorted(group)))
    return {
        node: index + 1
        for index, community in enumerate(ordered)
        for node in community
    }


def seleccionar_traza(trace, first=8, every=4):
    """Reduce una traza larga conservando inicio, muestras y final."""

    if len(trace) <= first + 2:
        return trace

    selected_indices = set(range(min(first, len(trace))))
    selected_indices.update(range(first, len(trace), every))
    selected_indices.add(len(trace) - 1)
    return [trace[index] for index in sorted(selected_indices)]


def crear_estado_analisis(
    phase,
    metric_label,
    scores,
    message,
    graph,
    current_node=None,
    iteration=None,
    delta=None,
    communities=None,
    modularity=0.0,
    active_merge=None,
    all_metrics=None,
    winners=None,
):
    """Crea una instantánea independiente para la animación."""

    ranking = ordenar_puntuaciones(scores) if scores else []
    communities = [frozenset(group) for group in (communities or [])]
    community_map = calcular_mapa_comunidades(communities) if communities else {}

    if phase in {"eigenvector", "pagerank"}:
        status = (
            f"iteración {iteration} · distribución inicial"
            if delta == float("inf")
            else f"iteración {iteration} · cambio {delta:.3e}"
        )
    elif phase == "communities":
        status = f"{len(communities)} comunidades · Q={modularity:.5f}"
    elif phase == "final":
        status = f"Resumen final · Q={modularity:.5f}"
    else:
        status = f"{graph.number_of_nodes()} vértices · {graph.number_of_edges()} aristas"

    return {
        "phase": phase,
        "metric_label": metric_label,
        "scores": dict(scores),
        "ranking": list(ranking),
        "current_node": current_node,
        "iteration": iteration,
        "delta": delta,
        "communities": communities,
        "community_map": community_map,
        "modularity": modularity,
        "active_merge": active_merge,
        "all_metrics": {name: dict(values) for name, values in (all_metrics or {}).items()},
        "winners": list(winners or []),
        "bridge_edges": list(graph.graph.get("bridge_edges", [])),
        "status_text": status,
        "table_footer": (
            "El tamaño del nodo representa la puntuación; el color representa la fase."
            if phase != "final"
            else "Valores normalizados o probabilidades; com. = comunidad."
        ),
        "message": message,
    }


def crear_estados_animacion(graph, results):
    """Construye una secuencia pedagógica de estados."""

    states = []

    metric_phases = [
        (
            "degree",
            "Centralidad de grado",
            results["degree"],
            "El grado mide conexiones directas. Los puentes E, F, J y K tienen una conexión adicional entre grupos.",
        ),
        (
            "closeness",
            "Centralidad de cercanía",
            results["closeness"],
            "La cercanía favorece los vértices con menor distancia media al resto de la red.",
        ),
        (
            "betweenness",
            "Centralidad de intermediación",
            results["betweenness"],
            "La intermediación destaca los vértices por los que pasan muchos caminos mínimos entre comunidades.",
        ),
    ]

    for phase, label, scores, intro in metric_phases:
        states.append(
            crear_estado_analisis(
                phase=phase,
                metric_label=label,
                scores=scores,
                message=intro,
                graph=graph,
            )
        )
        for rank, (node, value) in enumerate(ordenar_puntuaciones(scores)[:5], start=1):
            states.append(
                crear_estado_analisis(
                    phase=phase,
                    metric_label=label,
                    scores=scores,
                    current_node=node,
                    message=f"Puesto {rank}: {node} obtiene {value:.5f} en {label.lower()}.",
                    graph=graph,
                )
            )

    for iteration, scores, delta in seleccionar_traza(results["eigenvector_trace"], first=6, every=3):
        states.append(
            crear_estado_analisis(
                phase="eigenvector",
                metric_label="Centralidad de autovector",
                scores=scores,
                iteration=iteration,
                delta=delta,
                message=(
                    "El método de potencias propaga importancia desde los vecinos y normaliza el vector en cada iteración."
                    if iteration == 0
                    else f"Iteración {iteration}: la puntuación cambia como máximo {delta:.3e}."
                ),
                graph=graph,
            )
        )

    for iteration, scores, delta in seleccionar_traza(results["pagerank_trace"], first=8, every=4):
        states.append(
            crear_estado_analisis(
                phase="pagerank",
                metric_label=f"PageRank · d={DAMPING_FACTOR}",
                scores=scores,
                iteration=iteration,
                delta=delta,
                message=(
                    "PageRank comienza uniforme. Las flechas ponderadas transfieren importancia y O actúa como vértice colgante."
                    if iteration == 0
                    else f"Iteración {iteration}: cambio L1 = {delta:.3e}."
                ),
                graph=graph,
            )
        )

    for step_index, item in enumerate(results["community_trace"]):
        communities = item["communities"]
        merged = item["merged"]
        if merged is None:
            message = "La detección comienza con una comunidad independiente por vértice."
        else:
            first, second = merged
            message = (
                f"Fusión {step_index}: {{{', '.join(sorted(first))}}} y "
                f"{{{', '.join(sorted(second))}}}; modularidad Q={item['modularity']:.5f}."
            )

        states.append(
            crear_estado_analisis(
                phase="communities",
                metric_label="Comunidades por modularidad",
                scores={},
                message=message,
                graph=graph,
                communities=communities,
                modularity=item["modularity"],
                active_merge=merged,
            )
        )

    all_metrics = {
        "degree": results["degree"],
        "closeness": results["closeness"],
        "betweenness": results["betweenness"],
        "eigenvector": results["eigenvector"],
        "pagerank": results["pagerank"],
    }
    winners = [
        ("Grado", *ordenar_puntuaciones(results["degree"])[0]),
        ("Cercanía", *ordenar_puntuaciones(results["closeness"])[0]),
        ("Intermediación", *ordenar_puntuaciones(results["betweenness"])[0]),
        ("Autovector", *ordenar_puntuaciones(results["eigenvector"])[0]),
        ("PageRank", *ordenar_puntuaciones(results["pagerank"])[0]),
    ]

    states.append(
        crear_estado_analisis(
            phase="final",
            metric_label="Comparación final",
            scores=results["pagerank"],
            message=(
                "El tamaño final representa PageRank, los colores muestran las tres comunidades y las aristas rojas conectan grupos distintos."
            ),
            graph=graph,
            communities=results["communities"],
            modularity=results["modularity"],
            all_metrics=all_metrics,
            winners=winners,
        )
    )

    return states


def calcular_pagerank_referencia_networkx_sin_scipy(graph):
    """
    Calcula PageRank con la implementación iterativa pura de NetworkX.

    ``nx.pagerank`` delega en algunas versiones de NetworkX en una
    implementación basada en SciPy. En entornos donde SciPy está incompleto,
    mal instalado o sombreado por otro módulo, esa llamada puede fallar antes
    de comenzar el cálculo con un error similar a::

        AttributeError: module 'scipy' has no attribute 'sparse'

    Para que este ejemplo educativo no dependa de ``scipy.sparse``, se usa
    directamente la implementación de iteración de potencias escrita en
    Python que NetworkX incluye en el mismo módulo de PageRank.
    """

    try:
        from networkx.algorithms.link_analysis.pagerank_alg import (
            _pagerank_python,
        )
    except ImportError as error:
        raise RuntimeError(
            "La versión instalada de NetworkX no contiene la implementación "
            "iterativa pura de PageRank. Actualiza NetworkX o repara SciPy."
        ) from error

    return _pagerank_python(
        graph,
        alpha=DAMPING_FACTOR,
        max_iter=1000,
        tol=1e-12,
        weight="weight",
    )


def comprobar_con_networkx(graph, directed_graph, results):
    """Compara todas las métricas con implementaciones de referencia."""

    nx_degree = nx.degree_centrality(graph)
    nx_closeness = nx.closeness_centrality(graph)
    nx_betweenness = nx.betweenness_centrality(graph, normalized=True)
    nx_eigenvector = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-12)
    nx_pagerank = calcular_pagerank_referencia_networkx_sin_scipy(
        directed_graph
    )
    nx_communities = nx.community.greedy_modularity_communities(
        graph,
        cutoff=TARGET_COMMUNITIES,
        best_n=TARGET_COMMUNITIES,
    )

    def close_dict(first, second, tolerance=1e-7):
        return all(
            abs(first[node] - second[node]) <= tolerance
            for node in first
        )

    expected = normalizar_particion(graph.graph["expected_communities"])
    obtained = normalizar_particion(results["communities"])
    networkx_partition = normalizar_particion(nx_communities)

    return {
        "degree_matches": close_dict(results["degree"], nx_degree),
        "closeness_matches": close_dict(results["closeness"], nx_closeness),
        "betweenness_matches": close_dict(results["betweenness"], nx_betweenness),
        "eigenvector_matches": close_dict(results["eigenvector"], nx_eigenvector, 1e-6),
        "pagerank_matches": close_dict(results["pagerank"], nx_pagerank, 1e-7),
        "pagerank_sums_to_one": abs(sum(results["pagerank"].values()) - 1.0) <= 1e-10,
        "expected_communities_match": obtained == expected,
        "networkx_communities_match": obtained == networkx_partition,
        "modularity_matches": abs(
            results["modularity"]
            - nx.community.modularity(graph, nx_communities)
        ) <= 1e-12,
    }


def calcular_resultados(graph):
    """Ejecuta todas las métricas y la detección de comunidades."""

    validar_grafo_estructural(graph)
    directed_graph = crear_grafo_dirigido_pagerank(graph)

    degree = calcular_centralidad_grado(graph)
    closeness = calcular_centralidad_cercania(graph)
    harmonic = calcular_cercania_armonica(graph)
    betweenness = calcular_intermediacion_brandes(graph)
    eigenvector, eigenvector_trace = calcular_autovector_iterativo(graph)
    pagerank, pagerank_trace = ejecutar_pagerank_iterativo(directed_graph)
    communities, community_trace = detectar_comunidades_voraz(graph)
    modularity = nx.community.modularity(graph, communities)
    edge_betweenness = nx.edge_betweenness_centrality(graph, normalized=True)

    results = {
        "degree": degree,
        "closeness": closeness,
        "harmonic": harmonic,
        "betweenness": betweenness,
        "eigenvector": eigenvector,
        "eigenvector_trace": eigenvector_trace,
        "pagerank": pagerank,
        "pagerank_trace": pagerank_trace,
        "communities": communities,
        "community_trace": community_trace,
        "modularity": modularity,
        "edge_betweenness": edge_betweenness,
        "directed_graph": directed_graph,
    }
    results["states"] = crear_estados_animacion(graph, results)
    return results


def imprimir_resultados(graph, results, checks):
    """Muestra rankings y comunidades por terminal."""

    print("\n=== Centralidad, PageRank y comunidades ===")
    print(f"Vértices: {graph.number_of_nodes()}")
    print(f"Aristas: {graph.number_of_edges()}")

    metrics = [
        ("Grado", results["degree"]),
        ("Cercanía", results["closeness"]),
        ("Cercanía armónica", results["harmonic"]),
        ("Intermediación", results["betweenness"]),
        ("Autovector", results["eigenvector"]),
        ("PageRank", results["pagerank"]),
    ]

    for name, scores in metrics:
        print(f"\n{name}:")
        for rank, (node, value) in enumerate(ordenar_puntuaciones(scores), start=1):
            print(f"  {rank:>2}. {node}: {value:.8f}")

    print("\nComunidades:")
    for index, community in enumerate(
        sorted(results["communities"], key=lambda group: tuple(sorted(group))),
        start=1,
    ):
        print(f"  C{index}: {', '.join(sorted(community))}")

    print(f"\nModularidad: {results['modularity']:.8f}")

    print("\nAristas con mayor intermediación:")
    for edge, value in sorted(
        results["edge_betweenness"].items(),
        key=lambda item: (-item[1], tuple(sorted(item[0]))),
    )[:5]:
        print(f"  {edge[0]}—{edge[1]}: {value:.8f}")

    print("\nComprobación con NetworkX:")
    for name, value in checks.items():
        print(f"  {name}: {value}")


def main():
    graph = crear_grafo_estructural()
    positions = obtener_posiciones()
    results = calcular_resultados(graph)
    checks = comprobar_con_networkx(
        graph,
        results["directed_graph"],
        results,
    )

    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError("Fallaron las comprobaciones: " + ", ".join(failed))

    imprimir_resultados(graph, results, checks)

    animator = GraphAnimator(
        figsize=(16, 10),
        interval=780,
    )

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "03_algoritmos"
        / "10_centralidad_pagerank_comunidades.png"
    )

    animation = animator.animate_centrality_pagerank_communities(
        graph=graph,
        pagerank_graph=results["directed_graph"],
        pos=positions,
        states=results["states"],
        title="Centralidad, PageRank y comunidades",
        final_image_path=output_path,
        repeat=False,
    )

    _ = animation


if __name__ == "__main__":
    main()