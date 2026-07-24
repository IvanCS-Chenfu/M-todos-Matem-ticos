from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


def obtener_aristas_floyd_warshall():
    """
    Devuelve el mismo grafo dirigido utilizado con Bellman-Ford.

    Mantener los mismos pesos, incluidos los negativos, permite comparar:
    - Bellman-Ford: un origen hacia todos los destinos;
    - Floyd-Warshall: todos los orígenes hacia todos los destinos.
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


def crear_grafo_floyd_warshall(
    incluir_ciclo_negativo=False,
):
    """
    Crea el grafo dirigido ponderado del apartado.

    Por defecto contiene pesos negativos, pero no ciclos negativos.

    El modo opcional añade I→C con peso -5. Esa arista permite comprobar
    cómo Floyd-Warshall detecta un ciclo negativo mediante una entrada
    negativa en la diagonal de la matriz.
    """

    grafo = nx.DiGraph()

    aristas = obtener_aristas_floyd_warshall()

    if incluir_ciclo_negativo:
        aristas = list(aristas) + [
            ("I", "C", -5),
        ]

    grafo.add_weighted_edges_from(aristas)

    return grafo


def obtener_posiciones():
    """
    Mantiene la distribución visual del ejemplo de Bellman-Ford.
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


def crear_matrices_iniciales(grafo):
    """
    Construye las matrices iniciales de distancia y siguiente vértice.

    Reglas:
    - d(i, i) = 0;
    - d(i, j) = peso de la arista directa;
    - d(i, j) = infinito si no hay conexión directa;
    - siguiente(i, j) = j cuando existe una arista directa.
    """

    nodes = sorted(grafo.nodes())

    distances = {
        origin: {
            destination: float("inf")
            for destination in nodes
        }
        for origin in nodes
    }

    next_nodes = {
        origin: {
            destination: None
            for destination in nodes
        }
        for origin in nodes
    }

    for node in nodes:
        distances[node][node] = 0
        next_nodes[node][node] = node

    for origin, destination, data in grafo.edges(data=True):
        weight = data.get("weight", 1)

        if weight < distances[origin][destination]:
            distances[origin][destination] = weight
            next_nodes[origin][destination] = destination

    return nodes, distances, next_nodes


def copiar_matriz(matrix):
    """
    Crea una copia independiente de una matriz basada en diccionarios.
    """

    return {
        row_name: dict(row)
        for row_name, row in matrix.items()
    }


def reconstruir_camino_floyd(
    next_nodes,
    origin,
    destination,
):
    """
    Reconstruye un camino usando la matriz de siguientes.

    Se incluye una protección contra bucles para evitar una reconstrucción
    infinita si se intenta usar la matriz después de detectar un ciclo
    negativo.
    """

    if next_nodes[origin][destination] is None:
        return []

    path = [origin]
    current = origin
    maximum_steps = len(next_nodes) + 1

    for _ in range(maximum_steps):
        if current == destination:
            return path

        current = next_nodes[current][destination]

        if current is None:
            return []

        path.append(current)

    return []


def unir_caminos(
    first_path,
    second_path,
):
    """
    Une i→k y k→j evitando repetir el vértice intermedio.
    """

    if not first_path or not second_path:
        return []

    return first_path + second_path[1:]


def crear_estado_floyd_warshall(
    nodes,
    distances,
    next_nodes,
    message,
    phase,
    processed_intermediates,
    current_k_index,
    total_k,
    total_updates,
    updates_for_k,
    active_i=None,
    active_j=None,
    active_k=None,
    action=None,
    candidate=None,
    old_distance=None,
    distance_ik=None,
    distance_kj=None,
    candidate_path=None,
    final_path=None,
    negative_cycle_nodes=None,
):
    """
    Crea una copia independiente de un estado de Floyd-Warshall.
    """

    return {
        "nodes": list(nodes),
        "distances": copiar_matriz(distances),
        "next_nodes": copiar_matriz(next_nodes),
        "active_i": active_i,
        "active_j": active_j,
        "active_k": active_k,
        "action": action,
        "candidate": candidate,
        "old_distance": old_distance,
        "distance_ik": distance_ik,
        "distance_kj": distance_kj,
        "candidate_path": list(candidate_path or []),
        "final_path": list(final_path or []),
        "phase": phase,
        "processed_intermediates": set(processed_intermediates),
        "current_k_index": current_k_index,
        "total_k": total_k,
        "total_updates": total_updates,
        "updates_for_k": updates_for_k,
        "negative_cycle_nodes": set(negative_cycle_nodes or set()),
        "message": message,
    }


def ejecutar_floyd_warshall_por_pasos(
    grafo,
    origen_comparado,
    destino_comparado,
    registrar_comparaciones_sin_mejora=False,
):
    """
    Ejecuta Floyd-Warshall y registra sus estados principales.

    La actualización central es:

        d[i][j] = min(
            d[i][j],
            d[i][k] + d[k][j]
        )

    Para mantener una animación de duración razonable, por defecto se
    registran:
    - inicio y final de cada intermedio k;
    - todas las mejoras;
    - comprobación final de la diagonal.

    Si ``registrar_comparaciones_sin_mejora`` es verdadero, también se
    guarda cada comparación válida que no produce una mejora.
    """

    if origen_comparado not in grafo:
        raise ValueError(
            f"El origen {origen_comparado!r} no existe en el grafo."
        )

    if destino_comparado not in grafo:
        raise ValueError(
            f"El destino {destino_comparado!r} no existe en el grafo."
        )

    nodes, distances, next_nodes = crear_matrices_iniciales(grafo)

    states = []
    processed_intermediates = set()
    total_updates = 0

    states.append(
        crear_estado_floyd_warshall(
            nodes=nodes,
            distances=distances,
            next_nodes=next_nodes,
            message=(
                "Matriz inicial: cero en la diagonal, pesos directos "
                "en las aristas e infinito cuando no existe conexión."
            ),
            phase="initial",
            processed_intermediates=processed_intermediates,
            current_k_index=0,
            total_k=len(nodes),
            total_updates=0,
            updates_for_k=0,
        )
    )

    for k_index, k_node in enumerate(nodes, start=1):
        updates_for_k = 0

        states.append(
            crear_estado_floyd_warshall(
                nodes=nodes,
                distances=distances,
                next_nodes=next_nodes,
                message=(
                    f"Se habilita {k_node} como posible vértice "
                    "intermedio de todos los caminos."
                ),
                phase="intermediate_start",
                processed_intermediates=processed_intermediates,
                current_k_index=k_index,
                total_k=len(nodes),
                total_updates=total_updates,
                updates_for_k=updates_for_k,
                active_k=k_node,
            )
        )

        for i_node in nodes:
            if distances[i_node][k_node] == float("inf"):
                continue

            for j_node in nodes:
                if distances[k_node][j_node] == float("inf"):
                    continue

                old_distance = distances[i_node][j_node]
                distance_ik = distances[i_node][k_node]
                distance_kj = distances[k_node][j_node]
                candidate = distance_ik + distance_kj

                path_i_k = reconstruir_camino_floyd(
                    next_nodes=next_nodes,
                    origin=i_node,
                    destination=k_node,
                )

                path_k_j = reconstruir_camino_floyd(
                    next_nodes=next_nodes,
                    origin=k_node,
                    destination=j_node,
                )

                candidate_path = unir_caminos(
                    first_path=path_i_k,
                    second_path=path_k_j,
                )

                if candidate < old_distance:
                    distances[i_node][j_node] = candidate
                    next_nodes[i_node][j_node] = (
                        next_nodes[i_node][k_node]
                    )

                    updates_for_k += 1
                    total_updates += 1

                    states.append(
                        crear_estado_floyd_warshall(
                            nodes=nodes,
                            distances=distances,
                            next_nodes=next_nodes,
                            message=(
                                f"Mejora para {i_node}→{j_node}: "
                                f"pasar por {k_node} reduce el coste de "
                                f"{old_distance} a {candidate}."
                            ),
                            phase="iteration",
                            processed_intermediates=processed_intermediates,
                            current_k_index=k_index,
                            total_k=len(nodes),
                            total_updates=total_updates,
                            updates_for_k=updates_for_k,
                            active_i=i_node,
                            active_j=j_node,
                            active_k=k_node,
                            action="improvement",
                            candidate=candidate,
                            old_distance=old_distance,
                            distance_ik=distance_ik,
                            distance_kj=distance_kj,
                            candidate_path=candidate_path,
                        )
                    )

                elif registrar_comparaciones_sin_mejora:
                    states.append(
                        crear_estado_floyd_warshall(
                            nodes=nodes,
                            distances=distances,
                            next_nodes=next_nodes,
                            message=(
                                f"Pasar de {i_node} a {j_node} mediante "
                                f"{k_node} costaría {candidate}; no mejora "
                                f"el valor actual ({old_distance})."
                            ),
                            phase="iteration",
                            processed_intermediates=processed_intermediates,
                            current_k_index=k_index,
                            total_k=len(nodes),
                            total_updates=total_updates,
                            updates_for_k=updates_for_k,
                            active_i=i_node,
                            active_j=j_node,
                            active_k=k_node,
                            action="no_improvement",
                            candidate=candidate,
                            old_distance=old_distance,
                            distance_ik=distance_ik,
                            distance_kj=distance_kj,
                            candidate_path=candidate_path,
                        )
                    )

        processed_intermediates.add(k_node)

        states.append(
            crear_estado_floyd_warshall(
                nodes=nodes,
                distances=distances,
                next_nodes=next_nodes,
                message=(
                    f"Finaliza el intermedio {k_node} con "
                    f"{updates_for_k} mejora(s)."
                ),
                phase="intermediate_end",
                processed_intermediates=processed_intermediates,
                current_k_index=k_index,
                total_k=len(nodes),
                total_updates=total_updates,
                updates_for_k=updates_for_k,
                active_k=k_node,
            )
        )

    negative_cycle_nodes = {
        node
        for node in nodes
        if distances[node][node] < 0
    }

    if negative_cycle_nodes:
        final_path = []
        path_cost = None
        final_message = (
            "Floyd-Warshall ha detectado una entrada negativa en la "
            "diagonal. Existe al menos un ciclo negativo."
        )
    else:
        final_path = reconstruir_camino_floyd(
            next_nodes=next_nodes,
            origin=origen_comparado,
            destination=destino_comparado,
        )
        path_cost = distances[origen_comparado][destino_comparado]
        final_message = (
            "Floyd-Warshall ha terminado. La matriz contiene las "
            "distancias mínimas entre todos los pares y se resalta "
            f"el camino {origen_comparado}→{destino_comparado}."
        )

    states.append(
        crear_estado_floyd_warshall(
            nodes=nodes,
            distances=distances,
            next_nodes=next_nodes,
            message=final_message,
            phase="finished",
            processed_intermediates=processed_intermediates,
            current_k_index=len(nodes),
            total_k=len(nodes),
            total_updates=total_updates,
            updates_for_k=0,
            final_path=final_path,
            negative_cycle_nodes=negative_cycle_nodes,
        )
    )

    return {
        "states": states,
        "nodes": nodes,
        "distances": distances,
        "next_nodes": next_nodes,
        "negative_cycle_nodes": negative_cycle_nodes,
        "total_updates": total_updates,
        "path": final_path,
        "path_cost": path_cost,
    }


def comprobar_con_networkx(
    grafo,
    origen,
    destino,
    resultado,
):
    """
    Compara matrices, camino y detección de ciclos con NetworkX.
    """

    predecessors_nx, distances_nx = (
        nx.floyd_warshall_predecessor_and_distance(
            grafo,
            weight="weight",
        )
    )

    negative_cycle_nx = nx.negative_edge_cycle(
        grafo,
        weight="weight",
    )

    distances_match = all(
        resultado["distances"][origin][destination]
        == distances_nx[origin][destination]
        for origin in resultado["nodes"]
        for destination in resultado["nodes"]
    )

    if negative_cycle_nx:
        path_nx = []
        cost_nx = None
    else:
        path_nx = nx.reconstruct_path(
            origen,
            destino,
            predecessors_nx,
        )
        cost_nx = distances_nx[origen][destino]

    return {
        "negative_cycle_match": (
            bool(resultado["negative_cycle_nodes"])
            == negative_cycle_nx
        ),
        "distances_match": distances_match,
        "path_match": resultado["path"] == path_nx,
        "cost_match": resultado["path_cost"] == cost_nx,
        "networkx_path": path_nx,
        "networkx_cost": cost_nx,
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


def imprimir_matriz(nodes, distances):
    """
    Imprime una matriz de distancias alineada.
    """

    width = 6

    print(" " * width, end="")

    for destination in nodes:
        print(f"{destination:>{width}}", end="")

    print()

    for origin in nodes:
        print(f"{origin:>{width}}", end="")

        for destination in nodes:
            value = formatear_valor(
                distances[origin][destination]
            )
            print(f"{value:>{width}}", end="")

        print()


def imprimir_resultado_floyd_warshall(
    resultado,
    comprobacion,
):
    """
    Muestra los resultados principales por terminal.
    """

    print("\n=== Caminos mínimos con Floyd-Warshall ===")

    print("\nMatriz final de distancias:")
    imprimir_matriz(
        nodes=resultado["nodes"],
        distances=resultado["distances"],
    )

    print(
        "\nNúmero total de mejoras: "
        f"{resultado['total_updates']}"
    )

    print("\nDetección de ciclos negativos:")

    if resultado["negative_cycle_nodes"]:
        print(
            "  Sí. Diagonal negativa en: "
            + ", ".join(sorted(resultado["negative_cycle_nodes"]))
        )
    else:
        print("  No")

    print("\nCamino seleccionado para la comparación:")

    if resultado["path"]:
        print("  " + " -> ".join(resultado["path"]))
        print(
            "  Coste total: "
            + formatear_valor(resultado["path_cost"])
        )
    elif resultado["negative_cycle_nodes"]:
        print(
            "  No se reconstruye porque existe un ciclo negativo."
        )
    else:
        print("  El destino no es alcanzable.")

    print("\nComprobación con NetworkX:")
    print(
        "  ¿Coincide la detección de ciclo negativo?: "
        f"{comprobacion['negative_cycle_match']}"
    )
    print(
        "  ¿Coincide toda la matriz?: "
        f"{comprobacion['distances_match']}"
    )
    print(
        "  ¿Coincide el camino seleccionado?: "
        f"{comprobacion['path_match']}"
    )
    print(
        "  ¿Coincide el coste seleccionado?: "
        f"{comprobacion['cost_match']}"
    )


def main():
    # Cambiar a True permite probar la diagonal negativa.
    incluir_ciclo_negativo = False

    # Cambiar a True registra también todas las comparaciones sin mejora.
    # La animación será mucho más larga.
    registrar_comparaciones_sin_mejora = False

    grafo = crear_grafo_floyd_warshall(
        incluir_ciclo_negativo=incluir_ciclo_negativo,
    )
    posiciones = obtener_posiciones()

    origen_comparado = "A"
    destino_comparado = "I"

    resultado = ejecutar_floyd_warshall_por_pasos(
        grafo=grafo,
        origen_comparado=origen_comparado,
        destino_comparado=destino_comparado,
        registrar_comparaciones_sin_mejora=(
            registrar_comparaciones_sin_mejora
        ),
    )

    comprobacion = comprobar_con_networkx(
        grafo=grafo,
        origen=origen_comparado,
        destino=destino_comparado,
        resultado=resultado,
    )

    imprimir_resultado_floyd_warshall(
        resultado=resultado,
        comprobacion=comprobacion,
    )

    animador = GraphAnimator(
        figsize=(16, 10),
        interval=700,
    )

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "03_algoritmos"
        / "05_2_floyd_warshall.png"
    )

    animacion = animador.animate_floyd_warshall(
        graph=grafo,
        pos=posiciones,
        states=resultado["states"],
        source_node=origen_comparado,
        target_node=destino_comparado,
        title="Caminos mínimos con Floyd-Warshall",
        final_image_path=output_path,
        repeat=False,
    )

    _ = animacion


if __name__ == "__main__":
    main()
