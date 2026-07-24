import heapq
from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


def crear_grafo_astar():
    """
    Crea el mismo grafo ponderado utilizado en el ejemplo de Dijkstra.

    Mantener el mismo grafo, los mismos pesos, el mismo origen y el mismo
    destino permite comparar directamente ambos algoritmos.
    """

    grafo = nx.Graph()

    grafo.add_weighted_edges_from([
        ("A", "B", 7),
        ("A", "C", 3),
        ("A", "D", 12),

        ("B", "C", 2),
        ("B", "E", 6),
        ("B", "F", 8),

        ("C", "D", 4),
        ("C", "E", 3),
        ("C", "F", 9),

        ("D", "F", 2),
        ("D", "G", 7),

        ("E", "F", 1),
        ("E", "H", 5),

        ("F", "G", 3),
        ("F", "H", 4),
        ("F", "I", 7),

        ("G", "I", 2),
        ("G", "J", 6),

        ("H", "I", 2),
        ("H", "K", 7),

        ("I", "J", 3),
        ("I", "K", 4),
        ("I", "L", 8),

        ("J", "L", 2),
        ("K", "L", 2),
    ])

    return grafo


def obtener_posiciones():
    """
    Mantiene las posiciones del ejemplo de Dijkstra.
    """

    return {
        "A": (-6.0, 0.0),

        "B": (-3.0, 3.0),
        "C": (-3.0, 0.0),
        "D": (-3.0, -3.0),

        "E": (0.0, 3.0),
        "F": (0.0, 0.0),
        "G": (0.0, -3.0),

        "H": (3.0, 3.0),
        "I": (3.0, 0.0),
        "J": (3.0, -3.0),

        "K": (6.0, 2.0),
        "L": (6.0, -2.0),
    }


def obtener_heuristica_astar():
    """
    Devuelve una heurística admisible y consistente para llegar a L.

    Los valores son estimaciones inferiores del coste restante.
    No representan el coste exacto y no se calculan durante A* mediante
    un algoritmo de caminos mínimos.

    La heurística está preparada para que se observe claramente cómo A*
    prioriza la zona que conduce al objetivo y evita expandir algunos
    vértices que Dijkstra sí procesa.
    """

    return {
        "A": 14,
        "B": 12,
        "C": 12,
        "D": 10,
        "E": 9,
        "F": 8,
        "G": 6,
        "H": 6,
        "I": 4,
        "J": 1,
        "K": 1,
        "L": 0,
    }


def validar_pesos_no_negativos(grafo):
    """
    Comprueba que A* no reciba aristas con peso negativo.
    """

    for origen, destino, datos in grafo.edges(data=True):
        peso = datos.get("weight", 1)

        if peso < 0:
            raise ValueError(
                "A* no admite esta implementación con pesos negativos. "
                f"La arista {origen}-{destino} tiene peso {peso}."
            )


def validar_heuristica_consistente(
    grafo,
    heuristica,
    destino,
):
    """
    Comprueba que la heurística sea no negativa, completa y consistente.

    En un grafo no dirigido se comprueban ambos sentidos de cada arista:

        h(u) <= coste(u, v) + h(v)
        h(v) <= coste(u, v) + h(u)
    """

    if destino not in heuristica:
        raise ValueError(
            f"No existe valor heurístico para el destino {destino!r}."
        )

    if heuristica[destino] != 0:
        raise ValueError(
            "La heurística del destino debe ser igual a cero."
        )

    for nodo in grafo.nodes():
        if nodo not in heuristica:
            raise ValueError(
                f"Falta la heurística del vértice {nodo!r}."
            )

        if heuristica[nodo] < 0:
            raise ValueError(
                f"La heurística de {nodo!r} no puede ser negativa."
            )

    for origen, destino_arista, datos in grafo.edges(data=True):
        peso = datos.get("weight", 1)

        if heuristica[origen] > peso + heuristica[destino_arista]:
            raise ValueError(
                "La heurística no es consistente en la arista "
                f"{origen}-{destino_arista}."
            )

        if heuristica[destino_arista] > peso + heuristica[origen]:
            raise ValueError(
                "La heurística no es consistente en la arista "
                f"{destino_arista}-{origen}."
            )


def crear_cola_visible(
    open_nodes,
    f_scores,
    h_scores,
):
    """
    Construye una versión limpia de la cola para la animación.

    El montículo interno puede contener entradas antiguas. La cola visual
    muestra solamente el mejor valor actual de cada vértice abierto.
    """

    return sorted(
        (
            f_scores[nodo],
            h_scores[nodo],
            nodo,
        )
        for nodo in open_nodes
    )


def crear_estado_astar(
    current,
    g_scores,
    h_scores,
    f_scores,
    predecessors,
    open_nodes,
    closed_nodes,
    message,
    active_edge=None,
    action=None,
    candidate_g=None,
    candidate_f=None,
    old_g=None,
    edge_weight=None,
    final_path=None,
):
    """
    Crea una copia independiente del estado actual de A*.
    """

    return {
        "current": current,
        "g_scores": dict(g_scores),
        "h_scores": dict(h_scores),
        "f_scores": dict(f_scores),
        "predecessors": dict(predecessors),
        "open_nodes": set(open_nodes),
        "closed_nodes": set(closed_nodes),
        "priority_queue": crear_cola_visible(
            open_nodes=open_nodes,
            f_scores=f_scores,
            h_scores=h_scores,
        ),
        "active_edge": active_edge,
        "action": action,
        "candidate_g": candidate_g,
        "candidate_f": candidate_f,
        "old_g": old_g,
        "edge_weight": edge_weight,
        "final_path": list(final_path or []),
        "message": message,
    }


def reconstruir_camino(predecesores, origen, destino):
    """
    Reconstruye el camino siguiendo los predecesores hacia atrás.
    """

    if origen == destino:
        return [origen]

    if predecesores.get(destino) is None:
        return []

    camino_inverso = []
    actual = destino

    while actual is not None:
        camino_inverso.append(actual)

        if actual == origen:
            break

        actual = predecesores.get(actual)

    if not camino_inverso or camino_inverso[-1] != origen:
        return []

    return list(reversed(camino_inverso))


def ejecutar_astar_por_pasos(
    grafo,
    origen,
    destino,
    heuristica,
):
    """
    Ejecuta A* y registra selecciones, relajaciones y cambios de estado.

    La prioridad utilizada es:

        f(n) = g(n) + h(n)

    Para desempatar se utiliza primero el menor valor de h. Esto favorece
    al vértice que parece estar más cerca del objetivo.
    """

    if origen not in grafo:
        raise ValueError(f"El origen {origen!r} no existe en el grafo.")

    if destino not in grafo:
        raise ValueError(f"El destino {destino!r} no existe en el grafo.")

    validar_pesos_no_negativos(grafo)
    validar_heuristica_consistente(
        grafo=grafo,
        heuristica=heuristica,
        destino=destino,
    )

    g_scores = {
        nodo: float("inf")
        for nodo in grafo.nodes()
    }

    h_scores = {
        nodo: heuristica[nodo]
        for nodo in grafo.nodes()
    }

    f_scores = {
        nodo: float("inf")
        for nodo in grafo.nodes()
    }

    predecesores = {
        nodo: None
        for nodo in grafo.nodes()
    }

    g_scores[origen] = 0
    f_scores[origen] = h_scores[origen]

    # Entradas: (f, h, vértice)
    cola_prioridad = [
        (
            f_scores[origen],
            h_scores[origen],
            origen,
        )
    ]

    abiertos = {origen}
    cerrados = set()
    orden_expansion = []
    estados = []

    estados.append(
        crear_estado_astar(
            current=None,
            g_scores=g_scores,
            h_scores=h_scores,
            f_scores=f_scores,
            predecessors=predecesores,
            open_nodes=abiertos,
            closed_nodes=cerrados,
            action="initial",
            message=(
                f"Inicialización: g({origen}) = 0, "
                f"h({origen}) = {h_scores[origen]} y "
                f"f({origen}) = {f_scores[origen]}."
            ),
        )
    )

    camino_minimo = []

    while cola_prioridad:
        f_extraido, h_extraido, actual = heapq.heappop(
            cola_prioridad
        )

        # Descarta entradas antiguas del montículo.
        if (
            actual in cerrados
            or f_extraido != f_scores[actual]
            or h_extraido != h_scores[actual]
        ):
            estados.append(
                crear_estado_astar(
                    current=actual,
                    g_scores=g_scores,
                    h_scores=h_scores,
                    f_scores=f_scores,
                    predecessors=predecesores,
                    open_nodes=abiertos,
                    closed_nodes=cerrados,
                    action="stale",
                    message=(
                        f"Se descarta una entrada antigua de {actual}; "
                        "ya no representa sus mejores valores g y f."
                    ),
                )
            )
            continue

        abiertos.discard(actual)
        cerrados.add(actual)
        orden_expansion.append(actual)

        estados.append(
            crear_estado_astar(
                current=actual,
                g_scores=g_scores,
                h_scores=h_scores,
                f_scores=f_scores,
                predecessors=predecesores,
                open_nodes=abiertos,
                closed_nodes=cerrados,
                action="select",
                message=(
                    f"{actual} tiene la menor prioridad: "
                    f"f={f_scores[actual]} = "
                    f"g={g_scores[actual]} + h={h_scores[actual]}. "
                    "Pasa al conjunto cerrado."
                ),
            )
        )

        if actual == destino:
            camino_minimo = reconstruir_camino(
                predecesores=predecesores,
                origen=origen,
                destino=destino,
            )
            break

        for vecino in sorted(grafo.neighbors(actual)):
            peso = grafo[actual][vecino].get("weight", 1)

            if vecino in cerrados:
                estados.append(
                    crear_estado_astar(
                        current=actual,
                        g_scores=g_scores,
                        h_scores=h_scores,
                        f_scores=f_scores,
                        predecessors=predecesores,
                        open_nodes=abiertos,
                        closed_nodes=cerrados,
                        active_edge=(actual, vecino),
                        action="already_closed",
                        edge_weight=peso,
                        message=(
                            f"{vecino} ya pertenece al conjunto cerrado. "
                            f"La arista {actual}-{vecino} no se relaja."
                        ),
                    )
                )
                continue

            old_g = g_scores[vecino]
            candidate_g = g_scores[actual] + peso
            candidate_f = candidate_g + h_scores[vecino]

            if candidate_g < old_g:
                g_scores[vecino] = candidate_g
                f_scores[vecino] = candidate_f
                predecesores[vecino] = actual

                abiertos.add(vecino)

                heapq.heappush(
                    cola_prioridad,
                    (
                        candidate_f,
                        h_scores[vecino],
                        vecino,
                    ),
                )

                estados.append(
                    crear_estado_astar(
                        current=actual,
                        g_scores=g_scores,
                        h_scores=h_scores,
                        f_scores=f_scores,
                        predecessors=predecesores,
                        open_nodes=abiertos,
                        closed_nodes=cerrados,
                        active_edge=(actual, vecino),
                        action="improvement",
                        candidate_g=candidate_g,
                        candidate_f=candidate_f,
                        old_g=old_g,
                        edge_weight=peso,
                        message=(
                            f"Mejora para {vecino}: "
                            f"g pasa de {old_g} a {candidate_g}, "
                            f"f pasa a {candidate_f} y su "
                            f"predecesor pasa a ser {actual}."
                        ),
                    )
                )
            else:
                estados.append(
                    crear_estado_astar(
                        current=actual,
                        g_scores=g_scores,
                        h_scores=h_scores,
                        f_scores=f_scores,
                        predecessors=predecesores,
                        open_nodes=abiertos,
                        closed_nodes=cerrados,
                        active_edge=(actual, vecino),
                        action="no_improvement",
                        candidate_g=candidate_g,
                        candidate_f=candidate_f,
                        old_g=old_g,
                        edge_weight=peso,
                        message=(
                            f"La ruta hacia {vecino} pasando por {actual} "
                            f"produciría g={candidate_g} y f={candidate_f}; "
                            f"no mejora su g actual ({old_g})."
                        ),
                    )
                )

    if camino_minimo:
        final_message = (
            f"A* ha alcanzado {destino}. Se detiene tras cerrar "
            f"{len(cerrados)} de {grafo.number_of_nodes()} vértices. "
            f"Quedan {len(abiertos)} candidatos abiertos que no fue "
            "necesario expandir."
        )
    else:
        final_message = (
            "A* ha terminado sin encontrar un camino hasta el objetivo."
        )

    estados.append(
        crear_estado_astar(
            current=None,
            g_scores=g_scores,
            h_scores=h_scores,
            f_scores=f_scores,
            predecessors=predecesores,
            open_nodes=abiertos,
            closed_nodes=cerrados,
            action="finished",
            final_path=camino_minimo,
            message=final_message,
        )
    )

    return {
        "states": estados,
        "g_scores": g_scores,
        "h_scores": h_scores,
        "f_scores": f_scores,
        "predecessors": predecesores,
        "open_nodes": abiertos,
        "closed_nodes": cerrados,
        "expansion_order": orden_expansion,
        "path": camino_minimo,
        "path_cost": g_scores[destino],
    }


def comprobar_con_networkx(
    grafo,
    origen,
    destino,
    heuristica,
    resultado,
):
    """
    Compara la implementación manual con A* y Dijkstra de NetworkX.
    """

    def funcion_heuristica(nodo_actual, nodo_objetivo):
        _ = nodo_objetivo
        return heuristica[nodo_actual]

    camino_astar_networkx = nx.astar_path(
        grafo,
        source=origen,
        target=destino,
        heuristic=funcion_heuristica,
        weight="weight",
    )

    coste_astar_networkx = nx.path_weight(
        grafo,
        camino_astar_networkx,
        weight="weight",
    )

    camino_dijkstra_networkx = nx.dijkstra_path(
        grafo,
        source=origen,
        target=destino,
        weight="weight",
    )

    coste_dijkstra_networkx = nx.dijkstra_path_length(
        grafo,
        source=origen,
        target=destino,
        weight="weight",
    )

    return {
        "astar_path_match": (
            resultado["path"] == camino_astar_networkx
        ),
        "astar_cost_match": (
            resultado["path_cost"] == coste_astar_networkx
        ),
        "dijkstra_path_match": (
            resultado["path"] == camino_dijkstra_networkx
        ),
        "dijkstra_cost_match": (
            resultado["path_cost"] == coste_dijkstra_networkx
        ),
        "networkx_astar_path": camino_astar_networkx,
        "networkx_dijkstra_path": camino_dijkstra_networkx,
    }


def formatear_valor(valor):
    """
    Formatea infinito y números enteros para la terminal.
    """

    if valor == float("inf"):
        return "∞"

    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))

    return str(valor)


def imprimir_resultado_astar(resultado, comprobacion):
    """
    Muestra por terminal los resultados principales.
    """

    print("\n=== Caminos mínimos con A* ===")

    print("\nOrden de expansión:")
    print(" -> ".join(resultado["expansion_order"]))

    print("\nValores finales conocidos:")

    for nodo in sorted(resultado["g_scores"]):
        g_value = formatear_valor(resultado["g_scores"][nodo])
        h_value = formatear_valor(resultado["h_scores"][nodo])
        f_value = formatear_valor(resultado["f_scores"][nodo])

        predecessor = resultado["predecessors"][nodo]
        predecessor_text = "—" if predecessor is None else predecessor

        if nodo in resultado["closed_nodes"]:
            estado = "cerrado"
        elif nodo in resultado["open_nodes"]:
            estado = "abierto"
        else:
            estado = "no descubierto"

        print(
            f"  {nodo}: g={g_value}, h={h_value}, f={f_value}, "
            f"predecesor={predecessor_text}, estado={estado}"
        )

    print("\nCamino mínimo:")

    if resultado["path"]:
        print("  " + " -> ".join(resultado["path"]))
        print(
            "  Coste total: "
            + formatear_valor(resultado["path_cost"])
        )
    else:
        print("  El destino no es alcanzable.")

    print("\nComparación con NetworkX:")
    print(
        "  ¿Coincide con nx.astar_path?: "
        f"{comprobacion['astar_path_match']}"
    )
    print(
        "  ¿Coincide el coste de A*?: "
        f"{comprobacion['astar_cost_match']}"
    )
    print(
        "  ¿Coincide con el camino de Dijkstra?: "
        f"{comprobacion['dijkstra_path_match']}"
    )
    print(
        "  ¿Coincide el coste de Dijkstra?: "
        f"{comprobacion['dijkstra_cost_match']}"
    )

    print("\nComparación de exploración:")
    print(
        f"  Vértices cerrados por A*: "
        f"{len(resultado['closed_nodes'])}"
    )
    print(
        f"  Vértices que quedan abiertos sin expandir: "
        f"{len(resultado['open_nodes'])}"
    )


def main():
    grafo = crear_grafo_astar()
    posiciones = obtener_posiciones()
    heuristica = obtener_heuristica_astar()

    origen = "A"
    destino = "L"

    resultado = ejecutar_astar_por_pasos(
        grafo=grafo,
        origen=origen,
        destino=destino,
        heuristica=heuristica,
    )

    comprobacion = comprobar_con_networkx(
        grafo=grafo,
        origen=origen,
        destino=destino,
        heuristica=heuristica,
        resultado=resultado,
    )

    imprimir_resultado_astar(
        resultado=resultado,
        comprobacion=comprobacion,
    )

    animador = GraphAnimator(
        figsize=(16, 10),
        interval=850,
    )

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "03_algoritmos"
        / "04_caminos_minimos_astar.png"
    )

    animacion = animador.animate_astar(
        graph=grafo,
        pos=posiciones,
        states=resultado["states"],
        source_node=origen,
        target_node=destino,
        title="Caminos mínimos con A*",
        final_image_path=output_path,
        repeat=False,
    )

    _ = animacion


if __name__ == "__main__":
    main()
