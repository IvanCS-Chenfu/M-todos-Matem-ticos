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


def crear_grafo_dijkstra():
    """
    Crea un grafo ponderado no dirigido para estudiar Dijkstra.

    El grafo contiene varias rutas alternativas. Algunas distancias
    provisionales se actualizan más de una vez antes de convertirse
    en definitivas.
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
    Define posiciones manuales para obtener una visualización estable.

    El grafo se organiza de izquierda a derecha desde el origen A hasta
    el destino L.
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


def validar_pesos_no_negativos(grafo):
    """
    Comprueba la condición necesaria para utilizar Dijkstra.
    """

    for origen, destino, datos in grafo.edges(data=True):
        peso = datos.get("weight", 1)

        if peso < 0:
            raise ValueError(
                "Dijkstra no admite pesos negativos. "
                f"La arista {origen}-{destino} tiene peso {peso}."
            )


def crear_estado_dijkstra(
    current,
    distances,
    predecessors,
    finalized,
    priority_queue,
    message,
    active_edge=None,
    action=None,
    candidate=None,
    old_distance=None,
    edge_weight=None,
    final_path=None,
):
    """
    Crea una copia independiente de un estado de Dijkstra.

    Guardar copias es necesario porque las distancias, los predecesores
    y la cola cambian durante las siguientes iteraciones.
    """

    return {
        "current": current,
        "distances": dict(distances),
        "predecessors": dict(predecessors),
        "finalized": set(finalized),
        "priority_queue": sorted(list(priority_queue)),
        "active_edge": active_edge,
        "action": action,
        "candidate": candidate,
        "old_distance": old_distance,
        "edge_weight": edge_weight,
        "final_path": list(final_path or []),
        "message": message,
    }


def reconstruir_camino(predecesores, origen, destino):
    """
    Reconstruye el camino mínimo siguiendo predecesores hacia atrás.
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


def ejecutar_dijkstra_por_pasos(grafo, origen, destino):
    """
    Ejecuta Dijkstra y registra cada selección y relajación.

    La cola de prioridad contiene pares:

        (distancia_provisional, vértice)

    Cuando una distancia mejora, se inserta una nueva entrada. Las entradas
    antiguas permanecen en el montículo y se descartan al extraerlas.
    """

    if origen not in grafo:
        raise ValueError(f"El origen {origen!r} no existe en el grafo.")

    if destino not in grafo:
        raise ValueError(f"El destino {destino!r} no existe en el grafo.")

    validar_pesos_no_negativos(grafo)

    distancias = {
        nodo: float("inf")
        for nodo in grafo.nodes()
    }

    predecesores = {
        nodo: None
        for nodo in grafo.nodes()
    }

    distancias[origen] = 0

    cola_prioridad = [(0, origen)]
    definitivos = set()
    orden_definitivo = []
    estados = []

    estados.append(
        crear_estado_dijkstra(
            current=None,
            distances=distancias,
            predecessors=predecesores,
            finalized=definitivos,
            priority_queue=cola_prioridad,
            action="initial",
            message=(
                f"Inicialización: d({origen}) = 0 y el resto de "
                "distancias se establece a infinito."
            ),
        )
    )

    while cola_prioridad:
        distancia_extraida, actual = heapq.heappop(cola_prioridad)

        # Una misma entrada puede haber quedado desactualizada después
        # de encontrar una ruta mejor.
        if (
            actual in definitivos
            or distancia_extraida != distancias[actual]
        ):
            estados.append(
                crear_estado_dijkstra(
                    current=actual,
                    distances=distancias,
                    predecessors=predecesores,
                    finalized=definitivos,
                    priority_queue=cola_prioridad,
                    action="stale",
                    message=(
                        f"Se descarta la entrada ({distancia_extraida}, "
                        f"{actual}) porque ya no representa la mejor "
                        "distancia conocida."
                    ),
                )
            )
            continue

        definitivos.add(actual)
        orden_definitivo.append(actual)

        estados.append(
            crear_estado_dijkstra(
                current=actual,
                distances=distancias,
                predecessors=predecesores,
                finalized=definitivos,
                priority_queue=cola_prioridad,
                action="select",
                message=(
                    f"{actual} tiene la menor distancia provisional "
                    f"({distancias[actual]}) y pasa a ser definitivo."
                ),
            )
        )

        for vecino in sorted(grafo.neighbors(actual)):
            peso = grafo[actual][vecino].get("weight", 1)

            # Una distancia definitiva no necesita volver a relajarse.
            if vecino in definitivos:
                estados.append(
                    crear_estado_dijkstra(
                        current=actual,
                        distances=distancias,
                        predecessors=predecesores,
                        finalized=definitivos,
                        priority_queue=cola_prioridad,
                        active_edge=(actual, vecino),
                        action="already_finalized",
                        edge_weight=peso,
                        message=(
                            f"{vecino} ya tiene una distancia definitiva; "
                            f"la arista {actual}-{vecino} no se relaja."
                        ),
                    )
                )
                continue

            distancia_anterior = distancias[vecino]
            candidata = distancias[actual] + peso

            if candidata < distancia_anterior:
                distancias[vecino] = candidata
                predecesores[vecino] = actual

                heapq.heappush(
                    cola_prioridad,
                    (candidata, vecino),
                )

                estados.append(
                    crear_estado_dijkstra(
                        current=actual,
                        distances=distancias,
                        predecessors=predecesores,
                        finalized=definitivos,
                        priority_queue=cola_prioridad,
                        active_edge=(actual, vecino),
                        action="improvement",
                        candidate=candidata,
                        old_distance=distancia_anterior,
                        edge_weight=peso,
                        message=(
                            f"Mejora para {vecino}: su distancia pasa de "
                            f"{distancia_anterior} a {candidata} y su "
                            f"predecesor pasa a ser {actual}."
                        ),
                    )
                )
            else:
                estados.append(
                    crear_estado_dijkstra(
                        current=actual,
                        distances=distancias,
                        predecessors=predecesores,
                        finalized=definitivos,
                        priority_queue=cola_prioridad,
                        active_edge=(actual, vecino),
                        action="no_improvement",
                        candidate=candidata,
                        old_distance=distancia_anterior,
                        edge_weight=peso,
                        message=(
                            f"La ruta hacia {vecino} pasando por {actual} "
                            f"costaría {candidata}; no mejora su distancia "
                            f"actual ({distancia_anterior})."
                        ),
                    )
                )

    camino_minimo = reconstruir_camino(
        predecesores=predecesores,
        origen=origen,
        destino=destino,
    )

    estados.append(
        crear_estado_dijkstra(
            current=None,
            distances=distancias,
            predecessors=predecesores,
            finalized=definitivos,
            priority_queue=cola_prioridad,
            action="finished",
            final_path=camino_minimo,
            message=(
                "Dijkstra ha terminado: la cola está vacía, todas las "
                "distancias alcanzables son definitivas y se resalta el "
                f"camino mínimo hasta {destino}."
            ),
        )
    )

    return {
        "states": estados,
        "distances": distancias,
        "predecessors": predecesores,
        "finalization_order": orden_definitivo,
        "path": camino_minimo,
        "path_cost": distancias[destino],
    }


def comprobar_con_networkx(grafo, origen, resultado):
    """
    Verifica distancias y camino mínimo con la implementación de NetworkX.
    """

    distancias_networkx, caminos_networkx = nx.single_source_dijkstra(
        grafo,
        source=origen,
        weight="weight",
    )

    distancias_coinciden = all(
        resultado["distances"][nodo] == distancias_networkx.get(
            nodo,
            float("inf"),
        )
        for nodo in grafo.nodes()
    )

    camino_networkx = caminos_networkx.get(
        resultado["path"][-1],
        [],
    ) if resultado["path"] else []

    camino_coincide = resultado["path"] == camino_networkx

    return {
        "distances_match": distancias_coinciden,
        "path_match": camino_coincide,
        "networkx_path": camino_networkx,
    }


def formatear_valor(valor):
    """
    Formatea infinito y valores enteros para la salida por terminal.
    """

    if valor == float("inf"):
        return "∞"

    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))

    return str(valor)


def imprimir_resultado_dijkstra(resultado, comprobacion):
    """
    Muestra por terminal los resultados principales.
    """

    print("\n=== Caminos mínimos con Dijkstra ===")

    print("\nOrden en el que las distancias se vuelven definitivas:")
    print(" -> ".join(resultado["finalization_order"]))

    print("\nDistancias y predecesores finales:")

    for nodo in sorted(resultado["distances"]):
        distancia = formatear_valor(resultado["distances"][nodo])
        predecesor = resultado["predecessors"][nodo]
        predecesor_texto = "—" if predecesor is None else predecesor

        print(
            f"  {nodo}: distancia {distancia}, "
            f"predecesor {predecesor_texto}"
        )

    print("\nCamino mínimo hasta el destino:")

    if resultado["path"]:
        print("  " + " -> ".join(resultado["path"]))
        print(
            "  Coste total: "
            + formatear_valor(resultado["path_cost"])
        )
    else:
        print("  El destino no es alcanzable desde el origen.")

    print("\nComprobación con NetworkX:")
    print(
        "  ¿Coinciden todas las distancias?: "
        f"{comprobacion['distances_match']}"
    )
    print(
        "  ¿Coincide el camino mínimo?: "
        f"{comprobacion['path_match']}"
    )

    print("\nEstado final de la cola de prioridad:")
    print("  Cola vacía")


def main():
    grafo = crear_grafo_dijkstra()
    posiciones = obtener_posiciones()

    origen = "A"
    destino = "L"

    resultado = ejecutar_dijkstra_por_pasos(
        grafo=grafo,
        origen=origen,
        destino=destino,
    )

    comprobacion = comprobar_con_networkx(
        grafo=grafo,
        origen=origen,
        resultado=resultado,
    )

    imprimir_resultado_dijkstra(
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
        / "03_caminos_minimos_dijkstra.png"
    )

    # La imagen final se guarda antes de abrir la animación.
    animacion = animador.animate_dijkstra(
        graph=grafo,
        pos=posiciones,
        states=resultado["states"],
        source_node=origen,
        target_node=destino,
        title="Caminos mínimos con Dijkstra",
        final_image_path=output_path,
        repeat=False,
    )

    # Mantiene viva la referencia mientras la ventana está abierta.
    _ = animacion


if __name__ == "__main__":
    main()