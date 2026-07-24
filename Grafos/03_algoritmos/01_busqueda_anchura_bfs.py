from collections import deque
from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent
sys.path.append(str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


def crear_grafo_bfs():
    """
    Crea un grafo no dirigido suficientemente grande para visualizar BFS.

    El grafo tiene cuatro niveles claros desde el vértice A:
    - nivel 0: A,
    - nivel 1: B, C, D,
    - nivel 2: E, F, G, H, I, J,
    - nivel 3: K, L, M, N, O, P, Q, R, S, T.

    También se incluyen algunas aristas laterales entre nodos del mismo
    nivel. Estas aristas muestran que BFS no vuelve a insertar en la cola
    un vértice que ya fue descubierto.
    """

    grafo = nx.Graph()

    grafo.add_edges_from([
        # Nivel 0 -> nivel 1
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),

        # Nivel 1 -> nivel 2
        ("B", "E"),
        ("B", "F"),
        ("C", "G"),
        ("C", "H"),
        ("D", "I"),
        ("D", "J"),

        # Nivel 2 -> nivel 3
        ("E", "K"),
        ("E", "L"),
        ("F", "M"),
        ("G", "N"),
        ("G", "O"),
        ("H", "P"),
        ("I", "Q"),
        ("I", "R"),
        ("J", "S"),
        ("J", "T"),

        # Aristas laterales que no cambian los niveles BFS
        ("F", "G"),
        ("H", "I"),
        ("O", "P"),
        ("R", "S"),
    ])

    return grafo


def obtener_posiciones():
    """
    Define posiciones manuales para que los niveles de BFS se vean
    claramente de arriba hacia abajo.
    """

    return {
        # Nivel 0
        "A": (0.0, 4.5),

        # Nivel 1
        "B": (-6.0, 3.2),
        "C": (0.0, 3.2),
        "D": (6.0, 3.2),

        # Nivel 2
        "E": (-8.0, 1.8),
        "F": (-5.0, 1.8),
        "G": (-2.0, 1.8),
        "H": (1.0, 1.8),
        "I": (4.0, 1.8),
        "J": (7.0, 1.8),

        # Nivel 3
        "K": (-9.0, 0.2),
        "L": (-7.0, 0.2),
        "M": (-5.0, 0.2),
        "N": (-3.0, 0.2),
        "O": (-1.0, 0.2),
        "P": (1.0, 0.2),
        "Q": (3.0, 0.2),
        "R": (5.0, 0.2),
        "S": (7.0, 0.2),
        "T": (9.0, 0.2),
    }


def crear_estado(
    current,
    discovered,
    processed,
    queue,
    levels,
    tree_edges,
    message,
    active_edge=None,
):
    """
    Crea una copia inmutable del estado actual de BFS.

    Es importante copiar conjuntos, listas y diccionarios porque el
    algoritmo continúa modificándolos en los pasos posteriores.
    """

    return {
        "current": current,
        "discovered": set(discovered),
        "processed": set(processed),
        "queue": list(queue),
        "levels": dict(levels),
        "tree_edges": list(tree_edges),
        "active_edge": active_edge,
        "message": message,
    }


def ejecutar_bfs_por_pasos(grafo, vertice_inicial):
    """
    Ejecuta BFS y registra todos los estados necesarios para animarlo.

    La cola usada es FIFO:
    - se extrae por la izquierda con popleft(),
    - los nuevos vecinos se insertan por la derecha con append().

    Cada vecino nuevo genera un estado independiente. Así se observa cómo
    los vértices se descubren uno a uno y cómo cambia la cola.
    """

    if vertice_inicial not in grafo:
        raise ValueError(
            f"El vértice inicial {vertice_inicial!r} no existe en el grafo."
        )

    cola = deque([vertice_inicial])

    descubiertos = {vertice_inicial}
    procesados = set()

    niveles = {
        vertice_inicial: 0,
    }

    padres = {
        vertice_inicial: None,
    }

    aristas_arbol = []
    orden_bfs = []
    estados = []

    estados.append(
        crear_estado(
            current=None,
            discovered=descubiertos,
            processed=procesados,
            queue=cola,
            levels=niveles,
            tree_edges=aristas_arbol,
            message=(
                f"Se introduce {vertice_inicial} en la cola con nivel 0."
            ),
        )
    )

    while cola:
        actual = cola.popleft()
        orden_bfs.append(actual)

        estados.append(
            crear_estado(
                current=actual,
                discovered=descubiertos,
                processed=procesados,
                queue=cola,
                levels=niveles,
                tree_edges=aristas_arbol,
                message=(
                    f"Sale {actual} de la cola. "
                    "Ahora se examinan sus vecinos."
                ),
            )
        )

        # Ordenar hace que la ejecución sea determinista y fácil de seguir.
        for vecino in sorted(grafo.neighbors(actual)):
            if vecino in descubiertos:
                estados.append(
                    crear_estado(
                        current=actual,
                        discovered=descubiertos,
                        processed=procesados,
                        queue=cola,
                        levels=niveles,
                        tree_edges=aristas_arbol,
                        active_edge=(actual, vecino),
                        message=(
                            f"{vecino} ya había sido descubierto, "
                            "por lo que no se vuelve a insertar en la cola."
                        ),
                    )
                )
                continue

            descubiertos.add(vecino)
            padres[vecino] = actual
            niveles[vecino] = niveles[actual] + 1

            cola.append(vecino)
            aristas_arbol.append((actual, vecino))

            estados.append(
                crear_estado(
                    current=actual,
                    discovered=descubiertos,
                    processed=procesados,
                    queue=cola,
                    levels=niveles,
                    tree_edges=aristas_arbol,
                    active_edge=(actual, vecino),
                    message=(
                        f"Se descubre {vecino}: nivel {niveles[vecino]}. "
                        "Se añade al final de la cola."
                    ),
                )
            )

        procesados.add(actual)

        estados.append(
            crear_estado(
                current=None,
                discovered=descubiertos,
                processed=procesados,
                queue=cola,
                levels=niveles,
                tree_edges=aristas_arbol,
                message=(
                    f"{actual} queda procesado. "
                    f"La cola contiene {len(cola)} elemento(s)."
                ),
            )
        )

    estados.append(
        crear_estado(
            current=None,
            discovered=descubiertos,
            processed=procesados,
            queue=cola,
            levels=niveles,
            tree_edges=aristas_arbol,
            message=(
                "BFS ha terminado: la cola está vacía y todos los "
                "vértices alcanzables tienen asignado un nivel."
            ),
        )
    )

    return {
        "states": estados,
        "order": orden_bfs,
        "levels": niveles,
        "parents": padres,
        "tree_edges": aristas_arbol,
    }


def imprimir_resultado_bfs(resultado):
    """
    Muestra por terminal el resultado principal del recorrido.
    """

    print("\n=== Búsqueda en anchura (BFS) ===")

    print("\nOrden de procesamiento:")
    print(" -> ".join(resultado["order"]))

    print("\nNiveles respecto al vértice inicial:")
    for nodo, nivel in sorted(
        resultado["levels"].items(),
        key=lambda elemento: (elemento[1], elemento[0]),
    ):
        print(f"  {nodo}: nivel {nivel}")

    print("\nPadres en el árbol BFS:")
    for nodo, padre in resultado["parents"].items():
        if padre is None:
            print(f"  {nodo}: raíz")
        else:
            print(f"  {nodo}: padre {padre}")

    print("\nAristas del árbol BFS:")
    for origen, destino in resultado["tree_edges"]:
        print(f"  {origen} - {destino}")

    print("\nEstado final de la cola:")
    print("  Cola vacía")


def main():
    grafo = crear_grafo_bfs()
    posiciones = obtener_posiciones()
    vertice_inicial = "A"

    resultado = ejecutar_bfs_por_pasos(
        grafo,
        vertice_inicial,
    )

    imprimir_resultado_bfs(resultado)

    animador = GraphAnimator(
        figsize=(15, 9),
        interval=850,
    )

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "03_algoritmos"
        / "01_busqueda_anchura_bfs.png"
    )

    # La imagen final se guarda automáticamente antes de mostrar
    # la animación. El vídeo se puede grabar durante la ejecución.
    animacion = animador.animate_bfs(
        graph=grafo,
        pos=posiciones,
        states=resultado["states"],
        start_node=vertice_inicial,
        title="Búsqueda en anchura (BFS)",
        final_image_path=output_path,
        repeat=False,
    )

    # Mantener la referencia mientras la ventana está abierta.
    _ = animacion


if __name__ == "__main__":
    main()