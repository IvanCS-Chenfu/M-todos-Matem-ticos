from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator

def crear_grafo_dfs():
    """
    Crea el mismo grafo general utilizado en el ejemplo de BFS.

    Mantener el grafo permite comparar directamente ambos recorridos:
    - BFS explora por niveles,
    - DFS profundiza por una rama antes de retroceder.

    Las aristas laterales introducen ciclos. De esta forma, la animación
    también muestra cómo DFS encuentra conexiones hacia vértices que ya
    están activos o que ya fueron visitados.
    """

    grafo = nx.Graph()

    grafo.add_edges_from([
        # Conexiones desde la raíz
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),

        # Segunda zona del grafo
        ("B", "E"),
        ("B", "F"),
        ("C", "G"),
        ("C", "H"),
        ("D", "I"),
        ("D", "J"),

        # Ramas exteriores
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

        # Aristas laterales que generan ciclos
        ("F", "G"),
        ("H", "I"),
        ("O", "P"),
        ("R", "S"),
    ])

    return grafo


def obtener_posiciones():
    """
    Mantiene las mismas posiciones del ejemplo BFS.

    Esto facilita comparar visualmente los árboles de recorrido generados
    por BFS y DFS sobre el mismo grafo.
    """

    return {
        "A": (0.0, 4.5),

        "B": (-6.0, 3.2),
        "C": (0.0, 3.2),
        "D": (6.0, 3.2),

        "E": (-8.0, 1.8),
        "F": (-5.0, 1.8),
        "G": (-2.0, 1.8),
        "H": (1.0, 1.8),
        "I": (4.0, 1.8),
        "J": (7.0, 1.8),

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


def normalizar_arista(origen, destino):
    """
    Normaliza una arista de un grafo no dirigido.
    """

    return frozenset((origen, destino))


def crear_estado_dfs(
    current,
    discovered,
    finished,
    stack,
    depths,
    discovery_times,
    finish_times,
    tree_edges,
    cycle_edges,
    message,
    active_edge=None,
    edge_kind=None,
):
    """
    Crea una copia independiente del estado actual de DFS.

    Las copias evitan que los estados antiguos cambien cuando el algoritmo
    modifica sus conjuntos, listas y diccionarios.
    """

    return {
        "current": current,
        "discovered": set(discovered),
        "finished": set(finished),
        "stack": list(stack),
        "depths": dict(depths),
        "discovery_times": dict(discovery_times),
        "finish_times": dict(finish_times),
        "tree_edges": list(tree_edges),
        "cycle_edges": list(cycle_edges),
        "active_edge": active_edge,
        "edge_kind": edge_kind,
        "message": message,
    }


def ejecutar_dfs_por_pasos(grafo, vertice_inicial):
    """
    Ejecuta DFS de forma recursiva y registra todos sus estados.

    La pila de llamadas se representa explícitamente mediante
    ``camino_activo``. Cada vez que se descubre un vértice:

    - se añade al final de la pila,
    - se registra su profundidad,
    - se asigna un tiempo de descubrimiento.

    Cuando todos sus vecinos ya han sido examinados:

    - se asigna un tiempo de finalización,
    - se elimina de la pila,
    - se produce el retroceso hacia su padre.

    El orden de los vecinos se fija alfabéticamente para que la ejecución
    sea determinista y fácil de comparar.
    """

    if vertice_inicial not in grafo:
        raise ValueError(
            f"El vértice inicial {vertice_inicial!r} no existe en el grafo."
        )

    descubiertos = set()
    finalizados = set()
    camino_activo = []

    profundidades = {}
    padres = {}

    tiempos_descubrimiento = {}
    tiempos_finalizacion = {}

    aristas_arbol = []
    aristas_ciclo = []

    aristas_no_arbol_clasificadas = set()

    orden_descubrimiento = []
    orden_finalizacion = []
    estados = []

    reloj = 0

    def guardar_estado(
        actual,
        mensaje,
        arista_activa=None,
        tipo_arista=None,
    ):
        estados.append(
            crear_estado_dfs(
                current=actual,
                discovered=descubiertos,
                finished=finalizados,
                stack=camino_activo,
                depths=profundidades,
                discovery_times=tiempos_descubrimiento,
                finish_times=tiempos_finalizacion,
                tree_edges=aristas_arbol,
                cycle_edges=aristas_ciclo,
                active_edge=arista_activa,
                edge_kind=tipo_arista,
                message=mensaje,
            )
        )

    def visitar(actual, padre, profundidad):
        nonlocal reloj

        descubiertos.add(actual)
        camino_activo.append(actual)

        padres[actual] = padre
        profundidades[actual] = profundidad

        reloj += 1
        tiempos_descubrimiento[actual] = reloj
        orden_descubrimiento.append(actual)

        guardar_estado(
            actual=actual,
            mensaje=(
                f"Se descubre {actual}: profundidad {profundidad}, "
                f"tiempo de descubrimiento {reloj}."
            ),
        )

        for vecino in sorted(grafo.neighbors(actual)):
            # En un grafo no dirigido, la conexión directa con el padre
            # no debe interpretarse como un ciclo.
            if vecino == padre:
                continue

            arista_normalizada = normalizar_arista(actual, vecino)

            if vecino not in descubiertos:
                aristas_arbol.append((actual, vecino))

                guardar_estado(
                    actual=actual,
                    arista_activa=(actual, vecino),
                    tipo_arista="tree",
                    mensaje=(
                        f"{vecino} no estaba descubierto. "
                        f"DFS avanza desde {actual} hacia {vecino}."
                    ),
                )

                visitar(
                    actual=vecino,
                    padre=actual,
                    profundidad=profundidad + 1,
                )

                # El retroceso ya queda registrado dentro de la llamada hija.
                continue

            # Evita clasificar dos veces una misma arista no dirigida.
            if (
                arista_normalizada
                in aristas_no_arbol_clasificadas
            ):
                continue

            aristas_no_arbol_clasificadas.add(arista_normalizada)

            if vecino in camino_activo:
                aristas_ciclo.append((actual, vecino))

                guardar_estado(
                    actual=actual,
                    arista_activa=(actual, vecino),
                    tipo_arista="cycle",
                    mensaje=(
                        f"La arista {actual}-{vecino} llega a un antecesor "
                        "que todavía está activo: se identifica un ciclo."
                    ),
                )
            else:
                guardar_estado(
                    actual=actual,
                    arista_activa=(actual, vecino),
                    tipo_arista="visited",
                    mensaje=(
                        f"{vecino} ya había finalizado. "
                        "La arista no pertenece al árbol DFS."
                    ),
                )

        reloj += 1
        tiempos_finalizacion[actual] = reloj
        finalizados.add(actual)
        orden_finalizacion.append(actual)

        guardar_estado(
            actual=actual,
            mensaje=(
                f"{actual} queda finalizado en el tiempo {reloj}: "
                "ya no tiene vecinos pendientes."
            ),
        )

        camino_activo.pop()

        if padre is not None:
            guardar_estado(
                actual=padre,
                arista_activa=(actual, padre),
                tipo_arista="backtrack",
                mensaje=(
                    f"Retroceso desde {actual} hasta {padre}. "
                    f"La cima de la pila vuelve a ser {padre}."
                ),
            )

    padres[vertice_inicial] = None

    # Estado inicial: la pila todavía está vacía.
    guardar_estado(
        actual=None,
        mensaje=(
            f"DFS comenzará en {vertice_inicial}. "
            "La pila está inicialmente vacía."
        ),
    )

    visitar(
        actual=vertice_inicial,
        padre=None,
        profundidad=0,
    )

    guardar_estado(
        actual=None,
        mensaje=(
            "DFS ha terminado: la pila está vacía y todos los vértices "
            "alcanzables han sido finalizados."
        ),
    )

    return {
        "states": estados,
        "discovery_order": orden_descubrimiento,
        "finish_order": orden_finalizacion,
        "depths": profundidades,
        "parents": padres,
        "discovery_times": tiempos_descubrimiento,
        "finish_times": tiempos_finalizacion,
        "tree_edges": aristas_arbol,
        "cycle_edges": aristas_ciclo,
    }


def imprimir_resultado_dfs(resultado):
    """
    Muestra por terminal los resultados principales de DFS.
    """

    print("\n=== Búsqueda en profundidad (DFS) ===")

    print("\nOrden de descubrimiento:")
    print(" -> ".join(resultado["discovery_order"]))

    print("\nOrden de finalización:")
    print(" -> ".join(resultado["finish_order"]))

    print("\nProfundidad, descubrimiento y finalización:")
    for nodo in resultado["discovery_order"]:
        profundidad = resultado["depths"][nodo]
        descubrimiento = resultado["discovery_times"][nodo]
        finalizacion = resultado["finish_times"][nodo]

        print(
            f"  {nodo}: profundidad {profundidad}, "
            f"descubrimiento {descubrimiento}, "
            f"finalización {finalizacion}"
        )

    print("\nPadres en el árbol DFS:")
    for nodo in resultado["discovery_order"]:
        padre = resultado["parents"][nodo]

        if padre is None:
            print(f"  {nodo}: raíz")
        else:
            print(f"  {nodo}: padre {padre}")

    print("\nAristas del árbol DFS:")
    for origen, destino in resultado["tree_edges"]:
        print(f"  {origen} - {destino}")

    print("\nAristas que identifican ciclos durante el recorrido:")
    if resultado["cycle_edges"]:
        for origen, destino in resultado["cycle_edges"]:
            print(f"  {origen} - {destino}")
    else:
        print("  No se encontraron aristas de ciclo.")

    print("\nEstado final de la pila:")
    print("  Pila vacía")


def main():
    grafo = crear_grafo_dfs()
    posiciones = obtener_posiciones()
    vertice_inicial = "A"

    resultado = ejecutar_dfs_por_pasos(
        grafo,
        vertice_inicial,
    )

    imprimir_resultado_dfs(resultado)

    animador = GraphAnimator(
        figsize=(15, 9),
        interval=800,
    )

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "03_algoritmos"
        / "02_busqueda_profundidad_dfs.png"
    )

    # La imagen final se guarda automáticamente antes de mostrar
    # la animación. El vídeo se puede grabar durante la ejecución.
    animacion = animador.animate_dfs(
        graph=grafo,
        pos=posiciones,
        states=resultado["states"],
        start_node=vertice_inicial,
        title="Búsqueda en profundidad (DFS)",
        final_image_path=output_path,
        repeat=False,
    )

    # Mantener la referencia mientras la ventana está abierta.
    _ = animacion


if __name__ == "__main__":
    main()
