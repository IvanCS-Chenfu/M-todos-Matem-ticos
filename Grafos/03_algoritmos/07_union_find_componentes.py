from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


def obtener_aristas_union_find():
    """
    Devuelve las aristas en el orden utilizado por la animación.

    El orden se ha elegido para mostrar:
    - varias fusiones entre componentes;
    - árboles internos de más de un nivel;
    - compresión de caminos;
    - aristas que se rechazan por formar ciclos.
    """

    return [
        # Primera componente: A, B, C, D, E, F.
        ("A", "B"),
        ("C", "D"),
        ("B", "C"),
        ("E", "F"),
        ("D", "E"),
        ("A", "C"),
        ("B", "E"),
        ("C", "F"),

        # Segunda componente: G, H, I, J.
        ("G", "H"),
        ("I", "J"),
        ("H", "I"),
        ("G", "J"),
        ("H", "J"),

        # Tercera componente: K, L, M.
        ("K", "L"),
        ("L", "M"),
        ("K", "M"),
    ]


def crear_grafo_union_find():
    """
    Crea un grafo no dirigido con cuatro componentes conectadas.

    Componentes finales:
    - {A, B, C, D, E, F};
    - {G, H, I, J};
    - {K, L, M};
    - {N}, vértice aislado.
    """

    grafo = nx.Graph()

    grafo.add_nodes_from(list("ABCDEFGHIJKLMN"))
    grafo.add_edges_from(obtener_aristas_union_find())

    grafo.graph["edge_order"] = obtener_aristas_union_find()

    return grafo


def obtener_posiciones():
    """
    Define posiciones manuales separando visualmente las componentes.
    """

    return {
        # Componente 1.
        "A": (-6.0, 2.5),
        "B": (-4.0, 3.5),
        "C": (-2.0, 2.5),
        "D": (-2.0, 0.5),
        "E": (-4.0, -0.5),
        "F": (-6.0, 0.5),

        # Componente 2.
        "G": (1.0, 3.0),
        "H": (3.0, 3.0),
        "I": (3.0, 0.8),
        "J": (1.0, 0.8),

        # Componente 3.
        "K": (6.0, 2.8),
        "L": (7.5, 1.3),
        "M": (5.5, 0.4),

        # Vértice aislado.
        "N": (7.2, -1.7),
    }


class UnionFind:
    """
    Estructura de conjuntos disjuntos con:

    - compresión de caminos;
    - unión por rango;
    - tamaño almacenado en las raíces.
    """

    def __init__(self, elementos):
        self.parent = {
            elemento: elemento
            for elemento in elementos
        }
        self.rank = {
            elemento: 0
            for elemento in elementos
        }
        self.size = {
            elemento: 1
            for elemento in elementos
        }

    def raiz_sin_comprimir(self, elemento):
        """
        Obtiene la raíz sin modificar la estructura.
        """

        actual = elemento

        while self.parent[actual] != actual:
            actual = self.parent[actual]

        return actual

    def find(self, elemento):
        """
        Devuelve la raíz aplicando compresión de caminos.
        """

        if self.parent[elemento] != elemento:
            self.parent[elemento] = self.find(
                self.parent[elemento]
            )

        return self.parent[elemento]

    def find_con_traza(self, elemento):
        """
        Ejecuta find y devuelve:

        - raíz;
        - camino recorrido antes de comprimir;
        - cambios de padre producidos por la compresión.
        """

        camino = [elemento]
        actual = elemento

        while self.parent[actual] != actual:
            actual = self.parent[actual]
            camino.append(actual)

        raiz = actual
        cambios = []

        for nodo in camino[:-1]:
            padre_anterior = self.parent[nodo]

            if padre_anterior != raiz:
                self.parent[nodo] = raiz
                cambios.append(
                    (
                        nodo,
                        padre_anterior,
                        raiz,
                    )
                )

        return raiz, camino, cambios

    def union_raices(self, raiz_a, raiz_b):
        """
        Fusiona dos raíces mediante unión por rango.

        Devuelve información sobre la raíz que queda por encima.
        """

        if raiz_a == raiz_b:
            return {
                "merged": False,
                "new_root": raiz_a,
                "attached_root": None,
                "rank_increased": False,
            }

        if self.rank[raiz_a] < self.rank[raiz_b]:
            raiz_a, raiz_b = raiz_b, raiz_a

        self.parent[raiz_b] = raiz_a
        self.size[raiz_a] += self.size[raiz_b]

        rank_increased = False

        if self.rank[raiz_a] == self.rank[raiz_b]:
            self.rank[raiz_a] += 1
            rank_increased = True

        return {
            "merged": True,
            "new_root": raiz_a,
            "attached_root": raiz_b,
            "rank_increased": rank_increased,
        }

    def snapshot(self):
        """
        Devuelve una instantánea sin alterar los padres.
        """

        roots = {
            elemento: self.raiz_sin_comprimir(elemento)
            for elemento in self.parent
        }

        groups_by_root = {}

        for elemento, raiz in roots.items():
            groups_by_root.setdefault(raiz, []).append(elemento)

        components = []

        for raiz, elementos in groups_by_root.items():
            elementos_ordenados = sorted(elementos)
            components.append(
                (
                    elementos_ordenados[0],
                    elementos_ordenados,
                    raiz,
                )
            )

        components.sort(key=lambda item: item[0])

        component_map = {}

        for canonical_label, elementos, _ in components:
            for elemento in elementos:
                component_map[elemento] = canonical_label

        return {
            "parents": dict(self.parent),
            "ranks": dict(self.rank),
            "sizes": dict(self.size),
            "roots": roots,
            "components": [
                (canonical_label, elementos)
                for canonical_label, elementos, _ in components
            ],
            "component_map": component_map,
        }


def normalizar_componentes(componentes):
    """
    Convierte una colección de componentes en un conjunto comparable.
    """

    return {
        frozenset(componente)
        for componente in componentes
    }


def calcular_componentes_dfs(grafo):
    """
    Calcula las componentes mediante una DFS iterativa determinista.
    """

    visitados = set()
    componentes = []

    for inicio in sorted(grafo.nodes()):
        if inicio in visitados:
            continue

        componente = []
        pila = [inicio]

        while pila:
            actual = pila.pop()

            if actual in visitados:
                continue

            visitados.add(actual)
            componente.append(actual)

            vecinos = sorted(
                grafo.neighbors(actual),
                reverse=True,
            )

            for vecino in vecinos:
                if vecino not in visitados:
                    pila.append(vecino)

        componentes.append(sorted(componente))

    return componentes


def obtener_orden_aristas(grafo):
    """
    Recupera el orden explícito usado por Union-Find.
    """

    return list(
        grafo.graph.get(
            "edge_order",
            sorted(grafo.edges()),
        )
    )


def describir_compresion(cambios):
    """
    Convierte los cambios de compresión en una explicación breve.
    """

    if not cambios:
        return "sin cambios de padre"

    return ", ".join(
        f"{nodo}: {padre_anterior}→{raiz}"
        for nodo, padre_anterior, raiz in cambios
    )


def crear_estado_union_find(
    union_find,
    edge_order,
    edge_statuses,
    accepted_edges,
    rejected_edges,
    message,
    phase,
    active_edge=None,
    active_edge_index=None,
    action=None,
    find_path_u=None,
    find_path_v=None,
    root_u=None,
    root_v=None,
    compression_changes=None,
    union_info=None,
    dfs_components=None,
    networkx_components=None,
):
    """
    Crea una copia independiente del estado actual.
    """

    snapshot = union_find.snapshot()

    return {
        **snapshot,
        "edge_order": list(edge_order),
        "edge_statuses": dict(edge_statuses),
        "accepted_edges": list(accepted_edges),
        "rejected_edges": list(rejected_edges),
        "active_edge": active_edge,
        "active_edge_index": active_edge_index,
        "action": action,
        "find_path_u": list(find_path_u or []),
        "find_path_v": list(find_path_v or []),
        "root_u": root_u,
        "root_v": root_v,
        "compression_changes": list(
            compression_changes or []
        ),
        "union_info": dict(union_info or {}),
        "dfs_components": [
            list(component)
            for component in (dfs_components or [])
        ],
        "networkx_components": [
            list(component)
            for component in (networkx_components or [])
        ],
        "phase": phase,
        "message": message,
    }


def ejecutar_union_find_por_pasos(grafo):
    """
    Procesa todas las aristas con Union-Find.

    Para cada arista se registran dos estados principales:

    1. ejecución de find sobre ambos extremos;
    2. unión de componentes o rechazo por ciclo.

    Al final se comprimen todos los caminos y se comparan las componentes
    con una DFS manual y con NetworkX.
    """

    edge_order = obtener_orden_aristas(grafo)
    union_find = UnionFind(sorted(grafo.nodes()))

    edge_statuses = {}
    accepted_edges = []
    rejected_edges = []
    states = []

    states.append(
        crear_estado_union_find(
            union_find=union_find,
            edge_order=edge_order,
            edge_statuses=edge_statuses,
            accepted_edges=accepted_edges,
            rejected_edges=rejected_edges,
            message=(
                "Inicialización: cada vértice forma su propia componente. "
                "El vértice N permanecerá aislado."
            ),
            phase="initial",
        )
    )

    for edge_index, (origin, destination) in enumerate(edge_order):
        root_origin, path_origin, changes_origin = (
            union_find.find_con_traza(origin)
        )
        root_destination, path_destination, changes_destination = (
            union_find.find_con_traza(destination)
        )

        all_changes = changes_origin + changes_destination

        states.append(
            crear_estado_union_find(
                union_find=union_find,
                edge_order=edge_order,
                edge_statuses=edge_statuses,
                accepted_edges=accepted_edges,
                rejected_edges=rejected_edges,
                active_edge=(origin, destination),
                active_edge_index=edge_index,
                action="inspect",
                find_path_u=path_origin,
                find_path_v=path_destination,
                root_u=root_origin,
                root_v=root_destination,
                compression_changes=all_changes,
                message=(
                    f"Se examina {origin}—{destination}. "
                    f"Compresión: {describir_compresion(all_changes)}."
                ),
                phase="processing",
            )
        )

        if root_origin != root_destination:
            union_info = union_find.union_raices(
                root_origin,
                root_destination,
            )

            accepted_edges.append((origin, destination))
            edge_statuses[edge_index] = "accepted"

            new_root = union_info["new_root"]
            attached_root = union_info["attached_root"]

            rank_message = (
                " El rango de la nueva raíz aumenta."
                if union_info["rank_increased"]
                else ""
            )

            states.append(
                crear_estado_union_find(
                    union_find=union_find,
                    edge_order=edge_order,
                    edge_statuses=edge_statuses,
                    accepted_edges=accepted_edges,
                    rejected_edges=rejected_edges,
                    active_edge=(origin, destination),
                    active_edge_index=edge_index,
                    action="accepted",
                    find_path_u=path_origin,
                    find_path_v=path_destination,
                    root_u=root_origin,
                    root_v=root_destination,
                    compression_changes=all_changes,
                    union_info=union_info,
                    message=(
                        f"Raíces diferentes: {root_origin} y "
                        f"{root_destination}. Se acepta la arista y "
                        f"{attached_root} pasa a depender de {new_root}."
                        f"{rank_message}"
                    ),
                    phase="processing",
                )
            )
        else:
            rejected_edges.append((origin, destination))
            edge_statuses[edge_index] = "rejected"

            states.append(
                crear_estado_union_find(
                    union_find=union_find,
                    edge_order=edge_order,
                    edge_statuses=edge_statuses,
                    accepted_edges=accepted_edges,
                    rejected_edges=rejected_edges,
                    active_edge=(origin, destination),
                    active_edge_index=edge_index,
                    action="rejected",
                    find_path_u=path_origin,
                    find_path_v=path_destination,
                    root_u=root_origin,
                    root_v=root_destination,
                    compression_changes=all_changes,
                    message=(
                        f"Ambos extremos tienen la raíz {root_origin}. "
                        "La arista se rechaza porque cerraría un ciclo."
                    ),
                    phase="processing",
                )
            )

    # Compresión final de todos los caminos.
    final_compression_changes = []

    for node in sorted(grafo.nodes()):
        _, _, changes = union_find.find_con_traza(node)
        final_compression_changes.extend(changes)

    states.append(
        crear_estado_union_find(
            union_find=union_find,
            edge_order=edge_order,
            edge_statuses=edge_statuses,
            accepted_edges=accepted_edges,
            rejected_edges=rejected_edges,
            compression_changes=final_compression_changes,
            message=(
                "Compresión final: todos los vértices apuntan "
                "directamente al representante de su componente."
            ),
            phase="compression",
        )
    )

    dfs_components = calcular_componentes_dfs(grafo)
    networkx_components = [
        sorted(component)
        for component in nx.connected_components(grafo)
    ]
    networkx_components.sort(key=lambda component: component[0])

    final_snapshot = union_find.snapshot()
    union_find_components = [
        component
        for _, component in final_snapshot["components"]
    ]

    states.append(
        crear_estado_union_find(
            union_find=union_find,
            edge_order=edge_order,
            edge_statuses=edge_statuses,
            accepted_edges=accepted_edges,
            rejected_edges=rejected_edges,
            dfs_components=dfs_components,
            networkx_components=networkx_components,
            message=(
                "Resultado final: Union-Find, DFS y NetworkX coinciden "
                "en cuatro componentes. Las seis aristas rechazadas "
                "corresponden a conexiones que formarían ciclos."
            ),
            phase="finished",
        )
    )

    return {
        "states": states,
        "parents": dict(union_find.parent),
        "ranks": dict(union_find.rank),
        "sizes": dict(union_find.size),
        "components": union_find_components,
        "dfs_components": dfs_components,
        "networkx_components": networkx_components,
        "accepted_edges": accepted_edges,
        "rejected_edges": rejected_edges,
        "edge_statuses": edge_statuses,
    }


def comprobar_con_networkx(grafo, resultado):
    """
    Compara Union-Find con DFS y NetworkX.

    También verifica la identidad del rango cíclico:

        E - V + C

    que coincide con el número de aristas rechazadas al construir un
    bosque de expansión.
    """

    components_union_find = normalizar_componentes(
        resultado["components"]
    )
    components_dfs = normalizar_componentes(
        resultado["dfs_components"]
    )
    components_networkx = normalizar_componentes(
        resultado["networkx_components"]
    )

    number_of_components = nx.number_connected_components(grafo)

    expected_forest_edges = (
        grafo.number_of_nodes()
        - number_of_components
    )

    cyclomatic_number = (
        grafo.number_of_edges()
        - grafo.number_of_nodes()
        + number_of_components
    )

    forest = nx.Graph()
    forest.add_nodes_from(grafo.nodes())
    forest.add_edges_from(resultado["accepted_edges"])

    rejected_edges_are_cycles = all(
        nx.has_path(forest, origin, destination)
        for origin, destination in resultado["rejected_edges"]
    )

    return {
        "union_find_matches_dfs": (
            components_union_find == components_dfs
        ),
        "union_find_matches_networkx": (
            components_union_find == components_networkx
        ),
        "accepted_edge_count_matches": (
            len(resultado["accepted_edges"])
            == expected_forest_edges
        ),
        "rejected_edge_count_matches": (
            len(resultado["rejected_edges"])
            == cyclomatic_number
        ),
        "rejected_edges_are_cycles": rejected_edges_are_cycles,
        "forest_is_acyclic": nx.is_forest(forest),
        "forest_component_count_matches": (
            nx.number_connected_components(forest)
            == number_of_components
        ),
    }


def imprimir_resultado_union_find(resultado, comprobacion):
    """
    Muestra los resultados principales por terminal.
    """

    print("\n=== Union-Find y componentes conectadas ===")

    print("\nComponentes finales:")

    for index, component in enumerate(
        resultado["components"],
        start=1,
    ):
        print(
            f"  C{index}: "
            + ", ".join(component)
        )

    print(
        "\nNúmero de componentes: "
        f"{len(resultado['components'])}"
    )

    print(
        "Aristas aceptadas: "
        f"{len(resultado['accepted_edges'])}"
    )
    print(
        "Aristas rechazadas por ciclo: "
        f"{len(resultado['rejected_edges'])}"
    )

    print("\nAristas aceptadas:")

    for origin, destination in resultado["accepted_edges"]:
        print(f"  {origin} — {destination}")

    print("\nAristas rechazadas:")

    for origin, destination in resultado["rejected_edges"]:
        print(f"  {origin} — {destination}")

    print("\nPadres finales:")

    for node in sorted(resultado["parents"]):
        print(
            f"  {node}: padre={resultado['parents'][node]}, "
            f"rango={resultado['ranks'][node]}, "
            f"tamaño={resultado['sizes'][node]}"
        )

    print("\nComprobación:")
    for name, value in comprobacion.items():
        print(f"  {name}: {value}")


def main():
    grafo = crear_grafo_union_find()
    posiciones = obtener_posiciones()

    resultado = ejecutar_union_find_por_pasos(grafo)

    comprobacion = comprobar_con_networkx(
        grafo=grafo,
        resultado=resultado,
    )

    imprimir_resultado_union_find(
        resultado=resultado,
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
        / "07_union_find_componentes.png"
    )

    animacion = animador.animate_union_find_components(
        graph=grafo,
        pos=posiciones,
        states=resultado["states"],
        title="Union-Find y componentes conectadas",
        final_image_path=output_path,
        repeat=False,
    )

    _ = animacion


if __name__ == "__main__":
    main()
