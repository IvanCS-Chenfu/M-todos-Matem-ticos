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


def crear_grafo_mst():
    """
    Crea un grafo no dirigido y ponderado para comparar Prim y Kruskal.

    Todos los pesos son distintos. Por tanto, el árbol de expansión
    mínima es único y ambos algoritmos deben seleccionar exactamente
    las mismas aristas, aunque lo hagan en un orden diferente.
    """

    grafo = nx.Graph()

    grafo.add_weighted_edges_from([
        ("E", "F", 1),
        ("B", "C", 2),
        ("F", "G", 3),
        ("A", "C", 4),
        ("H", "I", 5),
        ("C", "D", 6),
        ("G", "I", 7),
        ("B", "E", 8),
        ("I", "J", 9),

        ("C", "E", 10),
        ("D", "F", 11),
        ("F", "H", 12),
        ("A", "B", 13),
        ("D", "G", 15),
        ("E", "H", 16),
        ("C", "F", 17),
        ("F", "I", 18),
        ("A", "D", 19),
        ("H", "J", 20),
        ("G", "J", 21),
    ])

    return grafo


def obtener_posiciones():
    """
    Define posiciones manuales comparables con los ejemplos anteriores.
    """

    return {
        "A": (-6.0, 0.0),

        "B": (-3.0, 3.0),
        "C": (-3.0, 0.0),
        "D": (-3.0, -3.0),

        "E": (0.0, 3.0),
        "F": (0.0, 0.0),
        "G": (0.0, -3.0),

        "H": (3.0, 2.5),
        "I": (3.0, -0.5),

        "J": (6.0, 0.5),
    }


def normalizar_arista(origen, destino):
    """
    Devuelve una representación independiente del sentido.
    """

    return frozenset((origen, destino))


def validar_grafo_mst(grafo):
    """
    Comprueba las condiciones del problema clásico de MST.
    """

    if grafo.is_directed():
        raise ValueError(
            "Prim y Kruskal se aplican aquí a un grafo no dirigido."
        )

    if grafo.number_of_nodes() == 0:
        raise ValueError("El grafo no puede estar vacío.")

    if not nx.is_connected(grafo):
        raise ValueError(
            "El grafo debe ser conexo para obtener un único árbol "
            "de expansión. En otro caso se obtendría un bosque."
        )


def crear_cola_visible_prim(
    grafo,
    incluidos,
    claves,
    padres,
):
    """
    Construye una cola limpia con la mejor conexión de cada vértice externo.

    El montículo interno puede contener entradas antiguas. Para visualizar
    Prim se muestra únicamente la mejor arista actual de cada vértice que
    todavía no ha sido incluido.
    """

    cola = []

    for destino in grafo.nodes():
        if destino in incluidos:
            continue

        padre = padres.get(destino)
        clave = claves.get(destino, float("inf"))

        if padre is not None and clave != float("inf"):
            cola.append((clave, padre, destino))

    return sorted(cola)


def crear_estado_prim(
    grafo,
    incluidos,
    seleccionadas,
    rechazadas,
    claves,
    padres,
    coste_total,
    mensaje,
    actual=None,
    arista_activa=None,
    accion=None,
):
    """
    Crea una copia independiente de un estado de Prim.
    """

    return {
        "algorithm": "prim",
        "included": set(incluidos),
        "selected_edges": list(seleccionadas),
        "rejected_edges": list(rechazadas),
        "keys": dict(claves),
        "parents": dict(padres),
        "priority_queue": crear_cola_visible_prim(
            grafo=grafo,
            incluidos=incluidos,
            claves=claves,
            padres=padres,
        ),
        "current_node": actual,
        "active_edge": arista_activa,
        "action": accion,
        "total_cost": coste_total,
        "message": mensaje,
    }


def ejecutar_prim_por_pasos(
    grafo,
    inicio,
):
    """
    Ejecuta Prim con una cola de prioridad y registra sus estados.

    La clave de un vértice es el peso de la arista más barata conocida
    que lo conecta con el árbol actual. No es una distancia acumulada.
    """

    validar_grafo_mst(grafo)

    if inicio not in grafo:
        raise ValueError(
            f"El vértice inicial {inicio!r} no existe en el grafo."
        )

    incluidos = {inicio}
    seleccionadas = []
    rechazadas = []
    coste_total = 0

    claves = {
        nodo: float("inf")
        for nodo in grafo.nodes()
    }
    padres = {
        nodo: None
        for nodo in grafo.nodes()
    }

    claves[inicio] = 0

    # Entradas internas: (peso, origen, destino).
    cola = []
    estados = []

    estados.append(
        crear_estado_prim(
            grafo=grafo,
            incluidos=incluidos,
            seleccionadas=seleccionadas,
            rechazadas=rechazadas,
            claves=claves,
            padres=padres,
            coste_total=coste_total,
            actual=inicio,
            accion="initial",
            mensaje=(
                f"Prim comienza en {inicio}. El árbol contiene un único "
                "vértice y todavía no tiene aristas."
            ),
        )
    )

    # Inicializa la frontera del vértice de inicio.
    for vecino in sorted(grafo.neighbors(inicio)):
        peso = grafo[inicio][vecino].get("weight", 1)

        if peso < claves[vecino]:
            claves[vecino] = peso
            padres[vecino] = inicio
            heapq.heappush(
                cola,
                (peso, inicio, vecino),
            )

            estados.append(
                crear_estado_prim(
                    grafo=grafo,
                    incluidos=incluidos,
                    seleccionadas=seleccionadas,
                    rechazadas=rechazadas,
                    claves=claves,
                    padres=padres,
                    coste_total=coste_total,
                    actual=inicio,
                    arista_activa=(inicio, vecino),
                    accion="frontier_update",
                    mensaje=(
                        f"{inicio}—{vecino} entra en la frontera con "
                        f"peso {peso}. La clave de {vecino} pasa a {peso}."
                    ),
                )
            )

    while cola and len(incluidos) < grafo.number_of_nodes():
        peso, origen, destino = heapq.heappop(cola)

        es_entrada_valida = (
            destino not in incluidos
            and padres[destino] == origen
            and claves[destino] == peso
        )

        if not es_entrada_valida:
            rechazadas.append((origen, destino))

            estados.append(
                crear_estado_prim(
                    grafo=grafo,
                    incluidos=incluidos,
                    seleccionadas=seleccionadas,
                    rechazadas=rechazadas,
                    claves=claves,
                    padres=padres,
                    coste_total=coste_total,
                    actual=destino,
                    arista_activa=(origen, destino),
                    accion="stale",
                    mensaje=(
                        f"Se descarta {origen}—{destino} con peso {peso}: "
                        "la entrada está obsoleta o su destino ya está "
                        "incluido en el árbol."
                    ),
                )
            )
            continue

        incluidos.add(destino)
        seleccionadas.append((origen, destino))
        coste_total += peso

        estados.append(
            crear_estado_prim(
                grafo=grafo,
                incluidos=incluidos,
                seleccionadas=seleccionadas,
                rechazadas=rechazadas,
                claves=claves,
                padres=padres,
                coste_total=coste_total,
                actual=destino,
                arista_activa=(origen, destino),
                accion="accepted",
                mensaje=(
                    f"Prim selecciona {origen}—{destino} con peso {peso}. "
                    f"{destino} se incorpora al árbol y el coste acumulado "
                    f"pasa a {coste_total}."
                ),
            )
        )

        # Actualiza las mejores conexiones de los vértices externos.
        for vecino in sorted(grafo.neighbors(destino)):
            if vecino in incluidos:
                continue

            peso_vecino = grafo[destino][vecino].get("weight", 1)
            clave_anterior = claves[vecino]

            if peso_vecino < clave_anterior:
                claves[vecino] = peso_vecino
                padres[vecino] = destino

                heapq.heappush(
                    cola,
                    (peso_vecino, destino, vecino),
                )

                estados.append(
                    crear_estado_prim(
                        grafo=grafo,
                        incluidos=incluidos,
                        seleccionadas=seleccionadas,
                        rechazadas=rechazadas,
                        claves=claves,
                        padres=padres,
                        coste_total=coste_total,
                        actual=destino,
                        arista_activa=(destino, vecino),
                        accion="frontier_update",
                        mensaje=(
                            f"Mejora la conexión de {vecino}: su clave "
                            f"pasa de {clave_anterior} a {peso_vecino} y "
                            f"su padre pasa a ser {destino}."
                        ),
                    )
                )
            else:
                estados.append(
                    crear_estado_prim(
                        grafo=grafo,
                        incluidos=incluidos,
                        seleccionadas=seleccionadas,
                        rechazadas=rechazadas,
                        claves=claves,
                        padres=padres,
                        coste_total=coste_total,
                        actual=destino,
                        arista_activa=(destino, vecino),
                        accion="no_improvement",
                        mensaje=(
                            f"{destino}—{vecino} pesa {peso_vecino}; "
                            f"no mejora la clave actual de {vecino} "
                            f"({clave_anterior})."
                        ),
                    )
                )

    if len(incluidos) != grafo.number_of_nodes():
        raise RuntimeError(
            "Prim no pudo abarcar todos los vértices."
        )

    estados.append(
        crear_estado_prim(
            grafo=grafo,
            incluidos=incluidos,
            seleccionadas=seleccionadas,
            rechazadas=rechazadas,
            claves=claves,
            padres=padres,
            coste_total=coste_total,
            actual=None,
            accion="finished",
            mensaje=(
                "Prim ha terminado: todos los vértices están incluidos, "
                f"se han seleccionado {len(seleccionadas)} aristas y el "
                f"coste total es {coste_total}."
            ),
        )
    )

    return {
        "states": estados,
        "edges": seleccionadas,
        "rejected_edges": rechazadas,
        "total_cost": coste_total,
        "parents": padres,
        "keys": claves,
    }


class UnionFind:
    """
    Estructura de conjuntos disjuntos con compresión de caminos y rango.
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

    def find(self, elemento):
        """
        Devuelve el representante de la componente de ``elemento``.
        """

        if self.parent[elemento] != elemento:
            self.parent[elemento] = self.find(
                self.parent[elemento]
            )

        return self.parent[elemento]

    def union(self, primero, segundo):
        """
        Une dos componentes. Devuelve False si ya eran la misma.
        """

        raiz_primera = self.find(primero)
        raiz_segunda = self.find(segundo)

        if raiz_primera == raiz_segunda:
            return False

        if self.rank[raiz_primera] < self.rank[raiz_segunda]:
            raiz_primera, raiz_segunda = (
                raiz_segunda,
                raiz_primera,
            )

        self.parent[raiz_segunda] = raiz_primera

        if self.rank[raiz_primera] == self.rank[raiz_segunda]:
            self.rank[raiz_primera] += 1

        return True

    def obtener_componentes(self):
        """
        Devuelve las componentes actuales como listas ordenadas.
        """

        grupos = {}

        for elemento in self.parent:
            raiz = self.find(elemento)
            grupos.setdefault(raiz, []).append(elemento)

        componentes = [
            tuple(sorted(grupo))
            for grupo in grupos.values()
        ]

        return sorted(componentes)

    def obtener_mapa_canonico(self):
        """
        Asigna a cada vértice la menor etiqueta de su componente.

        Se usa para que la representación visual sea estable y legible.
        """

        mapa = {}

        for componente in self.obtener_componentes():
            etiqueta = min(componente)

            for nodo in componente:
                mapa[nodo] = etiqueta

        return mapa


def obtener_aristas_ordenadas(grafo):
    """
    Devuelve las aristas en orden ascendente de peso.

    Las etiquetas de los extremos se normalizan para que el desempate sea
    determinista, aunque en este ejemplo todos los pesos son distintos.
    """

    aristas = []

    for origen, destino, datos in grafo.edges(data=True):
        primero, segundo = sorted((origen, destino))

        aristas.append(
            (
                datos.get("weight", 1),
                primero,
                segundo,
            )
        )

    return sorted(aristas)


def crear_estado_kruskal(
    union_find,
    aristas_ordenadas,
    seleccionadas,
    rechazadas,
    coste_total,
    procesadas,
    mensaje,
    arista_activa=None,
    indice_activo=None,
    accion=None,
):
    """
    Crea una copia independiente de un estado de Kruskal.
    """

    return {
        "algorithm": "kruskal",
        "component_map": union_find.obtener_mapa_canonico(),
        "components": union_find.obtener_componentes(),
        "sorted_edges": list(aristas_ordenadas),
        "selected_edges": list(seleccionadas),
        "rejected_edges": list(rechazadas),
        "total_cost": coste_total,
        "processed_count": procesadas,
        "active_edge": arista_activa,
        "active_edge_index": indice_activo,
        "action": accion,
        "message": mensaje,
    }


def ejecutar_kruskal_por_pasos(grafo):
    """
    Ejecuta Kruskal mediante Union-Find y registra todos los pasos.
    """

    validar_grafo_mst(grafo)

    aristas_ordenadas = obtener_aristas_ordenadas(grafo)
    union_find = UnionFind(grafo.nodes())

    seleccionadas = []
    rechazadas = []
    coste_total = 0
    estados = []

    estados.append(
        crear_estado_kruskal(
            union_find=union_find,
            aristas_ordenadas=aristas_ordenadas,
            seleccionadas=seleccionadas,
            rechazadas=rechazadas,
            coste_total=coste_total,
            procesadas=0,
            accion="initial",
            mensaje=(
                "Kruskal comienza con una componente independiente por "
                "vértice y las aristas ordenadas de menor a mayor peso."
            ),
        )
    )

    for indice, (peso, origen, destino) in enumerate(
        aristas_ordenadas
    ):
        raiz_origen = union_find.find(origen)
        raiz_destino = union_find.find(destino)

        if raiz_origen != raiz_destino:
            union_find.union(origen, destino)
            seleccionadas.append((origen, destino))
            coste_total += peso
            accion = "accepted"

            mensaje = (
                f"Kruskal selecciona {origen}—{destino} con peso {peso}: "
                "sus extremos pertenecían a componentes diferentes. "
                f"El coste acumulado pasa a {coste_total}."
            )
        else:
            rechazadas.append((origen, destino))
            accion = "rejected"

            mensaje = (
                f"Kruskal rechaza {origen}—{destino} con peso {peso}: "
                "sus extremos ya pertenecen a la misma componente y "
                "la arista formaría un ciclo."
            )

        estados.append(
            crear_estado_kruskal(
                union_find=union_find,
                aristas_ordenadas=aristas_ordenadas,
                seleccionadas=seleccionadas,
                rechazadas=rechazadas,
                coste_total=coste_total,
                procesadas=indice + 1,
                arista_activa=(origen, destino),
                indice_activo=indice,
                accion=accion,
                mensaje=mensaje,
            )
        )

        if len(seleccionadas) == grafo.number_of_nodes() - 1:
            break

    estados.append(
        crear_estado_kruskal(
            union_find=union_find,
            aristas_ordenadas=aristas_ordenadas,
            seleccionadas=seleccionadas,
            rechazadas=rechazadas,
            coste_total=coste_total,
            procesadas=(
                estados[-1]["processed_count"]
                if estados
                else 0
            ),
            accion="finished",
            mensaje=(
                "Kruskal ha terminado: todas las componentes se han "
                f"fusionado, se han seleccionado {len(seleccionadas)} "
                f"aristas y el coste total es {coste_total}."
            ),
        )
    )

    return {
        "states": estados,
        "edges": seleccionadas,
        "rejected_edges": rechazadas,
        "total_cost": coste_total,
        "components": union_find.obtener_componentes(),
    }


def coste_de_aristas(grafo, aristas):
    """
    Suma los pesos de una colección de aristas no dirigidas.
    """

    return sum(
        grafo[origen][destino].get("weight", 1)
        for origen, destino in aristas
    )


def normalizar_conjunto_aristas(aristas):
    """
    Normaliza una colección para comparar árboles no dirigidos.
    """

    return {
        normalizar_arista(origen, destino)
        for origen, destino in aristas
    }


def comprobar_con_networkx(
    grafo,
    resultado_prim,
    resultado_kruskal,
):
    """
    Compara Prim y Kruskal con el MST calculado por NetworkX.
    """

    mst_networkx = nx.minimum_spanning_tree(
        grafo,
        algorithm="kruskal",
        weight="weight",
    )

    aristas_networkx = list(mst_networkx.edges())
    coste_networkx = mst_networkx.size(weight="weight")

    aristas_prim = normalizar_conjunto_aristas(
        resultado_prim["edges"]
    )
    aristas_kruskal = normalizar_conjunto_aristas(
        resultado_kruskal["edges"]
    )
    aristas_nx = normalizar_conjunto_aristas(
        aristas_networkx
    )

    return {
        "prim_cost_match": (
            resultado_prim["total_cost"] == coste_networkx
        ),
        "kruskal_cost_match": (
            resultado_kruskal["total_cost"] == coste_networkx
        ),
        "prim_edges_match": aristas_prim == aristas_nx,
        "kruskal_edges_match": aristas_kruskal == aristas_nx,
        "same_cost": (
            resultado_prim["total_cost"]
            == resultado_kruskal["total_cost"]
        ),
        "same_edges": aristas_prim == aristas_kruskal,
        "networkx_edges": aristas_networkx,
        "networkx_cost": coste_networkx,
    }


def crear_estado_comparacion(
    grafo,
    resultado_prim,
    resultado_kruskal,
    comprobacion,
):
    """
    Crea el estado final que resume ambos algoritmos.
    """

    aristas_prim_normalizadas = normalizar_conjunto_aristas(
        resultado_prim["edges"]
    )
    aristas_kruskal_normalizadas = normalizar_conjunto_aristas(
        resultado_kruskal["edges"]
    )

    comunes_normalizadas = (
        aristas_prim_normalizadas
        & aristas_kruskal_normalizadas
    )

    aristas_comunes = []

    for origen, destino in resultado_prim["edges"]:
        if normalizar_arista(origen, destino) in comunes_normalizadas:
            aristas_comunes.append((origen, destino))

    aristas_comunes_ponderadas = sorted(
        (
            grafo[origen][destino].get("weight", 1),
            min(origen, destino),
            max(origen, destino),
        )
        for origen, destino in aristas_comunes
    )

    return {
        "algorithm": "comparison",
        "prim_edges": list(resultado_prim["edges"]),
        "kruskal_edges": list(resultado_kruskal["edges"]),
        "common_edges": aristas_comunes,
        "weighted_common_edges": aristas_comunes_ponderadas,
        "prim_cost": resultado_prim["total_cost"],
        "kruskal_cost": resultado_kruskal["total_cost"],
        "same_cost": comprobacion["same_cost"],
        "same_edges": comprobacion["same_edges"],
        "message": (
            "Comparación final: Prim y Kruskal han construido el mismo "
            f"árbol único con {len(aristas_comunes)} aristas y coste "
            f"{resultado_prim['total_cost']}."
        ),
    }


def formatear_aristas(grafo, aristas):
    """
    Devuelve una representación legible de las aristas y sus pesos.
    """

    aristas_ponderadas = sorted(
        (
            grafo[origen][destino].get("weight", 1),
            min(origen, destino),
            max(origen, destino),
        )
        for origen, destino in aristas
    )

    return [
        f"{origen}—{destino} ({peso})"
        for peso, origen, destino in aristas_ponderadas
    ]


def imprimir_resultados(
    grafo,
    resultado_prim,
    resultado_kruskal,
    comprobacion,
):
    """
    Muestra la comparación por terminal.
    """

    print("\n=== Árbol de expansión mínima ===")

    print("\nPrim:")
    print(
        "  Orden de selección: "
        + " -> ".join(
            f"{u}—{v}"
            for u, v in resultado_prim["edges"]
        )
    )
    print(
        f"  Coste total: {resultado_prim['total_cost']}"
    )

    print("\nKruskal:")
    print(
        "  Orden de selección: "
        + " -> ".join(
            f"{u}—{v}"
            for u, v in resultado_kruskal["edges"]
        )
    )
    print(
        f"  Coste total: {resultado_kruskal['total_cost']}"
    )

    print("\nAristas del MST ordenadas por peso:")

    for arista in formatear_aristas(
        grafo,
        resultado_prim["edges"],
    ):
        print(f"  {arista}")

    print("\nComprobación con NetworkX:")
    print(
        "  ¿Coincide el coste de Prim?: "
        f"{comprobacion['prim_cost_match']}"
    )
    print(
        "  ¿Coinciden las aristas de Prim?: "
        f"{comprobacion['prim_edges_match']}"
    )
    print(
        "  ¿Coincide el coste de Kruskal?: "
        f"{comprobacion['kruskal_cost_match']}"
    )
    print(
        "  ¿Coinciden las aristas de Kruskal?: "
        f"{comprobacion['kruskal_edges_match']}"
    )
    print(
        "  ¿Prim y Kruskal tienen el mismo coste?: "
        f"{comprobacion['same_cost']}"
    )
    print(
        "  ¿Prim y Kruskal seleccionan las mismas aristas?: "
        f"{comprobacion['same_edges']}"
    )


def main():
    grafo = crear_grafo_mst()
    posiciones = obtener_posiciones()

    inicio_prim = "A"

    resultado_prim = ejecutar_prim_por_pasos(
        grafo=grafo,
        inicio=inicio_prim,
    )

    resultado_kruskal = ejecutar_kruskal_por_pasos(
        grafo=grafo,
    )

    comprobacion = comprobar_con_networkx(
        grafo=grafo,
        resultado_prim=resultado_prim,
        resultado_kruskal=resultado_kruskal,
    )

    imprimir_resultados(
        grafo=grafo,
        resultado_prim=resultado_prim,
        resultado_kruskal=resultado_kruskal,
        comprobacion=comprobacion,
    )

    estado_comparacion = crear_estado_comparacion(
        grafo=grafo,
        resultado_prim=resultado_prim,
        resultado_kruskal=resultado_kruskal,
        comprobacion=comprobacion,
    )

    estados = (
        resultado_prim["states"]
        + resultado_kruskal["states"]
        + [estado_comparacion]
    )

    animador = GraphAnimator(
        figsize=(16, 10),
        interval=750,
    )

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "03_algoritmos"
        / "06_arbol_expansion_minima_prim_kruskal.png"
    )

    animacion = animador.animate_mst_comparison(
        graph=grafo,
        pos=posiciones,
        states=estados,
        start_node=inicio_prim,
        title="Árbol de expansión mínima: Prim y Kruskal",
        final_image_path=output_path,
        repeat=False,
    )

    _ = animacion


if __name__ == "__main__":
    main()
