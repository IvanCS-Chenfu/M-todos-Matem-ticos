from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent
sys.path.append(str(GRAFOS_DIR))

from utils.graph_visualizer import GraphVisualizer


def crear_grafo_elementos_basicos():
    """
    Crea un grafo mixto para estudiar los elementos básicos de un grafo.

    En este ejemplo:
    - Los nodos representan lugares de una casa.
    - Algunas aristas son no dirigidas, como conexiones físicas entre habitaciones.
    - Algunas aristas son dirigidas, como relaciones con sentido concreto.
    - Todas las aristas tienen un peso asociado.

    Importante:
    NetworkX no tiene un tipo de grafo mixto directo.
    Por eso usamos nx.MultiDiGraph y añadimos un atributo "directed" a cada arista.
    """

    grafo = nx.MultiDiGraph()

    grafo.add_nodes_from([
        "Entrada",
        "Pasillo",
        "Cocina",
        "Salón",
        "Habitación",
        "Terraza",
    ])

    # Aristas no dirigidas.
    # Representan conexiones físicas que pueden recorrerse en ambos sentidos.
    grafo.add_edge(
        "Entrada",
        "Pasillo",
        directed=False,
        weight=2,
    )

    grafo.add_edge(
        "Pasillo",
        "Salón",
        directed=False,
        weight=3,
    )

    grafo.add_edge(
        "Salón",
        "Habitación",
        directed=False,
        weight=4,
    )

    # Aristas dirigidas.
    # Representan relaciones con sentido.
    grafo.add_edge(
        "Pasillo",
        "Cocina",
        directed=True,
        weight=5,
    )

    grafo.add_edge(
        "Cocina",
        "Terraza",
        directed=True,
        weight=1,
    )

    grafo.add_edge(
        "Terraza",
        "Salón",
        directed=True,
        weight=6,
    )

    return grafo


def calcular_datos_importantes(grafo):
    """
    Calcula información básica de los vértices y aristas del grafo.

    Para cada vértice calcula:
    - vecinos,
    - grado total,
    - grado de entrada,
    - grado de salida.

    Criterio usado:
    - El grado total cuenta todas las aristas incidentes.
    - El grado de entrada solo cuenta aristas dirigidas que llegan al nodo.
    - El grado de salida solo cuenta aristas dirigidas que salen del nodo.
    - Las aristas no dirigidas cuentan para el grado total, pero no para
      el grado de entrada ni para el grado de salida.
    """

    datos_vertices = {}

    for nodo in grafo.nodes():
        datos_vertices[nodo] = {
            "vecinos": set(),
            "grado_total": 0,
            "grado_entrada": 0,
            "grado_salida": 0,
        }

    datos_aristas = {}

    for origen, destino, key, datos in grafo.edges(keys=True, data=True):
        directed = datos.get("directed", True)
        weight = datos.get("weight", None)

        # Toda arista conecta ambos nodos a nivel de vecindad general.
        datos_vertices[origen]["vecinos"].add(destino)
        datos_vertices[destino]["vecinos"].add(origen)

        # Toda arista cuenta como incidente para ambos nodos.
        datos_vertices[origen]["grado_total"] += 1
        datos_vertices[destino]["grado_total"] += 1

        if directed:
            datos_vertices[origen]["grado_salida"] += 1
            datos_vertices[destino]["grado_entrada"] += 1

        datos_aristas[(origen, destino, key)] = {
            "origen": origen,
            "destino": destino,
            "dirigida": directed,
            "peso": weight,
        }

    # Convertimos los conjuntos de vecinos en listas ordenadas para que
    # sean más fáciles de imprimir y mostrar.
    for nodo in datos_vertices:
        datos_vertices[nodo]["vecinos"] = sorted(datos_vertices[nodo]["vecinos"])

    return {
        "vertices": datos_vertices,
        "aristas": datos_aristas,
    }


def imprimir_datos_grafo(datos_grafo):
    """
    Imprime por terminal los datos principales del grafo.

    Esta salida no es lo importante de la demo, pero ayuda a comprobar
    que los cálculos son correctos.
    """

    print("\n=== Información de vértices ===")
    for nodo, datos in datos_grafo["vertices"].items():
        print(f"\nVértice: {nodo}")
        print(f"  Vecinos: {', '.join(datos['vecinos'])}")
        print(f"  Grado total: {datos['grado_total']}")
        print(f"  Grado de entrada: {datos['grado_entrada']}")
        print(f"  Grado de salida: {datos['grado_salida']}")

    print("\n=== Información de aristas ===")
    for (_, _, _), datos in datos_grafo["aristas"].items():
        tipo = "dirigida" if datos["dirigida"] else "no dirigida"
        print(
            f"  {datos['origen']} -> {datos['destino']} | "
            f"tipo: {tipo} | peso: {datos['peso']}"
        )


def main():
    grafo = crear_grafo_elementos_basicos()
    datos_grafo = calcular_datos_importantes(grafo)

    imprimir_datos_grafo(datos_grafo)

    visualizador = GraphVisualizer(figsize=(10, 6))

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "01_fundamentos"
        / "02_elementos_basicos_grafo.png"
    )

    visualizador.show_mixed_graph_with_info(
        grafo,
        node_info=datos_grafo["vertices"],
        edge_info=datos_grafo["aristas"],
        title="Elementos básicos de un grafo: vértices, aristas, grados y pesos",
        save_path=output_path,
    )


if __name__ == "__main__":
    main()