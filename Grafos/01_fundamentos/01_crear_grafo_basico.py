from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent
sys.path.append(str(GRAFOS_DIR))

from utils.graph_visualizer import GraphVisualizer


def crear_grafo_basico():
    """
    Crea un grafo básico no dirigido.

    En este ejemplo:
    - Los nodos representan lugares de una casa.
    - Las aristas representan conexiones entre esos lugares.
    """

    grafo = nx.Graph()

    grafo.add_nodes_from([
        "Entrada",
        "Pasillo",
        "Cocina",
        "Salón",
        "Habitación",
    ])

    grafo.add_edges_from([
        ("Entrada", "Pasillo"),
        ("Pasillo", "Cocina"),
        ("Pasillo", "Salón"),
        ("Salón", "Habitación"),
    ])

    return grafo


def main():
    grafo = crear_grafo_basico()

    visualizador = GraphVisualizer()

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "01_fundamentos"
        / "01_crear_grafo_basico.png"
    )

    visualizador.show_graph(
        grafo,
        title="Grafo básico: lugares conectados de una casa",
        save_path=output_path,
    )


if __name__ == "__main__":
    main()