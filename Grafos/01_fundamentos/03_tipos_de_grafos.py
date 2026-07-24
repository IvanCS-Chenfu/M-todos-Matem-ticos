from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent
sys.path.append(str(GRAFOS_DIR))

from utils.graph_visualizer import GraphVisualizer


def crear_grafo_no_dirigido():
    """
    Crea un grafo no dirigido.

    Una arista no dirigida representa una relación simétrica:
    si A está conectado con B, entonces B también está conectado con A.
    """

    grafo = nx.Graph()

    grafo.add_edges_from([
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
    ])

    return grafo


def crear_grafo_dirigido():
    """
    Crea un grafo dirigido.

    Una arista dirigida representa una relación con sentido:
    A puede apuntar hacia B sin que necesariamente B apunte hacia A.
    """

    grafo = nx.DiGraph()

    grafo.add_edges_from([
        ("A", "B"),
        ("B", "C"),
        ("A", "D"),
    ])

    return grafo


def crear_grafo_ponderado():
    """
    Crea un grafo ponderado.

    En un grafo ponderado, las aristas tienen un valor asociado.
    Ese valor puede representar distancia, tiempo, coste, energía,
    riesgo o cualquier otra magnitud.
    """

    grafo = nx.Graph()

    grafo.add_edge("A", "B", weight=2)
    grafo.add_edge("B", "C", weight=5)
    grafo.add_edge("A", "C", weight=8)
    grafo.add_edge("C", "D", weight=1)

    return grafo


def crear_grafo_ciclico():
    """
    Crea un grafo cíclico.

    Un grafo tiene un ciclo cuando se puede salir de un nodo,
    recorrer varias aristas y volver al mismo nodo.
    """

    grafo = nx.Graph()

    grafo.add_edges_from([
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "A"),
    ])

    return grafo


def crear_grafo_aciclico():
    """
    Crea un grafo acíclico.

    Un grafo acíclico no tiene ciclos. Si además es conectado,
    se trata de un árbol.
    """

    grafo = nx.Graph()

    grafo.add_edges_from([
        ("A", "B"),
        ("A", "C"),
        ("C", "D"),
        ("C", "E"),
    ])

    return grafo


def crear_grafo_no_conexo():
    """
    Crea un grafo no conexo.

    Un grafo no conexo tiene al menos dos componentes separadas.
    Es decir, existen nodos entre los que no hay ningún camino.
    """

    grafo = nx.Graph()

    grafo.add_edges_from([
        ("A", "B"),
        ("B", "C"),
        ("D", "E"),
    ])

    grafo.add_node("F")

    return grafo


def crear_tipos_de_grafos():
    """
    Crea una colección de grafos básicos para comparar visualmente
    distintos tipos de grafos.

    Cada grafo incluye posiciones manuales para que los ejemplos se vean
    claros y no dependan del layout automático.
    """

    return [
        {
            "title": "No dirigido",
            "graph": crear_grafo_no_dirigido(),
            "description": "Relación simétrica",
            "pos": {
                "A": (0.0, 3.0),
                "B": (0.0, 1.0),
                "C": (0.0, -1.0),
                "D": (0.0, -3.0),
            },
        },
        {
            "title": "Dirigido",
            "graph": crear_grafo_dirigido(),
            "description": "Relación con sentido",
            "pos": {
                "A": (0.0, 0.0),
                "B": (-2.7, 1.6),
                "C": (-5.2, 1.6),
                "D": (2.7, -1.6),
            },
        },
        {
            "title": "Ponderado",
            "graph": crear_grafo_ponderado(),
            "description": "Aristas con peso",
            "pos": {
                "A": (3.0, 2.2),
                "B": (-3.0, 2.2),
                "C": (-0.6, 0.0),
                "D": (-1.7, -3.0),
            },
        },
        {
            "title": "Cíclico",
            "graph": crear_grafo_ciclico(),
            "description": "Existe al menos un ciclo",
            "pos": {
                "A": (2.6, 0.0),
                "B": (0.0, 2.6),
                "C": (-2.6, 0.0),
                "D": (0.0, -2.6),
            },
        },
        {
            "title": "Acíclico",
            "graph": crear_grafo_aciclico(),
            "description": "No contiene ciclos",
            "pos": {
                "A": (0.0, 1.1),
                "B": (0.0, 3.2),
                "C": (0.0, -1.0),
                "D": (2.5, -3.0),
                "E": (-2.5, -3.0),
            },
        },
        {
            "title": "No conexo",
            "graph": crear_grafo_no_conexo(),
            "description": "Tiene componentes separadas",
            "pos": {
                "A": (-2.0, 1.6),
                "B": (-2.0, 0.0),
                "C": (-2.0, -1.6),
                "D": (1.4, 0.8),
                "E": (1.4, -0.8),
                "F": (4.0, 0.0),
            },
        },
    ]


def main():
    tipos_de_grafos = crear_tipos_de_grafos()

    visualizador = GraphVisualizer(figsize=(13, 8))

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "01_fundamentos"
        / "03_tipos_de_grafos.png"
    )

    visualizador.show_graph_collection(
        tipos_de_grafos,
        title="Tipos de grafos",
        save_path=output_path,
        layout_seed=4,
        rows=2,
        cols=3,
    )


if __name__ == "__main__":
    main()