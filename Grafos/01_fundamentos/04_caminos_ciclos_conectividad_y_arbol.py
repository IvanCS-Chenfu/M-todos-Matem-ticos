from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent
sys.path.append(str(GRAFOS_DIR))

from utils.graph_visualizer import GraphVisualizer


def crear_grafo_caminos_ciclos_conectividad_y_arboles():
    """
    Crea un grafo grande y didáctico para ilustrar:
    - caminos,
    - ciclos,
    - árboles,
    - bosque,
    - conectividad,
    - componentes no conexas,
    - vértices aislados.

    Estructura general:
    1. Componente principal conexa:
       - ciclo superior de 6 nodos,
       - dos subárboles unidos al ciclo.

    2. Dos árboles desconectados entre sí:
       - árbol 1,
       - árbol 2.
       Juntos forman un bosque.

    3. Dos vértices aislados:
       - cada uno es una componente separada,
       - ayudan a mostrar que el grafo completo no es conexo.
    """

    grafo = nx.Graph()

    # --- Ciclo principal superior ---
    grafo.add_edges_from([
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("E", "F"),
        ("F", "A"),
    ])

    # --- Subárbol izquierdo unido al ciclo ---
    grafo.add_edges_from([
        ("F", "G"),
        ("G", "H"),
        ("G", "I"),
    ])

    # --- Subárbol derecho unido al ciclo ---
    grafo.add_edges_from([
        ("E", "J"),
        ("J", "K"),
        ("J", "L"),
    ])

    # --- Árbol desconectado 1 ---
    grafo.add_edges_from([
        ("M", "N"),
        ("M", "O"),
        ("O", "P"),
    ])

    # --- Árbol desconectado 2 ---
    grafo.add_edges_from([
        ("Q", "R"),
        ("Q", "S"),
    ])

    # --- Vértices aislados ---
    grafo.add_node("T")
    grafo.add_node("U")

    return grafo


def obtener_posiciones():
    """
    Define posiciones manuales para que el dibujo quede claro.

    Se separan las distintas estructuras para que visualmente se identifiquen
    bien los conceptos de conectividad, no conectividad, árboles y bosque.
    """

    return {
        # Ciclo principal
        "A": (0.0, 8.0),
        "B": (2.0, 9.2),
        "C": (4.5, 9.2),
        "D": (6.5, 8.0),
        "E": (5.0, 6.6),
        "F": (1.5, 6.6),

        # Subárbol izquierdo unido al ciclo
        "G": (0.5, 5.0),
        "H": (-0.7, 3.7),
        "I": (1.7, 3.7),

        # Subárbol derecho unido al ciclo
        "J": (6.0, 5.0),
        "K": (4.8, 3.7),
        "L": (7.2, 3.7),

        # Árbol desconectado 1
        "M": (-6.0, 4.5),
        "N": (-7.5, 3.0),
        "O": (-4.5, 3.0),
        "P": (-3.4, 1.5),

        # Árbol desconectado 2
        "Q": (11.0, 4.5),
        "R": (9.6, 3.0),
        "S": (12.4, 3.0),

        # Vértices aislados
        "T": (-6.2, 0.0),
        "U": (11.0, 0.2),
    }


def calcular_elementos_destacados(grafo):
    """
    Define las estructuras que se quieren destacar en la visualización.
    """

    ciclo = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "A")]

    camino_destacado = ["H", "G", "F", "A", "B", "C", "D", "E", "J", "L"]

    cajas = [
        {
            "nodes": ["A", "B", "C", "D", "E", "F"],
            "label": "Ciclo",
            "color": "tab:blue",
            "pad_x": 0.8,
            "pad_y": 0.7,
        },
        {
            "nodes": ["G", "H", "I"],
            "label": "Subárbol unido al ciclo",
            "color": "tab:green",
            "pad_x": 0.7,
            "pad_y": 0.7,
        },
        {
            "nodes": ["J", "K", "L"],
            "label": "Subárbol unido al ciclo",
            "color": "tab:green",
            "pad_x": 0.7,
            "pad_y": 0.7,
        },
        {
            "nodes": ["M", "N", "O", "P"],
            "label": "Árbol 1",
            "color": "tab:orange",
            "pad_x": 0.8,
            "pad_y": 0.8,
        },
        {
            "nodes": ["Q", "R", "S"],
            "label": "Árbol 2",
            "color": "tab:orange",
            "pad_x": 0.8,
            "pad_y": 0.8,
        },
        {
            "nodes": ["T"],
            "label": "Vértice aislado",
            "color": "tab:red",
            "pad_x": 0.7,
            "pad_y": 0.7,
        },
        {
            "nodes": ["U"],
            "label": "Vértice aislado",
            "color": "tab:red",
            "pad_x": 0.7,
            "pad_y": 0.7,
        },
    ]

    componentes = list(nx.connected_components(grafo))
    num_componentes = len(componentes)

    notas = [
        "El camino resaltado en rojo conecta dos hojas del componente principal.",
        "La parte superior contiene un ciclo.",
        "Los nodos M-N-O-P forman un árbol y Q-R-S forman otro árbol.",
        "Árbol 1 y Árbol 2 forman un bosque porque están desconectados entre sí.",
        f"El grafo completo no es conexo: tiene {num_componentes} componentes conexas.",
        "T y U son vértices aislados.",
    ]

    return {
        "cycle_edges": ciclo,
        "highlighted_path": camino_destacado,
        "structure_boxes": cajas,
        "notes": notas,
    }


def imprimir_resumen_por_terminal(grafo, elementos):
    """
    Imprime un pequeño resumen por terminal.
    """

    print("\n=== Resumen del grafo ===")
    print(f"Número de nodos: {grafo.number_of_nodes()}")
    print(f"Número de aristas: {grafo.number_of_edges()}")

    componentes = list(nx.connected_components(grafo))
    print(f"Número de componentes conexas: {len(componentes)}")

    print("\nComponentes conexas:")
    for i, componente in enumerate(componentes, start=1):
        print(f"  Componente {i}: {sorted(componente)}")

    print("\nCamino destacado:")
    print("  " + " -> ".join(elementos["highlighted_path"]))

    print("\nCiclo destacado:")
    print("  " + " | ".join(f"{u}-{v}" for u, v in elementos["cycle_edges"]))


def main():
    grafo = crear_grafo_caminos_ciclos_conectividad_y_arboles()
    pos = obtener_posiciones()
    elementos = calcular_elementos_destacados(grafo)

    imprimir_resumen_por_terminal(grafo, elementos)

    visualizador = GraphVisualizer(figsize=(14, 9))

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "01_fundamentos"
        / "04_caminos_ciclos_conectividad_y_arboles.png"
    )

    visualizador.show_structures_graph(
        graph=grafo,
        pos=pos,
        title="Caminos, ciclos, conectividad y árboles",
        highlighted_path=elementos["highlighted_path"],
        cycle_edges=elementos["cycle_edges"],
        structure_boxes=elementos["structure_boxes"],
        notes=elementos["notes"],
        save_path=output_path,
    )


if __name__ == "__main__":
    main()