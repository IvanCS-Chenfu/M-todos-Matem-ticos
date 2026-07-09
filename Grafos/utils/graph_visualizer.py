from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


class GraphVisualizer:
    """
    Clase sencilla para visualizar grafos.

    Esta clase se reutilizará en distintos apartados de la wiki.
    Ahora solo muestra un grafo básico, pero más adelante se podrá ampliar
    para mostrar caminos, pesos, recorridos, árboles, errores, poses, etc.
    """

    def __init__(self, figsize=(8, 5)):
        self.figsize = figsize

    def show_graph(
        self,
        graph,
        title="Grafo",
        save_path=None,
        layout_seed=7,
    ):
        """
        Muestra un grafo usando NetworkX y Matplotlib.

        Parámetros:
        - graph: grafo de NetworkX.
        - title: título de la figura.
        - save_path: ruta opcional para guardar la imagen.
        - layout_seed: semilla para que el dibujo sea reproducible.
        """

        pos = nx.spring_layout(graph, seed=layout_seed)

        plt.figure(figsize=self.figsize)
        plt.title(title)

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=1800,
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            width=2,
        )

        nx.draw_networkx_labels(
            graph,
            pos,
            font_size=10,
            font_weight="bold",
        )

        plt.axis("off")
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Imagen guardada en: {save_path}")

        plt.show()