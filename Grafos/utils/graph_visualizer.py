from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Rectangle


class GraphVisualizer:
    """
    Clase sencilla para visualizar grafos.

    Esta clase se reutiliza en distintos apartados de la wiki.
    Incluye:
    - visualización simple de grafos,
    - comparación de varios grafos,
    - visualización de grafos mixtos con información interactiva,
    - visualización de estructuras grandes destacando ciclos, caminos,
      árboles, bosque, conectividad, etc.,
    - visualización de distintas representaciones computacionales
      de un mismo grafo.
    """

    def __init__(self, figsize=(8, 5)):
        self.figsize = figsize

    def _set_centered_limits(self, ax, pos, margin=0.35):
        """
        Centra el grafo dentro de la figura.
        """

        xs = [coord[0] for coord in pos.values()]
        ys = [coord[1] for coord in pos.values()]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max_x - min_x
        height = max_y - min_y

        if width == 0:
            width = 1.0
        if height == 0:
            height = 1.0

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        max_range = max(width, height)

        half_range = max_range / 2
        half_range_with_margin = half_range * (1 + margin)

        ax.set_xlim(
            center_x - half_range_with_margin,
            center_x + half_range_with_margin,
        )
        ax.set_ylim(
            center_y - half_range_with_margin,
            center_y + half_range_with_margin,
        )

        ax.set_aspect("equal", adjustable="box")

    def _draw_manual_node_labels(self, ax, graph, pos, font_size=9):
        """
        Dibuja las etiquetas de los nodos manualmente para controlar zorder.
        """

        for node, (x, y) in pos.items():
            ax.text(
                x,
                y,
                str(node),
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color="black",
                zorder=30,
            )

    def _draw_manual_edge_labels(self, ax, graph, pos, font_size=8):
        """
        Dibuja las etiquetas de peso manualmente.
        """

        edge_labels = nx.get_edge_attributes(graph, "weight")

        if not edge_labels:
            return

        for (u, v), weight in edge_labels.items():
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2

            dx = x2 - x1
            dy = y2 - y1

            length = (dx**2 + dy**2) ** 0.5

            if length == 0:
                offset_x = 0.0
                offset_y = 0.0
            else:
                offset_x = -dy / length * 0.16
                offset_y = dx / length * 0.16

            ax.text(
                mx + offset_x,
                my + offset_y,
                str(weight),
                fontsize=font_size,
                ha="center",
                va="center",
                color="black",
                zorder=40,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.95,
                },
            )

    def _draw_structure_box(
        self,
        ax,
        pos,
        nodes,
        label,
        color="tab:green",
        pad_x=0.7,
        pad_y=0.7,
        linewidth=2.0,
        linestyle="--",
    ):
        """
        Dibuja una caja resaltando un conjunto de nodos.
        """

        xs = [pos[node][0] for node in nodes]
        ys = [pos[node][1] for node in nodes]

        min_x = min(xs) - pad_x
        max_x = max(xs) + pad_x
        min_y = min(ys) - pad_y
        max_y = max(ys) + pad_y

        width = max_x - min_x
        height = max_y - min_y

        if width < 1.4:
            extra = (1.4 - width) / 2
            min_x -= extra
            width = 1.4

        if height < 1.4:
            extra = (1.4 - height) / 2
            min_y -= extra
            height = 1.4

        rect = Rectangle(
            (min_x, min_y),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=1,
        )
        ax.add_patch(rect)

        ax.text(
            min_x,
            max_y + 0.18,
            label,
            fontsize=9,
            fontweight="bold",
            color=color,
            ha="left",
            va="bottom",
            zorder=35,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": color,
                "alpha": 0.95,
            },
        )

    def _draw_weighted_graph_on_axis(
        self,
        ax,
        graph,
        pos,
        title,
        node_size=650,
        font_size=9,
    ):
        """
        Dibuja un grafo ponderado dentro de un eje concreto.

        Se usa para representar el mismo grafo en varios paneles.
        """

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

        for u, v in graph.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=2.2,
                color="black",
                zorder=10,
            )

        node_collection = nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=node_size,
            ax=ax,
        )
        node_collection.set_zorder(15)

        self._draw_manual_node_labels(
            ax=ax,
            graph=graph,
            pos=pos,
            font_size=font_size,
        )

        self._draw_manual_edge_labels(
            ax=ax,
            graph=graph,
            pos=pos,
            font_size=8,
        )

        self._set_centered_limits(ax, pos, margin=0.45)

    def show_graph(
        self,
        graph,
        title="Grafo",
        save_path=None,
        layout_seed=7,
    ):
        """
        Muestra un grafo básico usando NetworkX y Matplotlib.
        """

        pos = nx.spring_layout(graph, seed=layout_seed)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title)

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=1800,
            ax=ax,
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            width=2,
            ax=ax,
        )

        nx.draw_networkx_labels(
            graph,
            pos,
            font_size=10,
            font_weight="bold",
            ax=ax,
        )

        ax.axis("off")
        self._set_centered_limits(ax, pos)

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Imagen guardada en: {save_path}")

        plt.show()

    def show_graph_collection(
        self,
        graph_examples,
        title="Tipos de grafos",
        save_path=None,
        layout_seed=7,
        rows=2,
        cols=3,
    ):
        """
        Muestra varios grafos en una misma ventana.
        """

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=self.figsize,
        )

        fig.suptitle(title, fontsize=14, fontweight="bold")

        axes = axes.flatten()

        node_size = 520
        directed_shrink = 8

        for index, example in enumerate(graph_examples):
            ax = axes[index]

            graph = example["graph"]
            graph_title = example["title"]
            description = example.get("description", "")

            pos = example.get("pos")

            if pos is None:
                pos = nx.spring_layout(graph, seed=layout_seed, k=2.0)

            ax.set_title(graph_title, fontsize=11, fontweight="bold")
            ax.axis("off")

            if graph.is_directed():
                for u, v in graph.edges():
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]

                    arrow = FancyArrowPatch(
                        (x1, y1),
                        (x2, y2),
                        arrowstyle="-|>",
                        mutation_scale=18,
                        linewidth=2.2,
                        color="black",
                        shrinkA=directed_shrink,
                        shrinkB=directed_shrink,
                        zorder=20,
                    )

                    ax.add_patch(arrow)
            else:
                for u, v in graph.edges():
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        linewidth=2.2,
                        color="black",
                        zorder=10,
                    )

            node_collection = nx.draw_networkx_nodes(
                graph,
                pos,
                node_size=node_size,
                ax=ax,
            )
            node_collection.set_zorder(15)

            self._draw_manual_node_labels(
                ax,
                graph,
                pos,
                font_size=9,
            )

            self._draw_manual_edge_labels(
                ax,
                graph,
                pos,
                font_size=8,
            )

            if description:
                ax.text(
                    0.5,
                    -0.08,
                    description,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )

            self._set_centered_limits(ax, pos, margin=0.45)

        for empty_index in range(len(graph_examples), len(axes)):
            axes[empty_index].axis("off")

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Imagen guardada en: {save_path}")

        plt.show()

    def show_graph_representations(
        self,
        representation_examples,
        pos,
        title="Representaciones computacionales de un mismo grafo",
        save_path=None,
    ):
        """
        Muestra el mismo grafo creado desde distintas representaciones.

        Cada elemento de representation_examples debe tener:

        {
            "title": "Lista de aristas",
            "graph": grafo_networkx,
            "text": "Representación textual"
        }

        La figura final tiene dos filas:
        - arriba: el grafo dibujado,
        - abajo: la representación computacional usada para crearlo.
        """

        num_examples = len(representation_examples)

        fig = plt.figure(figsize=self.figsize)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        grid = fig.add_gridspec(
            2,
            num_examples,
            height_ratios=[2.0, 1.4],
            hspace=0.25,
            wspace=0.18,
        )

        for index, example in enumerate(representation_examples):
            graph_ax = fig.add_subplot(grid[0, index])
            text_ax = fig.add_subplot(grid[1, index])

            graph = example["graph"]
            example_title = example["title"]
            representation_text = example["text"]

            self._draw_weighted_graph_on_axis(
                ax=graph_ax,
                graph=graph,
                pos=pos,
                title=example_title,
                node_size=620,
                font_size=9,
            )

            text_ax.axis("off")
            text_ax.text(
                0.0,
                1.0,
                representation_text,
                transform=text_ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                family="monospace",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.98,
                },
            )

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Imagen guardada en: {save_path}")

        plt.show()

    def show_mixed_graph_with_info(
        self,
        graph,
        node_info=None,
        edge_info=None,
        title="Grafo mixto con información",
        save_path=None,
        layout_seed=7,
    ):
        """
        Muestra un grafo mixto y permite ver información al pasar el ratón
        por encima de nodos y aristas.
        """

        node_info = node_info or {}
        edge_info = edge_info or {}

        layout_graph = nx.Graph()
        layout_graph.add_nodes_from(graph.nodes())

        for u, v, *_ in graph.edges(keys=True, data=True):
            layout_graph.add_edge(u, v)

        pos = nx.spring_layout(layout_graph, seed=layout_seed)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title)
        ax.axis("off")

        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]

        ax.scatter(
            node_x,
            node_y,
            s=1800,
            zorder=3,
        )

        for node, (x, y) in pos.items():
            ax.text(
                x,
                y,
                node,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                zorder=4,
            )

        edge_artists = []

        for u, v, key, data in graph.edges(keys=True, data=True):
            directed = data.get("directed", True)
            weight = data.get("weight", None)

            x1, y1 = pos[u]
            x2, y2 = pos[v]

            if directed:
                artist = FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle="-|>",
                    mutation_scale=18,
                    linewidth=2,
                    shrinkA=25,
                    shrinkB=25,
                    zorder=2,
                )
                ax.add_patch(artist)
            else:
                line, = ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linewidth=2,
                    zorder=1,
                )
                artist = line

            if weight is not None:
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2

                ax.text(
                    mx,
                    my,
                    f"w={weight}",
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "fc": "white",
                        "ec": "none",
                        "alpha": 0.8,
                    },
                    zorder=5,
                )

            edge_artists.append(
                {
                    "u": u,
                    "v": v,
                    "key": key,
                    "data": data,
                    "artist": artist,
                    "p1": (x1, y1),
                    "p2": (x2, y2),
                }
            )

        self._set_centered_limits(ax, pos, margin=0.45)

        annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox={
                "boxstyle": "round,pad=0.4",
                "fc": "white",
                "ec": "black",
                "alpha": 0.95,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": "black",
            },
            fontsize=9,
            zorder=100,
        )

        annotation.set_zorder(100)

        if annotation.get_bbox_patch() is not None:
            annotation.get_bbox_patch().set_zorder(100)

        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_zorder(100)

        annotation.set_visible(False)

        self._set_centered_limits(ax, pos, margin=0.45)

        def format_node_text(node):
            info = node_info.get(node, {})

            vecinos = info.get("vecinos", [])
            vecinos_txt = ", ".join(vecinos) if vecinos else "Sin vecinos"

            return (
                f"Vértice: {node}\n"
                f"Grado total: {info.get('grado_total', '-')}\n"
                f"Grado de entrada: {info.get('grado_entrada', '-')}\n"
                f"Grado de salida: {info.get('grado_salida', '-')}\n"
                f"Vecinos: {vecinos_txt}"
            )

        def format_edge_text(u, v, key, data):
            info = edge_info.get((u, v, key), {})

            directed = data.get("directed", True)
            weight = data.get("weight", "-")

            if directed:
                tipo = "Dirigida"
                conexion = f"{u} → {v}"
            else:
                tipo = "No dirigida"
                conexion = f"{u} -- {v}"

            return (
                f"Arista: {conexion}\n"
                f"Tipo: {tipo}\n"
                f"Origen: {info.get('origen', u)}\n"
                f"Destino: {info.get('destino', v)}\n"
                f"Peso: {info.get('peso', weight)}"
            )

        def distance_pixels(point_a, point_b):
            ax_a = ax.transData.transform(point_a)
            ax_b = ax.transData.transform(point_b)
            return ((ax_a[0] - ax_b[0]) ** 2 + (ax_a[1] - ax_b[1]) ** 2) ** 0.5

        def point_to_segment_distance_pixels(point, seg_a, seg_b):
            p = ax.transData.transform(point)
            a = ax.transData.transform(seg_a)
            b = ax.transData.transform(seg_b)

            ab_x = b[0] - a[0]
            ab_y = b[1] - a[1]

            ap_x = p[0] - a[0]
            ap_y = p[1] - a[1]

            length_squared = ab_x**2 + ab_y**2

            if length_squared == 0:
                return ((p[0] - a[0])**2 + (p[1] - a[1])**2) ** 0.5

            t = (ap_x * ab_x + ap_y * ab_y) / length_squared
            t = max(0, min(1, t))

            projection = (
                a[0] + t * ab_x,
                a[1] + t * ab_y,
            )

            return ((p[0] - projection[0])**2 + (p[1] - projection[1])**2) ** 0.5

        def on_mouse_move(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                annotation.set_visible(False)
                fig.canvas.draw_idle()
                return

            mouse_point = (event.xdata, event.ydata)

            for node, node_position in pos.items():
                if distance_pixels(mouse_point, node_position) < 30:
                    annotation.xy = node_position
                    annotation.set_text(format_node_text(node))

                    annotation.set_zorder(100)
                    if annotation.get_bbox_patch() is not None:
                        annotation.get_bbox_patch().set_zorder(100)
                    if annotation.arrow_patch is not None:
                        annotation.arrow_patch.set_zorder(100)

                    annotation.set_visible(True)
                    fig.canvas.draw_idle()
                    return

            for edge in edge_artists:
                distance = point_to_segment_distance_pixels(
                    mouse_point,
                    edge["p1"],
                    edge["p2"],
                )

                if distance < 12:
                    mid_point = (
                        (edge["p1"][0] + edge["p2"][0]) / 2,
                        (edge["p1"][1] + edge["p2"][1]) / 2,
                    )

                    annotation.xy = mid_point
                    annotation.set_text(
                        format_edge_text(
                            edge["u"],
                            edge["v"],
                            edge["key"],
                            edge["data"],
                        )
                    )

                    annotation.set_zorder(100)
                    if annotation.get_bbox_patch() is not None:
                        annotation.get_bbox_patch().set_zorder(100)
                    if annotation.arrow_patch is not None:
                        annotation.arrow_patch.set_zorder(100)

                    annotation.set_visible(True)
                    fig.canvas.draw_idle()
                    return

            annotation.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Imagen guardada en: {save_path}")

        plt.show()

    def show_structures_graph(
        self,
        graph,
        pos,
        title="Caminos, ciclos, conectividad y árboles",
        highlighted_path=None,
        cycle_edges=None,
        structure_boxes=None,
        notes=None,
        save_path=None,
    ):
        """
        Muestra un grafo grande destacando estructuras importantes:
        - ciclo,
        - camino,
        - árboles,
        - bosque,
        - componentes no conexas,
        - vértices aislados.
        """

        cycle_edges = cycle_edges or []
        structure_boxes = structure_boxes or []
        notes = notes or []

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        for box in structure_boxes:
            self._draw_structure_box(
                ax=ax,
                pos=pos,
                nodes=box["nodes"],
                label=box["label"],
                color=box.get("color", "tab:green"),
                pad_x=box.get("pad_x", 0.8),
                pad_y=box.get("pad_y", 0.8),
            )

        for u, v in graph.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=2.2,
                color="#555555",
                zorder=10,
            )

        for u, v in cycle_edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=3.6,
                color="tab:blue",
                zorder=12,
            )

        path_nodes = set()

        if highlighted_path is not None and len(highlighted_path) >= 2:
            path_nodes = set(highlighted_path)
            path_edges = list(zip(highlighted_path[:-1], highlighted_path[1:]))

            for u, v in path_edges:
                x1, y1 = pos[u]
                x2, y2 = pos[v]

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linewidth=4.2,
                    color="tab:red",
                    zorder=14,
                )

        normal_nodes = [node for node in graph.nodes() if node not in path_nodes]

        normal_collection = nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=normal_nodes,
            node_size=700,
            node_color="#4C9ED9",
            ax=ax,
        )
        normal_collection.set_zorder(18)

        if path_nodes:
            path_collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=list(path_nodes),
                node_size=780,
                node_color="#FFB347",
                ax=ax,
            )
            path_collection.set_zorder(20)

        self._draw_manual_node_labels(
            ax=ax,
            graph=graph,
            pos=pos,
            font_size=9,
        )

        if notes:
            notes_text = "\n".join(f"• {line}" for line in notes)

            ax.text(
                0.02,
                0.02,
                notes_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                zorder=50,
                bbox={
                    "boxstyle": "round,pad=0.5",
                    "fc": "white",
                    "ec": "black",
                    "alpha": 0.95,
                },
            )

        self._set_centered_limits(ax, pos, margin=0.25)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Imagen guardada en: {save_path}")

        plt.show()