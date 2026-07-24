from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


class GraphAnimator:
    """
    Clase reutilizable para crear animaciones de algoritmos sobre grafos.

    Incluye métodos para:
    - búsqueda en anchura (BFS),
    - búsqueda en profundidad (DFS),
    - caminos mínimos con Dijkstra,
    - caminos mínimos con A*,
    - caminos mínimos con Bellman-Ford,
    - caminos mínimos con Floyd-Warshall,
    - árboles de expansión mínima con Prim y Kruskal,
    - Union-Find y componentes conectadas,
    - grafos dirigidos acíclicos y ordenamiento topológico,
    - flujo máximo y cortes mínimos con Edmonds-Karp,
    - centralidad, PageRank y detección de comunidades.

    Más adelante se podrá ampliar con otros algoritmos.
    """

    def __init__(self, figsize=(15, 9), interval=850):
        """
        Parameters
        ----------
        figsize:
            Tamaño de la figura de Matplotlib.
        interval:
            Tiempo, en milisegundos, entre dos estados consecutivos.
        """

        self.figsize = figsize
        self.interval = interval

        # Se conserva una referencia para evitar que Matplotlib elimine
        # la animación antes de mostrarla.
        self.animation = None

    @staticmethod
    def _normalizar_arista(origen, destino):
        """
        Devuelve una representación independiente del sentido.

        En estos ejemplos BFS y DFS se ejecutan sobre grafos no dirigidos,
        por lo que (A, B) y (B, A) representan la misma arista.
        """

        return frozenset((origen, destino))

    @staticmethod
    def _calcular_limites(pos, margin_x=1.2, margin_y=0.9):
        """
        Calcula límites adecuados a partir de posiciones manuales.
        """

        xs = [coordenada[0] for coordenada in pos.values()]
        ys = [coordenada[1] for coordenada in pos.values()]

        return (
            min(xs) - margin_x,
            max(xs) + margin_x,
            min(ys) - margin_y,
            max(ys) + margin_y,
        )

    def _preparar_figura(self, title):
        """
        Crea una figura con:
        - un área grande para el grafo,
        - un área inferior para la cola o la pila.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            1,
            height_ratios=[5.2, 1.25],
            hspace=0.10,
        )

        graph_ax = fig.add_subplot(grid[0])
        structure_ax = fig.add_subplot(grid[1])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        return fig, graph_ax, structure_ax

    # ------------------------------------------------------------------
    # Elementos comunes de BFS
    # ------------------------------------------------------------------

    def _dibujar_leyenda(self, ax):
        """
        Añade la leyenda utilizada por la animación BFS.

        Se mantiene este nombre para conservar compatibilidad con el código
        desarrollado anteriormente.
        """

        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="#666666",
                markersize=10,
                label="No descubierto",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F6C85F",
                markeredgecolor="#8A6D1D",
                markersize=10,
                label="Descubierto / en cola",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D",
                markersize=10,
                label="Vértice actual",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4C9ED9",
                markeredgecolor="#1F4F73",
                markersize=10,
                label="Procesado",
            ),
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=3,
                label="Arista del árbol BFS",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper left",
            fontsize=8,
            framealpha=0.96,
            ncol=2,
        )

    def _dibujar_cola(self, ax, cola):
        """
        Dibuja la cola FIFO en un panel separado.

        El primer elemento de la lista es el siguiente en salir.
        Los nuevos elementos entran por la derecha.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.02,
            0.82,
            "Cola FIFO",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.text(
            0.02,
            0.42,
            "Salida",
            fontsize=9,
            ha="left",
            va="center",
        )

        ax.text(
            0.98,
            0.42,
            "Entrada",
            fontsize=9,
            ha="right",
            va="center",
        )

        if not cola:
            ax.text(
                0.50,
                0.42,
                "Cola vacía",
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.98,
                },
            )
            return

        max_celdas = 12
        cola_visible = list(cola[:max_celdas])

        inicio_x = 0.12
        fin_x = 0.88
        ancho_total = fin_x - inicio_x
        ancho_celda = min(0.065, ancho_total / max(len(cola_visible), 1))
        separacion = 0.012

        ancho_ocupado = (
            len(cola_visible) * ancho_celda
            + max(0, len(cola_visible) - 1) * separacion
        )

        x_actual = 0.50 - ancho_ocupado / 2

        for indice, nodo in enumerate(cola_visible):
            rectangulo = Rectangle(
                (x_actual, 0.22),
                ancho_celda,
                0.40,
                facecolor="#F6C85F",
                edgecolor="#8A6D1D",
                linewidth=1.7,
            )
            ax.add_patch(rectangulo)

            ax.text(
                x_actual + ancho_celda / 2,
                0.42,
                str(nodo),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
            )

            if indice == 0:
                ax.text(
                    x_actual + ancho_celda / 2,
                    0.12,
                    "siguiente",
                    fontsize=7,
                    ha="center",
                    va="top",
                )

            x_actual += ancho_celda + separacion

        if len(cola) > max_celdas:
            ax.text(
                0.91,
                0.42,
                f"+{len(cola) - max_celdas}",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="center",
            )

    def _dibujar_estado_bfs(
        self,
        graph_ax,
        queue_ax,
        graph,
        pos,
        state,
        start_node,
    ):
        """
        Dibuja un estado completo de BFS.
        """

        graph_ax.clear()
        graph_ax.axis("off")

        limites = self._calcular_limites(pos)
        graph_ax.set_xlim(limites[0], limites[1])
        graph_ax.set_ylim(limites[2], limites[3])
        graph_ax.set_aspect("equal", adjustable="box")

        current = state.get("current")
        discovered = set(state.get("discovered", set()))
        processed = set(state.get("processed", set()))
        queue = list(state.get("queue", []))
        levels = dict(state.get("levels", {}))
        tree_edges = {
            self._normalizar_arista(u, v)
            for u, v in state.get("tree_edges", [])
        }

        active_edge = state.get("active_edge")
        active_edge_normalized = None

        if active_edge is not None:
            active_edge_normalized = self._normalizar_arista(*active_edge)

        for u, v in graph.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            edge_key = self._normalizar_arista(u, v)

            if edge_key == active_edge_normalized:
                color = "#E45756"
                linewidth = 4.2
                zorder = 16
            elif edge_key in tree_edges:
                color = "#2E8B57"
                linewidth = 3.0
                zorder = 14
            else:
                color = "#B8B8B8"
                linewidth = 1.7
                zorder = 10

            graph_ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=linewidth,
                zorder=zorder,
            )

        undiscovered_nodes = [
            node
            for node in graph.nodes()
            if node not in discovered
        ]

        queued_nodes = [
            node
            for node in discovered
            if node not in processed and node != current
        ]

        processed_nodes = [
            node
            for node in processed
            if node != current
        ]

        if undiscovered_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=undiscovered_nodes,
                node_size=720,
                node_color="#D9D9D9",
                edgecolors="#666666",
                linewidths=1.3,
                ax=graph_ax,
            )
            collection.set_zorder(20)

        if queued_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=queued_nodes,
                node_size=760,
                node_color="#F6C85F",
                edgecolors="#8A6D1D",
                linewidths=1.5,
                ax=graph_ax,
            )
            collection.set_zorder(22)

        if processed_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=processed_nodes,
                node_size=760,
                node_color="#4C9ED9",
                edgecolors="#1F4F73",
                linewidths=1.5,
                ax=graph_ax,
            )
            collection.set_zorder(22)

        if current is not None:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[current],
                node_size=900,
                node_color="#E45756",
                edgecolors="#7A1D1D",
                linewidths=2.4,
                ax=graph_ax,
            )
            collection.set_zorder(25)

        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

        for node, level in levels.items():
            x, y = pos[node]

            graph_ax.text(
                x,
                y + 0.35,
                str(level),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#222222",
                zorder=40,
                bbox={
                    "boxstyle": "circle,pad=0.20",
                    "fc": "white",
                    "ec": "#444444",
                    "alpha": 0.97,
                },
            )

        start_x, start_y = pos[start_node]
        graph_ax.text(
            start_x,
            start_y - 0.42,
            "inicio",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=10,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.40",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        graph_ax.text(
            0.99,
            0.985,
            f"Visitados/procesados: {len(processed)} de {graph.number_of_nodes()}",
            transform=graph_ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        self._dibujar_leyenda(graph_ax)
        self._dibujar_cola(queue_ax, queue)

    def animate_bfs(
        self,
        graph,
        pos,
        states,
        start_node,
        title="Búsqueda en anchura (BFS)",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima BFS a partir de una secuencia de estados ya calculada.

        También guarda una imagen del estado final, en el que:
        - la cola está vacía,
        - todos los vértices alcanzables están procesados,
        - aparece el nivel de cada vértice,
        - se ve el árbol BFS resultante.
        """

        if not states:
            raise ValueError("La lista de estados de BFS no puede estar vacía.")

        fig, graph_ax, queue_ax = self._preparar_figura(title)

        if final_image_path is not None:
            self._dibujar_estado_bfs(
                graph_ax=graph_ax,
                queue_ax=queue_ax,
                graph=graph,
                pos=pos,
                state=states[-1],
                start_node=start_node,
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_bfs(
                graph_ax=graph_ax,
                queue_ax=queue_ax,
                graph=graph,
                pos=pos,
                state=states[0],
                start_node=start_node,
            )
            return []

        def update(frame_index):
            self._dibujar_estado_bfs(
                graph_ax=graph_ax,
                queue_ax=queue_ax,
                graph=graph,
                pos=pos,
                state=states[frame_index],
                start_node=start_node,
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()

        return self.animation

    # ------------------------------------------------------------------
    # Elementos específicos de DFS
    # ------------------------------------------------------------------

    def _dibujar_leyenda_dfs(self, ax):
        """
        Añade una leyenda compacta para interpretar DFS.
        """

        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="#666666",
                markersize=10,
                label="No descubierto",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F6C85F",
                markeredgecolor="#8A6D1D",
                markersize=10,
                label="Activo / en pila",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D",
                markersize=10,
                label="Vértice actual",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4C9ED9",
                markeredgecolor="#1F4F73",
                markersize=10,
                label="Finalizado",
            ),
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=3,
                label="Arista del árbol DFS",
            ),
            Line2D(
                [0],
                [0],
                color="#8E5EA2",
                linewidth=3,
                label="Arista de ciclo",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper left",
            fontsize=8,
            framealpha=0.96,
            ncol=2,
        )

    def _dibujar_pila(self, ax, pila):
        """
        Dibuja la pila LIFO de DFS.

        La base aparece a la izquierda y la cima a la derecha.
        El último elemento es el siguiente que finalizará o retrocederá.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.02,
            0.82,
            "Pila LIFO / camino activo",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.text(
            0.02,
            0.42,
            "Base",
            fontsize=9,
            ha="left",
            va="center",
        )

        ax.text(
            0.98,
            0.42,
            "Cima",
            fontsize=9,
            ha="right",
            va="center",
        )

        if not pila:
            ax.text(
                0.50,
                0.42,
                "Pila vacía",
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.98,
                },
            )
            return

        max_celdas = 18
        pila_visible = list(pila[-max_celdas:])

        inicio_x = 0.11
        fin_x = 0.89
        ancho_total = fin_x - inicio_x
        ancho_celda = min(0.052, ancho_total / max(len(pila_visible), 1))
        separacion = 0.008

        ancho_ocupado = (
            len(pila_visible) * ancho_celda
            + max(0, len(pila_visible) - 1) * separacion
        )

        x_actual = 0.50 - ancho_ocupado / 2

        for indice, nodo in enumerate(pila_visible):
            es_cima = indice == len(pila_visible) - 1

            rectangulo = Rectangle(
                (x_actual, 0.22),
                ancho_celda,
                0.40,
                facecolor="#E45756" if es_cima else "#F6C85F",
                edgecolor="#7A1D1D" if es_cima else "#8A6D1D",
                linewidth=1.8,
            )
            ax.add_patch(rectangulo)

            ax.text(
                x_actual + ancho_celda / 2,
                0.42,
                str(nodo),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
            )

            if es_cima:
                ax.text(
                    x_actual + ancho_celda / 2,
                    0.12,
                    "actual",
                    fontsize=7,
                    ha="center",
                    va="top",
                )

            x_actual += ancho_celda + separacion

        if len(pila) > max_celdas:
            ax.text(
                0.08,
                0.42,
                f"+{len(pila) - max_celdas}",
                fontsize=9,
                fontweight="bold",
                ha="right",
                va="center",
            )

    def _dibujar_estado_dfs(
        self,
        graph_ax,
        stack_ax,
        graph,
        pos,
        state,
        start_node,
    ):
        """
        Dibuja un estado completo de DFS.

        Sobre cada nodo descubierto aparece:

            profundidad · descubrimiento/finalización

        Ejemplo:

            p3 · 7/12

        Si el nodo todavía no ha finalizado, el segundo tiempo aparece
        como un guion.
        """

        graph_ax.clear()
        graph_ax.axis("off")

        limites = self._calcular_limites(
            pos,
            margin_x=1.3,
            margin_y=1.15,
        )

        graph_ax.set_xlim(limites[0], limites[1])
        graph_ax.set_ylim(limites[2], limites[3])
        graph_ax.set_aspect("equal", adjustable="box")

        current = state.get("current")
        discovered = set(state.get("discovered", set()))
        finished = set(state.get("finished", set()))
        stack = list(state.get("stack", []))
        depths = dict(state.get("depths", {}))
        discovery_times = dict(state.get("discovery_times", {}))
        finish_times = dict(state.get("finish_times", {}))

        tree_edges = {
            self._normalizar_arista(u, v)
            for u, v in state.get("tree_edges", [])
        }

        cycle_edges = {
            self._normalizar_arista(u, v)
            for u, v in state.get("cycle_edges", [])
        }

        active_edge = state.get("active_edge")
        active_edge_normalized = None

        if active_edge is not None:
            active_edge_normalized = self._normalizar_arista(*active_edge)

        edge_kind = state.get("edge_kind")

        # 1. Aristas.
        for u, v in graph.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            edge_key = self._normalizar_arista(u, v)

            if edge_key == active_edge_normalized:
                if edge_kind == "backtrack":
                    color = "#F28E2B"
                elif edge_kind == "cycle":
                    color = "#8E5EA2"
                else:
                    color = "#E45756"

                linewidth = 4.2
                zorder = 18

            elif edge_key in tree_edges:
                color = "#2E8B57"
                linewidth = 3.0
                zorder = 15

            elif edge_key in cycle_edges:
                color = "#8E5EA2"
                linewidth = 2.8
                zorder = 14

            else:
                color = "#B8B8B8"
                linewidth = 1.7
                zorder = 10

            graph_ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=linewidth,
                zorder=zorder,
            )

        # 2. Estados de los nodos.
        undiscovered_nodes = [
            node
            for node in graph.nodes()
            if node not in discovered
        ]

        active_nodes = [
            node
            for node in stack
            if node != current
        ]

        finished_nodes = [
            node
            for node in finished
            if node != current
        ]

        if undiscovered_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=undiscovered_nodes,
                node_size=720,
                node_color="#D9D9D9",
                edgecolors="#666666",
                linewidths=1.3,
                ax=graph_ax,
            )
            collection.set_zorder(20)

        if active_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=active_nodes,
                node_size=780,
                node_color="#F6C85F",
                edgecolors="#8A6D1D",
                linewidths=1.6,
                ax=graph_ax,
            )
            collection.set_zorder(22)

        if finished_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=finished_nodes,
                node_size=760,
                node_color="#4C9ED9",
                edgecolors="#1F4F73",
                linewidths=1.5,
                ax=graph_ax,
            )
            collection.set_zorder(22)

        if current is not None:
            current_color = (
                "#4C9ED9"
                if current in finished and current not in stack
                else "#E45756"
            )

            current_edge_color = (
                "#1F4F73"
                if current in finished and current not in stack
                else "#7A1D1D"
            )

            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[current],
                node_size=920,
                node_color=current_color,
                edgecolors=current_edge_color,
                linewidths=2.4,
                ax=graph_ax,
            )
            collection.set_zorder(25)

        # 3. Etiquetas de los nodos.
        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

        # 4. Profundidad y tiempos DFS.
        for node in discovered:
            x, y = pos[node]

            profundidad = depths.get(node, "-")
            descubrimiento = discovery_times.get(node, "-")
            finalizacion = finish_times.get(node, "-")

            graph_ax.text(
                x,
                y + 0.36,
                f"p{profundidad} · {descubrimiento}/{finalizacion}",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#222222",
                zorder=40,
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "fc": "white",
                    "ec": "#555555",
                    "alpha": 0.97,
                },
            )

        # 5. Vértice inicial.
        start_x, start_y = pos[start_node]
        graph_ax.text(
            start_x,
            start_y - 0.43,
            "inicio",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        # 6. Explicación del paso.
        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=10,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.40",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        graph_ax.text(
            0.99,
            0.985,
            (
                f"Descubiertos: {len(discovered)} de {graph.number_of_nodes()}  |  "
                f"Finalizados: {len(finished)}"
            ),
            transform=graph_ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        graph_ax.text(
            0.99,
            0.925,
            "Etiqueta: profundidad · descubrimiento/finalización",
            transform=graph_ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            color="#444444",
            zorder=50,
        )

        self._dibujar_leyenda_dfs(graph_ax)
        self._dibujar_pila(stack_ax, stack)

    def animate_dfs(
        self,
        graph,
        pos,
        states,
        start_node,
        title="Búsqueda en profundidad (DFS)",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima DFS a partir de una secuencia de estados ya calculada.

        También guarda una imagen del estado final, en el que:
        - la pila está vacía,
        - todos los vértices alcanzables están finalizados,
        - aparecen profundidad y tiempos de descubrimiento/finalización,
        - se ve el árbol DFS,
        - se distinguen las aristas que cierran ciclos.
        """

        if not states:
            raise ValueError("La lista de estados de DFS no puede estar vacía.")

        fig, graph_ax, stack_ax = self._preparar_figura(title)

        if final_image_path is not None:
            self._dibujar_estado_dfs(
                graph_ax=graph_ax,
                stack_ax=stack_ax,
                graph=graph,
                pos=pos,
                state=states[-1],
                start_node=start_node,
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_dfs(
                graph_ax=graph_ax,
                stack_ax=stack_ax,
                graph=graph,
                pos=pos,
                state=states[0],
                start_node=start_node,
            )
            return []

        def update(frame_index):
            self._dibujar_estado_dfs(
                graph_ax=graph_ax,
                stack_ax=stack_ax,
                graph=graph,
                pos=pos,
                state=states[frame_index],
                start_node=start_node,
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()

        return self.animation
    # ------------------------------------------------------------------
    # Elementos específicos de Dijkstra
    # ------------------------------------------------------------------

    def _preparar_figura_dijkstra(self, title):
        """
        Crea una figura específica para Dijkstra.

        Distribución:
        - izquierda: leyenda y tarjetas compactas de distancias/predecesores;
        - derecha superior: grafo ponderado;
        - derecha inferior: cola de prioridad.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.55, 4.45],
            height_ratios=[5.2, 1.15],
            wspace=0.08,
            hspace=0.08,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        graph_ax = fig.add_subplot(grid[0, 1])
        queue_ax = fig.add_subplot(grid[1, 1])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.93,
            bottom=0.045,
        )

        return fig, graph_ax, info_ax, queue_ax

    @staticmethod
    def _formatear_distancia(valor):
        """
        Formatea una distancia para mostrarla en nodos y tarjetas.
        """

        if valor == float("inf"):
            return "∞"

        if isinstance(valor, float) and valor.is_integer():
            return str(int(valor))

        return str(valor)

    def _dibujar_leyenda_dijkstra(self, ax):
        """
        Dibuja la leyenda en el panel izquierdo.

        Al utilizar un eje independiente, la leyenda no puede quedar
        tapada por los nodos, aristas o etiquetas del grafo.
        """

        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="#666666",
                markersize=8,
                label="Distancia infinita",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F6C85F",
                markeredgecolor="#8A6D1D",
                markersize=8,
                label="Distancia provisional",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D",
                markersize=8,
                label="Vértice actual",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4C9ED9",
                markeredgecolor="#1F4F73",
                markersize=8,
                label="Distancia definitiva",
            ),
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=3,
                label="Árbol de predecesores",
            ),
            Line2D(
                [0],
                [0],
                color="#D62728",
                linewidth=4,
                label="Camino mínimo final",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            fontsize=7.2,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.9,
            handlelength=2.1,
            borderpad=0.55,
        )

    def _dibujar_peso_arista(
        self,
        ax,
        pos,
        origen,
        destino,
        peso,
    ):
        """
        Dibuja el peso de una arista ligeramente desplazado de su centro.
        """

        x1, y1 = pos[origen]
        x2, y2 = pos[destino]

        medio_x = (x1 + x2) / 2
        medio_y = (y1 + y2) / 2

        dx = x2 - x1
        dy = y2 - y1
        longitud = (dx**2 + dy**2) ** 0.5

        if longitud == 0:
            desplazamiento_x = 0
            desplazamiento_y = 0
        else:
            desplazamiento_x = -dy / longitud * 0.13
            desplazamiento_y = dx / longitud * 0.13

        ax.text(
            medio_x + desplazamiento_x,
            medio_y + desplazamiento_y,
            self._formatear_distancia(peso),
            fontsize=8,
            ha="center",
            va="center",
            color="#222222",
            zorder=35,
            bbox={
                "boxstyle": "round,pad=0.16",
                "fc": "white",
                "ec": "none",
                "alpha": 0.96,
            },
        )

    def _dibujar_tabla_dijkstra(
        self,
        ax,
        nodes,
        distances,
        predecessors,
        finalized,
        current,
        priority_queue=None,
    ):
        """
        Dibuja tarjetas compactas a la izquierda del grafo.

        Cada tarjeta contiene:
        - el vértice;
        - su distancia actual;
        - su predecesor actual.

        Colores:
        - gris: todavía no alcanzado;
        - amarillo: distancia provisional;
        - azul: distancia definitiva.

        La tarjeta del vértice actual se resalta con un borde rojo.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            "Distancias y predecesores",
            fontsize=11.5,
            fontweight="bold",
            ha="center",
            va="top",
        )

        number_of_columns = 2
        number_of_rows = (
            len(nodes) + number_of_columns - 1
        ) // number_of_columns

        card_width = 0.405
        card_height = 0.078
        horizontal_gap = 0.055
        vertical_gap = 0.018

        total_width = (
            number_of_columns * card_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.665

        for index, node in enumerate(nodes):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (card_width + horizontal_gap)
            y = top_y - row * (card_height + vertical_gap)

            distance = distances.get(node, float("inf"))
            predecessor = predecessors.get(node)

            if node in finalized:
                face_color = "#B7D7F0"
                edge_color = "#1F4F73"
            elif distance != float("inf"):
                face_color = "#FBE5A6"
                edge_color = "#8A6D1D"
            else:
                face_color = "#E5E5E5"
                edge_color = "#777777"

            line_width = 1.5

            if node == current:
                edge_color = "#C62828"
                line_width = 3.0

            rectangle = Rectangle(
                (x, y),
                card_width,
                card_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            predecessor_text = (
                "—"
                if predecessor is None
                else str(predecessor)
            )

            ax.text(
                x + card_width * 0.11,
                y + card_height / 2,
                str(node),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + card_width * 0.31,
                y + card_height / 2,
                f"d={self._formatear_distancia(distance)}",
                fontsize=7.4,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.62,
                y + card_height / 2,
                f"pred={predecessor_text}",
                fontsize=7.1,
                ha="left",
                va="center",
            )

        ax.text(
            0.50,
            0.055,
            (
                "Gris: sin alcanzar   ·   "
                "Amarillo: provisional   ·   "
                "Azul: definitiva"
            ),
            fontsize=6.8,
            ha="center",
            va="center",
            color="#444444",
        )

        self._dibujar_leyenda_dijkstra(ax)

    def _dibujar_cola_prioridad_dijkstra(
        self,
        ax,
        priority_queue,
    ):
        """
        Dibuja la cola de prioridad debajo del grafo.

        La cola se ordena únicamente para mostrarla. El primer elemento
        visible es el que tiene la menor prioridad y será el siguiente
        candidato a extraer.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.02,
            0.82,
            "Cola de prioridad",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.text(
            0.02,
            0.41,
            "Mínimo",
            fontsize=8.5,
            ha="left",
            va="center",
        )

        ax.text(
            0.98,
            0.41,
            "Prioridad mayor",
            fontsize=8.5,
            ha="right",
            va="center",
        )

        queue_sorted = sorted(priority_queue)

        if not queue_sorted:
            ax.text(
                0.50,
                0.41,
                "Cola vacía",
                fontsize=11.5,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.42",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.98,
                },
            )
            return

        max_cells = 10
        visible_queue = queue_sorted[:max_cells]

        initial_x = 0.12
        final_x = 0.88
        total_width = final_x - initial_x
        cell_width = min(
            0.072,
            total_width / max(len(visible_queue), 1),
        )
        gap = 0.010

        occupied_width = (
            len(visible_queue) * cell_width
            + max(0, len(visible_queue) - 1) * gap
        )

        current_x = 0.50 - occupied_width / 2

        for index, (distance, node) in enumerate(visible_queue):
            is_minimum = index == 0

            rectangle = Rectangle(
                (current_x, 0.22),
                cell_width,
                0.39,
                facecolor="#E45756" if is_minimum else "#F6C85F",
                edgecolor="#7A1D1D" if is_minimum else "#8A6D1D",
                linewidth=2.0 if is_minimum else 1.5,
            )
            ax.add_patch(rectangle)

            ax.text(
                current_x + cell_width / 2,
                0.46,
                str(node),
                fontsize=8.8,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                current_x + cell_width / 2,
                0.32,
                self._formatear_distancia(distance),
                fontsize=7.5,
                ha="center",
                va="center",
            )

            if is_minimum:
                ax.text(
                    current_x + cell_width / 2,
                    0.13,
                    "siguiente",
                    fontsize=6.6,
                    ha="center",
                    va="top",
                )

            current_x += cell_width + gap

        if len(queue_sorted) > max_cells:
            ax.text(
                0.91,
                0.41,
                f"+{len(queue_sorted) - max_cells}",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="center",
            )

    def _dibujar_estado_dijkstra(
        self,
        graph_ax,
        table_ax,
        queue_ax,
        graph,
        pos,
        state,
        source_node,
        target_node,
    ):
        """
        Dibuja un estado completo del algoritmo de Dijkstra.
        """

        graph_ax.clear()
        graph_ax.axis("off")

        limits = self._calcular_limites(
            pos,
            margin_x=1.2,
            margin_y=1.0,
        )

        graph_ax.set_xlim(limits[0], limits[1])
        graph_ax.set_ylim(limits[2], limits[3])
        graph_ax.set_aspect("equal", adjustable="box")

        current = state.get("current")
        distances = dict(state.get("distances", {}))
        predecessors = dict(state.get("predecessors", {}))
        finalized = set(state.get("finalized", set()))
        priority_queue = list(state.get("priority_queue", []))
        active_edge = state.get("active_edge")
        action = state.get("action")
        final_path = list(state.get("final_path", []))

        predecessor_edges = {
            self._normalizar_arista(predecessor, node)
            for node, predecessor in predecessors.items()
            if predecessor is not None
        }

        final_path_edges = {
            self._normalizar_arista(u, v)
            for u, v in zip(final_path[:-1], final_path[1:])
        }

        active_edge_normalized = None

        if active_edge is not None:
            active_edge_normalized = self._normalizar_arista(*active_edge)

        # 1. Aristas y pesos.
        for u, v, data in graph.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            edge_key = self._normalizar_arista(u, v)

            if edge_key == active_edge_normalized:
                color = (
                    "#F28E2B"
                    if action == "no_improvement"
                    else "#E45756"
                )
                line_width = 4.2
                zorder = 20
            elif edge_key in final_path_edges:
                color = "#D62728"
                line_width = 4.3
                zorder = 18
            elif edge_key in predecessor_edges:
                color = "#2E8B57"
                line_width = 3.0
                zorder = 15
            else:
                color = "#B8B8B8"
                line_width = 1.7
                zorder = 10

            graph_ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=line_width,
                zorder=zorder,
            )

            self._dibujar_peso_arista(
                ax=graph_ax,
                pos=pos,
                origen=u,
                destino=v,
                peso=data.get("weight", 1),
            )

        # 2. Clasificación de nodos.
        unreachable_nodes = [
            node
            for node in graph.nodes()
            if distances.get(node, float("inf")) == float("inf")
        ]

        provisional_nodes = [
            node
            for node in graph.nodes()
            if (
                distances.get(node, float("inf")) != float("inf")
                and node not in finalized
                and node != current
            )
        ]

        finalized_nodes = [
            node
            for node in finalized
            if node != current
        ]

        if unreachable_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=unreachable_nodes,
                node_size=760,
                node_color="#D9D9D9",
                edgecolors="#666666",
                linewidths=1.3,
                ax=graph_ax,
            )
            collection.set_zorder(22)

        if provisional_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=provisional_nodes,
                node_size=790,
                node_color="#F6C85F",
                edgecolors="#8A6D1D",
                linewidths=1.6,
                ax=graph_ax,
            )
            collection.set_zorder(23)

        if finalized_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=finalized_nodes,
                node_size=790,
                node_color="#4C9ED9",
                edgecolors="#1F4F73",
                linewidths=1.6,
                ax=graph_ax,
            )
            collection.set_zorder(23)

        if current is not None:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[current],
                node_size=930,
                node_color="#E45756",
                edgecolors="#7A1D1D",
                linewidths=2.5,
                ax=graph_ax,
            )
            collection.set_zorder(26)

        # 3. Etiquetas de vértices y distancias.
        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

            distance_text = self._formatear_distancia(
                distances.get(node, float("inf"))
            )

            graph_ax.text(
                x,
                y + 0.39,
                f"d={distance_text}",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#222222",
                zorder=40,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "#555555",
                    "alpha": 0.97,
                },
            )

        # 4. Marcas de origen y destino.
        source_x, source_y = pos[source_node]
        target_x, target_y = pos[target_node]

        graph_ax.text(
            source_x,
            source_y - 0.43,
            "origen",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        graph_ax.text(
            target_x,
            target_y - 0.43,
            "destino",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        # 5. Mensaje explicativo.
        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=9.3,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        candidate = state.get("candidate")
        old_distance = state.get("old_distance")
        edge_weight = state.get("edge_weight")

        if active_edge is not None and candidate is not None:
            origin, destination = active_edge

            operation_text = (
                f"Relajación {origin}→{destination}: "
                f"{self._formatear_distancia(distances.get(origin, float('inf')))}"
                f" + {self._formatear_distancia(edge_weight)}"
                f" = {self._formatear_distancia(candidate)}"
                f"  |  anterior: {self._formatear_distancia(old_distance)}"
            )

            graph_ax.text(
                0.50,
                0.965,
                operation_text,
                transform=graph_ax.transAxes,
                fontsize=8.3,
                ha="center",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
                zorder=50,
            )

        graph_ax.text(
            0.99,
            0.985,
            f"Definitivos: {len(finalized)} de {graph.number_of_nodes()}",
            transform=graph_ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        self._dibujar_tabla_dijkstra(
            ax=table_ax,
            nodes=sorted(graph.nodes()),
            distances=distances,
            predecessors=predecessors,
            finalized=finalized,
            current=current,
        )

        self._dibujar_cola_prioridad_dijkstra(
            ax=queue_ax,
            priority_queue=priority_queue,
        )

    def animate_dijkstra(
        self,
        graph,
        pos,
        states,
        source_node,
        target_node,
        title="Caminos mínimos con Dijkstra",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima Dijkstra a partir de los estados calculados por el script.

        También guarda una imagen del estado final:
        - cola de prioridad vacía;
        - distancias definitivas;
        - predecesores definitivos;
        - árbol de caminos mínimos;
        - camino mínimo al destino resaltado.
        """

        if not states:
            raise ValueError(
                "La lista de estados de Dijkstra no puede estar vacía."
            )

        (
            fig,
            graph_ax,
            table_ax,
            queue_ax,
        ) = self._preparar_figura_dijkstra(title)

        if final_image_path is not None:
            self._dibujar_estado_dijkstra(
                graph_ax=graph_ax,
                table_ax=table_ax,
                queue_ax=queue_ax,
                graph=graph,
                pos=pos,
                state=states[-1],
                source_node=source_node,
                target_node=target_node,
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(
                f"Imagen final guardada en: "
                f"{final_image_path}"
            )

        def init():
            self._dibujar_estado_dijkstra(
                graph_ax=graph_ax,
                table_ax=table_ax,
                queue_ax=queue_ax,
                graph=graph,
                pos=pos,
                state=states[0],
                source_node=source_node,
                target_node=target_node,
            )
            return []

        def update(frame_index):
            self._dibujar_estado_dijkstra(
                graph_ax=graph_ax,
                table_ax=table_ax,
                queue_ax=queue_ax,
                graph=graph,
                pos=pos,
                state=states[frame_index],
                source_node=source_node,
                target_node=target_node,
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()

        return self.animation

    # ------------------------------------------------------------------
    # Elementos específicos de A*
    # ------------------------------------------------------------------

    def _preparar_figura_astar(self, title):
        """
        Reutiliza la misma distribución visual empleada por Dijkstra.

        Esto permite comparar ambos algoritmos sobre:
        - el mismo grafo;
        - las mismas posiciones;
        - tarjetas equivalentes;
        - una cola de prioridad situada bajo el grafo.
        """

        return self._preparar_figura_dijkstra(title)

    def _dibujar_leyenda_astar(self, ax):
        """
        Dibuja la leyenda de A* en el panel izquierdo.
        """

        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="#666666",
                markersize=8,
                label="No descubierto",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F6C85F",
                markeredgecolor="#8A6D1D",
                markersize=8,
                label="Conjunto abierto",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D",
                markersize=8,
                label="Vértice actual",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4C9ED9",
                markeredgecolor="#1F4F73",
                markersize=8,
                label="Conjunto cerrado",
            ),
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=3,
                label="Árbol de predecesores",
            ),
            Line2D(
                [0],
                [0],
                color="#D62728",
                linewidth=4,
                label="Camino mínimo final",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            fontsize=7.1,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.8,
            handlelength=2.1,
            borderpad=0.55,
        )

    def _dibujar_tabla_astar(
        self,
        ax,
        nodes,
        g_scores,
        h_scores,
        f_scores,
        predecessors,
        open_nodes,
        closed_nodes,
        current,
    ):
        """
        Dibuja tarjetas compactas con los valores de A*.

        Cada tarjeta contiene:
        - g: coste real acumulado;
        - h: estimación restante;
        - f = g + h;
        - predecesor.

        Colores:
        - gris: no descubierto;
        - amarillo: conjunto abierto;
        - azul: conjunto cerrado;
        - borde rojo: vértice actual.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            "Valores de A*: g, h, f y predecesor",
            fontsize=10.8,
            fontweight="bold",
            ha="center",
            va="top",
        )

        number_of_columns = 2

        card_width = 0.405
        card_height = 0.086
        horizontal_gap = 0.055
        vertical_gap = 0.014

        total_width = (
            number_of_columns * card_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.675

        for index, node in enumerate(nodes):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (card_width + horizontal_gap)
            y = top_y - row * (card_height + vertical_gap)

            g_value = g_scores.get(node, float("inf"))
            h_value = h_scores.get(node, float("inf"))
            f_value = f_scores.get(node, float("inf"))
            predecessor = predecessors.get(node)

            if node in closed_nodes:
                face_color = "#B7D7F0"
                edge_color = "#1F4F73"
            elif node in open_nodes:
                face_color = "#FBE5A6"
                edge_color = "#8A6D1D"
            else:
                face_color = "#E5E5E5"
                edge_color = "#777777"

            line_width = 1.5

            if node == current:
                edge_color = "#C62828"
                line_width = 3.0

            rectangle = Rectangle(
                (x, y),
                card_width,
                card_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            predecessor_text = (
                "—"
                if predecessor is None
                else str(predecessor)
            )

            ax.text(
                x + card_width * 0.10,
                y + card_height * 0.66,
                str(node),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + card_width * 0.27,
                y + card_height * 0.66,
                f"g={self._formatear_distancia(g_value)}",
                fontsize=7.0,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.62,
                y + card_height * 0.66,
                f"h={self._formatear_distancia(h_value)}",
                fontsize=7.0,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.27,
                y + card_height * 0.28,
                f"f={self._formatear_distancia(f_value)}",
                fontsize=7.0,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.62,
                y + card_height * 0.28,
                f"pred={predecessor_text}",
                fontsize=6.9,
                ha="left",
                va="center",
            )

        ax.text(
            0.50,
            0.055,
            (
                "Gris: no descubierto   ·   "
                "Amarillo: abierto   ·   "
                "Azul: cerrado"
            ),
            fontsize=6.6,
            ha="center",
            va="center",
            color="#444444",
        )

        self._dibujar_leyenda_astar(ax)

    def _dibujar_cola_prioridad_astar(
        self,
        ax,
        priority_queue,
    ):
        """
        Dibuja la cola de prioridad de A* debajo del grafo.

        Cada entrada se representa mediante:
        - el vértice;
        - su prioridad f;
        - su valor heurístico h.

        La primera celda es el candidato con menor tupla (f, h).
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.02,
            0.82,
            "Cola de prioridad de A*",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.text(
            0.02,
            0.41,
            "Menor f",
            fontsize=8.5,
            ha="left",
            va="center",
        )

        ax.text(
            0.98,
            0.41,
            "Prioridad mayor",
            fontsize=8.5,
            ha="right",
            va="center",
        )

        queue_sorted = sorted(priority_queue)

        if not queue_sorted:
            ax.text(
                0.50,
                0.41,
                "Cola vacía",
                fontsize=11.5,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.42",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.98,
                },
            )
            return

        max_cells = 10
        visible_queue = queue_sorted[:max_cells]

        initial_x = 0.12
        final_x = 0.88
        total_width = final_x - initial_x
        cell_width = min(
            0.078,
            total_width / max(len(visible_queue), 1),
        )
        gap = 0.009

        occupied_width = (
            len(visible_queue) * cell_width
            + max(0, len(visible_queue) - 1) * gap
        )

        current_x = 0.50 - occupied_width / 2

        for index, (f_value, h_value, node) in enumerate(visible_queue):
            is_minimum = index == 0

            rectangle = Rectangle(
                (current_x, 0.20),
                cell_width,
                0.43,
                facecolor="#E45756" if is_minimum else "#F6C85F",
                edgecolor="#7A1D1D" if is_minimum else "#8A6D1D",
                linewidth=2.0 if is_minimum else 1.5,
            )
            ax.add_patch(rectangle)

            ax.text(
                current_x + cell_width / 2,
                0.49,
                str(node),
                fontsize=8.8,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                current_x + cell_width / 2,
                0.36,
                f"f={self._formatear_distancia(f_value)}",
                fontsize=7.1,
                ha="center",
                va="center",
            )

            ax.text(
                current_x + cell_width / 2,
                0.25,
                f"h={self._formatear_distancia(h_value)}",
                fontsize=6.8,
                ha="center",
                va="center",
            )

            if is_minimum:
                ax.text(
                    current_x + cell_width / 2,
                    0.12,
                    "siguiente",
                    fontsize=6.5,
                    ha="center",
                    va="top",
                )

            current_x += cell_width + gap

        if len(queue_sorted) > max_cells:
            ax.text(
                0.91,
                0.41,
                f"+{len(queue_sorted) - max_cells}",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="center",
            )

    def _dibujar_estado_astar(
        self,
        graph_ax,
        table_ax,
        queue_ax,
        graph,
        pos,
        state,
        source_node,
        target_node,
    ):
        """
        Dibuja un estado completo del algoritmo A*.
        """

        graph_ax.clear()
        graph_ax.axis("off")

        limits = self._calcular_limites(
            pos,
            margin_x=1.2,
            margin_y=1.0,
        )

        graph_ax.set_xlim(limits[0], limits[1])
        graph_ax.set_ylim(limits[2], limits[3])
        graph_ax.set_aspect("equal", adjustable="box")

        current = state.get("current")
        g_scores = dict(state.get("g_scores", {}))
        h_scores = dict(state.get("h_scores", {}))
        f_scores = dict(state.get("f_scores", {}))
        predecessors = dict(state.get("predecessors", {}))
        open_nodes = set(state.get("open_nodes", set()))
        closed_nodes = set(state.get("closed_nodes", set()))
        priority_queue = list(state.get("priority_queue", []))
        active_edge = state.get("active_edge")
        action = state.get("action")
        final_path = list(state.get("final_path", []))

        predecessor_edges = {
            self._normalizar_arista(predecessor, node)
            for node, predecessor in predecessors.items()
            if predecessor is not None
        }

        final_path_edges = {
            self._normalizar_arista(u, v)
            for u, v in zip(final_path[:-1], final_path[1:])
        }

        active_edge_normalized = None

        if active_edge is not None:
            active_edge_normalized = self._normalizar_arista(*active_edge)

        # 1. Aristas y pesos.
        for u, v, data in graph.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            edge_key = self._normalizar_arista(u, v)

            if edge_key == active_edge_normalized:
                color = (
                    "#F28E2B"
                    if action == "no_improvement"
                    else "#E45756"
                )
                line_width = 4.2
                zorder = 20
            elif edge_key in final_path_edges:
                color = "#D62728"
                line_width = 4.3
                zorder = 18
            elif edge_key in predecessor_edges:
                color = "#2E8B57"
                line_width = 3.0
                zorder = 15
            else:
                color = "#B8B8B8"
                line_width = 1.7
                zorder = 10

            graph_ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=line_width,
                zorder=zorder,
            )

            self._dibujar_peso_arista(
                ax=graph_ax,
                pos=pos,
                origen=u,
                destino=v,
                peso=data.get("weight", 1),
            )

        # 2. Estados de los nodos.
        undiscovered_nodes = [
            node
            for node in graph.nodes()
            if (
                node not in open_nodes
                and node not in closed_nodes
                and node != current
            )
        ]

        open_nodes_to_draw = [
            node
            for node in open_nodes
            if node != current
        ]

        closed_nodes_to_draw = [
            node
            for node in closed_nodes
            if node != current
        ]

        if undiscovered_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=undiscovered_nodes,
                node_size=760,
                node_color="#D9D9D9",
                edgecolors="#666666",
                linewidths=1.3,
                ax=graph_ax,
            )
            collection.set_zorder(22)

        if open_nodes_to_draw:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=open_nodes_to_draw,
                node_size=790,
                node_color="#F6C85F",
                edgecolors="#8A6D1D",
                linewidths=1.6,
                ax=graph_ax,
            )
            collection.set_zorder(23)

        if closed_nodes_to_draw:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=closed_nodes_to_draw,
                node_size=790,
                node_color="#4C9ED9",
                edgecolors="#1F4F73",
                linewidths=1.6,
                ax=graph_ax,
            )
            collection.set_zorder(23)

        if current is not None:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[current],
                node_size=930,
                node_color="#E45756",
                edgecolors="#7A1D1D",
                linewidths=2.5,
                ax=graph_ax,
            )
            collection.set_zorder(26)

        # 3. Etiquetas de nodos y valores g/f.
        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

            g_text = self._formatear_distancia(
                g_scores.get(node, float("inf"))
            )
            f_text = self._formatear_distancia(
                f_scores.get(node, float("inf"))
            )

            graph_ax.text(
                x,
                y + 0.39,
                f"g={g_text} | f={f_text}",
                fontsize=7.0,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#222222",
                zorder=40,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "#555555",
                    "alpha": 0.97,
                },
            )

        # 4. Origen y destino.
        source_x, source_y = pos[source_node]
        target_x, target_y = pos[target_node]

        graph_ax.text(
            source_x,
            source_y - 0.43,
            "origen",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        graph_ax.text(
            target_x,
            target_y - 0.43,
            "destino",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        # 5. Mensaje y relajación.
        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=9.2,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        candidate_g = state.get("candidate_g")
        candidate_f = state.get("candidate_f")
        old_g = state.get("old_g")
        edge_weight = state.get("edge_weight")

        if active_edge is not None and candidate_g is not None:
            origin, destination = active_edge

            operation_text = (
                f"Relajación {origin}→{destination}: "
                f"g candidato = "
                f"{self._formatear_distancia(g_scores.get(origin, float('inf')))}"
                f" + {self._formatear_distancia(edge_weight)}"
                f" = {self._formatear_distancia(candidate_g)}"
                f"  |  g anterior: {self._formatear_distancia(old_g)}"
                f"  |  f candidato: {self._formatear_distancia(candidate_f)}"
            )

            graph_ax.text(
                0.50,
                0.965,
                operation_text,
                transform=graph_ax.transAxes,
                fontsize=7.9,
                ha="center",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
                zorder=50,
            )

        graph_ax.text(
            0.99,
            0.985,
            (
                f"Cerrados: {len(closed_nodes)} de "
                f"{graph.number_of_nodes()}"
            ),
            transform=graph_ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        self._dibujar_tabla_astar(
            ax=table_ax,
            nodes=sorted(graph.nodes()),
            g_scores=g_scores,
            h_scores=h_scores,
            f_scores=f_scores,
            predecessors=predecessors,
            open_nodes=open_nodes,
            closed_nodes=closed_nodes,
            current=current,
        )

        self._dibujar_cola_prioridad_astar(
            ax=queue_ax,
            priority_queue=priority_queue,
        )

    def animate_astar(
        self,
        graph,
        pos,
        states,
        source_node,
        target_node,
        title="Caminos mínimos con A*",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima A* con la misma estructura visual utilizada por Dijkstra.

        La imagen final muestra:
        - valores g, h y f;
        - predecesores;
        - conjuntos abierto y cerrado;
        - candidatos que no fue necesario expandir;
        - camino mínimo final.
        """

        if not states:
            raise ValueError(
                "La lista de estados de A* no puede estar vacía."
            )

        (
            fig,
            graph_ax,
            table_ax,
            queue_ax,
        ) = self._preparar_figura_astar(title)

        if final_image_path is not None:
            self._dibujar_estado_astar(
                graph_ax=graph_ax,
                table_ax=table_ax,
                queue_ax=queue_ax,
                graph=graph,
                pos=pos,
                state=states[-1],
                source_node=source_node,
                target_node=target_node,
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(
                f"Imagen final guardada en: "
                f"{final_image_path}"
            )

        def init():
            self._dibujar_estado_astar(
                graph_ax=graph_ax,
                table_ax=table_ax,
                queue_ax=queue_ax,
                graph=graph,
                pos=pos,
                state=states[0],
                source_node=source_node,
                target_node=target_node,
            )
            return []

        def update(frame_index):
            self._dibujar_estado_astar(
                graph_ax=graph_ax,
                table_ax=table_ax,
                queue_ax=queue_ax,
                graph=graph,
                pos=pos,
                state=states[frame_index],
                source_node=source_node,
                target_node=target_node,
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()

        return self.animation

    # ------------------------------------------------------------------
    # Elementos específicos de Bellman-Ford
    # ------------------------------------------------------------------

    def _preparar_figura_bellman_ford(self, title):
        """
        Reutiliza la distribución visual de Dijkstra y A*.

        Distribución:
        - izquierda: leyenda y tarjetas de distancia/predecesor;
        - derecha superior: grafo dirigido ponderado;
        - derecha inferior: aristas de la pasada actual.
        """

        return self._preparar_figura_dijkstra(title)

    def _dibujar_leyenda_bellman_ford(self, ax):
        """
        Dibuja la leyenda de Bellman-Ford en el panel izquierdo.
        """

        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="#666666",
                markersize=8,
                label="No alcanzado",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F6C85F",
                markeredgecolor="#8A6D1D",
                markersize=8,
                label="Distancia provisional",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4C9ED9",
                markeredgecolor="#1F4F73",
                markersize=8,
                label="Distancia final",
            ),
            Line2D(
                [0],
                [0],
                color="#E45756",
                linewidth=4,
                label="Arista examinada",
            ),
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=3,
                label="Árbol de predecesores",
            ),
            Line2D(
                [0],
                [0],
                color="#D62728",
                linewidth=4,
                label="Camino mínimo final",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            fontsize=7.0,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.8,
            handlelength=2.1,
            borderpad=0.55,
        )

    def _dibujar_flecha_bellman_ford(
        self,
        ax,
        pos,
        origin,
        destination,
        color,
        line_width,
        zorder,
        line_style="solid",
    ):
        """
        Dibuja una arista dirigida evitando que la punta tape los nodos.
        """

        x1, y1 = pos[origin]
        x2, y2 = pos[destination]

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=line_width,
            linestyle=line_style,
            color=color,
            shrinkA=18,
            shrinkB=18,
            connectionstyle="arc3,rad=0.0",
            zorder=zorder,
        )
        ax.add_patch(arrow)

    def _dibujar_peso_arista_bellman_ford(
        self,
        ax,
        pos,
        origin,
        destination,
        weight,
    ):
        """
        Dibuja el peso de una arista dirigida.

        Los pesos negativos se muestran en morado para identificarlos
        inmediatamente.
        """

        x1, y1 = pos[origin]
        x2, y2 = pos[destination]

        middle_x = (x1 + x2) / 2
        middle_y = (y1 + y2) / 2

        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2) ** 0.5

        if length == 0:
            offset_x = 0
            offset_y = 0
        else:
            offset_x = -dy / length * 0.15
            offset_y = dx / length * 0.15

        text_color = "#7B2CBF" if weight < 0 else "#222222"

        ax.text(
            middle_x + offset_x,
            middle_y + offset_y,
            self._formatear_distancia(weight),
            fontsize=8,
            fontweight="bold" if weight < 0 else "normal",
            ha="center",
            va="center",
            color=text_color,
            zorder=35,
            bbox={
                "boxstyle": "round,pad=0.16",
                "fc": "white",
                "ec": "none",
                "alpha": 0.96,
            },
        )

    def _dibujar_tabla_bellman_ford(
        self,
        ax,
        nodes,
        distances,
        predecessors,
        final_distances,
        current_source,
        current_target,
        negative_cycle,
    ):
        """
        Dibuja tarjetas de distancia y predecesor.

        Durante las pasadas, las distancias alcanzables permanecen
        provisionales. Solo cambian a azul cuando Bellman-Ford termina
        sin detectar un ciclo negativo.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            "Distancias y predecesores",
            fontsize=11.2,
            fontweight="bold",
            ha="center",
            va="top",
        )

        number_of_columns = 2
        card_width = 0.405
        card_height = 0.078
        horizontal_gap = 0.055
        vertical_gap = 0.018

        total_width = (
            number_of_columns * card_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.665

        for index, node in enumerate(nodes):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (card_width + horizontal_gap)
            y = top_y - row * (card_height + vertical_gap)

            distance = distances.get(node, float("inf"))
            predecessor = predecessors.get(node)

            if negative_cycle and node == current_target:
                face_color = "#F6B4B4"
                edge_color = "#8B0000"
            elif final_distances and distance != float("inf"):
                face_color = "#B7D7F0"
                edge_color = "#1F4F73"
            elif distance != float("inf"):
                face_color = "#FBE5A6"
                edge_color = "#8A6D1D"
            else:
                face_color = "#E5E5E5"
                edge_color = "#777777"

            line_width = 1.5

            if node == current_target:
                edge_color = "#C62828"
                line_width = 3.0
            elif node == current_source:
                edge_color = "#F28E2B"
                line_width = 2.4

            rectangle = Rectangle(
                (x, y),
                card_width,
                card_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            predecessor_text = (
                "—"
                if predecessor is None
                else str(predecessor)
            )

            ax.text(
                x + card_width * 0.11,
                y + card_height / 2,
                str(node),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + card_width * 0.31,
                y + card_height / 2,
                f"d={self._formatear_distancia(distance)}",
                fontsize=7.4,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.62,
                y + card_height / 2,
                f"pred={predecessor_text}",
                fontsize=7.1,
                ha="left",
                va="center",
            )

        ax.text(
            0.50,
            0.055,
            (
                "Gris: no alcanzado   ·   "
                "Amarillo: provisional   ·   "
                "Azul: final"
            ),
            fontsize=6.7,
            ha="center",
            va="center",
            color="#444444",
        )

        self._dibujar_leyenda_bellman_ford(ax)

    def _dibujar_pasada_bellman_ford(
        self,
        ax,
        edge_order,
        processed_edges,
        active_edge_index,
        iteration,
        max_iterations,
        pass_changes,
        phase,
        negative_cycle,
    ):
        """
        Dibuja debajo del grafo todas las aristas de la pasada actual.

        Cada celda contiene:
        - origen y destino;
        - peso de la arista.

        Colores:
        - amarillo: pendiente;
        - azul: ya examinada;
        - rojo: arista actual;
        - morado: comprobación de ciclos negativos.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if phase == "negative_cycle_check":
            title = "Comprobación adicional de ciclos negativos"
        elif phase == "finished":
            title = "Bellman-Ford finalizado"
        elif phase == "initial":
            title = "Lista ordenada de aristas"
        else:
            title = (
                f"Pasada {iteration} de {max_iterations}"
                f"  ·  mejoras acumuladas: {pass_changes}"
            )

        ax.text(
            0.02,
            0.88,
            title,
            fontsize=11.5,
            fontweight="bold",
            ha="left",
            va="center",
        )

        if negative_cycle:
            ax.text(
                0.98,
                0.88,
                "Ciclo negativo alcanzable detectado",
                fontsize=9,
                fontweight="bold",
                color="#8B0000",
                ha="right",
                va="center",
            )
        elif phase == "finished":
            ax.text(
                0.98,
                0.88,
                "Sin ciclo negativo alcanzable",
                fontsize=9,
                fontweight="bold",
                color="#1F4F73",
                ha="right",
                va="center",
            )

        number_of_columns = 8
        number_of_rows = (
            len(edge_order) + number_of_columns - 1
        ) // number_of_columns

        cell_width = 0.097
        cell_height = 0.28
        horizontal_gap = 0.012
        vertical_gap = 0.08

        total_width = (
            number_of_columns * cell_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.48

        for index, (origin, destination, weight) in enumerate(edge_order):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (cell_width + horizontal_gap)
            y = top_y - row * (cell_height + vertical_gap)

            is_current = index == active_edge_index
            is_processed = index < processed_edges

            if is_current:
                face_color = "#E45756"
                edge_color = "#7A1D1D"
                line_width = 2.2
            elif phase == "negative_cycle_check" and not is_processed:
                face_color = "#E8D7F1"
                edge_color = "#8E5EA2"
                line_width = 1.4
            elif is_processed or phase == "finished":
                face_color = "#B7D7F0"
                edge_color = "#1F4F73"
                line_width = 1.4
            else:
                face_color = "#FBE5A6"
                edge_color = "#8A6D1D"
                line_width = 1.4

            rectangle = Rectangle(
                (x, y),
                cell_width,
                cell_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            ax.text(
                x + cell_width / 2,
                y + cell_height * 0.66,
                f"{origin}→{destination}",
                fontsize=7.2,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + cell_width / 2,
                y + cell_height * 0.30,
                f"w={self._formatear_distancia(weight)}",
                fontsize=6.8,
                ha="center",
                va="center",
                color="#7B2CBF" if weight < 0 else "#222222",
            )

        if number_of_rows == 1:
            ax.set_ylim(0.10, 1.0)

    def _dibujar_estado_bellman_ford(
        self,
        graph_ax,
        table_ax,
        pass_ax,
        graph,
        pos,
        state,
        source_node,
        target_node,
    ):
        """
        Dibuja un estado completo de Bellman-Ford.
        """

        graph_ax.clear()
        graph_ax.axis("off")

        limits = self._calcular_limites(
            pos,
            margin_x=1.2,
            margin_y=1.0,
        )

        graph_ax.set_xlim(limits[0], limits[1])
        graph_ax.set_ylim(limits[2], limits[3])
        graph_ax.set_aspect("equal", adjustable="box")

        distances = dict(state.get("distances", {}))
        predecessors = dict(state.get("predecessors", {}))
        edge_order = list(state.get("edge_order", []))
        active_edge = state.get("active_edge")
        active_edge_index = state.get("active_edge_index")
        current_source = state.get("current_source")
        current_target = state.get("current_target")
        action = state.get("action")
        final_path = list(state.get("final_path", []))
        final_distances = bool(state.get("final_distances", False))
        negative_cycle = bool(state.get("negative_cycle", False))
        phase = state.get("phase", "relaxation")

        predecessor_edges = {
            (predecessor, node)
            for node, predecessor in predecessors.items()
            if predecessor is not None
        }

        final_path_edges = {
            (u, v)
            for u, v in zip(final_path[:-1], final_path[1:])
        }

        # 1. Aristas dirigidas y pesos.
        for origin, destination, data in graph.edges(data=True):
            edge_key = (origin, destination)

            if edge_key == active_edge:
                if action == "negative_cycle":
                    color = "#8B0000"
                elif action == "no_improvement":
                    color = "#F28E2B"
                elif action == "unreachable_source":
                    color = "#888888"
                else:
                    color = "#E45756"

                line_width = 4.1
                zorder = 20
                line_style = (
                    "dashed"
                    if action == "unreachable_source"
                    else "solid"
                )
            elif edge_key in final_path_edges:
                color = "#D62728"
                line_width = 4.2
                zorder = 18
                line_style = "solid"
            elif edge_key in predecessor_edges:
                color = "#2E8B57"
                line_width = 3.0
                zorder = 15
                line_style = "solid"
            else:
                color = "#B8B8B8"
                line_width = 1.6
                zorder = 10
                line_style = "solid"

            self._dibujar_flecha_bellman_ford(
                ax=graph_ax,
                pos=pos,
                origin=origin,
                destination=destination,
                color=color,
                line_width=line_width,
                zorder=zorder,
                line_style=line_style,
            )

            self._dibujar_peso_arista_bellman_ford(
                ax=graph_ax,
                pos=pos,
                origin=origin,
                destination=destination,
                weight=data.get("weight", 1),
            )

        # 2. Estados de los nodos.
        unreachable_nodes = [
            node
            for node in graph.nodes()
            if distances.get(node, float("inf")) == float("inf")
        ]

        reached_nodes = [
            node
            for node in graph.nodes()
            if (
                distances.get(node, float("inf")) != float("inf")
                and node not in {current_source, current_target}
            )
        ]

        if unreachable_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=unreachable_nodes,
                node_size=760,
                node_color="#D9D9D9",
                edgecolors="#666666",
                linewidths=1.3,
                ax=graph_ax,
            )
            collection.set_zorder(22)

        if reached_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=reached_nodes,
                node_size=790,
                node_color=(
                    "#4C9ED9"
                    if final_distances
                    else "#F6C85F"
                ),
                edgecolors=(
                    "#1F4F73"
                    if final_distances
                    else "#8A6D1D"
                ),
                linewidths=1.6,
                ax=graph_ax,
            )
            collection.set_zorder(23)

        if current_source is not None:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[current_source],
                node_size=900,
                node_color="#E45756",
                edgecolors="#7A1D1D",
                linewidths=2.4,
                ax=graph_ax,
            )
            collection.set_zorder(26)

        if (
            current_target is not None
            and current_target != current_source
        ):
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[current_target],
                node_size=860,
                node_color="#F28E2B",
                edgecolors="#8A4B08",
                linewidths=2.3,
                ax=graph_ax,
            )
            collection.set_zorder(25)

        # 3. Etiquetas de nodos y distancias.
        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

            distance_text = self._formatear_distancia(
                distances.get(node, float("inf"))
            )

            graph_ax.text(
                x,
                y + 0.39,
                f"d={distance_text}",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#222222",
                zorder=40,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "#555555",
                    "alpha": 0.97,
                },
            )

        # 4. Origen y destino.
        source_x, source_y = pos[source_node]
        target_x, target_y = pos[target_node]

        graph_ax.text(
            source_x,
            source_y - 0.43,
            "origen",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        graph_ax.text(
            target_x,
            target_y - 0.43,
            "destino",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        # 5. Mensaje y operación de relajación.
        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=9.2,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        candidate = state.get("candidate")
        old_distance = state.get("old_distance")
        edge_weight = state.get("edge_weight")

        if active_edge is not None and candidate is not None:
            origin, destination = active_edge

            operation_text = (
                f"Relajación {origin}→{destination}: "
                f"{self._formatear_distancia(distances.get(origin, float('inf')))}"
                f" + {self._formatear_distancia(edge_weight)}"
                f" = {self._formatear_distancia(candidate)}"
                f"  |  anterior: "
                f"{self._formatear_distancia(old_distance)}"
            )

            graph_ax.text(
                0.50,
                0.965,
                operation_text,
                transform=graph_ax.transAxes,
                fontsize=8.1,
                ha="center",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
                zorder=50,
            )

        iteration = state.get("iteration", 0)
        max_iterations = state.get(
            "max_iterations",
            max(graph.number_of_nodes() - 1, 0),
        )
        pass_changes = state.get("pass_changes", 0)

        if phase == "negative_cycle_check":
            status_text = "Comprobando ciclos negativos"
        elif phase == "finished":
            status_text = (
                "Ciclo negativo detectado"
                if negative_cycle
                else "Distancias finales"
            )
        else:
            status_text = (
                f"Pasada {iteration}/{max_iterations}"
                f"  |  mejoras: {pass_changes}"
            )

        graph_ax.text(
            0.99,
            0.985,
            status_text,
            transform=graph_ax.transAxes,
            fontsize=8.8,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        self._dibujar_tabla_bellman_ford(
            ax=table_ax,
            nodes=sorted(graph.nodes()),
            distances=distances,
            predecessors=predecessors,
            final_distances=final_distances,
            current_source=current_source,
            current_target=current_target,
            negative_cycle=negative_cycle,
        )

        self._dibujar_pasada_bellman_ford(
            ax=pass_ax,
            edge_order=edge_order,
            processed_edges=state.get("processed_edges", 0),
            active_edge_index=active_edge_index,
            iteration=iteration,
            max_iterations=max_iterations,
            pass_changes=pass_changes,
            phase=phase,
            negative_cycle=negative_cycle,
        )

    def animate_bellman_ford(
        self,
        graph,
        pos,
        states,
        source_node,
        target_node,
        title="Caminos mínimos con Bellman-Ford",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima Bellman-Ford con una distribución comparable a Dijkstra y A*.

        La imagen final muestra:
        - distancias y predecesores;
        - aristas dirigidas y pesos negativos;
        - árbol de predecesores;
        - camino mínimo hacia el destino;
        - resultado de la comprobación de ciclos negativos.
        """

        if not states:
            raise ValueError(
                "La lista de estados de Bellman-Ford no puede estar vacía."
            )

        (
            fig,
            graph_ax,
            table_ax,
            pass_ax,
        ) = self._preparar_figura_bellman_ford(title)

        if final_image_path is not None:
            self._dibujar_estado_bellman_ford(
                graph_ax=graph_ax,
                table_ax=table_ax,
                pass_ax=pass_ax,
                graph=graph,
                pos=pos,
                state=states[-1],
                source_node=source_node,
                target_node=target_node,
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(
                f"Imagen final guardada en: "
                f"{final_image_path}"
            )

        def init():
            self._dibujar_estado_bellman_ford(
                graph_ax=graph_ax,
                table_ax=table_ax,
                pass_ax=pass_ax,
                graph=graph,
                pos=pos,
                state=states[0],
                source_node=source_node,
                target_node=target_node,
            )
            return []

        def update(frame_index):
            self._dibujar_estado_bellman_ford(
                graph_ax=graph_ax,
                table_ax=table_ax,
                pass_ax=pass_ax,
                graph=graph,
                pos=pos,
                state=states[frame_index],
                source_node=source_node,
                target_node=target_node,
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()

        return self.animation

    # ------------------------------------------------------------------
    # Elementos específicos de Floyd-Warshall
    # ------------------------------------------------------------------

    def _preparar_figura_floyd_warshall(self, title):
        """
        Crea una distribución comparable con Dijkstra, A* y Bellman-Ford.

        Distribución:
        - izquierda: leyenda y fila activa de la matriz;
        - derecha superior: grafo dirigido ponderado;
        - derecha inferior: matriz completa de distancias.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.65, 4.35],
            height_ratios=[4.55, 2.15],
            wspace=0.08,
            hspace=0.10,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        graph_ax = fig.add_subplot(grid[0, 1])
        matrix_ax = fig.add_subplot(grid[1, 1])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.93,
            bottom=0.045,
        )

        return fig, graph_ax, info_ax, matrix_ax

    def _dibujar_leyenda_floyd_warshall(self, ax):
        """
        Dibuja la leyenda de Floyd-Warshall en el panel izquierdo.
        """

        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="#666666",
                markersize=8,
                label="No procesado como k",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4C9ED9",
                markeredgecolor="#1F4F73",
                markersize=8,
                label="Ya procesado como k",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#8E5EA2",
                markeredgecolor="#5A316B",
                markersize=8,
                label="Intermedio k",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D",
                markersize=8,
                label="Origen i",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F28E2B",
                markeredgecolor="#8A4B08",
                markersize=8,
                label="Destino j",
            ),
            Line2D(
                [0],
                [0],
                color="#D62728",
                linewidth=4,
                label="Camino mínimo final",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            fontsize=6.9,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.7,
            handlelength=2.0,
            borderpad=0.50,
        )

    def _dibujar_fila_floyd_warshall(
        self,
        ax,
        nodes,
        distances,
        next_nodes,
        row_node,
        active_target,
        via_node,
        final_state,
        negative_cycle_nodes,
    ):
        """
        Dibuja tarjetas compactas para una fila de la matriz.

        Cada tarjeta contiene:
        - distancia desde la fila activa;
        - siguiente vértice del camino.

        La fila activa permite comparar visualmente Floyd-Warshall con
        las tarjetas de distancia/predecesor de Dijkstra y Bellman-Ford.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        row_label = "—" if row_node is None else str(row_node)

        ax.text(
            0.50,
            0.985,
            f"Fila activa: origen {row_label}",
            fontsize=11.0,
            fontweight="bold",
            ha="center",
            va="top",
        )

        number_of_columns = 2
        card_width = 0.405
        card_height = 0.080
        horizontal_gap = 0.055
        vertical_gap = 0.016

        total_width = (
            number_of_columns * card_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.660

        for index, node in enumerate(nodes):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (card_width + horizontal_gap)
            y = top_y - row * (card_height + vertical_gap)

            if row_node is None:
                distance = float("inf")
                next_node = None
            else:
                distance = distances[row_node][node]
                next_node = next_nodes[row_node][node]

            if row_node in negative_cycle_nodes:
                face_color = "#F6B4B4"
                edge_color = "#8B0000"
            elif final_state and distance != float("inf"):
                face_color = "#B7D7F0"
                edge_color = "#1F4F73"
            elif distance != float("inf"):
                face_color = "#FBE5A6"
                edge_color = "#8A6D1D"
            else:
                face_color = "#E5E5E5"
                edge_color = "#777777"

            line_width = 1.5

            if node == active_target:
                edge_color = "#C62828"
                line_width = 3.0
            elif node == via_node:
                edge_color = "#8E5EA2"
                line_width = 2.6

            rectangle = Rectangle(
                (x, y),
                card_width,
                card_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            next_text = "—" if next_node is None else str(next_node)

            ax.text(
                x + card_width * 0.10,
                y + card_height / 2,
                str(node),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + card_width * 0.30,
                y + card_height / 2,
                f"d={self._formatear_distancia(distance)}",
                fontsize=7.2,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.62,
                y + card_height / 2,
                f"sig={next_text}",
                fontsize=7.0,
                ha="left",
                va="center",
            )

        ax.text(
            0.50,
            0.055,
            (
                "Gris: ∞   ·   Amarillo: conocida   ·   "
                "Azul: resultado final"
            ),
            fontsize=6.6,
            ha="center",
            va="center",
            color="#444444",
        )

        self._dibujar_leyenda_floyd_warshall(ax)

    def _dibujar_matriz_floyd_warshall(
        self,
        ax,
        nodes,
        distances,
        active_i,
        active_j,
        active_k,
        action,
        updates_for_k,
        phase,
        negative_cycle_nodes,
    ):
        """
        Dibuja la matriz completa de distancias debajo del grafo.

        Resaltados:
        - morado: fila o columna del intermedio k;
        - amarillo: celdas d[i][k] y d[k][j];
        - rojo/verde: celda d[i][j] examinada o mejorada;
        - rojo oscuro: diagonal negativa.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if active_k is None:
            matrix_title = "Matriz de distancias"
        else:
            matrix_title = (
                f"Matriz de distancias · intermedio k={active_k}"
                f" · mejoras con k: {updates_for_k}"
            )

        ax.text(
            0.02,
            0.955,
            matrix_title,
            fontsize=11.2,
            fontweight="bold",
            ha="left",
            va="top",
        )

        number_of_nodes = len(nodes)
        cell_size = min(
            0.080,
            0.72 / max(number_of_nodes, 1),
        )

        matrix_width = number_of_nodes * cell_size
        matrix_height = number_of_nodes * cell_size

        start_x = 0.50 - matrix_width / 2
        start_y = 0.08

        # Cabeceras de columnas y filas.
        for index, node in enumerate(nodes):
            x = start_x + index * cell_size
            y = start_y + (number_of_nodes - 1 - index) * cell_size

            ax.text(
                x + cell_size / 2,
                start_y + matrix_height + 0.035,
                str(node),
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                start_x - 0.025,
                y + cell_size / 2,
                str(node),
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
            )

        node_index = {
            node: index
            for index, node in enumerate(nodes)
        }

        for row_index, origin in enumerate(nodes):
            for column_index, destination in enumerate(nodes):
                x = start_x + column_index * cell_size
                y = (
                    start_y
                    + (number_of_nodes - 1 - row_index) * cell_size
                )

                face_color = "white"
                edge_color = "#AAAAAA"
                line_width = 0.8

                if (
                    origin in negative_cycle_nodes
                    and origin == destination
                ):
                    face_color = "#F6B4B4"
                    edge_color = "#8B0000"
                    line_width = 2.2
                elif (
                    active_i == origin
                    and active_j == destination
                ):
                    if action == "improvement":
                        face_color = "#B7E4C7"
                        edge_color = "#2E8B57"
                    else:
                        face_color = "#F6B4B4"
                        edge_color = "#C62828"
                    line_width = 2.2
                elif (
                    active_i == origin
                    and active_k == destination
                ) or (
                    active_k == origin
                    and active_j == destination
                ):
                    face_color = "#FBE5A6"
                    edge_color = "#8A6D1D"
                    line_width = 1.8
                elif (
                    active_k == origin
                    or active_k == destination
                ):
                    face_color = "#EEE3F3"
                    edge_color = "#8E5EA2"
                    line_width = 1.2

                rectangle = Rectangle(
                    (x, y),
                    cell_size,
                    cell_size,
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=line_width,
                )
                ax.add_patch(rectangle)

                ax.text(
                    x + cell_size / 2,
                    y + cell_size / 2,
                    self._formatear_distancia(
                        distances[origin][destination]
                    ),
                    fontsize=6.7,
                    ha="center",
                    va="center",
                )

        if phase == "finished":
            status_text = (
                "Diagonal negativa: existe ciclo negativo"
                if negative_cycle_nodes
                else "Matriz final · sin ciclos negativos"
            )
        elif phase == "initial":
            status_text = "Solo caminos directos"
        elif active_i is not None and active_j is not None:
            status_text = (
                f"Se compara d[{active_i}][{active_j}] con "
                f"d[{active_i}][{active_k}] + d[{active_k}][{active_j}]"
            )
        else:
            status_text = "Se habilita un nuevo vértice intermedio"

        ax.text(
            0.98,
            0.955,
            status_text,
            fontsize=8.2,
            ha="right",
            va="top",
            color="#444444",
        )

    def _dibujar_estado_floyd_warshall(
        self,
        graph_ax,
        info_ax,
        matrix_ax,
        graph,
        pos,
        state,
        source_node,
        target_node,
    ):
        """
        Dibuja un estado completo de Floyd-Warshall.
        """

        graph_ax.clear()
        graph_ax.axis("off")

        limits = self._calcular_limites(
            pos,
            margin_x=1.2,
            margin_y=1.0,
        )

        graph_ax.set_xlim(limits[0], limits[1])
        graph_ax.set_ylim(limits[2], limits[3])
        graph_ax.set_aspect("equal", adjustable="box")

        nodes = list(state.get("nodes", sorted(graph.nodes())))
        distances = {
            origin: dict(row)
            for origin, row in state.get("distances", {}).items()
        }
        next_nodes = {
            origin: dict(row)
            for origin, row in state.get("next_nodes", {}).items()
        }

        active_i = state.get("active_i")
        active_j = state.get("active_j")
        active_k = state.get("active_k")
        action = state.get("action")
        phase = state.get("phase", "iteration")
        candidate_path = list(state.get("candidate_path", []))
        final_path = list(state.get("final_path", []))
        processed_intermediates = set(
            state.get("processed_intermediates", set())
        )
        negative_cycle_nodes = set(
            state.get("negative_cycle_nodes", set())
        )

        candidate_edges = {
            (u, v)
            for u, v in zip(
                candidate_path[:-1],
                candidate_path[1:],
            )
        }

        final_path_edges = {
            (u, v)
            for u, v in zip(
                final_path[:-1],
                final_path[1:],
            )
        }

        # 1. Aristas dirigidas y pesos.
        for origin, destination, data in graph.edges(data=True):
            edge_key = (origin, destination)

            if edge_key in final_path_edges:
                color = "#D62728"
                line_width = 4.2
                zorder = 20
            elif edge_key in candidate_edges:
                color = (
                    "#2E8B57"
                    if action == "improvement"
                    else "#F28E2B"
                )
                line_width = 3.8
                zorder = 18
            else:
                color = "#B8B8B8"
                line_width = 1.6
                zorder = 10

            self._dibujar_flecha_bellman_ford(
                ax=graph_ax,
                pos=pos,
                origin=origin,
                destination=destination,
                color=color,
                line_width=line_width,
                zorder=zorder,
                line_style="solid",
            )

            self._dibujar_peso_arista_bellman_ford(
                ax=graph_ax,
                pos=pos,
                origin=origin,
                destination=destination,
                weight=data.get("weight", 1),
            )

        # 2. Estados de nodos.
        default_nodes = [
            node
            for node in graph.nodes()
            if node not in {
                active_i,
                active_j,
                active_k,
            }
            and node not in processed_intermediates
        ]

        processed_nodes = [
            node
            for node in processed_intermediates
            if node not in {
                active_i,
                active_j,
                active_k,
            }
        ]

        if default_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=default_nodes,
                node_size=760,
                node_color="#D9D9D9",
                edgecolors="#666666",
                linewidths=1.3,
                ax=graph_ax,
            )
            collection.set_zorder(22)

        if processed_nodes:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=processed_nodes,
                node_size=790,
                node_color="#4C9ED9",
                edgecolors="#1F4F73",
                linewidths=1.6,
                ax=graph_ax,
            )
            collection.set_zorder(23)

        if active_k is not None:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[active_k],
                node_size=950,
                node_color="#8E5EA2",
                edgecolors="#5A316B",
                linewidths=2.6,
                ax=graph_ax,
            )
            collection.set_zorder(27)

        if active_i is not None and active_i != active_k:
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[active_i],
                node_size=900,
                node_color="#E45756",
                edgecolors="#7A1D1D",
                linewidths=2.4,
                ax=graph_ax,
            )
            collection.set_zorder(26)

        if (
            active_j is not None
            and active_j not in {active_i, active_k}
        ):
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[active_j],
                node_size=860,
                node_color="#F28E2B",
                edgecolors="#8A4B08",
                linewidths=2.3,
                ax=graph_ax,
            )
            collection.set_zorder(25)

        # 3. Etiquetas de los nodos.
        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

        # 4. Origen y destino seleccionados para la comparación.
        source_x, source_y = pos[source_node]
        target_x, target_y = pos[target_node]

        graph_ax.text(
            source_x,
            source_y - 0.43,
            "origen comparado",
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        graph_ax.text(
            target_x,
            target_y - 0.43,
            "destino comparado",
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        # 5. Mensaje y fórmula.
        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=9.0,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        candidate = state.get("candidate")
        old_distance = state.get("old_distance")
        distance_ik = state.get("distance_ik")
        distance_kj = state.get("distance_kj")

        if (
            active_i is not None
            and active_j is not None
            and active_k is not None
            and candidate is not None
        ):
            operation_text = (
                f"d[{active_i}][{active_j}] = min("
                f"{self._formatear_distancia(old_distance)}, "
                f"{self._formatear_distancia(distance_ik)} + "
                f"{self._formatear_distancia(distance_kj)}"
                f" = {self._formatear_distancia(candidate)})"
            )

            graph_ax.text(
                0.50,
                0.965,
                operation_text,
                transform=graph_ax.transAxes,
                fontsize=8.0,
                ha="center",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
                zorder=50,
            )

        current_k_index = state.get("current_k_index", 0)
        total_k = state.get("total_k", len(nodes))
        total_updates = state.get("total_updates", 0)

        graph_ax.text(
            0.99,
            0.985,
            (
                f"Intermedios: {current_k_index}/{total_k}"
                f"  |  mejoras: {total_updates}"
            ),
            transform=graph_ax.transAxes,
            fontsize=8.7,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        row_node = (
            active_i
            if active_i is not None
            else source_node
        )

        self._dibujar_fila_floyd_warshall(
            ax=info_ax,
            nodes=nodes,
            distances=distances,
            next_nodes=next_nodes,
            row_node=row_node,
            active_target=active_j,
            via_node=active_k,
            final_state=phase == "finished",
            negative_cycle_nodes=negative_cycle_nodes,
        )

        self._dibujar_matriz_floyd_warshall(
            ax=matrix_ax,
            nodes=nodes,
            distances=distances,
            active_i=active_i,
            active_j=active_j,
            active_k=active_k,
            action=action,
            updates_for_k=state.get("updates_for_k", 0),
            phase=phase,
            negative_cycle_nodes=negative_cycle_nodes,
        )

    def animate_floyd_warshall(
        self,
        graph,
        pos,
        states,
        source_node,
        target_node,
        title="Caminos mínimos con Floyd-Warshall",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima Floyd-Warshall con una estructura comparable a los ejemplos
        anteriores de caminos mínimos.

        La imagen final muestra:
        - la matriz completa de distancias;
        - la fila del origen seleccionado;
        - los siguientes vértices para reconstruir caminos;
        - el camino mínimo seleccionado;
        - la comprobación de ciclos negativos.
        """

        if not states:
            raise ValueError(
                "La lista de estados de Floyd-Warshall no puede estar vacía."
            )

        (
            fig,
            graph_ax,
            info_ax,
            matrix_ax,
        ) = self._preparar_figura_floyd_warshall(title)

        if final_image_path is not None:
            self._dibujar_estado_floyd_warshall(
                graph_ax=graph_ax,
                info_ax=info_ax,
                matrix_ax=matrix_ax,
                graph=graph,
                pos=pos,
                state=states[-1],
                source_node=source_node,
                target_node=target_node,
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(
                f"Imagen final guardada en: "
                f"{final_image_path}"
            )

        def init():
            self._dibujar_estado_floyd_warshall(
                graph_ax=graph_ax,
                info_ax=info_ax,
                matrix_ax=matrix_ax,
                graph=graph,
                pos=pos,
                state=states[0],
                source_node=source_node,
                target_node=target_node,
            )
            return []

        def update(frame_index):
            self._dibujar_estado_floyd_warshall(
                graph_ax=graph_ax,
                info_ax=info_ax,
                matrix_ax=matrix_ax,
                graph=graph,
                pos=pos,
                state=states[frame_index],
                source_node=source_node,
                target_node=target_node,
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()

        return self.animation

    # ------------------------------------------------------------------
    # Elementos específicos de Prim y Kruskal
    # ------------------------------------------------------------------

    def _preparar_figura_mst(self, title):
        """
        Reutiliza la distribución visual de Dijkstra y A*.

        Distribución:
        - izquierda: leyenda y tarjetas del algoritmo activo;
        - derecha superior: grafo ponderado;
        - derecha inferior: cola de Prim o lista de Kruskal.
        """

        return self._preparar_figura_dijkstra(title)

    def _dibujar_leyenda_mst(self, ax, algorithm):
        """
        Dibuja una leyenda compacta para Prim, Kruskal y la comparación.
        """

        if algorithm == "prim":
            elementos = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#D9D9D9",
                    markeredgecolor="#666666",
                    markersize=8,
                    label="Fuera del árbol",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#F6C85F",
                    markeredgecolor="#8A6D1D",
                    markersize=8,
                    label="En la frontera",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#4C9ED9",
                    markeredgecolor="#1F4F73",
                    markersize=8,
                    label="Incluido",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#2E8B57",
                    linewidth=3,
                    label="Arista seleccionada",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#E45756",
                    linewidth=4,
                    label="Arista actual",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#F28E2B",
                    linewidth=3,
                    linestyle="dashed",
                    label="Entrada obsoleta",
                ),
            ]
        elif algorithm == "kruskal":
            elementos = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#D9D9D9",
                    markeredgecolor="#666666",
                    markersize=8,
                    label="Componente",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#2E8B57",
                    linewidth=3,
                    label="Arista seleccionada",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#E45756",
                    linewidth=4,
                    label="Arista actual",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#F28E2B",
                    linewidth=3,
                    linestyle="dashed",
                    label="Rechazada por ciclo",
                ),
            ]
        else:
            elementos = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#4C9ED9",
                    markeredgecolor="#1F4F73",
                    markersize=8,
                    label="Vértice conectado",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#2E8B57",
                    linewidth=4,
                    label="Árbol de expansión mínima",
                ),
            ]

        ax.legend(
            handles=elementos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            fontsize=7.0,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.8,
            handlelength=2.1,
            borderpad=0.55,
        )

    def _dibujar_tabla_prim(
        self,
        ax,
        nodes,
        keys,
        parents,
        included,
        frontier_nodes,
        current_node,
        total_cost,
    ):
        """
        Dibuja las tarjetas de Prim.

        Cada tarjeta muestra:
        - clave: peso de la mejor arista conocida hacia el árbol;
        - padre: extremo incluido que proporciona esa arista.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            "Prim · claves y padres",
            fontsize=11.3,
            fontweight="bold",
            ha="center",
            va="top",
        )

        ax.text(
            0.50,
            0.935,
            f"Coste acumulado: {self._formatear_distancia(total_cost)}",
            fontsize=8.7,
            ha="center",
            va="top",
            color="#444444",
        )

        number_of_columns = 2
        card_width = 0.405
        card_height = 0.078
        horizontal_gap = 0.055
        vertical_gap = 0.016

        total_width = (
            number_of_columns * card_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.665

        for index, node in enumerate(nodes):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (card_width + horizontal_gap)
            y = top_y - row * (card_height + vertical_gap)

            key = keys.get(node, float("inf"))
            parent = parents.get(node)

            if node in included:
                face_color = "#B7D7F0"
                edge_color = "#1F4F73"
            elif node in frontier_nodes:
                face_color = "#FBE5A6"
                edge_color = "#8A6D1D"
            else:
                face_color = "#E5E5E5"
                edge_color = "#777777"

            line_width = 1.5

            if node == current_node:
                edge_color = "#C62828"
                line_width = 3.0

            rectangle = Rectangle(
                (x, y),
                card_width,
                card_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            parent_text = "—" if parent is None else str(parent)

            ax.text(
                x + card_width * 0.11,
                y + card_height / 2,
                str(node),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + card_width * 0.31,
                y + card_height / 2,
                f"k={self._formatear_distancia(key)}",
                fontsize=7.3,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.62,
                y + card_height / 2,
                f"padre={parent_text}",
                fontsize=6.9,
                ha="left",
                va="center",
            )

        ax.text(
            0.50,
            0.055,
            (
                "Gris: fuera   ·   Amarillo: frontera   ·   "
                "Azul: incluido"
            ),
            fontsize=6.7,
            ha="center",
            va="center",
            color="#444444",
        )

        self._dibujar_leyenda_mst(ax, "prim")

    def _dibujar_cola_prim(
        self,
        ax,
        priority_queue,
    ):
        """
        Dibuja la cola visible de Prim debajo del grafo.

        Las entradas tienen la forma:
            (peso, origen, destino)
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.02,
            0.82,
            "Cola de prioridad de Prim",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.text(
            0.02,
            0.41,
            "Menor peso",
            fontsize=8.5,
            ha="left",
            va="center",
        )

        ax.text(
            0.98,
            0.41,
            "Peso mayor",
            fontsize=8.5,
            ha="right",
            va="center",
        )

        queue_sorted = sorted(priority_queue)

        if not queue_sorted:
            ax.text(
                0.50,
                0.41,
                "Cola vacía",
                fontsize=11.5,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.42",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.98,
                },
            )
            return

        max_cells = 10
        visible_queue = queue_sorted[:max_cells]

        initial_x = 0.11
        final_x = 0.89
        total_width = final_x - initial_x
        cell_width = min(
            0.078,
            total_width / max(len(visible_queue), 1),
        )
        gap = 0.009

        occupied_width = (
            len(visible_queue) * cell_width
            + max(0, len(visible_queue) - 1) * gap
        )

        current_x = 0.50 - occupied_width / 2

        for index, (weight, origin, destination) in enumerate(visible_queue):
            is_minimum = index == 0

            rectangle = Rectangle(
                (current_x, 0.20),
                cell_width,
                0.43,
                facecolor="#E45756" if is_minimum else "#F6C85F",
                edgecolor="#7A1D1D" if is_minimum else "#8A6D1D",
                linewidth=2.0 if is_minimum else 1.5,
            )
            ax.add_patch(rectangle)

            ax.text(
                current_x + cell_width / 2,
                0.49,
                f"{origin}—{destination}",
                fontsize=7.6,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                current_x + cell_width / 2,
                0.33,
                f"w={self._formatear_distancia(weight)}",
                fontsize=7.1,
                ha="center",
                va="center",
            )

            if is_minimum:
                ax.text(
                    current_x + cell_width / 2,
                    0.12,
                    "siguiente",
                    fontsize=6.5,
                    ha="center",
                    va="top",
                )

            current_x += cell_width + gap

        if len(queue_sorted) > max_cells:
            ax.text(
                0.91,
                0.41,
                f"+{len(queue_sorted) - max_cells}",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="center",
            )

    def _dibujar_tabla_kruskal(
        self,
        ax,
        nodes,
        component_map,
        selected_edges,
        current_edge,
        total_cost,
    ):
        """
        Dibuja una tarjeta por vértice con su componente actual.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            "Kruskal · componentes",
            fontsize=11.3,
            fontweight="bold",
            ha="center",
            va="top",
        )

        ax.text(
            0.50,
            0.935,
            (
                f"Aristas: {len(selected_edges)}"
                f"  ·  coste: {self._formatear_distancia(total_cost)}"
            ),
            fontsize=8.7,
            ha="center",
            va="top",
            color="#444444",
        )

        number_of_columns = 2
        card_width = 0.405
        card_height = 0.078
        horizontal_gap = 0.055
        vertical_gap = 0.016

        total_width = (
            number_of_columns * card_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.665

        current_nodes = set(current_edge or [])

        for index, node in enumerate(nodes):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (card_width + horizontal_gap)
            y = top_y - row * (card_height + vertical_gap)

            component = component_map.get(node, node)

            face_color = "#E5E5E5"
            edge_color = "#777777"
            line_width = 1.5

            if node in current_nodes:
                face_color = "#FBE5A6"
                edge_color = "#C62828"
                line_width = 3.0

            rectangle = Rectangle(
                (x, y),
                card_width,
                card_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            ax.text(
                x + card_width * 0.15,
                y + card_height / 2,
                str(node),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + card_width * 0.43,
                y + card_height / 2,
                f"comp. {component}",
                fontsize=7.2,
                ha="left",
                va="center",
            )

        ax.text(
            0.50,
            0.055,
            "Dos extremos en la misma componente formarían un ciclo",
            fontsize=6.7,
            ha="center",
            va="center",
            color="#444444",
        )

        self._dibujar_leyenda_mst(ax, "kruskal")

    def _dibujar_lista_kruskal(
        self,
        ax,
        sorted_edges,
        processed_count,
        active_edge_index,
        action,
    ):
        """
        Dibuja la lista ordenada de aristas de Kruskal.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.02,
            0.87,
            "Aristas ordenadas de Kruskal",
            fontsize=11.8,
            fontweight="bold",
            ha="left",
            va="center",
        )

        number_of_columns = 10
        number_of_rows = (
            len(sorted_edges) + number_of_columns - 1
        ) // number_of_columns

        cell_width = 0.075
        cell_height = 0.25
        horizontal_gap = 0.009
        vertical_gap = 0.075

        total_width = (
            number_of_columns * cell_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.49

        for index, (weight, origin, destination) in enumerate(sorted_edges):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (cell_width + horizontal_gap)
            y = top_y - row * (cell_height + vertical_gap)

            is_current = index == active_edge_index
            is_processed = index < processed_count

            if is_current:
                if action == "accepted":
                    face_color = "#B7E4C7"
                    edge_color = "#2E8B57"
                else:
                    face_color = "#F6B4B4"
                    edge_color = "#C62828"
                line_width = 2.2
            elif is_processed:
                face_color = "#B7D7F0"
                edge_color = "#1F4F73"
                line_width = 1.4
            else:
                face_color = "#FBE5A6"
                edge_color = "#8A6D1D"
                line_width = 1.4

            rectangle = Rectangle(
                (x, y),
                cell_width,
                cell_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            ax.text(
                x + cell_width / 2,
                y + cell_height * 0.66,
                f"{origin}—{destination}",
                fontsize=6.8,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + cell_width / 2,
                y + cell_height * 0.28,
                f"w={self._formatear_distancia(weight)}",
                fontsize=6.5,
                ha="center",
                va="center",
            )

        if number_of_rows == 1:
            ax.set_ylim(0.10, 1.0)

    def _dibujar_grafo_mst_comun(
        self,
        graph_ax,
        graph,
        pos,
        selected_edges,
        rejected_edges,
        active_edge,
        action,
        node_colors,
        node_edge_colors,
        node_sizes,
        labels_above=None,
    ):
        """
        Dibuja el grafo ponderado común a Prim y Kruskal.
        """

        graph_ax.clear()
        graph_ax.axis("off")

        limits = self._calcular_limites(
            pos,
            margin_x=1.2,
            margin_y=1.0,
        )

        graph_ax.set_xlim(limits[0], limits[1])
        graph_ax.set_ylim(limits[2], limits[3])
        graph_ax.set_aspect("equal", adjustable="box")

        selected_normalized = {
            self._normalizar_arista(u, v)
            for u, v in selected_edges
        }
        rejected_normalized = {
            self._normalizar_arista(u, v)
            for u, v in rejected_edges
        }

        active_normalized = None
        if active_edge is not None:
            active_normalized = self._normalizar_arista(*active_edge)

        for u, v, data in graph.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            edge_key = self._normalizar_arista(u, v)

            if edge_key == active_normalized:
                if action in {"rejected", "stale", "no_improvement"}:
                    color = "#F28E2B"
                    line_style = "dashed"
                else:
                    color = "#E45756"
                    line_style = "solid"
                line_width = 4.2
                zorder = 20
            elif edge_key in selected_normalized:
                color = "#2E8B57"
                line_style = "solid"
                line_width = 3.4
                zorder = 17
            elif edge_key in rejected_normalized:
                color = "#C8A27A"
                line_style = "dashed"
                line_width = 1.8
                zorder = 12
            else:
                color = "#B8B8B8"
                line_style = "solid"
                line_width = 1.6
                zorder = 10

            graph_ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                zorder=zorder,
            )

            self._dibujar_peso_arista(
                ax=graph_ax,
                pos=pos,
                origen=u,
                destino=v,
                peso=data.get("weight", 1),
            )

        for node in graph.nodes():
            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[node],
                node_size=node_sizes.get(node, 780),
                node_color=node_colors.get(node, "#D9D9D9"),
                edgecolors=node_edge_colors.get(node, "#666666"),
                linewidths=2.2 if node_sizes.get(node, 780) > 850 else 1.5,
                ax=graph_ax,
            )
            collection.set_zorder(25)

        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

            if labels_above and node in labels_above:
                graph_ax.text(
                    x,
                    y + 0.39,
                    labels_above[node],
                    fontsize=7.1,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    color="#222222",
                    zorder=40,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "fc": "white",
                        "ec": "#555555",
                        "alpha": 0.97,
                    },
                )

    def _dibujar_estado_prim(
        self,
        graph_ax,
        info_ax,
        structure_ax,
        graph,
        pos,
        state,
        start_node,
    ):
        """
        Dibuja un estado completo de Prim.
        """

        included = set(state.get("included", set()))
        selected_edges = list(state.get("selected_edges", []))
        rejected_edges = list(state.get("rejected_edges", []))
        keys = dict(state.get("keys", {}))
        parents = dict(state.get("parents", {}))
        priority_queue = list(state.get("priority_queue", []))
        active_edge = state.get("active_edge")
        action = state.get("action")
        current_node = state.get("current_node")

        frontier_nodes = {
            destination
            for _, _, destination in priority_queue
        }

        node_colors = {}
        node_edge_colors = {}
        node_sizes = {}

        for node in graph.nodes():
            if node == current_node:
                node_colors[node] = "#E45756"
                node_edge_colors[node] = "#7A1D1D"
                node_sizes[node] = 930
            elif node in included:
                node_colors[node] = "#4C9ED9"
                node_edge_colors[node] = "#1F4F73"
                node_sizes[node] = 790
            elif node in frontier_nodes:
                node_colors[node] = "#F6C85F"
                node_edge_colors[node] = "#8A6D1D"
                node_sizes[node] = 790
            else:
                node_colors[node] = "#D9D9D9"
                node_edge_colors[node] = "#666666"
                node_sizes[node] = 760

        labels_above = {
            node: f"k={self._formatear_distancia(keys.get(node, float('inf')))}"
            for node in graph.nodes()
        }

        self._dibujar_grafo_mst_comun(
            graph_ax=graph_ax,
            graph=graph,
            pos=pos,
            selected_edges=selected_edges,
            rejected_edges=rejected_edges,
            active_edge=active_edge,
            action=action,
            node_colors=node_colors,
            node_edge_colors=node_edge_colors,
            node_sizes=node_sizes,
            labels_above=labels_above,
        )

        start_x, start_y = pos[start_node]
        graph_ax.text(
            start_x,
            start_y - 0.43,
            "inicio de Prim",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
            zorder=40,
        )

        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=9.2,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        graph_ax.text(
            0.99,
            0.985,
            (
                f"PRIM  ·  incluidos: {len(included)}/{graph.number_of_nodes()}"
                f"  ·  aristas: {len(selected_edges)}"
                f"  ·  coste: {self._formatear_distancia(state.get('total_cost', 0))}"
            ),
            transform=graph_ax.transAxes,
            fontsize=8.6,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        self._dibujar_tabla_prim(
            ax=info_ax,
            nodes=sorted(graph.nodes()),
            keys=keys,
            parents=parents,
            included=included,
            frontier_nodes=frontier_nodes,
            current_node=current_node,
            total_cost=state.get("total_cost", 0),
        )

        self._dibujar_cola_prim(
            ax=structure_ax,
            priority_queue=priority_queue,
        )

    def _dibujar_estado_kruskal(
        self,
        graph_ax,
        info_ax,
        structure_ax,
        graph,
        pos,
        state,
    ):
        """
        Dibuja un estado completo de Kruskal.
        """

        selected_edges = list(state.get("selected_edges", []))
        rejected_edges = list(state.get("rejected_edges", []))
        component_map = dict(state.get("component_map", {}))
        sorted_edges = list(state.get("sorted_edges", []))
        active_edge = state.get("active_edge")
        action = state.get("action")

        # Paleta estable basada en la etiqueta canónica de la componente.
        palette = [
            "#B7D7F0",
            "#FBE5A6",
            "#D8C4E8",
            "#B7E4C7",
            "#F7C6C7",
            "#CDE7E8",
            "#E7D6B8",
            "#D6E4B7",
            "#D8D8F0",
            "#F4D2A7",
        ]

        canonical_components = sorted(set(component_map.values()))
        component_color = {
            component: palette[index % len(palette)]
            for index, component in enumerate(canonical_components)
        }

        node_colors = {}
        node_edge_colors = {}
        node_sizes = {}

        active_nodes = set(active_edge or [])

        for node in graph.nodes():
            component = component_map.get(node, node)
            node_colors[node] = component_color.get(
                component,
                "#D9D9D9",
            )
            node_edge_colors[node] = (
                "#C62828"
                if node in active_nodes
                else "#666666"
            )
            node_sizes[node] = 900 if node in active_nodes else 780

        labels_above = {
            node: f"comp. {component_map.get(node, node)}"
            for node in graph.nodes()
        }

        self._dibujar_grafo_mst_comun(
            graph_ax=graph_ax,
            graph=graph,
            pos=pos,
            selected_edges=selected_edges,
            rejected_edges=rejected_edges,
            active_edge=active_edge,
            action=action,
            node_colors=node_colors,
            node_edge_colors=node_edge_colors,
            node_sizes=node_sizes,
            labels_above=labels_above,
        )

        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=9.2,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        graph_ax.text(
            0.99,
            0.985,
            (
                f"KRUSKAL  ·  componentes: "
                f"{len(set(component_map.values()))}"
                f"  ·  aristas: {len(selected_edges)}"
                f"  ·  coste: {self._formatear_distancia(state.get('total_cost', 0))}"
            ),
            transform=graph_ax.transAxes,
            fontsize=8.6,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        self._dibujar_tabla_kruskal(
            ax=info_ax,
            nodes=sorted(graph.nodes()),
            component_map=component_map,
            selected_edges=selected_edges,
            current_edge=active_edge,
            total_cost=state.get("total_cost", 0),
        )

        self._dibujar_lista_kruskal(
            ax=structure_ax,
            sorted_edges=sorted_edges,
            processed_count=state.get("processed_count", 0),
            active_edge_index=state.get("active_edge_index"),
            action=action,
        )

    def _dibujar_estado_comparacion_mst(
        self,
        graph_ax,
        info_ax,
        structure_ax,
        graph,
        pos,
        state,
    ):
        """
        Dibuja el resumen final de Prim y Kruskal.
        """

        prim_edges = list(state.get("prim_edges", []))
        kruskal_edges = list(state.get("kruskal_edges", []))
        common_edges = list(state.get("common_edges", []))

        node_colors = {
            node: "#4C9ED9"
            for node in graph.nodes()
        }
        node_edge_colors = {
            node: "#1F4F73"
            for node in graph.nodes()
        }
        node_sizes = {
            node: 790
            for node in graph.nodes()
        }

        self._dibujar_grafo_mst_comun(
            graph_ax=graph_ax,
            graph=graph,
            pos=pos,
            selected_edges=common_edges,
            rejected_edges=[],
            active_edge=None,
            action="finished",
            node_colors=node_colors,
            node_edge_colors=node_edge_colors,
            node_sizes=node_sizes,
            labels_above=None,
        )

        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=9.3,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        graph_ax.text(
            0.99,
            0.985,
            "COMPARACIÓN FINAL",
            transform=graph_ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        info_ax.clear()
        info_ax.axis("off")
        info_ax.set_xlim(0, 1)
        info_ax.set_ylim(0, 1)

        info_ax.text(
            0.50,
            0.985,
            "Comparación final",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="top",
        )

        cards = [
            (
                "Prim",
                len(prim_edges),
                state.get("prim_cost", 0),
                "#B7D7F0",
            ),
            (
                "Kruskal",
                len(kruskal_edges),
                state.get("kruskal_cost", 0),
                "#B7E4C7",
            ),
        ]

        y_positions = [0.70, 0.53]

        for (name, edge_count, cost, color), y in zip(cards, y_positions):
            rectangle = Rectangle(
                (0.12, y),
                0.76,
                0.12,
                facecolor=color,
                edgecolor="#555555",
                linewidth=1.5,
            )
            info_ax.add_patch(rectangle)

            info_ax.text(
                0.20,
                y + 0.06,
                name,
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="center",
            )

            info_ax.text(
                0.52,
                y + 0.06,
                f"{edge_count} aristas",
                fontsize=8,
                ha="center",
                va="center",
            )

            info_ax.text(
                0.80,
                y + 0.06,
                f"coste {self._formatear_distancia(cost)}",
                fontsize=8,
                ha="right",
                va="center",
            )

        same_cost = state.get("same_cost", False)
        same_edges = state.get("same_edges", False)

        info_ax.text(
            0.50,
            0.39,
            (
                f"Mismo coste: {'sí' if same_cost else 'no'}\n"
                f"Mismas aristas: {'sí' if same_edges else 'no'}\n"
                f"Vértices: {graph.number_of_nodes()}\n"
                f"Aristas del MST: {len(common_edges)}"
            ),
            fontsize=9,
            ha="center",
            va="center",
            linespacing=1.5,
            bbox={
                "boxstyle": "round,pad=0.45",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.98,
            },
        )

        self._dibujar_leyenda_mst(info_ax, "comparison")

        structure_ax.clear()
        structure_ax.axis("off")
        structure_ax.set_xlim(0, 1)
        structure_ax.set_ylim(0, 1)

        structure_ax.text(
            0.02,
            0.82,
            "Aristas del árbol de expansión mínima",
            fontsize=11.8,
            fontweight="bold",
            ha="left",
            va="center",
        )

        weighted_edges = list(state.get("weighted_common_edges", []))
        max_cells = 10
        visible_edges = weighted_edges[:max_cells]

        if visible_edges:
            initial_x = 0.10
            final_x = 0.90
            total_width = final_x - initial_x
            cell_width = min(
                0.078,
                total_width / max(len(visible_edges), 1),
            )
            gap = 0.009

            occupied_width = (
                len(visible_edges) * cell_width
                + max(0, len(visible_edges) - 1) * gap
            )
            current_x = 0.50 - occupied_width / 2

            for weight, origin, destination in visible_edges:
                rectangle = Rectangle(
                    (current_x, 0.22),
                    cell_width,
                    0.40,
                    facecolor="#B7E4C7",
                    edgecolor="#2E8B57",
                    linewidth=1.7,
                )
                structure_ax.add_patch(rectangle)

                structure_ax.text(
                    current_x + cell_width / 2,
                    0.47,
                    f"{origin}—{destination}",
                    fontsize=7.2,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

                structure_ax.text(
                    current_x + cell_width / 2,
                    0.31,
                    f"w={self._formatear_distancia(weight)}",
                    fontsize=6.8,
                    ha="center",
                    va="center",
                )

                current_x += cell_width + gap

    def _dibujar_estado_mst(
        self,
        graph_ax,
        info_ax,
        structure_ax,
        graph,
        pos,
        state,
        start_node,
    ):
        """
        Despacha el estado al dibujo de Prim, Kruskal o comparación.
        """

        algorithm = state.get("algorithm")

        if algorithm == "prim":
            self._dibujar_estado_prim(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=state,
                start_node=start_node,
            )
        elif algorithm == "kruskal":
            self._dibujar_estado_kruskal(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=state,
            )
        elif algorithm == "comparison":
            self._dibujar_estado_comparacion_mst(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=state,
            )
        else:
            raise ValueError(
                f"Algoritmo MST desconocido: {algorithm!r}"
            )

    def animate_mst_comparison(
        self,
        graph,
        pos,
        states,
        start_node,
        title="Árbol de expansión mínima: Prim y Kruskal",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima primero Prim, después Kruskal y termina con una comparación.

        La imagen final muestra:
        - el árbol de expansión mínima;
        - el coste obtenido por ambos algoritmos;
        - el número de aristas;
        - si coinciden el coste y la estructura.
        """

        if not states:
            raise ValueError(
                "La lista de estados de Prim/Kruskal no puede estar vacía."
            )

        (
            fig,
            graph_ax,
            info_ax,
            structure_ax,
        ) = self._preparar_figura_mst(title)

        if final_image_path is not None:
            self._dibujar_estado_mst(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[-1],
                start_node=start_node,
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(
                f"Imagen final guardada en: "
                f"{final_image_path}"
            )

        def init():
            self._dibujar_estado_mst(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[0],
                start_node=start_node,
            )
            return []

        def update(frame_index):
            self._dibujar_estado_mst(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[frame_index],
                start_node=start_node,
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()

        return self.animation

    # ------------------------------------------------------------------
    # Elementos específicos de Union-Find y componentes conectadas
    # ------------------------------------------------------------------

    def _preparar_figura_union_find(self, title):
        """
        Reutiliza la distribución visual de Dijkstra, A* y Prim.

        Distribución:
        - izquierda: padres, raíces, rangos y tamaños;
        - derecha superior: grafo y componentes actuales;
        - derecha inferior: aristas procesadas en orden.
        """

        return self._preparar_figura_dijkstra(title)

    @staticmethod
    def _paleta_componentes_union_find():
        """
        Devuelve una paleta suficientemente amplia para el ejemplo.
        """

        return [
            "#B7D7F0",
            "#FBE5A6",
            "#D8C4E8",
            "#B7E4C7",
            "#F7C6C7",
            "#CDE7E8",
            "#E7D6B8",
            "#D6E4B7",
            "#D8D8F0",
            "#F4D2A7",
            "#C8D6E5",
            "#D5E8D4",
            "#FFE0B2",
            "#E1BEE7",
        ]

    def _dibujar_leyenda_union_find(self, ax):
        """
        Dibuja la leyenda de Union-Find en el panel izquierdo.
        """

        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#B7D7F0",
                markeredgecolor="#666666",
                markersize=8,
                label="Componente actual",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D",
                markersize=8,
                label="Primer extremo",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F28E2B",
                markeredgecolor="#8A4B08",
                markersize=8,
                label="Segundo extremo",
            ),
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=3,
                label="Arista aceptada",
            ),
            Line2D(
                [0],
                [0],
                color="#E45756",
                linewidth=4,
                label="Arista examinada",
            ),
            Line2D(
                [0],
                [0],
                color="#F28E2B",
                linewidth=3,
                linestyle="dashed",
                label="Rechazada por ciclo",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            fontsize=6.9,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.7,
            handlelength=2.0,
            borderpad=0.50,
        )

    def _dibujar_tabla_union_find(
        self,
        ax,
        nodes,
        parents,
        ranks,
        sizes,
        roots,
        component_map,
        components,
        current_nodes,
    ):
        """
        Dibuja una tarjeta por vértice.

        Cada tarjeta contiene:
        - padre inmediato;
        - raíz de la componente;
        - rango si el vértice es raíz;
        - tamaño si el vértice es raíz.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            "Union-Find · padres, raíces y rangos",
            fontsize=10.7,
            fontweight="bold",
            ha="center",
            va="top",
        )

        ax.text(
            0.50,
            0.945,
            f"Componentes actuales: {len(components)}",
            fontsize=8.5,
            ha="center",
            va="top",
            color="#444444",
        )

        palette = self._paleta_componentes_union_find()
        canonical_components = sorted(set(component_map.values()))
        component_color = {
            component: palette[index % len(palette)]
            for index, component in enumerate(canonical_components)
        }

        number_of_columns = 2
        card_width = 0.405
        card_height = 0.074
        horizontal_gap = 0.055
        vertical_gap = 0.012

        total_width = (
            number_of_columns * card_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.685

        current_nodes = set(current_nodes or [])

        for index, node in enumerate(nodes):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (card_width + horizontal_gap)
            y = top_y - row * (card_height + vertical_gap)

            parent = parents.get(node, node)
            root = roots.get(node, node)
            component = component_map.get(node, node)
            is_root = parent == node

            face_color = component_color.get(component, "#E5E5E5")
            edge_color = "#666666"
            line_width = 1.4

            if node in current_nodes:
                edge_color = "#C62828"
                line_width = 3.0

            rectangle = Rectangle(
                (x, y),
                card_width,
                card_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            rank_text = ranks.get(node, 0) if is_root else "—"
            size_text = sizes.get(node, 1) if is_root else "—"

            ax.text(
                x + card_width * 0.10,
                y + card_height * 0.66,
                str(node),
                fontsize=8.8,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + card_width * 0.27,
                y + card_height * 0.66,
                f"p={parent}",
                fontsize=6.9,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.62,
                y + card_height * 0.66,
                f"raíz={root}",
                fontsize=6.8,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.27,
                y + card_height * 0.27,
                f"rango={rank_text}",
                fontsize=6.5,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.62,
                y + card_height * 0.27,
                f"tam={size_text}",
                fontsize=6.5,
                ha="left",
                va="center",
            )

        ax.text(
            0.50,
            0.050,
            (
                "p: padre inmediato   ·   "
                "raíz: representante   ·   "
                "tam: tamaño de la raíz"
            ),
            fontsize=6.2,
            ha="center",
            va="center",
            color="#444444",
        )

        self._dibujar_leyenda_union_find(ax)

    def _dibujar_lista_aristas_union_find(
        self,
        ax,
        edge_order,
        edge_statuses,
        active_edge_index,
        action,
        component_count,
        accepted_count,
        rejected_count,
    ):
        """
        Dibuja la secuencia de aristas procesadas.

        Estados:
        - amarillo: pendiente;
        - rojo: arista actual;
        - verde: unión aceptada;
        - naranja: rechazada porque formaría un ciclo.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.02,
            0.88,
            "Aristas procesadas por Union-Find",
            fontsize=11.7,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.text(
            0.98,
            0.88,
            (
                f"componentes: {component_count}"
                f"  ·  uniones: {accepted_count}"
                f"  ·  ciclos: {rejected_count}"
            ),
            fontsize=8.2,
            ha="right",
            va="center",
            color="#444444",
        )

        number_of_columns = 8
        number_of_rows = (
            len(edge_order) + number_of_columns - 1
        ) // number_of_columns

        cell_width = 0.095
        cell_height = 0.25
        horizontal_gap = 0.012
        vertical_gap = 0.075

        total_width = (
            number_of_columns * cell_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.49

        for index, (origin, destination) in enumerate(edge_order):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (cell_width + horizontal_gap)
            y = top_y - row * (cell_height + vertical_gap)

            status = edge_statuses.get(index, "pending")
            is_current = index == active_edge_index

            if is_current:
                if action == "rejected":
                    face_color = "#F6B4B4"
                    edge_color = "#C62828"
                elif action == "accepted":
                    face_color = "#B7E4C7"
                    edge_color = "#2E8B57"
                else:
                    face_color = "#E45756"
                    edge_color = "#7A1D1D"
                line_width = 2.2
            elif status == "accepted":
                face_color = "#B7E4C7"
                edge_color = "#2E8B57"
                line_width = 1.5
            elif status == "rejected":
                face_color = "#F8D7B5"
                edge_color = "#F28E2B"
                line_width = 1.5
            else:
                face_color = "#FBE5A6"
                edge_color = "#8A6D1D"
                line_width = 1.4

            rectangle = Rectangle(
                (x, y),
                cell_width,
                cell_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            ax.text(
                x + cell_width / 2,
                y + cell_height * 0.66,
                f"{origin}—{destination}",
                fontsize=7.0,
                fontweight="bold",
                ha="center",
                va="center",
            )

            if status == "accepted":
                status_text = "✓ unión"
            elif status == "rejected":
                status_text = "× ciclo"
            elif is_current:
                status_text = "find"
            else:
                status_text = "pendiente"

            ax.text(
                x + cell_width / 2,
                y + cell_height * 0.28,
                status_text,
                fontsize=6.4,
                ha="center",
                va="center",
            )

        if number_of_rows == 1:
            ax.set_ylim(0.10, 1.0)

    def _dibujar_estado_union_find(
        self,
        graph_ax,
        info_ax,
        structure_ax,
        graph,
        pos,
        state,
    ):
        """
        Dibuja un estado completo de Union-Find.
        """

        graph_ax.clear()
        graph_ax.axis("off")

        limits = self._calcular_limites(
            pos,
            margin_x=1.2,
            margin_y=1.0,
        )

        graph_ax.set_xlim(limits[0], limits[1])
        graph_ax.set_ylim(limits[2], limits[3])
        graph_ax.set_aspect("equal", adjustable="box")

        parents = dict(state.get("parents", {}))
        ranks = dict(state.get("ranks", {}))
        sizes = dict(state.get("sizes", {}))
        roots = dict(state.get("roots", {}))
        component_map = dict(state.get("component_map", {}))
        components = list(state.get("components", []))

        edge_order = list(state.get("edge_order", []))
        edge_statuses = dict(state.get("edge_statuses", {}))
        accepted_edges = list(state.get("accepted_edges", []))
        rejected_edges = list(state.get("rejected_edges", []))
        active_edge = state.get("active_edge")
        active_edge_index = state.get("active_edge_index")
        action = state.get("action")
        phase = state.get("phase", "processing")

        accepted_normalized = {
            self._normalizar_arista(u, v)
            for u, v in accepted_edges
        }
        rejected_normalized = {
            self._normalizar_arista(u, v)
            for u, v in rejected_edges
        }

        active_normalized = None
        current_nodes = set()

        if active_edge is not None:
            active_normalized = self._normalizar_arista(*active_edge)
            current_nodes = set(active_edge)

        # 1. Aristas.
        for origin, destination in graph.edges():
            x1, y1 = pos[origin]
            x2, y2 = pos[destination]
            edge_key = self._normalizar_arista(origin, destination)

            if edge_key == active_normalized:
                if action == "rejected":
                    color = "#F28E2B"
                    line_style = "dashed"
                else:
                    color = "#E45756"
                    line_style = "solid"
                line_width = 4.2
                zorder = 20
            elif edge_key in accepted_normalized:
                color = "#2E8B57"
                line_style = "solid"
                line_width = 3.2
                zorder = 17
            elif edge_key in rejected_normalized:
                color = "#F28E2B"
                line_style = "dashed"
                line_width = 2.2
                zorder = 14
            else:
                color = "#B8B8B8"
                line_style = "solid"
                line_width = 1.6
                zorder = 10

            graph_ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                zorder=zorder,
            )

        # 2. Colores de las componentes.
        palette = self._paleta_componentes_union_find()
        canonical_components = sorted(set(component_map.values()))
        component_color = {
            component: palette[index % len(palette)]
            for index, component in enumerate(canonical_components)
        }

        component_sizes = {
            component[0]: len(component[1])
            for component in components
        }

        first_endpoint = active_edge[0] if active_edge else None
        second_endpoint = active_edge[1] if active_edge else None

        for node in graph.nodes():
            component = component_map.get(node, node)
            color = component_color.get(component, "#D9D9D9")
            edge_color = "#666666"
            node_size = 790

            if (
                phase == "finished"
                and component_sizes.get(component, 1) == 1
            ):
                color = "#D9D9D9"

            if node == first_endpoint:
                color = "#E45756"
                edge_color = "#7A1D1D"
                node_size = 930
            elif node == second_endpoint:
                color = "#F28E2B"
                edge_color = "#8A4B08"
                node_size = 900

            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[node],
                node_size=node_size,
                node_color=color,
                edgecolors=edge_color,
                linewidths=2.4 if node in current_nodes else 1.5,
                ax=graph_ax,
            )
            collection.set_zorder(25)

        # 3. Etiquetas.
        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

            graph_ax.text(
                x,
                y + 0.39,
                f"raíz={roots.get(node, node)}",
                fontsize=7.0,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#222222",
                zorder=40,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "#555555",
                    "alpha": 0.97,
                },
            )

        # 4. Operación find/union.
        find_path_u = list(state.get("find_path_u", []))
        find_path_v = list(state.get("find_path_v", []))
        root_u = state.get("root_u")
        root_v = state.get("root_v")

        if active_edge is not None and find_path_u and find_path_v:
            origin, destination = active_edge

            operation_text = (
                f"find({origin}): {' → '.join(map(str, find_path_u))}"
                f" = {root_u}"
                f"   |   "
                f"find({destination}): {' → '.join(map(str, find_path_v))}"
                f" = {root_v}"
            )

            graph_ax.text(
                0.50,
                0.965,
                operation_text,
                transform=graph_ax.transAxes,
                fontsize=7.9,
                ha="center",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
                zorder=50,
            )

        # 5. Mensajes y resumen.
        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=9.0,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        graph_ax.text(
            0.99,
            0.985,
            (
                f"Componentes: {len(components)}"
                f"  ·  uniones: {len(accepted_edges)}"
                f"  ·  ciclos: {len(rejected_edges)}"
            ),
            transform=graph_ax.transAxes,
            fontsize=8.7,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        self._dibujar_tabla_union_find(
            ax=info_ax,
            nodes=sorted(graph.nodes()),
            parents=parents,
            ranks=ranks,
            sizes=sizes,
            roots=roots,
            component_map=component_map,
            components=components,
            current_nodes=current_nodes,
        )

        self._dibujar_lista_aristas_union_find(
            ax=structure_ax,
            edge_order=edge_order,
            edge_statuses=edge_statuses,
            active_edge_index=active_edge_index,
            action=action,
            component_count=len(components),
            accepted_count=len(accepted_edges),
            rejected_count=len(rejected_edges),
        )

    def animate_union_find_components(
        self,
        graph,
        pos,
        states,
        title="Union-Find y componentes conectadas",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima el procesamiento incremental de las aristas con Union-Find.

        La imagen final muestra:
        - componentes conectadas;
        - padres, raíces, rangos y tamaños;
        - bosque de aristas aceptadas;
        - aristas rechazadas porque formarían ciclos;
        - vértices aislados.
        """

        if not states:
            raise ValueError(
                "La lista de estados de Union-Find no puede estar vacía."
            )

        (
            fig,
            graph_ax,
            info_ax,
            structure_ax,
        ) = self._preparar_figura_union_find(title)

        if final_image_path is not None:
            self._dibujar_estado_union_find(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[-1],
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(
                f"Imagen final guardada en: "
                f"{final_image_path}"
            )

        def init():
            self._dibujar_estado_union_find(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_union_find(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[frame_index],
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()

        return self.animation

    # ------------------------------------------------------------------
    # Elementos específicos de DAG y ordenamiento topológico
    # ------------------------------------------------------------------

    def _preparar_figura_orden_topologico(self, title):
        """
        Mantiene la estructura visual de los algoritmos anteriores.

        Distribución:
        - izquierda: grado de entrada, nivel y estado de cada tarea;
        - derecha superior: DAG dirigido;
        - derecha inferior: cola de disponibles y orden construido.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.70, 4.30],
            height_ratios=[4.65, 1.65],
            wspace=0.08,
            hspace=0.09,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        graph_ax = fig.add_subplot(grid[0, 1])
        structure_ax = fig.add_subplot(grid[1, 1])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.93,
            bottom=0.045,
        )

        return fig, graph_ax, info_ax, structure_ax

    def _dibujar_leyenda_orden_topologico(self, ax):
        """
        Dibuja la leyenda del algoritmo de Kahn.
        """

        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="#666666",
                markersize=8,
                label="Tarea pendiente",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F6C85F",
                markeredgecolor="#8A6D1D",
                markersize=8,
                label="Disponible",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D",
                markersize=8,
                label="Tarea actual",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4C9ED9",
                markeredgecolor="#1F4F73",
                markersize=8,
                label="Procesada",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D8C4E8",
                markeredgecolor="#5A316B",
                markersize=8,
                label="Bloqueada por ciclo",
            ),
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=3,
                label="Dependencia satisfecha",
            ),
            Line2D(
                [0],
                [0],
                color="#E45756",
                linewidth=4,
                label="Dependencia actual",
            ),
            Line2D(
                [0],
                [0],
                color="#8E5EA2",
                linewidth=4,
                label="Ciclo dirigido",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            fontsize=6.6,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.65,
            handlelength=1.9,
            borderpad=0.48,
        )

    @staticmethod
    def _abreviar_tarea_orden_topologico(task, maximum_length=18):
        """
        Acorta una descripción para que quepa en una tarjeta.
        """

        task = str(task)

        if len(task) <= maximum_length:
            return task

        return task[: maximum_length - 1] + "…"

    def _dibujar_tabla_orden_topologico(
        self,
        ax,
        nodes,
        task_names,
        initial_in_degrees,
        in_degrees,
        levels,
        available_nodes,
        processed_nodes,
        blocked_nodes,
        current_node,
        phase,
        is_unique,
    ):
        """
        Dibuja una tarjeta por tarea.

        Cada tarjeta contiene:
        - identificador y descripción abreviada;
        - grado de entrada actual e inicial;
        - nivel topológico provisional o final.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            "Kahn · grados de entrada y niveles",
            fontsize=10.7,
            fontweight="bold",
            ha="center",
            va="top",
        )

        unique_text = (
            "sí"
            if is_unique
            else "no"
        )

        ax.text(
            0.50,
            0.948,
            f"Orden único hasta ahora: {unique_text}",
            fontsize=8.1,
            ha="center",
            va="top",
            color="#444444",
        )

        available_nodes = set(available_nodes)
        processed_nodes = set(processed_nodes)
        blocked_nodes = set(blocked_nodes)

        number_of_columns = 2
        card_width = 0.415
        card_height = 0.086
        horizontal_gap = 0.045
        vertical_gap = 0.012

        total_width = (
            number_of_columns * card_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.675

        for index, node in enumerate(nodes):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (card_width + horizontal_gap)
            y = top_y - row * (card_height + vertical_gap)

            if node in blocked_nodes:
                face_color = "#D8C4E8"
                edge_color = "#5A316B"
            elif node in processed_nodes:
                face_color = "#B7D7F0"
                edge_color = "#1F4F73"
            elif node in available_nodes:
                face_color = "#FBE5A6"
                edge_color = "#8A6D1D"
            else:
                face_color = "#E5E5E5"
                edge_color = "#777777"

            line_width = 1.45

            if node == current_node:
                face_color = "#F6B4B4"
                edge_color = "#C62828"
                line_width = 3.0

            rectangle = Rectangle(
                (x, y),
                card_width,
                card_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            task_text = self._abreviar_tarea_orden_topologico(
                task_names.get(node, node)
            )

            level = levels.get(node)
            level_text = "—" if level is None else str(level)

            current_in = in_degrees.get(node, 0)
            initial_in = initial_in_degrees.get(node, 0)

            ax.text(
                x + card_width * 0.08,
                y + card_height * 0.67,
                str(node),
                fontsize=8.7,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + card_width * 0.19,
                y + card_height * 0.67,
                task_text,
                fontsize=6.35,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.19,
                y + card_height * 0.27,
                f"entrada={current_in}/{initial_in}",
                fontsize=6.5,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.73,
                y + card_height * 0.27,
                f"nivel={level_text}",
                fontsize=6.45,
                ha="left",
                va="center",
            )

        if phase == "cycle":
            footer = (
                "Morado: tarea bloqueada por una dependencia circular"
            )
        else:
            footer = (
                "entrada actual/inicial · nivel = dependencia más profunda"
            )

        ax.text(
            0.50,
            0.048,
            footer,
            fontsize=6.25,
            ha="center",
            va="center",
            color="#444444",
        )

        self._dibujar_leyenda_orden_topologico(ax)

    def _dibujar_cola_y_orden_topologico(
        self,
        ax,
        available_nodes,
        order,
        levels,
        blocked_nodes,
        phase,
        multiple_choice_count,
    ):
        """
        Dibuja la cola de fuentes y el orden topológico parcial.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        available_nodes = list(available_nodes)
        order = list(order)
        blocked_nodes = list(blocked_nodes)

        ax.text(
            0.02,
            0.90,
            "Cola de tareas disponibles",
            fontsize=11.2,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.text(
            0.98,
            0.90,
            (
                f"elecciones múltiples detectadas: "
                f"{multiple_choice_count}"
            ),
            fontsize=7.9,
            ha="right",
            va="center",
            color="#444444",
        )

        def draw_cells(
            values,
            y,
            face_color,
            edge_color,
            first_is_current=False,
            labels_below=None,
        ):
            max_cells = 14
            visible_values = list(values[:max_cells])

            if not visible_values:
                ax.text(
                    0.50,
                    y + 0.085,
                    "vacía",
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color="#666666",
                )
                return

            start_x = 0.08
            end_x = 0.92
            total_width = end_x - start_x
            cell_width = min(
                0.052,
                total_width / max(len(visible_values), 1),
            )
            gap = 0.008

            occupied_width = (
                len(visible_values) * cell_width
                + max(0, len(visible_values) - 1) * gap
            )
            current_x = 0.50 - occupied_width / 2

            for index, value in enumerate(visible_values):
                is_current = first_is_current and index == 0

                rectangle = Rectangle(
                    (current_x, y),
                    cell_width,
                    0.17,
                    facecolor="#E45756" if is_current else face_color,
                    edgecolor="#7A1D1D" if is_current else edge_color,
                    linewidth=2.0 if is_current else 1.4,
                )
                ax.add_patch(rectangle)

                ax.text(
                    current_x + cell_width / 2,
                    y + 0.105,
                    str(value),
                    fontsize=8.2,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

                if labels_below is not None:
                    label = labels_below.get(value)
                    label_text = "—" if label is None else str(label)

                    ax.text(
                        current_x + cell_width / 2,
                        y + 0.035,
                        f"n{label_text}",
                        fontsize=5.9,
                        ha="center",
                        va="center",
                    )

                current_x += cell_width + gap

        draw_cells(
            values=available_nodes,
            y=0.60,
            face_color="#FBE5A6",
            edge_color="#8A6D1D",
            first_is_current=True,
        )

        ax.text(
            0.02,
            0.43,
            "Orden topológico construido",
            fontsize=11.2,
            fontweight="bold",
            ha="left",
            va="center",
        )

        draw_cells(
            values=order,
            y=0.13,
            face_color="#B7D7F0",
            edge_color="#1F4F73",
            first_is_current=False,
            labels_below=levels,
        )

        if phase == "cycle" and blocked_nodes:
            ax.text(
                0.98,
                0.43,
                "Bloqueadas: " + ", ".join(map(str, blocked_nodes)),
                fontsize=8.0,
                fontweight="bold",
                ha="right",
                va="center",
                color="#5A316B",
            )
        elif phase == "finished":
            ax.text(
                0.98,
                0.43,
                "Orden completo y validado",
                fontsize=8.0,
                fontweight="bold",
                ha="right",
                va="center",
                color="#1F4F73",
            )

    def _dibujar_estado_orden_topologico(
        self,
        graph_ax,
        info_ax,
        structure_ax,
        graph,
        pos,
        state,
    ):
        """
        Dibuja un estado completo del algoritmo de Kahn.
        """

        graph_ax.clear()
        graph_ax.axis("off")

        limits = self._calcular_limites(
            pos,
            margin_x=1.15,
            margin_y=1.05,
        )

        graph_ax.set_xlim(limits[0], limits[1])
        graph_ax.set_ylim(limits[2], limits[3])
        graph_ax.set_aspect("equal", adjustable="box")

        task_names = dict(state.get("task_names", {}))
        short_names = dict(state.get("short_names", {}))
        initial_in_degrees = dict(
            state.get("initial_in_degrees", {})
        )
        in_degrees = dict(state.get("in_degrees", {}))
        levels = dict(state.get("levels", {}))

        available_nodes = list(state.get("available_nodes", []))
        processed_nodes = set(state.get("processed_nodes", set()))
        blocked_nodes = set(state.get("blocked_nodes", set()))
        order = list(state.get("order", []))
        processed_edges = set(state.get("processed_edges", set()))
        cycle_edges = set(state.get("cycle_edges", set()))

        current_node = state.get("current_node")
        active_edge = state.get("active_edge")
        active_successor = state.get("active_successor")
        action = state.get("action")
        phase = state.get("phase", "processing")

        for origin, destination in graph.edges():
            edge_key = (origin, destination)

            if edge_key in cycle_edges:
                color = "#8E5EA2"
                line_width = 4.0
                line_style = "solid"
                zorder = 21
            elif edge_key == active_edge:
                color = (
                    "#2E8B57"
                    if action == "unlock"
                    else "#E45756"
                )
                line_width = 4.2
                line_style = "solid"
                zorder = 20
            elif edge_key in processed_edges:
                color = "#2E8B57"
                line_width = 2.8
                line_style = "solid"
                zorder = 15
            else:
                color = "#B8B8B8"
                line_width = 1.55
                line_style = "solid"
                zorder = 10

            self._dibujar_flecha_bellman_ford(
                ax=graph_ax,
                pos=pos,
                origin=origin,
                destination=destination,
                color=color,
                line_width=line_width,
                zorder=zorder,
                line_style=line_style,
            )

        available_set = set(available_nodes)

        for node in graph.nodes():
            if node == current_node:
                face_color = "#E45756"
                edge_color = "#7A1D1D"
                node_size = 930
            elif node == active_successor:
                face_color = "#F28E2B"
                edge_color = "#8A4B08"
                node_size = 890
            elif node in blocked_nodes:
                face_color = "#D8C4E8"
                edge_color = "#5A316B"
                node_size = 810
            elif node in processed_nodes:
                face_color = "#4C9ED9"
                edge_color = "#1F4F73"
                node_size = 790
            elif node in available_set:
                face_color = "#F6C85F"
                edge_color = "#8A6D1D"
                node_size = 800
            else:
                face_color = "#D9D9D9"
                edge_color = "#666666"
                node_size = 760

            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[node],
                node_size=node_size,
                node_color=face_color,
                edgecolors=edge_color,
                linewidths=2.5 if node in {
                    current_node,
                    active_successor,
                } else 1.5,
                ax=graph_ax,
            )
            collection.set_zorder(25)

        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=9.5,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

            graph_ax.text(
                x,
                y - 0.40,
                short_names.get(node, task_names.get(node, node)),
                fontsize=6.5,
                ha="center",
                va="top",
                color="#222222",
                zorder=37,
            )

            level = levels.get(node)
            level_text = "—" if level is None else str(level)

            graph_ax.text(
                x,
                y + 0.38,
                (
                    f"in={in_degrees.get(node, 0)}"
                    f" | niv={level_text}"
                ),
                fontsize=6.8,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#222222",
                zorder=40,
                bbox={
                    "boxstyle": "round,pad=0.17",
                    "fc": "white",
                    "ec": "#555555",
                    "alpha": 0.97,
                },
            )

        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=8.9,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        old_in_degree = state.get("old_in_degree")
        new_in_degree = state.get("new_in_degree")

        if active_edge is not None and old_in_degree is not None:
            origin, destination = active_edge

            operation_text = (
                f"Eliminar conceptualmente {origin}→{destination}: "
                f"entrada({destination}) "
                f"{old_in_degree} → {new_in_degree}"
            )

            graph_ax.text(
                0.50,
                0.965,
                operation_text,
                transform=graph_ax.transAxes,
                fontsize=8.0,
                ha="center",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
                zorder=50,
            )

        is_unique = bool(state.get("is_unique", True))
        uniqueness_text = "único" if is_unique else "no único"

        status_text = (
            f"Procesadas: {len(processed_nodes)}/{graph.number_of_nodes()}"
            f"  ·  disponibles: {len(available_nodes)}"
            f"  ·  orden {uniqueness_text}"
        )

        if phase == "cycle":
            status_text = (
                f"CICLO  ·  procesadas: {len(processed_nodes)}"
                f"  ·  bloqueadas: {len(blocked_nodes)}"
            )

        graph_ax.text(
            0.99,
            0.985,
            status_text,
            transform=graph_ax.transAxes,
            fontsize=8.5,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        self._dibujar_tabla_orden_topologico(
            ax=info_ax,
            nodes=sorted(graph.nodes()),
            task_names=task_names,
            initial_in_degrees=initial_in_degrees,
            in_degrees=in_degrees,
            levels=levels,
            available_nodes=available_nodes,
            processed_nodes=processed_nodes,
            blocked_nodes=blocked_nodes,
            current_node=current_node,
            phase=phase,
            is_unique=is_unique,
        )

        self._dibujar_cola_y_orden_topologico(
            ax=structure_ax,
            available_nodes=available_nodes,
            order=order,
            levels=levels,
            blocked_nodes=sorted(blocked_nodes),
            phase=phase,
            multiple_choice_count=state.get(
                "multiple_choice_count",
                0,
            ),
        )

    def animate_topological_sort(
        self,
        graph,
        pos,
        states,
        title="DAG y ordenamiento topológico",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima el algoritmo de Kahn.

        La imagen final muestra:
        - el orden topológico;
        - grados de entrada reducidos a cero;
        - niveles de dependencia;
        - dependencias satisfechas;
        - o, en el modo opcional, el ciclo y las tareas bloqueadas.
        """

        if not states:
            raise ValueError(
                "La lista de estados del ordenamiento topológico "
                "no puede estar vacía."
            )

        (
            fig,
            graph_ax,
            info_ax,
            structure_ax,
        ) = self._preparar_figura_orden_topologico(title)

        if final_image_path is not None:
            self._dibujar_estado_orden_topologico(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[-1],
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(
                f"Imagen final guardada en: "
                f"{final_image_path}"
            )

        def init():
            self._dibujar_estado_orden_topologico(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_orden_topologico(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[frame_index],
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()

        return self.animation
    # ------------------------------------------------------------------
    # Elementos específicos de flujo máximo y corte mínimo
    # ------------------------------------------------------------------

    def _preparar_figura_flujo_maximo(self, title):
        """
        Crea una distribución comparable a Dijkstra y Edmonds-Karp.

        Distribución:
        - izquierda: flujo, capacidad y residuales de cada arista;
        - derecha superior: red dirigida;
        - derecha inferior: cola BFS, camino aumentante o corte mínimo.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.90, 4.10],
            height_ratios=[4.55, 1.85],
            wspace=0.08,
            hspace=0.09,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        graph_ax = fig.add_subplot(grid[0, 1])
        structure_ax = fig.add_subplot(grid[1, 1])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.93,
            bottom=0.045,
        )

        return fig, graph_ax, info_ax, structure_ax

    def _dibujar_leyenda_flujo_maximo(self, ax):
        """
        Dibuja la leyenda de Edmonds-Karp.
        """

        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="#666666",
                markersize=8,
                label="No visitado por BFS",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F6C85F",
                markeredgecolor="#8A6D1D",
                markersize=8,
                label="Visitado por BFS",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D",
                markersize=8,
                label="Vértice actual",
            ),
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=3,
                label="Camino aumentante",
            ),
            Line2D(
                [0],
                [0],
                color="#8E5EA2",
                linewidth=3,
                linestyle="dashed",
                label="Residual inversa",
            ),
            Line2D(
                [0],
                [0],
                color="#C62828",
                linewidth=4,
                label="Arista del corte mínimo",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            fontsize=6.7,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.65,
            handlelength=2.0,
            borderpad=0.50,
        )

    @staticmethod
    def _clave_arista_flujo(origen, destino):
        """
        Devuelve una clave dirigida para una arista original.
        """

        return origen, destino

    def _dibujar_tabla_aristas_flujo(
        self,
        ax,
        graph,
        flows,
        active_residual_edge,
        augmenting_path,
        cut_edges,
        phase,
        flow_value,
        cut_capacity,
    ):
        """
        Dibuja tarjetas con flujo, capacidad y residuales.

        Cada tarjeta contiene:
        - ``f/c``: flujo actual y capacidad original;
        - ``r+``: residual directa;
        - ``r-``: residual inversa.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            "Edmonds-Karp · flujo y red residual",
            fontsize=10.7,
            fontweight="bold",
            ha="center",
            va="top",
        )

        if phase in {"min_cut", "finished"}:
            subtitle = (
                f"Flujo máximo: {flow_value} · "
                f"corte mínimo: {cut_capacity}"
            )
        else:
            subtitle = f"Valor actual del flujo: {flow_value}"

        ax.text(
            0.50,
            0.948,
            subtitle,
            fontsize=8.1,
            ha="center",
            va="top",
            color="#444444",
        )

        active_original = None
        active_direction = None

        if active_residual_edge is not None:
            active_original = tuple(
                active_residual_edge.get("original_edge", ())
            )
            active_direction = active_residual_edge.get("direction")

        direct_path_edges = {
            tuple(arco["original_edge"])
            for arco in augmenting_path
            if arco.get("direction") == "directo"
        }
        reverse_path_edges = {
            tuple(arco["original_edge"])
            for arco in augmenting_path
            if arco.get("direction") == "inverso"
        }
        cut_edge_keys = {
            (origen, destino)
            for origen, destino, _ in cut_edges
        }

        edge_order = list(
            graph.graph.get(
                "edge_order",
                [
                    (
                        origen,
                        destino,
                        datos.get("capacity", 0),
                    )
                    for origen, destino, datos
                    in graph.edges(data=True)
                ],
            )
        )

        number_of_columns = 2
        card_width = 0.420
        card_height = 0.091
        horizontal_gap = 0.045
        vertical_gap = 0.014

        total_width = (
            number_of_columns * card_width
            + (number_of_columns - 1) * horizontal_gap
        )

        initial_x = (1 - total_width) / 2
        top_y = 0.660

        for index, (origin, destination, capacity) in enumerate(edge_order):
            row = index // number_of_columns
            column = index % number_of_columns

            x = initial_x + column * (card_width + horizontal_gap)
            y = top_y - row * (card_height + vertical_gap)

            edge_key = (origin, destination)
            flow = flows.get(edge_key, 0)
            direct_residual = capacity - flow
            reverse_residual = flow

            if edge_key in cut_edge_keys:
                face_color = "#F6B4B4"
                edge_color = "#C62828"
                line_width = 2.4
            elif edge_key == active_original:
                if active_direction == "inverso":
                    face_color = "#E8D7F1"
                    edge_color = "#8E5EA2"
                else:
                    face_color = "#F6B4B4"
                    edge_color = "#C62828"
                line_width = 2.5
            elif edge_key in reverse_path_edges:
                face_color = "#E8D7F1"
                edge_color = "#8E5EA2"
                line_width = 1.9
            elif edge_key in direct_path_edges:
                face_color = "#B7E4C7"
                edge_color = "#2E8B57"
                line_width = 1.9
            elif flow == capacity:
                face_color = "#B7D7F0"
                edge_color = "#1F4F73"
                line_width = 1.5
            elif flow > 0:
                face_color = "#D5E8D4"
                edge_color = "#497A4A"
                line_width = 1.5
            else:
                face_color = "#E5E5E5"
                edge_color = "#777777"
                line_width = 1.4

            rectangle = Rectangle(
                (x, y),
                card_width,
                card_height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rectangle)

            ax.text(
                x + card_width * 0.08,
                y + card_height * 0.67,
                f"{origin}→{destination}",
                fontsize=7.4,
                fontweight="bold",
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.58,
                y + card_height * 0.67,
                f"f/c={flow}/{capacity}",
                fontsize=7.0,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.08,
                y + card_height * 0.27,
                f"r+={direct_residual}",
                fontsize=6.7,
                ha="left",
                va="center",
            )

            ax.text(
                x + card_width * 0.58,
                y + card_height * 0.27,
                f"r-={reverse_residual}",
                fontsize=6.7,
                ha="left",
                va="center",
            )

        ax.text(
            0.50,
            0.050,
            (
                "r+: aumentar flujo · r-: cancelar flujo previo · "
                "azul: saturada"
            ),
            fontsize=6.1,
            ha="center",
            va="center",
            color="#444444",
        )

        self._dibujar_leyenda_flujo_maximo(ax)

    def _dibujar_flecha_flujo(
        self,
        ax,
        pos,
        origin,
        destination,
        color,
        line_width,
        zorder,
        line_style="solid",
        curvature=0.0,
    ):
        """
        Dibuja una arista dirigida de la red o de la residual.
        """

        x1, y1 = pos[origin]
        x2, y2 = pos[destination]

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=line_width,
            linestyle=line_style,
            color=color,
            shrinkA=18,
            shrinkB=18,
            connectionstyle=f"arc3,rad={curvature}",
            zorder=zorder,
        )
        ax.add_patch(arrow)

    def _dibujar_etiqueta_flujo_arista(
        self,
        ax,
        pos,
        origin,
        destination,
        flow,
        capacity,
    ):
        """
        Dibuja la etiqueta ``flujo/capacidad`` de una arista original.
        """

        x1, y1 = pos[origin]
        x2, y2 = pos[destination]

        middle_x = (x1 + x2) / 2
        middle_y = (y1 + y2) / 2

        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2) ** 0.5

        if length == 0:
            offset_x = 0
            offset_y = 0
        else:
            offset_x = -dy / length * 0.16
            offset_y = dx / length * 0.16

        ax.text(
            middle_x + offset_x,
            middle_y + offset_y,
            f"{flow}/{capacity}",
            fontsize=7.7,
            fontweight="bold" if flow == capacity else "normal",
            ha="center",
            va="center",
            color="#1F4F73" if flow == capacity else "#222222",
            zorder=38,
            bbox={
                "boxstyle": "round,pad=0.16",
                "fc": "white",
                "ec": "none",
                "alpha": 0.97,
            },
        )

    def _dibujar_panel_edmonds_karp(
        self,
        ax,
        queue,
        augmenting_path,
        bottleneck,
        augmentation_index,
        flow_value,
        phase,
        reachable_nodes,
        all_nodes,
        cut_edges,
        cut_capacity,
    ):
        """
        Dibuja cola BFS, camino aumentante o corte mínimo.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.02,
            0.93,
            (
                f"Edmonds-Karp · aumento {augmentation_index} · "
                f"flujo {flow_value}"
            ),
            fontsize=11.3,
            fontweight="bold",
            ha="left",
            va="center",
        )

        if phase in {"min_cut", "finished"}:
            source_side = sorted(reachable_nodes)
            sink_side = sorted(set(all_nodes) - set(reachable_nodes))

            ax.text(
                0.02,
                0.70,
                "Lado S: " + ", ".join(source_side),
                fontsize=8.8,
                fontweight="bold",
                ha="left",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.30",
                    "fc": "#B7D7F0",
                    "ec": "#1F4F73",
                    "alpha": 0.98,
                },
            )

            ax.text(
                0.98,
                0.70,
                "Lado T: " + ", ".join(sink_side),
                fontsize=8.8,
                fontweight="bold",
                ha="right",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.30",
                    "fc": "#F8D7B5",
                    "ec": "#8A4B08",
                    "alpha": 0.98,
                },
            )

            ax.text(
                0.02,
                0.46,
                "Aristas del corte mínimo",
                fontsize=10.5,
                fontweight="bold",
                ha="left",
                va="center",
            )

            visible_cut_edges = list(cut_edges[:8])

            if visible_cut_edges:
                cell_width = 0.125
                gap = 0.018
                occupied_width = (
                    len(visible_cut_edges) * cell_width
                    + max(0, len(visible_cut_edges) - 1) * gap
                )
                current_x = 0.50 - occupied_width / 2

                for origin, destination, capacity in visible_cut_edges:
                    rectangle = Rectangle(
                        (current_x, 0.12),
                        cell_width,
                        0.23,
                        facecolor="#F6B4B4",
                        edgecolor="#C62828",
                        linewidth=1.8,
                    )
                    ax.add_patch(rectangle)

                    ax.text(
                        current_x + cell_width / 2,
                        0.255,
                        f"{origin}→{destination}",
                        fontsize=7.4,
                        fontweight="bold",
                        ha="center",
                        va="center",
                    )

                    ax.text(
                        current_x + cell_width / 2,
                        0.17,
                        f"c={capacity}",
                        fontsize=6.8,
                        ha="center",
                        va="center",
                    )

                    current_x += cell_width + gap

            ax.text(
                0.98,
                0.46,
                f"capacidad = {cut_capacity}",
                fontsize=9.2,
                fontweight="bold",
                ha="right",
                va="center",
                color="#C62828",
            )

            return

        ax.text(
            0.02,
            0.68,
            "Cola BFS residual",
            fontsize=10.4,
            fontweight="bold",
            ha="left",
            va="center",
        )

        queue = list(queue)

        if queue:
            max_cells = 10
            visible_queue = queue[:max_cells]
            cell_width = 0.060
            gap = 0.010
            occupied_width = (
                len(visible_queue) * cell_width
                + max(0, len(visible_queue) - 1) * gap
            )
            current_x = 0.50 - occupied_width / 2

            for index, node in enumerate(visible_queue):
                is_next = index == 0

                rectangle = Rectangle(
                    (current_x, 0.56),
                    cell_width,
                    0.18,
                    facecolor="#E45756" if is_next else "#FBE5A6",
                    edgecolor="#7A1D1D" if is_next else "#8A6D1D",
                    linewidth=2.0 if is_next else 1.4,
                )
                ax.add_patch(rectangle)

                ax.text(
                    current_x + cell_width / 2,
                    0.65,
                    str(node),
                    fontsize=8.2,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

                current_x += cell_width + gap
        else:
            ax.text(
                0.50,
                0.65,
                "cola vacía",
                fontsize=8.8,
                fontweight="bold",
                ha="center",
                va="center",
                color="#666666",
            )

        ax.text(
            0.02,
            0.38,
            "Camino aumentante residual",
            fontsize=10.4,
            fontweight="bold",
            ha="left",
            va="center",
        )

        augmenting_path = list(augmenting_path)

        if augmenting_path:
            max_cells = 8
            visible_path = augmenting_path[:max_cells]
            cell_width = 0.115
            gap = 0.012
            occupied_width = (
                len(visible_path) * cell_width
                + max(0, len(visible_path) - 1) * gap
            )
            current_x = 0.50 - occupied_width / 2

            for arco in visible_path:
                is_reverse = arco.get("direction") == "inverso"

                rectangle = Rectangle(
                    (current_x, 0.08),
                    cell_width,
                    0.22,
                    facecolor="#E8D7F1" if is_reverse else "#B7E4C7",
                    edgecolor="#8E5EA2" if is_reverse else "#2E8B57",
                    linewidth=1.7,
                )
                ax.add_patch(rectangle)

                ax.text(
                    current_x + cell_width / 2,
                    0.215,
                    (
                        f"{arco['origin']}→"
                        f"{arco['destination']}"
                    ),
                    fontsize=7.0,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

                ax.text(
                    current_x + cell_width / 2,
                    0.135,
                    (
                        f"r={arco['residual']} · "
                        f"{'inv' if is_reverse else 'dir'}"
                    ),
                    fontsize=6.2,
                    ha="center",
                    va="center",
                )

                current_x += cell_width + gap

            ax.text(
                0.98,
                0.38,
                (
                    "Δ = "
                    + ("—" if bottleneck is None else str(bottleneck))
                ),
                fontsize=9.2,
                fontweight="bold",
                ha="right",
                va="center",
                color="#2E8B57",
            )
        else:
            ax.text(
                0.50,
                0.19,
                "todavía no reconstruido",
                fontsize=8.6,
                ha="center",
                va="center",
                color="#666666",
            )

    def _dibujar_estado_flujo_maximo(
        self,
        graph_ax,
        info_ax,
        structure_ax,
        graph,
        pos,
        state,
        source_node,
        sink_node,
    ):
        """
        Dibuja un estado completo de Edmonds-Karp.
        """

        graph_ax.clear()
        graph_ax.axis("off")

        limits = self._calcular_limites(
            pos,
            margin_x=1.25,
            margin_y=1.10,
        )

        graph_ax.set_xlim(limits[0], limits[1])
        graph_ax.set_ylim(limits[2], limits[3])
        graph_ax.set_aspect("equal", adjustable="box")

        flows = dict(state.get("flows", {}))
        visited = set(state.get("visited", set()))
        queue = list(state.get("queue", []))
        current_node = state.get("current_node")
        active_residual_edge = state.get("active_residual_edge")
        augmenting_path = list(state.get("augmenting_path", []))
        reachable_nodes = set(state.get("reachable_nodes", set()))
        cut_edges = list(state.get("cut_edges", []))
        cut_capacity = state.get("cut_capacity")
        phase = state.get("phase", "bfs")
        action = state.get("action")

        active_original = None
        active_direction = None

        if active_residual_edge is not None:
            active_original = tuple(
                active_residual_edge.get("original_edge", ())
            )
            active_direction = active_residual_edge.get("direction")

        direct_path_edges = {
            tuple(arco["original_edge"])
            for arco in augmenting_path
            if arco.get("direction") == "directo"
        }
        reverse_path_arcs = [
            arco
            for arco in augmenting_path
            if arco.get("direction") == "inverso"
        ]
        path_nodes = set()

        for arco in augmenting_path:
            path_nodes.add(arco["origin"])
            path_nodes.add(arco["destination"])

        cut_edge_keys = {
            (origin, destination)
            for origin, destination, _ in cut_edges
        }

        # 1. Aristas originales.
        for origin, destination, data in graph.edges(data=True):
            edge_key = (origin, destination)
            capacity = data.get("capacity", 0)
            flow = flows.get(edge_key, 0)

            if edge_key in cut_edge_keys:
                color = "#C62828"
                line_width = 4.2
                line_style = "solid"
                zorder = 22
            elif (
                edge_key == active_original
                and active_direction == "directo"
            ):
                color = "#E45756"
                line_width = 4.2
                line_style = "solid"
                zorder = 21
            elif edge_key in direct_path_edges:
                color = "#2E8B57"
                line_width = 3.6
                line_style = "solid"
                zorder = 18
            elif flow == capacity:
                color = "#4C78A8"
                line_width = 2.7
                line_style = "solid"
                zorder = 15
            elif flow > 0:
                color = "#74A66A"
                line_width = 2.4
                line_style = "solid"
                zorder = 14
            else:
                color = "#B8B8B8"
                line_width = 1.6
                line_style = "solid"
                zorder = 10

            self._dibujar_flecha_flujo(
                ax=graph_ax,
                pos=pos,
                origin=origin,
                destination=destination,
                color=color,
                line_width=line_width,
                zorder=zorder,
                line_style=line_style,
                curvature=0.0,
            )

            self._dibujar_etiqueta_flujo_arista(
                ax=graph_ax,
                pos=pos,
                origin=origin,
                destination=destination,
                flow=flow,
                capacity=capacity,
            )

        # 2. Residuales inversas pertenecientes al camino.
        for arco in reverse_path_arcs:
            is_active = (
                active_residual_edge is not None
                and arco.get("origin")
                == active_residual_edge.get("origin")
                and arco.get("destination")
                == active_residual_edge.get("destination")
                and arco.get("original_edge")
                == active_residual_edge.get("original_edge")
            )

            self._dibujar_flecha_flujo(
                ax=graph_ax,
                pos=pos,
                origin=arco["origin"],
                destination=arco["destination"],
                color="#E45756" if is_active else "#8E5EA2",
                line_width=4.3 if is_active else 3.5,
                zorder=25,
                line_style="dashed",
                curvature=0.16,
            )

        # Una residual inversa puede estar activa durante BFS antes de que
        # el camino completo haya sido reconstruido.
        if (
            active_residual_edge is not None
            and active_direction == "inverso"
            and not any(
                arco.get("origin")
                == active_residual_edge.get("origin")
                and arco.get("destination")
                == active_residual_edge.get("destination")
                and arco.get("original_edge")
                == active_residual_edge.get("original_edge")
                for arco in reverse_path_arcs
            )
        ):
            self._dibujar_flecha_flujo(
                ax=graph_ax,
                pos=pos,
                origin=active_residual_edge["origin"],
                destination=active_residual_edge["destination"],
                color="#8E5EA2",
                line_width=4.1,
                zorder=25,
                line_style="dashed",
                curvature=0.16,
            )

        # 3. Estados de nodos.
        for node in graph.nodes():
            if phase in {"min_cut", "finished"}:
                if node in reachable_nodes:
                    face_color = "#B7D7F0"
                    edge_color = "#1F4F73"
                else:
                    face_color = "#F8D7B5"
                    edge_color = "#8A4B08"
            elif node == current_node:
                face_color = "#E45756"
                edge_color = "#7A1D1D"
            elif node in path_nodes:
                face_color = "#B7E4C7"
                edge_color = "#2E8B57"
            elif node in visited:
                face_color = "#F6C85F"
                edge_color = "#8A6D1D"
            else:
                face_color = "#D9D9D9"
                edge_color = "#666666"

            node_size = 800

            if node == source_node:
                face_color = "#7BC67B"
                edge_color = "#27632A"
                node_size = 930
            elif node == sink_node:
                face_color = "#D8C4E8"
                edge_color = "#5A316B"
                node_size = 930

            if node == current_node:
                face_color = "#E45756"
                edge_color = "#7A1D1D"
                node_size = 950

            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[node],
                node_size=node_size,
                node_color=face_color,
                edgecolors=edge_color,
                linewidths=2.5 if node in {
                    source_node,
                    sink_node,
                    current_node,
                } else 1.5,
                ax=graph_ax,
            )
            collection.set_zorder(30)

        # 4. Etiquetas de nodos.
        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=38,
            )

        source_x, source_y = pos[source_node]
        sink_x, sink_y = pos[sink_node]

        graph_ax.text(
            source_x,
            source_y - 0.45,
            "fuente",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#27632A",
            zorder=40,
        )

        graph_ax.text(
            sink_x,
            sink_y - 0.45,
            "sumidero",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
            color="#5A316B",
            zorder=40,
        )

        # 5. Operación activa y mensaje.
        if active_residual_edge is not None:
            original_origin, original_destination = (
                active_residual_edge["original_edge"]
            )
            direction_text = (
                "directa"
                if active_residual_edge.get("direction") == "directo"
                else "inversa"
            )

            operation_text = (
                f"Residual {direction_text} "
                f"{active_residual_edge['origin']}→"
                f"{active_residual_edge['destination']} "
                f"(original {original_origin}→{original_destination}) "
                f"· capacidad residual "
                f"{active_residual_edge['residual']}"
            )

            old_flow = state.get("old_flow")
            new_flow = state.get("new_flow")

            if old_flow is not None and new_flow is not None:
                operation_text += (
                    f" · flujo original {old_flow}→{new_flow}"
                )

            graph_ax.text(
                0.50,
                0.965,
                operation_text,
                transform=graph_ax.transAxes,
                fontsize=7.8,
                ha="center",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
                zorder=50,
            )

        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=8.8,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        flow_value = state.get("flow_value", 0)
        augmentation_index = state.get("augmentation_index", 0)

        if phase in {"min_cut", "finished"}:
            status_text = (
                f"Flujo máximo: {flow_value} · "
                f"corte mínimo: {cut_capacity}"
            )
        elif phase == "no_path":
            status_text = (
                f"Sin camino aumentante · flujo {flow_value}"
            )
        elif phase == "bfs":
            status_text = (
                f"Aumento {augmentation_index} · flujo {flow_value} · "
                f"visitados {len(visited)}"
            )
        else:
            bottleneck = state.get("bottleneck")
            bottleneck_text = (
                "—" if bottleneck is None else str(bottleneck)
            )
            status_text = (
                f"Aumento {augmentation_index} · flujo {flow_value} · "
                f"Δ={bottleneck_text}"
            )

        graph_ax.text(
            0.99,
            0.985,
            status_text,
            transform=graph_ax.transAxes,
            fontsize=8.6,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        self._dibujar_tabla_aristas_flujo(
            ax=info_ax,
            graph=graph,
            flows=flows,
            active_residual_edge=active_residual_edge,
            augmenting_path=augmenting_path,
            cut_edges=cut_edges,
            phase=phase,
            flow_value=flow_value,
            cut_capacity=cut_capacity,
        )

        self._dibujar_panel_edmonds_karp(
            ax=structure_ax,
            queue=queue,
            augmenting_path=augmenting_path,
            bottleneck=state.get("bottleneck"),
            augmentation_index=augmentation_index,
            flow_value=flow_value,
            phase=phase,
            reachable_nodes=reachable_nodes,
            all_nodes=list(graph.nodes()),
            cut_edges=cut_edges,
            cut_capacity=cut_capacity,
        )

    def animate_max_flow_min_cut(
        self,
        graph,
        pos,
        states,
        source_node,
        sink_node,
        title="Flujo máximo y corte mínimo con Edmonds-Karp",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima Edmonds-Karp y la obtención del corte mínimo.

        La imagen final muestra:
        - flujo y capacidad de todas las aristas;
        - capacidades residuales directas e inversas;
        - partición del corte mínimo;
        - aristas saturadas que cruzan el corte;
        - igualdad entre flujo máximo y corte mínimo.
        """

        if not states:
            raise ValueError(
                "La lista de estados de Edmonds-Karp no puede estar vacía."
            )

        (
            fig,
            graph_ax,
            info_ax,
            structure_ax,
        ) = self._preparar_figura_flujo_maximo(title)

        if final_image_path is not None:
            self._dibujar_estado_flujo_maximo(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[-1],
                source_node=source_node,
                sink_node=sink_node,
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )

            print(
                f"Imagen final guardada en: "
                f"{final_image_path}"
            )

        def init():
            self._dibujar_estado_flujo_maximo(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[0],
                source_node=source_node,
                sink_node=sink_node,
            )
            return []

        def update(frame_index):
            self._dibujar_estado_flujo_maximo(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pos=pos,
                state=states[frame_index],
                source_node=source_node,
                sink_node=sink_node,
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()

        return self.animation

    # ------------------------------------------------------------------
    # Elementos específicos de centralidad, PageRank y comunidades
    # ------------------------------------------------------------------

    def _preparar_figura_centralidad(self, title):
        """
        Mantiene una distribución comparable con los algoritmos anteriores.

        Distribución:
        - izquierda: puntuaciones y ranking del estado actual;
        - derecha superior: grafo con tamaños y colores significativos;
        - derecha inferior: ranking, convergencia o comunidades.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[2.05, 3.95],
            height_ratios=[4.65, 1.75],
            wspace=0.08,
            hspace=0.09,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        graph_ax = fig.add_subplot(grid[0, 1])
        structure_ax = fig.add_subplot(grid[1, 1])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.93,
            bottom=0.045,
        )

        return fig, graph_ax, info_ax, structure_ax

    @staticmethod
    def _paleta_comunidades_analisis():
        """Devuelve una paleta estable para las comunidades."""

        return [
            "#B7D7F0",
            "#FBE5A6",
            "#D8C4E8",
            "#B7E4C7",
            "#F7C6C7",
            "#CDE7E8",
            "#E7D6B8",
            "#D6E4B7",
            "#D8D8F0",
            "#F4D2A7",
            "#C8D6E5",
            "#D5E8D4",
            "#FFE0B2",
            "#E1BEE7",
            "#F8BBD0",
        ]

    @staticmethod
    def _normalizar_valores_analisis(scores):
        """Normaliza puntuaciones al intervalo [0, 1]."""

        if not scores:
            return {}

        values = list(scores.values())
        minimum = min(values)
        maximum = max(values)

        if abs(maximum - minimum) < 1e-15:
            return {node: 0.5 for node in scores}

        return {
            node: (value - minimum) / (maximum - minimum)
            for node, value in scores.items()
        }

    @staticmethod
    def _color_por_puntuacion_analisis(value):
        """Interpola entre gris claro y azul según una puntuación normalizada."""

        value = max(0.0, min(1.0, float(value)))
        start = (224, 224, 224)
        end = (76, 158, 217)
        rgb = tuple(
            round(start[index] + value * (end[index] - start[index]))
            for index in range(3)
        )
        return "#{:02X}{:02X}{:02X}".format(*rgb)

    def _dibujar_leyenda_centralidad(self, ax, phase):
        """Dibuja una leyenda adaptada a la fase activa."""

        if phase in {"communities", "final"}:
            elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#B7D7F0",
                    markeredgecolor="#666666",
                    markersize=8,
                    label="Comunidad",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#C62828",
                    linewidth=3,
                    label="Enlace entre comunidades",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#2E8B57",
                    linewidth=4,
                    label="Fusión examinada",
                ),
            ]
        elif phase == "pagerank":
            elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#4C9ED9",
                    markeredgecolor="#1F4F73",
                    markersize=8,
                    label="PageRank alto",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#D9D9D9",
                    markeredgecolor="#666666",
                    markersize=8,
                    label="PageRank bajo",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#777777",
                    linewidth=2,
                    label="Enlace dirigido",
                ),
            ]
        else:
            elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#4C9ED9",
                    markeredgecolor="#1F4F73",
                    markersize=8,
                    label="Puntuación alta",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#D9D9D9",
                    markeredgecolor="#666666",
                    markersize=8,
                    label="Puntuación baja",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="#E45756",
                    markeredgecolor="#7A1D1D",
                    markersize=8,
                    label="Vértice destacado",
                ),
            ]

        ax.legend(
            handles=elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            fontsize=6.7,
            framealpha=0.97,
            ncol=1 if len(elements) <= 3 else 2,
            columnspacing=0.65,
            handlelength=2.0,
            borderpad=0.50,
        )

    def _dibujar_tabla_centralidad(self, ax, graph, state):
        """Dibuja puntuaciones, posiciones y comunidades."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        phase = state.get("phase", "degree")
        metric_label = state.get("metric_label", "Análisis estructural")
        scores = dict(state.get("scores", {}))
        ranking = list(state.get("ranking", []))
        current_node = state.get("current_node")
        communities = list(state.get("communities", []))

        ax.text(
            0.50,
            0.985,
            metric_label,
            fontsize=11.0,
            fontweight="bold",
            ha="center",
            va="top",
        )

        if phase == "final":
            all_metrics = dict(state.get("all_metrics", {}))
            community_map = dict(state.get("community_map", {}))

            headers = ["v", "grado", "cerc.", "inter.", "autov.", "PR", "com."]
            x_positions = [0.055, 0.18, 0.33, 0.48, 0.63, 0.78, 0.925]

            for x, header in zip(x_positions, headers):
                ax.text(
                    x,
                    0.925,
                    header,
                    fontsize=6.6,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

            nodes = sorted(graph.nodes())
            top_y = 0.875
            row_height = 0.049

            for index, node in enumerate(nodes):
                y = top_y - index * row_height
                background = "#F7F7F7" if index % 2 == 0 else "white"
                rectangle = Rectangle(
                    (0.025, y - 0.020),
                    0.95,
                    0.041,
                    facecolor=background,
                    edgecolor="#DDDDDD",
                    linewidth=0.5,
                )
                ax.add_patch(rectangle)

                values = [
                    node,
                    f"{all_metrics['degree'][node]:.3f}",
                    f"{all_metrics['closeness'][node]:.3f}",
                    f"{all_metrics['betweenness'][node]:.3f}",
                    f"{all_metrics['eigenvector'][node]:.3f}",
                    f"{all_metrics['pagerank'][node]:.3f}",
                    str(community_map.get(node, "—")),
                ]

                for x, value in zip(x_positions, values):
                    ax.text(
                        x,
                        y,
                        str(value),
                        fontsize=5.9,
                        fontweight="bold" if x == x_positions[0] else "normal",
                        ha="center",
                        va="center",
                    )
        else:
            rank_position = {
                node: index + 1
                for index, (node, _) in enumerate(ranking)
            }

            nodes = sorted(graph.nodes())
            number_of_columns = 3
            card_width = 0.275
            card_height = 0.098
            horizontal_gap = 0.035
            vertical_gap = 0.018
            total_width = (
                number_of_columns * card_width
                + (number_of_columns - 1) * horizontal_gap
            )
            initial_x = (1 - total_width) / 2
            top_y = 0.675

            community_map = {}
            for community_index, community in enumerate(communities, start=1):
                for node in community:
                    community_map[node] = community_index

            normalized = self._normalizar_valores_analisis(scores)

            for index, node in enumerate(nodes):
                row = index // number_of_columns
                column = index % number_of_columns
                x = initial_x + column * (card_width + horizontal_gap)
                y = top_y - row * (card_height + vertical_gap)

                face_color = self._color_por_puntuacion_analisis(
                    normalized.get(node, 0.0)
                )
                edge_color = "#666666"
                line_width = 1.3

                if phase == "communities":
                    palette = self._paleta_comunidades_analisis()
                    community_number = community_map.get(node, 1)
                    face_color = palette[(community_number - 1) % len(palette)]

                if node == current_node:
                    edge_color = "#C62828"
                    line_width = 3.0

                rectangle = Rectangle(
                    (x, y),
                    card_width,
                    card_height,
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=line_width,
                )
                ax.add_patch(rectangle)

                score_text = (
                    "—"
                    if node not in scores
                    else f"{scores[node]:.5f}"
                )
                rank_text = rank_position.get(node, "—")

                ax.text(
                    x + card_width * 0.14,
                    y + card_height * 0.65,
                    str(node),
                    fontsize=8.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
                ax.text(
                    x + card_width * 0.34,
                    y + card_height * 0.65,
                    score_text,
                    fontsize=6.5,
                    ha="left",
                    va="center",
                )
                ax.text(
                    x + card_width * 0.14,
                    y + card_height * 0.27,
                    f"rango {rank_text}",
                    fontsize=6.2,
                    ha="left",
                    va="center",
                )

                if phase == "communities":
                    ax.text(
                        x + card_width * 0.62,
                        y + card_height * 0.27,
                        f"C{community_map.get(node, '—')}",
                        fontsize=6.4,
                        ha="left",
                        va="center",
                    )

        footer = state.get("table_footer", "")
        ax.text(
            0.50,
            0.047,
            footer,
            fontsize=6.2,
            ha="center",
            va="center",
            color="#444444",
        )

        if phase != "final":
            self._dibujar_leyenda_centralidad(ax, phase)

    def _dibujar_panel_inferior_centralidad(self, ax, state):
        """Dibuja ranking, convergencia, fusiones o resumen final."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        phase = state.get("phase", "degree")
        ranking = list(state.get("ranking", []))
        communities = list(state.get("communities", []))

        if phase == "final":
            winners = list(state.get("winners", []))
            ax.text(
                0.02,
                0.88,
                "Comparación final de métricas",
                fontsize=11.6,
                fontweight="bold",
                ha="left",
                va="center",
            )

            if winners:
                cell_width = 0.14
                gap = 0.012
                total = len(winners) * cell_width + (len(winners) - 1) * gap
                x = 0.50 - total / 2

                for label, node, value in winners:
                    rectangle = Rectangle(
                        (x, 0.23),
                        cell_width,
                        0.43,
                        facecolor="#EEF4FA",
                        edgecolor="#577590",
                        linewidth=1.4,
                    )
                    ax.add_patch(rectangle)
                    ax.text(
                        x + cell_width / 2,
                        0.54,
                        label,
                        fontsize=6.7,
                        fontweight="bold",
                        ha="center",
                        va="center",
                    )
                    ax.text(
                        x + cell_width / 2,
                        0.40,
                        str(node),
                        fontsize=10,
                        fontweight="bold",
                        ha="center",
                        va="center",
                    )
                    ax.text(
                        x + cell_width / 2,
                        0.28,
                        f"{value:.4f}",
                        fontsize=6.6,
                        ha="center",
                        va="center",
                    )
                    x += cell_width + gap

            ax.text(
                0.98,
                0.88,
                (
                    f"Comunidades: {len(communities)} · "
                    f"modularidad: {state.get('modularity', 0):.4f}"
                ),
                fontsize=8.3,
                ha="right",
                va="center",
                color="#444444",
            )
            return

        if phase == "communities":
            ax.text(
                0.02,
                0.88,
                "Optimización voraz de modularidad",
                fontsize=11.6,
                fontweight="bold",
                ha="left",
                va="center",
            )
            ax.text(
                0.98,
                0.88,
                (
                    f"comunidades: {len(communities)} · "
                    f"Q={state.get('modularity', 0):.5f}"
                ),
                fontsize=8.3,
                ha="right",
                va="center",
                color="#444444",
            )

            visible = communities[:8]
            cell_width = min(0.105, 0.80 / max(len(visible), 1))
            gap = 0.010
            occupied = len(visible) * cell_width + max(0, len(visible) - 1) * gap
            x = 0.50 - occupied / 2
            palette = self._paleta_comunidades_analisis()
            active_merge = state.get("active_merge")
            active_sets = [set(group) for group in active_merge] if active_merge else []

            for index, community in enumerate(visible):
                is_active = any(set(community) == group for group in active_sets)
                rectangle = Rectangle(
                    (x, 0.21),
                    cell_width,
                    0.43,
                    facecolor=palette[index % len(palette)],
                    edgecolor="#2E8B57" if is_active else "#666666",
                    linewidth=2.6 if is_active else 1.3,
                )
                ax.add_patch(rectangle)
                ax.text(
                    x + cell_width / 2,
                    0.50,
                    f"C{index + 1}",
                    fontsize=7.2,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
                ax.text(
                    x + cell_width / 2,
                    0.34,
                    "".join(map(str, sorted(community))),
                    fontsize=6.2,
                    ha="center",
                    va="center",
                )
                x += cell_width + gap
            return

        metric_label = state.get("metric_label", "Ranking")
        ax.text(
            0.02,
            0.88,
            f"Ranking · {metric_label}",
            fontsize=11.6,
            fontweight="bold",
            ha="left",
            va="center",
        )

        iteration = state.get("iteration")
        delta = state.get("delta")
        if iteration is not None:
            iteration_text = (
                f"iteración: {iteration} · distribución inicial"
                if delta == float("inf")
                else f"iteración: {iteration} · cambio: {delta:.3e}"
            )
            ax.text(
                0.98,
                0.88,
                iteration_text,
                fontsize=8.2,
                ha="right",
                va="center",
                color="#444444",
            )

        visible = ranking[:7]
        if not visible:
            return

        cell_width = 0.105
        gap = 0.012
        occupied = len(visible) * cell_width + (len(visible) - 1) * gap
        x = 0.50 - occupied / 2

        for index, (node, value) in enumerate(visible, start=1):
            rectangle = Rectangle(
                (x, 0.21),
                cell_width,
                0.43,
                facecolor="#B7D7F0" if index == 1 else "#EEF4FA",
                edgecolor="#1F4F73",
                linewidth=2.0 if index == 1 else 1.2,
            )
            ax.add_patch(rectangle)
            ax.text(
                x + cell_width / 2,
                0.53,
                f"#{index}",
                fontsize=6.5,
                ha="center",
                va="center",
            )
            ax.text(
                x + cell_width / 2,
                0.41,
                str(node),
                fontsize=9.5,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax.text(
                x + cell_width / 2,
                0.28,
                f"{value:.5f}",
                fontsize=6.2,
                ha="center",
                va="center",
            )
            x += cell_width + gap

    def _dibujar_grafo_centralidad(
        self,
        graph_ax,
        graph,
        pagerank_graph,
        pos,
        state,
    ):
        """Dibuja el grafo del estado actual."""

        graph_ax.clear()
        graph_ax.axis("off")

        limits = self._calcular_limites(
            pos,
            margin_x=1.15,
            margin_y=1.10,
        )
        graph_ax.set_xlim(limits[0], limits[1])
        graph_ax.set_ylim(limits[2], limits[3])
        graph_ax.set_aspect("equal", adjustable="box")

        phase = state.get("phase", "degree")
        scores = dict(state.get("scores", {}))
        current_node = state.get("current_node")
        communities = list(state.get("communities", []))
        active_merge = state.get("active_merge")
        bridge_edges = {
            self._normalizar_arista(*edge)
            for edge in state.get("bridge_edges", [])
        }

        community_map = {}
        for community_index, community in enumerate(communities):
            for node in community:
                community_map[node] = community_index

        active_nodes = set()
        if active_merge:
            for group in active_merge:
                active_nodes.update(group)

        use_directed = phase == "pagerank"
        drawing_graph = pagerank_graph if use_directed else graph

        if use_directed:
            for origin, destination, data in drawing_graph.edges(data=True):
                reverse_exists = drawing_graph.has_edge(destination, origin)
                if reverse_exists:
                    curvature = 0.10 if str(origin) < str(destination) else -0.10
                else:
                    curvature = 0.0

                weight = float(data.get("weight", 1.0))
                line_width = 0.8 + min(weight, 3.0) * 0.45
                self._dibujar_flecha_flujo(
                    ax=graph_ax,
                    pos=pos,
                    origin=origin,
                    destination=destination,
                    color="#9A9A9A",
                    line_width=line_width,
                    zorder=9,
                    line_style="solid",
                    curvature=curvature,
                )
        else:
            for origin, destination in graph.edges():
                edge_key = self._normalizar_arista(origin, destination)
                x1, y1 = pos[origin]
                x2, y2 = pos[destination]

                origin_community = community_map.get(origin)
                destination_community = community_map.get(destination)
                between_communities = (
                    origin_community is not None
                    and destination_community is not None
                    and origin_community != destination_community
                )

                if (
                    active_merge
                    and origin in active_nodes
                    and destination in active_nodes
                    and between_communities
                ):
                    color = "#2E8B57"
                    line_width = 4.0
                    zorder = 18
                elif phase in {"communities", "final"} and between_communities:
                    color = "#C62828"
                    line_width = 3.2
                    zorder = 16
                elif edge_key in bridge_edges:
                    color = "#F28E2B"
                    line_width = 2.7
                    zorder = 15
                else:
                    color = "#B8B8B8"
                    line_width = 1.45
                    zorder = 10

                graph_ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    linewidth=line_width,
                    zorder=zorder,
                )

        normalized = self._normalizar_valores_analisis(scores)
        palette = self._paleta_comunidades_analisis()

        for node in graph.nodes():
            normalized_value = normalized.get(node, 0.30)
            node_size = 650 + 1500 * normalized_value
            face_color = self._color_por_puntuacion_analisis(normalized_value)
            edge_color = "#555555"
            line_width = 1.4

            if phase in {"communities", "final"}:
                community_index = community_map.get(node, 0)
                face_color = palette[community_index % len(palette)]
                if phase == "final":
                    node_size = 700 + 1700 * normalized_value
                else:
                    node_size = 820

            if node in active_nodes:
                edge_color = "#2E8B57"
                line_width = 3.0
                node_size += 160

            if node == current_node:
                face_color = "#E45756"
                edge_color = "#7A1D1D"
                line_width = 3.0
                node_size += 200

            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[node],
                node_size=node_size,
                node_color=face_color,
                edgecolors=edge_color,
                linewidths=line_width,
                ax=graph_ax,
            )
            collection.set_zorder(25)

        for node, (x, y) in pos.items():
            graph_ax.text(
                x,
                y,
                str(node),
                fontsize=9.5,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

            if node in scores:
                label = f"{scores[node]:.4f}"
                if phase in {"communities"}:
                    label = f"C{community_map.get(node, 0) + 1}"
                elif phase == "final":
                    label = (
                        f"PR={scores[node]:.3f} · "
                        f"C{community_map.get(node, 0) + 1}"
                    )

                graph_ax.text(
                    x,
                    y + 0.39,
                    label,
                    fontsize=6.7,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    color="#222222",
                    zorder=40,
                    bbox={
                        "boxstyle": "round,pad=0.16",
                        "fc": "white",
                        "ec": "#666666",
                        "alpha": 0.96,
                    },
                )

        graph_ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=graph_ax.transAxes,
            fontsize=8.8,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        status = state.get("status_text", "")
        graph_ax.text(
            0.99,
            0.985,
            status,
            transform=graph_ax.transAxes,
            fontsize=8.3,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.30",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

    def _dibujar_estado_centralidad(
        self,
        graph_ax,
        info_ax,
        structure_ax,
        graph,
        pagerank_graph,
        pos,
        state,
    ):
        """Dibuja un estado completo del análisis estructural."""

        self._dibujar_grafo_centralidad(
            graph_ax=graph_ax,
            graph=graph,
            pagerank_graph=pagerank_graph,
            pos=pos,
            state=state,
        )
        self._dibujar_tabla_centralidad(
            ax=info_ax,
            graph=graph,
            state=state,
        )
        self._dibujar_panel_inferior_centralidad(
            ax=structure_ax,
            state=state,
        )

    def animate_centrality_pagerank_communities(
        self,
        graph,
        pagerank_graph,
        pos,
        states,
        title="Centralidad, PageRank y comunidades",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima las métricas locales y globales y termina con comunidades.

        La imagen final compara:
        - grado;
        - cercanía;
        - intermediación;
        - autovector;
        - PageRank;
        - comunidad detectada.
        """

        if not states:
            raise ValueError(
                "La lista de estados de centralidad no puede estar vacía."
            )

        (
            fig,
            graph_ax,
            info_ax,
            structure_ax,
        ) = self._preparar_figura_centralidad(title)

        if final_image_path is not None:
            self._dibujar_estado_centralidad(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pagerank_graph=pagerank_graph,
                pos=pos,
                state=states[-1],
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            fig.savefig(
                final_image_path,
                dpi=200,
                bbox_inches="tight",
            )
            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_centralidad(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pagerank_graph=pagerank_graph,
                pos=pos,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_centralidad(
                graph_ax=graph_ax,
                info_ax=info_ax,
                structure_ax=structure_ax,
                graph=graph,
                pagerank_graph=pagerank_graph,
                pos=pos,
                state=states[frame_index],
            )
            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        plt.show()
        return self.animation
