from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


class GraphAnimator:
    """
    Clase reutilizable para crear animaciones de algoritmos sobre grafos.

    Incluye métodos para:
    - búsqueda en anchura (BFS),
    - búsqueda en profundidad (DFS).

    Más adelante se podrá ampliar con animaciones para Dijkstra, A*,
    Prim, Kruskal y otros algoritmos.
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