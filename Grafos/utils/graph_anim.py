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
    - búsqueda en profundidad (DFS),
    - caminos mínimos con Dijkstra,
    - caminos mínimos con A*.

    Más adelante se podrá ampliar con animaciones para Prim,
    Kruskal y otros algoritmos.
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
