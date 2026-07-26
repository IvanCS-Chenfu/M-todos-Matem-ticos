from math import cos, degrees, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, Rectangle


class GraphAnimator:
    """
    Clase reutilizable para crear animaciones de algoritmos sobre grafos.

    Incluye métodos para:
    - búsqueda en anchura (BFS),
    - búsqueda en profundidad (DFS),
    - caminos mínimos con Dijkstra,
    - caminos mínimos con A*,
    - navegación en grid con A* y replanificación dinámica,
    - planificación y ejecución de tareas robóticas,
    - restricciones relativas entre poses y transición a Graph SLAM,
    - variables, mediciones, predicciones y errores en SE(2),
    - funciones de coste y mínimos cuadrados,
    - incertidumbre, covarianza y matrices de información,
    - priors, libertad de gauge y anclaje de pose graphs,
    - optimización no lineal iterativa con Gauss-Newton y Levenberg-Marquardt,
    - jacobianos, Hessianas y estructura dispersa en pose graphs,
    - introducción a SLAM mediante trayectoria real y deriva de odometría,
    - Pose Graph SLAM 2D con prior, odometría, cierre y optimización,
    - loop closure con reconocimiento, verificación geométrica y robustez,
    - landmarks en SLAM con referencias conocidas y variables estimadas,
    - asociación de datos con gating, matching global y RANSAC,
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

    # ------------------------------------------------------------------
    # Navegación en grid con A* y replanificación dinámica
    # ------------------------------------------------------------------

    def _preparar_figura_navegacion_astar(self, title):
        """
        Crea la distribución visual del ejemplo de navegación.

        Distribución:
        - izquierda: leyenda, significado del grid y métricas;
        - derecha superior: mapa de ocupación;
        - derecha inferior: fases de planificación y replanificación.
        """

        fig = plt.figure(figsize=self.figsize)

        grid_spec = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.75, 4.25],
            height_ratios=[5.10, 1.30],
            wspace=0.055,
            hspace=0.08,
        )

        info_ax = fig.add_subplot(grid_spec[:, 0])
        map_ax = fig.add_subplot(grid_spec[0, 1])
        phase_ax = fig.add_subplot(grid_spec[1, 1])

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

        return fig, map_ax, info_ax, phase_ax

    def _dibujar_leyenda_navegacion_astar(self, ax):
        """Dibuja la leyenda estable de la navegación en grid."""

        elementos = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#888888",
                markersize=9,
                label="Casilla libre",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#222222",
                markeredgecolor="#111111",
                markersize=9,
                label="Obstáculo",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#FBE5A6",
                markeredgecolor="#8A6D1D",
                markersize=9,
                label="Abierta / frontera",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#B7D7F0",
                markeredgecolor="#1F4F73",
                markersize=9,
                label="Explorada / cerrada",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#D8C4E8",
                markeredgecolor="#5A316B",
                markersize=9,
                label="Casilla actual",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#B7E4C7",
                markeredgecolor="#2E8B57",
                markersize=9,
                label="Ruta activa",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#D0D0D0",
                markeredgecolor="#777777",
                markersize=9,
                label="Ruta anterior",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F28E2B",
                markeredgecolor="#8A4B08",
                markersize=9,
                label="Robot",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.905),
            fontsize=6.8,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.7,
            handletextpad=0.45,
            borderpad=0.5,
        )

    @staticmethod
    def _formatear_celda_navegacion(cell):
        """Convierte una celda ``(fila, columna)`` en texto compacto."""

        if cell is None:
            return "—"

        return f"({cell[0]}, {cell[1]})"

    def _dibujar_info_navegacion_astar(
        self,
        ax,
        occupancy_grid,
        state,
        start,
        goal,
    ):
        """Dibuja la explicación del modelo y las métricas del estado."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        rows = len(occupancy_grid)
        columns = len(occupancy_grid[0])
        obstacle_count = sum(
            value == 1
            for row in occupancy_grid
            for value in row
        )
        free_count = rows * columns - obstacle_count

        ax.text(
            0.50,
            0.985,
            "Navegación robótica con A*",
            fontsize=11.4,
            fontweight="bold",
            ha="center",
            va="top",
        )

        ax.text(
            0.50,
            0.947,
            "Mapa de ocupación convertido en un grafo",
            fontsize=8.0,
            ha="center",
            va="top",
            color="#444444",
        )

        self._dibujar_leyenda_navegacion_astar(ax)

        ax.text(
            0.50,
            0.655,
            (
                "Casilla blanca = vértice transitable\n"
                "Adyacencia N/S/E/O = arista de coste 1\n"
                "Casilla negra = vértice no disponible"
            ),
            fontsize=7.5,
            ha="center",
            va="center",
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.45",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.98,
            },
        )

        cards = [
            (
                "MAPA",
                (
                    f"Grid: {rows} × {columns}\n"
                    f"Libres: {free_count} · Obstáculos: {obstacle_count}\n"
                    f"Vértices: {state.get('graph_nodes', free_count)}\n"
                    f"Aristas: {state.get('graph_edges', '—')}"
                ),
                "#E5E5E5",
            ),
            (
                "A*",
                (
                    f"Actual: {self._formatear_celda_navegacion(state.get('current'))}\n"
                    f"g={state.get('current_g', '—')} · "
                    f"h={state.get('current_h', '—')} · "
                    f"f={state.get('current_f', '—')}\n"
                    f"Abiertas: {len(state.get('open_nodes', []))}\n"
                    f"Cerradas: {len(state.get('closed_nodes', []))}"
                ),
                "#FBE5A6",
            ),
            (
                "RUTAS",
                (
                    f"Inicial: {state.get('initial_path_cost', '—')} movimientos\n"
                    f"Replanificada: {state.get('replanned_path_cost', '—')} movimientos\n"
                    f"Recorridos: {state.get('travelled_cost', 0)}\n"
                    f"Coste total: {state.get('total_travel_cost', '—')}"
                ),
                "#B7E4C7",
            ),
        ]

        card_y = [0.490, 0.315, 0.140]
        card_height = 0.145

        for (title, body, color), y in zip(cards, card_y):
            rectangle = Rectangle(
                (0.08, y),
                0.84,
                card_height,
                facecolor=color,
                edgecolor="#666666",
                linewidth=1.15,
            )
            ax.add_patch(rectangle)

            ax.text(
                0.12,
                y + card_height - 0.025,
                title,
                fontsize=7.7,
                fontweight="bold",
                ha="left",
                va="top",
            )

            ax.text(
                0.12,
                y + card_height - 0.052,
                body,
                fontsize=6.8,
                ha="left",
                va="top",
                linespacing=1.30,
            )

        ax.text(
            0.50,
            0.055,
            (
                f"Inicio {self._formatear_celda_navegacion(start)}  ·  "
                f"Objetivo {self._formatear_celda_navegacion(goal)}\n"
                "Heurística Manhattan: h = |Δfila| + |Δcolumna|"
            ),
            fontsize=6.6,
            ha="center",
            va="center",
            color="#444444",
        )

    def _dibujar_fases_navegacion_astar(self, ax, state):
        """Dibuja la secuencia global de planificación y replanificación."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        phases = [
            ("1", "Mapa"),
            ("2", "A* inicial"),
            ("3", "Movimiento"),
            ("4", "Obstáculo"),
            ("5", "Replanificación"),
            ("6", "Objetivo"),
        ]

        phase_to_index = {
            "map": 0,
            "initial_search": 1,
            "initial_path": 1,
            "movement": 2,
            "obstacle": 3,
            "replanning": 4,
            "replanned_path": 4,
            "final_movement": 5,
            "finished": 5,
        }

        active_index = phase_to_index.get(state.get("phase"), 0)

        ax.text(
            0.02,
            0.88,
            "Secuencia de navegación",
            fontsize=11.2,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.text(
            0.98,
            0.88,
            (
                f"Robot: {self._formatear_celda_navegacion(state.get('robot'))}"
                f"  ·  Obstáculo dinámico: "
                f"{self._formatear_celda_navegacion(state.get('dynamic_obstacle'))}"
            ),
            fontsize=7.8,
            ha="right",
            va="center",
            color="#444444",
        )

        start_x = 0.045
        gap = 0.015
        cell_width = (0.91 - gap * (len(phases) - 1)) / len(phases)
        y = 0.42
        height = 0.29

        for index, (number, label) in enumerate(phases):
            x = start_x + index * (cell_width + gap)

            if index < active_index:
                face_color = "#B7D7F0"
                edge_color = "#1F4F73"
            elif index == active_index:
                face_color = "#F6C85F"
                edge_color = "#8A6D1D"
            else:
                face_color = "#E5E5E5"
                edge_color = "#888888"

            rectangle = Rectangle(
                (x, y),
                cell_width,
                height,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=2.0 if index == active_index else 1.2,
            )
            ax.add_patch(rectangle)

            ax.text(
                x + cell_width / 2,
                y + height * 0.65,
                number,
                fontsize=8.5,
                fontweight="bold",
                ha="center",
                va="center",
            )

            ax.text(
                x + cell_width / 2,
                y + height * 0.28,
                label,
                fontsize=6.7,
                ha="center",
                va="center",
            )

        ax.text(
            0.50,
            0.13,
            state.get("message", ""),
            fontsize=8.5,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.35",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.97,
            },
        )

    def _dibujar_estado_navegacion_astar(
        self,
        map_ax,
        info_ax,
        phase_ax,
        occupancy_grid,
        state,
        start,
        goal,
    ):
        """Dibuja un estado completo del ejemplo de navegación."""

        map_ax.clear()
        map_ax.axis("off")

        rows = len(occupancy_grid)
        columns = len(occupancy_grid[0])

        map_ax.set_xlim(0, columns)
        map_ax.set_ylim(rows, 0)
        map_ax.set_aspect("equal", adjustable="box")

        open_nodes = set(state.get("open_nodes", set()))
        closed_nodes = set(state.get("closed_nodes", set()))
        current = state.get("current")
        active_path = set(state.get("active_path", []))
        previous_path = set(state.get("previous_path", []))
        traversed_path = set(state.get("traversed_path", []))
        dynamic_obstacle = state.get("dynamic_obstacle")
        robot = state.get("robot")

        for row in range(rows):
            for column in range(columns):
                cell = (row, column)
                is_static_obstacle = occupancy_grid[row][column] == 1

                face_color = "white"
                edge_color = "#C9C9C9"
                line_width = 0.45

                if is_static_obstacle:
                    face_color = "#222222"
                    edge_color = "#111111"
                else:
                    if cell in closed_nodes:
                        face_color = "#B7D7F0"
                        edge_color = "#8DB6D6"
                    if cell in open_nodes:
                        face_color = "#FBE5A6"
                        edge_color = "#D5B65A"
                    if cell in previous_path:
                        face_color = "#D0D0D0"
                        edge_color = "#777777"
                        line_width = 0.8
                    if cell in active_path:
                        face_color = "#B7E4C7"
                        edge_color = "#2E8B57"
                        line_width = 0.9
                    if cell in traversed_path:
                        face_color = "#80CBC4"
                        edge_color = "#287A73"
                        line_width = 0.9
                    if cell == current:
                        face_color = "#D8C4E8"
                        edge_color = "#5A316B"
                        line_width = 1.5

                if cell == start:
                    face_color = "#81C784"
                    edge_color = "#2E6B32"
                    line_width = 1.3

                if cell == goal:
                    face_color = "#EF9A9A"
                    edge_color = "#8B1A1A"
                    line_width = 1.3

                if cell == dynamic_obstacle:
                    face_color = "#111111"
                    edge_color = "#C62828"
                    line_width = 2.2

                rectangle = Rectangle(
                    (column, row),
                    1,
                    1,
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=line_width,
                    zorder=10,
                )
                map_ax.add_patch(rectangle)

                if cell == dynamic_obstacle:
                    map_ax.plot(
                        [column + 0.18, column + 0.82],
                        [row + 0.18, row + 0.82],
                        color="#E45756",
                        linewidth=1.6,
                        zorder=20,
                    )
                    map_ax.plot(
                        [column + 0.82, column + 0.18],
                        [row + 0.18, row + 0.82],
                        color="#E45756",
                        linewidth=1.6,
                        zorder=20,
                    )

        start_row, start_column = start
        goal_row, goal_column = goal

        map_ax.text(
            start_column + 0.5,
            start_row + 0.5,
            "I",
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            zorder=30,
        )

        map_ax.text(
            goal_column + 0.5,
            goal_row + 0.5,
            "O",
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            zorder=30,
        )

        if robot is not None:
            robot_row, robot_column = robot
            map_ax.scatter(
                [robot_column + 0.5],
                [robot_row + 0.5],
                s=180,
                marker="o",
                facecolor="#F28E2B",
                edgecolor="#8A4B08",
                linewidth=1.8,
                zorder=40,
            )
            map_ax.text(
                robot_column + 0.5,
                robot_row + 0.5,
                "R",
                fontsize=6.8,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=45,
            )

        phase_titles = {
            "map": "Mapa de ocupación",
            "initial_search": "A* busca la ruta inicial",
            "initial_path": "Ruta inicial encontrada",
            "movement": "El robot sigue la ruta inicial",
            "obstacle": "Nuevo obstáculo detectado",
            "replanning": "A* replantea desde la posición actual",
            "replanned_path": "Ruta alternativa encontrada",
            "final_movement": "El robot sigue la ruta replanificada",
            "finished": "Objetivo alcanzado",
        }

        map_ax.set_title(
            phase_titles.get(state.get("phase"), "Navegación con A*"),
            fontsize=12,
            fontweight="bold",
            pad=7,
        )

        map_ax.text(
            0.99,
            0.99,
            (
                f"Expandidas: {state.get('expanded_count', 0)}"
                f"  ·  Frontera: {len(open_nodes)}"
                f"  ·  f=g+h"
            ),
            transform=map_ax.transAxes,
            fontsize=7.8,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.28",
                "fc": "white",
                "ec": "#888888",
                "alpha": 0.95,
            },
            zorder=60,
        )

        self._dibujar_info_navegacion_astar(
            ax=info_ax,
            occupancy_grid=occupancy_grid,
            state=state,
            start=start,
            goal=goal,
        )

        self._dibujar_fases_navegacion_astar(
            ax=phase_ax,
            state=state,
        )

    def animate_grid_astar_replanning(
        self,
        occupancy_grid,
        states,
        start,
        goal,
        title="Navegación en grid con A* y replanificación",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima la planificación de una ruta, el movimiento parcial del robot,
        la aparición de un obstáculo y una segunda ejecución de A*.

        La imagen final muestra:
        - mapa de ocupación;
        - trayectoria recorrida;
        - obstáculo dinámico;
        - ruta anterior atenuada;
        - ruta replanificada;
        - métricas de ambas búsquedas.
        """

        if not states:
            raise ValueError(
                "La lista de estados de navegación no puede estar vacía."
            )

        if not occupancy_grid or not occupancy_grid[0]:
            raise ValueError("El mapa de ocupación no puede estar vacío.")

        (
            fig,
            map_ax,
            info_ax,
            phase_ax,
        ) = self._preparar_figura_navegacion_astar(title)

        if final_image_path is not None:
            self._dibujar_estado_navegacion_astar(
                map_ax=map_ax,
                info_ax=info_ax,
                phase_ax=phase_ax,
                occupancy_grid=occupancy_grid,
                state=states[-1],
                start=start,
                goal=goal,
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
            self._dibujar_estado_navegacion_astar(
                map_ax=map_ax,
                info_ax=info_ax,
                phase_ax=phase_ax,
                occupancy_grid=occupancy_grid,
                state=states[0],
                start=start,
                goal=goal,
            )
            return []

        def update(frame_index):
            self._dibujar_estado_navegacion_astar(
                map_ax=map_ax,
                info_ax=info_ax,
                phase_ax=phase_ax,
                occupancy_grid=occupancy_grid,
                state=states[frame_index],
                start=start,
                goal=goal,
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
    # Elementos específicos de planificación de tareas robóticas
    # ------------------------------------------------------------------

    def _preparar_figura_planificacion_tareas(self, title):
        """
        Crea una figura coherente con las animaciones anteriores.

        Distribución:
        - izquierda: estado de la misión, tarea destacada y leyenda;
        - derecha superior: grafo dirigido de tareas;
        - derecha inferior: línea temporal agrupada por recursos.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.85, 4.15],
            height_ratios=[4.75, 1.75],
            wspace=0.07,
            hspace=0.09,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        graph_ax = fig.add_subplot(grid[0, 1])
        timeline_ax = fig.add_subplot(grid[1, 1])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.99,
            top=0.93,
            bottom=0.055,
        )

        return fig, graph_ax, info_ax, timeline_ax

    def _dibujar_leyenda_planificacion_tareas(self, ax):
        """Dibuja la leyenda de estados y dependencias."""

        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#D9D9D9",
                markeredgecolor="#666666",
                markersize=8,
                label="Pendiente",
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
                markerfacecolor="#8E5EA2",
                markeredgecolor="#5A316B",
                markersize=8,
                label="En ejecución",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4C9ED9",
                markeredgecolor="#1F4F73",
                markersize=8,
                label="Completada",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F28E2B",
                markeredgecolor="#8A4B08",
                markersize=8,
                label="Fallida",
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
                color="#B8B8B8",
                linewidth=2,
                linestyle="dashed",
                label="Rama no activada",
            ),
            Line2D(
                [0],
                [0],
                color="#C62828",
                linewidth=4,
                label="Camino crítico final",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            fontsize=6.6,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.65,
            handlelength=1.9,
            borderpad=0.50,
        )

    @staticmethod
    def _abreviar_lista_tareas(values, maximum=5):
        """Convierte una colección de tareas en una línea compacta."""

        values = list(values)

        if not values:
            return "—"

        visible = values[:maximum]
        text = ", ".join(map(str, visible))

        if len(values) > maximum:
            text += f"  +{len(values) - maximum}"

        return text

    def _dibujar_panel_planificacion_tareas(
        self,
        ax,
        graph,
        state,
    ):
        """Dibuja métricas, listas y detalles de la tarea destacada."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        statuses = dict(state.get("statuses", {}))
        available = sorted(state.get("available", set()))
        running = sorted(state.get("running", set()))
        completed = set(state.get("completed", set()))
        failed = set(state.get("failed", set()))
        focus_task = state.get("focus_task")
        time = state.get("time", 0)
        phase = state.get("phase", "execution")

        ax.text(
            0.50,
            0.985,
            "Estado de la misión",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="top",
        )

        phase_labels = {
            "validation": "Validación del DAG",
            "ready": "Tareas desbloqueadas",
            "start": "Inicio",
            "running": "Ejecución",
            "completed": "Tarea completada",
            "failure": "Fallo y recuperación",
            "finished": "Finalización",
            "summary": "Resumen final",
        }

        ax.text(
            0.50,
            0.945,
            phase_labels.get(phase, phase),
            fontsize=8.3,
            ha="center",
            va="top",
            color="#444444",
        )

        summary_rectangle = Rectangle(
            (0.07, 0.815),
            0.86,
            0.095,
            facecolor="white",
            edgecolor="#777777",
            linewidth=1.3,
        )
        ax.add_patch(summary_rectangle)

        ax.text(
            0.13,
            0.872,
            f"t = {time}",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.text(
            0.43,
            0.872,
            f"Disponibles: {len(available)}",
            fontsize=7.4,
            ha="left",
            va="center",
        )

        ax.text(
            0.43,
            0.835,
            f"En ejecución: {len(running)}",
            fontsize=7.4,
            ha="left",
            va="center",
        )

        ax.text(
            0.73,
            0.872,
            f"Completadas: {len(completed)}",
            fontsize=7.4,
            ha="left",
            va="center",
        )

        ax.text(
            0.73,
            0.835,
            f"Fallidas: {len(failed)}",
            fontsize=7.4,
            ha="left",
            va="center",
        )

        if focus_task is None or focus_task not in graph:
            focus_name = "Sin tarea destacada"
            focus_category = "—"
            focus_resource = "—"
            focus_duration = "—"
            focus_status = "—"
            focus_remaining = "—"
        else:
            node_data = graph.nodes[focus_task]
            focus_name = node_data.get("name", focus_task)
            focus_category = node_data.get("category", "—")
            focus_resource = node_data.get("resource", "—")
            focus_duration = node_data.get("duration", "—")
            focus_status = statuses.get(focus_task, "—")
            focus_remaining = state.get("remaining", {}).get(
                focus_task,
                "—",
            )

        focus_rectangle = Rectangle(
            (0.07, 0.610),
            0.86,
            0.170,
            facecolor="#F7F7F7",
            edgecolor="#777777",
            linewidth=1.3,
        )
        ax.add_patch(focus_rectangle)

        ax.text(
            0.11,
            0.752,
            "Tarea destacada",
            fontsize=8.2,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax.text(
            0.11,
            0.713,
            (
                f"{focus_task or '—'} · {focus_name}"
                if focus_task is not None
                else focus_name
            ),
            fontsize=7.6,
            fontweight="bold",
            ha="left",
            va="center",
            wrap=True,
        )

        ax.text(
            0.11,
            0.672,
            f"Estado: {focus_status}",
            fontsize=7.0,
            ha="left",
            va="center",
        )

        ax.text(
            0.56,
            0.672,
            f"Recurso: {focus_resource}",
            fontsize=7.0,
            ha="left",
            va="center",
        )

        ax.text(
            0.11,
            0.632,
            f"Categoría: {focus_category}",
            fontsize=7.0,
            ha="left",
            va="center",
        )

        ax.text(
            0.56,
            0.632,
            f"Duración/restante: {focus_duration}/{focus_remaining}",
            fontsize=6.9,
            ha="left",
            va="center",
        )

        list_specs = [
            (
                "Disponibles",
                available,
                0.545,
                "#FBE5A6",
                "#8A6D1D",
            ),
            (
                "En ejecución",
                running,
                0.455,
                "#E8D7F1",
                "#5A316B",
            ),
            (
                "Últimas finalizadas",
                list(state.get("execution_order", []))[-5:],
                0.365,
                "#DDECF7",
                "#1F4F73",
            ),
        ]

        for label, values, y, face_color, edge_color in list_specs:
            rectangle = Rectangle(
                (0.07, y - 0.055),
                0.86,
                0.075,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.1,
            )
            ax.add_patch(rectangle)

            ax.text(
                0.10,
                y,
                f"{label}: ",
                fontsize=6.9,
                fontweight="bold",
                ha="left",
                va="center",
            )

            ax.text(
                0.34,
                y,
                self._abreviar_lista_tareas(values),
                fontsize=6.8,
                ha="left",
                va="center",
            )

        total_duration = state.get("total_duration")
        critical_path = list(state.get("critical_path", []))

        if total_duration is not None:
            result_text = (
                f"Duración total: {total_duration} unidades\n"
                f"Camino crítico activo: {len(critical_path)} tareas\n"
                "Incluye la rama de recuperación"
            )
        else:
            result_text = (
                f"Tareas: {graph.number_of_nodes()}\n"
                f"Dependencias: {graph.number_of_edges()}\n"
                "DAG válido: sí"
            )

        ax.text(
            0.50,
            0.245,
            result_text,
            fontsize=7.1,
            ha="center",
            va="center",
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.40",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.98,
            },
        )

        self._dibujar_leyenda_planificacion_tareas(ax)

    def _dibujar_flecha_planificacion_tareas(
        self,
        ax,
        pos,
        origin,
        destination,
        color,
        line_width,
        line_style,
        zorder,
        curvature=0.0,
    ):
        """Dibuja una dependencia dirigida evitando tapar los nodos."""

        x1, y1 = pos[origin]
        x2, y2 = pos[destination]

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=line_width,
            linestyle=line_style,
            color=color,
            shrinkA=17,
            shrinkB=17,
            connectionstyle=f"arc3,rad={curvature}",
            zorder=zorder,
        )
        ax.add_patch(arrow)

    def _dibujar_etiqueta_dependencia_tarea(
        self,
        ax,
        pos,
        origin,
        destination,
        label,
        curvature=0.0,
    ):
        """Añade la etiqueta éxito/fallo de una arista condicional."""

        if not label:
            return

        x1, y1 = pos[origin]
        x2, y2 = pos[destination]

        middle_x = (x1 + x2) / 2
        middle_y = (y1 + y2) / 2

        dx = x2 - x1
        dy = y2 - y1
        length = max((dx**2 + dy**2) ** 0.5, 1e-9)

        offset_x = -dy / length * 0.15
        offset_y = dx / length * 0.15

        if curvature != 0:
            offset_y += 0.42 if curvature < 0 else -0.42

        ax.text(
            middle_x + offset_x,
            middle_y + offset_y,
            label,
            fontsize=6.6,
            fontweight="bold",
            ha="center",
            va="center",
            color="#333333",
            zorder=35,
            bbox={
                "boxstyle": "round,pad=0.16",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
        )

    def _dibujar_grafo_planificacion_tareas(
        self,
        ax,
        graph,
        pos,
        state,
    ):
        """Dibuja el DAG y el estado dinámico de todas sus tareas."""

        ax.clear()
        ax.axis("off")

        limits = self._calcular_limites(
            pos,
            margin_x=0.85,
            margin_y=0.90,
        )

        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("auto", adjustable="box")

        statuses = dict(state.get("statuses", {}))
        satisfied_edges = set(state.get("satisfied_edges", set()))
        triggered_edges = set(
            state.get("triggered_condition_edges", set())
        )
        inactive_edges = set(
            state.get("inactive_condition_edges", set())
        )
        recent_edges = set(state.get("recent_edges", set()))
        critical_edges = set(state.get("critical_edges", set()))
        critical_nodes = set(state.get("critical_nodes", set()))
        focus_task = state.get("focus_task")
        phase = state.get("phase", "execution")

        curvature_map = {
            ("VAG1", "PLT"): -0.20,
            ("VAG1", "REC1"): 0.10,
            ("VAG2", "PLT"): 0.12,
        }

        for origin, destination, data in graph.edges(data=True):
            edge = (origin, destination)
            condition = data.get("condition", "siempre")
            curvature = curvature_map.get(edge, 0.0)

            if phase == "summary" and edge in critical_edges:
                color = "#C62828"
                line_width = 4.0
                line_style = "solid"
                zorder = 24
            elif edge in recent_edges:
                color = "#2E8B57"
                line_width = 4.2
                line_style = "solid"
                zorder = 23
            elif edge in inactive_edges:
                color = "#B8B8B8"
                line_width = 1.7
                line_style = "dashed"
                zorder = 9
            elif edge in triggered_edges:
                color = "#2E8B57"
                line_width = 3.2
                line_style = "solid"
                zorder = 18
            elif edge in satisfied_edges:
                color = "#2E8B57"
                line_width = 2.6
                line_style = "solid"
                zorder = 15
            elif condition != "siempre":
                color = "#AFAFAF"
                line_width = 1.6
                line_style = "dashed"
                zorder = 10
            else:
                color = "#B8B8B8"
                line_width = 1.45
                line_style = "solid"
                zorder = 10

            self._dibujar_flecha_planificacion_tareas(
                ax=ax,
                pos=pos,
                origin=origin,
                destination=destination,
                color=color,
                line_width=line_width,
                line_style=line_style,
                zorder=zorder,
                curvature=curvature,
            )

            self._dibujar_etiqueta_dependencia_tarea(
                ax=ax,
                pos=pos,
                origin=origin,
                destination=destination,
                label=data.get("label", ""),
                curvature=curvature,
            )

        status_style = {
            "pendiente": ("#D9D9D9", "#666666"),
            "disponible": ("#F6C85F", "#8A6D1D"),
            "en_ejecucion": ("#8E5EA2", "#5A316B"),
            "completada": ("#4C9ED9", "#1F4F73"),
            "fallida": ("#F28E2B", "#8A4B08"),
        }

        for node in graph.nodes():
            status = statuses.get(node, "pendiente")
            face_color, edge_color = status_style.get(
                status,
                ("#D9D9D9", "#666666"),
            )

            node_size = 720
            line_width = 1.5

            if node == focus_task:
                edge_color = "#C62828"
                node_size = 900
                line_width = 2.8
            elif phase == "summary" and node in critical_nodes:
                edge_color = "#C62828"
                node_size = 790
                line_width = 2.6

            collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[node],
                node_size=node_size,
                node_color=face_color,
                edgecolors=edge_color,
                linewidths=line_width,
                ax=ax,
            )
            collection.set_zorder(28)

        remaining = dict(state.get("remaining", {}))

        for node, (x, y) in pos.items():
            ax.text(
                x,
                y,
                str(node),
                fontsize=7.3,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                zorder=35,
            )

            ax.text(
                x,
                y - 0.42,
                graph.nodes[node].get("short_name", node),
                fontsize=5.8,
                ha="center",
                va="top",
                color="#222222",
                zorder=36,
            )

            duration = graph.nodes[node].get("duration", 0)
            status = statuses.get(node, "pendiente")

            if status == "en_ejecucion":
                upper_label = (
                    f"rest={remaining.get(node, duration)}"
                )
            else:
                upper_label = f"d={duration}"

            ax.text(
                x,
                y + 0.37,
                upper_label,
                fontsize=5.9,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#222222",
                zorder=38,
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.96,
                },
            )

        ax.text(
            0.50,
            0.012,
            state.get("message", ""),
            transform=ax.transAxes,
            fontsize=8.6,
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

        status_text = (
            f"t={state.get('time', 0)}"
            f"  ·  disponibles={len(state.get('available', set()))}"
            f"  ·  ejecutándose={len(state.get('running', set()))}"
            f"  ·  completadas={len(state.get('completed', set()))}"
            f"  ·  fallidas={len(state.get('failed', set()))}"
        )

        ax.text(
            0.995,
            0.985,
            status_text,
            transform=ax.transAxes,
            fontsize=7.8,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.28",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.96,
            },
            zorder=50,
        )

        ax.text(
            0.005,
            0.985,
            "Vértice = tarea  ·  Flecha = dependencia",
            transform=ax.transAxes,
            fontsize=7.5,
            ha="left",
            va="top",
            color="#444444",
            zorder=50,
        )

    def _dibujar_timeline_planificacion_tareas(
        self,
        ax,
        graph,
        state,
    ):
        """Dibuja una línea temporal compacta agrupada por recursos."""

        ax.clear()

        start_times = dict(state.get("start_times", {}))
        end_times = dict(state.get("end_times", {}))
        statuses = dict(state.get("statuses", {}))
        current_time = state.get("time", 0)
        total_duration = state.get("total_duration")

        resource_order = list(
            graph.graph.get(
                "resource_order",
                sorted(
                    {
                        data.get("resource", "recurso")
                        for _, data in graph.nodes(data=True)
                    }
                ),
            )
        )
        resource_labels = dict(
            graph.graph.get("resource_labels", {})
        )

        estimated_end = max(
            [
                start_times.get(node, 0)
                + graph.nodes[node].get("duration", 0)
                for node in start_times
            ]
            + [current_time, 1]
        )

        x_max = max(total_duration or 0, estimated_end, 1)
        x_margin = max(1.0, x_max * 0.025)

        ax.set_xlim(-x_margin, x_max + x_margin)
        ax.set_ylim(-0.65, len(resource_order) - 0.25)

        ax.set_yticks(range(len(resource_order)))
        ax.set_yticklabels(
            [
                resource_labels.get(resource, resource)
                for resource in resource_order
            ],
            fontsize=7.2,
        )

        tick_step = 1 if x_max <= 18 else 2 if x_max <= 36 else 5
        ax.set_xticks(range(0, int(x_max) + 1, tick_step))
        ax.tick_params(axis="x", labelsize=6.8)
        ax.grid(axis="x", alpha=0.20, linewidth=0.7)
        ax.set_xlabel("Tiempo simulado", fontsize=7.5)
        ax.set_title(
            "Uso de recursos y paralelismo",
            fontsize=10.2,
            fontweight="bold",
            pad=4,
        )

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        resource_to_y = {
            resource: index
            for index, resource in enumerate(resource_order)
        }

        bar_colors = {
            "en_ejecucion": "#8E5EA2",
            "completada": "#4C9ED9",
            "fallida": "#F28E2B",
            "disponible": "#F6C85F",
        }

        for node, start_time in sorted(
            start_times.items(),
            key=lambda item: (item[1], item[0]),
        ):
            duration = graph.nodes[node].get("duration", 0)
            resource = graph.nodes[node].get("resource")

            if duration <= 0 or resource not in resource_to_y:
                continue

            y = resource_to_y[resource]
            planned_end = start_time + duration
            actual_end = end_times.get(node, planned_end)
            width = max(actual_end - start_time, 0.15)
            status = statuses.get(node, "en_ejecucion")

            rectangle = Rectangle(
                (start_time, y - 0.28),
                width,
                0.56,
                facecolor=bar_colors.get(status, "#D9D9D9"),
                edgecolor="#555555",
                linewidth=1.0,
                alpha=0.92,
            )
            ax.add_patch(rectangle)

            ax.text(
                start_time + width / 2,
                y,
                node,
                fontsize=6.0,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
            )

        ax.axvline(
            current_time,
            color="#C62828",
            linewidth=2.0,
            linestyle="--",
            zorder=15,
        )

        ax.text(
            current_time,
            len(resource_order) - 0.38,
            f"t={current_time}",
            fontsize=6.7,
            fontweight="bold",
            ha="center",
            va="top",
            color="#C62828",
            bbox={
                "boxstyle": "round,pad=0.16",
                "fc": "white",
                "ec": "#C62828",
                "alpha": 0.95,
            },
        )

    def _dibujar_estado_planificacion_tareas(
        self,
        graph_ax,
        info_ax,
        timeline_ax,
        graph,
        pos,
        state,
    ):
        """Dibuja un estado completo de la planificación de tareas."""

        self._dibujar_grafo_planificacion_tareas(
            ax=graph_ax,
            graph=graph,
            pos=pos,
            state=state,
        )

        self._dibujar_panel_planificacion_tareas(
            ax=info_ax,
            graph=graph,
            state=state,
        )

        self._dibujar_timeline_planificacion_tareas(
            ax=timeline_ax,
            graph=graph,
            state=state,
        )

    def animate_robot_task_planning(
        self,
        graph,
        pos,
        states,
        title="Planificación de tareas de una misión robótica",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima la ejecución de un grafo de tareas robóticas.

        La demostración puede mostrar:
        - validación del DAG;
        - tareas disponibles y ejecución paralela;
        - uso exclusivo de recursos;
        - finalización y desbloqueo de sucesores;
        - fallo controlado y rama de recuperación;
        - camino crítico activo y duración total.
        """

        if not states:
            raise ValueError(
                "La lista de estados de planificación no puede estar vacía."
            )

        if not graph.is_directed():
            raise ValueError("El grafo de tareas debe ser dirigido.")

        (
            fig,
            graph_ax,
            info_ax,
            timeline_ax,
        ) = self._preparar_figura_planificacion_tareas(title)

        if final_image_path is not None:
            self._dibujar_estado_planificacion_tareas(
                graph_ax=graph_ax,
                info_ax=info_ax,
                timeline_ax=timeline_ax,
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

            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_planificacion_tareas(
                graph_ax=graph_ax,
                info_ax=info_ax,
                timeline_ax=timeline_ax,
                graph=graph,
                pos=pos,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_planificacion_tareas(
                graph_ax=graph_ax,
                info_ax=info_ax,
                timeline_ax=timeline_ax,
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
    # Restricción básica entre dos poses SE(2)
    # ------------------------------------------------------------------

    def _preparar_figura_restriccion_pose(self, title):
        """
        Crea una distribución comparable a las animaciones anteriores.

        Distribución:
        - izquierda: explicación y leyenda;
        - centro superior: interpretación geométrica;
        - derecha superior: grafo de restricciones;
        - zona inferior: medición, predicción, residuo y coste.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[1.62, 4.15, 2.35],
            height_ratios=[4.75, 1.65],
            wspace=0.09,
            hspace=0.12,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        geometry_ax = fig.add_subplot(grid[0, 1])
        graph_ax = fig.add_subplot(grid[0, 2])
        calculation_ax = fig.add_subplot(grid[1, 1:])

        fig.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.925,
            bottom=0.045,
        )

        return fig, geometry_ax, graph_ax, info_ax, calculation_ax

    @staticmethod
    def _formatear_pose_restriccion(pose, decimals=2):
        """Formatea (x, y, theta) usando metros y grados."""

        if pose is None:
            return "—"

        return (
            f"({pose[0]:.{decimals}f} m, "
            f"{pose[1]:.{decimals}f} m, "
            f"{degrees(pose[2]):.{decimals}f}°)"
        )

    @staticmethod
    def _formatear_vector_restriccion(vector, decimals=3):
        """Formatea un vector de residuo con el ángulo en grados."""

        if vector is None:
            return "—"

        return (
            f"({vector[0]:.{decimals}f}, "
            f"{vector[1]:.{decimals}f}, "
            f"{degrees(vector[2]):.{decimals}f}°)"
        )

    @staticmethod
    def _titulo_fase_restriccion(phase):
        """Traduce la clave de fase a un título breve."""

        titles = {
            "normal_graph": "1. Arista como conexión",
            "constraint_graph": "2. Arista como restricción",
            "pose_x0": "3. Pose de referencia x0",
            "pose_x1": "4. Estimación actual x1",
            "measurement": "5. Medición relativa z01",
            "prediction": "6. Predicción de las poses",
            "translation_residual": "7. Error de traslación",
            "angular_residual": "8. Error angular",
            "uncertainty": "9. Incertidumbre e información",
            "cost": "10. Coste de la arista",
            "prior": "11. Prior y libertad gauge",
            "correction": "12. Corrección de x1",
            "comparison": "13. Comparación final",
            "pose_graph_preview": "14. De dos poses a Pose Graph",
            "summary": "15. Idea final",
        }

        return titles.get(phase, str(phase))

    def _dibujar_leyenda_restriccion_pose(self, ax):
        """Dibuja una leyenda compacta en el panel izquierdo."""

        elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4C9ED9",
                markeredgecolor="#1F4F73",
                markersize=8,
                label="x0: pose fija",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F28E2B",
                markeredgecolor="#8A4B08",
                markersize=8,
                label="x1: estimación actual",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#2E8B57",
                markersize=8,
                label="x1*: pose esperada",
            ),
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=3,
                label="Medición z01",
            ),
            Line2D(
                [0],
                [0],
                color="#8E5EA2",
                linewidth=3,
                label="Predicción z_hat01",
            ),
            Line2D(
                [0],
                [0],
                color="#D62728",
                linewidth=3,
                label="Residuo",
            ),
            Line2D(
                [0],
                [0],
                color="#777777",
                linewidth=2,
                linestyle="dashed",
                label="Estado inicial / histórico",
            ),
        ]

        ax.legend(
            handles=elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            fontsize=6.8,
            framealpha=0.97,
            ncol=1,
            handlelength=2.4,
            borderpad=0.55,
            labelspacing=0.58,
        )

    def _dibujar_pose_restriccion(
        self,
        ax,
        pose,
        label,
        face_color,
        edge_color,
        *,
        alpha=1.0,
        zorder=30,
        radius=0.105,
        label_offset=(0.0, -0.28),
        line_style="solid",
    ):
        """Dibuja una pose como origen local con ejes x e y."""

        x, y, theta = pose

        body = Ellipse(
            (x, y),
            width=2.0 * radius,
            height=2.0 * radius,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=2.1,
            linestyle=line_style,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(body)

        axis_length = 0.42
        side_length = 0.28

        x_end = (
            x + axis_length * cos(theta),
            y + axis_length * sin(theta),
        )
        y_end = (
            x - side_length * sin(theta),
            y + side_length * cos(theta),
        )

        forward_arrow = FancyArrowPatch(
            (x, y),
            x_end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=2.0,
            linestyle=line_style,
            color=edge_color,
            alpha=alpha,
            zorder=zorder + 1,
        )
        ax.add_patch(forward_arrow)

        ax.plot(
            [x, y_end[0]],
            [y, y_end[1]],
            color=edge_color,
            linewidth=1.45,
            linestyle=line_style,
            alpha=alpha,
            zorder=zorder,
        )

        ax.text(
            x + label_offset[0],
            y + label_offset[1],
            label,
            fontsize=8.1,
            fontweight="bold",
            ha="center",
            va="top" if label_offset[1] < 0 else "bottom",
            color=edge_color,
            alpha=alpha,
            zorder=zorder + 3,
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": edge_color,
                "alpha": 0.92 * alpha,
            },
        )

    def _dibujar_flecha_transformacion_restriccion(
        self,
        ax,
        start_pose,
        end_pose,
        color,
        label,
        *,
        curvature=0.0,
        line_width=3.0,
        line_style="solid",
        label_offset=(0.0, 0.0),
        alpha=1.0,
        zorder=18,
    ):
        """Dibuja una transformación entre posiciones de dos poses."""

        start = (start_pose[0], start_pose[1])
        end = (end_pose[0], end_pose[1])

        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=line_width,
            linestyle=line_style,
            color=color,
            alpha=alpha,
            shrinkA=10,
            shrinkB=10,
            connectionstyle=f"arc3,rad={curvature}",
            zorder=zorder,
        )
        ax.add_patch(arrow)

        middle_x = (start[0] + end[0]) / 2 + label_offset[0]
        middle_y = (start[1] + end[1]) / 2 + label_offset[1]

        ax.text(
            middle_x,
            middle_y,
            label,
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="center",
            color=color,
            alpha=alpha,
            zorder=zorder + 4,
            bbox={
                "boxstyle": "round,pad=0.20",
                "fc": "white",
                "ec": color,
                "alpha": 0.93,
            },
        )

    def _dibujar_elipse_incertidumbre_restriccion(
        self,
        ax,
        expected_pose,
        sigmas,
    ):
        """Dibuja una elipse de 2 sigma alrededor de la pose esperada."""

        ellipse = Ellipse(
            (expected_pose[0], expected_pose[1]),
            width=4.0 * sigmas[0],
            height=4.0 * sigmas[1],
            angle=degrees(expected_pose[2]),
            facecolor="#B7E4C7",
            edgecolor="#2E8B57",
            linewidth=1.8,
            linestyle="dashed",
            alpha=0.23,
            zorder=8,
        )
        ax.add_patch(ellipse)

        ax.text(
            expected_pose[0] + 0.10,
            expected_pose[1] + 0.68,
            "incertidumbre 2σ",
            fontsize=7.2,
            fontweight="bold",
            color="#2E8B57",
            ha="center",
            va="bottom",
            zorder=20,
        )

    def _dibujar_prior_restriccion(self, ax, pose_x0):
        """Dibuja un prior como una caja conectada con x0."""

        x, y, _ = pose_x0
        box_x = x - 0.35
        box_y = y + 0.70

        rectangle = Rectangle(
            (box_x, box_y),
            0.70,
            0.30,
            facecolor="#E5E5E5",
            edgecolor="#555555",
            linewidth=1.6,
            zorder=35,
        )
        ax.add_patch(rectangle)

        ax.text(
            x,
            box_y + 0.15,
            "PRIOR\nx0 fija",
            fontsize=7.1,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=38,
        )

        arrow = FancyArrowPatch(
            (x, box_y),
            (x, y + 0.12),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.7,
            color="#555555",
            zorder=34,
        )
        ax.add_patch(arrow)

    def _dibujar_panel_informacion_restriccion(self, ax, state):
        """Muestra el concepto activo, la ecuación principal y la leyenda."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        phase = state.get("phase", "")
        step = state.get("step", 0)
        total_steps = state.get("total_steps", 0)

        ax.text(
            0.50,
            0.985,
            "De conexión a restricción",
            fontsize=12.0,
            fontweight="bold",
            ha="center",
            va="top",
        )

        ax.text(
            0.50,
            0.945,
            f"Estado {step} de {total_steps}",
            fontsize=8.0,
            ha="center",
            va="top",
            color="#444444",
        )

        ax.text(
            0.50,
            0.885,
            self._titulo_fase_restriccion(phase),
            fontsize=10.1,
            fontweight="bold",
            ha="center",
            va="top",
            color="#1F4F73",
            bbox={
                "boxstyle": "round,pad=0.35",
                "fc": "#EAF3F8",
                "ec": "#1F4F73",
                "alpha": 0.98,
            },
        )

        equation_map = {
            "normal_graph": r"$\mathrm{arista}=\mathrm{conexión}$",
            "constraint_graph": r"$\mathrm{arista}=(z_{01},\Sigma_{01},\Omega_{01})$",
            "pose_x0": r"$x_0=(x,y,\theta)$",
            "pose_x1": r"$x_1=(x,y,\theta)$",
            "measurement": r"$x_1^*=x_0\oplus z_{01}$",
            "prediction": r"$\hat z_{01}=x_0^{-1}\oplus x_1$",
            "translation_residual": r"$e_{01}=z_{01}^{-1}\oplus\hat z_{01}$",
            "angular_residual": r"$e_\theta\in[-\pi,\pi)$",
            "uncertainty": r"$\Omega_{01}=\Sigma_{01}^{-1}$",
            "cost": r"$E_{01}=e_{01}^{T}\Omega_{01}e_{01}$",
            "prior": r"$x_0=\bar{x}_0$",
            "correction": r"$x_1\longrightarrow x_1^*$",
            "comparison": r"$e_{01}\approx0,\quad E_{01}\approx0$",
            "pose_graph_preview": r"$F(x)=\sum e_{ij}^{T}\Omega_{ij}e_{ij}$",
            "summary": r"$\mathrm{arista}=\mathrm{medición}+\mathrm{incertidumbre}+\mathrm{error}$",
        }

        ax.text(
            0.50,
            0.785,
            equation_map.get(phase, ""),
            fontsize=12.0,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#888888",
                "alpha": 0.98,
            },
        )

        pose_x0 = state.get("pose_x0")
        pose_x1 = state.get("pose_x1")
        expected = state.get("pose_x1_expected")

        values_text = (
            f"x0\n{self._formatear_pose_restriccion(pose_x0)}\n\n"
            f"x1 actual\n{self._formatear_pose_restriccion(pose_x1)}\n\n"
            f"x1 esperada\n{self._formatear_pose_restriccion(expected)}"
        )

        ax.text(
            0.50,
            0.610,
            values_text,
            fontsize=7.8,
            ha="center",
            va="center",
            linespacing=1.35,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.45",
                "fc": "#FAFAFA",
                "ec": "#999999",
                "alpha": 0.98,
            },
        )

        focus = state.get("focus")
        focus_texts = {
            "connection": "La línea solo expresa conectividad.",
            "measurement": "El sensor define cómo deberían relacionarse las poses.",
            "x0": "x0 proporciona el sistema local de la medición.",
            "x1": "x1 es una variable que puede modificarse.",
            "prediction": "Las poses actuales generan una medición predicha.",
            "translation_error": "La posición actual no coincide con la esperada.",
            "angular_error": "La orientación también forma parte del residuo.",
            "uncertainty": "La precisión determina cuánto pesa cada error.",
            "cost": "El coste convierte el residuo en un escalar.",
            "prior": "El prior elimina la libertad de desplazar todo el grafo.",
            "correction": "Al mover x1 disminuyen el residuo y el coste.",
            "comparison": "La medición y la predicción ya coinciden.",
            "future_graph": "Graph SLAM combina muchas restricciones locales.",
            "summary": "Cada arista aporta un término al coste global.",
        }

        ax.text(
            0.50,
            0.355,
            focus_texts.get(focus, ""),
            fontsize=8.0,
            ha="center",
            va="center",
            wrap=True,
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.40",
                "fc": "#FFF8E7",
                "ec": "#C69C36",
                "alpha": 0.98,
            },
        )

        self._dibujar_leyenda_restriccion_pose(ax)

    def _dibujar_geometria_restriccion_pose(self, ax, state):
        """Dibuja las poses, las transformaciones y el residuo geométrico."""

        ax.clear()
        ax.set_xlim(0.30, 5.05)
        ax.set_ylim(0.20, 3.00)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.grid(True, linewidth=0.55, alpha=0.28)
        ax.tick_params(labelsize=7)
        ax.set_title(
            "Interpretación geométrica de la restricción",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )

        if not state.get("show_geometry", False):
            ax.text(
                0.50,
                0.53,
                "La misma arista se interpretará\ncomo una relación geométrica entre poses",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                color="#555555",
                bbox={
                    "boxstyle": "round,pad=0.55",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.98,
                },
            )
            ax.text(
                0.50,
                0.36,
                "posición + orientación + sistema de referencia",
                transform=ax.transAxes,
                fontsize=8.6,
                ha="center",
                va="center",
                color="#666666",
            )
            return

        pose_x0 = state["pose_x0"]
        pose_x1 = state["pose_x1"]
        pose_initial = state["pose_x1_initial"]
        expected = state["pose_x1_expected"]
        sigmas = state["sigmas"]
        alpha = state.get("correction_alpha", 0.0)

        if state.get("show_uncertainty", False):
            self._dibujar_elipse_incertidumbre_restriccion(
                ax=ax,
                expected_pose=expected,
                sigmas=sigmas,
            )

        if state.get("show_measurement", False):
            self._dibujar_flecha_transformacion_restriccion(
                ax=ax,
                start_pose=pose_x0,
                end_pose=expected,
                color="#2E8B57",
                label=r"medición $z_{01}$",
                curvature=0.08,
                line_width=3.2,
                label_offset=(0.0, 0.22),
                zorder=15,
            )

        if state.get("show_prediction", False):
            self._dibujar_flecha_transformacion_restriccion(
                ax=ax,
                start_pose=pose_x0,
                end_pose=pose_x1,
                color="#8E5EA2",
                label=r"predicción $\hat z_{01}$",
                curvature=-0.08,
                line_width=3.0,
                label_offset=(0.0, -0.20),
                zorder=16,
            )

        if state.get("show_comparison", False) or (
            state.get("phase") == "correction" and alpha > 0.02
        ):
            self._dibujar_pose_restriccion(
                ax=ax,
                pose=pose_initial,
                label="x1 inicial",
                face_color="#F7C6C7",
                edge_color="#8B3A3A",
                alpha=0.42,
                zorder=21,
                label_offset=(0.0, 0.33),
                line_style="dashed",
            )

        if state.get("show_expected_pose", False):
            self._dibujar_pose_restriccion(
                ax=ax,
                pose=expected,
                label="x1* esperada",
                face_color="white",
                edge_color="#2E8B57",
                alpha=0.95,
                zorder=24,
                label_offset=(-0.22, 0.38),
                line_style="dashed",
            )

        if state.get("show_pose_x0", False):
            self._dibujar_pose_restriccion(
                ax=ax,
                pose=pose_x0,
                label="x0 fija",
                face_color="#4C9ED9",
                edge_color="#1F4F73",
                zorder=31,
            )

        if state.get("show_pose_x1", False):
            if state.get("phase") in {"comparison", "pose_graph_preview", "summary"}:
                x1_face = "#B7E4C7"
                x1_edge = "#2E8B57"
            else:
                x1_face = "#F28E2B"
                x1_edge = "#8A4B08"

            self._dibujar_pose_restriccion(
                ax=ax,
                pose=pose_x1,
                label="x1 actual",
                face_color=x1_face,
                edge_color=x1_edge,
                zorder=33,
                label_offset=(0.18, -0.28),
            )

        if state.get("show_translation_error", False):
            dx = pose_x1[0] - expected[0]
            dy = pose_x1[1] - expected[1]
            error_norm = (dx**2 + dy**2) ** 0.5

            if error_norm > 1e-5:
                self._dibujar_flecha_transformacion_restriccion(
                    ax=ax,
                    start_pose=expected,
                    end_pose=pose_x1,
                    color="#D62728",
                    label=f"error {error_norm:.3f} m",
                    curvature=0.0,
                    line_width=3.5,
                    label_offset=(0.24, 0.02),
                    zorder=28,
                )
            else:
                ax.text(
                    expected[0] + 0.06,
                    expected[1] - 0.48,
                    "error de traslación ≈ 0",
                    fontsize=7.4,
                    fontweight="bold",
                    color="#2E8B57",
                    ha="center",
                    va="top",
                    zorder=40,
                )

        if state.get("show_angular_error", False):
            expected_angle = degrees(expected[2])
            current_angle = degrees(pose_x1[2])
            angular_error = current_angle - expected_angle

            if abs(angular_error) > 0.05:
                theta1 = min(expected_angle, current_angle)
                theta2 = max(expected_angle, current_angle)

                arc = Arc(
                    (pose_x1[0], pose_x1[1]),
                    width=0.76,
                    height=0.76,
                    angle=0.0,
                    theta1=theta1,
                    theta2=theta2,
                    linewidth=2.8,
                    color="#D62728",
                    zorder=36,
                )
                ax.add_patch(arc)

                ax.text(
                    pose_x1[0] + 0.46,
                    pose_x1[1] + 0.26,
                    f"eθ={angular_error:.2f}°",
                    fontsize=7.4,
                    fontweight="bold",
                    color="#D62728",
                    ha="left",
                    va="center",
                    zorder=40,
                )

        if state.get("show_prior", False):
            self._dibujar_prior_restriccion(ax, pose_x0)

        ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=ax.transAxes,
            fontsize=8.8,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=60,
        )

    def _dibujar_grafo_restriccion_pose(self, ax, graph, state):
        """Dibuja la arista como conexión, restricción o pose graph futuro."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(-1.1, 5.15)
        ax.set_ylim(-1.35, 1.75)

        if state.get("show_future_graph", False):
            ax.set_title(
                "Vista previa: Pose Graph SLAM",
                fontsize=10.4,
                fontweight="bold",
                pad=8,
            )

            positions = {
                "x0": (0.0, 0.0),
                "x1": (1.35, 0.0),
                "x2": (2.70, 0.55),
                "x3": (4.05, 0.05),
                "prior": (-0.70, 1.00),
            }

            edges = [
                ("x0", "x1", "z01"),
                ("x1", "x2", "z12"),
                ("x2", "x3", "z23"),
            ]

            for origin, destination, label in edges:
                arrow = FancyArrowPatch(
                    positions[origin],
                    positions[destination],
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=2.4,
                    color="#2E8B57",
                    shrinkA=14,
                    shrinkB=14,
                    zorder=12,
                )
                ax.add_patch(arrow)

                mx = (positions[origin][0] + positions[destination][0]) / 2
                my = (positions[origin][1] + positions[destination][1]) / 2
                ax.text(
                    mx,
                    my + 0.20,
                    label,
                    fontsize=7.0,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color="#2E8B57",
                )

            loop_arrow = FancyArrowPatch(
                positions["x3"],
                positions["x0"],
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=2.6,
                linestyle="dashed",
                color="#8E5EA2",
                shrinkA=14,
                shrinkB=14,
                connectionstyle="arc3,rad=-0.42",
                zorder=13,
            )
            ax.add_patch(loop_arrow)
            ax.text(
                2.05,
                -0.92,
                "cierre de ciclo z30",
                fontsize=7.2,
                fontweight="bold",
                color="#8E5EA2",
                ha="center",
                va="center",
            )

            prior_arrow = FancyArrowPatch(
                positions["prior"],
                positions["x0"],
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.8,
                color="#555555",
                shrinkA=12,
                shrinkB=14,
            )
            ax.add_patch(prior_arrow)

            for node in ("x0", "x1", "x2", "x3"):
                x, y = positions[node]
                face_color = "#4C9ED9" if node == "x0" else "#F6C85F"
                edge_color = "#1F4F73" if node == "x0" else "#8A6D1D"
                circle = Ellipse(
                    (x, y),
                    width=0.48,
                    height=0.48,
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=1.8,
                    zorder=20,
                )
                ax.add_patch(circle)
                ax.text(
                    x,
                    y,
                    node,
                    fontsize=8.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=25,
                )

            prior_box = Rectangle(
                (positions["prior"][0] - 0.33, positions["prior"][1] - 0.18),
                0.66,
                0.36,
                facecolor="#E5E5E5",
                edgecolor="#555555",
                linewidth=1.5,
                zorder=20,
            )
            ax.add_patch(prior_box)
            ax.text(
                *positions["prior"],
                "prior",
                fontsize=7.2,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=25,
            )

            ax.text(
                2.0,
                1.42,
                "Cada arista aporta un residuo al coste global",
                fontsize=7.6,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.30",
                    "fc": "white",
                    "ec": "#888888",
                    "alpha": 0.98,
                },
            )
            return

        ax.set_title(
            "Grafo de restricciones",
            fontsize=10.6,
            fontweight="bold",
            pad=8,
        )

        positions = {
            "prior": (-0.35, 0.95),
            "x0": (0.60, 0.0),
            "x1": (3.75, 0.0),
        }

        phase = state.get("phase", "")
        is_normal = phase == "normal_graph"
        cost = state.get("weighted_error", 0.0)

        if is_normal:
            edge_color = "#7F7F7F"
            edge_width = 2.6
            edge_label = "conexión"
        elif cost < 1e-8:
            edge_color = "#2E8B57"
            edge_width = 3.5
            edge_label = "z01, Ω01\nrestricción satisfecha"
        elif phase == "correction":
            edge_color = "#F28E2B"
            edge_width = 3.5
            edge_label = "z01, Ω01\nrestricción en corrección"
        else:
            edge_color = "#D62728"
            edge_width = 3.5
            edge_label = "z01, Ω01\nrestricción con error"

        edge_arrow = FancyArrowPatch(
            positions["x0"],
            positions["x1"],
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=edge_width,
            color=edge_color,
            shrinkA=24,
            shrinkB=24,
            zorder=12,
        )
        ax.add_patch(edge_arrow)

        ax.text(
            2.18,
            0.31,
            edge_label,
            fontsize=7.6,
            fontweight="bold",
            ha="center",
            va="center",
            color=edge_color,
            bbox={
                "boxstyle": "round,pad=0.22",
                "fc": "white",
                "ec": edge_color,
                "alpha": 0.96,
            },
        )

        node_specs = {
            "x0": ("#4C9ED9", "#1F4F73"),
            "x1": (
                ("#B7E4C7", "#2E8B57")
                if cost < 1e-8
                else ("#F28E2B", "#8A4B08")
            ),
        }

        for node in ("x0", "x1"):
            x, y = positions[node]
            face, edge = node_specs[node]
            circle = Ellipse(
                (x, y),
                width=0.72,
                height=0.72,
                facecolor=face,
                edgecolor=edge,
                linewidth=2.0,
                zorder=20,
            )
            ax.add_patch(circle)
            ax.text(
                x,
                y + 0.05,
                node,
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=25,
            )
            ax.text(
                x,
                y - 0.20,
                "fija" if node == "x0" else "variable",
                fontsize=6.3,
                ha="center",
                va="center",
                zorder=25,
            )

        if state.get("show_prior", False):
            prior_box = Rectangle(
                (positions["prior"][0] - 0.38, positions["prior"][1] - 0.20),
                0.76,
                0.40,
                facecolor="#E5E5E5",
                edgecolor="#555555",
                linewidth=1.6,
                zorder=20,
            )
            ax.add_patch(prior_box)
            ax.text(
                *positions["prior"],
                "prior",
                fontsize=7.4,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=25,
            )
            prior_arrow = FancyArrowPatch(
                positions["prior"],
                positions["x0"],
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.8,
                color="#555555",
                shrinkA=16,
                shrinkB=24,
                zorder=15,
            )
            ax.add_patch(prior_arrow)

        if state.get("show_constraint_details", False):
            measurement = state["measurement"]
            residual = state["residual"]
            details = (
                "ARISTA x0 → x1\n"
                f"sensor: {graph.edges['x0', 'x1'].get('sensor', '—')}\n"
                f"z01: {self._formatear_pose_restriccion(measurement)}\n"
                f"e01: {self._formatear_vector_restriccion(residual)}\n"
                f"E01: {state.get('weighted_error', 0.0):.4f}"
            )

            ax.text(
                2.18,
                -0.88,
                details,
                fontsize=7.0,
                family="monospace",
                ha="center",
                va="center",
                linespacing=1.35,
                bbox={
                    "boxstyle": "round,pad=0.42",
                    "fc": "#FAFAFA",
                    "ec": "#888888",
                    "alpha": 0.98,
                },
            )
        else:
            ax.text(
                2.18,
                -0.72,
                "Una arista normal no permite\nevaluar coherencia geométrica",
                fontsize=7.6,
                ha="center",
                va="center",
                color="#555555",
            )

    def _dibujar_panel_calculos_restriccion(self, ax, state):
        """Dibuja tarjetas de medición, predicción, residuo y coste."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.02,
            0.93,
            "Cálculo de la restricción",
            fontsize=11.3,
            fontweight="bold",
            ha="left",
            va="top",
        )

        if not state.get("show_constraint_details", False):
            ax.text(
                0.50,
                0.47,
                "La conexión todavía no contiene una medición, una incertidumbre ni una función de error.",
                fontsize=10.0,
                fontweight="bold",
                ha="center",
                va="center",
                color="#555555",
                bbox={
                    "boxstyle": "round,pad=0.48",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.98,
                },
            )
            return

        measurement = state["measurement"]
        prediction = state["prediction"]
        residual = state["residual"]
        visual_error = state["visual_error"]
        cost = state.get("weighted_error", 0.0)
        unweighted = state.get("unweighted_error", 0.0)

        card_specs = [
            (
                "Medición z01",
                self._formatear_pose_restriccion(measurement),
                "#D5E8D4",
                "#2E8B57",
            ),
            (
                "Predicción z_hat01",
                self._formatear_pose_restriccion(prediction),
                "#E8D7F1",
                "#8E5EA2",
            ),
            (
                "Residuo SE(2)",
                self._formatear_vector_restriccion(residual),
                "#F7C6C7",
                "#C62828",
            ),
            (
                "Coste ponderado",
                f"E01 = {cost:.6f}\n||e||² = {unweighted:.6f}",
                "#FBE5A6",
                "#8A6D1D",
            ),
        ]

        card_width = 0.215
        gap = 0.018
        total_width = 4 * card_width + 3 * gap
        start_x = 0.50 - total_width / 2

        for index, (title, value, face, edge) in enumerate(card_specs):
            x = start_x + index * (card_width + gap)

            rectangle = Rectangle(
                (x, 0.36),
                card_width,
                0.40,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.6,
            )
            ax.add_patch(rectangle)

            ax.text(
                x + card_width / 2,
                0.675,
                title,
                fontsize=7.6,
                fontweight="bold",
                ha="center",
                va="center",
                color=edge,
            )

            ax.text(
                x + card_width / 2,
                0.515,
                value,
                fontsize=6.9,
                family="monospace",
                ha="center",
                va="center",
                linespacing=1.35,
            )

        if state.get("show_uncertainty", False):
            sigmas = state["sigmas"]
            information = state["information"]
            uncertainty_text = (
                f"σ = ({sigmas[0]:.2f} m, {sigmas[1]:.2f} m, "
                f"{degrees(sigmas[2]):.2f}°)"
                f"   ·   diag(Ω) = "
                f"({information[0][0]:.2f}, "
                f"{information[1][1]:.2f}, "
                f"{information[2][2]:.2f})"
            )
            ax.text(
                0.50,
                0.270,
                uncertainty_text,
                fontsize=7.4,
                ha="center",
                va="center",
                color="#444444",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.96,
                },
            )

        visual_text = (
            "Error visual global: "
            f"Δx={visual_error[0]:.3f} m, "
            f"Δy={visual_error[1]:.3f} m, "
            f"Δθ={degrees(visual_error[2]):.3f}°"
        )
        ax.text(
            0.02,
            0.175,
            visual_text,
            fontsize=7.3,
            ha="left",
            va="center",
            color="#444444",
        )

        if state.get("phase") == "correction":
            alpha = state.get("correction_alpha", 0.0)
            bar_x = 0.58
            bar_y = 0.105
            bar_width = 0.36
            bar_height = 0.075

            background = Rectangle(
                (bar_x, bar_y),
                bar_width,
                bar_height,
                facecolor="#E5E5E5",
                edgecolor="#777777",
                linewidth=1.2,
            )
            ax.add_patch(background)

            progress = Rectangle(
                (bar_x, bar_y),
                bar_width * alpha,
                bar_height,
                facecolor="#B7E4C7",
                edgecolor="#2E8B57",
                linewidth=1.0,
            )
            ax.add_patch(progress)

            ax.text(
                bar_x + bar_width / 2,
                bar_y + bar_height / 2,
                f"corrección {100 * alpha:.0f}%",
                fontsize=7.0,
                fontweight="bold",
                ha="center",
                va="center",
            )

        if state.get("show_comparison", False):
            initial_cost = state.get("initial_weighted_error", 0.0)
            final_cost = state.get("final_weighted_error", 0.0)
            ax.text(
                0.98,
                0.175,
                (
                    f"ANTES: E={initial_cost:.6f}   →   "
                    f"DESPUÉS: E={final_cost:.6f}"
                ),
                fontsize=7.6,
                fontweight="bold",
                ha="right",
                va="center",
                color="#1F4F73",
            )

    def _dibujar_estado_restriccion_pose(
        self,
        geometry_ax,
        graph_ax,
        info_ax,
        calculation_ax,
        graph,
        state,
    ):
        """Dibuja un estado completo de la transición a un factor de pose."""

        self._dibujar_panel_informacion_restriccion(
            ax=info_ax,
            state=state,
        )
        self._dibujar_geometria_restriccion_pose(
            ax=geometry_ax,
            state=state,
        )
        self._dibujar_grafo_restriccion_pose(
            ax=graph_ax,
            graph=graph,
            state=state,
        )
        self._dibujar_panel_calculos_restriccion(
            ax=calculation_ax,
            state=state,
        )

    def animate_basic_pose_constraint(
        self,
        graph,
        states,
        title="De una arista normal a una restricción entre poses",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima la transición de una conexión a una restricción SE(2).

        La secuencia muestra:
        - dos vértices unidos por una arista normal;
        - la medición, la covarianza y la información de la arista;
        - las poses x0 y x1 en el plano;
        - la pose esperada y la relación predicha;
        - el residuo traslacional y angular;
        - el coste eᵀΩe;
        - el prior que fija x0;
        - la corrección progresiva de x1;
        - una vista previa de un pose graph con cierre de ciclo.
        """

        if not states:
            raise ValueError(
                "La lista de estados de la restricción no puede estar vacía."
            )

        if not graph.is_directed():
            raise ValueError("El grafo de restricciones debe ser dirigido.")

        if not graph.has_edge("x0", "x1"):
            raise ValueError("Debe existir la restricción dirigida x0→x1.")

        (
            fig,
            geometry_ax,
            graph_ax,
            info_ax,
            calculation_ax,
        ) = self._preparar_figura_restriccion_pose(title)

        if final_image_path is not None:
            self._dibujar_estado_restriccion_pose(
                geometry_ax=geometry_ax,
                graph_ax=graph_ax,
                info_ax=info_ax,
                calculation_ax=calculation_ax,
                graph=graph,
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
            self._dibujar_estado_restriccion_pose(
                geometry_ax=geometry_ax,
                graph_ax=graph_ax,
                info_ax=info_ax,
                calculation_ax=calculation_ax,
                graph=graph,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_restriccion_pose(
                geometry_ax=geometry_ax,
                graph_ax=graph_ax,
                info_ax=info_ax,
                calculation_ax=calculation_ax,
                graph=graph,
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
    # Variables, medición, predicción y error en SE(2)
    # ------------------------------------------------------------------

    def _preparar_figura_medicion_prediccion(self, title):
        """Crea cuatro zonas comparables con el apartado anterior."""

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[1.70, 4.15, 2.45],
            height_ratios=[4.75, 1.75],
            wspace=0.09,
            hspace=0.12,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        geometry_ax = fig.add_subplot(grid[0, 1])
        flow_ax = fig.add_subplot(grid[0, 2])
        calculation_ax = fig.add_subplot(grid[1, 1:])

        fig.suptitle(title, fontsize=16, fontweight="bold")
        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.925,
            bottom=0.045,
        )

        return fig, geometry_ax, flow_ax, info_ax, calculation_ax

    @staticmethod
    def _titulo_fase_medicion_prediccion(phase):
        """Traduce la fase del guion a una etiqueta breve."""

        titles = {
            "variables": "1. Variables estimadas",
            "estimate_vs_truth": "2. Estimación y estado real",
            "measurement": "3. Medición del sensor",
            "local_frame": "4. Sistema local de x0",
            "expected_pose": "5. Pose esperada",
            "model": "6. Modelo de medición",
            "prediction": "7. Predicción",
            "comparison": "8. Medición frente a predicción",
            "translation_error": "9. Error de traslación",
            "angular_error": "10. Error angular",
            "angle_wrap": "11. Normalización angular",
            "residual": "12. Residuo en SE(2)",
            "uncertainty": "13. Incertidumbre",
            "cost": "14. Información y coste",
            "estimation_experiment": "15. Varias estimaciones",
            "correction": "16. Cambio de la variable x1",
            "compatible": "17. Medición compatible",
            "future_graph": "18. Muchas restricciones",
            "summary": "19. Idea final",
        }
        return titles.get(phase, str(phase))

    def _dibujar_leyenda_medicion_prediccion(self, ax):
        """Dibuja la leyenda semántica del ejemplo."""

        elements = [
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor="#4C9ED9", markeredgecolor="#1F4F73",
                markersize=8, label="Variable x0 fija",
            ),
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor="#F28E2B", markeredgecolor="#8A4B08",
                markersize=8, label="Variable x1 estimada",
            ),
            Line2D(
                [0], [0], color="#2E8B57", linewidth=3,
                label="Medición fija z01",
            ),
            Line2D(
                [0], [0], color="#8E5EA2", linewidth=3,
                label="Predicción z_hat01",
            ),
            Line2D(
                [0], [0], color="#D62728", linewidth=3,
                label="Error / residuo",
            ),
            Line2D(
                [0], [0], color="#777777", linewidth=2,
                linestyle="dashed", label="Estimación inicial",
            ),
        ]

        ax.legend(
            handles=elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.035),
            fontsize=6.8,
            framealpha=0.97,
            ncol=1,
            handlelength=2.4,
            borderpad=0.55,
            labelspacing=0.58,
        )

    def _dibujar_panel_info_medicion_prediccion(self, ax, state):
        """Explica qué magnitudes son fijas y cuáles dependen del estado."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            self._titulo_fase_medicion_prediccion(state.get("phase")),
            fontsize=11.4,
            fontweight="bold",
            ha="center",
            va="top",
        )

        ax.text(
            0.50,
            0.925,
            f"Estado {state.get('step', 0)} de {state.get('total_steps', 0)}",
            fontsize=7.5,
            ha="center",
            va="top",
            color="#555555",
        )

        ax.text(
            0.50,
            0.825,
            state.get("message", ""),
            fontsize=8.2,
            fontweight="bold",
            ha="center",
            va="top",
            wrap=True,
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.48",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.98,
            },
        )

        fixed_box = Rectangle(
            (0.09, 0.585), 0.82, 0.105,
            facecolor="#D5E8D4", edgecolor="#2E8B57", linewidth=1.5,
        )
        ax.add_patch(fixed_box)
        ax.text(
            0.50, 0.638,
            "FIJO: z01, Σ01 y Ω01\n(datos registrados por el sensor)",
            fontsize=7.2, fontweight="bold", ha="center", va="center",
            color="#245B3A",
        )

        dynamic_box = Rectangle(
            (0.09, 0.445), 0.82, 0.105,
            facecolor="#E8D7F1", edgecolor="#8E5EA2", linewidth=1.5,
        )
        ax.add_patch(dynamic_box)
        ax.text(
            0.50, 0.498,
            "CAMBIA CON x1: z_hat01, e01 y E01\n(valores calculados por el modelo)",
            fontsize=7.2, fontweight="bold", ha="center", va="center",
            color="#5A316B",
        )

        experiment_label = state.get("experiment_label")
        if experiment_label:
            ax.text(
                0.50,
                0.390,
                experiment_label,
                fontsize=7.4,
                fontweight="bold",
                ha="center",
                va="center",
                color="#8A4B08",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "#FBE5A6",
                    "ec": "#8A6D1D",
                },
            )

        if state.get("show_angle_wrap", False):
            ax.text(
                0.50,
                0.305,
                "179° - (-179°) = 358°\nwrap(358°) = -2°",
                fontsize=7.5,
                family="monospace",
                fontweight="bold",
                ha="center",
                va="center",
                color="#7A1D1D",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "fc": "#F7C6C7",
                    "ec": "#C62828",
                },
            )
        else:
            ax.text(
                0.50,
                0.305,
                "z01 = sensor\nz_hat01 = h(x0, x1)\ne01 = z01^-1 ⊕ z_hat01",
                fontsize=7.2,
                family="monospace",
                ha="center",
                va="center",
                linespacing=1.45,
                color="#333333",
            )

        self._dibujar_leyenda_medicion_prediccion(ax)

    def _dibujar_geometria_medicion_prediccion(self, ax, state):
        """Dibuja variables, medición, predicción y discrepancia geométrica."""

        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.20, 5.20)
        ax.set_ylim(0.20, 4.15)
        ax.grid(True, linewidth=0.55, alpha=0.22)
        ax.set_xlabel("x global [m]", fontsize=8)
        ax.set_ylabel("y global [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("Interpretación geométrica", fontsize=11.0, fontweight="bold")

        if not state.get("show_geometry", False):
            ax.text(
                0.50, 0.50,
                "Las variables se representarán como poses en el plano.",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                ha="center", va="center", color="#555555",
            )
            return

        pose_x0 = state["pose_x0"]
        pose_x1 = state["pose_x1"]
        initial_pose = state["pose_x1_initial"]
        expected_pose = state["pose_x1_expected"]

        if state.get("show_uncertainty", False):
            self._dibujar_elipse_incertidumbre_restriccion(
                ax=ax,
                expected_pose=expected_pose,
                sigmas=state["sigmas"],
            )

        if state.get("show_initial_history", False):
            self._dibujar_pose_restriccion(
                ax=ax,
                pose=initial_pose,
                label="x1 inicial",
                face_color="white",
                edge_color="#777777",
                alpha=0.72,
                zorder=15,
                radius=0.09,
                label_offset=(0.02, 0.30),
                line_style="dashed",
            )
            self._dibujar_flecha_transformacion_restriccion(
                ax=ax,
                start_pose=pose_x0,
                end_pose=initial_pose,
                color="#777777",
                label="z_hat inicial",
                curvature=0.09,
                line_width=1.8,
                line_style="dashed",
                label_offset=(0.0, 0.20),
                alpha=0.62,
                zorder=11,
            )

        if state.get("show_measurement", False):
            self._dibujar_flecha_transformacion_restriccion(
                ax=ax,
                start_pose=pose_x0,
                end_pose=expected_pose,
                color="#2E8B57",
                label="medición z01",
                curvature=-0.055,
                line_width=3.2,
                label_offset=(-0.02, -0.17),
                zorder=19,
            )

        if state.get("show_prediction", False):
            self._dibujar_flecha_transformacion_restriccion(
                ax=ax,
                start_pose=pose_x0,
                end_pose=pose_x1,
                color="#8E5EA2",
                label="predicción z_hat01",
                curvature=0.055,
                line_width=3.0,
                label_offset=(0.02, 0.19),
                zorder=20,
            )

        if state.get("show_expected", False):
            self._dibujar_pose_restriccion(
                ax=ax,
                pose=expected_pose,
                label="x1* esperada",
                face_color="white",
                edge_color="#2E8B57",
                alpha=0.95,
                zorder=24,
                radius=0.10,
                label_offset=(-0.05, -0.32),
                line_style="dashed",
            )

        if state.get("show_x0", False):
            self._dibujar_pose_restriccion(
                ax=ax,
                pose=pose_x0,
                label="x0 variable fija",
                face_color="#B7D7F0",
                edge_color="#1F4F73",
                zorder=30,
                radius=0.11,
                label_offset=(0.0, -0.32),
            )

        if state.get("show_x1", False):
            self._dibujar_pose_restriccion(
                ax=ax,
                pose=pose_x1,
                label="x1 estimada",
                face_color="#FBE5A6",
                edge_color="#8A4B08",
                zorder=31,
                radius=0.11,
                label_offset=(0.05, 0.32),
            )

        if state.get("show_translation_error", False):
            dx = pose_x1[0] - expected_pose[0]
            dy = pose_x1[1] - expected_pose[1]
            distance = (dx * dx + dy * dy) ** 0.5

            if distance > 1e-8:
                error_arrow = FancyArrowPatch(
                    (expected_pose[0], expected_pose[1]),
                    (pose_x1[0], pose_x1[1]),
                    arrowstyle="-|>", mutation_scale=15,
                    linewidth=3.0, color="#D62728",
                    shrinkA=8, shrinkB=8, zorder=28,
                )
                ax.add_patch(error_arrow)
                ax.text(
                    (expected_pose[0] + pose_x1[0]) / 2 + 0.10,
                    (expected_pose[1] + pose_x1[1]) / 2 - 0.10,
                    f"||Δp||={state['translation_error']:.3f} m",
                    fontsize=7.6, fontweight="bold", color="#C62828",
                    ha="left", va="top",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "fc": "white", "ec": "#C62828", "alpha": 0.94,
                    },
                    zorder=35,
                )
            else:
                ax.text(
                    expected_pose[0] + 0.25,
                    expected_pose[1] - 0.45,
                    "error de posición ≈ 0",
                    fontsize=7.4, fontweight="bold", color="#2E8B57",
                    ha="center", va="center",
                )

        if state.get("show_angular_error", False):
            angle_error_deg = degrees(state["visual_error"][2])
            if abs(angle_error_deg) > 1e-7:
                start_deg = degrees(expected_pose[2])
                end_deg = start_deg + angle_error_deg
                theta1, theta2 = sorted((start_deg, end_deg))
                arc = Arc(
                    (pose_x1[0], pose_x1[1]),
                    width=0.75, height=0.75,
                    angle=0.0, theta1=theta1, theta2=theta2,
                    linewidth=2.6, color="#D62728", zorder=29,
                )
                ax.add_patch(arc)
                ax.text(
                    pose_x1[0] + 0.42,
                    pose_x1[1] + 0.18,
                    f"Δθ={angle_error_deg:.2f}°",
                    fontsize=7.4, fontweight="bold", color="#C62828",
                    ha="left", va="center",
                )

        if state.get("show_local_frame", False):
            ax.text(
                pose_x0[0] - 0.45,
                pose_x0[1] + 0.55,
                "z01 se expresa\nen los ejes de x0",
                fontsize=7.2, fontweight="bold", color="#1F4F73",
                ha="center", va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "white", "ec": "#1F4F73", "alpha": 0.95,
                },
            )

        if state.get("show_angle_wrap", False):
            ax.text(
                0.03, 0.97,
                "Ángulos periódicos\n179° ↔ -179°\ndiferencia real: 2°",
                transform=ax.transAxes,
                fontsize=7.5, fontweight="bold", color="#7A1D1D",
                ha="left", va="top",
                bbox={
                    "boxstyle": "round,pad=0.30",
                    "fc": "#F7C6C7", "ec": "#C62828", "alpha": 0.96,
                },
            )

        ax.text(
            0.99, 0.02,
            "Verde = sensor fijo  ·  Morado = modelo(x0, x1)",
            transform=ax.transAxes, fontsize=7.2,
            ha="right", va="bottom", color="#444444",
        )

    @staticmethod
    def _dibujar_caja_flujo(ax, xy, width, height, text, face, edge, active):
        """Dibuja una caja del flujo conceptual."""

        rectangle = Rectangle(
            xy, width, height,
            facecolor=face, edgecolor=edge,
            linewidth=2.8 if active else 1.4,
        )
        ax.add_patch(rectangle)
        ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            text,
            fontsize=7.0,
            fontweight="bold" if active else "normal",
            ha="center",
            va="center",
            linespacing=1.25,
            color=edge,
        )

    def _dibujar_grafo_flujo_medicion_prediccion(self, ax, graph, state):
        """Muestra el grafo y el flujo variable→predicción→residuo."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Grafo y flujo de cálculo", fontsize=10.8, fontweight="bold")

        focus = state.get("focus")

        # Grafo de variables en la parte superior.
        node_positions = {
            "prior": (0.13, 0.82),
            "x0": (0.47, 0.82),
            "x1": (0.83, 0.82),
        }

        prior_box = Rectangle(
            (0.04, 0.755), 0.18, 0.13,
            facecolor="#E5E5E5", edgecolor="#666666", linewidth=1.4,
        )
        ax.add_patch(prior_box)
        ax.text(0.13, 0.82, "prior", fontsize=7.6, fontweight="bold", ha="center", va="center")

        for node, color, edge_color in (
            ("x0", "#B7D7F0", "#1F4F73"),
            ("x1", "#FBE5A6", "#8A4B08"),
        ):
            x, y = node_positions[node]
            ellipse = Ellipse(
                (x, y), 0.20, 0.13,
                facecolor=color, edgecolor=edge_color, linewidth=2.0,
            )
            ax.add_patch(ellipse)
            ax.text(x, y, node, fontsize=8.3, fontweight="bold", ha="center", va="center")

        ax.add_patch(FancyArrowPatch(
            (0.22, 0.82), (0.37, 0.82),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.6, color="#666666",
        ))
        ax.add_patch(FancyArrowPatch(
            (0.57, 0.82), (0.73, 0.82),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=2.4, color="#2E8B57",
        ))
        ax.text(
            0.65, 0.875,
            "z01, Σ01, Ω01",
            fontsize=6.4, fontweight="bold",
            ha="center", va="bottom", color="#2E8B57",
        )

        if state.get("show_future_graph", False):
            for index, x in enumerate((0.18, 0.39, 0.60, 0.81)):
                ellipse = Ellipse(
                    (x, 0.665), 0.12, 0.075,
                    facecolor="#E5E5E5", edgecolor="#666666", linewidth=1.2,
                )
                ax.add_patch(ellipse)
                ax.text(x, 0.665, f"x{index}", fontsize=6.4, fontweight="bold", ha="center", va="center")
                if index > 0:
                    ax.add_patch(FancyArrowPatch(
                        (x - 0.15, 0.665), (x - 0.065, 0.665),
                        arrowstyle="-|>", mutation_scale=9,
                        linewidth=1.3, color="#8E5EA2",
                    ))
            ax.add_patch(FancyArrowPatch(
                (0.81, 0.63), (0.18, 0.63),
                arrowstyle="-|>", mutation_scale=9,
                linewidth=1.5, color="#D62728",
                connectionstyle="arc3,rad=-0.35",
            ))
            ax.text(0.50, 0.555, "muchos residuos → coste global", fontsize=6.7, fontweight="bold", ha="center", color="#C62828")

        # Flujo conceptual inferior.
        boxes = {
            "variables": ((0.05, 0.34), 0.25, 0.12, "variables\nx0, x1", "#B7D7F0", "#1F4F73"),
            "model": ((0.38, 0.34), 0.24, 0.12, "modelo\nh(x0, x1)", "#E8D7F1", "#8E5EA2"),
            "prediction": ((0.70, 0.34), 0.25, 0.12, "predicción\nz_hat01", "#E8D7F1", "#8E5EA2"),
            "measurement": ((0.05, 0.10), 0.25, 0.12, "sensor\nmedición z01", "#D5E8D4", "#2E8B57"),
            "residual": ((0.38, 0.10), 0.24, 0.12, "comparación\nresiduo e01", "#F7C6C7", "#C62828"),
            "cost": ((0.70, 0.10), 0.25, 0.12, "ponderación\ncoste E01", "#FBE5A6", "#8A6D1D"),
        }

        active_keys = {
            "variables": {"variables"},
            "measurement": {"measurement", "local_frame"},
            "model": {"model"},
            "prediction": {"prediction"},
            "comparison": {"comparison", "translation_error", "angular_error", "angle_wrap"},
            "residual": {"residual"},
            "cost": {"uncertainty", "cost"},
            "experiment": {"variables", "prediction", "measurement", "residual", "cost"},
            "correction": {"variables", "prediction", "measurement", "residual", "cost"},
            "compatible": {"measurement", "prediction", "residual", "cost"},
            "future_graph": {"residual", "cost"},
            "summary": set(boxes),
        }.get(focus, set())

        for key, (xy, width, height, text, face, edge) in boxes.items():
            self._dibujar_caja_flujo(
                ax, xy, width, height, text, face, edge, key in active_keys
            )

        arrows = [
            ((0.30, 0.40), (0.38, 0.40), "#1F4F73"),
            ((0.62, 0.40), (0.70, 0.40), "#8E5EA2"),
            ((0.82, 0.34), (0.58, 0.22), "#8E5EA2"),
            ((0.30, 0.16), (0.38, 0.16), "#2E8B57"),
            ((0.62, 0.16), (0.70, 0.16), "#C62828"),
        ]
        for start, end, color in arrows:
            ax.add_patch(FancyArrowPatch(
                start, end,
                arrowstyle="-|>", mutation_scale=10,
                linewidth=1.6, color=color,
            ))

    def _dibujar_panel_calculos_medicion_prediccion(self, ax, state):
        """Dibuja valores y muestra qué cambia al modificar x1."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.02, 0.94,
            "Valores de la medición y del modelo",
            fontsize=11.2, fontweight="bold", ha="left", va="top",
        )

        values = [
            (
                "Variable x1",
                self._formatear_pose_restriccion(state["pose_x1"]),
                "#FBE5A6", "#8A4B08",
            ),
            (
                "Medición z01 · FIJA",
                self._formatear_pose_restriccion(state["measurement"]),
                "#D5E8D4", "#2E8B57",
            ),
            (
                "Predicción z_hat01",
                self._formatear_pose_restriccion(state["prediction"]),
                "#E8D7F1", "#8E5EA2",
            ),
            (
                "Residuo e01",
                self._formatear_vector_restriccion(state["residual"]),
                "#F7C6C7", "#C62828",
            ),
            (
                "Coste ponderado",
                f"E01={state['weighted_error']:.6f}\n||e||²={state['unweighted_error']:.6f}",
                "#FBE5A6", "#8A6D1D",
            ),
        ]

        card_width = 0.175
        gap = 0.016
        total_width = len(values) * card_width + (len(values) - 1) * gap
        start_x = 0.50 - total_width / 2

        for index, (title, value, face, edge) in enumerate(values):
            x = start_x + index * (card_width + gap)
            rectangle = Rectangle(
                (x, 0.42), card_width, 0.36,
                facecolor=face, edgecolor=edge, linewidth=1.6,
            )
            ax.add_patch(rectangle)
            ax.text(
                x + card_width / 2, 0.695,
                title, fontsize=7.0, fontweight="bold",
                ha="center", va="center", color=edge,
            )
            ax.text(
                x + card_width / 2, 0.545,
                value, fontsize=6.3, family="monospace",
                ha="center", va="center", linespacing=1.35,
            )

        if state.get("show_information", False):
            info = state["information"]
            contrib = state["contributions"]
            ax.text(
                0.02, 0.300,
                (
                    "diag(Ω01) = "
                    f"({info[0][0]:.2f}, {info[1][1]:.2f}, {info[2][2]:.2f})"
                    "   ·   contribuciones = "
                    f"({contrib[0]:.3f}, {contrib[1]:.3f}, {contrib[2]:.3f})"
                ),
                fontsize=7.0, ha="left", va="center", color="#444444",
            )

        ax.text(
            0.02, 0.205,
            (
                f"Error visual: ||Δp||={state['translation_error']:.4f} m"
                f"   ·   |Δθ|={degrees(state['angular_error']):.4f}°"
            ),
            fontsize=7.1, ha="left", va="center", color="#444444",
        )

        initial_cost = state["initial_weighted_error"]
        current_cost = state["weighted_error"]
        final_cost = state["final_weighted_error"]
        ratio = 0.0 if initial_cost <= 0.0 else max(0.0, min(1.0, current_cost / initial_cost))

        bar_x = 0.58
        bar_y = 0.135
        bar_width = 0.36
        bar_height = 0.075
        ax.add_patch(Rectangle(
            (bar_x, bar_y), bar_width, bar_height,
            facecolor="#B7E4C7", edgecolor="#2E8B57", linewidth=1.2,
        ))
        ax.add_patch(Rectangle(
            (bar_x, bar_y), bar_width * ratio, bar_height,
            facecolor="#F6B4B4", edgecolor="#C62828", linewidth=1.0,
        ))
        ax.text(
            bar_x + bar_width / 2, bar_y + bar_height / 2,
            f"coste actual / inicial = {100 * ratio:.1f}%",
            fontsize=7.0, fontweight="bold", ha="center", va="center",
        )

        ax.text(
            0.98, 0.300,
            f"INICIAL E={initial_cost:.6f}   →   FINAL E={final_cost:.6f}",
            fontsize=7.3, fontweight="bold",
            ha="right", va="center", color="#1F4F73",
        )

        ax.text(
            0.98, 0.070,
            "La medición z01 no se modifica en ningún fotograma.",
            fontsize=7.2, fontweight="bold",
            ha="right", va="center", color="#2E8B57",
        )

    def _dibujar_estado_medicion_prediccion(
        self,
        geometry_ax,
        flow_ax,
        info_ax,
        calculation_ax,
        graph,
        state,
    ):
        """Dibuja un estado completo del apartado 5.2."""

        self._dibujar_panel_info_medicion_prediccion(info_ax, state)
        self._dibujar_geometria_medicion_prediccion(geometry_ax, state)
        self._dibujar_grafo_flujo_medicion_prediccion(flow_ax, graph, state)
        self._dibujar_panel_calculos_medicion_prediccion(calculation_ax, state)

    def animate_measurement_prediction_error(
        self,
        graph,
        states,
        title="Variables, medición, predicción y error",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima la diferencia entre variables, medición, predicción y residuo.

        La medición permanece fija mientras la estimación x1 cambia. En cada
        estado se recalculan la predicción, el residuo y el coste ponderado.
        """

        if not states:
            raise ValueError(
                "La lista de estados de medición/predicción no puede estar vacía."
            )

        if not graph.is_directed():
            raise ValueError("El grafo de mediciones debe ser dirigido.")

        if not graph.has_edge("x0", "x1"):
            raise ValueError("Debe existir la medición dirigida x0→x1.")

        (
            fig,
            geometry_ax,
            flow_ax,
            info_ax,
            calculation_ax,
        ) = self._preparar_figura_medicion_prediccion(title)

        if final_image_path is not None:
            self._dibujar_estado_medicion_prediccion(
                geometry_ax=geometry_ax,
                flow_ax=flow_ax,
                info_ax=info_ax,
                calculation_ax=calculation_ax,
                graph=graph,
                state=states[-1],
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(final_image_path, dpi=200, bbox_inches="tight")
            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_medicion_prediccion(
                geometry_ax=geometry_ax,
                flow_ax=flow_ax,
                info_ax=info_ax,
                calculation_ax=calculation_ax,
                graph=graph,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_medicion_prediccion(
                geometry_ax=geometry_ax,
                flow_ax=flow_ax,
                info_ax=info_ax,
                calculation_ax=calculation_ax,
                graph=graph,
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
    # Elementos específicos de funciones de coste y mínimos cuadrados
    # ------------------------------------------------------------------

    def _preparar_figura_funcion_coste(self, title):
        """
        Crea una figura comparable con los apartados 5.1 y 5.2.

        Distribución:
        - izquierda: mediciones, residuos y valores numéricos;
        - centro superior: coste no ponderado;
        - derecha superior: coste ponderado;
        - extremo derecho: grafo de factores;
        - zona inferior: evolución del coste y flujo conceptual.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            4,
            width_ratios=[1.85, 3.05, 3.05, 2.10],
            height_ratios=[4.65, 1.65],
            wspace=0.12,
            hspace=0.16,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        unweighted_ax = fig.add_subplot(grid[0, 1])
        weighted_ax = fig.add_subplot(grid[0, 2])
        graph_ax = fig.add_subplot(grid[0, 3])
        history_ax = fig.add_subplot(grid[1, 1:])

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

        return (
            fig,
            info_ax,
            unweighted_ax,
            weighted_ax,
            graph_ax,
            history_ax,
        )

    def _dibujar_leyenda_funcion_coste(self, ax, weighted=False):
        """Dibuja una leyenda compacta para las gráficas de coste."""

        total_label = (
            "Coste ponderado total"
            if weighted
            else "Coste total"
        )

        elements = [
            Line2D(
                [0],
                [0],
                color="#9A9A9A",
                linewidth=1.4,
                label="Costes individuales",
            ),
            Line2D(
                [0],
                [0],
                color="#1F4F73" if not weighted else "#6A3D9A",
                linewidth=3.0,
                label=total_label,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D",
                markersize=7,
                label="Estimación actual",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="#2E8B57",
                markeredgecolor="#1D5A38",
                markersize=10,
                label="Mínimo",
            ),
        ]

        ax.legend(
            handles=elements,
            loc="upper center",
            fontsize=6.7,
            framealpha=0.95,
            ncol=2,
            columnspacing=0.8,
            handlelength=2.0,
        )

    def _dibujar_panel_datos_funcion_coste(self, ax, state):
        """Muestra mediciones, pesos, residuos y resumen de la estimación."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            "Datos y evaluación",
            fontsize=11.5,
            fontweight="bold",
            ha="center",
            va="top",
        )

        mode = state.get("mode", "unweighted")
        mode_text = (
            "Mínimos cuadrados ponderados"
            if mode == "weighted"
            else "Mínimos cuadrados"
        )

        ax.text(
            0.50,
            0.947,
            mode_text,
            fontsize=8.2,
            ha="center",
            va="top",
            color="#444444",
        )

        measurements = list(state.get("measurements", []))
        weights = list(state.get("weights", []))
        residuals = list(state.get("residuals", []))
        individual_costs = list(state.get("individual_costs", []))
        weighted_costs = list(
            state.get("weighted_individual_costs", [])
        )

        show_weights = bool(state.get("show_weights", False))
        show_residuals = bool(state.get("show_residuals", False))
        show_individual_costs = bool(
            state.get("show_individual_costs", False)
        )

        card_colors = ["#DCEAF5", "#FBE5A6", "#E8D7F1"]
        top_y = 0.835
        card_height = 0.120
        vertical_gap = 0.018

        for index, measurement in enumerate(measurements):
            y = top_y - index * (card_height + vertical_gap)
            rectangle = Rectangle(
                (0.08, y),
                0.84,
                card_height,
                facecolor=card_colors[index % len(card_colors)],
                edgecolor="#666666",
                linewidth=1.3,
            )
            ax.add_patch(rectangle)

            ax.text(
                0.13,
                y + card_height * 0.70,
                f"z{index + 1} = {measurement:.2f}",
                fontsize=8.2,
                fontweight="bold",
                ha="left",
                va="center",
            )

            if show_weights:
                weight_text = f"w{index + 1} = {weights[index]:.2f}"
            else:
                weight_text = "peso = 1"

            ax.text(
                0.87,
                y + card_height * 0.70,
                weight_text,
                fontsize=6.8,
                ha="right",
                va="center",
            )

            if show_residuals:
                residual_text = f"e{index + 1} = {residuals[index]:+.3f}"
            else:
                residual_text = "e = x - z"

            ax.text(
                0.13,
                y + card_height * 0.30,
                residual_text,
                fontsize=6.9,
                ha="left",
                va="center",
            )

            if show_individual_costs:
                contribution = (
                    weighted_costs[index]
                    if show_weights
                    else individual_costs[index]
                )
                cost_text = f"F{index + 1} = {contribution:.3f}"
            else:
                cost_text = "F = e²"

            ax.text(
                0.87,
                y + card_height * 0.30,
                cost_text,
                fontsize=6.9,
                ha="right",
                va="center",
            )

        estimate = float(state.get("estimate", 0.0))
        active_cost = float(state.get("active_cost", 0.0))
        derivative = float(state.get("active_derivative", 0.0))
        curvature = float(state.get("active_second_derivative", 0.0))
        unweighted_minimum = float(state.get("unweighted_minimum", 0.0))
        weighted_minimum = float(state.get("weighted_minimum", 0.0))

        summary_y = 0.245
        summary = Rectangle(
            (0.08, summary_y),
            0.84,
            0.205,
            facecolor="white",
            edgecolor="#666666",
            linewidth=1.4,
        )
        ax.add_patch(summary)

        ax.text(
            0.13,
            summary_y + 0.165,
            f"x actual = {estimate:.5f}",
            fontsize=8.3,
            fontweight="bold",
            ha="left",
            va="center",
        )
        ax.text(
            0.13,
            summary_y + 0.120,
            f"F(x) = {active_cost:.5f}",
            fontsize=7.5,
            ha="left",
            va="center",
        )
        ax.text(
            0.13,
            summary_y + 0.078,
            f"F'(x) = {derivative:+.5f}",
            fontsize=7.2,
            ha="left",
            va="center",
        )
        ax.text(
            0.13,
            summary_y + 0.037,
            f"F''(x) = {curvature:.3f} > 0",
            fontsize=7.0,
            ha="left",
            va="center",
        )

        minima_y = 0.105
        ax.text(
            0.50,
            minima_y + 0.075,
            f"media = {unweighted_minimum:.5f}",
            fontsize=7.5,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.28",
                "fc": "#DCEAF5",
                "ec": "#1F4F73",
                "alpha": 0.97,
            },
        )
        ax.text(
            0.50,
            minima_y + 0.015,
            f"media ponderada = {weighted_minimum:.5f}",
            fontsize=7.2,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.28",
                "fc": "#E8D7F1",
                "ec": "#6A3D9A",
                "alpha": 0.97,
            },
        )

        ax.text(
            0.50,
            0.018,
            f"Paso {state.get('step', 0)} de {state.get('total_steps', 0)}",
            fontsize=6.8,
            ha="center",
            va="bottom",
            color="#555555",
        )

    def _dibujar_curva_funcion_coste(
        self,
        ax,
        state,
        *,
        weighted,
    ):
        """Dibuja la función no ponderada o la ponderada."""

        ax.clear()
        domain = list(state.get("domain", []))

        if not domain:
            ax.axis("off")
            return

        measurements = list(state.get("measurements", []))
        estimate = float(state.get("estimate", 0.0))
        show_measurements = bool(state.get("show_measurements", False))
        show_individual = bool(state.get("show_individual_costs", False))
        show_total = bool(state.get("show_total_cost", False))
        show_estimate = bool(state.get("show_estimate", False))
        show_gradient = bool(state.get("show_gradient", False))
        show_weights = bool(state.get("show_weights", False))
        show_weighted_curve = bool(state.get("show_weighted_curve", False))

        if weighted:
            ax.set_title(
                "Coste ponderado",
                fontsize=11,
                fontweight="bold",
            )

            if not show_weights:
                ax.set_xlim(min(domain), max(domain))
                ax.set_ylim(0, 1)
                ax.grid(alpha=0.18)
                ax.text(
                    0.50,
                    0.52,
                    "Los pesos se introducirán\ndespués del mínimo no ponderado",
                    transform=ax.transAxes,
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.45",
                        "fc": "white",
                        "ec": "#999999",
                        "alpha": 0.97,
                    },
                )
                ax.set_xlabel("Variable x", fontsize=8)
                ax.set_ylabel("Coste", fontsize=8)
                return

            individual_curves = list(
                state.get("weighted_individual_curves", [])
            )
            total_curve = list(state.get("weighted_curve", []))
            total_color = "#6A3D9A"
            active_cost = float(state.get("weighted_cost", 0.0))
            derivative = float(state.get("weighted_derivative", 0.0))
            minimum = float(state.get("weighted_minimum", 0.0))
            minimum_cost = float(state.get("weighted_minimum_cost", 0.0))
            show_minimum = bool(state.get("show_weighted_minimum", False))
        else:
            ax.set_title(
                "Coste no ponderado",
                fontsize=11,
                fontweight="bold",
            )
            individual_curves = list(state.get("individual_curves", []))
            total_curve = list(state.get("unweighted_curve", []))
            total_color = "#1F4F73"
            active_cost = float(state.get("unweighted_cost", 0.0))
            derivative = float(state.get("unweighted_derivative", 0.0))
            minimum = float(state.get("unweighted_minimum", 0.0))
            minimum_cost = float(state.get("unweighted_minimum_cost", 0.0))
            show_minimum = bool(
                state.get("show_unweighted_minimum", False)
            )

        curve_colors = ["#7FA6C2", "#D1A940", "#9B79B4"]

        if show_individual:
            for index, curve in enumerate(individual_curves):
                ax.plot(
                    domain,
                    curve,
                    color=curve_colors[index % len(curve_colors)],
                    linewidth=1.25,
                    alpha=0.78,
                    label=f"F{index + 1}",
                )

        if show_total and (not weighted or show_weighted_curve):
            ax.plot(
                domain,
                total_curve,
                color=total_color,
                linewidth=3.0,
                zorder=12,
            )

        if show_measurements:
            for index, measurement in enumerate(measurements):
                ax.scatter(
                    [measurement],
                    [0.0],
                    s=45,
                    marker="v",
                    color=curve_colors[index % len(curve_colors)],
                    edgecolors="#444444",
                    linewidths=0.7,
                    zorder=20,
                )
                ax.text(
                    measurement,
                    0.0,
                    f"  z{index + 1}",
                    fontsize=6.8,
                    ha="left",
                    va="bottom",
                )

        if show_estimate and show_total and (not weighted or show_weighted_curve):
            ax.scatter(
                [estimate],
                [active_cost],
                s=78,
                color="#E45756",
                edgecolors="#7A1D1D",
                linewidths=1.4,
                zorder=25,
            )
            ax.plot(
                [estimate, estimate],
                [0.0, active_cost],
                color="#E45756",
                linewidth=1.2,
                linestyle="dashed",
                alpha=0.75,
                zorder=14,
            )
            ax.text(
                estimate,
                active_cost,
                f"  x={estimate:.3f}\n  F={active_cost:.3f}",
                fontsize=6.7,
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "fc": "white",
                    "ec": "#C62828",
                    "alpha": 0.94,
                },
                zorder=30,
            )

        if show_minimum and show_total and (not weighted or show_weighted_curve):
            ax.scatter(
                [minimum],
                [minimum_cost],
                s=135,
                marker="*",
                color="#2E8B57",
                edgecolors="#1D5A38",
                linewidths=1.2,
                zorder=28,
            )
            ax.axvline(
                minimum,
                color="#2E8B57",
                linewidth=1.4,
                linestyle="dashed",
                alpha=0.8,
                zorder=13,
            )
            ax.text(
                minimum,
                minimum_cost,
                f"mínimo\n{minimum:.3f}",
                fontsize=6.8,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#1D5A38",
            )

        if weighted and state.get("show_unweighted_minimum", False):
            unweighted_minimum = float(
                state.get("unweighted_minimum", 0.0)
            )
            ax.axvline(
                unweighted_minimum,
                color="#1F4F73",
                linewidth=1.2,
                linestyle=":",
                alpha=0.85,
            )
            ax.text(
                unweighted_minimum,
                0.96,
                "media",
                transform=ax.get_xaxis_transform(),
                fontsize=6.3,
                ha="center",
                va="top",
                color="#1F4F73",
            )

        if (
            show_gradient
            and show_estimate
            and show_total
            and (not weighted or show_weighted_curve)
        ):
            span = (max(domain) - min(domain)) * 0.14
            tangent_x = [estimate - span, estimate + span]
            tangent_y = [
                active_cost + derivative * (value - estimate)
                for value in tangent_x
            ]
            ax.plot(
                tangent_x,
                tangent_y,
                color="#F28E2B",
                linewidth=2.2,
                linestyle="dashed",
                zorder=19,
            )

            direction = -1.0 if derivative > 0.0 else 1.0
            arrow_start = (estimate, active_cost)
            arrow_end = (
                estimate + direction * span * 0.75,
                max(0.0, active_cost - abs(derivative) * span * 0.20),
            )
            arrow = FancyArrowPatch(
                arrow_start,
                arrow_end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=2.0,
                color="#F28E2B",
                zorder=24,
            )
            ax.add_patch(arrow)

        ax.set_xlim(min(domain), max(domain))

        if total_curve:
            curve_max = max(total_curve)
            current_limit = max(active_cost, minimum_cost)
            y_max = max(1.0, min(curve_max * 1.04, current_limit * 2.2 + 20.0))
        else:
            y_max = 1.0

        if weighted:
            y_max = max(y_max, float(state.get("weighted_minimum_cost", 0.0)) * 2.8)
        else:
            y_max = max(y_max, float(state.get("unweighted_minimum_cost", 0.0)) * 3.0)

        ax.set_ylim(0.0, y_max)
        ax.set_xlabel("Variable x", fontsize=8)
        ax.set_ylabel("Coste", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.22)
        self._dibujar_leyenda_funcion_coste(ax, weighted=weighted)

    def _dibujar_grafo_factores_coste(self, ax, graph, state):
        """Dibuja la variable, los factores y la suma de costes."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            "Grafo de factores",
            fontsize=10.8,
            fontweight="bold",
            ha="center",
            va="top",
        )

        if not state.get("show_factor_graph", False):
            ax.text(
                0.50,
                0.52,
                "Cada medición terminará\nsiendo un factor de coste",
                fontsize=8.5,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.97,
                },
            )
            return

        estimate = float(state.get("estimate", 0.0))
        measurements = list(state.get("measurements", []))
        weights = list(state.get("weights", []))
        residuals = list(state.get("residuals", []))
        show_weights = bool(state.get("show_weights", False))
        contributions = (
            list(state.get("weighted_individual_costs", []))
            if show_weights
            else list(state.get("individual_costs", []))
        )

        variable_position = (0.50, 0.79)
        factor_positions = {
            "f1": (0.18, 0.50),
            "f2": (0.50, 0.50),
            "f3": (0.82, 0.50),
        }

        for index, factor in enumerate(("f1", "f2", "f3")):
            fx, fy = factor_positions[factor]
            arrow = FancyArrowPatch(
                variable_position,
                (fx, fy + 0.055),
                arrowstyle="-",
                linewidth=1.8,
                color="#777777",
                shrinkA=19,
                shrinkB=8,
                zorder=10,
            )
            ax.add_patch(arrow)

            middle_x = (variable_position[0] + fx) / 2
            middle_y = (variable_position[1] + fy) / 2
            ax.text(
                middle_x,
                middle_y,
                f"e{index + 1}={residuals[index]:+.2f}",
                fontsize=5.8,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.92,
                },
            )

        ax.scatter(
            [variable_position[0]],
            [variable_position[1]],
            s=1150,
            color="#E45756",
            edgecolors="#7A1D1D",
            linewidths=2.2,
            zorder=20,
        )
        ax.text(
            variable_position[0],
            variable_position[1],
            "x",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=25,
        )
        ax.text(
            variable_position[0],
            variable_position[1] - 0.10,
            f"{estimate:.3f}",
            fontsize=7.0,
            ha="center",
            va="top",
        )

        factor_colors = ["#DCEAF5", "#FBE5A6", "#E8D7F1"]

        for index, factor in enumerate(("f1", "f2", "f3")):
            fx, fy = factor_positions[factor]
            rectangle = Rectangle(
                (fx - 0.105, fy - 0.055),
                0.21,
                0.11,
                facecolor=factor_colors[index],
                edgecolor="#555555",
                linewidth=1.5,
                zorder=18,
            )
            ax.add_patch(rectangle)
            ax.text(
                fx,
                fy + 0.020,
                factor,
                fontsize=8.0,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=22,
            )

            weight_text = (
                f"w={weights[index]:.0f}"
                if show_weights
                else "w=1"
            )
            ax.text(
                fx,
                fy - 0.020,
                f"z={measurements[index]:.1f} · {weight_text}",
                fontsize=5.8,
                ha="center",
                va="center",
                zorder=22,
            )
            ax.text(
                fx,
                fy - 0.085,
                f"F{index + 1}={contributions[index]:.2f}",
                fontsize=6.2,
                ha="center",
                va="top",
            )

        sum_rectangle = Rectangle(
            (0.27, 0.235),
            0.46,
            0.105,
            facecolor="#D5E8D4",
            edgecolor="#2E8B57",
            linewidth=1.8,
        )
        ax.add_patch(sum_rectangle)
        ax.text(
            0.50,
            0.287,
            "F(x) = F1 + F2 + F3",
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="center",
        )

        for factor in ("f1", "f2", "f3"):
            fx, fy = factor_positions[factor]
            arrow = FancyArrowPatch(
                (fx, fy - 0.065),
                (0.50, 0.34),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color="#2E8B57",
                shrinkA=4,
                shrinkB=4,
                zorder=12,
            )
            ax.add_patch(arrow)

        active_cost = float(state.get("active_cost", 0.0))
        ax.text(
            0.50,
            0.205,
            f"coste actual = {active_cost:.4f}",
            fontsize=7.3,
            ha="center",
            va="top",
            color="#1D5A38",
        )

        if state.get("show_pose_graph_connection", False):
            ax.text(
                0.50,
                0.115,
                "Pose Graph SLAM",
                fontsize=8.2,
                fontweight="bold",
                ha="center",
                va="center",
            )

            pose_x = [0.18, 0.39, 0.61, 0.82]
            pose_y = 0.055

            for index, x_value in enumerate(pose_x):
                ax.scatter(
                    [x_value],
                    [pose_y],
                    s=150,
                    color="#4C9ED9",
                    edgecolors="#1F4F73",
                    linewidths=1.0,
                    zorder=20,
                )
                ax.text(
                    x_value,
                    pose_y,
                    f"x{index}",
                    fontsize=5.7,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=22,
                )

            for x1, x2 in zip(pose_x[:-1], pose_x[1:]):
                ax.plot(
                    [x1, x2],
                    [pose_y, pose_y],
                    color="#777777",
                    linewidth=1.2,
                    zorder=10,
                )

            ax.plot(
                [pose_x[0], pose_x[-1]],
                [pose_y + 0.010, pose_y + 0.010],
                color="#8E5EA2",
                linewidth=1.3,
                linestyle="dashed",
                zorder=11,
            )

    def _dibujar_historial_coste(self, ax, state):
        """Dibuja el historial del descenso o el flujo conceptual."""

        ax.clear()
        history_estimates = list(state.get("history_estimates", []))
        history_costs = list(state.get("history_costs", []))

        if history_costs:
            iterations = list(range(1, len(history_costs) + 1))
            ax.plot(
                iterations,
                history_costs,
                marker="o",
                linewidth=2.2,
                markersize=4.8,
                color=(
                    "#6A3D9A"
                    if state.get("mode") == "weighted"
                    else "#1F4F73"
                ),
            )
            ax.set_xlabel("Iteración", fontsize=8)
            ax.set_ylabel("Coste", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.22)
            ax.set_title(
                "Evolución del coste durante el descenso",
                fontsize=10.0,
                fontweight="bold",
                loc="left",
            )

            if history_estimates:
                ax.text(
                    0.99,
                    0.92,
                    (
                        f"x: {history_estimates[0]:.4f}"
                        f" → {history_estimates[-1]:.4f}\n"
                        f"F: {history_costs[0]:.4f}"
                        f" → {history_costs[-1]:.4f}"
                    ),
                    transform=ax.transAxes,
                    fontsize=7.5,
                    ha="right",
                    va="top",
                    bbox={
                        "boxstyle": "round,pad=0.32",
                        "fc": "white",
                        "ec": "#777777",
                        "alpha": 0.96,
                    },
                )
        else:
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            boxes = [
                (0.03, "mediciones"),
                (0.24, "residuos"),
                (0.45, "cuadrados"),
                (0.66, "suma"),
                (0.84, "mínimo"),
            ]

            for x, label in boxes:
                width = 0.13 if label != "mínimo" else 0.12
                rectangle = Rectangle(
                    (x, 0.34),
                    width,
                    0.30,
                    facecolor="#F4F4F4",
                    edgecolor="#666666",
                    linewidth=1.4,
                )
                ax.add_patch(rectangle)
                ax.text(
                    x + width / 2,
                    0.49,
                    label,
                    fontsize=8.3,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

            arrow_pairs = [
                (0.16, 0.24),
                (0.37, 0.45),
                (0.58, 0.66),
                (0.79, 0.84),
            ]

            for start, end in arrow_pairs:
                arrow = FancyArrowPatch(
                    (start, 0.49),
                    (end, 0.49),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.5,
                    color="#555555",
                )
                ax.add_patch(arrow)

            ax.text(
                0.50,
                0.82,
                "Construcción de la función objetivo",
                fontsize=10.0,
                fontweight="bold",
                ha="center",
                va="center",
            )

        ax.text(
            0.50,
            0.02,
            state.get("message", ""),
            transform=ax.transAxes,
            fontsize=8.5,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.36",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

    def _dibujar_estado_funcion_coste(
        self,
        info_ax,
        unweighted_ax,
        weighted_ax,
        graph_ax,
        history_ax,
        graph,
        state,
    ):
        """Dibuja un fotograma completo del apartado 5.3."""

        self._dibujar_panel_datos_funcion_coste(info_ax, state)
        self._dibujar_curva_funcion_coste(
            unweighted_ax,
            state,
            weighted=False,
        )
        self._dibujar_curva_funcion_coste(
            weighted_ax,
            state,
            weighted=True,
        )
        self._dibujar_grafo_factores_coste(graph_ax, graph, state)
        self._dibujar_historial_coste(history_ax, state)

    def animate_cost_function_least_squares(
        self,
        graph,
        states,
        title="Funciones de coste y mínimos cuadrados",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima la construcción y minimización de una función de coste.

        La imagen final muestra:
        - costes individuales y coste total;
        - mínimo no ponderado y ponderado;
        - mediciones, residuos, pesos y contribuciones;
        - grafo de factores;
        - conexión con la suma de costes de Pose Graph SLAM.
        """

        if not states:
            raise ValueError(
                "La lista de estados de la función de coste no puede estar vacía."
            )

        if graph.is_directed():
            raise ValueError("El grafo de factores debe ser no dirigido.")

        if "x" not in graph:
            raise ValueError("El grafo debe contener la variable x.")

        (
            fig,
            info_ax,
            unweighted_ax,
            weighted_ax,
            graph_ax,
            history_ax,
        ) = self._preparar_figura_funcion_coste(title)

        if final_image_path is not None:
            self._dibujar_estado_funcion_coste(
                info_ax=info_ax,
                unweighted_ax=unweighted_ax,
                weighted_ax=weighted_ax,
                graph_ax=graph_ax,
                history_ax=history_ax,
                graph=graph,
                state=states[-1],
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(final_image_path, dpi=200, bbox_inches="tight")
            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_funcion_coste(
                info_ax=info_ax,
                unweighted_ax=unweighted_ax,
                weighted_ax=weighted_ax,
                graph_ax=graph_ax,
                history_ax=history_ax,
                graph=graph,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_funcion_coste(
                info_ax=info_ax,
                unweighted_ax=unweighted_ax,
                weighted_ax=weighted_ax,
                graph_ax=graph_ax,
                history_ax=history_ax,
                graph=graph,
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
    # Incertidumbre, covarianza y matriz de información
    # ------------------------------------------------------------------

    def _preparar_figura_incertidumbre(self, title):
        """Crea una distribución visual comparable a los apartados 5.1-5.3."""

        fig = plt.figure(figsize=self.figsize)
        grid = fig.add_gridspec(
            3,
            3,
            width_ratios=[1.60, 3.55, 2.05],
            height_ratios=[2.25, 2.10, 1.45],
            wspace=0.12,
            hspace=0.18,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        geometry_ax = fig.add_subplot(grid[0:2, 1])
        comparison_ax = fig.add_subplot(grid[2, 1])
        matrices_ax = fig.add_subplot(grid[0, 2])
        graph_ax = fig.add_subplot(grid[1, 2])
        flow_ax = fig.add_subplot(grid[2, 2])

        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.93,
            bottom=0.045,
        )

        return (
            fig,
            info_ax,
            geometry_ax,
            comparison_ax,
            matrices_ax,
            graph_ax,
            flow_ax,
        )

    @staticmethod
    def _formatear_vector_2d_incertidumbre(vector, decimals=3):
        """Formatea un vector bidimensional para tarjetas y etiquetas."""

        if vector is None or len(vector) != 2:
            return "(—, —)"

        return f"({vector[0]:.{decimals}f}, {vector[1]:.{decimals}f})"

    @staticmethod
    def _formatear_matriz_2d_incertidumbre(matrix, decimals=3):
        """Formatea una matriz 2x2 en dos líneas compactas."""

        if matrix is None or len(matrix) != 2:
            return "[[—, —],\n [—, —]]"

        return (
            f"[[{matrix[0][0]:.{decimals}f}, {matrix[0][1]:.{decimals}f}],\n"
            f" [{matrix[1][0]:.{decimals}f}, {matrix[1][1]:.{decimals}f}]]"
        )

    def _dibujar_panel_incertidumbre(self, ax, state):
        """Dibuja las magnitudes numéricas y la explicación del fotograma."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        step = state.get("step", 1)
        total_steps = state.get("total_steps", 1)
        measurements = state.get("measurements", [[0, 0], [0, 0]])
        sigmas = state.get("sigmas", [[0, 0], [0, 0]])
        information_traces = state.get("information_traces", [0, 0])
        mahalanobis = state.get("mahalanobis_squared", [0, 0])
        weighted_mean = state.get("weighted_mean", [0, 0])
        unweighted_mean = state.get("unweighted_mean", [0, 0])

        ax.text(
            0.50,
            0.985,
            "Datos y confianza",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="top",
        )

        ax.text(
            0.50,
            0.948,
            f"Estado {step} de {total_steps}",
            fontsize=8.2,
            ha="center",
            va="top",
            color="#444444",
        )

        cards = [
            (
                0.825,
                "Medición fiable",
                (
                    f"μ₁ = {self._formatear_vector_2d_incertidumbre(measurements[0])}\n"
                    f"σ₁ = {self._formatear_vector_2d_incertidumbre(sigmas[0])}\n"
                    f"tr(Ω₁) = {information_traces[0]:.3f}"
                ),
                "#DDF1E5",
                "#2E8B57",
            ),
            (
                0.660,
                "Medición poco fiable",
                (
                    f"μ₂ = {self._formatear_vector_2d_incertidumbre(measurements[1])}\n"
                    f"σ₂ = {self._formatear_vector_2d_incertidumbre(sigmas[1])}\n"
                    f"tr(Ω₂) = {information_traces[1]:.3f}"
                ),
                "#FCE6D4",
                "#F28E2B",
            ),
            (
                0.495,
                "Mismo residuo",
                (
                    f"e = {self._formatear_vector_2d_incertidumbre(state.get('residual'))}\n"
                    f"||e||₂ = {state.get('euclidean_distance', 0.0):.4f}\n"
                    f"eᵀΩ₁e = {mahalanobis[0]:.4f}"
                    f" · eᵀΩ₂e = {mahalanobis[1]:.4f}"
                ),
                "#EEE7F4",
                "#8E5EA2",
            ),
            (
                0.330,
                "Fusión",
                (
                    f"sin pesos = {self._formatear_vector_2d_incertidumbre(unweighted_mean)}\n"
                    f"con Ω = {self._formatear_vector_2d_incertidumbre(weighted_mean)}\n"
                    "la solución se acerca a la más fiable"
                ),
                "#DDEAF4",
                "#1F4F73",
            ),
        ]

        for y, title, body, face_color, edge_color in cards:
            rectangle = Rectangle(
                (0.07, y - 0.115),
                0.86,
                0.135,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.5,
            )
            ax.add_patch(rectangle)
            ax.text(
                0.11,
                y,
                title,
                fontsize=8.7,
                fontweight="bold",
                ha="left",
                va="top",
            )
            ax.text(
                0.11,
                y - 0.032,
                body,
                fontsize=6.9,
                ha="left",
                va="top",
                linespacing=1.35,
            )

        phase_labels = {
            "introduction": "Una medición necesita valor e incertidumbre",
            "reliable_measurement": "Medición fiable",
            "reliable_sigmas": "Desviaciones estándar",
            "reliable_ellipse": "Elipse de incertidumbre",
            "uncertain_measurement": "Medición poco fiable",
            "compare_ellipses": "Comparación de dispersiones",
            "information_inverse": "Información como inversa",
            "same_residual": "Mismo residuo geométrico",
            "mahalanobis": "Coste de Mahalanobis",
            "scale_experiment": "Cambio de covarianza",
            "unweighted_fusion": "Fusión sin incertidumbre",
            "weighted_fusion": "Fusión ponderada",
            "factor_graph": "Grafo de factores",
            "sensor_connection": "Conexión con sensores",
            "summary": "Resumen",
        }

        ax.text(
            0.50,
            0.175,
            phase_labels.get(state.get("phase"), state.get("phase", "")),
            fontsize=9.2,
            fontweight="bold",
            ha="center",
            va="center",
            color="#333333",
        )

        ax.text(
            0.50,
            0.075,
            state.get("message", ""),
            fontsize=7.8,
            ha="center",
            va="center",
            wrap=True,
            bbox={
                "boxstyle": "round,pad=0.42",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.97,
            },
        )

    def _dibujar_elipse_incertidumbre(
        self,
        ax,
        center,
        ellipse_data,
        face_color,
        edge_color,
        alpha=0.20,
        line_width=2.2,
        zorder=8,
    ):
        """Dibuja una elipse precomputada por el script principal."""

        ellipse = Ellipse(
            xy=(center[0], center[1]),
            width=ellipse_data.get("width", 0.0),
            height=ellipse_data.get("height", 0.0),
            angle=ellipse_data.get("angle_deg", 0.0),
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=line_width,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(ellipse)

        angle = ellipse_data.get("angle_deg", 0.0) * pi / 180.0
        semi_major = ellipse_data.get("semi_major", 0.0)
        semi_minor = ellipse_data.get("semi_minor", 0.0)

        major_dx = semi_major * cos(angle)
        major_dy = semi_major * sin(angle)
        minor_dx = -semi_minor * sin(angle)
        minor_dy = semi_minor * cos(angle)

        ax.plot(
            [center[0] - major_dx, center[0] + major_dx],
            [center[1] - major_dy, center[1] + major_dy],
            color=edge_color,
            linewidth=1.0,
            alpha=0.75,
            zorder=zorder + 1,
        )
        ax.plot(
            [center[0] - minor_dx, center[0] + minor_dx],
            [center[1] - minor_dy, center[1] + minor_dy],
            color=edge_color,
            linewidth=1.0,
            alpha=0.75,
            zorder=zorder + 1,
        )

    def _dibujar_geometria_incertidumbre(self, ax, state):
        """Dibuja puntos, elipses, residuos y resultados de la fusión."""

        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(1.25, 6.90)
        ax.set_ylim(0.35, 5.55)
        ax.grid(alpha=0.20)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(
            "Dos mediciones 2D con distinta incertidumbre",
            fontsize=11.0,
            fontweight="bold",
            loc="left",
        )

        measurements = state.get("measurements", [[0, 0], [0, 0]])
        ellipses = state.get("ellipses", [{}, {}])
        residual_points = state.get("same_residual_points", [[0, 0], [0, 0]])

        show_reliable = state.get("show_measurement_reliable", False)
        show_uncertain = state.get("show_measurement_uncertain", False)

        if state.get("show_ellipse_reliable", False):
            self._dibujar_elipse_incertidumbre(
                ax,
                measurements[0],
                ellipses[0],
                face_color="#7BC8A4",
                edge_color="#2E8B57",
                alpha=0.24,
            )

        if state.get("show_ellipse_uncertain", False):
            self._dibujar_elipse_incertidumbre(
                ax,
                measurements[1],
                ellipses[1],
                face_color="#F5B97A",
                edge_color="#F28E2B",
                alpha=0.20,
            )

        if show_reliable:
            ax.scatter(
                [measurements[0][0]],
                [measurements[0][1]],
                s=155,
                marker="o",
                color="#4CAF7A",
                edgecolors="#1D5A38",
                linewidths=1.7,
                zorder=20,
            )
            ax.text(
                measurements[0][0] - 0.08,
                measurements[0][1] - 0.24,
                "z₁ fiable",
                fontsize=8.0,
                fontweight="bold",
                ha="right",
                va="top",
                color="#1D5A38",
                zorder=25,
            )

        if show_uncertain:
            ax.scatter(
                [measurements[1][0]],
                [measurements[1][1]],
                s=155,
                marker="o",
                color="#F28E2B",
                edgecolors="#8A4B08",
                linewidths=1.7,
                zorder=20,
            )
            ax.text(
                measurements[1][0] + 0.08,
                measurements[1][1] + 0.18,
                "z₂ poco fiable",
                fontsize=8.0,
                fontweight="bold",
                ha="left",
                va="bottom",
                color="#8A4B08",
                zorder=25,
            )

        if state.get("show_same_residual", False):
            colors = ["#8E5EA2", "#7B2CBF"]

            for index in range(2):
                start = measurements[index]
                end = residual_points[index]
                arrow = FancyArrowPatch(
                    (start[0], start[1]),
                    (end[0], end[1]),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=2.2,
                    color=colors[index],
                    zorder=28,
                )
                ax.add_patch(arrow)
                ax.scatter(
                    [end[0]],
                    [end[1]],
                    s=55,
                    marker="x",
                    color=colors[index],
                    linewidths=1.8,
                    zorder=29,
                )

            ax.text(
                0.02,
                0.97,
                "Mismo vector e aplicado a ambas mediciones",
                transform=ax.transAxes,
                fontsize=7.7,
                ha="left",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#8E5EA2",
                    "alpha": 0.95,
                },
            )

        if state.get("show_unweighted_fusion", False):
            point = state.get("unweighted_mean", [0, 0])
            ax.scatter(
                [point[0]],
                [point[1]],
                s=145,
                marker="s",
                color="#BDBDBD",
                edgecolors="#555555",
                linewidths=1.6,
                zorder=30,
            )
            ax.text(
                point[0],
                point[1] - 0.19,
                "media sin pesos",
                fontsize=7.5,
                ha="center",
                va="top",
                color="#444444",
            )

        if state.get("show_weighted_fusion", False):
            point = state.get("weighted_mean", [0, 0])
            fused_ellipse = state.get("fused_ellipse", {})
            self._dibujar_elipse_incertidumbre(
                ax,
                point,
                fused_ellipse,
                face_color="#80B1D3",
                edge_color="#1F4F73",
                alpha=0.18,
                line_width=2.0,
                zorder=14,
            )
            ax.scatter(
                [point[0]],
                [point[1]],
                s=190,
                marker="*",
                color="#4C9ED9",
                edgecolors="#1F4F73",
                linewidths=1.4,
                zorder=32,
            )
            ax.text(
                point[0] + 0.07,
                point[1] + 0.17,
                "fusión con Ω",
                fontsize=8.0,
                fontweight="bold",
                ha="left",
                va="bottom",
                color="#1F4F73",
            )

            for measurement in measurements:
                ax.plot(
                    [measurement[0], point[0]],
                    [measurement[1], point[1]],
                    color="#999999",
                    linewidth=1.0,
                    linestyle="dashed",
                    zorder=11,
                )

        if state.get("show_scale_experiment", False):
            ax.text(
                0.98,
                0.03,
                (
                    f"escala σ₂ = {state.get('uncertainty_scale', 1.0):.2f}\n"
                    f"área elipse₂ = {state.get('ellipse_areas', [0, 0])[1]:.3f}\n"
                    f"tr(Ω₂) = {state.get('information_traces', [0, 0])[1]:.3f}"
                ),
                transform=ax.transAxes,
                fontsize=7.5,
                ha="right",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.32",
                    "fc": "white",
                    "ec": "#F28E2B",
                    "alpha": 0.96,
                },
            )

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#4CAF7A",
                markeredgecolor="#1D5A38",
                markersize=8,
                label="Medición fiable",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F28E2B",
                markeredgecolor="#8A4B08",
                markersize=8,
                label="Medición poco fiable",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="#4C9ED9",
                markeredgecolor="#1F4F73",
                markersize=10,
                label="Fusión ponderada",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=7.0,
            framealpha=0.95,
        )

    def _dibujar_comparacion_incertidumbre(self, ax, state):
        """Compara costes y muestra el experimento de escalado de Σ."""

        ax.clear()

        history = list(state.get("scale_history", []))

        if state.get("show_scale_experiment", False) and history:
            scales = [item.get("scale", 0.0) for item in history]
            costs = [item.get("cost", 0.0) for item in history]
            traces = [item.get("information_trace", 0.0) for item in history]

            ax.plot(
                scales,
                costs,
                marker="o",
                linewidth=2.1,
                markersize=4.3,
                color="#8E5EA2",
                label="eᵀΩ₂e",
            )
            ax.plot(
                scales,
                traces,
                marker="s",
                linewidth=1.8,
                markersize=3.8,
                color="#F28E2B",
                label="tr(Ω₂)",
            )
            ax.set_xlabel("Factor aplicado a σ₂", fontsize=8)
            ax.set_ylabel("Valor", fontsize=8)
            ax.set_title(
                "Al crecer Σ disminuyen información y coste",
                fontsize=9.6,
                fontweight="bold",
                loc="left",
            )
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.22)
            ax.legend(fontsize=7, loc="upper right")
        elif state.get("show_mahalanobis", False):
            costs = state.get("mahalanobis_squared", [0.0, 0.0])
            labels = ["fiable", "poco fiable"]
            bars = ax.barh(
                labels,
                costs,
                color=["#4CAF7A", "#F28E2B"],
                edgecolor=["#1D5A38", "#8A4B08"],
                linewidth=1.2,
            )
            maximum = max(max(costs), 1.0)
            ax.set_xlim(0, maximum * 1.18)
            ax.set_xlabel("Coste de Mahalanobis eᵀΩe", fontsize=8)
            ax.set_title(
                "Mismo residuo, distinto peso estadístico",
                fontsize=9.6,
                fontweight="bold",
                loc="left",
            )
            ax.tick_params(labelsize=7)
            ax.grid(axis="x", alpha=0.20)

            for bar, value in zip(bars, costs):
                ax.text(
                    value + maximum * 0.025,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}",
                    fontsize=7.4,
                    va="center",
                    ha="left",
                )

            ax.text(
                0.99,
                0.08,
                f"||e||₂ = {state.get('euclidean_distance', 0.0):.4f} en ambos casos",
                transform=ax.transAxes,
                fontsize=7.2,
                ha="right",
                va="bottom",
                color="#444444",
            )
        else:
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            labels = [
                (0.04, "residuo e"),
                (0.29, "covarianza Σ"),
                (0.56, "información Ω"),
                (0.81, "coste"),
            ]

            for x, label in labels:
                width = 0.16
                rectangle = Rectangle(
                    (x, 0.34),
                    width,
                    0.30,
                    facecolor="#F4F4F4",
                    edgecolor="#666666",
                    linewidth=1.3,
                )
                ax.add_patch(rectangle)
                ax.text(
                    x + width / 2,
                    0.49,
                    label,
                    fontsize=8.0,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

            for start, end in [(0.20, 0.29), (0.45, 0.56), (0.72, 0.81)]:
                ax.add_patch(
                    FancyArrowPatch(
                        (start, 0.49),
                        (end, 0.49),
                        arrowstyle="-|>",
                        mutation_scale=12,
                        linewidth=1.4,
                        color="#555555",
                    )
                )

            ax.text(
                0.50,
                0.82,
                "La geometría del error no basta: también importa la confianza",
                fontsize=9.5,
                fontweight="bold",
                ha="center",
                va="center",
            )

    def _dibujar_matrices_incertidumbre(self, ax, state):
        """Muestra Σ, Ω, áreas y correlaciones de las dos mediciones."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.975,
            "Covarianza e información",
            fontsize=10.7,
            fontweight="bold",
            ha="center",
            va="top",
        )

        covariances = state.get("covariances", [[[0, 0], [0, 0]]] * 2)
        informations = state.get("informations", [[[0, 0], [0, 0]]] * 2)
        correlations = state.get("correlations", [0, 0])
        areas = state.get("ellipse_areas", [0, 0])

        show_covariances = state.get("show_covariances", False)
        show_information = state.get("show_information", False)

        rows = [
            (
                0.57,
                "1 · fiable",
                "#DDF1E5",
                "#2E8B57",
                covariances[0],
                informations[0],
                correlations[0],
                areas[0],
            ),
            (
                0.12,
                "2 · poco fiable",
                "#FCE6D4",
                "#F28E2B",
                covariances[1],
                informations[1],
                correlations[1],
                areas[1],
            ),
        ]

        for y, title, face, edge, covariance, information, rho, area in rows:
            rectangle = Rectangle(
                (0.05, y),
                0.90,
                0.35,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.5,
            )
            ax.add_patch(rectangle)
            ax.text(
                0.09,
                y + 0.315,
                title,
                fontsize=8.7,
                fontweight="bold",
                ha="left",
                va="top",
            )

            covariance_text = (
                self._formatear_matriz_2d_incertidumbre(covariance)
                if show_covariances
                else "Σ = pendiente"
            )
            information_text = (
                self._formatear_matriz_2d_incertidumbre(information)
                if show_information
                else "Ω = pendiente"
            )

            ax.text(
                0.09,
                y + 0.245,
                "Σ = " + covariance_text,
                fontsize=6.6,
                family="monospace",
                ha="left",
                va="top",
            )
            ax.text(
                0.09,
                y + 0.135,
                "Ω = " + information_text,
                fontsize=6.6,
                family="monospace",
                ha="left",
                va="top",
            )
            ax.text(
                0.91,
                y + 0.045,
                f"ρ={rho:.3f} · área₂σ={area:.3f}",
                fontsize=6.7,
                ha="right",
                va="bottom",
                color="#444444",
            )

        ax.text(
            0.50,
            0.035,
            "Σ pequeña ⇄ Ω grande",
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="center",
        )

    def _dibujar_grafo_incertidumbre(self, ax, graph, state):
        """Dibuja el grafo de factores con pesos derivados de Ω."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.96,
            "Grafo de factores",
            fontsize=10.6,
            fontweight="bold",
            ha="center",
            va="top",
        )

        if not state.get("show_factor_graph", False):
            ax.text(
                0.50,
                0.50,
                "Cada medición se convertirá\nen un factor ponderado",
                fontsize=9.0,
                ha="center",
                va="center",
                color="#666666",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "fc": "white",
                    "ec": "#AAAAAA",
                },
            )
            return

        positions = {
            "p": (0.50, 0.72),
            "f_fiable": (0.25, 0.29),
            "f_poco_fiable": (0.75, 0.29),
        }

        information_traces = state.get("information_traces", [0, 0])
        max_trace = max(max(information_traces), 1e-9)

        edge_data = [
            ("p", "f_fiable", information_traces[0], "#2E8B57"),
            ("p", "f_poco_fiable", information_traces[1], "#F28E2B"),
        ]

        for origin, destination, trace_value, color in edge_data:
            x1, y1 = positions[origin]
            x2, y2 = positions[destination]
            line_width = 1.4 + 4.2 * trace_value / max_trace
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=line_width,
                alpha=0.85,
                zorder=8,
            )
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.03,
                f"tr(Ω)={trace_value:.2f}",
                fontsize=6.7,
                ha="center",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.93,
                },
            )

        ax.scatter(
            [positions["p"][0]],
            [positions["p"][1]],
            s=430,
            marker="o",
            color="#4C9ED9",
            edgecolors="#1F4F73",
            linewidths=1.7,
            zorder=20,
        )
        ax.text(
            positions["p"][0],
            positions["p"][1],
            "p",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=25,
        )
        ax.text(
            positions["p"][0],
            positions["p"][1] + 0.12,
            "variable 2D",
            fontsize=7,
            ha="center",
            va="bottom",
        )

        factor_specs = [
            ("f_fiable", "f₁\nfiable", "#B7E4C7", "#2E8B57"),
            (
                "f_poco_fiable",
                "f₂\npoco fiable",
                "#F8D7B5",
                "#F28E2B",
            ),
        ]

        for node, label, face, edge in factor_specs:
            x, y = positions[node]
            rectangle = Rectangle(
                (x - 0.13, y - 0.09),
                0.26,
                0.18,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.6,
                zorder=18,
            )
            ax.add_patch(rectangle)
            ax.text(
                x,
                y,
                label,
                fontsize=7.7,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=22,
            )

        ax.text(
            0.50,
            0.075,
            "F(p) = Σ eₖᵀ Ωₖ eₖ",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
        )

    def _dibujar_flujo_incertidumbre(self, ax, state):
        """Dibuja el flujo conceptual y la conexión con sensores."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        boxes = [
            (0.02, 0.16, "sensor"),
            (0.22, 0.17, "z, Σ"),
            (0.43, 0.17, "Ω=Σ⁻¹"),
            (0.65, 0.16, "eᵀΩe"),
            (0.85, 0.13, "SLAM"),
        ]

        for x, width, label in boxes:
            rectangle = Rectangle(
                (x, 0.37),
                width,
                0.30,
                facecolor="#F4F4F4",
                edgecolor="#666666",
                linewidth=1.25,
            )
            ax.add_patch(rectangle)
            ax.text(
                x + width / 2,
                0.52,
                label,
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
            )

        for start, end in [(0.18, 0.22), (0.39, 0.43), (0.60, 0.65), (0.81, 0.85)]:
            ax.add_patch(
                FancyArrowPatch(
                    (start, 0.52),
                    (end, 0.52),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.3,
                    color="#555555",
                )
            )

        if state.get("show_sensor_connection", False):
            ax.text(
                0.50,
                0.17,
                "odometría · LiDAR · cámara · IMU",
                fontsize=7.2,
                fontweight="bold",
                ha="center",
                va="center",
                color="#1F4F73",
            )
        else:
            ax.text(
                0.50,
                0.17,
                "cada sensor necesita su propio modelo de incertidumbre",
                fontsize=7.0,
                ha="center",
                va="center",
                color="#555555",
            )

        ax.text(
            0.50,
            0.86,
            "De la medición al peso de una arista",
            fontsize=9.3,
            fontweight="bold",
            ha="center",
            va="center",
        )

    def _dibujar_estado_incertidumbre(
        self,
        info_ax,
        geometry_ax,
        comparison_ax,
        matrices_ax,
        graph_ax,
        flow_ax,
        graph,
        state,
    ):
        """Dibuja un fotograma completo del apartado 5.4."""

        self._dibujar_panel_incertidumbre(info_ax, state)
        self._dibujar_geometria_incertidumbre(geometry_ax, state)
        self._dibujar_comparacion_incertidumbre(comparison_ax, state)
        self._dibujar_matrices_incertidumbre(matrices_ax, state)
        self._dibujar_grafo_incertidumbre(graph_ax, graph, state)
        self._dibujar_flujo_incertidumbre(flow_ax, state)

    def animate_uncertainty_information(
        self,
        graph,
        states,
        title="Incertidumbre, covarianza y matriz de información",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima dos mediciones con distinta incertidumbre.

        La imagen final muestra:
        - puntos y elipses de incertidumbre;
        - covarianzas y matrices de información;
        - el mismo residuo con costes de Mahalanobis distintos;
        - fusión no ponderada y ponderada;
        - grafo de factores y conexión con sensores de SLAM.
        """

        if not states:
            raise ValueError(
                "La lista de estados de incertidumbre no puede estar vacía."
            )

        if graph.is_directed():
            raise ValueError("El grafo de incertidumbre debe ser no dirigido.")

        required_nodes = {"p", "f_fiable", "f_poco_fiable"}

        if not required_nodes.issubset(graph.nodes()):
            raise ValueError(
                "El grafo debe contener p y los dos factores de medición."
            )

        (
            fig,
            info_ax,
            geometry_ax,
            comparison_ax,
            matrices_ax,
            graph_ax,
            flow_ax,
        ) = self._preparar_figura_incertidumbre(title)

        if final_image_path is not None:
            self._dibujar_estado_incertidumbre(
                info_ax=info_ax,
                geometry_ax=geometry_ax,
                comparison_ax=comparison_ax,
                matrices_ax=matrices_ax,
                graph_ax=graph_ax,
                flow_ax=flow_ax,
                graph=graph,
                state=states[-1],
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(final_image_path, dpi=200, bbox_inches="tight")
            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_incertidumbre(
                info_ax=info_ax,
                geometry_ax=geometry_ax,
                comparison_ax=comparison_ax,
                matrices_ax=matrices_ax,
                graph_ax=graph_ax,
                flow_ax=flow_ax,
                graph=graph,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_incertidumbre(
                info_ax=info_ax,
                geometry_ax=geometry_ax,
                comparison_ax=comparison_ax,
                matrices_ax=matrices_ax,
                graph_ax=graph_ax,
                flow_ax=flow_ax,
                graph=graph,
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
    # Priors, libertad de gauge y anclaje del grafo
    # ------------------------------------------------------------------

    def _preparar_figura_prior_anclaje(self, title):
        """Crea dos paneles principales y dos paneles de análisis."""

        fig = plt.figure(figsize=self.figsize)
        grid = fig.add_gridspec(
            3,
            3,
            width_ratios=[1.55, 3.15, 3.15],
            height_ratios=[2.45, 2.15, 1.55],
            wspace=0.12,
            hspace=0.18,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        without_prior_ax = fig.add_subplot(grid[0:2, 1])
        with_prior_ax = fig.add_subplot(grid[0:2, 2])
        cost_ax = fig.add_subplot(grid[2, 1])
        algebra_ax = fig.add_subplot(grid[2, 2])

        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.93,
            bottom=0.045,
        )

        return (
            fig,
            info_ax,
            without_prior_ax,
            with_prior_ax,
            cost_ax,
            algebra_ax,
        )

    @staticmethod
    def _formatear_pose_prior(pose, decimals=2):
        """Formatea una pose serializada y convierte theta a grados."""

        if pose is None or len(pose) != 3:
            return "(—, —, —)"

        return (
            f"({pose[0]:.{decimals}f}, {pose[1]:.{decimals}f}, "
            f"{degrees(pose[2]):.{decimals}f}°)"
        )

    @staticmethod
    def _obtener_configuracion_prior(state, identifier):
        """Busca una configuración A, B o C dentro del estado."""

        for configuration in state.get("configurations", []):
            if configuration.get("id") == identifier:
                return configuration

        return None

    @staticmethod
    def _envolver_texto_prior(text, maximum_length=30):
        """Inserta saltos de línea sencillos para tarjetas estrechas."""

        words = str(text).split()
        lines = []
        current = []
        current_length = 0

        for word in words:
            extra = len(word) if not current else len(word) + 1
            if current and current_length + extra > maximum_length:
                lines.append(" ".join(current))
                current = [word]
                current_length = len(word)
            else:
                current.append(word)
                current_length += extra

        if current:
            lines.append(" ".join(current))

        return "\n".join(lines)

    @staticmethod
    def _limites_comunes_prior(state):
        """Calcula límites estables para todas las copias del pose graph."""

        xs = []
        ys = []

        configurations = list(state.get("configurations", []))
        dynamic = state.get("dynamic")

        if dynamic:
            configurations.append({"poses": dynamic.get("poses", {})})

        for configuration in configurations:
            for pose in configuration.get("poses", {}).values():
                if len(pose) >= 2:
                    xs.append(pose[0])
                    ys.append(pose[1])

        prior_pose = state.get("prior_pose", [0.0, 0.0, 0.0])
        xs.append(prior_pose[0])
        ys.append(prior_pose[1])

        if not xs:
            return (-4.0, 8.0, -1.5, 7.0)

        return (
            min(xs) - 1.25,
            max(xs) + 1.25,
            min(ys) - 1.20,
            max(ys) + 1.20,
        )

    def _dibujar_pose_prior(
        self,
        ax,
        pose,
        label,
        color,
        edge_color,
        alpha=1.0,
        zorder=20,
        anchored=False,
    ):
        """Dibuja una pose como punto orientado con su etiqueta."""

        x, y, theta = pose
        node_size = 95 if anchored else 70

        ax.scatter(
            [x],
            [y],
            s=node_size,
            c=[color],
            edgecolors=edge_color,
            linewidths=2.2 if anchored else 1.4,
            alpha=alpha,
            zorder=zorder,
        )

        arrow_length = 0.42
        end_x = x + arrow_length * cos(theta)
        end_y = y + arrow_length * sin(theta)

        ax.add_patch(
            FancyArrowPatch(
                (x, y),
                (end_x, end_y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.8 if anchored else 1.35,
                color=edge_color,
                alpha=alpha,
                zorder=zorder + 1,
            )
        )

        ax.text(
            x,
            y + 0.30,
            label,
            fontsize=7.4,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="#222222",
            alpha=alpha,
            zorder=zorder + 2,
            bbox={
                "boxstyle": "round,pad=0.14",
                "fc": "white",
                "ec": edge_color,
                "alpha": 0.86 * alpha,
            },
        )

        if anchored:
            ax.text(
                x,
                y - 0.36,
                "anclada",
                fontsize=6.6,
                fontweight="bold",
                ha="center",
                va="top",
                color="#8B0000",
                zorder=zorder + 2,
            )

    def _dibujar_pose_graph_prior(
        self,
        ax,
        state,
        poses,
        color,
        edge_color,
        label_prefix,
        alpha=1.0,
        line_style="solid",
        visible_pose_count=4,
        visible_edge_count=4,
        anchored_x0=False,
        zorder=10,
    ):
        """Dibuja una copia completa o parcial del pequeño pose graph."""

        ordered_pose_names = sorted(
            poses,
            key=lambda name: int(name[1:]),
        )[:visible_pose_count]
        visible_names = set(ordered_pose_names)

        relative_edges = state.get("relative_edges", [])[:visible_edge_count]

        for edge in relative_edges:
            origin = edge.get("origin")
            target = edge.get("target")

            if origin not in visible_names or target not in visible_names:
                continue

            pose_origin = poses[origin]
            pose_target = poses[target]

            ax.plot(
                [pose_origin[0], pose_target[0]],
                [pose_origin[1], pose_target[1]],
                color=edge_color,
                linewidth=2.0 if alpha > 0.75 else 1.35,
                linestyle=line_style,
                alpha=alpha,
                zorder=zorder,
            )

            if edge.get("sensor") == "cierre de ciclo":
                middle_x = (pose_origin[0] + pose_target[0]) / 2
                middle_y = (pose_origin[1] + pose_target[1]) / 2
                ax.text(
                    middle_x,
                    middle_y,
                    "loop",
                    fontsize=5.8,
                    ha="center",
                    va="center",
                    color=edge_color,
                    alpha=alpha,
                    zorder=zorder + 1,
                    bbox={
                        "boxstyle": "round,pad=0.10",
                        "fc": "white",
                        "ec": "none",
                        "alpha": 0.76,
                    },
                )

        for name in ordered_pose_names:
            self._dibujar_pose_prior(
                ax=ax,
                pose=poses[name],
                label=f"{label_prefix}{name}",
                color=color,
                edge_color=edge_color,
                alpha=alpha,
                zorder=zorder + 3,
                anchored=anchored_x0 and name == "x0",
            )

    def _dibujar_ejes_mundo_prior(self, ax, origin=(0.0, 0.0)):
        """Dibuja un pequeño sistema de referencia mundial."""

        x, y = origin
        length = 0.85

        ax.add_patch(
            FancyArrowPatch(
                (x, y),
                (x + length, y),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.3,
                color="#333333",
                zorder=6,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (x, y),
                (x, y + length),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.3,
                color="#333333",
                zorder=6,
            )
        )
        ax.text(x + length + 0.08, y, "Xw", fontsize=6.7, va="center")
        ax.text(x, y + length + 0.08, "Yw", fontsize=6.7, ha="center")

    def _dibujar_panel_prior(self, ax, state):
        """Dibuja la explicación, la transformación y los costes del paso."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        step = state.get("step", 1)
        total_steps = state.get("total_steps", 1)
        dynamic = state.get("dynamic", {})
        transform = dynamic.get("transform", [0.0, 0.0, 0.0])

        ax.text(
            0.50,
            0.985,
            "Gauge y referencia",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="top",
        )
        ax.text(
            0.50,
            0.948,
            f"Estado {step} de {total_steps}",
            fontsize=8.2,
            ha="center",
            va="top",
            color="#444444",
        )

        cards = [
            (
                0.825,
                "Transformación global g",
                (
                    f"g = {self._formatear_pose_prior(transform)}\n"
                    "xᵢ' = g ⊕ xᵢ"
                ),
                "#EEE7F4",
                "#8E5EA2",
            ),
            (
                0.665,
                "Coste de la copia activa",
                (
                    f"Frel = {dynamic.get('relative_cost', 0.0):.6f}\n"
                    f"Fprior = {dynamic.get('prior_cost', 0.0):.3f}\n"
                    f"Ftotal = {dynamic.get('total_cost', 0.0):.3f}"
                ),
                "#DDEAF4",
                "#1F4F73",
            ),
            (
                0.500,
                "Sin prior",
                (
                    f"rango = {state.get('rank_without_prior', 0)}\n"
                    f"nulidad = {state.get('nullity_without_prior', 0)}\n"
                    "tx · ty · rotación"
                ),
                "#FCE6D4",
                "#F28E2B",
            ),
            (
                0.335,
                "Con prior sobre x0",
                (
                    f"rango = {state.get('rank_with_prior', 0)}\n"
                    f"nulidad = {state.get('nullity_with_prior', 0)}\n"
                    "origen y orientación definidos"
                ),
                "#DDF1E5",
                "#2E8B57",
            ),
        ]

        for y, title, body, face_color, edge_color in cards:
            rectangle = Rectangle(
                (0.07, y - 0.115),
                0.86,
                0.138,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.5,
            )
            ax.add_patch(rectangle)
            ax.text(
                0.11,
                y,
                title,
                fontsize=8.5,
                fontweight="bold",
                ha="left",
                va="top",
            )
            ax.text(
                0.11,
                y - 0.035,
                body,
                fontsize=6.9,
                ha="left",
                va="top",
                linespacing=1.35,
            )

        phase_labels = {
            "introduction": "Restricciones relativas",
            "build_poses": "Variables del pose graph",
            "build_edges": "Aristas relativas",
            "relative_cost": "Coste interno",
            "translate_gauge": "Traslación global",
            "rotate_gauge": "Rotación global",
            "equivalent_solutions": "Soluciones equivalentes",
            "nullspace": "Espacio nulo",
            "add_prior": "Factor prior",
            "prior_penalty": "Penalización absoluta",
            "anchored_solution": "Mapa anclado",
            "physical_priors": "Referencias externas",
            "summary": "Resumen",
        }

        ax.text(
            0.50,
            0.175,
            phase_labels.get(state.get("phase"), state.get("phase", "")),
            fontsize=9.2,
            fontweight="bold",
            ha="center",
            va="center",
            color="#333333",
        )
        ax.text(
            0.50,
            0.074,
            self._envolver_texto_prior(state.get("message", ""), 29),
            fontsize=7.3,
            ha="center",
            va="center",
            wrap=True,
            bbox={
                "boxstyle": "round,pad=0.42",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.97,
            },
        )

    def _dibujar_sin_prior(self, ax, state):
        """Dibuja el grafo flotante y sus copias equivalentes."""

        ax.clear()
        limits = self._limites_comunes_prior(state)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.14)
        ax.set_title(
            "Sin prior: el grafo puede flotar",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.set_xlabel("x global [m]", fontsize=8)
        ax.set_ylabel("y global [m]", fontsize=8)
        ax.tick_params(labelsize=7)

        original = self._obtener_configuracion_prior(state, "A")
        translated = self._obtener_configuracion_prior(state, "B")
        rotated = self._obtener_configuracion_prior(state, "C")

        visible_pose_count = state.get("visible_pose_count", 4)
        visible_edge_count = state.get("visible_edge_count", 4)

        if state.get("show_original") and original:
            self._dibujar_pose_graph_prior(
                ax,
                state,
                original["poses"],
                "#4C9ED9",
                "#1F4F73",
                "A·",
                alpha=0.96,
                visible_pose_count=visible_pose_count,
                visible_edge_count=visible_edge_count,
                zorder=20,
            )

        if state.get("show_translated") and translated:
            self._dibujar_pose_graph_prior(
                ax,
                state,
                translated["poses"],
                "#F6C85F",
                "#8A6D1D",
                "B·",
                alpha=0.70,
                line_style="dashed",
                zorder=12,
            )

        if state.get("show_rotated") and rotated:
            self._dibujar_pose_graph_prior(
                ax,
                state,
                rotated["poses"],
                "#D8C4E8",
                "#5A316B",
                "C·",
                alpha=0.70,
                line_style="dashed",
                zorder=11,
            )

        if state.get("show_dynamic"):
            dynamic = state.get("dynamic", {})
            self._dibujar_pose_graph_prior(
                ax,
                state,
                dynamic.get("poses", {}),
                "#F6B4B4",
                "#C62828",
                "g·",
                alpha=0.90,
                line_style="solid",
                zorder=30,
            )

        if state.get("show_gauge"):
            gauge_x = limits[0] + 0.75
            gauge_y = limits[2] + 0.72
            ax.add_patch(
                FancyArrowPatch(
                    (gauge_x, gauge_y),
                    (gauge_x + 1.05, gauge_y),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=1.6,
                    color="#8E5EA2",
                    zorder=45,
                )
            )
            ax.add_patch(
                FancyArrowPatch(
                    (gauge_x, gauge_y),
                    (gauge_x, gauge_y + 1.05),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=1.6,
                    color="#8E5EA2",
                    zorder=45,
                )
            )
            ax.add_patch(
                Arc(
                    (gauge_x, gauge_y),
                    1.15,
                    1.15,
                    angle=0,
                    theta1=15,
                    theta2=285,
                    linewidth=1.6,
                    color="#8E5EA2",
                    zorder=44,
                )
            )
            ax.text(
                gauge_x + 0.55,
                gauge_y - 0.32,
                "tx",
                fontsize=7,
                color="#5A316B",
                ha="center",
            )
            ax.text(
                gauge_x - 0.28,
                gauge_y + 0.55,
                "ty",
                fontsize=7,
                color="#5A316B",
                va="center",
            )
            ax.text(
                gauge_x + 0.50,
                gauge_y + 0.55,
                "θ",
                fontsize=7,
                color="#5A316B",
                ha="center",
            )

        if state.get("show_equal_cost"):
            ax.text(
                0.50,
                0.025,
                "Frel(A) = Frel(B) = Frel(C)",
                transform=ax.transAxes,
                fontsize=8.3,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#5A316B",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#8E5EA2",
                    "alpha": 0.95,
                },
            )
        else:
            ax.text(
                0.50,
                0.025,
                "Las aristas observan relaciones, no coordenadas absolutas",
                transform=ax.transAxes,
                fontsize=7.5,
                ha="center",
                va="bottom",
                color="#555555",
            )

    def _dibujar_factor_prior(self, ax, state, x0_pose):
        """Dibuja el factor unario y su conexión con x0."""

        prior_pose = state.get("prior_pose", [0.0, 0.0, 0.0])
        factor_x = prior_pose[0] - 0.78
        factor_y = prior_pose[1] - 0.75

        rectangle = Rectangle(
            (factor_x - 0.40, factor_y - 0.25),
            0.80,
            0.50,
            facecolor="#F6B4B4",
            edgecolor="#8B0000",
            linewidth=2.0,
            zorder=50,
        )
        ax.add_patch(rectangle)
        ax.text(
            factor_x,
            factor_y + 0.06,
            "PRIOR",
            fontsize=7.7,
            fontweight="bold",
            ha="center",
            va="center",
            color="#8B0000",
            zorder=51,
        )
        ax.text(
            factor_x,
            factor_y - 0.10,
            "x0 = (0,0,0)",
            fontsize=5.9,
            ha="center",
            va="center",
            color="#4A0000",
            zorder=51,
        )
        ax.plot(
            [factor_x + 0.40, x0_pose[0]],
            [factor_y, x0_pose[1]],
            color="#C62828",
            linewidth=2.3,
            zorder=48,
        )

    def _dibujar_con_prior(self, ax, state):
        """Dibuja la comparación con el factor prior activado."""

        ax.clear()
        limits = self._limites_comunes_prior(state)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.14)
        ax.set_title(
            "Con prior: x0 define el marco del mapa",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.set_xlabel("x mundial [m]", fontsize=8)
        ax.set_ylabel("y mundial [m]", fontsize=8)
        ax.tick_params(labelsize=7)

        original = self._obtener_configuracion_prior(state, "A")
        translated = self._obtener_configuracion_prior(state, "B")
        rotated = self._obtener_configuracion_prior(state, "C")

        self._dibujar_ejes_mundo_prior(ax)

        if original and state.get("show_original"):
            self._dibujar_pose_graph_prior(
                ax,
                state,
                original["poses"],
                "#B7E4C7" if state.get("show_prior") else "#D9D9D9",
                "#2E8B57" if state.get("show_prior") else "#777777",
                "A·",
                alpha=0.98,
                anchored_x0=state.get("show_prior", False),
                zorder=25,
            )

        if state.get("show_translated") and translated:
            self._dibujar_pose_graph_prior(
                ax,
                state,
                translated["poses"],
                "#FBE5A6",
                "#8A6D1D",
                "B·",
                alpha=0.34 if state.get("show_prior") else 0.58,
                line_style="dashed",
                zorder=11,
            )

        if state.get("show_rotated") and rotated:
            self._dibujar_pose_graph_prior(
                ax,
                state,
                rotated["poses"],
                "#E8D7F1",
                "#8E5EA2",
                "C·",
                alpha=0.34 if state.get("show_prior") else 0.58,
                line_style="dashed",
                zorder=10,
            )

        if state.get("show_prior") and original:
            self._dibujar_factor_prior(
                ax,
                state,
                original["poses"]["x0"],
            )

        if state.get("show_prior_costs"):
            prior_pose = state.get("prior_pose", [0.0, 0.0, 0.0])

            for configuration, color in [
                (translated, "#8A6D1D"),
                (rotated, "#5A316B"),
            ]:
                if not configuration:
                    continue

                x0 = configuration["poses"]["x0"]
                ax.plot(
                    [prior_pose[0], x0[0]],
                    [prior_pose[1], x0[1]],
                    color=color,
                    linewidth=1.8,
                    linestyle="dotted",
                    alpha=0.85,
                    zorder=42,
                )
                ax.text(
                    (prior_pose[0] + x0[0]) / 2,
                    (prior_pose[1] + x0[1]) / 2 + 0.18,
                    f"Fprior={configuration['prior_cost']:.1f}",
                    fontsize=6.6,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    color=color,
                    zorder=43,
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "fc": "white",
                        "ec": color,
                        "alpha": 0.90,
                    },
                )

        if state.get("show_prior"):
            ax.text(
                0.50,
                0.025,
                "Solo A es compatible con el prior sin coste absoluto",
                transform=ax.transAxes,
                fontsize=8.1,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#8B0000",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#C62828",
                    "alpha": 0.95,
                },
            )
        else:
            ax.text(
                0.50,
                0.025,
                "El sistema mundial todavía no está conectado al grafo",
                transform=ax.transAxes,
                fontsize=7.5,
                ha="center",
                va="bottom",
                color="#555555",
            )

    def _dibujar_costes_prior(self, ax, state):
        """Compara el coste relativo y el coste total de A, B y C."""

        ax.clear()
        ax.set_title(
            "Mismo coste relativo, distinto coste total",
            fontsize=9.7,
            fontweight="bold",
        )

        if not state.get("show_cost_comparison"):
            ax.axis("off")
            ax.text(
                0.50,
                0.55,
                "Cada arista aporta un término relativo.\n"
                "El prior añadirá un término absoluto.",
                fontsize=8.2,
                ha="center",
                va="center",
                color="#555555",
                bbox={
                    "boxstyle": "round,pad=0.42",
                    "fc": "white",
                    "ec": "#999999",
                },
            )
            return

        configurations = state.get("configurations", [])
        labels = [configuration.get("id", "?") for configuration in configurations]
        relative_costs = [
            max(configuration.get("relative_cost", 0.0), 1e-5)
            for configuration in configurations
        ]
        total_costs = [
            max(configuration.get("total_cost", 0.0), 1e-5)
            for configuration in configurations
        ]
        positions = list(range(len(configurations)))

        ax.bar(
            [position - 0.18 for position in positions],
            relative_costs,
            width=0.34,
            label="F relativo",
            color="#4C9ED9",
            edgecolor="#1F4F73",
            linewidth=1.0,
        )
        ax.bar(
            [position + 0.18 for position in positions],
            total_costs,
            width=0.34,
            label="F total con prior",
            color="#F6C85F",
            edgecolor="#8A6D1D",
            linewidth=1.0,
        )

        ax.set_yscale("log")
        minimum_cost = min(relative_costs + total_costs)
        maximum_cost = max(relative_costs + total_costs)
        ax.set_ylim(max(minimum_cost * 0.55, 1e-6), maximum_cost * 4.0)
        ax.set_xticks(positions, labels)
        ax.set_ylabel("coste (escala log)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis="y", alpha=0.20)
        ax.legend(fontsize=6.8, loc="upper left", ncol=2)

        for position, configuration in zip(positions, configurations):
            ax.text(
                position - 0.18,
                relative_costs[position] * 1.25,
                f"{configuration.get('relative_cost', 0.0):.3f}",
                fontsize=6.0,
                ha="center",
                va="bottom",
                rotation=90,
            )
            ax.text(
                position + 0.18,
                total_costs[position] * 1.20,
                f"{configuration.get('total_cost', 0.0):.1f}",
                fontsize=6.0,
                ha="center",
                va="bottom",
                rotation=90,
            )

    def _dibujar_algebra_prior(self, ax, state):
        """Muestra invariancia, rango, nulidad y conexiones externas."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.955,
            "Observabilidad y anclaje",
            fontsize=9.7,
            fontweight="bold",
            ha="center",
            va="top",
        )

        ax.text(
            0.04,
            0.805,
            r"Sin prior:  $F_{rel}(g\oplus X)=F_{rel}(X)$",
            fontsize=8.2,
            ha="left",
            va="center",
            color="#5A316B",
        )

        if state.get("show_rank_without"):
            left_text = (
                f"J: {tuple(state.get('shape_without_prior', []))}\n"
                f"rango: {state.get('rank_without_prior', 0)}\n"
                f"nulidad: {state.get('nullity_without_prior', 0)}"
            )
        else:
            left_text = "J relativo\n3 direcciones globales\naún no observables"

        if state.get("show_rank_with"):
            right_text = (
                f"J: {tuple(state.get('shape_with_prior', []))}\n"
                f"rango: {state.get('rank_with_prior', 0)}\n"
                f"nulidad: {state.get('nullity_with_prior', 0)}"
            )
        else:
            right_text = "J con prior\nse mostrará al conectar\nel factor absoluto"

        cards = [
            (0.06, 0.43, "SIN PRIOR", left_text, "#FCE6D4", "#F28E2B"),
            (0.54, 0.43, "CON PRIOR", right_text, "#DDF1E5", "#2E8B57"),
        ]

        for x, y, title, body, face_color, edge_color in cards:
            rectangle = Rectangle(
                (x, y),
                0.40,
                0.27,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=1.5,
            )
            ax.add_patch(rectangle)
            ax.text(
                x + 0.20,
                y + 0.225,
                title,
                fontsize=7.7,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax.text(
                x + 0.20,
                y + 0.105,
                body,
                fontsize=6.9,
                ha="center",
                va="center",
                linespacing=1.35,
            )

        if state.get("show_connections"):
            labels = ["origen mapa", "landmark", "GPS", "Graph SLAM"]
            starts = [0.05, 0.285, 0.52, 0.755]

            for x, label in zip(starts, labels):
                rectangle = Rectangle(
                    (x, 0.105),
                    0.19,
                    0.17,
                    facecolor="#F4F4F4",
                    edgecolor="#666666",
                    linewidth=1.1,
                )
                ax.add_patch(rectangle)
                ax.text(
                    x + 0.095,
                    0.19,
                    label,
                    fontsize=6.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

            ax.text(
                0.50,
                0.045,
                "Un prior de gauge elige coordenadas; un prior físico aporta una observación absoluta.",
                fontsize=6.7,
                ha="center",
                va="center",
                color="#444444",
            )
        else:
            norms = state.get("gauge_projection_norms", {})
            ax.text(
                0.50,
                0.185,
                (
                    "||J·tx|| = "
                    f"{norms.get('translation_x', 0.0):.1e}   ·   "
                    "||J·ty|| = "
                    f"{norms.get('translation_y', 0.0):.1e}   ·   "
                    "||J·rot|| = "
                    f"{norms.get('rotation', 0.0):.1e}"
                ),
                fontsize=6.8,
                ha="center",
                va="center",
                color="#5A316B",
            )
            ax.text(
                0.50,
                0.070,
                "El prior añade tres ecuaciones y elimina el gauge de SE(2).",
                fontsize=7.1,
                fontweight="bold",
                ha="center",
                va="center",
                color="#1F4F73",
            )

    def _dibujar_estado_prior_anclaje(
        self,
        info_ax,
        without_prior_ax,
        with_prior_ax,
        cost_ax,
        algebra_ax,
        state,
    ):
        """Dibuja un fotograma completo del apartado 5.5."""

        self._dibujar_panel_prior(info_ax, state)
        self._dibujar_sin_prior(without_prior_ax, state)
        self._dibujar_con_prior(with_prior_ax, state)
        self._dibujar_costes_prior(cost_ax, state)
        self._dibujar_algebra_prior(algebra_ax, state)

    def animate_prior_graph_anchoring(
        self,
        graph_without_prior,
        graph_with_prior,
        states,
        title="Priors y anclaje del grafo",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima un pose graph antes y después de añadir un prior.

        La imagen final muestra:
        - varias copias globales con el mismo coste relativo;
        - la traslación y la rotación de gauge de SE(2);
        - un factor prior unario conectado a x0;
        - el coste absoluto de las copias incompatibles;
        - rango y nulidad antes y después del anclaje.
        """

        if not states:
            raise ValueError(
                "La lista de estados de priors y anclaje no puede estar vacía."
            )

        if graph_without_prior.is_directed() or graph_with_prior.is_directed():
            raise ValueError("Los pose graphs del ejemplo deben ser no dirigidos.")

        required_poses = {"x0", "x1", "x2", "x3"}

        if not required_poses.issubset(graph_without_prior.nodes()):
            raise ValueError("El grafo relativo debe contener x0, x1, x2 y x3.")
        if "prior_x0" not in graph_with_prior.nodes():
            raise ValueError("El grafo anclado debe contener prior_x0.")
        if not graph_with_prior.has_edge("prior_x0", "x0"):
            raise ValueError("El factor prior debe estar conectado a x0.")

        (
            fig,
            info_ax,
            without_prior_ax,
            with_prior_ax,
            cost_ax,
            algebra_ax,
        ) = self._preparar_figura_prior_anclaje(title)

        if final_image_path is not None:
            self._dibujar_estado_prior_anclaje(
                info_ax=info_ax,
                without_prior_ax=without_prior_ax,
                with_prior_ax=with_prior_ax,
                cost_ax=cost_ax,
                algebra_ax=algebra_ax,
                state=states[-1],
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(final_image_path, dpi=200, bbox_inches="tight")
            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_prior_anclaje(
                info_ax=info_ax,
                without_prior_ax=without_prior_ax,
                with_prior_ax=with_prior_ax,
                cost_ax=cost_ax,
                algebra_ax=algebra_ax,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_prior_anclaje(
                info_ax=info_ax,
                without_prior_ax=without_prior_ax,
                with_prior_ax=with_prior_ax,
                cost_ax=cost_ax,
                algebra_ax=algebra_ax,
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
    # Elementos específicos de optimización no lineal iterativa
    # ------------------------------------------------------------------

    def _preparar_figura_optimizacion_no_lineal(self, title):
        """
        Crea una figura didáctica para el ajuste no lineal.

        Distribución:
        - izquierda: fase, parámetros y magnitudes de la iteración;
        - derecha superior: puntos, residuos y curvas;
        - derecha inferior izquierda: coste por intento;
        - derecha inferior derecha: sistema local y conexiones con SLAM.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[1.55, 3.10, 2.10],
            height_ratios=[3.85, 2.15],
            wspace=0.10,
            hspace=0.13,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        curve_ax = fig.add_subplot(grid[0, 1:])
        cost_ax = fig.add_subplot(grid[1, 1])
        diagnostics_ax = fig.add_subplot(grid[1, 2])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.925,
            bottom=0.055,
        )

        return fig, info_ax, curve_ax, cost_ax, diagnostics_ax

    @staticmethod
    def _formatear_numero_optimizacion(valor, precision=4):
        """Formatea magnitudes finitas y valores especiales."""

        if valor is None:
            return "—"

        valor = float(valor)

        if valor != valor:
            return "NaN"
        if valor == float("inf"):
            return "∞"
        if valor == float("-inf"):
            return "-∞"

        magnitud = abs(valor)

        if magnitud != 0.0 and (magnitud >= 1e4 or magnitud < 1e-3):
            return f"{valor:.2e}"

        return f"{valor:.{precision}f}"

    def _dibujar_panel_optimizacion_no_lineal(self, ax, state):
        """Dibuja la fase, los parámetros y las métricas actuales."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        phase_titles = {
            "introduction": "PROBLEMA NO LINEAL",
            "true_model": "MODELO DE REFERENCIA",
            "measurements": "MEDICIONES CON RUIDO",
            "initial_curve": "ESTIMACIÓN INICIAL",
            "initial_residuals": "RESIDUOS INICIALES",
            "linearization": "LINEALIZACIÓN LOCAL",
            "proposal": "PROPUESTA DE PASO",
            "accepted": "PASO ACEPTADO",
            "updated": "NUEVA ESTIMACIÓN",
            "rejected": "PASO RECHAZADO",
            "convergence": "CONVERGENCIA",
            "connections": "CONEXIÓN CON SLAM",
            "summary": "RESULTADO FINAL",
        }

        phase = state.get("phase", "")
        title = phase_titles.get(phase, phase.upper())

        if state.get("accepted") is True:
            title_color = "#2E8B57"
        elif state.get("accepted") is False:
            title_color = "#C62828"
        else:
            title_color = "#1F4F73"

        ax.text(
            0.50,
            0.975,
            title,
            fontsize=11.2,
            fontweight="bold",
            ha="center",
            va="top",
            color=title_color,
        )

        ax.text(
            0.50,
            0.932,
            (
                f"Estado {state.get('step', 1)}"
                f" de {state.get('total_steps', 1)}"
            ),
            fontsize=7.5,
            ha="center",
            va="top",
            color="#555555",
        )

        message = state.get("message", "")

        ax.text(
            0.50,
            0.855,
            message,
            fontsize=7.6,
            ha="center",
            va="center",
            wrap=True,
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.42",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.98,
            },
        )

        parameters = list(state.get("current_parameters", [0, 0, 0, 0]))
        names = [
            ("a", "amplitud"),
            ("b", "frecuencia"),
            ("c", "fase"),
            ("d", "offset"),
        ]

        ax.text(
            0.08,
            0.735,
            "Parámetros actuales",
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            va="center",
        )

        y_positions = [0.665, 0.590, 0.515, 0.440]

        for (symbol, label), value, y in zip(names, parameters, y_positions):
            rectangle = Rectangle(
                (0.08, y - 0.032),
                0.84,
                0.064,
                facecolor="#F4F4F4",
                edgecolor="#777777",
                linewidth=1.0,
            )
            ax.add_patch(rectangle)
            ax.text(
                0.14,
                y,
                symbol,
                fontsize=8.4,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax.text(
                0.25,
                y,
                label,
                fontsize=6.5,
                ha="left",
                va="center",
                color="#555555",
            )
            ax.text(
                0.87,
                y,
                self._formatear_numero_optimizacion(value, 5),
                fontsize=7.2,
                fontweight="bold",
                ha="right",
                va="center",
            )

        trial = state.get("trial")
        trial_text = "—" if trial is None else str(int(trial))

        metrics = [
            ("intento", trial_text),
            (
                "coste",
                self._formatear_numero_optimizacion(
                    state.get("current_cost", 0.0),
                    5,
                ),
            ),
            (
                "λ",
                self._formatear_numero_optimizacion(
                    state.get("lambda", 0.0),
                    3,
                ),
            ),
            (
                "||Δθ||",
                self._formatear_numero_optimizacion(
                    state.get("step_norm", 0.0),
                    3,
                ),
            ),
            (
                "||g||",
                self._formatear_numero_optimizacion(
                    state.get("gradient_norm", 0.0),
                    3,
                ),
            ),
            (
                "cond(H)",
                self._formatear_numero_optimizacion(
                    state.get("condition_number", 0.0),
                    3,
                ),
            ),
        ]

        ax.text(
            0.08,
            0.355,
            "Magnitudes de la iteración",
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            va="center",
        )

        metric_y = 0.300
        for index, (label, value) in enumerate(metrics):
            column = index % 2
            row = index // 2
            x = 0.08 + column * 0.44
            y = metric_y - row * 0.073

            rectangle = Rectangle(
                (x, y - 0.027),
                0.40,
                0.054,
                facecolor="#EDF3F8",
                edgecolor="#7A9CB8",
                linewidth=0.9,
            )
            ax.add_patch(rectangle)
            ax.text(
                x + 0.03,
                y,
                label,
                fontsize=6.2,
                ha="left",
                va="center",
                color="#555555",
            )
            ax.text(
                x + 0.37,
                y,
                value,
                fontsize=6.4,
                fontweight="bold",
                ha="right",
                va="center",
            )

        accepted = state.get("accepted")

        if accepted is True:
            status_text = "ACEPTADO · el coste disminuye"
            face_color = "#DDF1E5"
            edge_color = "#2E8B57"
        elif accepted is False:
            status_text = "RECHAZADO · aumenta el damping"
            face_color = "#FCE0E0"
            edge_color = "#C62828"
        else:
            status_text = "Se evalúa una aproximación local"
            face_color = "#FBEBCB"
            edge_color = "#8A6D1D"

        ax.text(
            0.50,
            0.055,
            status_text,
            fontsize=7.4,
            fontweight="bold",
            ha="center",
            va="center",
            color=edge_color,
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": face_color,
                "ec": edge_color,
                "alpha": 0.98,
            },
        )

    def _dibujar_curvas_optimizacion_no_lineal(self, ax, state):
        """Dibuja puntos, residuos y curvas inicial, candidata y final."""

        ax.clear()
        ax.grid(True, alpha=0.18)
        ax.set_title(
            "Ajuste de una curva sinusoidal a puntos con ruido",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)

        x_values = list(state.get("x_values", []))
        y_values = list(state.get("y_values", []))
        visible_points = min(
            int(state.get("visible_points", len(x_values))),
            len(x_values),
        )

        if x_values:
            x_min = min(x_values)
            x_max = max(x_values)
            y_candidates = list(y_values)

            for key in [
                "true_curve",
                "initial_curve",
                "current_curve",
                "candidate_curve",
                "final_curve",
            ]:
                values = state.get(key)
                if values:
                    y_candidates.extend(values)

            y_min = min(y_candidates)
            y_max = max(y_candidates)
            margin = max(0.25, 0.12 * (y_max - y_min))

            ax.set_xlim(x_min - 0.10, x_max + 0.10)
            ax.set_ylim(y_min - margin, y_max + margin)

        if state.get("show_true_curve"):
            ax.plot(
                x_values,
                state.get("true_curve", []),
                color="#555555",
                linewidth=2.0,
                linestyle="dashed",
                label="curva verdadera",
                zorder=10,
            )

        if visible_points > 0:
            ax.scatter(
                x_values[:visible_points],
                y_values[:visible_points],
                s=28,
                color="#222222",
                edgecolors="white",
                linewidths=0.7,
                label="mediciones",
                zorder=35,
            )

        if state.get("show_initial_curve"):
            ax.plot(
                x_values,
                state.get("initial_curve", []),
                color="#F28E2B",
                linewidth=2.0,
                linestyle="dotted",
                alpha=0.75,
                label="curva inicial",
                zorder=13,
            )

        if state.get("show_residuals") and visible_points > 0:
            current_curve = list(state.get("current_curve", []))

            for index in range(0, visible_points, 2):
                ax.plot(
                    [x_values[index], x_values[index]],
                    [current_curve[index], y_values[index]],
                    color="#8E5EA2",
                    linewidth=1.15,
                    alpha=0.55,
                    zorder=20,
                )

        if (
            state.get("show_current_curve")
            and not state.get("show_final_curve")
        ):
            current_label = (
                "curva final"
                if state.get("phase") in {
                    "convergence",
                    "connections",
                    "summary",
                }
                else "estimación actual"
            )
            ax.plot(
                x_values,
                state.get("current_curve", []),
                color="#4C9ED9",
                linewidth=2.8,
                label=current_label,
                zorder=28,
            )

        if state.get("show_candidate_curve") and state.get("candidate_curve"):
            accepted = state.get("accepted")
            candidate_color = (
                "#2E8B57"
                if accepted is True
                else "#C62828"
                if accepted is False
                else "#8E5EA2"
            )
            ax.plot(
                x_values,
                state.get("candidate_curve", []),
                color=candidate_color,
                linewidth=2.4,
                linestyle="dashdot",
                alpha=0.88,
                label="candidata",
                zorder=30,
            )

        if state.get("show_final_curve"):
            ax.plot(
                x_values,
                state.get("final_curve", []),
                color="#2E8B57",
                linewidth=3.4,
                label="ajuste final",
                zorder=32,
            )

        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for handle, label in zip(handles, labels):
            unique[label] = handle

        if unique:
            ax.legend(
                unique.values(),
                unique.keys(),
                fontsize=7.0,
                loc="upper right",
                ncol=2,
                framealpha=0.96,
            )

        ax.text(
            0.015,
            0.025,
            r"$y=a\sin(bx+c)+d$",
            transform=ax.transAxes,
            fontsize=8.2,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.95,
            },
        )

        ax.text(
            0.985,
            0.025,
            (
                "F actual = "
                + self._formatear_numero_optimizacion(
                    state.get("current_cost", 0.0),
                    5,
                )
            ),
            transform=ax.transAxes,
            fontsize=7.5,
            fontweight="bold",
            ha="right",
            va="bottom",
            color="#1F4F73",
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "#7A9CB8",
                "alpha": 0.95,
            },
        )

    def _dibujar_coste_optimizacion_no_lineal(self, ax, state):
        """Dibuja el coste de pasos aceptados y candidatos rechazados."""

        ax.clear()
        ax.set_title(
            "Coste por intento",
            fontsize=9.8,
            fontweight="bold",
        )
        ax.set_xlabel("intento", fontsize=7.5)
        ax.set_ylabel("F (escala log)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.20)

        if not state.get("show_cost_history"):
            ax.text(
                0.50,
                0.52,
                "El historial aparecerá\nal iniciar las iteraciones.",
                transform=ax.transAxes,
                fontsize=8.0,
                ha="center",
                va="center",
                color="#555555",
                bbox={
                    "boxstyle": "round,pad=0.40",
                    "fc": "white",
                    "ec": "#999999",
                },
            )
            return

        history = list(state.get("history", []))
        visible_count = min(
            int(state.get("visible_history_count", 0)),
            len(history),
        )
        visible = history[:visible_count]

        accepted_x = [0]
        accepted_y = [float(state.get("initial_cost", 1.0))]
        accepted_candidate_x = []
        accepted_candidate_y = []
        rejected_x = []
        rejected_y = []

        for record in visible:
            x_position = int(record.get("trial", 0)) + 1
            candidate_cost = float(record.get("candidate_cost", float("inf")))

            if candidate_cost <= 0.0 or candidate_cost == float("inf"):
                continue

            if record.get("accepted"):
                accepted_candidate_x.append(x_position)
                accepted_candidate_y.append(candidate_cost)
                accepted_x.append(x_position)
                accepted_y.append(candidate_cost)
            else:
                rejected_x.append(x_position)
                rejected_y.append(candidate_cost)

        ax.plot(
            accepted_x,
            accepted_y,
            color="#2E8B57",
            linewidth=2.3,
            marker="o",
            markersize=4.2,
            label="coste aceptado",
            zorder=25,
        )

        if accepted_candidate_x:
            ax.scatter(
                accepted_candidate_x,
                accepted_candidate_y,
                s=30,
                color="#2E8B57",
                edgecolors="white",
                linewidths=0.6,
                label="propuesta aceptada",
                zorder=30,
            )

        if rejected_x:
            ax.scatter(
                rejected_x,
                rejected_y,
                s=38,
                marker="x",
                color="#C62828",
                linewidths=1.8,
                label="propuesta rechazada",
                zorder=32,
            )

            for x_position, y_value in zip(rejected_x, rejected_y):
                current_cost = None
                record = history[x_position - 1]
                current_cost = float(record.get("cost", y_value))
                ax.plot(
                    [x_position, x_position],
                    [current_cost, y_value],
                    color="#C62828",
                    linewidth=1.0,
                    linestyle="dotted",
                    alpha=0.65,
                    zorder=15,
                )

        all_positive = [
            value
            for value in accepted_y + accepted_candidate_y + rejected_y
            if value > 0.0
        ]

        if all_positive:
            ax.set_yscale("log")
            minimum = min(all_positive)
            maximum = max(all_positive)
            ax.set_ylim(minimum * 0.55, maximum * 2.2)

        total_trials = max(len(history), 1)
        ax.set_xlim(-0.4, total_trials + 0.8)

        if visible:
            last_record = visible[-1]
            status = (
                "aceptado"
                if last_record.get("accepted")
                else "rechazado"
            )
            ax.text(
                0.98,
                0.96,
                (
                    f"último: {status}\n"
                    f"λ={self._formatear_numero_optimizacion(last_record.get('lambda'), 2)}"
                ),
                transform=ax.transAxes,
                fontsize=6.7,
                ha="right",
                va="top",
                color=(
                    "#2E8B57"
                    if last_record.get("accepted")
                    else "#C62828"
                ),
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.95,
                },
            )

        ax.legend(
            fontsize=6.4,
            loc="upper right",
            framealpha=0.95,
        )

    def _dibujar_diagnostico_optimizacion_no_lineal(self, ax, state):
        """Muestra el sistema local, el damping y la conexión con Graph SLAM."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.965,
            "Linealización y actualización",
            fontsize=9.8,
            fontweight="bold",
            ha="center",
            va="top",
        )

        if state.get("show_linearization"):
            equations = [
                r"$e(\theta+\Delta\theta)\approx e+J\Delta\theta$",
                r"$H=J^TWJ$",
                r"$g=J^TWe$",
                r"$(H+\lambda\,\mathrm{diag}(H))\Delta\theta=-g$",
            ]

            y_positions = [0.84, 0.72, 0.60, 0.48]

            for equation, y in zip(equations, y_positions):
                ax.text(
                    0.50,
                    y,
                    equation,
                    fontsize=8.3,
                    ha="center",
                    va="center",
                    color="#1F4F73",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "fc": "#EDF3F8",
                        "ec": "#7A9CB8",
                        "alpha": 0.96,
                    },
                )
        else:
            ax.text(
                0.50,
                0.68,
                "Cada iteración aproxima\nlocalmente el problema.",
                fontsize=8.3,
                ha="center",
                va="center",
                color="#555555",
                linespacing=1.5,
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "fc": "white",
                    "ec": "#999999",
                },
            )

        if state.get("show_damping"):
            rho = state.get("rho", 0.0)
            accepted = state.get("accepted")
            if accepted is True:
                damping_text = "paso bueno → λ disminuye"
                damping_color = "#2E8B57"
            elif accepted is False:
                damping_text = "paso malo → λ aumenta"
                damping_color = "#C62828"
            else:
                damping_text = "λ controla la confianza en el modelo local"
                damping_color = "#8A6D1D"

            ax.text(
                0.50,
                0.355,
                damping_text,
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
                color=damping_color,
            )

            ax.text(
                0.50,
                0.275,
                (
                    "ρ = reducción real / predicha = "
                    + self._formatear_numero_optimizacion(rho, 4)
                ),
                fontsize=6.8,
                ha="center",
                va="center",
                color="#444444",
            )

        if state.get("show_connections"):
            labels = [
                ("ajuste curva", "#FBE5A6", "#8A6D1D"),
                ("Gauss-Newton", "#B7D7F0", "#1F4F73"),
                ("Levenberg-\nMarquardt", "#D8C4E8", "#5A316B"),
                ("Graph SLAM", "#B7E4C7", "#2E8B57"),
            ]
            x_positions = [0.03, 0.275, 0.52, 0.765]

            for index, ((label, face, edge), x) in enumerate(
                zip(labels, x_positions)
            ):
                rectangle = Rectangle(
                    (x, 0.055),
                    0.205,
                    0.135,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.2,
                )
                ax.add_patch(rectangle)
                ax.text(
                    x + 0.1025,
                    0.122,
                    label,
                    fontsize=6.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

                if index < len(labels) - 1:
                    ax.add_patch(
                        FancyArrowPatch(
                            (x + 0.205, 0.122),
                            (x_positions[index + 1], 0.122),
                            arrowstyle="-|>",
                            mutation_scale=10,
                            linewidth=1.2,
                            color="#666666",
                        )
                    )
        else:
            current = list(state.get("current_parameters", [0, 0, 0, 0]))
            final_values = list(state.get("final_parameters", [0, 0, 0, 0]))

            current_text = "  ".join(
                f"{name}={self._formatear_numero_optimizacion(value, 3)}"
                for name, value in zip(["a", "b", "c", "d"], current)
            )
            final_text = "  ".join(
                f"{name}={self._formatear_numero_optimizacion(value, 3)}"
                for name, value in zip(["a", "b", "c", "d"], final_values)
            )

            ax.text(
                0.50,
                0.145,
                "actual: " + current_text,
                fontsize=6.5,
                ha="center",
                va="center",
                color="#1F4F73",
            )
            ax.text(
                0.50,
                0.075,
                "final: " + final_text,
                fontsize=6.5,
                ha="center",
                va="center",
                color="#2E8B57",
            )

    def _dibujar_estado_optimizacion_no_lineal(
        self,
        info_ax,
        curve_ax,
        cost_ax,
        diagnostics_ax,
        state,
    ):
        """Dibuja un fotograma completo del apartado 5.6."""

        self._dibujar_panel_optimizacion_no_lineal(info_ax, state)
        self._dibujar_curvas_optimizacion_no_lineal(curve_ax, state)
        self._dibujar_coste_optimizacion_no_lineal(cost_ax, state)
        self._dibujar_diagnostico_optimizacion_no_lineal(
            diagnostics_ax,
            state,
        )

    def animate_nonlinear_optimization(
        self,
        x_values,
        y_values,
        states,
        title="Optimización no lineal iterativa",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima el ajuste de una curva sinusoidal mediante Levenberg-Marquardt.

        La imagen final muestra:
        - puntos con ruido;
        - curva inicial, verdadera y final;
        - residuos finales;
        - coste por intento;
        - propuestas aceptadas y rechazadas;
        - ecuaciones de Gauss-Newton y Levenberg-Marquardt;
        - conexión con Graph SLAM.
        """

        if not states:
            raise ValueError(
                "La lista de estados de optimización no puede estar vacía."
            )

        x_values = list(x_values)
        y_values = list(y_values)

        if len(x_values) != len(y_values):
            raise ValueError("x_values e y_values deben tener la misma longitud.")
        if len(x_values) < 8:
            raise ValueError("Se necesitan al menos ocho puntos.")

        (
            fig,
            info_ax,
            curve_ax,
            cost_ax,
            diagnostics_ax,
        ) = self._preparar_figura_optimizacion_no_lineal(title)

        if final_image_path is not None:
            self._dibujar_estado_optimizacion_no_lineal(
                info_ax=info_ax,
                curve_ax=curve_ax,
                cost_ax=cost_ax,
                diagnostics_ax=diagnostics_ax,
                state=states[-1],
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
            self._dibujar_estado_optimizacion_no_lineal(
                info_ax=info_ax,
                curve_ax=curve_ax,
                cost_ax=cost_ax,
                diagnostics_ax=diagnostics_ax,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_optimizacion_no_lineal(
                info_ax=info_ax,
                curve_ax=curve_ax,
                cost_ax=cost_ax,
                diagnostics_ax=diagnostics_ax,
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
    # Elementos específicos de jacobianos y estructura dispersa
    # ------------------------------------------------------------------

    def _preparar_figura_matrices_slam(self, title):
        """Crea una figura con grafo, jacobiano, Hessiana e información."""

        fig = plt.figure(figsize=self.figsize)
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[1.55, 3.05, 3.80],
            height_ratios=[1.0, 1.0],
            wspace=0.12,
            hspace=0.16,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        graph_ax = fig.add_subplot(grid[:, 1])
        jacobian_ax = fig.add_subplot(grid[0, 2])
        hessian_ax = fig.add_subplot(grid[1, 2])

        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.925,
            bottom=0.055,
        )

        return fig, info_ax, graph_ax, jacobian_ax, hessian_ax

    @staticmethod
    def _formatear_porcentaje_dispersion(value):
        return f"{100.0 * float(value):.1f}%"

    def _dibujar_info_matrices_slam(self, ax, state):
        """Dibuja métricas, ecuaciones y conexiones del apartado 5.7."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        metrics = dict(state.get("metrics", {}))
        phase = state.get("phase", "")

        ax.text(
            0.50,
            0.985,
            "Estructura del sistema",
            fontsize=11.0,
            fontweight="bold",
            ha="center",
            va="top",
        )

        ax.text(
            0.50,
            0.935,
            f"Paso {state.get('step', 0)} de {state.get('total_steps', 0)}",
            fontsize=7.3,
            ha="center",
            va="top",
            color="#555555",
        )

        cards = [
            ("poses", metrics.get("pose_count", 0)),
            ("factores", metrics.get("factor_count", 0)),
            ("dim(X)", metrics.get("state_dimension", 0)),
            ("dim(e)", metrics.get("residual_dimension", 0)),
            ("nnz(J)", metrics.get("jacobian_nnz", 0)),
            (
                "densidad J",
                self._formatear_porcentaje_dispersion(
                    metrics.get("jacobian_density", 0.0)
                ),
            ),
            ("nnz(H)", metrics.get("hessian_nnz", 0)),
            (
                "densidad H",
                self._formatear_porcentaje_dispersion(
                    metrics.get("hessian_density", 0.0)
                ),
            ),
        ]

        y = 0.865
        for index, (label, value) in enumerate(cards):
            column = index % 2
            row = index // 2
            x = 0.055 + column * 0.46
            card_y = y - row * 0.074

            rectangle = Rectangle(
                (x, card_y - 0.027),
                0.405,
                0.055,
                facecolor="#EDF3F8",
                edgecolor="#7A9CB8",
                linewidth=0.9,
            )
            ax.add_patch(rectangle)
            ax.text(
                x + 0.025,
                card_y,
                label,
                fontsize=6.1,
                ha="left",
                va="center",
                color="#555555",
            )
            ax.text(
                x + 0.375,
                card_y,
                str(value),
                fontsize=6.4,
                fontweight="bold",
                ha="right",
                va="center",
            )

        equations = [
            r"$e_k=e_k(x_{i_1},\ldots,x_{i_p})$",
            r"$J=\partial e/\partial X$",
            r"$H=J^T\Omega J$",
            r"$g=J^T\Omega e$",
            r"$H\Delta X=-g$",
        ]

        for equation, equation_y in zip(
            equations,
            [0.535, 0.465, 0.395, 0.325, 0.255],
        ):
            ax.text(
                0.50,
                equation_y,
                equation,
                fontsize=8.0,
                ha="center",
                va="center",
                color="#1F4F73",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "fc": "#EDF3F8",
                    "ec": "#7A9CB8",
                    "alpha": 0.96,
                },
            )

        gauge_text = (
            "sin prior: rango "
            f"{metrics.get('rank_without_prior', 0)}, "
            f"nulidad {metrics.get('nullity_without_prior', 0)}\n"
            "con prior: rango "
            f"{metrics.get('rank_with_prior', 0)}, "
            f"nulidad {metrics.get('nullity_with_prior', 0)}"
        )
        ax.text(
            0.50,
            0.175,
            gauge_text,
            fontsize=6.9,
            ha="center",
            va="center",
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.32",
                "fc": "#FBEBCB",
                "ec": "#8A6D1D",
                "alpha": 0.97,
            },
        )

        if state.get("show_fill_in"):
            active = state.get("active_elimination_variable") or "—"
            fill_count = len(state.get("fill_edges", []))
            fill_text = (
                f"eliminando: {active}\n"
                f"fill-in acumulado: {fill_count}\n"
                f"orden bueno/malo: "
                f"{metrics.get('good_fill_count', 0)} / "
                f"{metrics.get('bad_fill_count', 0)}"
            )
        else:
            fill_text = (
                "bloques J: "
                f"{metrics.get('block_jacobian_nnz', 0)}/"
                f"{metrics.get('block_jacobian_total', 0)}\n"
                "bloques H: "
                f"{metrics.get('block_hessian_nnz', 0)}/"
                f"{metrics.get('block_hessian_total', 0)}"
            )

        ax.text(
            0.50,
            0.085,
            fill_text,
            fontsize=6.7,
            fontweight="bold",
            ha="center",
            va="center",
            color="#5A316B" if state.get("show_fill_in") else "#444444",
        )

        if state.get("show_connections"):
            labels = [
                ("GTSAM", "#B7D7F0", "#1F4F73"),
                ("g2o", "#D8C4E8", "#5A316B"),
                ("Ceres", "#FBE5A6", "#8A6D1D"),
                ("SLAM grande", "#B7E4C7", "#2E8B57"),
            ]
            positions = [0.03, 0.275, 0.52, 0.765]
            for label, face, edge in labels:
                pass
            for (label, face, edge), x in zip(labels, positions):
                rectangle = Rectangle(
                    (x, 0.005),
                    0.205,
                    0.042,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.0,
                )
                ax.add_patch(rectangle)
                ax.text(
                    x + 0.1025,
                    0.026,
                    label,
                    fontsize=5.8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )

        phase_labels = {
            "introduction": "localidad de los factores",
            "poses": "vector global de poses",
            "prior": "factor unario",
            "factor": "jacobiano local",
            "jacobian": "ensamblaje de J",
            "normal_equations": "ecuaciones normales",
            "assembly": "contribuciones a H",
            "gauge": "rango y anclaje",
            "good_elimination": "orden favorable",
            "bad_elimination": "orden desfavorable",
            "connections": "bibliotecas de optimización",
            "summary": "resumen final",
        }
        ax.text(
            0.50,
            0.615,
            phase_labels.get(phase, phase),
            fontsize=7.1,
            fontweight="bold",
            ha="center",
            va="center",
            color="#7A1D1D",
        )

    @staticmethod
    def _dibujar_orientacion_pose_dispersa(ax, x, y, theta, color):
        longitud = 0.34
        ax.arrow(
            x,
            y,
            longitud * cos(theta),
            longitud * sin(theta),
            width=0.018,
            head_width=0.11,
            head_length=0.12,
            length_includes_head=True,
            color=color,
            zorder=36,
        )

    def _dibujar_grafo_matrices_slam(self, ax, state):
        """Dibuja poses, factores, cierres y aristas de fill-in."""

        ax.clear()
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            "Pose graph y dependencias locales",
            fontsize=11.2,
            fontweight="bold",
        )

        poses = dict(state.get("poses", {}))
        variable_names = list(state.get("variable_names", []))
        factors = list(state.get("factors", []))
        visible_poses = min(
            int(state.get("visible_poses", 0)),
            len(variable_names),
        )
        visible_factors = min(
            int(state.get("visible_factors", 0)),
            len(factors),
        )
        active_factor = state.get("active_factor")
        eliminated = set(state.get("eliminated_variables", []))
        active_elimination = state.get("active_elimination_variable")

        coordinates = {
            name: (poses[name][0], poses[name][1])
            for name in variable_names
            if name in poses
        }

        if coordinates:
            xs = [value[0] for value in coordinates.values()]
            ys = [value[1] for value in coordinates.values()]
            ax.set_xlim(min(xs) - 1.1, max(xs) + 1.1)
            ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)

        visible_names = set(variable_names[:visible_poses])

        for factor in factors[:visible_factors]:
            name = factor.get("name")
            variables = list(factor.get("variables", []))
            factor_type = factor.get("type")

            if name == "prior_x0":
                if "x0" not in visible_names:
                    continue
                x0, y0 = coordinates["x0"]
                prior_position = (x0 - 0.72, y0 - 0.58)
                ax.plot(
                    [prior_position[0], x0],
                    [prior_position[1], y0],
                    color="#C62828" if name == active_factor else "#E45756",
                    linewidth=3.5 if name == active_factor else 2.4,
                    zorder=17,
                )
                rectangle = Rectangle(
                    (prior_position[0] - 0.23, prior_position[1] - 0.15),
                    0.46,
                    0.30,
                    facecolor="#F6B4B4",
                    edgecolor="#7A1D1D",
                    linewidth=2.3 if name == active_factor else 1.4,
                    zorder=26,
                )
                ax.add_patch(rectangle)
                ax.text(
                    prior_position[0],
                    prior_position[1],
                    "prior",
                    fontsize=7.2,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=30,
                )
                continue

            if len(variables) != 2 or not set(variables).issubset(visible_names):
                continue

            origin, target = variables
            x1, y1 = coordinates[origin]
            x2, y2 = coordinates[target]

            if factor_type == "loop_closure":
                color = "#8E5EA2"
                line_style = "dashed"
                line_width = 2.8
            else:
                color = "#2E8B57"
                line_style = "solid"
                line_width = 2.4

            if name == active_factor:
                color = "#E45756"
                line_width = 4.2

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                zorder=15,
            )
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.10,
                name,
                fontsize=6.1,
                fontweight="bold" if name == active_factor else "normal",
                ha="center",
                va="center",
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.13",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.88,
                },
                zorder=24,
            )

        for origin, target in state.get("fill_edges", []):
            if origin not in coordinates or target not in coordinates:
                continue
            x1, y1 = coordinates[origin]
            x2, y2 = coordinates[target]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="#F28E2B",
                linewidth=2.7,
                linestyle="dotted",
                alpha=0.95,
                zorder=19,
            )

        for name in variable_names[:visible_poses]:
            x, y = coordinates[name]

            if name == active_elimination:
                face = "#E45756"
                edge = "#7A1D1D"
                size = 980
            elif name in eliminated:
                face = "#D9D9D9"
                edge = "#888888"
                size = 720
            elif name == "x0" and visible_factors >= 1:
                face = "#F6C85F"
                edge = "#8A6D1D"
                size = 850
            else:
                face = "#4C9ED9"
                edge = "#1F4F73"
                size = 820

            collection = nx.draw_networkx_nodes(
                nx.Graph([(name, name)]),
                {name: (x, y)},
                nodelist=[name],
                node_size=size,
                node_color=face,
                edgecolors=edge,
                linewidths=2.2 if name == active_elimination else 1.5,
                ax=ax,
            )
            collection.set_zorder(28)
            ax.text(
                x,
                y,
                name,
                fontsize=9.0,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=34,
            )
            self._dibujar_orientacion_pose_dispersa(
                ax,
                x,
                y,
                poses[name][2],
                edge,
            )

        ax.text(
            0.50,
            0.015,
            state.get("message", ""),
            transform=ax.transAxes,
            fontsize=8.4,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.36",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=50,
        )

        legend_handles = [
            Line2D([0], [0], color="#2E8B57", linewidth=2.5, label="odometría"),
            Line2D([0], [0], color="#8E5EA2", linewidth=2.5, linestyle="dashed", label="cierre"),
            Line2D([0], [0], color="#E45756", linewidth=2.5, label="prior / activo"),
            Line2D([0], [0], color="#F28E2B", linewidth=2.5, linestyle="dotted", label="fill-in"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            fontsize=6.8,
            framealpha=0.95,
            ncol=2,
        )

    @staticmethod
    def _dibujar_rejilla_bloques(ax, rows, columns, block_size=3):
        for value in range(0, rows + 1, block_size):
            ax.axhline(value - 0.5, color="#777777", linewidth=0.55, alpha=0.65)
        for value in range(0, columns + 1, block_size):
            ax.axvline(value - 0.5, color="#777777", linewidth=0.55, alpha=0.65)

    def _dibujar_jacobiano_disperso(self, ax, state):
        """Dibuja el patrón escalar de J y resalta los bloques activos."""

        ax.clear()
        ax.set_title(
            "Jacobiano global J · filas=factores, columnas=poses",
            fontsize=9.8,
            fontweight="bold",
        )

        pattern = np.asarray(state.get("jacobian_pattern", []), dtype=bool)
        factor_names = list(state.get("factor_names", []))
        variable_names = list(state.get("variable_names", []))

        if not state.get("show_jacobian") or pattern.size == 0:
            ax.axis("off")
            ax.text(
                0.50,
                0.52,
                "El patrón de J aparecerá\nal añadir los factores.",
                transform=ax.transAxes,
                fontsize=8.5,
                ha="center",
                va="center",
                color="#555555",
                bbox={
                    "boxstyle": "round,pad=0.40",
                    "fc": "white",
                    "ec": "#999999",
                },
            )
            return

        visible_factors = min(
            int(state.get("visible_factors", 0)),
            len(factor_names),
        )
        visible_rows = 3 * visible_factors
        shown = np.zeros_like(pattern)
        shown[:visible_rows, :] = pattern[:visible_rows, :]

        ax.imshow(
            shown,
            cmap="Greys",
            interpolation="nearest",
            aspect="auto",
            vmin=0,
            vmax=1,
        )

        if state.get("show_block_grid"):
            self._dibujar_rejilla_bloques(
                ax,
                shown.shape[0],
                shown.shape[1],
                block_size=3,
            )

        for factor_index, variable_index in state.get("active_j_blocks", []):
            rectangle = Rectangle(
                (3 * variable_index - 0.5, 3 * factor_index - 0.5),
                3,
                3,
                fill=False,
                edgecolor="#E45756",
                linewidth=2.2,
            )
            ax.add_patch(rectangle)

        ax.set_xticks(
            [3 * index + 1 for index in range(len(variable_names))]
        )
        ax.set_xticklabels(variable_names, fontsize=6.8)
        ax.set_yticks(
            [3 * index + 1 for index in range(len(factor_names))]
        )
        ax.set_yticklabels(factor_names, fontsize=5.8)
        ax.tick_params(length=0)
        ax.set_xlabel("bloques de variables", fontsize=7.0)
        ax.set_ylabel("bloques de residuos", fontsize=7.0)

        metrics = state.get("metrics", {})
        ax.text(
            0.99,
            0.02,
            (
                f"forma {tuple(metrics.get('jacobian_shape', []))} · "
                f"nnz {metrics.get('jacobian_nnz', 0)} · "
                f"densidad "
                f"{self._formatear_porcentaje_dispersion(metrics.get('jacobian_density', 0.0))}"
            ),
            transform=ax.transAxes,
            fontsize=6.4,
            ha="right",
            va="bottom",
            color="#1F4F73",
            bbox={
                "boxstyle": "round,pad=0.20",
                "fc": "white",
                "ec": "#7A9CB8",
                "alpha": 0.94,
            },
        )

    def _dibujar_hessiana_dispersa(self, ax, state):
        """Dibuja H, bloques activos y enlaces simbólicos de fill-in."""

        ax.clear()
        ax.set_title(
            r"Hessiana aproximada $H=J^T\Omega J$",
            fontsize=9.8,
            fontweight="bold",
        )

        pattern = np.asarray(state.get("hessian_pattern", []), dtype=bool)
        variable_names = list(state.get("variable_names", []))

        if not state.get("show_hessian") or pattern.size == 0:
            ax.axis("off")
            ax.text(
                0.50,
                0.52,
                "H aparecerá al formar\nlas ecuaciones normales.",
                transform=ax.transAxes,
                fontsize=8.5,
                ha="center",
                va="center",
                color="#555555",
                bbox={
                    "boxstyle": "round,pad=0.40",
                    "fc": "white",
                    "ec": "#999999",
                },
            )
            return

        shown = pattern.astype(float)

        for origin, target in state.get("fill_edges", []):
            if origin not in variable_names or target not in variable_names:
                continue
            i = variable_names.index(origin)
            j = variable_names.index(target)
            shown[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = 0.55
            shown[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = 0.55

        ax.imshow(
            shown,
            cmap="Purples",
            interpolation="nearest",
            aspect="equal",
            vmin=0,
            vmax=1,
        )

        if state.get("show_block_grid"):
            self._dibujar_rejilla_bloques(
                ax,
                shown.shape[0],
                shown.shape[1],
                block_size=3,
            )

        for row_block, column_block in state.get("active_h_blocks", []):
            rectangle = Rectangle(
                (3 * column_block - 0.5, 3 * row_block - 0.5),
                3,
                3,
                fill=False,
                edgecolor="#E45756",
                linewidth=2.2,
            )
            ax.add_patch(rectangle)

        ticks = [3 * index + 1 for index in range(len(variable_names))]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(variable_names, fontsize=6.8)
        ax.set_yticklabels(variable_names, fontsize=6.8)
        ax.tick_params(length=0)
        ax.set_xlabel("variables", fontsize=7.0)
        ax.set_ylabel("variables", fontsize=7.0)

        metrics = state.get("metrics", {})
        extra = ""
        if state.get("show_fill_in"):
            extra = f" · fill-in {len(state.get('fill_edges', []))}"

        ax.text(
            0.99,
            0.02,
            (
                f"forma {tuple(metrics.get('hessian_shape', []))} · "
                f"nnz {metrics.get('hessian_nnz', 0)} · "
                f"densidad "
                f"{self._formatear_porcentaje_dispersion(metrics.get('hessian_density', 0.0))}"
                + extra
            ),
            transform=ax.transAxes,
            fontsize=6.4,
            ha="right",
            va="bottom",
            color="#5A316B",
            bbox={
                "boxstyle": "round,pad=0.20",
                "fc": "white",
                "ec": "#8E5EA2",
                "alpha": 0.94,
            },
        )

    def _dibujar_estado_matrices_slam(
        self,
        info_ax,
        graph_ax,
        jacobian_ax,
        hessian_ax,
        state,
    ):
        """Dibuja un fotograma completo del apartado 5.7."""

        self._dibujar_info_matrices_slam(info_ax, state)
        self._dibujar_grafo_matrices_slam(graph_ax, state)
        self._dibujar_jacobiano_disperso(jacobian_ax, state)
        self._dibujar_hessiana_dispersa(hessian_ax, state)

    def animate_sparse_slam_matrices(
        self,
        graph,
        states,
        title="Jacobianos y estructura dispersa en SLAM",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima la relación entre un pose graph y sus matrices dispersas.

        La imagen final muestra:
        - pose graph con prior, odometría y cierres de ciclo;
        - patrón escalar y por bloques del jacobiano;
        - Hessiana aproximada simétrica;
        - métricas de densidad, rango y fill-in;
        - conexión con GTSAM, g2o, Ceres y SLAM a gran escala.
        """

        if not states:
            raise ValueError(
                "La lista de estados de matrices dispersas no puede estar vacía."
            )
        if graph is None or graph.number_of_nodes() == 0:
            raise ValueError("El pose graph no puede estar vacío.")

        (
            fig,
            info_ax,
            graph_ax,
            jacobian_ax,
            hessian_ax,
        ) = self._preparar_figura_matrices_slam(title)

        if final_image_path is not None:
            self._dibujar_estado_matrices_slam(
                info_ax=info_ax,
                graph_ax=graph_ax,
                jacobian_ax=jacobian_ax,
                hessian_ax=hessian_ax,
                state=states[-1],
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
            self._dibujar_estado_matrices_slam(
                info_ax=info_ax,
                graph_ax=graph_ax,
                jacobian_ax=jacobian_ax,
                hessian_ax=hessian_ax,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_matrices_slam(
                info_ax=info_ax,
                graph_ax=graph_ax,
                jacobian_ax=jacobian_ax,
                hessian_ax=hessian_ax,
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
    # Elementos específicos de introducción a SLAM y deriva de odometría
    # ------------------------------------------------------------------

    def _preparar_figura_deriva_odometria(self, title):
        """Crea una figura comparable con los apartados de optimización."""

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[1.55, 3.75, 2.40],
            height_ratios=[3.45, 2.55],
            wspace=0.10,
            hspace=0.13,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        trajectory_ax = fig.add_subplot(grid[:, 1])
        error_ax = fig.add_subplot(grid[0, 2])
        diagnostic_ax = fig.add_subplot(grid[1, 2])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.925,
            bottom=0.055,
        )

        return fig, info_ax, trajectory_ax, error_ax, diagnostic_ax

    @staticmethod
    def _dibujar_flecha_pose_deriva(
        ax,
        pose,
        color,
        length=0.30,
        line_width=1.6,
        alpha=1.0,
        zorder=30,
    ):
        """Dibuja la orientación de una pose sin ocultar la trayectoria."""

        x, y, theta = np.asarray(pose, dtype=float)
        dx = length * np.cos(theta)
        dy = length * np.sin(theta)

        arrow = FancyArrowPatch(
            (x, y),
            (x + dx, y + dy),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=line_width,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(arrow)

    def _dibujar_leyenda_deriva_odometria(self, ax):
        """Dibuja la leyenda de las trayectorias y del error de cierre."""

        elementos = [
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=3.0,
                label="Trayectoria real",
            ),
            Line2D(
                [0],
                [0],
                color="#D62728",
                linewidth=3.0,
                linestyle="dashed",
                label="Odometría integrada",
            ),
            Line2D(
                [0],
                [0],
                color="#8E5EA2",
                linewidth=2.2,
                linestyle="dotted",
                label="Error entre poses",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#F6C85F",
                markeredgecolor="#8A6D1D",
                markersize=8,
                label="Inicio",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#D62728",
                markeredgecolor="#7A1D1D",
                markersize=8,
                label="Final odométrico",
            ),
        ]

        ax.legend(
            handles=elementos,
            loc="upper left",
            fontsize=7.3,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.8,
            handlelength=2.4,
        )

    @staticmethod
    def _limites_trayectorias_deriva(true_trajectory, estimated_trajectory):
        """Calcula límites comunes y estables para toda la animación."""

        puntos = np.vstack(
            (
                np.asarray(true_trajectory, dtype=float)[:, :2],
                np.asarray(estimated_trajectory, dtype=float)[:, :2],
            )
        )
        min_x, min_y = np.min(puntos, axis=0)
        max_x, max_y = np.max(puntos, axis=0)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)

        return (
            min_x - 0.12 * span_x - 0.35,
            max_x + 0.12 * span_x + 0.35,
            min_y - 0.12 * span_y - 0.35,
            max_y + 0.12 * span_y + 0.35,
        )

    def _dibujar_panel_info_deriva(self, ax, simulation, state):
        """Muestra conceptos, fase actual y métricas de la simulación."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        metrics = simulation["metrics"]
        phase = state.get("phase", "introduction")
        active_index = state.get("active_index")

        ax.text(
            0.50,
            0.985,
            "SLAM y deriva",
            fontsize=12.2,
            fontweight="bold",
            ha="center",
            va="top",
        )

        phase_labels = {
            "introduction": "problema SLAM",
            "ground_truth": "trayectoria real",
            "odometry_model": "medición relativa",
            "integration": "integración odométrica",
            "drift": "deriva acumulada",
            "loop_closure": "cierre de ciclo",
            "summary": "necesidad de Graph SLAM",
        }

        ax.text(
            0.50,
            0.936,
            phase_labels.get(phase, phase),
            fontsize=8.2,
            fontweight="bold",
            ha="center",
            va="top",
            color="#7A1D1D",
        )

        cards = [
            ("REAL", "movimiento físico", "#B7E4C7", "#2E8B57"),
            ("ODOMETRÍA", "incrementos relativos", "#F7C6C7", "#7A1D1D"),
            ("DERIVA", "error acumulado", "#D8C4E8", "#5A316B"),
            ("SLAM", "restricciones globales", "#B7D7F0", "#1F4F73"),
        ]

        y_positions = [0.820, 0.710, 0.600, 0.490]

        for (title, subtitle, face, edge), y in zip(cards, y_positions):
            rectangle = Rectangle(
                (0.10, y),
                0.80,
                0.085,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.4,
            )
            ax.add_patch(rectangle)
            ax.text(
                0.16,
                y + 0.056,
                title,
                fontsize=7.9,
                fontweight="bold",
                ha="left",
                va="center",
            )
            ax.text(
                0.16,
                y + 0.025,
                subtitle,
                fontsize=6.7,
                ha="left",
                va="center",
                color="#333333",
            )

        if phase in {"integration", "drift", "loop_closure", "summary"}:
            current_index = (
                len(simulation["true_trajectory"]) - 1
                if active_index is None
                else max(0, min(int(active_index), len(simulation["true_trajectory"]) - 1))
            )
            position_error = simulation["position_errors"][current_index]
            orientation_error = np.rad2deg(
                simulation["orientation_errors"][current_index]
            )
            current_text = (
                f"Pose actual: {current_index}\n"
                f"error posición: {position_error:.3f} m\n"
                f"error orientación: {orientation_error:.2f}°"
            )
        else:
            current_text = (
                "La estimación comienza\n"
                "en la misma pose que\n"
                "la trayectoria real."
            )

        ax.text(
            0.50,
            0.405,
            current_text,
            fontsize=7.6,
            ha="center",
            va="center",
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.42",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.98,
            },
        )

        if state.get("show_metrics"):
            metric_text = (
                f"poses: {metrics['pose_count']}\n"
                f"recorrido: {metrics['real_length']:.2f} m\n"
                f"error final: {metrics['position_final']:.3f} m\n"
                f"giro final: {metrics['orientation_final_deg']:.2f}°\n"
                f"RMSE: {metrics['position_rmse']:.3f} m\n"
                f"deriva: {metrics['drift_percent']:.2f} %"
            )
        else:
            metric_text = (
                f"poses: {metrics['pose_count']}\n"
                f"incrementos: {metrics['increment_count']}\n"
                "sin referencias absolutas\n"
                "ni cierres de ciclo"
            )

        ax.text(
            0.50,
            0.225,
            metric_text,
            fontsize=7.4,
            ha="center",
            va="center",
            linespacing=1.38,
            bbox={
                "boxstyle": "round,pad=0.42",
                "fc": "#F7F7F7",
                "ec": "#888888",
                "alpha": 0.98,
            },
        )

        if state.get("show_connections"):
            connection_text = (
                "odometría\n"
                "↓\n"
                "cierre de ciclo\n"
                "↓\n"
                "Graph SLAM"
            )
            ax.text(
                0.50,
                0.070,
                connection_text,
                fontsize=7.1,
                fontweight="bold",
                ha="center",
                va="center",
                color="#1F4F73",
            )
        else:
            ax.text(
                0.50,
                0.065,
                "Los errores locales pequeños\nse componen durante todo el recorrido.",
                fontsize=6.6,
                ha="center",
                va="center",
                color="#444444",
            )

    def _dibujar_trayectorias_deriva(self, ax, simulation, state):
        """Dibuja la trayectoria real, la odometría y el error actual."""

        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.tick_params(labelsize=7)

        true_trajectory = np.asarray(
            simulation["true_trajectory"],
            dtype=float,
        )
        estimated_trajectory = np.asarray(
            simulation["estimated_trajectory"],
            dtype=float,
        )

        limits = self._limites_trayectorias_deriva(
            true_trajectory,
            estimated_trajectory,
        )
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])

        true_count = min(
            max(int(state.get("visible_true_count", 0)), 0),
            len(true_trajectory),
        )
        estimated_count = min(
            max(int(state.get("visible_estimated_count", 0)), 0),
            len(estimated_trajectory),
        )

        if state.get("show_true") and true_count > 0:
            true_visible = true_trajectory[:true_count]
            ax.plot(
                true_visible[:, 0],
                true_visible[:, 1],
                color="#2E8B57",
                linewidth=2.8,
                label="Trayectoria real",
                zorder=15,
            )

        if state.get("show_estimated") and estimated_count > 0:
            estimated_visible = estimated_trajectory[:estimated_count]
            ax.plot(
                estimated_visible[:, 0],
                estimated_visible[:, 1],
                color="#D62728",
                linewidth=2.6,
                linestyle="dashed",
                label="Odometría integrada",
                zorder=16,
            )

        # Poses de referencia repartidas a lo largo de la trayectoria.
        stride = max(1, len(true_trajectory) // 12)
        if state.get("show_true"):
            for index in range(0, max(true_count, 1), stride):
                if index < true_count:
                    self._dibujar_flecha_pose_deriva(
                        ax,
                        true_trajectory[index],
                        color="#2E8B57",
                        length=0.24,
                        line_width=1.2,
                        alpha=0.75,
                        zorder=28,
                    )

        if state.get("show_estimated"):
            for index in range(0, max(estimated_count, 1), stride):
                if index < estimated_count:
                    self._dibujar_flecha_pose_deriva(
                        ax,
                        estimated_trajectory[index],
                        color="#D62728",
                        length=0.24,
                        line_width=1.1,
                        alpha=0.68,
                        zorder=29,
                    )

        start = true_trajectory[0]
        ax.scatter(
            [start[0]],
            [start[1]],
            s=95,
            marker="o",
            facecolor="#F6C85F",
            edgecolor="#8A6D1D",
            linewidth=1.8,
            zorder=40,
        )
        ax.text(
            start[0] + 0.12,
            start[1] - 0.18,
            "inicio",
            fontsize=7.5,
            fontweight="bold",
            color="#7A1D1D",
            zorder=42,
        )

        active_index = state.get("active_index")
        if active_index is not None:
            active_index = max(
                0,
                min(int(active_index), len(true_trajectory) - 1),
            )

            if state.get("show_true") and active_index < true_count:
                true_pose = true_trajectory[active_index]
                ax.scatter(
                    [true_pose[0]],
                    [true_pose[1]],
                    s=64,
                    marker="o",
                    facecolor="#2E8B57",
                    edgecolor="#174F32",
                    linewidth=1.4,
                    zorder=43,
                )

            if state.get("show_estimated") and active_index < estimated_count:
                estimated_pose = estimated_trajectory[active_index]
                ax.scatter(
                    [estimated_pose[0]],
                    [estimated_pose[1]],
                    s=74,
                    marker="o",
                    facecolor="#E45756",
                    edgecolor="#7A1D1D",
                    linewidth=1.6,
                    zorder=44,
                )

            if (
                state.get("show_error_vector")
                and active_index < true_count
                and active_index < estimated_count
            ):
                true_pose = true_trajectory[active_index]
                estimated_pose = estimated_trajectory[active_index]
                ax.plot(
                    [true_pose[0], estimated_pose[0]],
                    [true_pose[1], estimated_pose[1]],
                    color="#8E5EA2",
                    linewidth=2.0,
                    linestyle="dotted",
                    zorder=32,
                )

            if (
                state.get("show_increment")
                and active_index > 0
                and active_index < estimated_count
            ):
                previous = estimated_trajectory[active_index - 1]
                current = estimated_trajectory[active_index]
                arrow = FancyArrowPatch(
                    (previous[0], previous[1]),
                    (current[0], current[1]),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=2.0,
                    color="#E45756",
                    zorder=35,
                )
                ax.add_patch(arrow)

        if state.get("show_loop_closure"):
            final_estimated = estimated_trajectory[-1]
            ax.scatter(
                [final_estimated[0]],
                [final_estimated[1]],
                s=115,
                marker="X",
                facecolor="#D62728",
                edgecolor="#7A1D1D",
                linewidth=1.7,
                zorder=48,
            )
            ax.plot(
                [start[0], final_estimated[0]],
                [start[1], final_estimated[1]],
                color="#8E5EA2",
                linewidth=2.4,
                linestyle="dotted",
                zorder=34,
            )
            midpoint = 0.5 * (start[:2] + final_estimated[:2])
            ax.text(
                midpoint[0],
                midpoint[1] + 0.18,
                (
                    "error de cierre\n"
                    f"{simulation['metrics']['estimated_closure_error']:.3f} m"
                ),
                fontsize=7.2,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#5A316B",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "fc": "white",
                    "ec": "#8E5EA2",
                    "alpha": 0.95,
                },
                zorder=50,
            )

        ax.text(
            0.50,
            0.018,
            state.get("message", ""),
            transform=ax.transAxes,
            fontsize=8.5,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=60,
        )

        ax.text(
            0.99,
            0.985,
            (
                f"Paso {state.get('step', 0)}/{state.get('total_steps', 0)}"
            ),
            transform=ax.transAxes,
            fontsize=7.8,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.95,
            },
            zorder=60,
        )

        self._dibujar_leyenda_deriva_odometria(ax)

    def _dibujar_error_posicion_deriva(self, ax, simulation, state):
        """Dibuja el error global de posición acumulado por la deriva."""

        ax.clear()
        ax.grid(True, alpha=0.24)
        ax.set_title("Error global de posición", fontsize=10, fontweight="bold")
        ax.set_xlabel("Índice de pose", fontsize=7.5)
        ax.set_ylabel("Error [m]", fontsize=7.5)
        ax.tick_params(labelsize=7)

        errors = np.asarray(simulation["position_errors"], dtype=float)
        active_index = state.get("active_index")

        if active_index is None:
            visible_count = 1
        else:
            visible_count = max(1, min(int(active_index) + 1, len(errors)))

        if state.get("show_error_history"):
            indices = np.arange(visible_count)
            ax.plot(
                indices,
                errors[:visible_count],
                color="#8E5EA2",
                linewidth=2.3,
                zorder=15,
            )
            ax.fill_between(
                indices,
                0.0,
                errors[:visible_count],
                color="#D8C4E8",
                alpha=0.35,
                zorder=10,
            )
            current_index = visible_count - 1
            ax.scatter(
                [current_index],
                [errors[current_index]],
                s=48,
                facecolor="#E45756",
                edgecolor="#7A1D1D",
                linewidth=1.2,
                zorder=20,
            )

        upper = max(0.25, 1.18 * float(np.max(errors)))
        ax.set_xlim(0, len(errors) - 1)
        ax.set_ylim(0, upper)

        if state.get("show_metrics"):
            ax.axhline(
                simulation["metrics"]["position_rmse"],
                color="#1F4F73",
                linewidth=1.3,
                linestyle="dashed",
                label="RMSE",
            )
            ax.legend(loc="upper left", fontsize=6.7, framealpha=0.95)

    def _dibujar_diagnostico_deriva(self, ax, simulation, state):
        """Dibuja deriva angular y errores locales de la pose activa."""

        ax.clear()
        ax.grid(True, alpha=0.22)
        ax.set_title(
            "Deriva angular y error local",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Índice de pose", fontsize=7.5)
        ax.set_ylabel("Error angular [°]", fontsize=7.5)
        ax.tick_params(labelsize=7)

        orientation_errors = np.rad2deg(
            np.asarray(simulation["orientation_errors"], dtype=float)
        )
        active_index = state.get("active_index")

        if active_index is None:
            visible_count = 1
            active_index = 0
        else:
            active_index = max(
                0,
                min(int(active_index), len(orientation_errors) - 1),
            )
            visible_count = active_index + 1

        if state.get("show_error_history"):
            indices = np.arange(visible_count)
            ax.plot(
                indices,
                orientation_errors[:visible_count],
                color="#F28E2B",
                linewidth=2.1,
                zorder=15,
            )
            ax.scatter(
                [active_index],
                [orientation_errors[active_index]],
                s=42,
                facecolor="#E45756",
                edgecolor="#7A1D1D",
                linewidth=1.1,
                zorder=20,
            )

        max_abs = max(1.0, float(np.max(np.abs(orientation_errors))))
        ax.set_xlim(0, len(orientation_errors) - 1)
        ax.set_ylim(-0.12 * max_abs, 1.18 * max_abs)
        ax.axhline(0.0, color="#777777", linewidth=0.8)

        if active_index > 0:
            local_index = active_index - 1
            local_translation = (
                1000.0
                * simulation["increment_translation_errors"][local_index]
            )
            local_orientation = np.rad2deg(
                simulation["increment_orientation_errors"][local_index]
            )
            scale = simulation["scale_factors"][local_index]
            diagnostic_text = (
                f"incremento {local_index + 1}\n"
                f"error local: {local_translation:.2f} mm\n"
                f"error giro: {local_orientation:.3f}°\n"
                f"escala: {scale:.5f}"
            )
        else:
            diagnostic_text = (
                "La pose inicial\n"
                "no contiene deriva."
            )

        ax.text(
            0.97,
            0.06,
            diagnostic_text,
            transform=ax.transAxes,
            fontsize=6.8,
            ha="right",
            va="bottom",
            linespacing=1.32,
            bbox={
                "boxstyle": "round,pad=0.32",
                "fc": "white",
                "ec": "#888888",
                "alpha": 0.96,
            },
            zorder=30,
        )

        if state.get("show_connections"):
            ax.text(
                0.03,
                0.94,
                "cierre de ciclo → optimización global",
                transform=ax.transAxes,
                fontsize=6.8,
                fontweight="bold",
                ha="left",
                va="top",
                color="#1F4F73",
            )

    def _dibujar_estado_deriva_odometria(
        self,
        info_ax,
        trajectory_ax,
        error_ax,
        diagnostic_ax,
        simulation,
        state,
    ):
        """Dibuja un estado completo de la introducción a SLAM."""

        self._dibujar_panel_info_deriva(
            info_ax,
            simulation,
            state,
        )
        self._dibujar_trayectorias_deriva(
            trajectory_ax,
            simulation,
            state,
        )
        self._dibujar_error_posicion_deriva(
            error_ax,
            simulation,
            state,
        )
        self._dibujar_diagnostico_deriva(
            diagnostic_ax,
            simulation,
            state,
        )

    def animate_slam_odometry_drift(
        self,
        simulation,
        states,
        title="Introducción a SLAM: deriva de odometría",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima una trayectoria real y su estimación mediante odometría.

        La imagen final muestra:
        - trayectoria real cerrada;
        - trayectoria odométrica con deriva;
        - error de cierre;
        - error de posición por pose;
        - deriva angular;
        - métricas que motivan cierres de ciclo y Graph SLAM.
        """

        if not states:
            raise ValueError(
                "La lista de estados de deriva de odometría no puede estar vacía."
            )
        if simulation is None:
            raise ValueError("La simulación de SLAM no puede ser nula.")

        required = {
            "true_trajectory",
            "estimated_trajectory",
            "position_errors",
            "orientation_errors",
            "metrics",
        }
        missing = required.difference(simulation)
        if missing:
            raise ValueError(
                "Faltan datos de la simulación: "
                + ", ".join(sorted(missing))
            )

        (
            fig,
            info_ax,
            trajectory_ax,
            error_ax,
            diagnostic_ax,
        ) = self._preparar_figura_deriva_odometria(title)

        if final_image_path is not None:
            self._dibujar_estado_deriva_odometria(
                info_ax=info_ax,
                trajectory_ax=trajectory_ax,
                error_ax=error_ax,
                diagnostic_ax=diagnostic_ax,
                simulation=simulation,
                state=states[-1],
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
            self._dibujar_estado_deriva_odometria(
                info_ax=info_ax,
                trajectory_ax=trajectory_ax,
                error_ax=error_ax,
                diagnostic_ax=diagnostic_ax,
                simulation=simulation,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_deriva_odometria(
                info_ax=info_ax,
                trajectory_ax=trajectory_ax,
                error_ax=error_ax,
                diagnostic_ax=diagnostic_ax,
                simulation=simulation,
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
    # Elementos específicos de Pose Graph SLAM 2D
    # ------------------------------------------------------------------

    def _preparar_figura_pose_graph_slam(self, title):
        """
        Crea una figura centrada en la comparación antes/después.

        Distribución:
        - izquierda: explicación, métricas y leyenda;
        - centro superior: pose graph antes de optimizar;
        - derecha superior: pose graph durante/después de optimizar;
        - parte inferior: evolución del coste, RMSE y error de cierre.
        """

        fig = plt.figure(figsize=self.figsize)
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[1.55, 3.15, 3.15],
            height_ratios=[4.55, 1.55],
            wspace=0.10,
            hspace=0.14,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        initial_ax = fig.add_subplot(grid[0, 1])
        optimized_ax = fig.add_subplot(grid[0, 2])
        history_ax = fig.add_subplot(grid[1, 1:])

        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.925,
            bottom=0.055,
        )

        return fig, info_ax, initial_ax, optimized_ax, history_ax

    @staticmethod
    def _limites_pose_graph_slam(result):
        """Calcula límites comunes para los dos paneles geométricos."""

        trajectories = [
            np.asarray(result["true_trajectory"], dtype=float),
            np.asarray(result["initial_trajectory"], dtype=float),
            np.asarray(result["optimized_trajectory"], dtype=float),
        ]
        points = np.vstack([trajectory[:, :2] for trajectory in trajectories])
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        span = np.maximum(maximum - minimum, 1.0)
        margin = 0.12 * span + np.array([0.45, 0.45])
        return (
            minimum[0] - margin[0],
            maximum[0] + margin[0],
            minimum[1] - margin[1],
            maximum[1] + margin[1],
        )

    def _dibujar_leyenda_pose_graph_slam(self, ax):
        """Dibuja una leyenda compacta para el ejemplo."""

        elements = [
            Line2D(
                [0],
                [0],
                color="#777777",
                linewidth=2.0,
                linestyle="dashed",
                label="Trayectoria real",
            ),
            Line2D(
                [0],
                [0],
                color="#F28E2B",
                linewidth=2.8,
                label="Estimación inicial",
            ),
            Line2D(
                [0],
                [0],
                color="#4C9ED9",
                linewidth=2.8,
                label="Estimación optimizada",
            ),
            Line2D(
                [0],
                [0],
                color="#2E8B57",
                linewidth=2.5,
                label="Odometría",
            ),
            Line2D(
                [0],
                [0],
                color="#8E5EA2",
                linewidth=2.8,
                linestyle="dashed",
                label="Cierre de ciclo",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D",
                markersize=8,
                label="Prior sobre x0",
            ),
        ]

        ax.legend(
            handles=elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            fontsize=6.7,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.7,
            handlelength=2.2,
            borderpad=0.55,
        )

    @staticmethod
    def _formatear_numero_pose_graph(valor, precision=4):
        """Formatea magnitudes opcionales del panel de métricas."""

        if valor is None:
            return "—"
        valor = float(valor)
        if not np.isfinite(valor):
            return "∞"
        return f"{valor:.{precision}f}"

    def _dibujar_panel_pose_graph_slam(
        self,
        ax,
        result,
        trajectory,
        state,
        panel_kind,
        title,
    ):
        """Dibuja una trayectoria como pose graph 2D."""

        ax.clear()
        ax.set_title(title, fontsize=11.3, fontweight="bold")
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.grid(True, alpha=0.22)
        ax.set_aspect("equal", adjustable="box")

        limits = self._limites_pose_graph_slam(result)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])

        graph = result["graph"]
        true_trajectory = np.asarray(result["true_trajectory"], dtype=float)
        initial_trajectory = np.asarray(result["initial_trajectory"], dtype=float)
        trajectory = np.asarray(trajectory, dtype=float)

        visible_pose_count = int(
            np.clip(
                state.get("visible_pose_count", len(trajectory)),
                0,
                len(trajectory),
            )
        )
        visible_odometry_count = int(
            np.clip(
                state.get("visible_odometry_count", len(trajectory) - 1),
                0,
                len(trajectory) - 1,
            )
        )

        if state.get("show_true", True) and visible_pose_count > 0:
            ax.plot(
                true_trajectory[:visible_pose_count, 0],
                true_trajectory[:visible_pose_count, 1],
                color="#777777",
                linewidth=1.8,
                linestyle="dashed",
                alpha=0.82,
                zorder=4,
                label="Trayectoria real",
            )

        if visible_pose_count > 0:
            if panel_kind == "initial":
                trajectory_color = "#F28E2B"
                node_face = "#F6C85F"
                node_edge = "#8A4B08"
            else:
                trajectory_color = "#4C9ED9"
                node_face = "#B7D7F0"
                node_edge = "#1F4F73"

            ax.plot(
                trajectory[:visible_pose_count, 0],
                trajectory[:visible_pose_count, 1],
                color=trajectory_color,
                linewidth=2.5,
                alpha=0.92,
                zorder=9,
            )

            # Aristas de odometría visibles.
            for index in range(visible_odometry_count):
                if index + 1 >= visible_pose_count:
                    break
                p0 = trajectory[index, :2]
                p1 = trajectory[index + 1, :2]
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    color="#2E8B57",
                    linewidth=1.7,
                    alpha=0.70,
                    zorder=10,
                )

            ax.scatter(
                trajectory[:visible_pose_count, 0],
                trajectory[:visible_pose_count, 1],
                s=38,
                color=node_face,
                edgecolors=node_edge,
                linewidths=1.1,
                zorder=20,
            )

            # Etiquetas y orientaciones de una selección de poses.
            label_indices = set(range(0, visible_pose_count, 2))
            label_indices.update({0, visible_pose_count - 1})

            for index in sorted(label_indices):
                x, y, theta = trajectory[index]
                ax.text(
                    x,
                    y + 0.22,
                    f"x{index}",
                    fontsize=6.4,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    color="#222222",
                    zorder=30,
                )
                length = 0.28
                ax.arrow(
                    x,
                    y,
                    length * np.cos(theta),
                    length * np.sin(theta),
                    width=0.010,
                    head_width=0.085,
                    head_length=0.090,
                    color=node_edge,
                    length_includes_head=True,
                    zorder=24,
                    alpha=0.85,
                )

        if state.get("show_prior") and visible_pose_count > 0:
            x0, y0 = trajectory[0, :2]
            ax.scatter(
                [x0],
                [y0],
                s=115,
                marker="s",
                color="#E45756",
                edgecolors="#7A1D1D",
                linewidths=2.0,
                zorder=35,
            )
            ax.text(
                x0,
                y0 - 0.36,
                "prior",
                fontsize=7.0,
                fontweight="bold",
                ha="center",
                va="top",
                color="#7A1D1D",
                zorder=36,
            )

        if state.get("show_loop") and visible_pose_count == len(trajectory):
            start = trajectory[-1, :2]
            end = trajectory[0, :2]
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=2.8,
                linestyle="dashed",
                color="#8E5EA2",
                connectionstyle="arc3,rad=0.28",
                shrinkA=7,
                shrinkB=8,
                zorder=32,
            )
            ax.add_patch(arrow)
            middle = 0.5 * (start + end)
            ax.text(
                middle[0],
                middle[1] - 0.48,
                "loop  x15 → x0",
                fontsize=7.0,
                fontweight="bold",
                ha="center",
                va="top",
                color="#5A316B",
                zorder=35,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "fc": "white",
                    "ec": "#8E5EA2",
                    "alpha": 0.94,
                },
            )

            # Vector de cierre espacial, útil en el panel inicial.
            ax.plot(
                [trajectory[-1, 0], trajectory[0, 0]],
                [trajectory[-1, 1], trajectory[0, 1]],
                color="#C62828",
                linewidth=2.0,
                linestyle=":",
                zorder=31,
            )

        phase = state.get("phase", "")
        if phase in {"optimization", "comparison", "summary"}:
            if panel_kind == "initial":
                panel_cost = result["initial_system"]["cost"]
                panel_rmse = result["initial_metrics"]["position_rmse"]
                panel_closure = result["initial_closure"]["translation"]
            else:
                panel_cost = state.get("cost")
                panel_rmse = state.get("rmse")
                panel_closure = state.get("closure_error")

            ax.text(
                0.02,
                0.02,
                (
                    f"coste={self._formatear_numero_pose_graph(panel_cost, 3)}\n"
                    f"RMSE={self._formatear_numero_pose_graph(panel_rmse, 3)} m\n"
                    f"cierre={self._formatear_numero_pose_graph(panel_closure, 3)} m"
                ),
                transform=ax.transAxes,
                fontsize=7.1,
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.32",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.95,
                },
                zorder=50,
            )

    def _dibujar_info_pose_graph_slam(self, ax, result, state):
        """Dibuja explicación, métricas y conexiones conceptuales."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        initial_cost = float(result["initial_system"]["cost"])
        final_cost = float(result["final_system"]["cost"])
        initial_rmse = float(result["initial_metrics"]["position_rmse"])
        final_rmse = float(result["final_metrics"]["position_rmse"])
        initial_closure = float(result["initial_closure"]["translation"])
        final_closure = float(result["final_closure"]["translation"])

        ax.text(
            0.50,
            0.985,
            "Pose Graph SLAM",
            fontsize=12.2,
            fontweight="bold",
            ha="center",
            va="top",
        )

        ax.text(
            0.50,
            0.940,
            "poses = vértices\nmediciones = aristas",
            fontsize=8.2,
            ha="center",
            va="top",
            color="#444444",
            linespacing=1.35,
        )

        cards = [
            (
                "Estructura",
                (
                    f"{result['graph'].number_of_nodes()} poses\n"
                    f"15 odometrías\n1 cierre · 1 prior"
                ),
                "#E5E5E5",
            ),
            (
                "Coste",
                f"{initial_cost:.3f}\n→ {final_cost:.3f}",
                "#D5E8D4",
            ),
            (
                "RMSE de posición",
                f"{initial_rmse:.3f} m\n→ {final_rmse:.3f} m",
                "#B7D7F0",
            ),
            (
                "Error de cierre",
                f"{initial_closure:.3f} m\n→ {final_closure:.3f} m",
                "#E8D7F1",
            ),
        ]

        y_positions = [0.815, 0.675, 0.535, 0.395]
        for (title, value, color), y in zip(cards, y_positions):
            rectangle = Rectangle(
                (0.10, y),
                0.80,
                0.105,
                facecolor=color,
                edgecolor="#666666",
                linewidth=1.2,
            )
            ax.add_patch(rectangle)
            ax.text(
                0.15,
                y + 0.072,
                title,
                fontsize=7.8,
                fontweight="bold",
                ha="left",
                va="center",
            )
            ax.text(
                0.85,
                y + 0.045,
                value,
                fontsize=7.3,
                ha="right",
                va="center",
                linespacing=1.35,
            )

        iteration = state.get("iteration")
        iteration_text = "—" if iteration is None else str(iteration)
        accepted = state.get("accepted")
        if accepted is True:
            accepted_text = "aceptado"
        elif accepted is False:
            accepted_text = "rechazado"
        else:
            accepted_text = "—"

        ax.text(
            0.50,
            0.315,
            (
                f"Iteración: {iteration_text}\n"
                f"λ: {self._formatear_numero_pose_graph(state.get('damping'), 3)}\n"
                f"||ΔX||: {self._formatear_numero_pose_graph(state.get('step_norm'), 4)}\n"
                f"Paso: {accepted_text}"
            ),
            fontsize=7.5,
            ha="center",
            va="top",
            linespacing=1.42,
            bbox={
                "boxstyle": "round,pad=0.42",
                "fc": "white",
                "ec": "#888888",
                "alpha": 0.97,
            },
        )

        ax.text(
            0.50,
            0.165,
            state.get("message", ""),
            fontsize=7.7,
            ha="center",
            va="center",
            wrap=True,
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.97,
            },
        )

        if state.get("show_connections"):
            ax.text(
                0.50,
                0.095,
                "odometría → prior → loop closure → optimización global",
                fontsize=6.8,
                fontweight="bold",
                ha="center",
                va="center",
                color="#333333",
                wrap=True,
            )

        ax.text(
            0.50,
            0.015,
            (
                f"Estado {state.get('step', 1)} de "
                f"{state.get('total_steps', 1)}"
            ),
            fontsize=6.7,
            ha="center",
            va="bottom",
            color="#555555",
        )

        self._dibujar_leyenda_pose_graph_slam(ax)

    def _dibujar_historial_pose_graph_slam(self, ax, result, state):
        """Dibuja coste, RMSE y error de cierre a lo largo de las iteraciones."""

        ax.clear()
        ax.grid(True, alpha=0.22)
        ax.set_title(
            "Convergencia: coste, RMSE y error de cierre",
            fontsize=10.4,
            fontweight="bold",
        )
        ax.set_xlabel("Iteración", fontsize=8)

        history = list(result["optimization"]["history"])
        cost_values = [float(result["initial_system"]["cost"])]
        rmse_values = [float(result["initial_metrics"]["position_rmse"])]
        closure_values = [float(result["initial_closure"]["translation"])]

        for entry in history:
            cost_values.append(float(entry["cost_after"]))
            rmse_values.append(float(entry["rmse_after"]))
            closure_values.append(float(entry["closure_after"]))

        iteration = state.get("iteration")
        if state.get("show_cost_history"):
            visible_count = len(cost_values) if iteration is None else min(
                len(cost_values), int(iteration) + 1
            )
        else:
            visible_count = 1

        x_values = np.arange(visible_count)
        cost_visible = np.asarray(cost_values[:visible_count], dtype=float)
        rmse_visible = np.asarray(rmse_values[:visible_count], dtype=float)
        closure_visible = np.asarray(closure_values[:visible_count], dtype=float)

        # Se normalizan las tres magnitudes para compararlas en un mismo eje.
        def normalize(values):
            base = max(float(values[0]), 1e-12)
            return values / base

        ax.plot(
            x_values,
            normalize(cost_visible),
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            color="#E45756",
            label="coste / coste inicial",
        )
        ax.plot(
            x_values,
            normalize(rmse_visible),
            marker="o",
            linewidth=2.0,
            markersize=4.2,
            color="#4C9ED9",
            label="RMSE / RMSE inicial",
        )
        ax.plot(
            x_values,
            normalize(closure_visible),
            marker="o",
            linewidth=2.0,
            markersize=4.2,
            color="#8E5EA2",
            label="cierre / cierre inicial",
        )

        ax.axhline(1.0, color="#999999", linewidth=1.0, linestyle="dotted")
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(-0.2, max(len(cost_values) - 0.8, 1.2))
        ax.legend(loc="upper right", fontsize=7.0, ncol=3, framealpha=0.95)

        ax.text(
            0.01,
            0.03,
            (
                "Cada valor está dividido por su magnitud inicial. "
                "Las tres curvas deben descender."
            ),
            transform=ax.transAxes,
            fontsize=6.9,
            ha="left",
            va="bottom",
            color="#444444",
        )

    def _dibujar_estado_pose_graph_slam(
        self,
        info_ax,
        initial_ax,
        optimized_ax,
        history_ax,
        result,
        state,
    ):
        """Dibuja un estado completo del ejemplo de Pose Graph SLAM."""

        initial = np.asarray(result["initial_trajectory"], dtype=float)
        current = state.get("current_poses")
        if current is None:
            current = initial
        current = np.asarray(current, dtype=float)

        self._dibujar_info_pose_graph_slam(info_ax, result, state)

        self._dibujar_panel_pose_graph_slam(
            ax=initial_ax,
            result=result,
            trajectory=initial,
            state=state,
            panel_kind="initial",
            title="Antes de optimizar: odometría con deriva",
        )

        phase = state.get("phase")
        if phase in {"comparison", "summary"}:
            optimized_title = "Después de optimizar"
        elif state.get("show_current") or phase == "optimization":
            optimized_title = "Optimización: iteración actual"
        else:
            optimized_title = "Optimización global"

        self._dibujar_panel_pose_graph_slam(
            ax=optimized_ax,
            result=result,
            trajectory=current,
            state=state,
            panel_kind="optimized",
            title=optimized_title,
        )

        if not state.get("show_current") and state.get("phase") not in {
            "optimization",
            "comparison",
            "summary",
        }:
            optimized_ax.text(
                0.50,
                0.50,
                "La trayectoria optimizada\naparecerá al resolver\nel pose graph",
                transform=optimized_ax.transAxes,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                color="#666666",
                bbox={
                    "boxstyle": "round,pad=0.55",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
                zorder=80,
            )

        self._dibujar_historial_pose_graph_slam(history_ax, result, state)

    def animate_pose_graph_slam(
        self,
        result,
        states,
        title="Pose Graph SLAM 2D",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima la construcción y optimización de un pose graph 2D.

        La imagen final muestra:
        - trayectoria inicial y trayectoria optimizada en paneles separados;
        - prior, odometría y cierre de ciclo;
        - coste, RMSE y error de cierre;
        - evolución completa de la optimización.
        """

        if not states:
            raise ValueError(
                "La lista de estados de Pose Graph SLAM no puede estar vacía."
            )
        if result is None:
            raise ValueError("El resultado de Pose Graph SLAM no puede ser nulo.")

        required = {
            "graph",
            "true_trajectory",
            "initial_trajectory",
            "optimized_trajectory",
            "initial_system",
            "final_system",
            "initial_metrics",
            "final_metrics",
            "initial_closure",
            "final_closure",
            "optimization",
        }
        missing = required.difference(result)
        if missing:
            raise ValueError(
                "Faltan datos del resultado: " + ", ".join(sorted(missing))
            )

        (
            fig,
            info_ax,
            initial_ax,
            optimized_ax,
            history_ax,
        ) = self._preparar_figura_pose_graph_slam(title)

        if final_image_path is not None:
            self._dibujar_estado_pose_graph_slam(
                info_ax=info_ax,
                initial_ax=initial_ax,
                optimized_ax=optimized_ax,
                history_ax=history_ax,
                result=result,
                state=states[-1],
            )

            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(final_image_path, dpi=200, bbox_inches="tight")
            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_pose_graph_slam(
                info_ax=info_ax,
                initial_ax=initial_ax,
                optimized_ax=optimized_ax,
                history_ax=history_ax,
                result=result,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_pose_graph_slam(
                info_ax=info_ax,
                initial_ax=initial_ax,
                optimized_ax=optimized_ax,
                history_ax=history_ax,
                result=result,
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
    # Elementos específicos de loop closure
    # ------------------------------------------------------------------

    def _preparar_figura_loop_closure(self, title):
        """Crea la figura de detección, arista de loop y corrección global."""

        fig = plt.figure(figsize=self.figsize)
        grid = fig.add_gridspec(
            2,
            4,
            width_ratios=[1.55, 2.75, 2.75, 2.75],
            height_ratios=[4.65, 1.55],
            wspace=0.11,
            hspace=0.15,
        )

        info_ax = fig.add_subplot(grid[:, 0])
        drift_ax = fig.add_subplot(grid[0, 1])
        detection_ax = fig.add_subplot(grid[0, 2])
        corrected_ax = fig.add_subplot(grid[0, 3])
        history_ax = fig.add_subplot(grid[1, 1:])

        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.subplots_adjust(
            left=0.022,
            right=0.988,
            top=0.925,
            bottom=0.055,
        )
        return fig, info_ax, drift_ax, detection_ax, corrected_ax, history_ax

    @staticmethod
    def _limites_loop_closure(result):
        """Calcula límites comunes para las trayectorias del ejemplo."""

        trajectories = [
            np.asarray(result["true_trajectory"], dtype=float),
            np.asarray(result["initial_trajectory"], dtype=float),
            np.asarray(result["optimized_trajectory"], dtype=float),
        ]
        points = np.vstack([trajectory[:, :2] for trajectory in trajectories])
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        span = np.maximum(maximum - minimum, 1.0)
        margin = 0.10 * span + np.array([0.45, 0.45])
        return (
            minimum[0] - margin[0],
            maximum[0] + margin[0],
            minimum[1] - margin[1],
            maximum[1] + margin[1],
        )

    def _dibujar_leyenda_loop_closure(self, ax):
        """Dibuja la leyenda del apartado de cierre de bucle."""

        elements = [
            Line2D(
                [0], [0], color="#777777", linewidth=1.9,
                linestyle="dashed", label="Trayectoria real",
            ),
            Line2D(
                [0], [0], color="#F28E2B", linewidth=2.7,
                label="Odometría con deriva",
            ),
            Line2D(
                [0], [0], color="#4C9ED9", linewidth=2.7,
                label="Trayectoria corregida",
            ),
            Line2D(
                [0], [0], color="#2E8B57", linewidth=2.3,
                label="Aristas de odometría",
            ),
            Line2D(
                [0], [0], color="#8E5EA2", linewidth=2.7,
                linestyle="dashed", label="Loop closure aceptado",
            ),
            Line2D(
                [0], [0], color="#C62828", linewidth=2.4,
                linestyle="dotted", label="Candidato rechazado",
            ),
            Line2D(
                [0], [0], marker="s", color="none",
                markerfacecolor="#E45756", markeredgecolor="#7A1D1D",
                markersize=8, label="Prior sobre x0",
            ),
        ]
        ax.legend(
            handles=elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            fontsize=6.3,
            framealpha=0.97,
            ncol=2,
            columnspacing=0.65,
            handlelength=2.2,
            borderpad=0.50,
        )

    def _configurar_eje_loop_closure(self, ax, result, title):
        """Configura un panel geométrico con límites comunes."""

        ax.clear()
        ax.set_title(title, fontsize=10.7, fontweight="bold")
        ax.set_xlabel("x [m]", fontsize=7.5)
        ax.set_ylabel("y [m]", fontsize=7.5)
        ax.grid(True, alpha=0.22)
        ax.set_aspect("equal", adjustable="box")
        limits = self._limites_loop_closure(result)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])

    def _dibujar_trayectoria_loop(
        self,
        ax,
        trajectory,
        color,
        node_face,
        node_edge,
        visible_pose_count,
        visible_odometry_count,
        draw_labels=True,
        alpha=1.0,
    ):
        """Dibuja poses, orientaciones y restricciones consecutivas."""

        trajectory = np.asarray(trajectory, dtype=float)
        visible_pose_count = int(
            np.clip(visible_pose_count, 0, len(trajectory))
        )
        visible_odometry_count = int(
            np.clip(visible_odometry_count, 0, len(trajectory) - 1)
        )
        if visible_pose_count <= 0:
            return

        ax.plot(
            trajectory[:visible_pose_count, 0],
            trajectory[:visible_pose_count, 1],
            color=color,
            linewidth=2.45,
            alpha=0.92 * alpha,
            zorder=9,
        )
        for index in range(visible_odometry_count):
            if index + 1 >= visible_pose_count:
                break
            p0 = trajectory[index, :2]
            p1 = trajectory[index + 1, :2]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color="#2E8B57",
                linewidth=1.45,
                alpha=0.65 * alpha,
                zorder=10,
            )

        ax.scatter(
            trajectory[:visible_pose_count, 0],
            trajectory[:visible_pose_count, 1],
            s=31,
            color=node_face,
            edgecolors=node_edge,
            linewidths=1.0,
            alpha=alpha,
            zorder=20,
        )

        if not draw_labels:
            return

        indices = set(range(0, visible_pose_count, 4))
        indices.update({0, visible_pose_count - 1})
        for index in sorted(indices):
            x, y, theta = trajectory[index]
            ax.text(
                x,
                y + 0.20,
                f"x{index}",
                fontsize=5.9,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="#222222",
                zorder=30,
            )
            length = 0.24
            ax.arrow(
                x,
                y,
                length * np.cos(theta),
                length * np.sin(theta),
                width=0.008,
                head_width=0.070,
                head_length=0.075,
                color=node_edge,
                length_includes_head=True,
                alpha=0.82 * alpha,
                zorder=24,
            )

    def _dibujar_prior_loop(self, ax, trajectory):
        """Marca el prior sobre la primera pose."""

        x0, y0 = np.asarray(trajectory, dtype=float)[0, :2]
        ax.scatter(
            [x0], [y0], s=105, marker="s", color="#E45756",
            edgecolors="#7A1D1D", linewidths=1.9, zorder=36,
        )
        ax.text(
            x0, y0 - 0.32, "prior", fontsize=6.5, fontweight="bold",
            ha="center", va="top", color="#7A1D1D", zorder=37,
        )

    def _dibujar_arista_loop(self, ax, trajectory, label="loop x24 → x0"):
        """Dibuja la nueva restricción de largo alcance."""

        trajectory = np.asarray(trajectory, dtype=float)
        start = trajectory[-1, :2]
        end = trajectory[0, :2]
        distance = float(np.linalg.norm(start - end))
        if distance < 0.20:
            center = 0.5 * (start + end)
            loop_arc = Arc(
                center,
                width=0.95,
                height=0.72,
                angle=0.0,
                theta1=25,
                theta2=335,
                linewidth=2.7,
                linestyle="dashed",
                color="#8E5EA2",
                zorder=34,
            )
            ax.add_patch(loop_arc)
            arrow = FancyArrowPatch(
                (center[0] + 0.42, center[1] - 0.16),
                (center[0] + 0.46, center[1] + 0.02),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=2.2,
                color="#8E5EA2",
                zorder=35,
            )
            ax.add_patch(arrow)
            middle = center
            label_y = center[1] - 0.54
        else:
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=2.7,
                linestyle="dashed",
                color="#8E5EA2",
                connectionstyle="arc3,rad=0.28",
                shrinkA=7,
                shrinkB=8,
                zorder=34,
            )
            ax.add_patch(arrow)
            middle = 0.5 * (start + end)
            label_y = middle[1] - 0.43
        ax.text(
            middle[0],
            label_y,
            label,
            fontsize=6.4,
            fontweight="bold",
            ha="center",
            va="top",
            color="#5A316B",
            bbox={
                "boxstyle": "round,pad=0.20",
                "fc": "white",
                "ec": "#8E5EA2",
                "alpha": 0.94,
            },
            zorder=36,
        )

    def _dibujar_panel_deriva_loop(self, ax, result, state):
        """Dibuja la trayectoria odométrica antes de reconocer el lugar."""

        self._configurar_eje_loop_closure(
            ax, result, "1. Antes: odometría con deriva"
        )
        true_trajectory = np.asarray(result["true_trajectory"], dtype=float)
        initial = np.asarray(result["initial_trajectory"], dtype=float)
        visible_pose_count = int(
            np.clip(state.get("visible_pose_count", len(initial)), 0, len(initial))
        )
        visible_odometry_count = int(
            np.clip(
                state.get("visible_odometry_count", len(initial) - 1),
                0,
                len(initial) - 1,
            )
        )

        if state.get("show_true", True) and visible_pose_count > 0:
            ax.plot(
                true_trajectory[:visible_pose_count, 0],
                true_trajectory[:visible_pose_count, 1],
                color="#777777",
                linewidth=1.7,
                linestyle="dashed",
                alpha=0.78,
                zorder=4,
            )
        if state.get("show_initial"):
            self._dibujar_trayectoria_loop(
                ax,
                initial,
                "#F28E2B",
                "#F6C85F",
                "#8A4B08",
                visible_pose_count,
                visible_odometry_count,
            )
        if state.get("show_prior") and visible_pose_count > 0:
            self._dibujar_prior_loop(ax, initial)

        if visible_pose_count == len(initial):
            ax.plot(
                [initial[-1, 0], initial[0, 0]],
                [initial[-1, 1], initial[0, 1]],
                color="#C62828",
                linewidth=2.0,
                linestyle=":",
                zorder=31,
            )
            ax.text(
                0.02,
                0.02,
                (
                    f"RMSE={result['initial_metrics']['position_rmse']:.3f} m\n"
                    f"cierre={result['initial_closure']['translation']:.3f} m"
                ),
                transform=ax.transAxes,
                fontsize=6.8,
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.95,
                },
                zorder=50,
            )

    def _dibujar_panel_deteccion_loop(self, ax, result, state):
        """Dibuja candidatos visuales, verificación y arista aceptada."""

        self._configurar_eje_loop_closure(
            ax, result, "2. Detección y verificación"
        )
        initial = np.asarray(result["initial_trajectory"], dtype=float)
        true_trajectory = np.asarray(result["true_trajectory"], dtype=float)
        numero_poses = len(initial)

        ax.plot(
            initial[:, 0], initial[:, 1], color="#C7C7C7",
            linewidth=1.6, alpha=0.72, zorder=5,
        )
        ax.scatter(
            initial[:, 0], initial[:, 1], s=20, color="#E5E5E5",
            edgecolors="#888888", linewidths=0.8, zorder=12,
        )

        current_index = numero_poses - 1
        current = initial[current_index, :2]
        ax.scatter(
            [current[0]], [current[1]], s=105, marker="*",
            color="#E45756", edgecolors="#7A1D1D", linewidths=1.4,
            zorder=32,
        )
        ax.text(
            current[0], current[1] + 0.28, f"actual x{current_index}",
            fontsize=6.5, fontweight="bold", ha="center", va="bottom",
            color="#7A1D1D", zorder=34,
        )

        all_evaluations = list(result["detection"]["evaluations"])
        accepted_index = result["detection"]["accepted"]["candidate_index"]
        false_index = result["detection"]["false_candidate"]["candidate_index"]
        evaluations = [
            evaluation
            for evaluation in all_evaluations
            if evaluation["candidate_index"] in {accepted_index, false_index}
        ]
        active_candidate = state.get("active_candidate")
        show_candidates = state.get("show_candidates")

        if show_candidates:
            for evaluation in evaluations:
                index = evaluation["candidate_index"]
                point = initial[index, :2]
                is_active = index == active_candidate
                is_accepted = evaluation["accepted"]

                if is_accepted:
                    color = "#8E5EA2"
                    marker = "o"
                else:
                    color = "#C62828"
                    marker = "X"

                ax.scatter(
                    [point[0]], [point[1]],
                    s=95 if is_active else 62,
                    marker=marker,
                    color=color,
                    edgecolors="#333333",
                    linewidths=1.1,
                    zorder=30,
                )
                ax.plot(
                    [current[0], point[0]],
                    [current[1], point[1]],
                    color=color,
                    linewidth=2.4 if is_active else 1.2,
                    linestyle="--" if is_accepted else ":",
                    alpha=0.95 if is_active else 0.45,
                    zorder=24,
                )
                ax.text(
                    point[0], point[1] - 0.28,
                    f"x{index}  s={evaluation['similarity']:.3f}",
                    fontsize=5.8, fontweight="bold" if is_active else "normal",
                    ha="center", va="top", color=color, zorder=35,
                )

        if state.get("show_loop"):
            self._dibujar_arista_loop(ax, initial)

        if state.get("show_database") and not show_candidates:
            ax.text(
                0.50,
                0.50,
                "Consulta de la base\nde lugares anteriores",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="#555555",
                bbox={
                    "boxstyle": "round,pad=0.50",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
                zorder=60,
            )

        if state.get("show_matches") and active_candidate is not None:
            evaluation = next(
                item
                for item in evaluations
                if item["candidate_index"] == active_candidate
            )
            decision = "ACEPTADO" if evaluation["accepted"] else "RECHAZADO"
            decision_color = "#2E8B57" if evaluation["accepted"] else "#C62828"
            ax.text(
                0.02,
                0.02,
                (
                    f"candidato x{active_candidate}\n"
                    f"similitud={evaluation['similarity']:.3f}\n"
                    f"inliers={evaluation['inliers']}/"
                    f"{evaluation['inliers'] + evaluation['outliers']}\n"
                    f"RMSE geom.={evaluation['rmse']:.3f} m\n"
                    f"{decision}"
                ),
                transform=ax.transAxes,
                fontsize=6.8,
                fontweight="bold",
                ha="left",
                va="bottom",
                color=decision_color,
                bbox={
                    "boxstyle": "round,pad=0.30",
                    "fc": "white",
                    "ec": decision_color,
                    "alpha": 0.96,
                },
                zorder=60,
            )

        # El lugar físico real se dibuja discretamente para recordar la revisita.
        ax.scatter(
            [true_trajectory[0, 0]], [true_trajectory[0, 1]],
            s=35, facecolors="none", edgecolors="#777777",
            linewidths=1.1, zorder=16,
        )

    def _dibujar_panel_corregido_loop(self, ax, result, state):
        """Dibuja la trayectoria actual o la solución optimizada."""

        self._configurar_eje_loop_closure(
            ax, result, "3. Después: corrección global"
        )
        true_trajectory = np.asarray(result["true_trajectory"], dtype=float)
        initial = np.asarray(result["initial_trajectory"], dtype=float)
        current = state.get("current_poses")
        if current is None:
            current = initial
        current = np.asarray(current, dtype=float)

        ax.plot(
            true_trajectory[:, 0], true_trajectory[:, 1],
            color="#777777", linewidth=1.7, linestyle="dashed",
            alpha=0.78, zorder=4,
        )
        ax.plot(
            initial[:, 0], initial[:, 1],
            color="#F28E2B", linewidth=1.5, alpha=0.28, zorder=5,
        )

        phase = state.get("phase", "")
        show_current = state.get("show_current") or phase in {
            "optimization", "robustness", "comparison", "summary"
        }
        if show_current:
            self._dibujar_trayectoria_loop(
                ax,
                current,
                "#4C9ED9",
                "#B7D7F0",
                "#1F4F73",
                len(current),
                len(current) - 1,
            )
            if state.get("show_prior"):
                self._dibujar_prior_loop(ax, current)
            if state.get("show_loop"):
                self._dibujar_arista_loop(ax, current)

            ax.text(
                0.02,
                0.02,
                (
                    f"coste={self._formatear_numero_pose_graph(state.get('cost'), 3)}\n"
                    f"RMSE={self._formatear_numero_pose_graph(state.get('rmse'), 3)} m\n"
                    f"cierre={self._formatear_numero_pose_graph(state.get('closure_error'), 3)} m"
                ),
                transform=ax.transAxes,
                fontsize=6.8,
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.95,
                },
                zorder=50,
            )
        else:
            ax.text(
                0.50,
                0.50,
                "La trayectoria corregida\naparecerá después de\nañadir y optimizar el loop",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                color="#666666",
                bbox={
                    "boxstyle": "round,pad=0.50",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
                zorder=70,
            )

    def _dibujar_info_loop_closure(self, ax, result, state):
        """Dibuja explicación, detección, métricas y conexiones."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        accepted = result["detection"]["accepted"]
        false_candidate = result["detection"]["false_candidate"]
        initial_cost = float(result["initial_system"]["cost"])
        final_cost = float(result["final_system"]["cost"])
        initial_rmse = float(result["initial_metrics"]["position_rmse"])
        final_rmse = float(result["final_metrics"]["position_rmse"])
        initial_closure = float(result["initial_closure"]["translation"])
        final_closure = float(result["final_closure"]["translation"])

        ax.text(
            0.50, 0.985, "Loop closure", fontsize=12.2,
            fontweight="bold", ha="center", va="top",
        )
        ax.text(
            0.50, 0.944,
            "reconocer → verificar → añadir arista → optimizar",
            fontsize=7.2, ha="center", va="top", color="#444444", wrap=True,
        )

        cards = [
            (
                "Lugar aceptado",
                (
                    f"x{accepted['candidate_index']} · s={accepted['similarity']:.3f}\n"
                    f"{accepted['inliers']} inliers · {accepted['rmse']:.3f} m"
                ),
                "#D5E8D4",
            ),
            (
                "Alias rechazado",
                (
                    f"x{false_candidate['candidate_index']} · s={false_candidate['similarity']:.3f}\n"
                    f"{false_candidate['inliers']} inliers"
                ),
                "#F6D5D5",
            ),
            (
                "Coste",
                f"{initial_cost:.3f}\n→ {final_cost:.3f}",
                "#E5E5E5",
            ),
            (
                "RMSE",
                f"{initial_rmse:.3f} m\n→ {final_rmse:.3f} m",
                "#B7D7F0",
            ),
            (
                "Error de cierre",
                f"{initial_closure:.3f} m\n→ {final_closure:.3f} m",
                "#E8D7F1",
            ),
        ]
        y_positions = [0.835, 0.710, 0.585, 0.460, 0.335]
        for (title, value, color), y in zip(cards, y_positions):
            rectangle = Rectangle(
                (0.09, y), 0.82, 0.095, facecolor=color,
                edgecolor="#666666", linewidth=1.1,
            )
            ax.add_patch(rectangle)
            ax.text(
                0.14, y + 0.064, title, fontsize=7.2,
                fontweight="bold", ha="left", va="center",
            )
            ax.text(
                0.86, y + 0.040, value, fontsize=6.8,
                ha="right", va="center", linespacing=1.30,
            )

        accepted_state = state.get("accepted")
        if accepted_state is True:
            decision = "aceptado"
        elif accepted_state is False:
            decision = "rechazado"
        else:
            decision = "—"

        ax.text(
            0.50,
            0.265,
            (
                f"Iteración: {state.get('iteration', '—')}\n"
                f"λ: {self._formatear_numero_pose_graph(state.get('damping'), 3)}\n"
                f"||ΔX||: {self._formatear_numero_pose_graph(state.get('step_norm'), 4)}\n"
                f"peso loop: {self._formatear_numero_pose_graph(state.get('loop_weight'), 3)}\n"
                f"decisión: {decision}"
            ),
            fontsize=7.0,
            ha="center",
            va="top",
            linespacing=1.36,
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#888888",
                "alpha": 0.97,
            },
        )

        ax.text(
            0.50,
            0.125,
            state.get("message", ""),
            fontsize=7.2,
            ha="center",
            va="center",
            wrap=True,
            bbox={
                "boxstyle": "round,pad=0.36",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.97,
            },
        )

        if state.get("show_connections"):
            ax.text(
                0.50,
                0.072,
                "ORB-SLAM3 · reconocimiento · asociación · robustez",
                fontsize=6.3,
                fontweight="bold",
                ha="center",
                va="center",
                color="#333333",
                wrap=True,
            )

        ax.text(
            0.50,
            0.012,
            f"Estado {state.get('step', 1)} de {state.get('total_steps', 1)}",
            fontsize=6.4,
            ha="center",
            va="bottom",
            color="#555555",
        )
        self._dibujar_leyenda_loop_closure(ax)

    def _dibujar_historial_loop_closure(self, ax, result, state):
        """Dibuja coste, RMSE, cierre y peso robusto por iteración."""

        ax.clear()
        ax.grid(True, alpha=0.22)
        ax.set_title(
            "Optimización después del loop closure",
            fontsize=10.2,
            fontweight="bold",
        )
        ax.set_xlabel("Iteración", fontsize=7.5)

        history = list(result["optimization"]["history"])
        factor_name = result["loop_factor_name"]
        costs = [float(result["initial_system"]["cost"])]
        rmses = [float(result["initial_metrics"]["position_rmse"])]
        closures = [float(result["initial_closure"]["translation"])]
        weights = [
            float(result["initial_system"]["robust_weights"].get(factor_name, 1.0))
        ]
        for entry in history:
            costs.append(float(entry["cost_after"]))
            rmses.append(float(entry["rmse_after"]))
            closures.append(float(entry["closure_after"]))
            weights.append(float(entry.get("loop_weight_after", 1.0)))

        iteration = state.get("iteration")
        if state.get("show_history"):
            visible_count = len(costs) if iteration is None else min(
                len(costs), int(iteration) + 1
            )
        else:
            visible_count = 1

        x_values = np.arange(visible_count)

        def normalize(values):
            values = np.asarray(values[:visible_count], dtype=float)
            base = max(abs(float(values[0])), 1e-12)
            return values / base

        ax.plot(
            x_values, normalize(costs), marker="o", linewidth=2.1,
            markersize=4.2, color="#E45756", label="coste / inicial",
        )
        ax.plot(
            x_values, normalize(rmses), marker="o", linewidth=1.9,
            markersize=4.0, color="#4C9ED9", label="RMSE / inicial",
        )
        ax.plot(
            x_values, normalize(closures), marker="o", linewidth=1.9,
            markersize=4.0, color="#8E5EA2", label="cierre / inicial",
        )
        ax.plot(
            x_values, np.asarray(weights[:visible_count]), marker="s",
            linewidth=1.8, markersize=3.8, color="#2E8B57",
            label="peso robusto del loop",
        )
        ax.axhline(1.0, color="#999999", linewidth=1.0, linestyle="dotted")
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(-0.2, max(len(costs) - 0.8, 1.2))
        ax.legend(loc="upper right", fontsize=6.8, ncol=4, framealpha=0.95)
        ax.text(
            0.01,
            0.03,
            "La arista de loop reduce la inconsistencia; Huber limita outliers extremos.",
            transform=ax.transAxes,
            fontsize=6.7,
            ha="left",
            va="bottom",
            color="#444444",
        )

    def _dibujar_estado_loop_closure(
        self,
        info_ax,
        drift_ax,
        detection_ax,
        corrected_ax,
        history_ax,
        result,
        state,
    ):
        """Dibuja un estado completo de la demostración de loop closure."""

        self._dibujar_info_loop_closure(info_ax, result, state)
        self._dibujar_panel_deriva_loop(drift_ax, result, state)
        self._dibujar_panel_deteccion_loop(detection_ax, result, state)
        self._dibujar_panel_corregido_loop(corrected_ax, result, state)
        self._dibujar_historial_loop_closure(history_ax, result, state)

    def animate_loop_closure(
        self,
        result,
        states,
        title="Loop closure: detección y corrección",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima el proceso completo de cierre de bucle.

        La imagen final muestra:
        - trayectoria odométrica con deriva;
        - candidato verdadero y alias perceptual rechazado;
        - arista de loop closure;
        - trayectoria corregida;
        - coste, RMSE, cierre y peso robusto.
        """

        if not states:
            raise ValueError(
                "La lista de estados de loop closure no puede estar vacía."
            )
        if result is None:
            raise ValueError("El resultado de loop closure no puede ser nulo.")

        required = {
            "graph",
            "graph_before_loop",
            "true_trajectory",
            "initial_trajectory",
            "optimized_trajectory",
            "detection",
            "initial_system",
            "final_system",
            "initial_metrics",
            "final_metrics",
            "initial_closure",
            "final_closure",
            "optimization",
            "loop_factor_name",
        }
        missing = required.difference(result)
        if missing:
            raise ValueError(
                "Faltan datos del resultado: " + ", ".join(sorted(missing))
            )

        (
            fig,
            info_ax,
            drift_ax,
            detection_ax,
            corrected_ax,
            history_ax,
        ) = self._preparar_figura_loop_closure(title)

        if final_image_path is not None:
            self._dibujar_estado_loop_closure(
                info_ax=info_ax,
                drift_ax=drift_ax,
                detection_ax=detection_ax,
                corrected_ax=corrected_ax,
                history_ax=history_ax,
                result=result,
                state=states[-1],
            )
            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(final_image_path, dpi=200, bbox_inches="tight")
            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_loop_closure(
                info_ax=info_ax,
                drift_ax=drift_ax,
                detection_ax=detection_ax,
                corrected_ax=corrected_ax,
                history_ax=history_ax,
                result=result,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_loop_closure(
                info_ax=info_ax,
                drift_ax=drift_ax,
                detection_ax=detection_ax,
                corrected_ax=corrected_ax,
                history_ax=history_ax,
                result=result,
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
    # Elementos específicos de landmarks en SLAM
    # ------------------------------------------------------------------

    def _preparar_figura_landmarks_slam(self, title):
        """Crea tres paneles geométricos, información y convergencia."""

        fig = plt.figure(figsize=self.figsize)
        grid = fig.add_gridspec(
            2,
            4,
            width_ratios=[1.30, 2.35, 2.35, 2.35],
            height_ratios=[4.75, 1.55],
            wspace=0.10,
            hspace=0.16,
        )
        info_ax = fig.add_subplot(grid[:, 0])
        true_ax = fig.add_subplot(grid[0, 1])
        initial_ax = fig.add_subplot(grid[0, 2])
        optimized_ax = fig.add_subplot(grid[0, 3])
        history_ax = fig.add_subplot(grid[1, 1:])
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.925,
            bottom=0.055,
        )
        return fig, info_ax, true_ax, initial_ax, optimized_ax, history_ax

    @staticmethod
    def _limites_landmarks_slam(result):
        """Calcula límites comunes para poses y landmarks."""

        points = []
        for key in ("true_state", "initial_state", "optimized_state"):
            state = result[key]
            points.append(np.asarray(state["poses"], dtype=float)[:, :2])
            points.append(
                np.asarray(list(state["landmarks"].values()), dtype=float)
            )
        all_points = np.vstack(points)
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)
        return (
            min_x - 0.11 * width - 0.35,
            max_x + 0.11 * width + 0.35,
            min_y - 0.16 * height - 0.35,
            max_y + 0.16 * height + 0.35,
        )

    def _configurar_eje_landmarks_slam(self, ax, title, limits):
        """Aplica una configuración geométrica común a cada panel."""

        ax.clear()
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
        ax.set_xlabel("x [m]", fontsize=7.5)
        ax.set_ylabel("y [m]", fontsize=7.5)
        ax.tick_params(labelsize=6.8)

    @staticmethod
    def _dibujar_flechas_poses_landmarks(
        ax,
        poses,
        count,
        color,
        label,
        alpha=1.0,
        line_width=2.0,
    ):
        """Dibuja una trayectoria y pequeñas flechas de orientación."""

        poses = np.asarray(poses, dtype=float)
        count = max(0, min(int(count), len(poses)))
        if count == 0:
            return
        visible = poses[:count]
        ax.plot(
            visible[:, 0],
            visible[:, 1],
            color=color,
            linewidth=line_width,
            alpha=alpha,
            label=label,
            zorder=16,
        )
        ax.scatter(
            visible[:, 0],
            visible[:, 1],
            s=21,
            color=color,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.45,
            zorder=21,
        )
        step = max(1, len(visible) // 7)
        sampled = visible[::step]
        ax.quiver(
            sampled[:, 0],
            sampled[:, 1],
            np.cos(sampled[:, 2]),
            np.sin(sampled[:, 2]),
            angles="xy",
            scale_units="xy",
            scale=1.0 / 0.32,
            color=color,
            width=0.006,
            alpha=alpha,
            zorder=24,
        )

    @staticmethod
    def _dibujar_landmarks_por_tipo(
        ax,
        positions,
        known_names,
        visible_count,
        *,
        unknown_color,
        unknown_label,
        known_label="Conocido y fijo",
        alpha=1.0,
        annotate=True,
        active_landmark=None,
    ):
        """Dibuja referencias conocidas y desconocidas con símbolos distintos."""

        names = sorted(positions, key=lambda name: int(name[1:]))
        names = names[: max(0, min(int(visible_count), len(names)))]
        known_names = set(known_names)
        known = [name for name in names if name in known_names]
        unknown = [name for name in names if name not in known_names]

        if known:
            xy = np.asarray([positions[name] for name in known], dtype=float)
            ax.scatter(
                xy[:, 0], xy[:, 1], s=95, marker="s",
                color="#E45756", edgecolors="#7A1D1D", linewidths=1.4,
                alpha=alpha, label=known_label, zorder=30,
            )
        if unknown:
            xy = np.asarray([positions[name] for name in unknown], dtype=float)
            ax.scatter(
                xy[:, 0], xy[:, 1], s=90, marker="D",
                color=unknown_color, edgecolors="#4F3562", linewidths=1.2,
                alpha=alpha, label=unknown_label, zorder=29,
            )

        if annotate:
            for name in names:
                x, y = positions[name]
                active = name == active_landmark
                ax.text(
                    x,
                    y + 0.24,
                    name,
                    fontsize=7.0 if not active else 8.2,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    color="#222222",
                    zorder=36,
                    bbox=(
                        {
                            "boxstyle": "round,pad=0.17",
                            "fc": "#FFF3CD",
                            "ec": "#C28A00",
                            "alpha": 0.98,
                        }
                        if active
                        else None
                    ),
                )

    @staticmethod
    def _dibujar_observaciones_landmarks_slam(
        ax,
        observations,
        poses,
        landmarks,
        visible_count,
        color="#2E8B57",
        alpha=0.25,
        line_width=0.9,
        active_pose=None,
        active_landmark=None,
    ):
        """Dibuja aristas pose-landmark visibles en un estado."""

        observations = list(observations)[: max(0, int(visible_count))]
        for observation in observations:
            pose_index = int(observation["pose_name"][1:])
            landmark_name = observation["landmark_name"]
            p = np.asarray(poses[pose_index][:2], dtype=float)
            l = np.asarray(landmarks[landmark_name], dtype=float)
            active = (
                pose_index == active_pose
                or landmark_name == active_landmark
            )
            ax.plot(
                [p[0], l[0]],
                [p[1], l[1]],
                color="#E45756" if active else color,
                linewidth=2.2 if active else line_width,
                alpha=0.88 if active else alpha,
                linestyle="solid" if active else "-",
                zorder=12 if active else 8,
            )

    @staticmethod
    def _dibujar_campo_vision_landmarks_slam(
        ax,
        pose,
        campo_vision_grados,
        alcance,
    ):
        """Dibuja dos rayos y un arco para el campo de visión activo."""

        pose = np.asarray(pose, dtype=float)
        semiangulo = np.deg2rad(float(campo_vision_grados)) / 2.0
        for angle in (pose[2] - semiangulo, pose[2] + semiangulo):
            end = pose[:2] + float(alcance) * np.array(
                [np.cos(angle), np.sin(angle)], dtype=float
            )
            ax.plot(
                [pose[0], end[0]], [pose[1], end[1]],
                color="#F28E2B", linewidth=1.1, linestyle="dashed",
                alpha=0.55, zorder=7,
            )
        arc = Arc(
            (pose[0], pose[1]),
            2.0 * float(alcance),
            2.0 * float(alcance),
            angle=0.0,
            theta1=degrees(pose[2] - semiangulo),
            theta2=degrees(pose[2] + semiangulo),
            color="#F28E2B",
            linewidth=1.1,
            linestyle="dashed",
            alpha=0.55,
            zorder=7,
        )
        ax.add_patch(arc)

    def _dibujar_leyenda_landmarks_slam(self, ax):
        """Añade una leyenda compacta y estable."""

        handles = [
            Line2D([0], [0], color="#777777", linewidth=2.0, label="Real"),
            Line2D([0], [0], color="#F28E2B", linewidth=2.0, label="Inicial"),
            Line2D([0], [0], color="#4C9ED9", linewidth=2.0, label="Optimizada"),
            Line2D(
                [0], [0], marker="s", color="none", markerfacecolor="#E45756",
                markeredgecolor="#7A1D1D", markersize=7, label="Landmark conocido",
            ),
            Line2D(
                [0], [0], marker="D", color="none", markerfacecolor="#8E5EA2",
                markeredgecolor="#4F3562", markersize=7, label="Landmark real",
            ),
            Line2D(
                [0], [0], color="#2E8B57", linewidth=1.6,
                alpha=0.65, label="Observación",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=5.9,
            ncol=2,
            framealpha=0.94,
            borderpad=0.42,
            columnspacing=0.7,
            handlelength=1.8,
        )

    def _dibujar_panel_real_landmarks(self, ax, result, state, limits):
        """Dibuja la geometría verdadera y las observaciones disponibles."""

        self._configurar_eje_landmarks_slam(ax, "1. Geometría real", limits)
        true_state = result["true_state"]
        pose_count = state.get("visible_pose_count", 0)
        landmark_count = state.get("visible_landmark_count", 0)
        observation_count = state.get("visible_observation_count", 0)

        self._dibujar_observaciones_landmarks_slam(
            ax,
            result["observations"],
            true_state["poses"],
            true_state["landmarks"],
            observation_count,
            color="#2E8B57",
            alpha=0.23,
            active_pose=state.get("active_pose"),
            active_landmark=state.get("active_landmark"),
        )
        self._dibujar_flechas_poses_landmarks(
            ax,
            true_state["poses"],
            pose_count,
            "#777777",
            "Trayectoria real",
            alpha=0.92,
        )
        self._dibujar_landmarks_por_tipo(
            ax,
            true_state["landmarks"],
            result["graph"].graph["known_landmarks"],
            landmark_count,
            unknown_color="#8E5EA2",
            unknown_label="Landmark real",
            alpha=0.95,
            active_landmark=state.get("active_landmark"),
        )

        active_pose = state.get("active_pose")
        if state.get("show_fov") and active_pose is not None:
            self._dibujar_campo_vision_landmarks_slam(
                ax,
                true_state["poses"][active_pose],
                250.0,
                2.0,
            )
        ax.text(
            0.02,
            0.02,
            "Las líneas verdes son factores pose-landmark.",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="left",
            va="bottom",
            color="#444444",
        )
        self._dibujar_leyenda_landmarks_slam(ax)

    def _dibujar_panel_inicial_landmarks(self, ax, result, state, limits):
        """Dibuja la odometría con deriva y landmarks inicializados."""

        self._configurar_eje_landmarks_slam(ax, "2. Estimación inicial", limits)
        true_state = result["true_state"]
        initial_state = result["initial_state"]
        pose_count = state.get("visible_pose_count", 0)
        landmark_count = state.get("visible_landmark_count", 0)
        observation_count = state.get("visible_observation_count", 0)

        if state.get("show_true"):
            self._dibujar_flechas_poses_landmarks(
                ax,
                true_state["poses"],
                pose_count,
                "#999999",
                "Real",
                alpha=0.48,
                line_width=1.4,
            )
            self._dibujar_landmarks_por_tipo(
                ax,
                true_state["landmarks"],
                result["graph"].graph["known_landmarks"],
                landmark_count,
                unknown_color="#BCA8C9",
                unknown_label="Landmark real",
                alpha=0.52,
                annotate=False,
            )

        if state.get("show_initial"):
            self._dibujar_observaciones_landmarks_slam(
                ax,
                result["observations"],
                initial_state["poses"],
                initial_state["landmarks"],
                observation_count,
                color="#7AAE74",
                alpha=0.18,
                active_pose=state.get("active_pose"),
                active_landmark=state.get("active_landmark"),
            )
            self._dibujar_flechas_poses_landmarks(
                ax,
                initial_state["poses"],
                pose_count,
                "#F28E2B",
                "Inicial",
                alpha=0.95,
                line_width=2.1,
            )
            self._dibujar_landmarks_por_tipo(
                ax,
                initial_state["landmarks"],
                result["graph"].graph["known_landmarks"],
                landmark_count,
                unknown_color="#F6C85F",
                unknown_label="Landmark inicial",
                alpha=0.96,
                active_landmark=state.get("active_landmark"),
            )

        metrics = result["initial_pose_metrics"]
        landmark_metrics = result["initial_landmark_metrics"]
        ax.text(
            0.98,
            0.02,
            (
                f"RMSE poses: {metrics['position_rmse']:.3f} m\n"
                f"RMSE landmarks: {landmark_metrics['rmse']:.3f} m"
            ),
            transform=ax.transAxes,
            fontsize=6.6,
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.28",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.94,
            },
        )
        self._dibujar_leyenda_landmarks_slam(ax)

    def _dibujar_panel_optimizado_landmarks(self, ax, result, state, limits):
        """Dibuja el estado actual o la solución final optimizada."""

        self._configurar_eje_landmarks_slam(ax, "3. Optimización conjunta", limits)
        true_state = result["true_state"]
        current = state.get("current_state")
        if current is None:
            current = result["initial_state"]
        pose_count = state.get("visible_pose_count", 0)
        landmark_count = state.get("visible_landmark_count", 0)
        observation_count = state.get("visible_observation_count", 0)

        self._dibujar_flechas_poses_landmarks(
            ax,
            true_state["poses"],
            pose_count,
            "#999999",
            "Real",
            alpha=0.42,
            line_width=1.4,
        )
        self._dibujar_landmarks_por_tipo(
            ax,
            true_state["landmarks"],
            result["graph"].graph["known_landmarks"],
            landmark_count,
            unknown_color="#BCA8C9",
            unknown_label="Landmark real",
            alpha=0.45,
            annotate=False,
        )

        if state.get("show_current") or state.get("phase") in {
            "unknown_landmarks", "factor_graph", "optimization",
            "comparison", "summary",
        }:
            self._dibujar_observaciones_landmarks_slam(
                ax,
                result["observations"],
                current["poses"],
                current["landmarks"],
                observation_count,
                color="#2E8B57",
                alpha=0.23,
                active_pose=state.get("active_pose"),
                active_landmark=state.get("active_landmark"),
            )
            self._dibujar_flechas_poses_landmarks(
                ax,
                current["poses"],
                pose_count,
                "#4C9ED9",
                "Actual / optimizada",
                alpha=0.96,
                line_width=2.2,
            )
            self._dibujar_landmarks_por_tipo(
                ax,
                current["landmarks"],
                result["graph"].graph["known_landmarks"],
                landmark_count,
                unknown_color="#4C9ED9",
                unknown_label="Landmark estimado",
                alpha=0.97,
                active_landmark=state.get("active_landmark"),
            )

        iteration = state.get("iteration")
        status = "Sin optimizar" if iteration is None else f"Iteración {iteration}"
        ax.text(
            0.98,
            0.02,
            (
                f"{status}\n"
                f"coste: {state.get('cost') if state.get('cost') is not None else result['initial_system']['cost']:.3f}\n"
                f"obs.: {state.get('observation_rmse') if state.get('observation_rmse') is not None else result['initial_observation_metrics']['rmse']:.3f} m"
            ),
            transform=ax.transAxes,
            fontsize=6.5,
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.28",
                "fc": "white",
                "ec": "#999999",
                "alpha": 0.94,
            },
        )
        self._dibujar_leyenda_landmarks_slam(ax)

    def _dibujar_info_landmarks_slam(self, ax, result, state):
        """Dibuja tarjetas explicativas y métricas del estado actual."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        graph = result["graph"]
        ax.text(
            0.50,
            0.985,
            "Landmarks en SLAM",
            fontsize=12.0,
            fontweight="bold",
            ha="center",
            va="top",
        )
        ax.text(
            0.50,
            0.945,
            "poses + referencias + observaciones",
            fontsize=7.2,
            ha="center",
            va="top",
            color="#444444",
        )

        cards = [
            ("Poses", len(result["true_state"]["poses"]), "#B7D7F0"),
            ("Landmarks", len(result["true_state"]["landmarks"]), "#D8C4E8"),
            ("Conocidos", len(graph.graph["known_landmarks"]), "#F6B4B4"),
            ("Variables", len(graph.graph["unknown_landmarks"]), "#FBE5A6"),
            ("Observaciones", len(result["observations"]), "#B7E4C7"),
            ("Estado", graph.graph["state_dimension"], "#CDE7E8"),
        ]
        y = 0.870
        for label, value, color in cards:
            rectangle = Rectangle(
                (0.10, y), 0.80, 0.065,
                facecolor=color, edgecolor="#666666", linewidth=1.0,
            )
            ax.add_patch(rectangle)
            ax.text(
                0.17, y + 0.0325, label,
                fontsize=7.2, fontweight="bold", ha="left", va="center",
            )
            ax.text(
                0.83, y + 0.0325, str(value),
                fontsize=7.5, fontweight="bold", ha="right", va="center",
            )
            y -= 0.074

        cost = state.get("cost")
        pose_rmse = state.get("pose_rmse")
        landmark_rmse = state.get("landmark_rmse")
        observation_rmse = state.get("observation_rmse")
        if cost is None:
            cost = result["initial_system"]["cost"]
        if pose_rmse is None:
            pose_rmse = result["initial_pose_metrics"]["position_rmse"]
        if landmark_rmse is None:
            landmark_rmse = result["initial_landmark_metrics"]["rmse"]
        if observation_rmse is None:
            observation_rmse = result["initial_observation_metrics"]["rmse"]

        ax.text(
            0.50,
            0.405,
            (
                f"Coste: {cost:.5f}\n"
                f"RMSE poses: {pose_rmse:.5f} m\n"
                f"RMSE landmarks: {landmark_rmse:.5f} m\n"
                f"RMSE observación: {observation_rmse:.5f} m"
            ),
            fontsize=7.1,
            ha="center",
            va="center",
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.42",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.98,
            },
        )

        damping = state.get("damping")
        step_norm = state.get("step_norm")
        iteration = state.get("iteration")
        if iteration is not None:
            damping_text = "—" if damping is None else f"{damping:.3g}"
            step_text = "—" if step_norm is None else f"{step_norm:.3g}"
            ax.text(
                0.50,
                0.278,
                (
                    f"iteración {iteration}\n"
                    f"λ={damping_text} · ‖Δ‖={step_text}\n"
                    f"paso {'aceptado' if state.get('accepted') else 'resumen'}"
                ),
                fontsize=6.8,
                ha="center",
                va="center",
                linespacing=1.35,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "fc": "#EEF6FB",
                    "ec": "#4C9ED9",
                    "alpha": 0.97,
                },
            )

        message_words = str(state.get("message", "")).split()
        message_lines = []
        current_line = []
        for word in message_words:
            candidate = " ".join(current_line + [word])
            if current_line and len(candidate) > 31:
                message_lines.append(" ".join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            message_lines.append(" ".join(current_line))
        wrapped_message = "\n".join(message_lines)

        ax.text(
            0.50,
            0.166,
            wrapped_message,
            fontsize=6.8,
            ha="center",
            va="center",
            wrap=True,
            bbox={
                "boxstyle": "round,pad=0.38",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.97,
            },
        )
        if state.get("show_graph_connections"):
            ax.text(
                0.50,
                0.082,
                "OpenCV · AprilTags · características · mapas semánticos",
                fontsize=6.0,
                fontweight="bold",
                ha="center",
                va="center",
                color="#333333",
                wrap=True,
            )
        ax.text(
            0.50,
            0.018,
            f"Estado {state.get('step', 1)} de {state.get('total_steps', 1)}",
            fontsize=6.4,
            ha="center",
            va="bottom",
            color="#555555",
        )

    def _dibujar_historial_landmarks_slam(self, ax, result, state):
        """Dibuja coste y errores normalizados por iteración."""

        ax.clear()
        ax.grid(True, alpha=0.22)
        ax.set_title(
            "Convergencia de poses, landmarks y observaciones",
            fontsize=10.1,
            fontweight="bold",
        )
        ax.set_xlabel("Iteración", fontsize=7.5)

        history = list(result["optimization"]["history"])
        costs = [float(result["initial_system"]["cost"])]
        pose_rmses = [float(result["initial_pose_metrics"]["position_rmse"])]
        landmark_rmses = [float(result["initial_landmark_metrics"]["rmse"])]
        observation_rmses = [
            float(result["initial_observation_metrics"]["rmse"])
        ]
        for entry in history:
            costs.append(float(entry["cost_after"]))
            pose_rmses.append(float(entry["pose_rmse_after"]))
            landmark_rmses.append(float(entry["landmark_rmse_after"]))
            observation_rmses.append(float(entry["observation_rmse_after"]))

        iteration = state.get("iteration")
        if state.get("show_history"):
            visible_count = len(costs) if iteration is None else min(
                len(costs), int(iteration) + 1
            )
        else:
            visible_count = 1
        x_values = np.arange(visible_count)

        def normalize(values):
            values = np.asarray(values[:visible_count], dtype=float)
            base = max(abs(float(values[0])), 1e-12)
            return values / base

        ax.plot(
            x_values, normalize(costs), marker="o", linewidth=2.1,
            markersize=4.0, color="#E45756", label="coste / inicial",
        )
        ax.plot(
            x_values, normalize(pose_rmses), marker="o", linewidth=1.9,
            markersize=3.8, color="#4C9ED9", label="RMSE poses / inicial",
        )
        ax.plot(
            x_values, normalize(landmark_rmses), marker="D", linewidth=1.9,
            markersize=3.6, color="#8E5EA2", label="RMSE landmarks / inicial",
        )
        ax.plot(
            x_values, normalize(observation_rmses), marker="s", linewidth=1.8,
            markersize=3.6, color="#2E8B57", label="RMSE observación / inicial",
        )
        ax.axhline(1.0, color="#999999", linewidth=1.0, linestyle="dotted")
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(-0.2, max(len(costs) - 0.8, 1.2))
        ax.legend(loc="upper right", fontsize=6.7, ncol=4, framealpha=0.95)
        ax.text(
            0.01,
            0.03,
            "Los landmarks conocidos permanecen fijos; poses y referencias desconocidas convergen.",
            transform=ax.transAxes,
            fontsize=6.6,
            ha="left",
            va="bottom",
            color="#444444",
        )

    def _dibujar_estado_landmarks_slam(
        self,
        info_ax,
        true_ax,
        initial_ax,
        optimized_ax,
        history_ax,
        result,
        state,
    ):
        """Dibuja un estado completo del ejemplo de landmarks."""

        limits = self._limites_landmarks_slam(result)
        self._dibujar_info_landmarks_slam(info_ax, result, state)
        self._dibujar_panel_real_landmarks(true_ax, result, state, limits)
        self._dibujar_panel_inicial_landmarks(initial_ax, result, state, limits)
        self._dibujar_panel_optimizado_landmarks(
            optimized_ax, result, state, limits
        )
        self._dibujar_historial_landmarks_slam(history_ax, result, state)

    def animate_landmarks_slam(
        self,
        result,
        states,
        title="Landmarks en SLAM",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima la construcción y optimización de un grafo pose-landmark.

        La imagen final muestra:
        - geometría real;
        - odometría y landmarks iniciales;
        - trayectoria y landmarks optimizados;
        - observaciones pose-landmark;
        - convergencia de coste y errores.
        """

        if not states:
            raise ValueError(
                "La lista de estados de landmarks SLAM no puede estar vacía."
            )
        if result is None:
            raise ValueError("El resultado de landmarks SLAM no puede ser nulo.")

        required = {
            "graph",
            "true_state",
            "initial_state",
            "optimized_state",
            "observations",
            "initial_system",
            "final_system",
            "initial_pose_metrics",
            "final_pose_metrics",
            "initial_landmark_metrics",
            "final_landmark_metrics",
            "initial_observation_metrics",
            "final_observation_metrics",
            "optimization",
        }
        missing = required.difference(result)
        if missing:
            raise ValueError(
                "Faltan datos del resultado: " + ", ".join(sorted(missing))
            )

        (
            fig,
            info_ax,
            true_ax,
            initial_ax,
            optimized_ax,
            history_ax,
        ) = self._preparar_figura_landmarks_slam(title)

        if final_image_path is not None:
            self._dibujar_estado_landmarks_slam(
                info_ax=info_ax,
                true_ax=true_ax,
                initial_ax=initial_ax,
                optimized_ax=optimized_ax,
                history_ax=history_ax,
                result=result,
                state=states[-1],
            )
            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(final_image_path, dpi=200, bbox_inches="tight")
            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_landmarks_slam(
                info_ax=info_ax,
                true_ax=true_ax,
                initial_ax=initial_ax,
                optimized_ax=optimized_ax,
                history_ax=history_ax,
                result=result,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_landmarks_slam(
                info_ax=info_ax,
                true_ax=true_ax,
                initial_ax=initial_ax,
                optimized_ax=optimized_ax,
                history_ax=history_ax,
                result=result,
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
    # Error de observación pose-landmark
    # ------------------------------------------------------------------

    def _preparar_figura_error_pose_landmark(self, title):
        """Crea paneles para geometría global, marco local y valores."""

        fig = plt.figure(figsize=self.figsize)
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[1.65, 3.80, 3.15],
            height_ratios=[4.75, 2.05],
            wspace=0.10,
            hspace=0.13,
        )
        info_ax = fig.add_subplot(grid[:, 0])
        global_ax = fig.add_subplot(grid[0, 1])
        local_ax = fig.add_subplot(grid[0, 2])
        metrics_ax = fig.add_subplot(grid[1, 1:])

        fig.suptitle(title, fontsize=16, fontweight="bold")
        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.925,
            bottom=0.045,
        )
        return fig, info_ax, global_ax, local_ax, metrics_ax

    @staticmethod
    def _titulo_fase_error_pose_landmark(phase):
        """Traduce las fases internas a títulos didácticos."""

        titles = {
            "introduction": "1. Factor pose-landmark",
            "true_geometry": "2. Geometría real",
            "measurement": "3. Medición fija",
            "estimated_state": "4. Estado estimado",
            "prediction": "5. Predicción del modelo",
            "range_error": "6. Error de distancia",
            "bearing_error": "7. Error angular",
            "residual": "8. Vector de residuo",
            "angle_wrap": "9. Normalización angular",
            "uncertainty": "10. Incertidumbre",
            "mahalanobis": "11. Mahalanobis y coste",
            "jacobians": "12. Jacobianos",
            "sensitivity": "13. Sensibilidad geométrica",
            "observability": "14. Observabilidad local",
            "landmark_correction": "15. Corrección del landmark",
            "optimized": "16. Observación compatible",
            "calibration_correct": "17. Calibración correcta",
            "calibration_wrong": "18. Error de calibración",
            "summary": "19. Resumen final",
        }
        return titles.get(phase, str(phase))

    @staticmethod
    def _formatear_pose_error_pose_landmark(pose):
        """Formatea una pose 2D para tarjetas."""

        pose = np.asarray(pose, dtype=float)
        return (
            f"({pose[0]:.3f}, {pose[1]:.3f}, "
            f"{degrees(pose[2]):.2f}°)"
        )

    @staticmethod
    def _formatear_landmark_error_pose_landmark(landmark):
        """Formatea una posición de landmark."""

        landmark = np.asarray(landmark, dtype=float)
        return f"({landmark[0]:.3f}, {landmark[1]:.3f})"

    @staticmethod
    def _dibujar_pose_error_pose_landmark(
        ax,
        pose,
        color,
        edge_color,
        label,
        alpha=1.0,
        line_style="solid",
        zorder=20,
    ):
        """Dibuja una pose como círculo y flecha orientada."""

        pose = np.asarray(pose, dtype=float)
        x, y, theta = pose
        length = 0.50
        ax.scatter(
            [x],
            [y],
            s=115,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.7,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(
            FancyArrowPatch(
                (x, y),
                (x + length * cos(theta), y + length * sin(theta)),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=2.2,
                linestyle=line_style,
                color=edge_color,
                alpha=alpha,
                zorder=zorder + 1,
            )
        )
        ax.text(
            x,
            y - 0.30,
            label,
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="top",
            color=edge_color,
            alpha=alpha,
            zorder=zorder + 2,
        )

    @staticmethod
    def _dibujar_sensor_error_pose_landmark(
        ax,
        sensor_pose,
        color,
        label,
        alpha=1.0,
        zorder=24,
    ):
        """Dibuja el marco del sensor mediante un cuadrado y dos ejes."""

        sensor_pose = np.asarray(sensor_pose, dtype=float)
        x, y, theta = sensor_pose
        ax.scatter(
            [x],
            [y],
            marker="s",
            s=52,
            facecolor="white",
            edgecolor=color,
            linewidth=1.6,
            alpha=alpha,
            zorder=zorder,
        )
        axis_length = 0.28
        ax.plot(
            [x, x + axis_length * cos(theta)],
            [y, y + axis_length * sin(theta)],
            color=color,
            linewidth=1.8,
            alpha=alpha,
            zorder=zorder + 1,
        )
        ax.plot(
            [x, x - axis_length * sin(theta)],
            [y, y + axis_length * cos(theta)],
            color=color,
            linewidth=1.2,
            alpha=0.72 * alpha,
            zorder=zorder + 1,
        )
        ax.text(
            x + 0.08,
            y + 0.16,
            label,
            fontsize=6.5,
            fontweight="bold",
            color=color,
            alpha=alpha,
            zorder=zorder + 2,
        )

    def _dibujar_leyenda_error_pose_landmark(self, ax):
        """Dibuja una leyenda compacta y estable."""

        elements = [
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor="#4C9ED9", markeredgecolor="#1F4F73",
                markersize=8, label="Pose estimada",
            ),
            Line2D(
                [0], [0], marker="D", color="none",
                markerfacecolor="#F28E2B", markeredgecolor="#8A4B08",
                markersize=8, label="Landmark estimado",
            ),
            Line2D(
                [0], [0], color="#2E8B57", linewidth=2.6,
                linestyle="dashed", label="Medición fija",
            ),
            Line2D(
                [0], [0], color="#8E5EA2", linewidth=2.8,
                label="Predicción",
            ),
            Line2D(
                [0], [0], color="#C62828", linewidth=2.8,
                label="Error",
            ),
            Line2D(
                [0], [0], color="#777777", linewidth=1.7,
                linestyle="dashed", label="Valor real / inicial",
            ),
        ]
        ax.legend(
            handles=elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            fontsize=6.6,
            framealpha=0.97,
            ncol=1,
            handlelength=2.4,
            labelspacing=0.55,
        )

    def _dibujar_info_error_pose_landmark(self, ax, result, state):
        """Dibuja fase, mensaje y flujo conceptual del factor."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.50,
            0.985,
            self._titulo_fase_error_pose_landmark(state.get("phase")),
            fontsize=11.2,
            fontweight="bold",
            ha="center",
            va="top",
        )
        ax.text(
            0.50,
            0.943,
            f"Estado {state.get('step', 0)} de {state.get('total_steps', 0)}",
            fontsize=7.3,
            color="#555555",
            ha="center",
            va="top",
        )
        ax.text(
            0.50,
            0.835,
            state.get("message", ""),
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="top",
            wrap=True,
            linespacing=1.42,
            bbox={
                "boxstyle": "round,pad=0.45",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.98,
            },
        )

        # Esquema mínimo del factor.
        ax.scatter(
            [0.20], [0.665], s=220,
            facecolor="#B7D7F0", edgecolor="#1F4F73", linewidth=1.7,
        )
        ax.scatter(
            [0.80], [0.665], s=205, marker="D",
            facecolor="#FBE5A6", edgecolor="#8A4B08", linewidth=1.7,
        )
        ax.plot(
            [0.28, 0.72], [0.665, 0.665],
            color="#2E8B57", linewidth=3.0,
        )
        ax.text(0.20, 0.665, "x0", fontsize=8.4, fontweight="bold", ha="center", va="center")
        ax.text(0.80, 0.665, "l0", fontsize=8.4, fontweight="bold", ha="center", va="center")
        ax.text(0.50, 0.700, "z=(r, β)", fontsize=7.0, fontweight="bold", ha="center", color="#245B3A")

        measurement = result["measurement"]
        evaluation = state["evaluation"]
        fixed_box = Rectangle(
            (0.08, 0.500), 0.84, 0.105,
            facecolor="#D5E8D4", edgecolor="#2E8B57", linewidth=1.5,
        )
        ax.add_patch(fixed_box)
        ax.text(
            0.50, 0.553,
            (
                "MEDICIÓN FIJA\n"
                f"r={measurement[0]:.3f} m · β={degrees(measurement[1]):.3f}°"
            ),
            fontsize=7.1,
            fontweight="bold",
            color="#245B3A",
            ha="center",
            va="center",
        )

        prediction = evaluation["prediction"]
        dynamic_box = Rectangle(
            (0.08, 0.360), 0.84, 0.105,
            facecolor="#E8D7F1", edgecolor="#8E5EA2", linewidth=1.5,
        )
        ax.add_patch(dynamic_box)
        ax.text(
            0.50, 0.413,
            (
                "PREDICCIÓN h(x0,l0)\n"
                f"r̂={prediction[0]:.3f} m · β̂={degrees(prediction[1]):.3f}°"
            ),
            fontsize=7.1,
            fontweight="bold",
            color="#5A316B",
            ha="center",
            va="center",
        )

        label = state.get("experiment_label")
        if label:
            ax.text(
                0.50,
                0.303,
                label,
                fontsize=7.2,
                fontweight="bold",
                ha="center",
                va="center",
                color="#8A4B08",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "#FBE5A6",
                    "ec": "#8A6D1D",
                },
            )

        if state.get("show_wrap", False):
            wrap = result["angle_wrap"]
            ax.text(
                0.50,
                0.225,
                (
                    f"{degrees(wrap['predicted_bearing']):.0f}° - "
                    f"({degrees(wrap['measured_bearing']):.0f}°)\n"
                    f"= {degrees(wrap['raw_error']):.0f}° → "
                    f"wrap = {degrees(wrap['normalized_error']):.0f}°"
                ),
                fontsize=7.4,
                family="monospace",
                fontweight="bold",
                color="#7A1D1D",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.33",
                    "fc": "#F7C6C7",
                    "ec": "#C62828",
                },
            )
        elif state.get("show_calibration", False):
            ext = state["sensor_extrinsic"]
            ax.text(
                0.50,
                0.225,
                (
                    "T_RS usada\n"
                    f"({ext[0]:.2f} m, {ext[1]:.2f} m, {degrees(ext[2]):.1f}°)"
                ),
                fontsize=7.2,
                family="monospace",
                ha="center",
                va="center",
                color="#7A1D1D" if state.get("phase") == "calibration_wrong" else "#245B3A",
            )
        else:
            ax.text(
                0.50,
                0.225,
                "e = h(x0,l0) - z\nF = 1/2 · eᵀΩe",
                fontsize=7.5,
                family="monospace",
                ha="center",
                va="center",
                color="#333333",
                linespacing=1.45,
            )

        self._dibujar_leyenda_error_pose_landmark(ax)

    def _dibujar_geometria_global_error_pose_landmark(self, ax, result, state):
        """Dibuja pose, sensor, landmark y rayos en el marco global."""

        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.15, 5.70)
        ax.set_ylim(0.15, 5.15)
        ax.grid(True, linewidth=0.55, alpha=0.22)
        ax.set_xlabel("x global [m]", fontsize=8)
        ax.set_ylabel("y global [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("Geometría global", fontsize=11.0, fontweight="bold")

        evaluation = state["evaluation"]
        pose = state["pose"]
        landmark = state["landmark"]
        sensor = evaluation["sensor_pose"]

        if state.get("show_true", False):
            self._dibujar_pose_error_pose_landmark(
                ax,
                result["true_pose"],
                "white",
                "#777777",
                "pose real",
                alpha=0.72,
                line_style="dashed",
                zorder=11,
            )
            ax.scatter(
                [result["true_landmark"][0]],
                [result["true_landmark"][1]],
                s=135,
                marker="x",
                color="#555555",
                linewidth=2.0,
                alpha=0.75,
                zorder=12,
            )
            ax.text(
                result["true_landmark"][0] + 0.08,
                result["true_landmark"][1] + 0.16,
                "landmark real",
                fontsize=6.8,
                color="#555555",
                zorder=13,
            )

        if state.get("show_initial_history", False):
            initial_landmark = result["initial_landmark"]
            if np.linalg.norm(landmark - initial_landmark) > 1e-8:
                ax.scatter(
                    [initial_landmark[0]],
                    [initial_landmark[1]],
                    s=80,
                    marker="D",
                    facecolor="white",
                    edgecolor="#8A4B08",
                    linewidth=1.3,
                    alpha=0.55,
                    zorder=13,
                )
                ax.plot(
                    [initial_landmark[0], landmark[0]],
                    [initial_landmark[1], landmark[1]],
                    color="#777777",
                    linestyle="dashed",
                    linewidth=1.4,
                    alpha=0.65,
                    zorder=12,
                )
                ax.text(
                    initial_landmark[0] - 0.12,
                    initial_landmark[1] + 0.20,
                    "l0 inicial",
                    fontsize=6.4,
                    color="#777777",
                    ha="right",
                )

        self._dibujar_pose_error_pose_landmark(
            ax,
            pose,
            "#4C9ED9",
            "#1F4F73",
            "x0 estimada",
            zorder=20,
        )
        self._dibujar_sensor_error_pose_landmark(
            ax,
            sensor,
            "#1F4F73",
            "sensor",
            zorder=24,
        )

        if state.get("phase") == "calibration_wrong":
            true_sensor = result["true_evaluation"]["sensor_pose"]
            self._dibujar_sensor_error_pose_landmark(
                ax,
                true_sensor,
                "#777777",
                "sensor real",
                alpha=0.58,
                zorder=14,
            )

        ax.scatter(
            [landmark[0]],
            [landmark[1]],
            s=125,
            marker="D",
            facecolor="#F28E2B",
            edgecolor="#8A4B08",
            linewidth=1.8,
            zorder=25,
        )
        ax.text(
            landmark[0] + 0.12,
            landmark[1] - 0.24,
            "l0 estimado",
            fontsize=7.0,
            fontweight="bold",
            color="#8A4B08",
            ha="left",
            va="top",
            zorder=26,
        )

        if state.get("show_measurement", False):
            endpoint = evaluation["measurement_endpoint_global"]
            ax.plot(
                [sensor[0], endpoint[0]],
                [sensor[1], endpoint[1]],
                color="#2E8B57",
                linewidth=2.8,
                linestyle="dashed",
                zorder=18,
            )
            ax.scatter(
                [endpoint[0]], [endpoint[1]],
                s=74, marker="o", facecolor="white",
                edgecolor="#2E8B57", linewidth=1.7, zorder=21,
            )
            ax.text(
                endpoint[0] - 0.12,
                endpoint[1] + 0.20,
                "extremo medido",
                fontsize=6.6,
                color="#245B3A",
                ha="right",
                va="bottom",
                zorder=22,
            )

        if state.get("show_prediction", False):
            ax.plot(
                [sensor[0], landmark[0]],
                [sensor[1], landmark[1]],
                color="#8E5EA2",
                linewidth=2.8,
                zorder=19,
            )

        if state.get("show_range_error", False) or state.get("show_bearing_error", False):
            endpoint = evaluation["measurement_endpoint_global"]
            ax.add_patch(
                FancyArrowPatch(
                    endpoint,
                    landmark,
                    arrowstyle="<->",
                    mutation_scale=11,
                    linewidth=2.2,
                    color="#C62828",
                    connectionstyle="arc3,rad=0.08",
                    zorder=28,
                )
            )
            ax.text(
                (endpoint[0] + landmark[0]) / 2 - 0.08,
                (endpoint[1] + landmark[1]) / 2 + 0.32,
                "discrepancia geométrica",
                fontsize=6.5,
                fontweight="bold",
                color="#7A1D1D",
                ha="center",
                zorder=29,
            )

        ax.text(
            0.02,
            0.02,
            "El sensor no coincide necesariamente con el origen del robot.",
            transform=ax.transAxes,
            fontsize=7.0,
            color="#444444",
            ha="left",
            va="bottom",
        )

    @staticmethod
    def _covarianza_polar_a_cartesiana(measurement, covariance):
        """Propaga localmente una covarianza rango-rumbo a cartesiano."""

        measurement = np.asarray(measurement, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        r, beta = measurement
        jacobian = np.array(
            [
                [cos(beta), -r * sin(beta)],
                [sin(beta), r * cos(beta)],
            ],
            dtype=float,
        )
        return jacobian @ covariance @ jacobian.T

    def _dibujar_marco_local_error_pose_landmark(self, ax, result, state):
        """Dibuja medida, predicción, distancias y ángulos en el sensor."""

        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.45, 5.35)
        ax.set_ylim(-2.60, 3.35)
        ax.grid(True, linewidth=0.55, alpha=0.20)
        ax.set_xlabel("eje x del sensor [m]", fontsize=8)
        ax.set_ylabel("eje y del sensor [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("Medición frente a predicción", fontsize=11.0, fontweight="bold")

        evaluation = state["evaluation"]
        measurement_vector = evaluation["measurement_cartesian"]
        prediction_vector = evaluation["prediction_cartesian"]
        measurement = evaluation["measurement"]
        prediction = evaluation["prediction"]

        ax.axhline(0.0, color="#1F4F73", linewidth=1.3, alpha=0.70)
        ax.axvline(0.0, color="#1F4F73", linewidth=1.0, alpha=0.45)
        ax.scatter(
            [0.0], [0.0], marker="s", s=70,
            facecolor="white", edgecolor="#1F4F73", linewidth=1.7, zorder=25,
        )
        ax.text(0.08, 0.16, "sensor", fontsize=7.0, fontweight="bold", color="#1F4F73")

        if state.get("show_uncertainty", False):
            covariance_cartesian = self._covarianza_polar_a_cartesiana(
                measurement,
                result["covariance"],
            )
            values, vectors = np.linalg.eigh(covariance_cartesian)
            order = np.argsort(values)[::-1]
            values = values[order]
            vectors = vectors[:, order]
            angle = degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
            width, height = 2.0 * 2.0 * np.sqrt(np.maximum(values, 0.0))
            ellipse = Ellipse(
                measurement_vector,
                width=width,
                height=height,
                angle=angle,
                facecolor="#B7E4C7",
                edgecolor="#2E8B57",
                linewidth=1.5,
                alpha=0.28,
                zorder=10,
            )
            ax.add_patch(ellipse)

        if state.get("show_measurement", False):
            ax.add_patch(
                FancyArrowPatch(
                    (0.0, 0.0),
                    measurement_vector,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=2.8,
                    linestyle="dashed",
                    color="#2E8B57",
                    zorder=20,
                )
            )
            ax.scatter(
                [measurement_vector[0]], [measurement_vector[1]],
                s=75, facecolor="white", edgecolor="#2E8B57",
                linewidth=1.7, zorder=24,
            )
            midpoint = 0.52 * measurement_vector
            ax.text(
                midpoint[0], midpoint[1] + 0.12,
                f"r={measurement[0]:.3f} m",
                fontsize=6.9, fontweight="bold", color="#245B3A",
                ha="center", zorder=25,
            )

        if state.get("show_prediction", False):
            ax.add_patch(
                FancyArrowPatch(
                    (0.0, 0.0),
                    prediction_vector,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=2.9,
                    color="#8E5EA2",
                    zorder=21,
                )
            )
            ax.scatter(
                [prediction_vector[0]], [prediction_vector[1]],
                s=75, marker="D", facecolor="#FBE5A6", edgecolor="#8E5EA2",
                linewidth=1.7, zorder=24,
            )
            midpoint = 0.56 * prediction_vector
            ax.text(
                midpoint[0], midpoint[1] - 0.16,
                f"r̂={prediction[0]:.3f} m",
                fontsize=6.9, fontweight="bold", color="#5A316B",
                ha="center", zorder=25,
            )

        if state.get("show_range_error", False):
            unit_prediction = prediction_vector / max(np.linalg.norm(prediction_vector), 1e-12)
            measured_on_prediction = measurement[0] * unit_prediction
            ax.add_patch(
                FancyArrowPatch(
                    measured_on_prediction,
                    prediction_vector,
                    arrowstyle="<->",
                    mutation_scale=12,
                    linewidth=2.8,
                    color="#C62828",
                    zorder=28,
                )
            )
            center = 0.5 * (measured_on_prediction + prediction_vector)
            ax.text(
                center[0] + 0.10,
                center[1] - 0.15,
                f"e_r={evaluation['range_error']:+.3f} m",
                fontsize=7.0,
                fontweight="bold",
                color="#7A1D1D",
                zorder=29,
            )

        max_arc_radius = 0.90
        if state.get("show_measurement", False):
            beta_deg = degrees(measurement[1])
            theta1, theta2 = sorted((0.0, beta_deg))
            ax.add_patch(
                Arc(
                    (0.0, 0.0),
                    2 * max_arc_radius,
                    2 * max_arc_radius,
                    angle=0.0,
                    theta1=theta1,
                    theta2=theta2,
                    linewidth=2.1,
                    linestyle="dashed",
                    color="#2E8B57",
                    zorder=16,
                )
            )
            ax.text(
                0.64 * cos(measurement[1] / 2),
                0.64 * sin(measurement[1] / 2) + 0.05,
                f"β={beta_deg:.2f}°",
                fontsize=6.7,
                color="#245B3A",
                ha="center",
            )

        if state.get("show_prediction", False):
            beta_hat_deg = degrees(prediction[1])
            theta1, theta2 = sorted((0.0, beta_hat_deg))
            ax.add_patch(
                Arc(
                    (0.0, 0.0),
                    2 * (max_arc_radius + 0.18),
                    2 * (max_arc_radius + 0.18),
                    angle=0.0,
                    theta1=theta1,
                    theta2=theta2,
                    linewidth=2.2,
                    color="#8E5EA2",
                    zorder=17,
                )
            )
            ax.text(
                0.88 * cos(prediction[1] / 2),
                0.88 * sin(prediction[1] / 2) - 0.08,
                f"β̂={beta_hat_deg:.2f}°",
                fontsize=6.7,
                color="#5A316B",
                ha="center",
            )

        if state.get("show_bearing_error", False):
            start_deg = degrees(measurement[1])
            error_deg = degrees(evaluation["bearing_error"])
            end_deg = start_deg + error_deg
            theta1, theta2 = sorted((start_deg, end_deg))
            ax.add_patch(
                Arc(
                    (0.0, 0.0),
                    2 * 1.40,
                    2 * 1.40,
                    angle=0.0,
                    theta1=theta1,
                    theta2=theta2,
                    linewidth=3.0,
                    color="#C62828",
                    zorder=30,
                )
            )
            middle = np.deg2rad((start_deg + end_deg) / 2.0)
            ax.text(
                1.52 * cos(middle),
                1.52 * sin(middle),
                f"e_β={error_deg:+.2f}°",
                fontsize=7.0,
                fontweight="bold",
                color="#7A1D1D",
                ha="center",
                va="center",
                zorder=31,
            )

        ax.text(
            0.98,
            0.02,
            "Todos los vectores se expresan en el mismo marco del sensor.",
            transform=ax.transAxes,
            fontsize=6.9,
            color="#444444",
            ha="right",
            va="bottom",
        )

    def _dibujar_metricas_error_pose_landmark(self, ax, result, state):
        """Muestra residuos, incertidumbre, jacobianos y convergencia."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        evaluation = state["evaluation"]
        measurement = evaluation["measurement"]
        prediction = evaluation["prediction"]
        residual = evaluation["residual"]

        ax.text(
            0.015,
            0.955,
            "Valores calculados",
            fontsize=11.2,
            fontweight="bold",
            ha="left",
            va="top",
        )

        cards = [
            (
                "Medición z · fija",
                f"r={measurement[0]:.4f} m\nβ={degrees(measurement[1]):.4f}°",
                "#D5E8D4",
                "#2E8B57",
            ),
            (
                "Predicción h(x,l)",
                f"r̂={prediction[0]:.4f} m\nβ̂={degrees(prediction[1]):.4f}°",
                "#E8D7F1",
                "#8E5EA2",
            ),
            (
                "Residuo",
                f"e_r={residual[0]:+.4f} m\ne_β={degrees(residual[1]):+.4f}°",
                "#F7C6C7",
                "#C62828",
            ),
            (
                "Mahalanobis y coste",
                f"eᵀΩe={evaluation['mahalanobis']:.4f}\nF={evaluation['quadratic_cost']:.4f}",
                "#FBE5A6",
                "#8A6D1D",
            ),
        ]

        card_width = 0.145
        card_height = 0.48
        gap = 0.012
        start_x = 0.015
        for index, (title, value, face, edge) in enumerate(cards):
            x = start_x + index * (card_width + gap)
            ax.add_patch(
                Rectangle(
                    (x, 0.36),
                    card_width,
                    card_height,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.5,
                )
            )
            ax.text(
                x + card_width / 2,
                0.745,
                title,
                fontsize=6.9,
                fontweight="bold",
                color=edge,
                ha="center",
                va="center",
            )
            ax.text(
                x + card_width / 2,
                0.570,
                value,
                fontsize=6.6,
                family="monospace",
                ha="center",
                va="center",
                linespacing=1.45,
            )

        information = result["information"]
        contributions = evaluation["contributions"]
        ax.text(
            0.015,
            0.245,
            (
                "Σ = diag("
                f"{result['covariance'][0,0]:.5f}, {result['covariance'][1,1]:.7f})"
                "   ·   Ω = diag("
                f"{information[0,0]:.2f}, {information[1,1]:.2f})"
            ),
            fontsize=6.8,
            family="monospace",
            ha="left",
            va="center",
            color="#444444",
        )
        ax.text(
            0.015,
            0.145,
            (
                "contribuciones Mahalanobis: "
                f"rango={contributions[0]:.4f} · rumbo={contributions[1]:.4f}"
                f"   ·   Huber: w={evaluation['huber_weight']:.4f}, "
                f"ρ={evaluation['huber_cost']:.4f}"
            ),
            fontsize=6.8,
            ha="left",
            va="center",
            color="#444444",
        )

        # Zona derecha: jacobianos u optimización.
        right_x = 0.665
        right_width = 0.320
        ax.add_patch(
            Rectangle(
                (right_x, 0.36),
                right_width,
                0.48,
                facecolor="white",
                edgecolor="#777777",
                linewidth=1.3,
            )
        )

        if state.get("show_wrap", False):
            wrap = result["angle_wrap"]
            ax.text(
                right_x + right_width / 2,
                0.735,
                "Normalización angular",
                fontsize=8.0,
                fontweight="bold",
                color="#7A1D1D",
                ha="center",
            )
            ax.text(
                right_x + right_width / 2,
                0.575,
                (
                    f"crudo: {degrees(wrap['raw_error']):.1f}°\n"
                    f"normalizado: {degrees(wrap['normalized_error']):.1f}°"
                ),
                fontsize=8.5,
                family="monospace",
                fontweight="bold",
                ha="center",
                va="center",
                color="#7A1D1D",
            )
        elif state.get("show_jacobians", False) or state.get("focus") in {"observability", "summary"}:
            jac = result["jacobians"]
            obs = result["observability"]
            ax.text(
                right_x + right_width / 2,
                0.775,
                "Jacobianos y rango local",
                fontsize=7.8,
                fontweight="bold",
                color="#1F4F73",
                ha="center",
            )
            ax.text(
                right_x + 0.018,
                0.670,
                (
                    "A=∂e/∂x: 2×3\n"
                    "B=∂e/∂l: 2×2\n"
                    f"error A: {jac['pose_max_error']:.2e}\n"
                    f"error B: {jac['landmark_max_error']:.2e}"
                ),
                fontsize=6.5,
                family="monospace",
                ha="left",
                va="top",
            )
            ax.text(
                right_x + 0.185,
                0.670,
                (
                    f"conjunto: r={obs['joint']['rank']}, n={obs['joint']['nullity']}\n"
                    f"solo pose: r={obs['pose_only']['rank']}, n={obs['pose_only']['nullity']}\n"
                    f"solo landmark: r={obs['landmark_only']['rank']}, n={obs['landmark_only']['nullity']}\n"
                    f"lin. error: {result['linearization']['error_norm']:.2e}"
                ),
                fontsize=6.5,
                family="monospace",
                ha="left",
                va="top",
            )
        else:
            history = result["optimization"]["history"]
            costs = [result["optimization"]["initial_evaluation"]["quadratic_cost"]]
            costs.extend(entry["cost_after"] for entry in history)
            max_cost = max(costs) if costs else 1.0
            normalized = [value / max(max_cost, 1e-15) for value in costs]
            x_values = np.linspace(right_x + 0.04, right_x + right_width - 0.04, len(normalized))
            y_values = 0.42 + 0.29 * (1.0 - np.asarray(normalized))
            ax.plot(
                x_values,
                y_values,
                color="#4C9ED9",
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                zorder=4,
            )
            ax.text(
                right_x + right_width / 2,
                0.775,
                "Corrección del landmark",
                fontsize=7.8,
                fontweight="bold",
                color="#1F4F73",
                ha="center",
            )
            ax.text(
                right_x + right_width / 2,
                0.385,
                (
                    f"{result['optimization']['iterations']} iteraciones · "
                    f"F final={result['optimized_evaluation']['quadratic_cost']:.2e}"
                ),
                fontsize=6.6,
                color="#444444",
                ha="center",
            )

        ax.text(
            0.985,
            0.115,
            (
                "e_cart = "
                f"({evaluation['cartesian_residual'][0]:+.4f}, "
                f"{evaluation['cartesian_residual'][1]:+.4f}) m"
            ),
            fontsize=6.8,
            family="monospace",
            ha="right",
            va="center",
            color="#444444",
        )

    def _dibujar_estado_error_pose_landmark(
        self,
        info_ax,
        global_ax,
        local_ax,
        metrics_ax,
        result,
        state,
    ):
        """Dibuja un estado completo del apartado 6.5."""

        self._dibujar_info_error_pose_landmark(info_ax, result, state)
        self._dibujar_geometria_global_error_pose_landmark(global_ax, result, state)
        self._dibujar_marco_local_error_pose_landmark(local_ax, result, state)
        self._dibujar_metricas_error_pose_landmark(metrics_ax, result, state)

    def animate_pose_landmark_error(
        self,
        result,
        states,
        title="Error de observación pose-landmark",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima medida, predicción, residuo, incertidumbre y sensibilidad.

        La imagen final conserva la discrepancia inicial y muestra:
        - pose y landmark estimados;
        - rango y rumbo medidos;
        - rango y rumbo predichos;
        - errores de distancia y ángulo;
        - Mahalanobis, coste, jacobianos y observabilidad local.
        """

        if not states:
            raise ValueError(
                "La lista de estados del error pose-landmark no puede estar vacía."
            )
        if result is None:
            raise ValueError("El resultado pose-landmark no puede ser nulo.")

        required = {
            "graph",
            "true_pose",
            "initial_pose",
            "true_landmark",
            "initial_landmark",
            "optimized_landmark",
            "measurement",
            "covariance",
            "information",
            "initial_evaluation",
            "optimized_evaluation",
            "jacobians",
            "observability",
            "optimization",
            "angle_wrap",
        }
        missing = required.difference(result)
        if missing:
            raise ValueError(
                "Faltan datos del resultado: " + ", ".join(sorted(missing))
            )

        (
            fig,
            info_ax,
            global_ax,
            local_ax,
            metrics_ax,
        ) = self._preparar_figura_error_pose_landmark(title)

        if final_image_path is not None:
            self._dibujar_estado_error_pose_landmark(
                info_ax=info_ax,
                global_ax=global_ax,
                local_ax=local_ax,
                metrics_ax=metrics_ax,
                result=result,
                state=states[-1],
            )
            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(final_image_path, dpi=200, bbox_inches="tight")
            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_error_pose_landmark(
                info_ax=info_ax,
                global_ax=global_ax,
                local_ax=local_ax,
                metrics_ax=metrics_ax,
                result=result,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_error_pose_landmark(
                info_ax=info_ax,
                global_ax=global_ax,
                local_ax=local_ax,
                metrics_ax=metrics_ax,
                result=result,
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
    # Elementos específicos de asociación de datos
    # ------------------------------------------------------------------

    def _preparar_figura_asociacion_datos(self, title):
        """Crea la distribución visual del apartado de asociación de datos."""

        fig = plt.figure(figsize=self.figsize)
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=[1.50, 3.15, 3.15],
            height_ratios=[1.0, 1.0],
            wspace=0.11,
            hspace=0.16,
        )
        info_ax = fig.add_subplot(grid[:, 0])
        scene_ax = fig.add_subplot(grid[0, 1])
        matching_ax = fig.add_subplot(grid[0, 2])
        matrix_ax = fig.add_subplot(grid[1, 1])
        verification_ax = fig.add_subplot(grid[1, 2])

        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.925,
            bottom=0.055,
        )
        return (
            fig,
            info_ax,
            scene_ax,
            matching_ax,
            matrix_ax,
            verification_ax,
        )

    @staticmethod
    def _rotacion_asociacion_datos(theta):
        """Devuelve una matriz de rotación 2D para la visualización."""

        c = cos(float(theta))
        s = sin(float(theta))
        return np.array([[c, -s], [s, c]], dtype=float)

    @staticmethod
    def _colores_asociacion_datos():
        """Devuelve una paleta estable para candidatos y decisiones."""

        return {
            "landmark": "#222222",
            "observation": "#4C9ED9",
            "candidate": "#B8B8B8",
            "correct": "#2E8B57",
            "doubtful": "#F28E2B",
            "false": "#C62828",
            "new": "#8E5EA2",
            "selected": "#E45756",
            "gate": "#7AA6C2",
            "independent": "#8E5EA2",
            "global": "#2E8B57",
        }

    def _dibujar_leyenda_asociacion_datos(self, ax):
        """Dibuja una leyenda compacta en el panel de información."""

        colores = self._colores_asociacion_datos()
        elementos = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=colores["landmark"],
                markersize=8,
                label="Landmark del mapa",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor=colores["observation"],
                markeredgecolor="#1F4F73",
                markersize=8,
                label="Observación",
            ),
            Line2D(
                [0],
                [0],
                color=colores["candidate"],
                linewidth=2,
                label="Candidato",
            ),
            Line2D(
                [0],
                [0],
                color=colores["correct"],
                linewidth=3,
                label="Correcta aceptada",
            ),
            Line2D(
                [0],
                [0],
                color=colores["doubtful"],
                linewidth=3,
                linestyle="dashed",
                label="Dudosa",
            ),
            Line2D(
                [0],
                [0],
                color=colores["false"],
                linewidth=3,
                label="Falsa / rechazada",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=colores["new"],
                markeredgecolor="#5A316B",
                markersize=10,
                label="Observación nueva",
            ),
        ]
        ax.legend(
            handles=elementos,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            fontsize=6.8,
            framealpha=0.97,
            ncol=1,
            borderpad=0.55,
        )

    def _dibujar_info_asociacion_datos(self, ax, result, state):
        """Muestra fase, criterios, selección y métricas acumuladas."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        phase_titles = {
            "intro": "Problema",
            "landmarks": "Mapa conocido",
            "observations": "Observaciones",
            "descriptor_candidates": "Propuesta visual",
            "gating": "Gate geométrico",
            "cost_matrix": "Matriz de costes",
            "independent_matching": "Vecino independiente",
            "global_matching": "Matching global",
            "false_alias": "Aliasing perceptual",
            "ransac": "Verificación RANSAC",
            "decisions": "Decisión final",
            "false_effect": "Peligro del falso match",
            "metrics": "Evaluación",
            "summary": "Resumen",
        }
        phase = state.get("phase", "intro")
        ax.text(
            0.50,
            0.985,
            "Asociación de datos",
            fontsize=12.5,
            fontweight="bold",
            ha="center",
            va="top",
        )
        ax.text(
            0.50,
            0.940,
            phase_titles.get(phase, phase),
            fontsize=9.2,
            fontweight="bold",
            color="#1F4F73",
            ha="center",
            va="top",
        )

        ax.text(
            0.06,
            0.865,
            state.get("message", ""),
            fontsize=7.6,
            ha="left",
            va="top",
            wrap=True,
            bbox={
                "boxstyle": "round,pad=0.42",
                "fc": "white",
                "ec": "#888888",
                "alpha": 0.98,
            },
        )

        selected = state.get("selected_observation")
        if selected is not None:
            truth = result["true_associations"].get(selected)
            decision = result["decisions"].get(selected)
            lines = [
                f"Observación: {selected}",
                f"Origen real: {truth if truth is not None else 'nuevo'}",
            ]
            if decision is not None and state.get("show_global", False):
                assigned = decision.get("landmark")
                lines.append(
                    f"Asignación: {assigned if assigned is not None else '∅'}"
                )
                lines.append(f"Estado: {decision.get('status')}")
            ax.text(
                0.06,
                0.710,
                "\n".join(lines),
                fontsize=7.5,
                family="monospace",
                ha="left",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "fc": "#F7F7F7",
                    "ec": "#999999",
                },
            )

        ax.text(
            0.06,
            0.535,
            (
                f"Landmarks: {len(result['landmarks'])}\n"
                f"Observaciones: {len(result['observations'])}\n"
                f"Candidatos visuales: {len(result['candidates'])}\n"
                f"Pares dentro del gate: "
                f"{int(np.sum(np.isfinite(result['cost_matrix'])))}"
            ),
            fontsize=7.3,
            family="monospace",
            ha="left",
            va="top",
        )

        ax.text(
            0.06,
            0.395,
            (
                "Criterios\n"
                "─────────\n"
                "descriptor: top-3\n"
                "Mahalanobis ≤ 9.21\n"
                "asignación uno-a-uno\n"
                "RANSAC: error ≤ 0.26 m"
            ),
            fontsize=7.1,
            family="monospace",
            ha="left",
            va="top",
        )

        metrics = result["metrics"]
        if state.get("show_metrics", False) or phase == "summary":
            ax.text(
                0.06,
                0.235,
                (
                    f"precision = {metrics['precision']:.3f}\n"
                    f"recall    = {metrics['recall']:.3f}\n"
                    f"F1        = {metrics['f1']:.3f}\n"
                    f"factores  = {result['factor_graph'].number_of_edges()}"
                ),
                fontsize=7.4,
                family="monospace",
                fontweight="bold",
                color="#1F4F73",
                ha="left",
                va="top",
            )

        self._dibujar_leyenda_asociacion_datos(ax)

    def _dibujar_pose_asociacion_datos(self, ax, pose, color, label=None):
        """Dibuja una pose como punto y flecha orientada."""

        x, y, theta = pose
        longitud = 0.65
        ax.scatter(
            [x],
            [y],
            s=90,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=2.2,
            zorder=22,
        )
        ax.arrow(
            x,
            y,
            longitud * np.cos(theta),
            longitud * np.sin(theta),
            width=0.035,
            head_width=0.20,
            head_length=0.25,
            length_includes_head=True,
            color=color,
            zorder=23,
        )
        if label:
            ax.text(
                x,
                y - 0.42,
                label,
                fontsize=7.2,
                fontweight="bold",
                color=color,
                ha="center",
                va="top",
                zorder=25,
            )

    def _configurar_eje_escena_asociacion(self, ax, result, title):
        """Configura límites y cuadrícula de la escena geométrica."""

        ax.clear()
        ax.set_title(title, fontsize=10.2, fontweight="bold")
        ax.grid(True, alpha=0.20)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]", fontsize=7.5)
        ax.set_ylabel("y [m]", fontsize=7.5)
        ax.tick_params(labelsize=6.7)

        puntos = [result["true_pose"][:2], result["estimated_pose"][:2]]
        puntos.extend(result["landmarks"].values())
        puntos.extend(result["observed_global_estimated"].values())
        puntos = np.asarray(puntos, dtype=float)
        ax.set_xlim(np.min(puntos[:, 0]) - 1.0, np.max(puntos[:, 0]) + 1.0)
        ax.set_ylim(np.min(puntos[:, 1]) - 1.0, np.max(puntos[:, 1]) + 1.0)

    def _dibujar_escena_asociacion_datos(self, ax, result, state):
        """Dibuja robot, landmarks, observaciones, gates y asociaciones."""

        self._configurar_eje_escena_asociacion(
            ax,
            result,
            "Escena geométrica y asociaciones",
        )
        colores = self._colores_asociacion_datos()
        numero_lm = min(
            int(state.get("visible_landmarks", 0)),
            len(result["landmark_names"]),
        )
        numero_obs = min(
            int(state.get("visible_observations", 0)),
            len(result["observation_names"]),
        )
        landmarks_visibles = result["landmark_names"][:numero_lm]
        observaciones_visibles = result["observation_names"][:numero_obs]

        self._dibujar_pose_asociacion_datos(
            ax,
            result["true_pose"],
            "#555555",
            "pose real",
        )
        if numero_obs > 0:
            self._dibujar_pose_asociacion_datos(
                ax,
                result["estimated_pose"],
                "#4C9ED9",
                "pose estimada",
            )

        # Campo de visión aproximado.
        pose = result["estimated_pose"]
        for angulo in (
            pose[2] - np.deg2rad(62),
            pose[2] + np.deg2rad(62),
        ):
            ax.plot(
                [pose[0], pose[0] + 9.2 * np.cos(angulo)],
                [pose[1], pose[1] + 9.2 * np.sin(angulo)],
                color="#BBBBBB",
                linewidth=1.0,
                linestyle="dashed",
                zorder=2,
            )

        for nombre in landmarks_visibles:
            posicion = result["landmarks"][nombre]
            ax.scatter(
                [posicion[0]],
                [posicion[1]],
                s=105,
                marker="o",
                facecolor="white",
                edgecolor=colores["landmark"],
                linewidth=2.0,
                zorder=20,
            )
            ax.text(
                posicion[0],
                posicion[1] + 0.25,
                nombre,
                fontsize=7.4,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=25,
            )

        selected = state.get("selected_observation")
        for nombre in observaciones_visibles:
            punto = result["observed_global_estimated"][nombre]
            is_selected = nombre == selected
            ax.scatter(
                [punto[0]],
                [punto[1]],
                s=125 if is_selected else 90,
                marker="^",
                facecolor=(
                    colores["selected"]
                    if is_selected
                    else colores["observation"]
                ),
                edgecolor="#1F4F73",
                linewidth=1.4,
                zorder=24,
            )
            ax.text(
                punto[0],
                punto[1] - 0.25,
                nombre,
                fontsize=7.0,
                fontweight="bold",
                ha="center",
                va="top",
                zorder=25,
            )
            if state.get("show_gates", False):
                ax.add_patch(
                    Ellipse(
                        xy=punto,
                        width=0.72,
                        height=0.48,
                        angle=np.rad2deg(result["estimated_pose"][2]),
                        facecolor="none",
                        edgecolor=colores["gate"],
                        linewidth=1.0,
                        linestyle="dashed",
                        alpha=0.75,
                        zorder=5,
                    )
                )

        # Candidatos visuales en el orden en que aparecen.
        numero_candidatos = min(
            int(state.get("visible_candidates", 0)),
            len(result["candidates"]),
        )
        for candidato in result["candidates"][:numero_candidatos]:
            nombre_obs = candidato["observation"]
            nombre_lm = candidato["landmark"]
            if (
                nombre_obs not in observaciones_visibles
                or nombre_lm not in landmarks_visibles
            ):
                continue
            punto_obs = result["observed_global_estimated"][nombre_obs]
            punto_lm = result["landmarks"][nombre_lm]
            if candidato.get("false_visual_alias") and state.get("show_false_alias", False):
                color = colores["false"]
                width = 3.0
                style = "solid"
                alpha = 0.95
            elif candidato["inside_geometry_gate"]:
                color = "#7F9F7F"
                width = 1.7
                style = "solid"
                alpha = 0.75
            else:
                color = colores["candidate"]
                width = 1.0
                style = "dotted"
                alpha = 0.45
            ax.plot(
                [punto_obs[0], punto_lm[0]],
                [punto_obs[1], punto_lm[1]],
                color=color,
                linewidth=width,
                linestyle=style,
                alpha=alpha,
                zorder=7,
            )

        mapping = None
        mapping_color = None
        if state.get("show_independent", False):
            mapping = result["independent_associations"]
            mapping_color = colores["independent"]
        if state.get("show_global", False):
            mapping = result["global_assignment"]["associations"]
            mapping_color = colores["global"]

        if mapping is not None:
            for nombre_obs, nombre_lm in mapping.items():
                if nombre_obs not in observaciones_visibles or nombre_lm is None:
                    continue
                punto_obs = result["observed_global_estimated"][nombre_obs]
                punto_lm = result["landmarks"][nombre_lm]
                ax.plot(
                    [punto_obs[0], punto_lm[0]],
                    [punto_obs[1], punto_lm[1]],
                    color=mapping_color,
                    linewidth=2.7,
                    alpha=0.85,
                    zorder=12,
                )

        if state.get("show_decisions", False):
            for nombre_obs, decision in result["decisions"].items():
                if nombre_obs not in observaciones_visibles:
                    continue
                punto_obs = result["observed_global_estimated"][nombre_obs]
                status = decision["status"]
                nombre_lm = decision["landmark"]
                if status == "new":
                    ax.scatter(
                        [punto_obs[0]],
                        [punto_obs[1]],
                        s=210,
                        marker="*",
                        facecolor=colores["new"],
                        edgecolor="#5A316B",
                        linewidth=1.5,
                        zorder=31,
                    )
                    continue
                if nombre_lm is None:
                    continue
                punto_lm = result["landmarks"][nombre_lm]
                color = colores.get(status, colores["candidate"])
                style = "dashed" if status == "doubtful" else "solid"
                ax.plot(
                    [punto_obs[0], punto_lm[0]],
                    [punto_obs[1], punto_lm[1]],
                    color=color,
                    linewidth=3.2,
                    linestyle=style,
                    alpha=0.98,
                    zorder=30,
                )

        ax.text(
            0.02,
            0.02,
            "○ landmark   ▲ observación   elipse: gate local",
            transform=ax.transAxes,
            fontsize=6.6,
            color="#444444",
            ha="left",
            va="bottom",
        )

    def _dibujar_grafo_matching_asociacion_datos(self, ax, result, state):
        """Dibuja el grafo bipartito de candidatos y asignaciones."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Grafo bipartito de matching", fontsize=10.2, fontweight="bold")
        colores = self._colores_asociacion_datos()

        nombres_lm = result["landmark_names"]
        nombres_obs = result["observation_names"]
        ys_lm = np.linspace(0.84, 0.16, len(nombres_lm))
        ys_obs = np.linspace(0.87, 0.13, len(nombres_obs))
        pos_lm = {nombre: (0.17, y) for nombre, y in zip(nombres_lm, ys_lm)}
        pos_obs = {nombre: (0.83, y) for nombre, y in zip(nombres_obs, ys_obs)}

        numero_lm = min(state.get("visible_landmarks", 0), len(nombres_lm))
        numero_obs = min(state.get("visible_observations", 0), len(nombres_obs))
        visibles_lm = set(nombres_lm[:numero_lm])
        visibles_obs = set(nombres_obs[:numero_obs])
        selected = state.get("selected_observation")

        numero_candidatos = min(
            int(state.get("visible_candidates", 0)),
            len(result["candidates"]),
        )
        for candidato in result["candidates"][:numero_candidatos]:
            obs = candidato["observation"]
            lm = candidato["landmark"]
            if obs not in visibles_obs or lm not in visibles_lm:
                continue
            x1, y1 = pos_lm[lm]
            x2, y2 = pos_obs[obs]
            if candidato.get("false_visual_alias") and state.get("show_false_alias", False):
                color = colores["false"]
                width = 2.8
                style = "solid"
                alpha = 0.95
            elif candidato["inside_geometry_gate"]:
                color = "#7F9F7F"
                width = 1.6
                style = "solid"
                alpha = 0.75
            else:
                color = colores["candidate"]
                width = 0.9
                style = "dotted"
                alpha = 0.38
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=width,
                linestyle=style,
                alpha=alpha,
                zorder=5,
            )

        mapping = None
        color_mapping = None
        if state.get("show_independent", False):
            mapping = result["independent_associations"]
            color_mapping = colores["independent"]
        if state.get("show_global", False):
            mapping = result["global_assignment"]["associations"]
            color_mapping = colores["global"]
        if mapping is not None:
            for obs, lm in mapping.items():
                if obs not in visibles_obs or lm not in visibles_lm or lm is None:
                    continue
                ax.plot(
                    [pos_lm[lm][0], pos_obs[obs][0]],
                    [pos_lm[lm][1], pos_obs[obs][1]],
                    color=color_mapping,
                    linewidth=2.8,
                    alpha=0.90,
                    zorder=12,
                )

        if state.get("show_decisions", False):
            for obs, decision in result["decisions"].items():
                lm = decision["landmark"]
                if obs not in visibles_obs:
                    continue
                if decision["status"] == "new":
                    continue
                if lm is None or lm not in visibles_lm:
                    continue
                color = colores.get(decision["status"], colores["candidate"])
                style = "dashed" if decision["status"] == "doubtful" else "solid"
                ax.plot(
                    [pos_lm[lm][0], pos_obs[obs][0]],
                    [pos_lm[lm][1], pos_obs[obs][1]],
                    color=color,
                    linewidth=3.2,
                    linestyle=style,
                    zorder=20,
                )

        for nombre in nombres_lm:
            if nombre not in visibles_lm:
                continue
            x, y = pos_lm[nombre]
            ax.scatter(
                [x],
                [y],
                s=280,
                marker="o",
                facecolor="white",
                edgecolor=colores["landmark"],
                linewidth=1.7,
                zorder=25,
            )
            ax.text(x, y, nombre, fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=30)

        for nombre in nombres_obs:
            if nombre not in visibles_obs:
                continue
            x, y = pos_obs[nombre]
            decision = result["decisions"][nombre]
            if state.get("show_decisions", False) and decision["status"] == "new":
                marker = "*"
                face = colores["new"]
                size = 340
            else:
                marker = "^"
                face = colores["selected"] if nombre == selected else colores["observation"]
                size = 300 if nombre == selected else 250
            ax.scatter(
                [x],
                [y],
                s=size,
                marker=marker,
                facecolor=face,
                edgecolor="#1F4F73",
                linewidth=1.4,
                zorder=26,
            )
            ax.text(x, y, nombre, fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=30)

        ax.text(0.17, 0.96, "LANDMARKS", fontsize=7.4, fontweight="bold", ha="center")
        ax.text(0.83, 0.96, "OBSERVACIONES", fontsize=7.4, fontweight="bold", ha="center")

        if state.get("show_independent", False):
            subtitle = (
                f"Vecino independiente · duplicidades: "
                f"{result['method_comparison']['independent_duplicates']}"
            )
        elif state.get("show_global", False):
            subtitle = "Matching global · restricción uno-a-uno"
        else:
            subtitle = "Apariencia propone; geometría filtra"
        ax.text(
            0.50,
            0.035,
            subtitle,
            fontsize=7.2,
            color="#444444",
            ha="center",
            va="bottom",
        )

    def _dibujar_matriz_costes_asociacion_datos(self, ax, result, state):
        """Dibuja la matriz de costes y las celdas asignadas."""

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Matriz de costes después del gating", fontsize=10.2, fontweight="bold")

        nombres_obs = result["observation_names"]
        nombres_lm = result["landmark_names"]
        matriz = result["cost_matrix"]
        filas_visibles = (
            int(state.get("matrix_rows", 0))
            if state.get("show_cost_matrix", False)
            else 0
        )

        start_x = 0.16
        start_y = 0.12
        cell_w = 0.112
        cell_h = 0.095
        colores = self._colores_asociacion_datos()

        for columna, nombre_lm in enumerate(nombres_lm):
            ax.text(
                start_x + columna * cell_w + cell_w / 2,
                start_y + len(nombres_obs) * cell_h + 0.038,
                nombre_lm,
                fontsize=7.3,
                fontweight="bold",
                ha="center",
                va="center",
            )

        for fila, nombre_obs in enumerate(nombres_obs):
            y = start_y + (len(nombres_obs) - 1 - fila) * cell_h
            ax.text(
                start_x - 0.035,
                y + cell_h / 2,
                nombre_obs,
                fontsize=7.3,
                fontweight="bold",
                ha="right",
                va="center",
            )
            for columna, nombre_lm in enumerate(nombres_lm):
                x = start_x + columna * cell_w
                visible = fila < filas_visibles
                coste = matriz[fila, columna]
                face = "#F7F7F7"
                edge = "#BBBBBB"
                width = 0.8
                text = ""
                text_color = "#333333"

                if visible:
                    if np.isfinite(coste):
                        face = "#EAF2F7"
                        text = f"{coste:.2f}"
                    else:
                        face = "#ECECEC"
                        text = "×"
                        text_color = "#999999"

                    if state.get("show_independent", False):
                        if result["independent_associations"].get(nombre_obs) == nombre_lm:
                            edge = colores["independent"]
                            width = 2.5
                    if state.get("show_global", False):
                        if result["global_assignment"]["associations"].get(nombre_obs) == nombre_lm:
                            edge = colores["global"]
                            width = 2.7
                    if state.get("show_decisions", False):
                        decision = result["decisions"][nombre_obs]
                        if decision["landmark"] == nombre_lm:
                            edge = colores.get(decision["status"], edge)
                            width = 3.0

                ax.add_patch(
                    Rectangle(
                        (x, y),
                        cell_w,
                        cell_h,
                        facecolor=face,
                        edgecolor=edge,
                        linewidth=width,
                    )
                )
                if visible:
                    ax.text(
                        x + cell_w / 2,
                        y + cell_h / 2,
                        text,
                        fontsize=6.7,
                        family="monospace",
                        fontweight="bold" if width > 2.0 else "normal",
                        color=text_color,
                        ha="center",
                        va="center",
                    )

        ax.text(
            0.50,
            0.055,
            (
                "×: fuera del gate   ·   borde morado: NN independiente   ·   "
                "borde verde: matching global"
            ),
            fontsize=6.4,
            color="#444444",
            ha="center",
            va="center",
        )

    def _dibujar_verificacion_asociacion_datos(self, ax, result, state):
        """Muestra RANSAC, métricas y el efecto de un falso factor."""

        ax.clear()
        ax.set_title("Verificación geométrica y riesgo", fontsize=10.2, fontweight="bold")
        colores = self._colores_asociacion_datos()

        if state.get("show_ransac", False):
            corr = result["ransac_correspondences"]
            ransac = result["ransac"]
            transformados = (
                np.asarray(
                    [
                        result["true_pose"][:2]
                    ]
                )
            )
            transformados = (
                np.asarray(corr["source_points"]) @ self._rotacion_asociacion_datos(ransac["pose"][2]).T
                + ransac["pose"][:2]
            )
            destinos = np.asarray(corr["target_points"])
            todos = np.vstack([transformados, destinos])
            ax.set_xlim(np.min(todos[:, 0]) - 0.6, np.max(todos[:, 0]) + 0.6)
            ax.set_ylim(np.min(todos[:, 1]) - 0.6, np.max(todos[:, 1]) + 0.6)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.18)
            ax.tick_params(labelsize=6.2)

            for indice, ((obs, lm), p, q) in enumerate(
                zip(corr["pairs"], transformados, destinos)
            ):
                es_inlier = bool(ransac["inliers"][indice])
                color = colores["correct"] if es_inlier else colores["false"]
                ax.plot(
                    [p[0], q[0]],
                    [p[1], q[1]],
                    color=color,
                    linewidth=2.0 if es_inlier else 2.5,
                    linestyle="solid" if es_inlier else "dashed",
                    alpha=0.85,
                    zorder=8,
                )
                ax.scatter(
                    [p[0]],
                    [p[1]],
                    s=55,
                    marker="^",
                    facecolor="#4C9ED9",
                    edgecolor="#1F4F73",
                    zorder=12,
                )
                ax.scatter(
                    [q[0]],
                    [q[1]],
                    s=55,
                    marker="o",
                    facecolor="white",
                    edgecolor="#222222",
                    zorder=12,
                )
                ax.text(
                    (p[0] + q[0]) / 2,
                    (p[1] + q[1]) / 2,
                    f"{obs}-{lm}",
                    fontsize=5.5,
                    color=color,
                    ha="center",
                    va="center",
                )

            ax.text(
                0.02,
                0.98,
                (
                    f"RANSAC: {ransac['inlier_count']} inliers · "
                    f"{ransac['outlier_count']} outliers\n"
                    f"RMSE = {ransac['rmse']:.3f} m"
                ),
                transform=ax.transAxes,
                fontsize=7.0,
                fontweight="bold",
                ha="left",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.30",
                    "fc": "white",
                    "ec": "#777777",
                    "alpha": 0.96,
                },
            )
        else:
            ax.axis("off")
            ax.text(
                0.50,
                0.58,
                "La geometría conjunta\nse mostrará mediante RANSAC",
                fontsize=10,
                fontweight="bold",
                color="#666666",
                ha="center",
                va="center",
            )

        if state.get("show_false_effect", False):
            efecto = result["false_effect"]
            valores = [
                efecto["translation_shift_false"],
                efecto["translation_shift_robust"],
            ]
            maximo = max(max(valores), 1e-9)
            panel_x = 0.57
            panel_y = 0.04
            panel_w = 0.40
            panel_h = 0.27
            ax.add_patch(
                Rectangle(
                    (panel_x, panel_y),
                    panel_w,
                    panel_h,
                    transform=ax.transAxes,
                    facecolor="white",
                    edgecolor="#777777",
                    linewidth=1.0,
                    alpha=0.95,
                    zorder=40,
                )
            )
            ax.text(
                panel_x + panel_w / 2,
                panel_y + panel_h - 0.025,
                "Desplazamiento por z5-l2",
                transform=ax.transAxes,
                fontsize=6.3,
                fontweight="bold",
                ha="center",
                va="top",
                zorder=42,
            )
            base_y = panel_y + 0.055
            usable_h = panel_h - 0.105
            centers = [panel_x + panel_w * 0.31, panel_x + panel_w * 0.72]
            labels = ["sin robustez", "Huber"]
            for center, valor, label in zip(centers, valores, labels):
                altura = usable_h * valor / maximo
                ax.add_patch(
                    Rectangle(
                        (center - 0.035, base_y),
                        0.070,
                        altura,
                        transform=ax.transAxes,
                        facecolor="#4C9ED9",
                        edgecolor="#1F4F73",
                        linewidth=0.8,
                        zorder=41,
                    )
                )
                ax.text(
                    center,
                    base_y + altura + 0.008,
                    f"{valor:.3f} m",
                    transform=ax.transAxes,
                    fontsize=5.5,
                    ha="center",
                    va="bottom",
                    zorder=42,
                )
                ax.text(
                    center,
                    panel_y + 0.018,
                    label,
                    transform=ax.transAxes,
                    fontsize=5.2,
                    ha="center",
                    va="bottom",
                    zorder=42,
                )

        if state.get("show_metrics", False):
            metrics = result["metrics"]
            ax.text(
                0.02,
                0.02,
                (
                    f"precision={metrics['precision']:.3f} · "
                    f"recall={metrics['recall']:.3f} · F1={metrics['f1']:.3f}"
                ),
                transform=ax.transAxes,
                fontsize=6.8,
                fontweight="bold",
                color="#1F4F73",
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.95,
                },
            )

    def _dibujar_estado_asociacion_datos(
        self,
        info_ax,
        scene_ax,
        matching_ax,
        matrix_ax,
        verification_ax,
        result,
        state,
    ):
        """Dibuja un estado completo de asociación de datos."""

        self._dibujar_info_asociacion_datos(info_ax, result, state)
        self._dibujar_escena_asociacion_datos(scene_ax, result, state)
        self._dibujar_grafo_matching_asociacion_datos(matching_ax, result, state)
        self._dibujar_matriz_costes_asociacion_datos(matrix_ax, result, state)
        self._dibujar_verificacion_asociacion_datos(verification_ax, result, state)

    def animate_data_association(
        self,
        result,
        states,
        title="Asociación de datos",
        final_image_path=None,
        repeat=False,
    ):
        """
        Anima propuesta visual, gating, matching global y RANSAC.

        La imagen final muestra:
        - landmarks y observaciones;
        - candidatos visuales y geométricos;
        - matriz de costes;
        - asociaciones correctas, dudosas y nuevas;
        - alias perceptual rechazado;
        - inliers y outliers de RANSAC;
        - riesgo de aceptar una asociación falsa.
        """

        if not states:
            raise ValueError(
                "La lista de estados de asociación de datos no puede estar vacía."
            )
        if result is None:
            raise ValueError("El resultado de asociación no puede ser nulo.")

        required = {
            "true_pose",
            "estimated_pose",
            "landmarks",
            "observations",
            "candidates",
            "cost_matrix",
            "independent_associations",
            "global_assignment",
            "decisions",
            "ransac",
            "false_effect",
            "metrics",
        }
        missing = required.difference(result)
        if missing:
            raise ValueError(
                "Faltan datos del resultado: " + ", ".join(sorted(missing))
            )

        (
            fig,
            info_ax,
            scene_ax,
            matching_ax,
            matrix_ax,
            verification_ax,
        ) = self._preparar_figura_asociacion_datos(title)

        if final_image_path is not None:
            self._dibujar_estado_asociacion_datos(
                info_ax=info_ax,
                scene_ax=scene_ax,
                matching_ax=matching_ax,
                matrix_ax=matrix_ax,
                verification_ax=verification_ax,
                result=result,
                state=states[-1],
            )
            final_image_path = Path(final_image_path)
            final_image_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(final_image_path, dpi=200, bbox_inches="tight")
            print(f"Imagen final guardada en: {final_image_path}")

        def init():
            self._dibujar_estado_asociacion_datos(
                info_ax=info_ax,
                scene_ax=scene_ax,
                matching_ax=matching_ax,
                matrix_ax=matrix_ax,
                verification_ax=verification_ax,
                result=result,
                state=states[0],
            )
            return []

        def update(frame_index):
            self._dibujar_estado_asociacion_datos(
                info_ax=info_ax,
                scene_ax=scene_ax,
                matching_ax=matching_ax,
                matrix_ax=matrix_ax,
                verification_ax=verification_ax,
                result=result,
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
