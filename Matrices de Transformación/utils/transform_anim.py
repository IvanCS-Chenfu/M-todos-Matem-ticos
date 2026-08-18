from pathlib import Path
import shutil
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class TransformAnimator:
    """
    Clase reutilizable para crear animaciones de geometría y transformaciones.

    La clase se ocupa únicamente de la visualización. Los scripts de cada
    apartado deben calcular los puntos, vectores, frames y estados que desean
    mostrar y entregarlos al animador.

    La animación 2D genérica es capaz de representar:
    - sistemas de referencia cartesianos,
    - puntos y vectores,
    - segmentos, polilíneas y polígonos,
    - cuadrículas deformadas calculadas por cada demo,
    - textos y leyendas pedagógicas,
    - información numérica asociada a cada estado,
    - imagen final y vídeo WebM/MP4.

    Más adelante se podrá ampliar con métodos específicos para:
    - transformaciones homogéneas,
    - composición de transformaciones,
    - rotaciones 3D,
    - SE(2) y SE(3),
    - trayectorias y cadenas de frames.
    """

    def __init__(self, figsize=(15, 9), interval=50):
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
    def _vector_2d(valor, nombre):
        """
        Convierte un valor a vector NumPy 2D y valida sus dimensiones.
        """

        vector = np.asarray(valor, dtype=float).reshape(-1)

        if vector.shape != (2,):
            raise ValueError(
                f"{nombre} debe contener exactamente dos componentes. "
                f"Se recibió una forma {vector.shape}."
            )

        return vector

    @staticmethod
    def _ejes_desde_angulo(angulo):
        """
        Devuelve los versores x e y de un frame 2D con orientación `angulo`.
        """

        coseno = np.cos(angulo)
        seno = np.sin(angulo)

        eje_x = np.array([coseno, seno], dtype=float)
        eje_y = np.array([-seno, coseno], dtype=float)

        return eje_x, eje_y

    @staticmethod
    def _crear_directorio_salida(path):
        """
        Crea el directorio padre de una salida si todavía no existe.
        """

        if path is None:
            return

        Path(path).expanduser().resolve().parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _preparar_figura(self, title):
        """
        Crea una figura con una escena geométrica y un panel de información.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[3.4, 1.55],
            wspace=0.08,
        )

        scene_ax = fig.add_subplot(grid[0])
        info_ax = fig.add_subplot(grid[1])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        return fig, scene_ax, info_ax

    @staticmethod
    def _configurar_escena(ax, limits):
        """
        Configura límites, cuadrícula y aspecto cartesiano de la escena 2D.
        """

        x_min, x_max, y_min, y_max = limits

        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, linewidth=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.axhline(0.0, linewidth=0.8, alpha=0.18)
        ax.axvline(0.0, linewidth=0.8, alpha=0.18)

    def _dibujar_frame_2d(self, ax, frame):
        """
        Dibuja un sistema de referencia cartesiano 2D.

        El diccionario `frame` admite:
        - name: nombre visible,
        - origin: origen (x, y),
        - angle: orientación en radianes,
        - length: longitud de los ejes,
        - alpha: transparencia,
        - x_color / y_color: colores de los ejes,
        - linewidth: grosor de las flechas.
        """

        name = frame.get("name", "frame")
        origin = self._vector_2d(frame.get("origin", (0.0, 0.0)), "origin")
        angle = float(frame.get("angle", 0.0))
        length = float(frame.get("length", 1.5))
        alpha = float(frame.get("alpha", 1.0))
        linewidth = float(frame.get("linewidth", 2.4))

        x_color = frame.get("x_color", "#C63C3C")
        y_color = frame.get("y_color", "#2A8F5B")

        eje_x, eje_y = self._ejes_desde_angulo(angle)

        extremo_x = origin + length * eje_x
        extremo_y = origin + length * eje_y

        flecha_x = FancyArrowPatch(
            posA=origin,
            posB=extremo_x,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=linewidth,
            color=x_color,
            alpha=alpha,
            zorder=30,
        )
        flecha_y = FancyArrowPatch(
            posA=origin,
            posB=extremo_y,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=linewidth,
            color=y_color,
            alpha=alpha,
            zorder=30,
        )

        ax.add_patch(flecha_x)
        ax.add_patch(flecha_y)

        ax.scatter(
            [origin[0]],
            [origin[1]],
            s=32,
            color="#222222",
            alpha=alpha,
            zorder=35,
        )

        ax.text(
            extremo_x[0],
            extremo_x[1],
            f"  x_{name}",
            fontsize=10,
            fontweight="bold",
            color=x_color,
            alpha=alpha,
            ha="left",
            va="center",
            zorder=40,
        )

        ax.text(
            extremo_y[0],
            extremo_y[1],
            f"  y_{name}",
            fontsize=10,
            fontweight="bold",
            color=y_color,
            alpha=alpha,
            ha="left",
            va="center",
            zorder=40,
        )

        ax.text(
            origin[0] + 0.10,
            origin[1] - 0.18,
            f"{{{name}}}",
            fontsize=10,
            fontweight="bold",
            color="#222222",
            alpha=alpha,
            ha="left",
            va="top",
            zorder=40,
            bbox={
                "boxstyle": "round,pad=0.20",
                "fc": "white",
                "ec": "#BBBBBB",
                "alpha": 0.88 * alpha,
            },
        )

    def _dibujar_punto_2d(self, ax, point):
        """
        Dibuja un punto geométrico 2D.
        """

        position = self._vector_2d(point["position"], "point.position")
        name = point.get("name", "P")
        color = point.get("color", "#7B2CBF")
        alpha = float(point.get("alpha", 1.0))
        size = float(point.get("size", 95))

        ax.scatter(
            [position[0]],
            [position[1]],
            s=size,
            color=color,
            edgecolor="#222222",
            linewidth=1.0,
            alpha=alpha,
            zorder=45,
        )

        if name:
            label_offset = self._vector_2d(
                point.get("label_offset", (0.14, 0.14)),
                "point.label_offset",
            )
            ax.text(
                position[0] + label_offset[0],
                position[1] + label_offset[1],
                name,
                fontsize=float(point.get("fontsize", 11)),
                fontweight=point.get("fontweight", "bold"),
                color=color,
                alpha=alpha,
                zorder=float(point.get("zorder", 50)),
            )

    def _dibujar_vector_2d(self, ax, vector):
        """
        Dibuja un vector geométrico 2D desde un punto de anclaje.
        """

        origin = self._vector_2d(vector.get("origin", (0.0, 0.0)), "vector.origin")
        value = self._vector_2d(vector["value"], "vector.value")
        name = vector.get("name", "v")
        color = vector.get("color", "#E07A1F")
        alpha = float(vector.get("alpha", 1.0))
        linewidth = float(vector.get("linewidth", 3.0))

        end = origin + value

        flecha = FancyArrowPatch(
            posA=origin,
            posB=end,
            arrowstyle=vector.get("arrowstyle", "-|>"),
            mutation_scale=float(vector.get("mutation_scale", 18)),
            linewidth=linewidth,
            linestyle=vector.get("linestyle", "-"),
            color=color,
            alpha=alpha,
            zorder=float(vector.get("zorder", 42)),
        )
        ax.add_patch(flecha)

        if vector.get("show_origin", True):
            ax.scatter(
                [origin[0]],
                [origin[1]],
                s=float(vector.get("origin_size", 20)),
                color=color,
                alpha=alpha,
                zorder=float(vector.get("zorder", 42)) + 1,
            )

        if name:
            label_offset = self._vector_2d(
                vector.get("label_offset", (0.14, 0.10)),
                "vector.label_offset",
            )
            ax.text(
                end[0] + label_offset[0],
                end[1] + label_offset[1],
                name,
                fontsize=float(vector.get("fontsize", 11)),
                fontweight=vector.get("fontweight", "bold"),
                color=color,
                alpha=alpha,
                zorder=float(vector.get("zorder", 42)) + 8,
            )

    def _dibujar_segmento_2d(self, ax, segment):
        """
        Dibuja una línea auxiliar entre dos puntos.
        """

        start = self._vector_2d(segment["start"], "segment.start")
        end = self._vector_2d(segment["end"], "segment.end")

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            linestyle=segment.get("linestyle", "--"),
            linewidth=float(segment.get("linewidth", 1.4)),
            color=segment.get("color", "#777777"),
            alpha=float(segment.get("alpha", 0.7)),
            zorder=12,
        )

    def _dibujar_arco_2d(self, ax, arc):
        """
        Dibuja un arco 2D, útil para representar ángulos de rotación.

        El diccionario `arc` admite:
        - center: centro del arco,
        - radius: radio si se quiere un arco circular,
        - width / height: dimensiones opcionales para un arco elíptico,
        - theta1 / theta2: ángulos inicial y final en grados,
        - color, linewidth, linestyle, alpha y zorder.
        """

        center = self._vector_2d(arc.get("center", (0.0, 0.0)), "arc.center")
        radius = float(arc.get("radius", 1.0))
        width = float(arc.get("width", 2.0 * radius))
        height = float(arc.get("height", 2.0 * radius))

        patch = Arc(
            xy=center,
            width=width,
            height=height,
            angle=float(arc.get("angle", 0.0)),
            theta1=float(arc.get("theta1", 0.0)),
            theta2=float(arc.get("theta2", 90.0)),
            color=arc.get("color", "#7B2CBF"),
            linewidth=float(arc.get("linewidth", 2.0)),
            linestyle=arc.get("linestyle", "-"),
            alpha=float(arc.get("alpha", 0.9)),
            zorder=float(arc.get("zorder", 25)),
        )
        ax.add_patch(patch)

    def _dibujar_polilinea_2d(self, ax, polyline):
        """
        Dibuja una polilínea formada por una secuencia de puntos 2D.

        Este elemento es útil para cuadrículas, contornos y trayectorias. La
        geometría se calcula en el script del temario; el animador solo dibuja.
        """

        points = np.asarray(polyline["points"], dtype=float)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("polyline.points debe tener forma (N, 2).")

        ax.plot(
            points[:, 0],
            points[:, 1],
            linestyle=polyline.get("linestyle", "-"),
            linewidth=float(polyline.get("linewidth", 1.2)),
            color=polyline.get("color", "#777777"),
            alpha=float(polyline.get("alpha", 0.65)),
            zorder=float(polyline.get("zorder", 10)),
        )

    def _dibujar_poligono_2d(self, ax, polygon):
        """
        Dibuja un polígono 2D, por ejemplo el paralelogramo de una base.
        """

        points = np.asarray(polygon["points"], dtype=float)

        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("polygon.points debe tener forma (N, 2), N >= 3.")

        patch = Polygon(
            points,
            closed=bool(polygon.get("closed", True)),
            facecolor=polygon.get("facecolor", "#D9EAF7"),
            edgecolor=polygon.get("edgecolor", "#4472A8"),
            linewidth=float(polygon.get("linewidth", 1.5)),
            linestyle=polygon.get("linestyle", "-"),
            alpha=float(polygon.get("alpha", 0.28)),
            zorder=float(polygon.get("zorder", 8)),
        )
        ax.add_patch(patch)

    def _dibujar_texto_2d(self, ax, text_item):
        """
        Dibuja una anotación breve dentro de la escena geométrica.
        """

        position = self._vector_2d(text_item["position"], "text.position")
        text = str(text_item.get("text", ""))

        if not text:
            return

        kwargs = {
            "fontsize": float(text_item.get("fontsize", 10)),
            "fontweight": text_item.get("fontweight", "normal"),
            "color": text_item.get("color", "#222222"),
            "alpha": float(text_item.get("alpha", 1.0)),
            "ha": text_item.get("ha", "left"),
            "va": text_item.get("va", "center"),
            "zorder": float(text_item.get("zorder", 60)),
        }

        bbox = text_item.get("bbox")
        if bbox is not None:
            kwargs["bbox"] = bbox

        ax.text(position[0], position[1], text, **kwargs)

    @staticmethod
    def _crear_elemento_leyenda(item):
        """
        Construye un Line2D sencillo a partir de una especificación de leyenda.
        """

        kind = item.get("kind", "line")
        label = item.get("label", "")
        color = item.get("color", "#555555")

        if kind == "point":
            return Line2D(
                [0],
                [0],
                marker=item.get("marker", "o"),
                color="none",
                markerfacecolor=color,
                markeredgecolor=item.get("edgecolor", "#222222"),
                markersize=float(item.get("markersize", 8)),
                label=label,
            )

        return Line2D(
            [0],
            [0],
            color=color,
            linewidth=float(item.get("linewidth", 2.5)),
            linestyle=item.get("linestyle", "-"),
            marker=item.get("marker", None),
            label=label,
        )

    def _dibujar_mensaje(self, ax, state):
        """
        Añade el mensaje pedagógico asociado al estado actual.
        """

        message = state.get("message", "")

        if not message:
            return

        message = textwrap.fill(str(message), width=76)

        ax.text(
            0.50,
            0.025,
            message,
            transform=ax.transAxes,
            fontsize=9.5,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.45",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
            zorder=80,
        )

    @staticmethod
    def _dibujar_info(ax, state):
        """
        Dibuja el panel textual con coordenadas y observaciones del estado.
        """

        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(
            0.04,
            0.96,
            state.get("info_title", "Información del estado"),
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
        )

        lines = list(state.get("info_lines", []))

        y = float(state.get("info_start_y", 0.89))
        line_height = float(state.get("info_line_height", 0.052))
        info_fontsize = float(state.get("info_fontsize", 10))

        for line in lines:
            if isinstance(line, dict):
                text = str(line.get("text", ""))
                bold = bool(line.get("bold", False))
                color = line.get("color", "#222222")
                spacing = float(line.get("spacing", 1.0))
            else:
                text = str(line)
                bold = False
                color = "#222222"
                spacing = 1.0

            ax.text(
                0.04,
                y,
                text,
                fontsize=info_fontsize,
                fontweight="bold" if bold else "normal",
                color=color,
                family="monospace" if line and not isinstance(line, dict) else None,
                ha="left",
                va="top",
            )

            y -= line_height * spacing

        phase = state.get("phase")
        if phase:
            # El indicador de fase se coloca en el borde inferior del panel.
            # La última línea de información puede llegar aproximadamente a
            # y=0.11, por lo que usar y=0.02 evita que el recuadro la tape.
            ax.text(
                0.04,
                0.02,
                phase,
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.40",
                    "fc": "white",
                    "ec": "#999999",
                    "alpha": 0.96,
                },
            )

    def _dibujar_estado_2d(self, scene_ax, info_ax, state, limits):
        """
        Dibuja por completo un estado de una animación geométrica 2D.
        """

        self._configurar_escena(scene_ax, limits)

        for polygon in state.get("polygons", []):
            self._dibujar_poligono_2d(scene_ax, polygon)

        for polyline in state.get("polylines", []):
            self._dibujar_polilinea_2d(scene_ax, polyline)

        for segment in state.get("segments", []):
            self._dibujar_segmento_2d(scene_ax, segment)

        for arc in state.get("arcs", []):
            self._dibujar_arco_2d(scene_ax, arc)

        for frame in state.get("frames", []):
            self._dibujar_frame_2d(scene_ax, frame)

        for point in state.get("points", []):
            self._dibujar_punto_2d(scene_ax, point)

        for vector in state.get("vectors", []):
            self._dibujar_vector_2d(scene_ax, vector)

        for text_item in state.get("texts", []):
            self._dibujar_texto_2d(scene_ax, text_item)

        self._dibujar_mensaje(scene_ax, state)
        self._dibujar_info(info_ax, state)

        legend_elements = []

        if state.get("legend"):
            legend_elements = [
                self._crear_elemento_leyenda(item)
                for item in state["legend"]
                if item.get("label")
            ]
        else:
            # Compatibilidad con las primeras demos: si no se proporciona una
            # leyenda específica se mantiene la leyenda mínima original.
            if state.get("points"):
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor="#7B2CBF",
                        markeredgecolor="#222222",
                        markersize=8,
                        label="Punto geométrico",
                    )
                )

            if state.get("vectors"):
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="#E07A1F",
                        linewidth=3,
                        label="Vector geométrico",
                    )
                )

        if legend_elements:
            scene_ax.legend(
                handles=legend_elements,
                loc=state.get("legend_loc", "upper left"),
                fontsize=float(state.get("legend_fontsize", 9)),
                framealpha=0.95,
                ncol=int(state.get("legend_ncol", 1)),
            )

    def _guardar_video(self, animation, video_path, fps, dpi):
        """
        Guarda una animación mediante ffmpeg.

        Para WebM se utiliza VP9. Si ffmpeg no está disponible se muestra un
        aviso y la animación sigue pudiéndose visualizar en pantalla.
        """

        video_path = Path(video_path).expanduser().resolve()
        self._crear_directorio_salida(video_path)

        if shutil.which("ffmpeg") is None:
            print(
                "\n[AVISO] No se ha encontrado ffmpeg. "
                "No se guardará el vídeo."
            )
            return False

        suffix = video_path.suffix.lower()

        if suffix == ".webm":
            writer = FFMpegWriter(
                fps=fps,
                codec="libvpx-vp9",
                bitrate=-1,
                extra_args=[
                    "-crf",
                    "32",
                    "-b:v",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                ],
                metadata={"artist": "M-todos-Matem-ticos"},
            )
        else:
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=2400,
                extra_args=["-pix_fmt", "yuv420p"],
                metadata={"artist": "M-todos-Matem-ticos"},
            )

        print(f"\nGuardando vídeo en:\n  {video_path}")
        animation.save(str(video_path), writer=writer, dpi=dpi)
        print("Vídeo guardado correctamente.")

        return True

    def animate_2d_states(
        self,
        states,
        title,
        limits=(-5.0, 5.0, -4.0, 4.0),
        final_image_path=None,
        video_path=None,
        repeat=False,
        fps=None,
        dpi=130,
        show=True,
    ):
        """
        Anima una secuencia genérica de estados geométricos 2D.

        Parameters
        ----------
        states:
            Lista de diccionarios. Cada estado puede contener `frames`,
            `points`, `vectors`, `segments`, `arcs`, `polylines`, `polygons`, `texts`,
            `legend`, `message`, `info_lines` y `phase`.
        title:
            Título general de la figura.
        limits:
            (xmin, xmax, ymin, ymax) de la escena.
        final_image_path:
            Ruta opcional para guardar como PNG el último estado.
        video_path:
            Ruta opcional para guardar la animación. Se recomienda `.webm`.
        repeat:
            Si la animación debe reiniciarse al terminar.
        fps:
            Fotogramas por segundo del vídeo. Si se omite se deduce de
            `interval`.
        dpi:
            Resolución utilizada al guardar imagen y vídeo.
        show:
            Si se debe abrir la ventana interactiva de Matplotlib.
        """

        states = list(states)

        if not states:
            raise ValueError("La animación necesita al menos un estado.")

        fig, scene_ax, info_ax = self._preparar_figura(title)

        # La captura estática se genera antes de construir FuncAnimation.
        # FuncAnimation registra callbacks de dibujo que pueden forzar el
        # primer frame al ejecutar fig.canvas.draw(); hacerlo en este orden
        # garantiza que el PNG corresponda realmente al último estado.
        if final_image_path is not None:
            self._dibujar_estado_2d(
                scene_ax,
                info_ax,
                states[-1],
                limits,
            )
            final_image_path = Path(final_image_path).expanduser().resolve()
            self._crear_directorio_salida(final_image_path)
            fig.savefig(
                final_image_path,
                dpi=dpi,
                bbox_inches="tight",
            )
            print(f"\nImagen final guardada en:\n  {final_image_path}")

        def actualizar(frame_index):
            self._dibujar_estado_2d(
                scene_ax,
                info_ax,
                states[frame_index],
                limits,
            )

            return []

        self.animation = FuncAnimation(
            fig,
            actualizar,
            frames=len(states),
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        if fps is None:
            fps = max(1, int(round(1000.0 / self.interval)))

        if video_path is not None:
            self._guardar_video(
                self.animation,
                video_path=video_path,
                fps=fps,
                dpi=dpi,
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

        return self.animation

    # ------------------------------------------------------------------
    # Visualización 3D
    # ------------------------------------------------------------------

    @staticmethod
    def _vector_3d(valor, nombre):
        """
        Convierte un valor a vector NumPy 3D y valida sus dimensiones.
        """

        vector = np.asarray(valor, dtype=float).reshape(-1)

        if vector.shape != (3,):
            raise ValueError(
                f"{nombre} debe contener exactamente tres componentes. "
                f"Se recibió una forma {vector.shape}."
            )

        return vector

    @staticmethod
    def _matriz_rotacion_3d(valor, nombre):
        """
        Convierte un valor a matriz de orientación 3x3.
        """

        matriz = np.asarray(valor, dtype=float)

        if matriz.shape != (3, 3):
            raise ValueError(
                f"{nombre} debe tener forma (3, 3). "
                f"Se recibió una forma {matriz.shape}."
            )

        return matriz

    def _preparar_figura_3d(self, title):
        """
        Crea una figura con una escena 3D y un panel lateral de información.
        """

        fig = plt.figure(figsize=self.figsize)

        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[3.5, 1.55],
            wspace=0.05,
        )

        scene_ax = fig.add_subplot(grid[0], projection="3d")
        info_ax = fig.add_subplot(grid[1])

        fig.suptitle(
            title,
            fontsize=15,
            fontweight="bold",
        )

        return fig, scene_ax, info_ax

    @staticmethod
    def _configurar_escena_3d(ax, limits, view):
        """
        Configura límites, etiquetas y cámara de una escena cartesiana 3D.
        """

        x_min, x_max, y_min, y_max, z_min, z_max = limits

        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.grid(True, alpha=0.22)
        ax.view_init(
            elev=float(view[0]),
            azim=float(view[1]),
        )

        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(
                (
                    max(x_max - x_min, 1e-9),
                    max(y_max - y_min, 1e-9),
                    max(z_max - z_min, 1e-9),
                )
            )

        try:
            ax.set_proj_type("persp")
        except AttributeError:
            pass

    def _dibujar_frame_3d(self, ax, frame):
        """
        Dibuja un sistema de referencia cartesiano 3D.

        El frame se define mediante un origen y una matriz de orientación R cuyas
        columnas son los ejes x, y, z expresados en las coordenadas de la escena.
        """

        name = frame.get("name", "frame")
        origin = self._vector_3d(
            frame.get("origin", (0.0, 0.0, 0.0)),
            "frame.origin",
        )
        rotation = self._matriz_rotacion_3d(
            frame.get("rotation", np.eye(3)),
            "frame.rotation",
        )

        length = float(frame.get("length", 1.5))
        alpha = float(frame.get("alpha", 1.0))
        linewidth = float(frame.get("linewidth", 2.5))
        colors = frame.get(
            "colors",
            ("#C63C3C", "#2A8F5B", "#1F77B4"),
        )
        labels = frame.get("axis_labels", ("x", "y", "z"))

        for indice, (color, label) in enumerate(zip(colors, labels)):
            direction = length * rotation[:, indice]

            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                arrow_length_ratio=0.12,
                normalize=False,
            )

            end = origin + direction
            axis_name = f"{label}_{name}" if name else str(label)

            ax.text(
                end[0],
                end[1],
                end[2],
                f" {axis_name}",
                fontsize=float(frame.get("fontsize", 9)),
                fontweight="bold",
                color=color,
                alpha=alpha,
            )

        ax.scatter(
            [origin[0]],
            [origin[1]],
            [origin[2]],
            s=float(frame.get("origin_size", 28)),
            color=frame.get("origin_color", "#222222"),
            alpha=alpha,
        )

        if name:
            offset = self._vector_3d(
                frame.get("label_offset", (0.10, 0.10, -0.12)),
                "frame.label_offset",
            )
            label_position = origin + offset
            ax.text(
                label_position[0],
                label_position[1],
                label_position[2],
                f"{{{name}}}",
                fontsize=float(frame.get("label_fontsize", 9)),
                fontweight="bold",
                color=frame.get("label_color", "#222222"),
                alpha=alpha,
            )

    def _dibujar_punto_3d(self, ax, point):
        """
        Dibuja un punto geométrico 3D.
        """

        position = self._vector_3d(point["position"], "point.position")
        name = point.get("name", "P")
        color = point.get("color", "#7B2CBF")
        alpha = float(point.get("alpha", 1.0))

        ax.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            s=float(point.get("size", 70)),
            color=color,
            edgecolor=point.get("edgecolor", "#222222"),
            linewidth=float(point.get("linewidth", 0.8)),
            alpha=alpha,
            depthshade=False,
        )

        if name:
            offset = self._vector_3d(
                point.get("label_offset", (0.12, 0.12, 0.12)),
                "point.label_offset",
            )
            label_position = position + offset
            ax.text(
                label_position[0],
                label_position[1],
                label_position[2],
                name,
                fontsize=float(point.get("fontsize", 10)),
                fontweight=point.get("fontweight", "bold"),
                color=color,
                alpha=alpha,
            )

    def _dibujar_vector_3d(self, ax, vector):
        """
        Dibuja un vector 3D desde un punto de anclaje.
        """

        origin = self._vector_3d(
            vector.get("origin", (0.0, 0.0, 0.0)),
            "vector.origin",
        )
        value = self._vector_3d(vector["value"], "vector.value")
        name = vector.get("name", "v")
        color = vector.get("color", "#E07A1F")
        alpha = float(vector.get("alpha", 1.0))
        linewidth = float(vector.get("linewidth", 2.8))

        if np.linalg.norm(value) > 1e-12:
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                value[0],
                value[1],
                value[2],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                arrow_length_ratio=float(vector.get("arrow_length_ratio", 0.10)),
                normalize=False,
            )

        if vector.get("show_origin", True):
            ax.scatter(
                [origin[0]],
                [origin[1]],
                [origin[2]],
                s=float(vector.get("origin_size", 16)),
                color=color,
                alpha=alpha,
                depthshade=False,
            )

        if name:
            end = origin + value
            offset = self._vector_3d(
                vector.get("label_offset", (0.12, 0.10, 0.10)),
                "vector.label_offset",
            )
            label_position = end + offset
            ax.text(
                label_position[0],
                label_position[1],
                label_position[2],
                name,
                fontsize=float(vector.get("fontsize", 10)),
                fontweight=vector.get("fontweight", "bold"),
                color=color,
                alpha=alpha,
            )

    def _dibujar_segmento_3d(self, ax, segment):
        """
        Dibuja un segmento auxiliar 3D.
        """

        start = self._vector_3d(segment["start"], "segment.start")
        end = self._vector_3d(segment["end"], "segment.end")

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            linestyle=segment.get("linestyle", "--"),
            linewidth=float(segment.get("linewidth", 1.4)),
            color=segment.get("color", "#777777"),
            alpha=float(segment.get("alpha", 0.7)),
        )

    def _dibujar_polilinea_3d(self, ax, polyline):
        """
        Dibuja una polilínea 3D, útil para trayectorias, arcos y contornos.
        """

        points = np.asarray(polyline["points"], dtype=float)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("polyline.points debe tener forma (N, 3).")

        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            linestyle=polyline.get("linestyle", "-"),
            linewidth=float(polyline.get("linewidth", 1.4)),
            color=polyline.get("color", "#777777"),
            alpha=float(polyline.get("alpha", 0.75)),
        )

    def _dibujar_malla_3d(self, ax, mesh):
        """
        Dibuja una malla poligonal 3D a partir de vértices y caras.

        `faces` es una lista de listas de índices sobre `vertices`.
        """

        vertices = np.asarray(mesh["vertices"], dtype=float)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("mesh.vertices debe tener forma (N, 3).")

        faces = list(mesh.get("faces", []))

        if not faces:
            raise ValueError("mesh.faces debe contener al menos una cara.")

        polygons = [
            vertices[np.asarray(face, dtype=int)]
            for face in faces
        ]

        collection = Poly3DCollection(
            polygons,
            facecolors=mesh.get("facecolor", "#9CC7E8"),
            edgecolors=mesh.get("edgecolor", "#315A7D"),
            linewidths=float(mesh.get("linewidth", 1.0)),
            alpha=float(mesh.get("alpha", 0.30)),
        )
        ax.add_collection3d(collection)

    def _dibujar_texto_3d(self, ax, text_item):
        """
        Dibuja una anotación breve en coordenadas 3D.
        """

        position = self._vector_3d(text_item["position"], "text.position")
        text = str(text_item.get("text", ""))

        if not text:
            return

        ax.text(
            position[0],
            position[1],
            position[2],
            text,
            fontsize=float(text_item.get("fontsize", 9)),
            fontweight=text_item.get("fontweight", "normal"),
            color=text_item.get("color", "#222222"),
            alpha=float(text_item.get("alpha", 1.0)),
        )

    @staticmethod
    def _dibujar_mensaje_3d(ax, state):
        """
        Añade el mensaje pedagógico en la zona inferior de una escena 3D.
        """

        message = state.get("message", "")

        if not message:
            return

        message = textwrap.fill(str(message), width=74)

        ax.text2D(
            0.50,
            0.02,
            message,
            transform=ax.transAxes,
            fontsize=9.3,
            ha="center",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.45",
                "fc": "white",
                "ec": "#777777",
                "alpha": 0.96,
            },
        )

    def _dibujar_estado_3d(self, scene_ax, info_ax, state, limits, view):
        """
        Dibuja por completo un estado de una animación geométrica 3D.
        """

        current_view = state.get("view", view)
        self._configurar_escena_3d(scene_ax, limits, current_view)

        for mesh in state.get("meshes3d", []):
            self._dibujar_malla_3d(scene_ax, mesh)

        for polyline in state.get("polylines3d", []):
            self._dibujar_polilinea_3d(scene_ax, polyline)

        for segment in state.get("segments3d", []):
            self._dibujar_segmento_3d(scene_ax, segment)

        for frame in state.get("frames3d", []):
            self._dibujar_frame_3d(scene_ax, frame)

        for point in state.get("points3d", []):
            self._dibujar_punto_3d(scene_ax, point)

        for vector in state.get("vectors3d", []):
            self._dibujar_vector_3d(scene_ax, vector)

        for text_item in state.get("texts3d", []):
            self._dibujar_texto_3d(scene_ax, text_item)

        self._dibujar_mensaje_3d(scene_ax, state)
        self._dibujar_info(info_ax, state)

        if state.get("legend"):
            legend_elements = [
                self._crear_elemento_leyenda(item)
                for item in state["legend"]
                if item.get("label")
            ]

            if legend_elements:
                scene_ax.legend(
                    handles=legend_elements,
                    loc=state.get("legend_loc", "upper left"),
                    fontsize=float(state.get("legend_fontsize", 8.5)),
                    framealpha=0.95,
                    ncol=int(state.get("legend_ncol", 1)),
                )

    def animate_3d_states(
        self,
        states,
        title,
        limits=(-4.0, 4.0, -4.0, 4.0, -3.0, 4.0),
        view=(24.0, -58.0),
        final_image_path=None,
        video_path=None,
        repeat=False,
        fps=None,
        dpi=125,
        show=True,
    ):
        """
        Anima una secuencia genérica de estados geométricos 3D.

        Cada estado puede contener:
        - frames3d,
        - points3d,
        - vectors3d,
        - segments3d,
        - polylines3d,
        - meshes3d,
        - texts3d,
        - legend,
        - message,
        - info_lines,
        - phase.

        Los cálculos geométricos pertenecen a los scripts del temario; este
        método se limita a representar y exportar los estados recibidos.
        """

        states = list(states)

        if not states:
            raise ValueError("La animación 3D necesita al menos un estado.")

        fig, scene_ax, info_ax = self._preparar_figura_3d(title)

        if final_image_path is not None:
            self._dibujar_estado_3d(
                scene_ax,
                info_ax,
                states[-1],
                limits,
                view,
            )
            final_image_path = Path(final_image_path).expanduser().resolve()
            self._crear_directorio_salida(final_image_path)
            fig.savefig(
                final_image_path,
                dpi=dpi,
                bbox_inches="tight",
            )
            print(f"\nImagen final guardada en:\n  {final_image_path}")

        def actualizar(frame_index):
            self._dibujar_estado_3d(
                scene_ax,
                info_ax,
                states[frame_index],
                limits,
                view,
            )
            return []

        self.animation = FuncAnimation(
            fig,
            actualizar,
            frames=len(states),
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        if fps is None:
            fps = max(1, int(round(1000.0 / self.interval)))

        if video_path is not None:
            self._guardar_video(
                self.animation,
                video_path=video_path,
                fps=fps,
                dpi=dpi,
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

        return self.animation
