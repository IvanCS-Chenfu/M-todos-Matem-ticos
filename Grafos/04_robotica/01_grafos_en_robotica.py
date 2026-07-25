from pathlib import Path
import sys

import networkx as nx


CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_visualizer import GraphVisualizer


def crear_grafo_navegacion():
    """
    Crea un mapa topológico.

    Vértices:
        lugares significativos.

    Aristas:
        caminos transitables.

    Pesos:
        distancia aproximada.
    """

    graph = nx.Graph()

    graph.add_nodes_from([
        "Entrada",
        "Pasillo",
        "Taller",
        "Almacén",
        "Carga",
    ])

    graph.add_edges_from([
        ("Entrada", "Pasillo", {"distance": 4, "label": "camino · 4 m"}),
        ("Pasillo", "Taller", {"distance": 6, "label": "camino · 6 m"}),
        ("Pasillo", "Almacén", {"distance": 5, "label": "camino · 5 m"}),
        ("Almacén", "Carga", {"distance": 3, "label": "camino · 3 m"}),
        ("Taller", "Carga", {"distance": 7, "label": "camino · 7 m"}),
    ])

    positions = {
        "Entrada": (-2.5, 0.0),
        "Pasillo": (-0.8, 0.0),
        "Taller": (1.1, 1.25),
        "Almacén": (1.1, -1.25),
        "Carga": (2.8, 0.0),
    }

    return graph, positions


def crear_grafo_tareas():
    """
    Crea un DAG de tareas robóticas.

    Vértices:
        tareas.

    Aristas:
        relaciones de precedencia.
    """

    graph = nx.DiGraph()

    graph.add_nodes_from([
        "Encender",
        "Sensores",
        "Localizar",
        "Planificar",
        "Navegar",
        "Recoger",
    ])

    graph.add_edges_from([
        ("Encender", "Sensores", {"label": "antes de"}),
        ("Sensores", "Localizar", {"label": "habilita"}),
        ("Localizar", "Planificar", {"label": "antes de"}),
        ("Planificar", "Navegar", {"label": "genera ruta"}),
        ("Navegar", "Recoger", {"label": "permite"}),
    ])

    positions = {
        "Encender": (-3.0, 0.0),
        "Sensores": (-1.8, 1.15),
        "Localizar": (-0.6, 0.0),
        "Planificar": (0.7, 1.15),
        "Navegar": (1.9, 0.0),
        "Recoger": (3.1, 1.15),
    }

    return graph, positions


def crear_grafo_poses():
    """
    Crea un pequeño grafo de poses.

    Vértices:
        poses del robot.

    Aristas:
        mediciones relativas.

    La arista x4→x0 representa un cierre de bucle.
    """

    graph = nx.DiGraph()

    graph.add_nodes_from([
        "x0",
        "x1",
        "x2",
        "x3",
        "x4",
    ])

    graph.add_edges_from([
        ("x0", "x1", {"label": "odometría"}),
        ("x1", "x2", {"label": "odometría"}),
        ("x2", "x3", {"label": "odometría"}),
        ("x3", "x4", {"label": "odometría"}),
        ("x4", "x0", {"label": "cierre de bucle"}),
    ])

    positions = {
        "x0": (-2.1, -1.1),
        "x1": (-2.0, 1.1),
        "x2": (0.0, 1.65),
        "x3": (2.0, 1.1),
        "x4": (2.1, -1.1),
    }

    return graph, positions


def crear_grafo_semantico():
    """
    Crea un grafo de escena semántico.

    Vértices:
        objetos, robot y lugares.

    Aristas:
        relaciones espaciales o semánticas.
    """

    graph = nx.DiGraph()

    graph.add_nodes_from([
        "Robot",
        "Taza",
        "Mesa",
        "Cocina",
        "Puerta",
    ])

    graph.add_edges_from([
        ("Robot", "Taza", {"label": "observa"}),
        ("Taza", "Mesa", {"label": "sobre"}),
        ("Mesa", "Cocina", {"label": "dentro de"}),
        ("Robot", "Cocina", {"label": "está en"}),
        ("Puerta", "Cocina", {"label": "da acceso"}),
    ])

    positions = {
        "Robot": (-2.2, 0.0),
        "Taza": (-0.4, 1.35),
        "Mesa": (1.3, 1.15),
        "Cocina": (1.3, -1.2),
        "Puerta": (-0.5, -1.35),
    }

    return graph, positions


def crear_grafo_multi_robot():
    """
    Crea una red de comunicación multi-robot.

    Vértices:
        robots y estación base.

    Aristas:
        enlaces de comunicación.

    El atributo quality representa la calidad estimada del enlace.
    """

    graph = nx.Graph()

    graph.add_nodes_from([
        "R1",
        "R2",
        "R3",
        "R4",
        "Base",
    ])

    graph.add_edges_from([
        ("R1", "R2", {"quality": 0.9, "label": "radio · 0,9"}),
        ("R2", "R3", {"quality": 0.7, "label": "radio · 0,7"}),
        ("R2", "R4", {"quality": 0.8, "label": "relé · 0,8"}),
        ("R4", "Base", {"quality": 0.95, "label": "Wi-Fi · 0,95"}),
        ("R3", "Base", {"quality": 0.6, "label": "radio · 0,6"}),
    ])

    positions = {
        "R1": (-2.6, 0.9),
        "R2": (-0.9, 0.9),
        "R3": (0.8, 1.4),
        "R4": (0.5, -0.9),
        "Base": (2.6, -0.1),
    }

    return graph, positions


def crear_grafo_planificacion():
    """
    Crea un grafo de configuraciones.

    Vértices:
        configuraciones válidas del robot.

    Aristas:
        movimientos locales sin colisión.

    El camino destacado conecta la configuración inicial con el objetivo.
    """

    graph = nx.Graph()

    graph.add_nodes_from([
        "q0",
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
    ])

    graph.add_edges_from([
        ("q0", "q1", {"cost": 1.0, "label": "giro"}),
        ("q1", "q2", {"cost": 1.2, "label": "extender"}),
        ("q2", "q5", {"cost": 1.0, "label": "aproximar"}),
        ("q0", "q3", {"cost": 1.4, "label": "alternativa"}),
        ("q3", "q4", {"cost": 1.1, "label": "reorientar"}),
        ("q4", "q5", {"cost": 1.3, "label": "aproximar"}),
        ("q2", "q4", {"cost": 0.9, "label": "transición"}),
    ])

    positions = {
        "q0": (-2.7, 0.0),
        "q1": (-1.2, 1.25),
        "q2": (0.5, 1.25),
        "q3": (-1.2, -1.25),
        "q4": (0.5, -1.25),
        "q5": (2.5, 0.0),
    }

    return graph, positions


def analizar_grafo(graph):
    """
    Calcula propiedades estructurales sencillas del ejemplo.
    """

    if graph.is_directed():
        connected = nx.is_weakly_connected(graph)
        acyclic = nx.is_directed_acyclic_graph(graph)
        cycles = 0 if acyclic else len(list(nx.simple_cycles(graph)))
    else:
        connected = nx.is_connected(graph)
        cycles = len(nx.cycle_basis(graph))
        acyclic = cycles == 0

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed": graph.is_directed(),
        "connected": connected,
        "acyclic": acyclic,
        "cycles": cycles,
    }


def validar_ejemplo(example):
    """
    Comprueba que posiciones y etiquetas cubran el grafo completo.
    """

    graph = example["graph"]
    positions = example["pos"]
    node_labels = example["node_labels"]

    if set(positions) != set(graph.nodes()):
        raise ValueError(
            f"Las posiciones de {example['title']} "
            "no cubren todos los vértices."
        )

    if set(node_labels) != set(graph.nodes()):
        raise ValueError(
            f"Las etiquetas de {example['title']} "
            "no cubren todos los vértices."
        )

    for u, v in graph.edges():
        if "label" not in graph.edges[u, v]:
            raise ValueError(
                f"La arista {u}→{v} de {example['title']} "
                "no tiene una etiqueta semántica."
            )


def crear_ejemplos_robotica():
    """
    Crea los seis paneles comparativos del apartado.
    """

    navigation, navigation_pos = crear_grafo_navegacion()
    tasks, tasks_pos = crear_grafo_tareas()
    poses, poses_pos = crear_grafo_poses()
    semantic, semantic_pos = crear_grafo_semantico()
    multi_robot, multi_robot_pos = crear_grafo_multi_robot()
    planning, planning_pos = crear_grafo_planificacion()

    examples = [
        {
            "title": "1. Navegación",
            "subtitle": "Grafo no dirigido y ponderado",
            "graph": navigation,
            "pos": navigation_pos,
            "node_labels": {node: node for node in navigation.nodes()},
            "edge_labels": nx.get_edge_attributes(navigation, "label"),
            "highlight_nodes": ["Entrada", "Carga"],
            "description": (
                "Vértices = lugares · Aristas = caminos\n"
                "Uso: Dijkstra o A* para calcular una ruta"
            ),
            "edge_font_size": 6.8,
            "node_font_size": 8.2,
            "node_size": 1900,
            "spread_factor": 1.18,
            "margin": 0.20,
            "edge_label_offsets": {
                ("Entrada", "Pasillo"): (0.0, 0.20),
                ("Pasillo", "Taller"): (-0.10, 0.18),
                ("Pasillo", "Almacén"): (-0.10, -0.18),
                ("Almacén", "Carga"): (0.08, -0.18),
                ("Taller", "Carga"): (0.08, 0.18),
            },
        },
        {
            "title": "2. Planificación de tareas",
            "subtitle": "DAG de precedencias",
            "graph": tasks,
            "pos": tasks_pos,
            "node_labels": {node: node for node in tasks.nodes()},
            "edge_labels": nx.get_edge_attributes(tasks, "label"),
            "highlight_nodes": ["Encender", "Recoger"],
            "description": (
                "Vértices = tareas · Aristas = dependencias\n"
                "Uso: ordenamiento topológico"
            ),
            "edge_font_size": 6.7,
            "node_font_size": 8.2,
            "node_size": 1900,
            "spread_factor": 1.26,
            "margin": 0.16,
            "edge_label_offsets": {
                ("Encender", "Sensores"): (-0.10, 0.18),
                ("Sensores", "Localizar"): (0.10, -0.18),
                ("Localizar", "Planificar"): (-0.10, 0.18),
                ("Planificar", "Navegar"): (0.10, -0.18),
                ("Navegar", "Recoger"): (-0.10, 0.18),
            },
        },
        {
            "title": "3. Grafo de poses",
            "subtitle": "Mediciones relativas y cierre de bucle",
            "graph": poses,
            "pos": poses_pos,
            "node_labels": {
                "x0": "x0\ninicio",
                "x1": "x1",
                "x2": "x2",
                "x3": "x3",
                "x4": "x4",
            },
            "edge_labels": nx.get_edge_attributes(poses, "label"),
            "edge_label_offsets": {
                ("x0", "x1"): (-0.28, 0.0),
                ("x1", "x2"): (0.0, 0.20),
                ("x2", "x3"): (0.0, 0.20),
                ("x3", "x4"): (0.28, 0.0),
                ("x4", "x0"): (0.0, -0.30),
            },
            "edge_rads": {
                ("x4", "x0"): -0.20,
            },
            "highlight_nodes": ["x0"],
            "highlight_edges": [("x4", "x0")],
            "description": (
                "Vértices = poses · Aristas = mediciones\n"
                "Uso: optimización de Graph SLAM"
            ),
            "edge_font_size": 6.7,
            "node_font_size": 8.2,
            "node_size": 1900,
            "spread_factor": 1.16,
            "margin": 0.20,
        },
        {
            "title": "4. Percepción semántica",
            "subtitle": "Grafo dirigido de escena",
            "graph": semantic,
            "pos": semantic_pos,
            "node_labels": {node: node for node in semantic.nodes()},
            "edge_labels": nx.get_edge_attributes(semantic, "label"),
            "edge_label_offsets": {
                ("Robot", "Cocina"): (0.0, -0.22),
                ("Mesa", "Cocina"): (0.28, 0.0),
            },
            "highlight_nodes": ["Robot"],
            "description": (
                "Vértices = entidades · Aristas = relaciones\n"
                "Uso: percepción, memoria y razonamiento"
            ),
            "edge_font_size": 6.8,
            "node_font_size": 8.2,
            "node_size": 1900,
            "spread_factor": 1.17,
            "margin": 0.20,
        },
        {
            "title": "5. Coordinación multi-robot",
            "subtitle": "Red de comunicación",
            "graph": multi_robot,
            "pos": multi_robot_pos,
            "node_labels": {
                "R1": "R1",
                "R2": "R2\npuente",
                "R3": "R3",
                "R4": "R4\nrelé",
                "Base": "Base",
            },
            "edge_labels": nx.get_edge_attributes(multi_robot, "label"),
            "edge_label_offsets": {
                ("R2", "R4"): (-0.18, 0.0),
                ("R3", "Base"): (0.15, 0.18),
            },
            "highlight_nodes": ["R2", "R4", "Base"],
            "description": (
                "Vértices = robots · Aristas = comunicación\n"
                "Uso: conectividad, centralidad y asignación"
            ),
            "edge_font_size": 6.7,
            "node_font_size": 8.2,
            "node_size": 1900,
            "spread_factor": 1.18,
            "margin": 0.20,
        },
        {
            "title": "6. Planificación de movimiento",
            "subtitle": "Grafo de configuraciones válidas",
            "graph": planning,
            "pos": planning_pos,
            "node_labels": {
                "q0": "q0\ninicio",
                "q1": "q1",
                "q2": "q2",
                "q3": "q3",
                "q4": "q4",
                "q5": "q5\nobjetivo",
            },
            "edge_labels": nx.get_edge_attributes(planning, "label"),
            "edge_label_offsets": {
                ("q2", "q4"): (0.25, 0.0),
            },
            "highlight_nodes": ["q0", "q5"],
            "highlight_edges": [
                ("q0", "q1"),
                ("q1", "q2"),
                ("q2", "q5"),
            ],
            "description": (
                "Vértices = configuraciones · Aristas = movimientos\n"
                "Uso: búsqueda de un camino sin colisiones"
            ),
            "edge_font_size": 6.7,
            "node_font_size": 8.2,
            "node_size": 1900,
            "spread_factor": 1.18,
            "margin": 0.20,
        },
    ]

    for example in examples:
        validar_ejemplo(example)

    return examples


def imprimir_resumen(examples):
    """
    Muestra las propiedades principales de cada grafo.
    """

    print("\n=== Grafos aplicados a robótica ===")

    for example in examples:
        analysis = analizar_grafo(example["graph"])

        graph_type = (
            "dirigido"
            if analysis["directed"]
            else "no dirigido"
        )

        cycle_text = (
            "acíclico"
            if analysis["acyclic"]
            else f"{analysis['cycles']} ciclo(s)"
        )

        print(
            f"\n{example['title']}\n"
            f"  Tipo: {graph_type}\n"
            f"  Vértices: {analysis['nodes']}\n"
            f"  Aristas: {analysis['edges']}\n"
            f"  Conexo: {analysis['connected']}\n"
            f"  Estructura: {cycle_text}"
        )


def main():
    examples = crear_ejemplos_robotica()
    imprimir_resumen(examples)

    visualizer = GraphVisualizer(
        figsize=(26.5, 14.2),
    )

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "04_robotica"
        / "01_grafos_en_robotica.png"
    )

    visualizer.show_robotics_graph_collection(
        graph_examples=examples,
        title="Por qué los grafos son importantes en robótica",
        subtitle=(
            "La misma estructura matemática representa lugares, tareas, "
            "poses, objetos, robots y configuraciones"
        ),
        save_path=output_path,
        rows=2,
        cols=3,
    )


if __name__ == "__main__":
    main()