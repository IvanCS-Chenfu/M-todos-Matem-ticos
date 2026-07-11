from pathlib import Path
import sys

import networkx as nx
import numpy as np


# Permite importar módulos desde la carpeta Grafos/
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent
sys.path.append(str(GRAFOS_DIR))

from utils.graph_visualizer import GraphVisualizer


def obtener_nodos():
    """
    Devuelve los nodos del grafo en un orden fijo.

    Mantener un orden fijo es importante para que las matrices sean
    fáciles de leer.
    """

    return ["A", "B", "C", "D", "E"]


def obtener_aristas_ponderadas():
    """
    Devuelve las aristas ponderadas del grafo.

    Cada tupla tiene la forma:

        (origen, destino, peso)

    El grafo será no dirigido.
    """

    return [
        ("A", "B", 2),
        ("A", "C", 4),
        ("B", "D", 3),
        ("C", "D", 1),
        ("C", "E", 5),
        ("D", "E", 2),
    ]


def obtener_posiciones():
    """
    Define posiciones manuales para que todos los grafos se vean iguales.
    """

    return {
        "A": (0.0, 2.0),
        "B": (-2.0, 0.7),
        "C": (2.0, 0.7),
        "D": (0.0, -0.8),
        "E": (2.0, -2.0),
    }


def crear_grafo_desde_lista_de_aristas():
    """
    Crea el grafo usando una lista de aristas ponderadas.

    Esta es una de las formas más directas de construir un grafo.
    """

    grafo = nx.Graph()
    grafo.add_nodes_from(obtener_nodos())
    grafo.add_weighted_edges_from(obtener_aristas_ponderadas())

    return grafo


def crear_grafo_desde_lista_de_adyacencia():
    """
    Crea el mismo grafo usando una lista/diccionario de adyacencia.

    Cada nodo guarda sus vecinos y el peso de la conexión.
    """

    lista_adyacencia = {
        "A": {"B": 2, "C": 4},
        "B": {"A": 2, "D": 3},
        "C": {"A": 4, "D": 1, "E": 5},
        "D": {"B": 3, "C": 1, "E": 2},
        "E": {"C": 5, "D": 2},
    }

    grafo = nx.Graph()
    grafo.add_nodes_from(obtener_nodos())

    for nodo, vecinos in lista_adyacencia.items():
        for vecino, peso in vecinos.items():
            grafo.add_edge(nodo, vecino, weight=peso)

    return grafo


def crear_matriz_ponderada():
    """
    Crea la matriz ponderada del grafo.

    Un 0 indica que no hay arista directa entre dos nodos.
    Un valor distinto de 0 indica el peso de la arista.
    """

    nodos = obtener_nodos()
    indice = {nodo: i for i, nodo in enumerate(nodos)}

    matriz = np.zeros((len(nodos), len(nodos)), dtype=int)

    for origen, destino, peso in obtener_aristas_ponderadas():
        i = indice[origen]
        j = indice[destino]

        matriz[i, j] = peso
        matriz[j, i] = peso

    return matriz


def crear_grafo_desde_matriz_ponderada():
    """
    Crea el mismo grafo a partir de una matriz ponderada.
    """

    nodos = obtener_nodos()
    matriz = crear_matriz_ponderada()

    grafo = nx.Graph()
    grafo.add_nodes_from(nodos)

    for i, origen in enumerate(nodos):
        for j, destino in enumerate(nodos):
            peso = matriz[i, j]

            if j > i and peso != 0:
                grafo.add_edge(origen, destino, weight=int(peso))

    return grafo


def crear_matriz_incidencia():
    """
    Crea la matriz de incidencia del grafo.

    En un grafo no dirigido:
    - cada columna representa una arista,
    - cada fila representa un nodo,
    - aparece un 1 si el nodo pertenece a esa arista.
    """

    nodos = obtener_nodos()
    aristas = obtener_aristas_ponderadas()

    indice_nodos = {nodo: i for i, nodo in enumerate(nodos)}

    matriz = np.zeros((len(nodos), len(aristas)), dtype=int)

    for j, (origen, destino, _) in enumerate(aristas):
        matriz[indice_nodos[origen], j] = 1
        matriz[indice_nodos[destino], j] = 1

    return matriz


def crear_grafo_desde_matriz_incidencia():
    """
    Crea el mismo grafo a partir de una matriz de incidencia.

    Para reconstruir los pesos, usamos la lista de aristas original como
    referencia. En un caso real, los pesos podrían almacenarse en una lista
    adicional asociada a cada columna de la matriz de incidencia.
    """

    nodos = obtener_nodos()
    aristas = obtener_aristas_ponderadas()
    matriz_incidencia = crear_matriz_incidencia()

    grafo = nx.Graph()
    grafo.add_nodes_from(nodos)

    for columna in range(matriz_incidencia.shape[1]):
        filas_con_uno = np.where(matriz_incidencia[:, columna] == 1)[0]

        if len(filas_con_uno) != 2:
            continue

        origen = nodos[filas_con_uno[0]]
        destino = nodos[filas_con_uno[1]]
        peso = aristas[columna][2]

        grafo.add_edge(origen, destino, weight=peso)

    return grafo


def formatear_lista_de_aristas():
    """
    Devuelve una versión textual de la lista de aristas.
    """

    lineas = ["Lista de aristas:"]

    for origen, destino, peso in obtener_aristas_ponderadas():
        lineas.append(f"({origen}, {destino}, peso={peso})")

    return "\n".join(lineas)


def formatear_lista_de_adyacencia():
    """
    Devuelve una versión textual de la lista de adyacencia.
    """

    grafo = crear_grafo_desde_lista_de_aristas()

    lineas = ["Lista de adyacencia:"]

    for nodo in obtener_nodos():
        vecinos = []

        for vecino in sorted(grafo.neighbors(nodo)):
            peso = grafo[nodo][vecino]["weight"]
            vecinos.append(f"{vecino}({peso})")

        lineas.append(f"{nodo}: " + ", ".join(vecinos))

    return "\n".join(lineas)


def formatear_matriz(nombre, matriz, columnas):
    """
    Formatea una matriz para mostrarla como texto monoespaciado.
    """

    lineas = [nombre]

    cabecera = "    " + " ".join(f"{columna:>3}" for columna in columnas)
    lineas.append(cabecera)

    for etiqueta_fila, fila in zip(obtener_nodos(), matriz):
        valores = " ".join(f"{valor:>3}" for valor in fila)
        lineas.append(f"{etiqueta_fila:>3} {valores}")

    return "\n".join(lineas)


def formatear_matriz_ponderada():
    """
    Devuelve la matriz ponderada como texto.
    """

    matriz = crear_matriz_ponderada()
    return formatear_matriz(
        "Matriz ponderada:",
        matriz,
        obtener_nodos(),
    )


def formatear_matriz_incidencia():
    """
    Devuelve la matriz de incidencia como texto.
    """

    matriz = crear_matriz_incidencia()

    columnas = [
        f"e{i + 1}"
        for i in range(matriz.shape[1])
    ]

    return formatear_matriz(
        "Matriz de incidencia:",
        matriz,
        columnas,
    )


def crear_ejemplos_de_representaciones():
    """
    Crea el mismo grafo desde varias representaciones distintas.

    Aunque la forma de construirlo cambia, el resultado visual es el mismo.
    """

    return [
        {
            "title": "Desde lista de aristas",
            "graph": crear_grafo_desde_lista_de_aristas(),
            "text": formatear_lista_de_aristas(),
        },
        {
            "title": "Desde lista de adyacencia",
            "graph": crear_grafo_desde_lista_de_adyacencia(),
            "text": formatear_lista_de_adyacencia(),
        },
        {
            "title": "Desde matriz ponderada",
            "graph": crear_grafo_desde_matriz_ponderada(),
            "text": formatear_matriz_ponderada(),
        },
        {
            "title": "Desde matriz de incidencia",
            "graph": crear_grafo_desde_matriz_incidencia(),
            "text": formatear_matriz_incidencia(),
        },
    ]


def comprobar_grafos_equivalentes(grafos):
    """
    Comprueba que todos los grafos generados tienen los mismos nodos,
    las mismas aristas y los mismos pesos.
    """

    grafo_referencia = grafos[0]

    for grafo in grafos[1:]:
        if set(grafo.nodes()) != set(grafo_referencia.nodes()):
            return False

        for origen, destino, datos in grafo_referencia.edges(data=True):
            if not grafo.has_edge(origen, destino):
                return False

            peso_referencia = datos["weight"]
            peso_actual = grafo[origen][destino]["weight"]

            if peso_referencia != peso_actual:
                return False

    return True


def main():
    ejemplos = crear_ejemplos_de_representaciones()
    grafos = [ejemplo["graph"] for ejemplo in ejemplos]

    son_equivalentes = comprobar_grafos_equivalentes(grafos)

    print("\n=== Representación computacional de grafos ===")
    print(f"Número de nodos: {grafos[0].number_of_nodes()}")
    print(f"Número de aristas: {grafos[0].number_of_edges()}")
    print(f"¿Todos los grafos generados son equivalentes?: {son_equivalentes}")

    print("\n--- Lista de aristas ---")
    print(formatear_lista_de_aristas())

    print("\n--- Lista de adyacencia ---")
    print(formatear_lista_de_adyacencia())

    print("\n--- Matriz ponderada ---")
    print(formatear_matriz_ponderada())

    print("\n--- Matriz de incidencia ---")
    print(formatear_matriz_incidencia())

    visualizador = GraphVisualizer(figsize=(16, 9))

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "02_representacion_computacional"
        / "01_representaciones_grafo.png"
    )

    visualizador.show_graph_representations(
        representation_examples=ejemplos,
        pos=obtener_posiciones(),
        title="Representaciones computacionales de un mismo grafo",
        save_path=output_path,
    )


if __name__ == "__main__":
    main()