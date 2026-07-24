from collections import deque
from pathlib import Path
import sys

import networkx as nx


# Permite importar módulos desde la carpeta Grafos/.
CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_anim import GraphAnimator


SOURCE_NODE = "S"
SINK_NODE = "T"


def obtener_aristas_flujo_maximo():
    """
    Devuelve las aristas de la red en un orden determinista.

    El ejemplo se ha diseñado para que Edmonds-Karp necesite utilizar
    una arista residual inversa. De esta manera, la animación muestra
    cómo puede corregirse una decisión tomada en un aumento anterior.
    """

    return [
        ("S", "A", 4),
        ("S", "B", 3),
        ("S", "E", 2),
        ("A", "C", 2),
        ("A", "D", 3),
        ("B", "C", 2),
        ("E", "F", 2),
        ("C", "T", 2),
        ("D", "T", 3),
        ("F", "T", 2),
    ]


def crear_red_flujo():
    """
    Crea una red dirigida con capacidades enteras.

    Fuente:
        S

    Sumidero:
        T

    El flujo máximo esperado es 7.
    """

    grafo = nx.DiGraph()
    aristas = obtener_aristas_flujo_maximo()

    for origen, destino, capacidad in aristas:
        grafo.add_edge(
            origen,
            destino,
            capacity=capacidad,
        )

    grafo.graph["edge_order"] = list(aristas)
    grafo.graph["source"] = SOURCE_NODE
    grafo.graph["sink"] = SINK_NODE

    return grafo


def obtener_posiciones():
    """
    Define posiciones manuales por capas para representar la red.
    """

    return {
        "S": (-6.0, 0.0),
        "A": (-3.2, 2.7),
        "B": (-3.2, 0.0),
        "E": (-3.2, -2.7),
        "D": (0.0, 3.1),
        "C": (0.0, 1.0),
        "F": (0.0, -2.7),
        "T": (4.2, 0.2),
    }


def validar_red_flujo(
    grafo,
    fuente,
    sumidero,
):
    """
    Comprueba las condiciones utilizadas por este ejemplo.

    Para mantener una representación residual inequívoca, el ejemplo no
    admite aristas antiparalelas originales. Las aristas inversas que
    aparecen durante el algoritmo pertenecen a la red residual.
    """

    if not grafo.is_directed():
        raise ValueError(
            "La red de flujo debe ser un grafo dirigido."
        )

    if grafo.number_of_nodes() == 0:
        raise ValueError("La red de flujo no puede estar vacía.")

    if fuente not in grafo:
        raise ValueError(f"La fuente {fuente!r} no existe.")

    if sumidero not in grafo:
        raise ValueError(f"El sumidero {sumidero!r} no existe.")

    if fuente == sumidero:
        raise ValueError(
            "La fuente y el sumidero deben ser distintos."
        )

    for origen, destino, datos in grafo.edges(data=True):
        capacidad = datos.get("capacity")

        if capacidad is None:
            raise ValueError(
                f"La arista {origen}→{destino} no tiene capacidad."
            )

        if capacidad <= 0:
            raise ValueError(
                "Todas las capacidades deben ser positivas."
            )

        if origen == destino:
            raise ValueError(
                "El ejemplo no admite bucles propios."
            )

        if grafo.has_edge(destino, origen):
            raise ValueError(
                "El ejemplo no admite aristas antiparalelas "
                "originales."
            )

    if grafo.in_degree(fuente) != 0:
        raise ValueError(
            "La fuente del ejemplo no debe tener aristas entrantes."
        )

    if grafo.out_degree(sumidero) != 0:
        raise ValueError(
            "El sumidero del ejemplo no debe tener aristas salientes."
        )


def obtener_orden_aristas(grafo):
    """
    Recupera el orden explícito de las aristas originales.
    """

    return list(
        grafo.graph.get(
            "edge_order",
            [
                (
                    origen,
                    destino,
                    datos["capacity"],
                )
                for origen, destino, datos
                in grafo.edges(data=True)
            ],
        )
    )


def inicializar_flujo(grafo):
    """
    Inicializa a cero el flujo de todas las aristas originales.
    """

    return {
        (origen, destino): 0
        for origen, destino in grafo.edges()
    }


def calcular_valor_flujo(
    grafo,
    flujo,
    fuente,
):
    """
    Calcula el flujo neto que sale de la fuente.
    """

    flujo_saliente = sum(
        flujo[(fuente, destino)]
        for destino in grafo.successors(fuente)
    )

    flujo_entrante = sum(
        flujo[(origen, fuente)]
        for origen in grafo.predecessors(fuente)
    )

    return flujo_saliente - flujo_entrante


def crear_arco_residual(
    origen,
    destino,
    arista_original,
    sentido,
    capacidad_residual,
):
    """
    Crea la representación de una arista residual.

    sentido puede ser:
    - ``directo``: permite aumentar el flujo original;
    - ``inverso``: permite reducir el flujo original.
    """

    return {
        "origin": origen,
        "destination": destino,
        "original_edge": tuple(arista_original),
        "direction": sentido,
        "residual": capacidad_residual,
    }


def obtener_arcos_residuales_desde(
    grafo,
    flujo,
    nodo,
):
    """
    Devuelve los arcos residuales positivos que salen de ``nodo``.

    Se respeta el orden original de las aristas para que las BFS sean
    deterministas y la animación siempre produzca los mismos caminos.
    """

    arcos = []

    for origen, destino, capacidad in obtener_orden_aristas(grafo):
        flujo_actual = flujo[(origen, destino)]

        if origen == nodo:
            residual_directo = capacidad - flujo_actual

            if residual_directo > 0:
                arcos.append(
                    crear_arco_residual(
                        origen=origen,
                        destino=destino,
                        arista_original=(origen, destino),
                        sentido="directo",
                        capacidad_residual=residual_directo,
                    )
                )

        if destino == nodo and flujo_actual > 0:
            arcos.append(
                crear_arco_residual(
                    origen=destino,
                    destino=origen,
                    arista_original=(origen, destino),
                    sentido="inverso",
                    capacidad_residual=flujo_actual,
                )
            )

    return arcos


def obtener_todos_arcos_residuales(
    grafo,
    flujo,
):
    """
    Construye la lista completa de arcos residuales positivos.
    """

    arcos = []

    for nodo in grafo.nodes():
        arcos.extend(
            obtener_arcos_residuales_desde(
                grafo=grafo,
                flujo=flujo,
                nodo=nodo,
            )
        )

    return arcos


def reconstruir_camino_residual(
    padres,
    fuente,
    sumidero,
):
    """
    Reconstruye un camino aumentante a partir de los padres de BFS.
    """

    if sumidero not in padres:
        return []

    camino = []
    actual = sumidero

    while actual != fuente:
        arco = dict(padres[actual])
        camino.append(arco)
        actual = arco["origin"]

    camino.reverse()

    return camino


def formatear_camino_residual(camino):
    """
    Devuelve una representación legible de un camino residual.
    """

    if not camino:
        return "—"

    nodos = [camino[0]["origin"]]
    nodos.extend(
        arco["destination"]
        for arco in camino
    )

    return " → ".join(nodos)


def crear_estado_flujo(
    grafo,
    flujo,
    fuente,
    sumidero,
    message,
    phase,
    augmentation_index,
    queue=None,
    visited=None,
    parents=None,
    current_node=None,
    active_residual_edge=None,
    augmenting_path=None,
    bottleneck=None,
    action=None,
    old_flow=None,
    new_flow=None,
    reachable_nodes=None,
    cut_edges=None,
    cut_capacity=None,
):
    """
    Crea una copia independiente de un estado de Edmonds-Karp.
    """

    return {
        "flows": dict(flujo),
        "residual_arcs": [
            dict(arco)
            for arco in obtener_todos_arcos_residuales(
                grafo,
                flujo,
            )
        ],
        "source": fuente,
        "sink": sumidero,
        "flow_value": calcular_valor_flujo(
            grafo,
            flujo,
            fuente,
        ),
        "augmentation_index": augmentation_index,
        "queue": list(queue or []),
        "visited": set(visited or set()),
        "parents": {
            nodo: dict(arco)
            for nodo, arco in (parents or {}).items()
        },
        "current_node": current_node,
        "active_residual_edge": (
            dict(active_residual_edge)
            if active_residual_edge is not None
            else None
        ),
        "augmenting_path": [
            dict(arco)
            for arco in (augmenting_path or [])
        ],
        "bottleneck": bottleneck,
        "action": action,
        "old_flow": old_flow,
        "new_flow": new_flow,
        "reachable_nodes": set(reachable_nodes or set()),
        "cut_edges": list(cut_edges or []),
        "cut_capacity": cut_capacity,
        "phase": phase,
        "message": message,
    }


def buscar_camino_aumentante_bfs(
    grafo,
    flujo,
    fuente,
    sumidero,
    states,
    augmentation_index,
):
    """
    Ejecuta una BFS en la red residual y registra sus descubrimientos.

    Devuelve:
    - el camino aumentante, o una lista vacía;
    - el conjunto de vértices alcanzables;
    - el diccionario de padres de la BFS.
    """

    cola = deque([fuente])
    visitados = {fuente}
    padres = {}

    states.append(
        crear_estado_flujo(
            grafo=grafo,
            flujo=flujo,
            fuente=fuente,
            sumidero=sumidero,
            queue=cola,
            visited=visitados,
            parents=padres,
            augmentation_index=augmentation_index,
            message=(
                "Comienza una BFS en la red residual para buscar "
                "un nuevo camino aumentante."
            ),
            phase="bfs",
            action="bfs_start",
        )
    )

    while cola and sumidero not in visitados:
        actual = cola.popleft()

        for arco in obtener_arcos_residuales_desde(
            grafo=grafo,
            flujo=flujo,
            nodo=actual,
        ):
            vecino = arco["destination"]

            if vecino in visitados:
                continue

            visitados.add(vecino)
            padres[vecino] = dict(arco)
            cola.append(vecino)

            direction_text = (
                "directa"
                if arco["direction"] == "directo"
                else "inversa"
            )

            states.append(
                crear_estado_flujo(
                    grafo=grafo,
                    flujo=flujo,
                    fuente=fuente,
                    sumidero=sumidero,
                    queue=cola,
                    visited=visitados,
                    parents=padres,
                    current_node=actual,
                    active_residual_edge=arco,
                    augmentation_index=augmentation_index,
                    message=(
                        f"BFS descubre {vecino} desde {actual} mediante "
                        f"una residual {direction_text} de capacidad "
                        f"{arco['residual']}."
                    ),
                    phase="bfs",
                    action="discover",
                )
            )

            if vecino == sumidero:
                break

    camino = reconstruir_camino_residual(
        padres=padres,
        fuente=fuente,
        sumidero=sumidero,
    )

    return camino, visitados, padres, list(cola)


def actualizar_flujo_con_arco(
    flujo,
    arco,
    incremento,
):
    """
    Aplica un incremento sobre una residual directa o inversa.
    """

    arista_original = tuple(arco["original_edge"])
    flujo_anterior = flujo[arista_original]

    if arco["direction"] == "directo":
        flujo_nuevo = flujo_anterior + incremento
    elif arco["direction"] == "inverso":
        flujo_nuevo = flujo_anterior - incremento
    else:
        raise ValueError(
            f"Sentido residual desconocido: {arco['direction']!r}"
        )

    flujo[arista_original] = flujo_nuevo

    return flujo_anterior, flujo_nuevo


def calcular_capacidad_corte(
    grafo,
    lado_fuente,
):
    """
    Calcula la capacidad y las aristas de un corte s-t.
    """

    lado_fuente = set(lado_fuente)
    aristas_corte = []

    for origen, destino, datos in grafo.edges(data=True):
        if origen in lado_fuente and destino not in lado_fuente:
            aristas_corte.append(
                (
                    origen,
                    destino,
                    datos["capacity"],
                )
            )

    capacidad = sum(
        capacidad_arista
        for _, _, capacidad_arista in aristas_corte
    )

    return capacidad, aristas_corte


def ejecutar_edmonds_karp_por_pasos(
    grafo,
    fuente=SOURCE_NODE,
    sumidero=SINK_NODE,
):
    """
    Ejecuta Edmonds-Karp y registra la animación completa.

    Cada iteración realiza:
    1. BFS en la red residual;
    2. reconstrucción del camino aumentante;
    3. cálculo del cuello de botella;
    4. actualización de aristas directas o inversas.

    Al finalizar, la última BFS proporciona el lado alcanzable del corte
    mínimo.
    """

    validar_red_flujo(
        grafo=grafo,
        fuente=fuente,
        sumidero=sumidero,
    )

    flujo = inicializar_flujo(grafo)
    states = []
    augmentations = []
    augmentation_index = 0

    states.append(
        crear_estado_flujo(
            grafo=grafo,
            flujo=flujo,
            fuente=fuente,
            sumidero=sumidero,
            augmentation_index=augmentation_index,
            message=(
                "Inicialización: todas las aristas tienen flujo cero. "
                "La capacidad residual directa coincide con la capacidad."
            ),
            phase="initial",
            action="initial",
        )
    )

    final_reachable = {fuente}
    final_parents = {}

    while True:
        camino, visitados, padres, cola_final = (
            buscar_camino_aumentante_bfs(
                grafo=grafo,
                flujo=flujo,
                fuente=fuente,
                sumidero=sumidero,
                states=states,
                augmentation_index=augmentation_index + 1,
            )
        )

        final_reachable = set(visitados)
        final_parents = dict(padres)

        if not camino:
            states.append(
                crear_estado_flujo(
                    grafo=grafo,
                    flujo=flujo,
                    fuente=fuente,
                    sumidero=sumidero,
                    queue=cola_final,
                    visited=visitados,
                    parents=padres,
                    augmentation_index=augmentation_index,
                    message=(
                        "La BFS no puede alcanzar T. Ya no existe ningún "
                        "camino aumentante y el flujo actual es máximo."
                    ),
                    phase="no_path",
                    action="no_path",
                )
            )
            break

        augmentation_index += 1
        cuello_botella = min(
            arco["residual"]
            for arco in camino
        )

        states.append(
            crear_estado_flujo(
                grafo=grafo,
                flujo=flujo,
                fuente=fuente,
                sumidero=sumidero,
                queue=cola_final,
                visited=visitados,
                parents=padres,
                augmenting_path=camino,
                bottleneck=cuello_botella,
                augmentation_index=augmentation_index,
                message=(
                    f"Camino aumentante {augmentation_index}: "
                    f"{formatear_camino_residual(camino)}."
                ),
                phase="path_found",
                action="path_found",
            )
        )

        states.append(
            crear_estado_flujo(
                grafo=grafo,
                flujo=flujo,
                fuente=fuente,
                sumidero=sumidero,
                augmenting_path=camino,
                bottleneck=cuello_botella,
                augmentation_index=augmentation_index,
                message=(
                    "El cuello de botella es la menor capacidad residual "
                    f"del camino: Δ = {cuello_botella}."
                ),
                phase="bottleneck",
                action="bottleneck",
            )
        )

        used_reverse_edge = False

        for arco in camino:
            old_flow, new_flow = actualizar_flujo_con_arco(
                flujo=flujo,
                arco=arco,
                incremento=cuello_botella,
            )

            if arco["direction"] == "inverso":
                used_reverse_edge = True
                action = "update_reverse"
                explanation = (
                    "La residual inversa cancela parte del flujo "
                    "enviado previamente."
                )
            else:
                action = "update_direct"
                explanation = (
                    "La residual directa aumenta el flujo de la "
                    "arista original."
                )

            original_origin, original_destination = (
                arco["original_edge"]
            )

            states.append(
                crear_estado_flujo(
                    grafo=grafo,
                    flujo=flujo,
                    fuente=fuente,
                    sumidero=sumidero,
                    active_residual_edge=arco,
                    augmenting_path=camino,
                    bottleneck=cuello_botella,
                    augmentation_index=augmentation_index,
                    old_flow=old_flow,
                    new_flow=new_flow,
                    message=(
                        f"Actualizar {original_origin}→"
                        f"{original_destination}: {old_flow} → "
                        f"{new_flow}. {explanation}"
                    ),
                    phase="update",
                    action=action,
                )
            )

        flow_value = calcular_valor_flujo(
            grafo,
            flujo,
            fuente,
        )

        augmentations.append(
            {
                "index": augmentation_index,
                "path": [dict(arco) for arco in camino],
                "bottleneck": cuello_botella,
                "flow_value": flow_value,
                "used_reverse_edge": used_reverse_edge,
            }
        )

        reverse_text = (
            " Se ha utilizado una residual inversa para redirigir "
            "flujo anterior."
            if used_reverse_edge
            else ""
        )

        states.append(
            crear_estado_flujo(
                grafo=grafo,
                flujo=flujo,
                fuente=fuente,
                sumidero=sumidero,
                augmenting_path=camino,
                bottleneck=cuello_botella,
                augmentation_index=augmentation_index,
                message=(
                    f"Finaliza el aumento {augmentation_index}. "
                    f"Valor total del flujo: {flow_value}."
                    f"{reverse_text}"
                ),
                phase="augmentation_complete",
                action="augmentation_complete",
            )
        )

    cut_capacity, cut_edges = calcular_capacidad_corte(
        grafo=grafo,
        lado_fuente=final_reachable,
    )

    states.append(
        crear_estado_flujo(
            grafo=grafo,
            flujo=flujo,
            fuente=fuente,
            sumidero=sumidero,
            visited=final_reachable,
            parents=final_parents,
            reachable_nodes=final_reachable,
            cut_edges=cut_edges,
            cut_capacity=cut_capacity,
            augmentation_index=augmentation_index,
            message=(
                "Los vértices todavía alcanzables desde S forman el "
                "lado S del corte mínimo. Las aristas que salen de ese "
                "conjunto están saturadas."
            ),
            phase="min_cut",
            action="min_cut",
        )
    )

    flow_value = calcular_valor_flujo(
        grafo,
        flujo,
        fuente,
    )

    states.append(
        crear_estado_flujo(
            grafo=grafo,
            flujo=flujo,
            fuente=fuente,
            sumidero=sumidero,
            reachable_nodes=final_reachable,
            cut_edges=cut_edges,
            cut_capacity=cut_capacity,
            augmentation_index=augmentation_index,
            message=(
                f"Resultado final: flujo máximo = {flow_value} y "
                f"capacidad del corte mínimo = {cut_capacity}."
            ),
            phase="finished",
            action="finished",
        )
    )

    return {
        "states": states,
        "flows": dict(flujo),
        "flow_value": flow_value,
        "augmentations": augmentations,
        "reachable_nodes": set(final_reachable),
        "cut_edges": list(cut_edges),
        "cut_capacity": cut_capacity,
        "source": fuente,
        "sink": sumidero,
    }


def comprobar_flujo_factible(
    grafo,
    flujo,
    fuente,
    sumidero,
):
    """
    Comprueba capacidades y conservación del flujo.
    """

    capacity_constraints = all(
        0 <= flujo[(origen, destino)] <= datos["capacity"]
        for origen, destino, datos in grafo.edges(data=True)
    )

    conservation = True

    for node in grafo.nodes():
        if node in {fuente, sumidero}:
            continue

        incoming = sum(
            flujo[(origen, node)]
            for origen in grafo.predecessors(node)
        )
        outgoing = sum(
            flujo[(node, destino)]
            for destino in grafo.successors(node)
        )

        if incoming != outgoing:
            conservation = False
            break

    source_value = calcular_valor_flujo(
        grafo,
        flujo,
        fuente,
    )

    sink_value = sum(
        flujo[(origen, sumidero)]
        for origen in grafo.predecessors(sumidero)
    ) - sum(
        flujo[(sumidero, destino)]
        for destino in grafo.successors(sumidero)
    )

    return {
        "capacity_constraints": capacity_constraints,
        "flow_conservation": conservation,
        "source_sink_values_match": source_value == sink_value,
    }


def existe_camino_aumentante(
    grafo,
    flujo,
    fuente,
    sumidero,
):
    """
    Comprueba si el sumidero es alcanzable en la red residual.
    """

    cola = deque([fuente])
    visitados = {fuente}

    while cola:
        actual = cola.popleft()

        for arco in obtener_arcos_residuales_desde(
            grafo,
            flujo,
            actual,
        ):
            vecino = arco["destination"]

            if vecino in visitados:
                continue

            if vecino == sumidero:
                return True

            visitados.add(vecino)
            cola.append(vecino)

    return False


def comprobar_con_networkx(
    grafo,
    resultado,
):
    """
    Compara el resultado con el flujo máximo y corte mínimo de NetworkX.
    """

    fuente = resultado["source"]
    sumidero = resultado["sink"]

    networkx_flow_value = nx.maximum_flow_value(
        grafo,
        fuente,
        sumidero,
        capacity="capacity",
    )

    networkx_cut_value, _ = nx.minimum_cut(
        grafo,
        fuente,
        sumidero,
        capacity="capacity",
    )

    feasibility = comprobar_flujo_factible(
        grafo=grafo,
        flujo=resultado["flows"],
        fuente=fuente,
        sumidero=sumidero,
    )

    cut_edges_saturated = all(
        resultado["flows"][(origen, destino)] == capacidad
        for origen, destino, capacidad in resultado["cut_edges"]
    )

    reverse_augmentation_exists = any(
        augmentation["used_reverse_edge"]
        for augmentation in resultado["augmentations"]
    )

    return {
        **feasibility,
        "maximum_flow_matches_networkx": (
            resultado["flow_value"] == networkx_flow_value
        ),
        "minimum_cut_matches_networkx": (
            resultado["cut_capacity"] == networkx_cut_value
        ),
        "max_flow_equals_min_cut": (
            resultado["flow_value"] == resultado["cut_capacity"]
        ),
        "cut_edges_are_saturated": cut_edges_saturated,
        "no_augmenting_path_remains": not existe_camino_aumentante(
            grafo,
            resultado["flows"],
            fuente,
            sumidero,
        ),
        "reverse_residual_was_used": reverse_augmentation_exists,
    }


def imprimir_resultado_flujo_maximo(
    grafo,
    resultado,
    comprobacion,
):
    """
    Muestra por terminal el flujo, los aumentos y el corte mínimo.
    """

    print("\n=== Flujo máximo y corte mínimo ===")

    print("\nAumentos de Edmonds-Karp:")

    for augmentation in resultado["augmentations"]:
        reverse_text = (
            " · usa residual inversa"
            if augmentation["used_reverse_edge"]
            else ""
        )

        print(
            f"  {augmentation['index']}. "
            f"{formatear_camino_residual(augmentation['path'])} "
            f"· Δ={augmentation['bottleneck']} "
            f"· flujo={augmentation['flow_value']}"
            f"{reverse_text}"
        )

    print("\nFlujo final por arista:")

    for origen, destino, capacidad in obtener_orden_aristas(grafo):
        flujo_arista = resultado["flows"][(origen, destino)]
        residual_directo = capacidad - flujo_arista
        residual_inverso = flujo_arista

        print(
            f"  {origen} → {destino}: "
            f"{flujo_arista}/{capacidad} "
            f"· r+={residual_directo} "
            f"· r-={residual_inverso}"
        )

    print(
        "\nValor del flujo máximo: "
        f"{resultado['flow_value']}"
    )

    print(
        "Lado S del corte mínimo: "
        + ", ".join(sorted(resultado["reachable_nodes"]))
    )

    other_side = sorted(
        set(grafo.nodes()) - resultado["reachable_nodes"]
    )

    print(
        "Lado T del corte mínimo: "
        + ", ".join(other_side)
    )

    print("Aristas del corte mínimo:")

    for origen, destino, capacidad in resultado["cut_edges"]:
        print(f"  {origen} → {destino}: {capacidad}")

    print(
        "Capacidad del corte mínimo: "
        f"{resultado['cut_capacity']}"
    )

    print("\nComprobación con NetworkX:")

    for name, value in comprobacion.items():
        print(f"  {name}: {value}")


def main():
    grafo = crear_red_flujo()
    posiciones = obtener_posiciones()

    resultado = ejecutar_edmonds_karp_por_pasos(
        grafo=grafo,
        fuente=SOURCE_NODE,
        sumidero=SINK_NODE,
    )

    comprobacion = comprobar_con_networkx(
        grafo=grafo,
        resultado=resultado,
    )

    imprimir_resultado_flujo_maximo(
        grafo=grafo,
        resultado=resultado,
        comprobacion=comprobacion,
    )

    animador = GraphAnimator(
        figsize=(16, 10),
        interval=700,
    )

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "03_algoritmos"
        / "09_flujo_maximo_corte_minimo.png"
    )

    animacion = animador.animate_max_flow_min_cut(
        graph=grafo,
        pos=posiciones,
        states=resultado["states"],
        source_node=SOURCE_NODE,
        sink_node=SINK_NODE,
        title="Flujo máximo y corte mínimo con Edmonds-Karp",
        final_image_path=output_path,
        repeat=False,
    )

    _ = animacion


if __name__ == "__main__":
    main()
