from pathlib import Path
import sys

import networkx as nx


CURRENT_DIR = Path(__file__).resolve().parent
GRAFOS_DIR = CURRENT_DIR.parent

if str(GRAFOS_DIR) not in sys.path:
    sys.path.insert(0, str(GRAFOS_DIR))

from utils.graph_visualizer import GraphVisualizer


ENTITY_OBJECT = "objeto"
ENTITY_PLACE = "lugar"
ENTITY_AGENT = "agente"
ENTITY_CLASS = "clase"

FACT_DECLARED = "declarado"
FACT_OBSERVED = "observado"
FACT_INFERRED = "inferido"
FACT_ACTION = "accion"


ENTITIES = {
    "Objeto": {
        "label": "Objeto",
        "entity_type": ENTITY_CLASS,
        "description": "Clase general de entidades físicas.",
    },
    "Recipiente": {
        "label": "Recipiente",
        "entity_type": ENTITY_CLASS,
        "description": "Objeto capaz de contener otros elementos.",
    },
    "Taza": {
        "label": "Taza",
        "entity_type": ENTITY_CLASS,
        "description": "Clase concreta de recipiente.",
    },
    "Mueble": {
        "label": "Mueble",
        "entity_type": ENTITY_CLASS,
        "description": "Objeto del entorno que organiza o soporta elementos.",
    },
    "Mesa": {
        "label": "Mesa",
        "entity_type": ENTITY_CLASS,
        "description": "Clase de mueble con una superficie de apoyo.",
    },
    "Habitacion": {
        "label": "Habitación",
        "entity_type": ENTITY_CLASS,
        "description": "Lugar delimitado dentro de una vivienda.",
    },
    "Agente": {
        "label": "Agente",
        "entity_type": ENTITY_CLASS,
        "description": "Entidad capaz de percibir, decidir o actuar.",
    },
    "Robot": {
        "label": "Robot",
        "entity_type": ENTITY_CLASS,
        "description": "Agente artificial con capacidades físicas.",
    },
    "Persona": {
        "label": "Persona",
        "entity_type": ENTITY_CLASS,
        "description": "Agente humano del entorno.",
    },
    "robot_1": {
        "label": "robot_1\nRobot doméstico",
        "entity_type": ENTITY_AGENT,
        "persistent": True,
        "capabilities": ["observar", "agarrar", "transportar"],
    },
    "persona_1": {
        "label": "persona_1\nPersona",
        "entity_type": ENTITY_AGENT,
        "persistent": True,
    },
    "taza_1": {
        "label": "taza_1\nTaza roja",
        "entity_type": ENTITY_OBJECT,
        "persistent": True,
        "color": "rojo",
        "material": "ceramica",
    },
    "mesa_1": {
        "label": "mesa_1\nMesa cocina",
        "entity_type": ENTITY_OBJECT,
        "persistent": True,
    },
    "cocina": {
        "label": "cocina",
        "entity_type": ENTITY_PLACE,
        "persistent": True,
    },
    "salon": {
        "label": "salón",
        "entity_type": ENTITY_PLACE,
        "persistent": True,
    },
}


CLASS_RELATIONS = (
    ("Taza", "Recipiente", "subclase_de"),
    ("Recipiente", "Objeto", "subclase_de"),
    ("Mesa", "Mueble", "subclase_de"),
    ("Mueble", "Objeto", "subclase_de"),
    ("Robot", "Agente", "subclase_de"),
    ("Persona", "Agente", "subclase_de"),
)


INSTANCE_RELATIONS = (
    ("taza_1", "Taza", "es_un"),
    ("mesa_1", "Mesa", "es_un"),
    ("robot_1", "Robot", "es_un"),
    ("persona_1", "Persona", "es_un"),
    ("cocina", "Habitacion", "es_un"),
    ("salon", "Habitacion", "es_un"),
)


SCENE_POSITIONS = {
    "robot_1": (-4.0, -1.55),
    "taza_1": (-1.35, 0.25),
    "mesa_1": (1.55, 0.20),
    "cocina": (4.30, 1.45),
    "persona_1": (4.15, -1.55),
    "salon": (6.60, -0.15),
}


ONTOLOGY_POSITIONS = {
    "Objeto": (0.0, 3.55),
    "Recipiente": (-2.25, 2.30),
    "Taza": (-2.25, 0.95),
    "Mueble": (2.25, 2.30),
    "Mesa": (2.25, 0.95),
    "Agente": (6.45, 3.55),
    "Robot": (5.30, 2.10),
    "Persona": (7.60, 2.10),
    "Habitacion": (10.50, 3.55),
    "taza_1": (-2.25, -0.95),
    "mesa_1": (2.25, -0.95),
    "robot_1": (5.30, -0.95),
    "persona_1": (7.60, -0.95),
    "cocina": (9.65, -0.95),
    "salon": (11.45, -0.95),
}


SCENE_EDGE_LABEL_OFFSETS = {
    ("robot_1", "persona_1", "asiste_a"): (0.0, 0.56),
    ("persona_1", "taza_1", "quiere"): (0.05, 0.34),
    ("taza_1", "mesa_1", "encima_de"): (0.0, 0.26),
    ("mesa_1", "cocina", "esta_en"): (0.12, 0.20),
    ("persona_1", "salon", "esta_en"): (0.08, -0.23),
    ("taza_1", "cocina", "esta_en"): (0.0, 0.36),
}


SCENE_EDGE_RADS = {
    ("robot_1", "taza_1", "observa"): -0.30,
    ("robot_1", "taza_1", "puede_agarrar"): 0.0,
    ("robot_1", "taza_1", "sostiene"): 0.30,
    ("robot_1", "persona_1", "asiste_a"): -0.24,
    ("persona_1", "taza_1", "quiere"): -0.12,
    ("taza_1", "cocina", "esta_en"): -0.16,
}


ONTOLOGY_EDGE_RADS = {
    ("taza_1", "Recipiente", "es_un"): -0.34,
    ("mesa_1", "Mueble", "es_un"): 0.34,
    ("robot_1", "Agente", "es_un"): -0.30,
    ("persona_1", "Agente", "es_un"): 0.30,
}


ONTOLOGY_EDGE_LABEL_OFFSETS = {
    ("Recipiente", "Objeto", "subclase_de"): (-0.20, 0.08),
    ("Mueble", "Objeto", "subclase_de"): (0.20, 0.08),
    ("Robot", "Agente", "subclase_de"): (-0.24, 0.10),
    ("Persona", "Agente", "subclase_de"): (0.24, 0.10),
    ("cocina", "Habitacion", "es_un"): (-0.12, 0.0),
    ("salon", "Habitacion", "es_un"): (0.14, 0.0),
    ("taza_1", "Recipiente", "es_un"): (-0.25, 0.0),
    ("mesa_1", "Mueble", "es_un"): (0.25, 0.0),
    ("robot_1", "Agente", "es_un"): (-0.25, 0.0),
    ("persona_1", "Agente", "es_un"): (0.25, 0.0),
}


def anadir_entidad(graph, identifier, attributes):
    """Añade una entidad validando su identificador y su tipo."""

    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Cada entidad debe tener un identificador no vacío.")

    if identifier in graph:
        raise ValueError(f"La entidad {identifier} ya existe.")

    entity_type = attributes.get("entity_type")
    valid_types = {
        ENTITY_OBJECT,
        ENTITY_PLACE,
        ENTITY_AGENT,
        ENTITY_CLASS,
    }

    if entity_type not in valid_types:
        raise ValueError(
            f"Tipo desconocido para {identifier}: {entity_type}"
        )

    node_attributes = dict(attributes)
    node_attributes.setdefault("label", identifier)
    graph.add_node(identifier, **node_attributes)


def anadir_relacion(
    graph,
    origin,
    destination,
    relation,
    *,
    fact_type=FACT_DECLARED,
    confidence=1.0,
    source="conocimiento_inicial",
    timestamp=0,
    active=True,
    rule=None,
    support_edges=None,
):
    """Añade una relación semántica con procedencia y estado explícitos."""

    if origin not in graph or destination not in graph:
        raise ValueError(
            f"La relación {origin}→{destination} referencia una entidad ausente."
        )

    if not relation:
        raise ValueError("Toda relación debe tener una etiqueta semántica.")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("La confianza debe pertenecer al intervalo [0, 1].")

    key = relation
    suffix = 2

    while graph.has_edge(origin, destination, key=key):
        key = f"{relation}_{suffix}"
        suffix += 1

    graph.add_edge(
        origin,
        destination,
        key=key,
        relation=relation,
        fact_type=fact_type,
        confidence=float(confidence),
        source=source,
        timestamp=timestamp,
        active=bool(active),
        rule=rule,
        support_edges=list(support_edges or []),
    )

    return origin, destination, key


def iterar_relaciones(
    graph,
    *,
    origin=None,
    destination=None,
    relation=None,
    active=None,
):
    """Itera sobre relaciones que cumplen un patrón de búsqueda."""

    for u, v, key, data in graph.edges(keys=True, data=True):
        if origin is not None and u != origin:
            continue
        if destination is not None and v != destination:
            continue
        if relation is not None and data.get("relation") != relation:
            continue
        if active is not None and data.get("active", True) != active:
            continue

        yield u, v, key, data


def existe_relacion_activa(graph, origin, destination, relation):
    """Comprueba si existe una relación activa concreta."""

    return any(
        True
        for _ in iterar_relaciones(
            graph,
            origin=origin,
            destination=destination,
            relation=relation,
            active=True,
        )
    )


def crear_grafo_conocimiento():
    """Crea el grafo semántico inicial de un robot doméstico."""

    graph = nx.MultiDiGraph()
    graph.graph["name"] = "Conocimiento doméstico del robot"
    graph.graph["current_time"] = 12

    for identifier, attributes in ENTITIES.items():
        anadir_entidad(graph, identifier, attributes)

    for origin, destination, relation in CLASS_RELATIONS:
        anadir_relacion(
            graph,
            origin,
            destination,
            relation,
            fact_type=FACT_DECLARED,
            source="ontologia_domestica",
        )

    for origin, destination, relation in INSTANCE_RELATIONS:
        anadir_relacion(
            graph,
            origin,
            destination,
            relation,
            fact_type=FACT_DECLARED,
            source="registro_de_entidades",
        )

    anadir_relacion(
        graph,
        "robot_1",
        "taza_1",
        "observa",
        fact_type=FACT_OBSERVED,
        confidence=0.94,
        source="camara_frontal",
        timestamp=12,
    )
    anadir_relacion(
        graph,
        "taza_1",
        "mesa_1",
        "encima_de",
        fact_type=FACT_OBSERVED,
        confidence=0.92,
        source="percepcion_3d",
        timestamp=12,
    )
    anadir_relacion(
        graph,
        "mesa_1",
        "cocina",
        "esta_en",
        fact_type=FACT_DECLARED,
        source="mapa_semantico",
        timestamp=1,
    )
    anadir_relacion(
        graph,
        "persona_1",
        "salon",
        "esta_en",
        fact_type=FACT_OBSERVED,
        confidence=0.97,
        source="camara_salon",
        timestamp=11,
    )
    anadir_relacion(
        graph,
        "persona_1",
        "taza_1",
        "quiere",
        fact_type=FACT_DECLARED,
        source="instruccion_usuario",
        timestamp=10,
    )
    anadir_relacion(
        graph,
        "robot_1",
        "taza_1",
        "puede_agarrar",
        fact_type=FACT_DECLARED,
        source="modelo_capacidades",
        timestamp=0,
    )
    anadir_relacion(
        graph,
        "robot_1",
        "persona_1",
        "asiste_a",
        fact_type=FACT_DECLARED,
        source="configuracion_mision",
        timestamp=0,
    )

    return graph


def validar_grafo_semantico(graph):
    """Valida la estructura, atributos y procedencia de todo el grafo."""

    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("El ejemplo debe utilizar nx.MultiDiGraph.")

    valid_types = {
        ENTITY_OBJECT,
        ENTITY_PLACE,
        ENTITY_AGENT,
        ENTITY_CLASS,
    }
    valid_fact_types = {
        FACT_DECLARED,
        FACT_OBSERVED,
        FACT_INFERRED,
        FACT_ACTION,
    }

    for node, data in graph.nodes(data=True):
        if data.get("entity_type") not in valid_types:
            raise ValueError(f"La entidad {node} tiene un tipo inválido.")
        if not data.get("label"):
            raise ValueError(f"La entidad {node} no tiene etiqueta visual.")

    for u, v, key, data in graph.edges(keys=True, data=True):
        if u not in graph or v not in graph:
            raise ValueError(f"La arista {u}→{v} contiene referencias rotas.")
        if not data.get("relation"):
            raise ValueError(f"La arista {u}→{v} no tiene relación semántica.")
        if data.get("fact_type") not in valid_fact_types:
            raise ValueError(f"Tipo de hecho inválido en {u}→{v}.")
        if not 0.0 <= data.get("confidence", -1.0) <= 1.0:
            raise ValueError(f"Confianza inválida en {u}→{v}.")
        if not isinstance(data.get("active"), bool):
            raise ValueError(f"El estado de {u}→{v} debe ser booleano.")
        if not data.get("source"):
            raise ValueError(f"La relación {u}→{v} no declara procedencia.")

    if not nx.is_weakly_connected(nx.DiGraph(graph)):
        raise ValueError("El grafo semántico debe formar un conjunto conectado.")

    return {
        "entities": graph.number_of_nodes(),
        "relations": graph.number_of_edges(),
        "classes": sum(
            data.get("entity_type") == ENTITY_CLASS
            for _, data in graph.nodes(data=True)
        ),
        "instances": sum(
            data.get("entity_type") != ENTITY_CLASS
            for _, data in graph.nodes(data=True)
        ),
    }


def inferir_ubicaciones(graph):
    """Aplica una regla espacial: encima_de(X,Y) y esta_en(Y,Z)."""

    inferences = []
    above_edges = list(
        iterar_relaciones(graph, relation="encima_de", active=True)
    )

    for x, y, above_key, _ in above_edges:
        for _, z, location_key, _ in iterar_relaciones(
            graph,
            origin=y,
            relation="esta_en",
            active=True,
        ):
            if existe_relacion_activa(graph, x, z, "esta_en"):
                continue

            edge = anadir_relacion(
                graph,
                x,
                z,
                "esta_en",
                fact_type=FACT_INFERRED,
                confidence=0.88,
                source="motor_de_inferencia",
                timestamp=graph.graph["current_time"],
                rule="encima_de(X,Y) ∧ esta_en(Y,Z) → esta_en(X,Z)",
                support_edges=[
                    (x, y, above_key),
                    (y, z, location_key),
                ],
            )
            inferences.append(edge)

    return inferences


def inferir_clases_generales(graph):
    """Infere una clase superior a partir de es_un y subclase_de."""

    inferences = []
    instance_edges = list(
        iterar_relaciones(graph, relation="es_un", active=True)
    )

    for entity, direct_class, instance_key, instance_data in instance_edges:
        if instance_data.get("fact_type") == FACT_INFERRED:
            continue

        for _, general_class, subclass_key, _ in iterar_relaciones(
            graph,
            origin=direct_class,
            relation="subclase_de",
            active=True,
        ):
            if existe_relacion_activa(
                graph,
                entity,
                general_class,
                "es_un",
            ):
                continue

            edge = anadir_relacion(
                graph,
                entity,
                general_class,
                "es_un",
                fact_type=FACT_INFERRED,
                confidence=1.0,
                source="motor_de_inferencia",
                timestamp=graph.graph["current_time"],
                rule="es_un(X,A) ∧ subclase_de(A,B) → es_un(X,B)",
                support_edges=[
                    (entity, direct_class, instance_key),
                    (direct_class, general_class, subclass_key),
                ],
            )
            inferences.append(edge)

    return inferences


def objetivos_de_relacion(graph, origin, relation):
    """Obtiene los destinos activos de una relación desde un origen."""

    return {
        destination
        for _, destination, _, _ in iterar_relaciones(
            graph,
            origin=origin,
            relation=relation,
            active=True,
        )
    }


def consultar_objeto_deseado(graph, person):
    """Consulta qué objetos quiere una persona."""

    return sorted(objetivos_de_relacion(graph, person, "quiere"))


def consultar_objetos_deseados_y_agarrables(graph, person, robot):
    """Busca objetos que satisfacen simultáneamente dos relaciones."""

    desired = objetivos_de_relacion(graph, person, "quiere")
    grabbable = objetivos_de_relacion(graph, robot, "puede_agarrar")
    return sorted(desired & grabbable)


def consultar_ubicaciones(graph, entity):
    """Devuelve los lugares activos asociados mediante esta_en."""

    return sorted(objetivos_de_relacion(graph, entity, "esta_en"))


def consultar_quien_sostiene(graph, object_id):
    """Consulta qué agentes mantienen una relación sostiene activa."""

    holders = {
        origin
        for origin, _, _, _ in iterar_relaciones(
            graph,
            destination=object_id,
            relation="sostiene",
            active=True,
        )
    }
    return sorted(holders)


def proponer_objetivo_entrega(graph, person, robot):
    """Propone un objetivo simbólico a partir del conocimiento disponible."""

    candidates = consultar_objetos_deseados_y_agarrables(
        graph,
        person,
        robot,
    )

    if not candidates:
        return None

    object_id = candidates[0]
    object_locations = consultar_ubicaciones(graph, object_id)
    person_locations = consultar_ubicaciones(graph, person)

    if not object_locations or not person_locations:
        return None

    return {
        "action": "entregar",
        "object": object_id,
        "recipient": person,
        "origin": object_locations[0],
        "destination": person_locations[0],
        "text": (
            f"entregar {object_id} a {person} "
            f"({object_locations[0]} → {person_locations[0]})"
        ),
    }


def desactivar_relaciones(graph, *, origin, relation, fact_type=None):
    """Desactiva hechos y devuelve sus identificadores."""

    deactivated = []

    for u, v, key, data in iterar_relaciones(
        graph,
        origin=origin,
        relation=relation,
        active=True,
    ):
        if fact_type is not None and data.get("fact_type") != fact_type:
            continue

        data["active"] = False
        data["invalidated_at"] = graph.graph["current_time"]
        deactivated.append((u, v, key))

    return deactivated


def actualizar_despues_del_agarre(graph, robot, object_id):
    """Actualiza el mundo cuando el robot pasa a sostener un objeto."""

    graph.graph["current_time"] = 20

    deactivated_support = desactivar_relaciones(
        graph,
        origin=object_id,
        relation="encima_de",
    )
    invalidated_inferences = desactivar_relaciones(
        graph,
        origin=object_id,
        relation="esta_en",
        fact_type=FACT_INFERRED,
    )

    added_edge = anadir_relacion(
        graph,
        robot,
        object_id,
        "sostiene",
        fact_type=FACT_ACTION,
        confidence=1.0,
        source="resultado_accion_agarrar",
        timestamp=graph.graph["current_time"],
    )

    return {
        "deactivated_support": deactivated_support,
        "invalidated_inferences": invalidated_inferences,
        "added_edge": added_edge,
    }


def obtener_estadisticas(graph):
    """Calcula estadísticas finales separadas por estado y procedencia."""

    edges = list(graph.edges(keys=True, data=True))

    def count_fact(fact_type):
        return sum(
            data.get("fact_type") == fact_type
            for _, _, _, data in edges
        )

    active_count = sum(
        data.get("active", True)
        for _, _, _, data in edges
    )

    return {
        "entities": graph.number_of_nodes(),
        "relations_total": graph.number_of_edges(),
        "relations_active": active_count,
        "relations_inactive": graph.number_of_edges() - active_count,
        "observed": count_fact(FACT_OBSERVED),
        "declared": count_fact(FACT_DECLARED),
        "inferred": count_fact(FACT_INFERRED),
        "actions": count_fact(FACT_ACTION),
    }


def validar_resultados(
    graph,
    desired,
    desired_and_grabbable,
    locations_before_grab,
    goal,
    update_result,
    holders,
):
    """Comprueba que consultas, inferencias y actualización sean coherentes."""

    if desired != ["taza_1"]:
        raise ValueError("persona_1 debe querer exactamente taza_1.")

    if desired_and_grabbable != ["taza_1"]:
        raise ValueError("taza_1 debe ser deseada y agarrable.")

    if locations_before_grab != ["cocina"]:
        raise ValueError("La ubicación inferida inicial debe ser cocina.")

    if goal is None or goal["object"] != "taza_1":
        raise ValueError("Debe proponerse la entrega de taza_1.")

    if not update_result["deactivated_support"]:
        raise ValueError("El agarre debe desactivar la relación encima_de.")

    if not update_result["invalidated_inferences"]:
        raise ValueError("Debe invalidarse la ubicación espacial inferida.")

    if holders != ["robot_1"]:
        raise ValueError("robot_1 debe sostener taza_1 al final.")

    if existe_relacion_activa(graph, "taza_1", "mesa_1", "encima_de"):
        raise ValueError("taza_1 ya no puede seguir encima de mesa_1.")

    if not existe_relacion_activa(graph, "robot_1", "taza_1", "sostiene"):
        raise ValueError("Debe existir la relación activa sostiene.")

    if not existe_relacion_activa(graph, "taza_1", "Recipiente", "es_un"):
        raise ValueError("Debe mantenerse la inferencia taxonómica.")


def imprimir_resumen(
    validation,
    desired,
    desired_and_grabbable,
    locations_before_grab,
    goal,
    holders,
    statistics,
):
    """Imprime un resumen determinista del conocimiento y las consultas."""

    print("\n=== Grafo de conocimiento de un robot doméstico ===")
    print(f"Entidades iniciales: {validation['entities']}")
    print(f"Clases: {validation['classes']}")
    print(f"Instancias: {validation['instances']}")
    print(f"Objeto deseado por persona_1: {desired}")
    print(f"Deseado y agarrable por robot_1: {desired_and_grabbable}")
    print(f"Ubicación inferida antes del agarre: {locations_before_grab}")
    print(f"Objetivo candidato: {goal['text']}")
    print(f"Quién sostiene taza_1 al final: {holders}")
    print(f"Relaciones totales: {statistics['relations_total']}")
    print(f"Relaciones activas: {statistics['relations_active']}")
    print(f"Relaciones históricas: {statistics['relations_inactive']}")
    print(f"Hechos inferidos: {statistics['inferred']}")


def main():
    graph = crear_grafo_conocimiento()
    validation = validar_grafo_semantico(graph)

    spatial_inferences = inferir_ubicaciones(graph)
    class_inferences = inferir_clases_generales(graph)

    desired = consultar_objeto_deseado(graph, "persona_1")
    desired_and_grabbable = consultar_objetos_deseados_y_agarrables(
        graph,
        "persona_1",
        "robot_1",
    )
    locations_before_grab = consultar_ubicaciones(graph, "taza_1")
    goal = proponer_objetivo_entrega(graph, "persona_1", "robot_1")

    update_result = actualizar_despues_del_agarre(
        graph,
        "robot_1",
        "taza_1",
    )
    holders = consultar_quien_sostiene(graph, "taza_1")
    statistics = obtener_estadisticas(graph)

    validar_grafo_semantico(graph)
    validar_resultados(
        graph=graph,
        desired=desired,
        desired_and_grabbable=desired_and_grabbable,
        locations_before_grab=locations_before_grab,
        goal=goal,
        update_result=update_result,
        holders=holders,
    )

    imprimir_resumen(
        validation=validation,
        desired=desired,
        desired_and_grabbable=desired_and_grabbable,
        locations_before_grab=locations_before_grab,
        goal=goal,
        holders=holders,
        statistics=statistics,
    )

    inference_results = []

    for u, v, key in spatial_inferences:
        data = graph.edges[u, v, key]
        inference_results.append({
            "fact": f"{u} esta_en {v}",
            "rule": data["rule"],
            "active": data["active"],
        })

    for u, v, key in class_inferences:
        if u != "taza_1":
            continue

        data = graph.edges[u, v, key]
        inference_results.append({
            "fact": f"{u} es_un {v}",
            "rule": data["rule"],
            "active": data["active"],
        })

    query_results = [
        {
            "question": "¿Qué objeto quiere persona_1?",
            "answer": ", ".join(desired),
        },
        {
            "question": "¿Qué objeto quiere y puede agarrar el robot?",
            "answer": ", ".join(desired_and_grabbable),
        },
        {
            "question": "¿Dónde estaba la taza antes del agarre?",
            "answer": ", ".join(locations_before_grab),
        },
        {
            "question": "¿Quién sostiene taza_1 al final?",
            "answer": ", ".join(holders),
        },
    ]

    update_summary = {
        "event": "robot_1 agarra taza_1",
        "deactivated": "taza_1 encima_de mesa_1",
        "invalidated": "taza_1 esta_en cocina (inferida)",
        "added": "robot_1 sostiene taza_1",
        "goal": goal["text"],
    }

    visualizer = GraphVisualizer(figsize=(22, 14.5))

    output_path = (
        GRAFOS_DIR
        / "assets"
        / "04_robotica"
        / "05_grafo_conocimiento_robot.png"
    )

    visualizer.show_robot_knowledge_graph(
        graph=graph,
        scene_positions=SCENE_POSITIONS,
        ontology_positions=ONTOLOGY_POSITIONS,
        query_results=query_results,
        inference_results=inference_results,
        update_summary=update_summary,
        statistics=statistics,
        title="Grafo de conocimiento de un robot doméstico",
        subtitle=(
            "La percepción crea hechos, las reglas generan inferencias "
            "y las acciones actualizan el conocimiento"
        ),
        save_path=output_path,
        scene_edge_label_offsets=SCENE_EDGE_LABEL_OFFSETS,
        scene_edge_rads=SCENE_EDGE_RADS,
        ontology_edge_label_offsets=ONTOLOGY_EDGE_LABEL_OFFSETS,
        ontology_edge_rads=ONTOLOGY_EDGE_RADS,
        highlight_nodes={"robot_1", "taza_1", "persona_1"},
        highlight_edges={
            ("persona_1", "taza_1", "quiere"),
            ("taza_1", "Recipiente", "es_un"),
        },
    )


if __name__ == "__main__":
    main()
