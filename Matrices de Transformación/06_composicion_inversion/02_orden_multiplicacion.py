from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rotacion_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s],[s,c]], dtype=float)

def H(R=None, t=None):
    M = np.eye(3)
    if R is not None:
        M[:2,:2] = R
    if t is not None:
        M[:2,2] = np.asarray(t, dtype=float)
    return M

def suavizar(p):
    return 0.5 - 0.5*np.cos(np.pi*p)

def aplicar(M, p):
    return (M @ np.r_[p,1.0])[:2]

def fmt(v):
    return "[" + ", ".join(f"{x:5.2f}" for x in np.asarray(v)) + "]"

def trayectoria_orden(p0, R, t, orden, n=60):
    puntos = [p0.copy()]
    if orden == "TR":
        # T @ R: primero rota, después traslada.
        for s in np.linspace(0,1,n):
            puntos.append(rotacion_2d(s*np.pi/2) @ p0)
        p_r = R @ p0
        for s in np.linspace(0,1,n):
            puntos.append(p_r + s*t)
    else:
        # R @ T: primero traslada, después rota todo alrededor del origen.
        for s in np.linspace(0,1,n):
            puntos.append(p0 + s*t)
        p_t = p0 + t
        for s in np.linspace(0,1,n):
            puntos.append(rotacion_2d(s*np.pi/2) @ p_t)
    return np.asarray(puntos)

def crear_estado(fase_idx, progreso, fase, mensaje):
    theta = np.pi/2
    R = rotacion_2d(theta)
    t = np.array([3.0, 1.0])
    p0 = np.array([1.0, 0.0])

    HR = H(R=R)
    HT = H(t=t)
    M_TR = HT @ HR  # derecha primero: R y luego T
    M_RT = HR @ HT  # derecha primero: T y luego R
    p_TR = aplicar(M_TR, p0)
    p_RT = aplicar(M_RT, p0)

    s = suavizar(progreso)
    if fase_idx == 0:
        p = p0
        path = np.array([p0,p0])
        color = "#6B7280"
        etiqueta = "p inicial"
    elif fase_idx == 1:
        # T@R: primero rotar
        p = rotacion_2d(s*theta) @ p0
        path = trayectoria_orden(p0,R,t,"TR")[:max(2,int(60*s)+1)]
        color = "#2563EB"
        etiqueta = "T·R: rotando"
    elif fase_idx == 2:
        p_r = R @ p0
        p = p_r + s*t
        full = trayectoria_orden(p0,R,t,"TR")
        path = full[:60 + max(2,int(60*s)+1)]
        color = "#2563EB"
        etiqueta = "T·R"
    elif fase_idx == 3:
        p = p0 + s*t
        full = trayectoria_orden(p0,R,t,"RT")
        path = full[:max(2,int(60*s)+1)]
        color = "#D97706"
        etiqueta = "R·T: trasladando"
    elif fase_idx == 4:
        p_t = p0 + t
        p = rotacion_2d(s*theta) @ p_t
        full = trayectoria_orden(p0,R,t,"RT")
        path = full[:60 + max(2,int(60*s)+1)]
        color = "#D97706"
        etiqueta = "R·T"
    else:
        p = p_TR
        path = trayectoria_orden(p0,R,t,"TR")
        color = "#2563EB"
        etiqueta = "comparación"

    state = {
        "points": [
            {"name":"p0","position":p0,"color":"#6B7280","alpha":0.45,"size":70},
        ],
        "polylines": [],
        "vectors": [
            {"name":"t","origin":np.zeros(2),"value":t,"color":"#7B2CBF","alpha":0.35,"linewidth":2.2},
        ],
        "message": mensaje,
        "info_title": "El orden de multiplicación importa",
        "info_lines": [
            {"text":"MATRICES","bold":True},
            "T·R -> primero R, luego T",
            "R·T -> primero T, luego R",
            "",
            f"p0       = {fmt(p0)}",
            f"(T·R)p0  = {fmt(p_TR)}",
            f"(R·T)p0  = {fmt(p_RT)}",
            "",
            f"diferencia = {np.linalg.norm(p_TR-p_RT):.3f}",
            "",
            "Con vectores columna,",
            "la derecha actúa primero.",
        ],
        "phase": fase,
        "info_line_height": 0.047,
        "info_fontsize": 9.0,
    }

    if fase_idx in (1,2):
        state["points"].append({"name":etiqueta,"position":p,"color":"#2563EB","size":90})
        state["polylines"].append({"points":path,"color":"#2563EB","linewidth":2.3,"alpha":0.85})
    elif fase_idx in (3,4):
        state["points"].append({"name":etiqueta,"position":p,"color":"#D97706","size":90})
        state["polylines"].append({"points":path,"color":"#D97706","linewidth":2.3,"alpha":0.85})
    elif fase_idx == 5:
        state["points"].extend([
            {"name":"(T·R)p","position":p_TR,"color":"#2563EB","size":100},
            {"name":"(R·T)p","position":p_RT,"color":"#D97706","size":100},
        ])
        state["polylines"].extend([
            {"points":trayectoria_orden(p0,R,t,"TR"),"color":"#2563EB","linewidth":2.2,"alpha":0.80},
            {"points":trayectoria_orden(p0,R,t,"RT"),"color":"#D97706","linewidth":2.2,"alpha":0.80},
        ])
        state["legend"] = [
            {"kind":"line","label":"T·R: rotar y trasladar","color":"#2563EB"},
            {"kind":"line","label":"R·T: trasladar y rotar","color":"#D97706"},
        ]
        state["legend_fontsize"] = 8.1
    return state

def crear_estados_demostracion():
    estados=[]
    for _ in range(25):
        estados.append(crear_estado(0,0.0,"1/6 · Punto inicial",
            "Usaremos el mismo punto, una rotación de 90° y la misma traslación. Solo cambiaremos el orden de las matrices."))
    for p in np.linspace(0,1,70):
        estados.append(crear_estado(1,p,"2/6 · T·R: primero rotar",
            "En T·R, la matriz R está a la derecha y actúa primero. El punto gira alrededor del origen."))
    for p in np.linspace(0,1,65):
        estados.append(crear_estado(2,p,"3/6 · T·R: después trasladar",
            "Tras la rotación se aplica T. La traslación conserva su dirección global."))
    for p in np.linspace(0,1,65):
        estados.append(crear_estado(3,p,"4/6 · R·T: primero trasladar",
            "En R·T, T actúa primero. El punto se aleja del origen antes de realizar el giro."))
    for p in np.linspace(0,1,80):
        estados.append(crear_estado(4,p,"5/6 · R·T: después rotar",
            "Ahora R gira alrededor del origen tanto el punto como el desplazamiento previo, produciendo una trayectoria distinta."))
    for _ in range(65):
        estados.append(crear_estado(5,1.0,"6/6 · No conmutatividad",
            "Los resultados finales y las trayectorias son diferentes: T·R ≠ R·T. El orden forma parte del significado de la cadena."))
    return {"states":estados}

def main():
    resultado=crear_estados_demostracion()
    animador=TransformAnimator(figsize=(15.5,8.8), interval=50)
    image_path=MATRICES_DIR/"assets"/"06_composicion_inversion"/"02_orden_multiplicacion.png"
    video_path=MATRICES_DIR/"assets"/"06_composicion_inversion"/"02_orden_multiplicacion.webm"
    animador.animate_2d_states(
        states=resultado["states"], title="6.2. El orden de multiplicación importa",
        limits=(-5.4,5.4,-4.7,5.4), final_image_path=image_path, video_path=video_path,
        repeat=False, fps=20, dpi=130, show=True)

if __name__=="__main__":
    main()
