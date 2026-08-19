from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rx(a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],dtype=float)
def ry(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],dtype=float)
def rz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=float)

def T(R,t):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=np.asarray(t); return M

def invT(M):
    R=M[:3,:3]; t=M[:3,3]
    N=np.eye(4); N[:3,:3]=R.T; N[:3,3]=-R.T@t
    return N

def transformar(M,p):
    return (M@np.r_[p,1.0])[:3]

def suavizar(p):
    return 0.5-0.5*np.cos(np.pi*p)

def fmt(v):
    return "["+", ".join(f"{x:5.2f}" for x in np.asarray(v))+"]"

def crear_estado(s, fase, mensaje):
    T_AB=T(rz(np.radians(35))@ry(np.radians(-10)), np.array([2.0,0.8,0.6]))
    T_AC=T(rz(np.radians(-25))@rx(np.radians(18)), np.array([-0.8,2.5,1.1]))
    T_BC=invT(T_AB)@T_AC

    oA=np.zeros(3)
    oB=T_AB[:3,3]
    oC=T_AC[:3,3]
    # Punto local en C para comprobar la relación relativa.
    pC=np.array([0.75,0.20,0.35])
    pB=transformar(T_BC,pC)
    pA=transformar(T_AC,pC)
    pA_viaB=transformar(T_AB,pB)

    # La línea directa B->C aparece progresivamente.
    inter=oB + s*(oC-oB)
    segs=[
        {"start":oA,"end":oB,"color":"#7B2CBF","alpha":0.45,"linestyle":"--"},
        {"start":oA,"end":oC,"color":"#2D7F5E","alpha":0.45,"linestyle":"--"},
    ]
    if s>1e-6:
        segs.append({"start":oB,"end":inter,"color":"#E07A1F","alpha":0.90,"linestyle":"-","linewidth":2.4})

    return {
        "frames3d":[
            {"name":"A","origin":oA,"rotation":np.eye(3),"length":1.25,"alpha":0.55},
            {"name":"B","origin":oB,"rotation":T_AB[:3,:3],"length":1.25,"alpha":1.0},
            {"name":"C","origin":oC,"rotation":T_AC[:3,:3],"length":1.25,"alpha":1.0,
             "colors":("#D97706","#0F766E","#2563EB")},
        ],
        "segments3d":segs,
        "points3d":[{"name":"p físico","position":pA,"color":"#7B2CBF","size":82}],
        "message":mensaje,
        "info_title":"Transformación relativa B -> C",
        "info_lines":[
            {"text":"DATOS CONOCIDOS","bold":True},
            "^A T_B",
            "^A T_C",
            "",
            {"text":"RELACIÓN BUSCADA","bold":True},
            "^B T_C = (^A T_B)^-1 ^A T_C",
            "          B->A      A->C",
            "",
            f"t_BC = {fmt(T_BC[:3,3])}",
            f"^C p = {fmt(pC)}",
            f"^B p = {fmt(pB)}",
            "",
            f"error vía B = {np.linalg.norm(pA-pA_viaB):.2e}",
        ],
        "phase":fase,
        "info_line_height":0.0405,
        "info_fontsize":8.7,
        "legend":[
            {"kind":"line","label":"poses conocidas desde A","color":"#7B2CBF"},
            {"kind":"line","label":"relación B -> C","color":"#E07A1F"},
        ],
        "legend_fontsize":8.0,
    }

def crear_estados_demostracion():
    estados=[]
    for _ in range(35):
        estados.append(crear_estado(0.0,"1/3 · B y C respecto a A",
            "Conocemos por separado las poses de {B} y {C} respecto al mismo frame común {A}. Queremos eliminar ese intermediario."))
    for p in np.linspace(0,1,115):
        estados.append(crear_estado(suavizar(p),"2/3 · B -> A -> C",
            "Primero invertimos ^A T_B para pasar de B a A y después componemos con ^A T_C. Los índices internos A se cancelan."))
    for _ in range(80):
        estados.append(crear_estado(1.0,"3/3 · ^B T_C",
            "La matriz resultante expresa directamente la pose de {C} vista desde {B}. Un punto C->B->A coincide con C->A."))
    return {"states":estados}

def main():
    resultado=crear_estados_demostracion()
    animador=TransformAnimator(figsize=(15.7,8.9), interval=50)
    image_path=MATRICES_DIR/"assets"/"06_composicion_inversion"/"04_transformacion_relativa.png"
    video_path=MATRICES_DIR/"assets"/"06_composicion_inversion"/"04_transformacion_relativa.webm"
    animador.animate_3d_states(
        states=resultado["states"], title="6.4. Transformación relativa entre dos frames",
        limits=(-3.3,4.3,-2.6,4.6,-1.5,3.8), view=(24,-58),
        final_image_path=image_path, video_path=video_path, repeat=False, fps=20, dpi=125, show=True)

if __name__=="__main__":
    main()
