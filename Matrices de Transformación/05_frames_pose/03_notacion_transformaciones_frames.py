from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rz(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])


def ry(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,0.0,s],[0.0,1.0,0.0],[-s,0.0,c]])


def T(R,t):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=t; return M


def interp_pose(s, R_final, t_final, tipo="z"):
    if tipo=="z":
        ang=np.arctan2(R_final[1,0], R_final[0,0])
        R=rz(s*ang)
    else:
        ang=np.arctan2(R_final[0,2], R_final[0,0])
        R=ry(s*ang)
    return T(R, s*np.asarray(t_final))


def suavizar(p):
    return 0.5-0.5*np.cos(np.pi*p)


def transformar(M,p):
    return (M@np.r_[p,1.0])[:3]


def fmt(v):
    return "["+", ".join(f"{x:5.2f}" for x in np.asarray(v))+"]"


def crear_estado(s_ab, s_bc, fase, mensaje):
    R_AB=rz(np.radians(35.0)); t_AB=np.array([2.2,0.7,0.45])
    R_BC=ry(np.radians(-28.0)); t_BC=np.array([1.25,0.25,0.85])

    T_AB=interp_pose(s_ab,R_AB,t_AB,"z")
    T_BC=interp_pose(s_bc,R_BC,t_BC,"y")
    T_AC=T_AB@T_BC

    p_C=np.array([0.8,-0.35,0.55])
    p_B=transformar(T_BC,p_C)
    p_A=transformar(T_AB,p_B)
    p_A_directo=transformar(T_AC,p_C)

    oA=np.zeros(3); oB=T_AB[:3,3]; oC=T_AC[:3,3]

    return {
        "frames3d":[
            {"name":"A","origin":oA,"rotation":np.eye(3),"length":1.35,"alpha":0.75},
            {"name":"B","origin":oB,"rotation":T_AB[:3,:3],"length":1.25,"alpha":1.0},
            {"name":"C","origin":oC,"rotation":T_AC[:3,:3],"length":1.15,"alpha":1.0,
             "colors":("#D97706","#0F766E","#2563EB")},
        ],
        "segments3d":[
            {"start":oA,"end":oB,"color":"#7B2CBF","alpha":0.55,"linestyle":"--"},
            {"start":oB,"end":oC,"color":"#E07A1F","alpha":0.65,"linestyle":"--"},
            {"start":oC,"end":p_A,"color":"#2D7F5E","alpha":0.55,"linestyle":"--"},
        ],
        "points3d":[
            {"name":"p","position":p_A,"color":"#7B2CBF","size":82},
        ],
        "message":mensaje,
        "info_title":"Notación y cancelación de frames",
        "info_lines":[
            {"text":"TRANSFORMACIONES", "bold":True},
            "^A T_B : B -> A",
            "^B T_C : C -> B",
            "",
            {"text":"COMPOSICIÓN", "bold":True},
            "^A T_C = ^A T_B  ^B T_C",
            "           B se cancela",
            "",
            {"text":"PUNTO", "bold":True},
            f"^C p = {fmt(p_C)}",
            f"^B p = {fmt(p_B)}",
            f"^A p = {fmt(p_A)}",
            "",
            f"error directo = {np.linalg.norm(p_A-p_A_directo):.2e}",
        ],
        "phase":fase,
        "info_line_height":0.041,
        "info_fontsize":8.8,
        "legend":[
            {"kind":"line","label":"A -> B","color":"#7B2CBF"},
            {"kind":"line","label":"B -> C","color":"#E07A1F"},
            {"kind":"point","label":"mismo punto p","color":"#7B2CBF"},
        ],
        "legend_fontsize":8.0,
    }


def crear_estados_demostracion():
    estados=[]
    for p in np.linspace(0.0,1.0,90):
        estados.append(crear_estado(suavizar(p),0.0,"1/3 · ^A T_B","Primero situamos el frame {B} respecto a {A}. La notación ^A T_B indica que convierte coordenadas de B hacia A."))
    for p in np.linspace(0.0,1.0,100):
        estados.append(crear_estado(1.0,suavizar(p),"2/3 · ^B T_C","Ahora situamos {C} respecto a {B}. Su pose global se obtiene componiendo ^A T_B con ^B T_C."))
    for _ in range(70):
        estados.append(crear_estado(1.0,1.0,"3/3 · Regla de cancelación","La cadena ^A T_B ^B T_C produce ^A T_C. El punto puede transformarse C -> B -> A o directamente C -> A con el mismo resultado."))
    return {"states":estados}


def main():
    resultado=crear_estados_demostracion()
    animador=TransformAnimator(figsize=(15.6,8.9), interval=50)
    image_path=MATRICES_DIR/"assets"/"05_frames_pose"/"03_notacion_transformaciones_frames.png"
    video_path=MATRICES_DIR/"assets"/"05_frames_pose"/"03_notacion_transformaciones_frames.webm"
    animador.animate_3d_states(
        states=resultado["states"], title="5.3. Notación de transformaciones entre frames",
        limits=(-1.8,5.8,-2.7,3.7,-1.5,3.8), view=(24.0,-58.0),
        final_image_path=image_path, video_path=video_path, repeat=False, fps=20, dpi=125, show=True)


if __name__=="__main__":
    main()
