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


def suavizar(p):
    return 0.5-0.5*np.cos(np.pi*p)


def fmt(v):
    return "["+", ".join(f"{x:5.2f}" for x in np.asarray(v))+"]"


def crear_estado(theta, fase, mensaje):
    R=rz(theta)
    p=np.array([1.8,0.8,0.45])
    p_act=R@p
    p_B=R.T@p

    o_act=np.array([-3.1,0.0,0.0])
    o_pas=np.array([3.1,0.0,0.0])

    return {
        "frames3d":[
            {"name":"A","origin":o_act,"rotation":np.eye(3),"length":1.30,"alpha":0.75},
            {"name":"B","origin":o_pas,"rotation":R,"length":1.30,"alpha":1.0,
             "colors":("#D97706","#0F766E","#2563EB")},
            {"name":"A","origin":o_pas,"rotation":np.eye(3),"length":1.15,"alpha":0.16,
             "colors":("#9CA3AF","#9CA3AF","#9CA3AF")},
        ],
        "vectors3d":[
            {"name":"p inicial","origin":o_act,"value":p,"color":"#6B7280","alpha":0.28,"linewidth":1.8},
            {"name":"Rp","origin":o_act,"value":p_act,"color":"#7B2CBF","linewidth":3.0},
            {"name":"p físico","origin":o_pas,"value":p,"color":"#E07A1F","linewidth":3.0},
        ],
        "texts3d":[
            {"position":o_act+np.array([0.0,0.0,2.25]),"text":"ACTIVA: rota el vector","fontweight":"bold","color":"#7B2CBF"},
            {"position":o_pas+np.array([0.0,0.0,2.25]),"text":"PASIVA: rota el frame","fontweight":"bold","color":"#D97706"},
        ],
        "message":mensaje,
        "info_title":"Activa frente a pasiva",
        "info_lines":[
            {"text":"ÁNGULO", "bold":True},
            f"theta = {np.degrees(theta):6.1f}°",
            "",
            {"text":"ROTACIÓN ACTIVA", "bold":True},
            f"p      = {fmt(p)}",
            f"p' = Rp= {fmt(p_act)}",
            "frame A permanece fijo",
            "",
            {"text":"CAMBIO PASIVO", "bold":True},
            f"^A p = {fmt(p)}",
            f"^B p = {fmt(p_B)}",
            "^B p = R^T ^A p",
            "vector físico fijo",
        ],
        "phase":fase,
        "info_line_height":0.041,
        "info_fontsize":8.8,
        "legend":[
            {"kind":"line","label":"vector rotado Rp","color":"#7B2CBF"},
            {"kind":"line","label":"vector físico fijo","color":"#E07A1F"},
        ],
        "legend_fontsize":8.0,
    }


def crear_estados_demostracion():
    estados=[]
    for _ in range(28):
        estados.append(crear_estado(0.0,"1/3 · Misma situación inicial","Partimos de dos copias equivalentes: en la activa moveremos el vector; en la pasiva mantendremos el vector físico y moveremos el frame."))
    for p in np.linspace(0.0,1.0,125):
        estados.append(crear_estado(suavizar(p)*np.radians(65.0),"2/3 · R frente a R^T","A la izquierda p' = R p. A la derecha el frame gira con R y las coordenadas del vector fijo pasan a ^B p = R^T ^A p."))
    for _ in range(70):
        estados.append(crear_estado(np.radians(65.0),"3/3 · Misma matriz, preguntas distintas","No hay contradicción: R puede rotar activamente un vector o describir la orientación de un frame. El significado depende de qué se mantiene fijo."))
    return {"states":estados}


def main():
    resultado=crear_estados_demostracion()
    animador=TransformAnimator(figsize=(15.8,8.9), interval=50)
    image_path=MATRICES_DIR/"assets"/"05_frames_pose"/"04_rotacion_activa_pasiva.png"
    video_path=MATRICES_DIR/"assets"/"05_frames_pose"/"04_rotacion_activa_pasiva.webm"
    animador.animate_3d_states(
        states=resultado["states"], title="5.4. Rotación activa frente a cambio pasivo de coordenadas",
        limits=(-5.3,5.4,-3.0,3.2,-1.5,3.5), view=(24.0,-58.0),
        final_image_path=image_path, video_path=video_path, repeat=False, fps=20, dpi=125, show=True)


if __name__=="__main__":
    main()
