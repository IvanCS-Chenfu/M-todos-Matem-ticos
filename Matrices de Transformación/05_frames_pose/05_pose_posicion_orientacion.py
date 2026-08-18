from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


BOX_FACES=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]


def rx(a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],dtype=float)


def ry(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],dtype=float)


def rz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=float)


def suavizar(p):
    return 0.5-0.5*np.cos(np.pi*p)


def T(R,t):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=t; return M


def caja(size):
    sx,sy,sz=np.asarray(size)/2.0
    return np.array([[-sx,-sy,-sz],[sx,-sy,-sz],[sx,sy,-sz],[-sx,sy,-sz],[-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]])


def transformar_vertices(vertices,M):
    h=np.c_[vertices,np.ones(len(vertices))]; return (M@h.T).T[:,:3]


def fmt(v):
    return "["+", ".join(f"{x:5.2f}" for x in np.asarray(v))+"]"


def crear_estado(progreso, fase, mensaje):
    s=suavizar(progreso)
    roll=s*np.radians(8.0); pitch=s*np.radians(-12.0); yaw=s*np.radians(20.0)
    R=rz(yaw)@ry(pitch)@rx(roll)
    t=s*np.array([0.30,0.0,0.20])
    T_base_camera=T(R,t)

    p_camera=np.array([1.50,0.20,0.40])
    p_base=(T_base_camera@np.r_[p_camera,1.0])[:3]

    cuerpo=transformar_vertices(caja((1.15,0.78,0.35)), T(np.eye(3),np.array([0.0,0.0,0.18])))
    camara=transformar_vertices(caja((0.30,0.24,0.18)), T_base_camera@T(np.eye(3),np.array([0.0,0.0,0.0])))

    return {
        "frames3d":[
            {"name":"base_link","origin":np.zeros(3),"rotation":np.eye(3),"length":0.95,"alpha":0.85},
            {"name":"camera_link","origin":t,"rotation":R,"length":0.78,"alpha":1.0,
             "colors":("#D97706","#0F766E","#2563EB")},
        ],
        "meshes3d":[
            {"vertices":cuerpo,"faces":BOX_FACES,"facecolor":"#CBD5E1","edgecolor":"#64748B","alpha":0.22,"linewidth":1.0},
            {"vertices":camara,"faces":BOX_FACES,"facecolor":"#F8CFA7","edgecolor":"#C2410C","alpha":0.45,"linewidth":1.1},
        ],
        "points3d":[
            {"name":"p detectado","position":p_base,"color":"#7B2CBF","size":82},
        ],
        "segments3d":[
            {"start":t,"end":p_base,"color":"#7B2CBF","alpha":0.55,"linestyle":"--"},
        ],
        "vectors3d":[
            {"name":"t","origin":np.zeros(3),"value":t,"color":"#E07A1F","linewidth":2.8},
        ],
        "message":mensaje,
        "info_title":"Pose de camera_link respecto a base_link",
        "info_lines":[
            {"text":"POSICIÓN", "bold":True},
            f"t = {fmt(t)} m",
            "",
            {"text":"ORIENTACIÓN", "bold":True},
            f"roll  = {np.degrees(roll):6.1f}°",
            f"pitch = {np.degrees(pitch):6.1f}°",
            f"yaw   = {np.degrees(yaw):6.1f}°",
            "",
            {"text":"PUNTO DE CÁMARA", "bold":True},
            f"^camera p = {fmt(p_camera)}",
            f"^base p   = {fmt(p_base)}",
            "",
            "^base p = ^base T_camera",
            "          · ^camera p",
        ],
        "phase":fase,
        "info_line_height":0.0395,
        "info_fontsize":8.7,
        "legend":[
            {"kind":"line","label":"traslación de la cámara","color":"#E07A1F"},
            {"kind":"point","label":"punto transformado a base","color":"#7B2CBF"},
        ],
        "legend_fontsize":8.0,
    }


def crear_estados_demostracion():
    estados=[]
    for _ in range(30):
        estados.append(crear_estado(0.0,"1/3 · Posición + orientación","Una pose combina dónde está un origen y cómo están orientados sus ejes. La cámara parte coincidiendo con base_link."))
    for p in np.linspace(0.0,1.0,130):
        estados.append(crear_estado(p,"2/3 · Construir ^base T_camera","La cámara adquiere simultáneamente una posición y una orientación relativas. Ambas quedan agrupadas en una sola matriz homogénea."))
    for _ in range(75):
        estados.append(crear_estado(1.0,"3/3 · Transformar una detección","Un punto medido en camera_link se expresa en base_link multiplicando por ^base T_camera. La pose es una relación entre frames, no una posición absoluta."))
    return {"states":estados}


def main():
    resultado=crear_estados_demostracion()
    animador=TransformAnimator(figsize=(15.8,8.9), interval=50)
    image_path=MATRICES_DIR/"assets"/"05_frames_pose"/"05_pose_posicion_orientacion.png"
    video_path=MATRICES_DIR/"assets"/"05_frames_pose"/"05_pose_posicion_orientacion.webm"
    animador.animate_3d_states(
        states=resultado["states"], title="5.5. Pose: posición y orientación en una sola matriz",
        limits=(-1.6,3.4,-2.1,2.2,-1.0,2.8), view=(23.0,-58.0),
        final_image_path=image_path, video_path=video_path, repeat=False, fps=20, dpi=125, show=True)


if __name__=="__main__":
    main()
