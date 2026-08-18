from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


BOX_FACES = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]


def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])


def ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0.0,s],[0.0,1.0,0.0],[-s,0.0,c]])


def suavizar(p):
    return 0.5 - 0.5 * np.cos(np.pi * p)


def T(R, t):
    M = np.eye(4)
    M[:3,:3] = R
    M[:3,3] = t
    return M


def invT(M):
    R = M[:3,:3]
    t = M[:3,3]
    N = np.eye(4)
    N[:3,:3] = R.T
    N[:3,3] = -R.T @ t
    return N


def transformar_punto(M, p):
    return (M @ np.r_[p,1.0])[:3]


def caja_local(size):
    sx, sy, sz = np.asarray(size, dtype=float) / 2.0
    return np.array([[-sx,-sy,-sz],[sx,-sy,-sz],[sx,sy,-sz],[-sx,sy,-sz],[-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]])


def transformar_vertices(vertices, M):
    h = np.c_[vertices, np.ones(len(vertices))]
    return (M @ h.T).T[:,:3]


def fmt(v):
    return "[" + ", ".join(f"{x:5.2f}" for x in np.asarray(v)) + "]"


def crear_estado(progreso, fase, mensaje):
    s = suavizar(progreso)

    T_map_odom = T(rz(np.radians(6.0)), np.array([0.45,-0.35,0.0]))
    R_odom_base = rz(s * np.radians(42.0))
    t_odom_base = s * np.array([3.0, 1.15, 0.25])
    T_odom_base = T(R_odom_base, t_odom_base)
    T_map_base = T_map_odom @ T_odom_base

    T_base_camera = T(ry(np.radians(-12.0)), np.array([0.48,0.0,0.48]))
    T_base_laser = T(np.eye(3), np.array([0.18,0.0,0.30]))
    T_map_camera = T_map_base @ T_base_camera
    T_map_laser = T_map_base @ T_base_laser

    P_map = np.array([4.7, 2.0, 0.85])
    P_base = transformar_punto(invT(T_map_base), P_map)
    P_camera = transformar_punto(invT(T_map_camera), P_map)

    robot = transformar_vertices(caja_local((1.15,0.72,0.40)), T_map_base @ T(np.eye(3), np.array([0.0,0.0,0.20])))

    return {
        "frames3d": [
            {"name":"map","origin":np.zeros(3),"rotation":np.eye(3),"length":1.25,"alpha":0.65},
            {"name":"odom","origin":T_map_odom[:3,3],"rotation":T_map_odom[:3,:3],"length":1.15,"alpha":0.45,
             "colors":("#9CA3AF","#9CA3AF","#9CA3AF")},
            {"name":"base_link","origin":T_map_base[:3,3],"rotation":T_map_base[:3,:3],"length":1.10,"alpha":1.0},
            {"name":"camera","origin":T_map_camera[:3,3],"rotation":T_map_camera[:3,:3],"length":0.72,"alpha":1.0,
             "colors":("#D97706","#0F766E","#2563EB")},
            {"name":"laser","origin":T_map_laser[:3,3],"rotation":T_map_laser[:3,:3],"length":0.62,"alpha":0.95,
             "colors":("#B23A48","#2D7F5E","#7B2CBF")},
        ],
        "meshes3d": [
            {"vertices":robot,"faces":BOX_FACES,"facecolor":"#93C5FD","edgecolor":"#1D4ED8","alpha":0.28,"linewidth":1.1},
        ],
        "points3d": [
            {"name":"P físico","position":P_map,"color":"#7B2CBF","size":82},
        ],
        "segments3d": [
            {"start":T_map_base[:3,3],"end":T_map_camera[:3,3],"color":"#D97706","alpha":0.65,"linestyle":"--"},
            {"start":T_map_base[:3,3],"end":T_map_laser[:3,3],"color":"#B23A48","alpha":0.65,"linestyle":"--"},
        ],
        "message": mensaje,
        "info_title": "Frames de un robot móvil",
        "info_lines": [
            {"text":"JERARQUÍA", "bold":True},
            "map -> odom -> base_link",
            "base_link -> camera",
            "base_link -> laser",
            "",
            {"text":"MISMO PUNTO FÍSICO", "bold":True},
            f"p_map    = {fmt(P_map)}",
            f"p_base   = {fmt(P_base)}",
            f"p_camera = {fmt(P_camera)}",
            "",
            "map y odom son referencias",
            "globales en esta demo.",
            "Los sensores viajan",
            "solidarios con base_link.",
        ],
        "phase": fase,
        "info_line_height":0.041,
        "info_fontsize":8.8,
        "legend":[
            {"kind":"point","label":"mismo punto físico P","color":"#7B2CBF"},
            {"kind":"line","label":"robot/base_link","color":"#1D4ED8"},
        ],
        "legend_fontsize":8.0,
    }


def crear_estados_demostracion():
    estados=[]
    for _ in range(30):
        estados.append(crear_estado(0.0,"1/3 · Jerarquía inicial","Un frame combina origen y orientación. Los datos de cada sensor son naturales en su propio sistema de referencia."))
    for p in np.linspace(0.0,1.0,135):
        estados.append(crear_estado(p,"2/3 · El robot se mueve","base_link se mueve respecto a odom; camera y laser conservan su montaje relativo y se desplazan con el robot."))
    for _ in range(65):
        estados.append(crear_estado(1.0,"3/3 · Mismo punto, distintas coordenadas","El punto físico P no cambia, pero sus números dependen del frame desde el que se expresa."))
    return {"states":estados}


def main():
    resultado=crear_estados_demostracion()
    animador=TransformAnimator(figsize=(15.8,8.9), interval=50)
    image_path=MATRICES_DIR/"assets"/"05_frames_pose"/"02_sistemas_referencia_frames.png"
    video_path=MATRICES_DIR/"assets"/"05_frames_pose"/"02_sistemas_referencia_frames.webm"
    animador.animate_3d_states(
        states=resultado["states"], title="5.2. Qué es un sistema de referencia o frame",
        limits=(-2.0,6.4,-3.0,4.3,-1.1,3.8), view=(24.0,-58.0),
        final_image_path=image_path, video_path=video_path, repeat=False, fps=20, dpi=125, show=True)


if __name__=="__main__":
    main()
