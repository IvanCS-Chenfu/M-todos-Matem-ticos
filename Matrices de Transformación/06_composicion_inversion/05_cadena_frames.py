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

def T(R,t):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=np.asarray(t); return M

def suavizar(p):
    return 0.5-0.5*np.cos(np.pi*p)

def transformar(M,p):
    return (M@np.r_[p,1.0])[:3]

def caja(size):
    sx,sy,sz=np.asarray(size,dtype=float)/2
    return np.array([[-sx,-sy,-sz],[sx,-sy,-sz],[sx,sy,-sz],[-sx,sy,-sz],
                     [-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]])

def transformar_vertices(vertices,M):
    h=np.c_[vertices,np.ones(len(vertices))]
    return (M@h.T).T[:,:3]

def fmt(v):
    return "["+", ".join(f"{x:5.2f}" for x in np.asarray(v))+"]"

def crear_estado(progreso, fase, mensaje):
    s=suavizar(progreso)

    T_world_base=T(rz(s*np.radians(42.0)), s*np.array([2.8,1.15,0.25]))
    # Cámara montada sobre un pequeño soporte orientable: también cambia respecto a base.
    T_base_camera=T(ry(np.radians(-10.0)-s*np.radians(12.0)),
                    np.array([0.48,0.0,0.48]))
    # El objeto está descrito localmente respecto a la cámara (por ejemplo, una detección).
    T_camera_obj=T(rz(np.radians(18.0)), np.array([1.55,0.30,0.15]))

    T_world_camera=T_world_base@T_base_camera
    T_world_obj=T_world_camera@T_camera_obj

    p_camera=np.array([1.35,-0.20,0.10])
    p_world=transformar(T_world_camera,p_camera)

    robot=transformar_vertices(caja((1.15,0.75,0.38)),
                               T_world_base@T(np.eye(3),np.array([0,0,0.19])))
    objeto=transformar_vertices(caja((0.35,0.35,0.35)),
                                T_world_obj@T(np.eye(3),np.zeros(3)))

    ow=np.zeros(3); ob=T_world_base[:3,3]; oc=T_world_camera[:3,3]; oo=T_world_obj[:3,3]
    return {
        "frames3d":[
            {"name":"world","origin":ow,"rotation":np.eye(3),"length":1.30,"alpha":0.55},
            {"name":"base_link","origin":ob,"rotation":T_world_base[:3,:3],"length":1.05,"alpha":1.0},
            {"name":"camera_link","origin":oc,"rotation":T_world_camera[:3,:3],"length":0.75,"alpha":1.0,
             "colors":("#D97706","#0F766E","#2563EB")},
            {"name":"obj","origin":oo,"rotation":T_world_obj[:3,:3],"length":0.55,"alpha":0.90,
             "colors":("#B23A48","#2D7F5E","#7B2CBF")},
        ],
        "meshes3d":[
            {"vertices":robot,"faces":BOX_FACES,"facecolor":"#93C5FD","edgecolor":"#1D4ED8","alpha":0.28,"linewidth":1.1},
            {"vertices":objeto,"faces":BOX_FACES,"facecolor":"#F8CFA7","edgecolor":"#C2410C","alpha":0.42,"linewidth":1.1},
        ],
        "segments3d":[
            {"start":ow,"end":ob,"color":"#7B2CBF","alpha":0.45,"linestyle":"--"},
            {"start":ob,"end":oc,"color":"#E07A1F","alpha":0.65,"linestyle":"--"},
            {"start":oc,"end":oo,"color":"#2D7F5E","alpha":0.70,"linestyle":"--"},
        ],
        "points3d":[
            {"name":"p detectado en world","position":p_world,"color":"#7B2CBF","size":80},
        ],
        "message":mensaje,
        "info_title":"Cadena world -> base -> camera -> obj",
        "info_lines":[
            {"text":"POSE DEL OBJETO","bold":True},
            "^world T_obj =",
            "^world T_base · ^base T_camera",
            "               · ^camera T_obj",
            "",
            f"origen obj = {fmt(oo)}",
            "",
            {"text":"PUNTO DETECTADO","bold":True},
            f"^camera p = {fmt(p_camera)}",
            f"^world p  = {fmt(p_world)}",
            "",
            "^world p = ^world T_base",
            "          ^base T_camera",
            "          ^camera p",
            "",
            "TF2 automatizará esta idea.",
        ],
        "phase":fase,
        "info_line_height":0.0365,
        "info_fontsize":8.4,
        "legend":[
            {"kind":"line","label":"world -> base","color":"#7B2CBF"},
            {"kind":"line","label":"base -> camera","color":"#E07A1F"},
            {"kind":"line","label":"camera -> objeto","color":"#2D7F5E"},
        ],
        "legend_fontsize":7.8,
    }

def crear_estados_demostracion():
    estados=[]
    for _ in range(30):
        estados.append(crear_estado(0.0,"1/3 · Cadena inicial",
            "Cada elemento solo conoce una relación local: world->base_link, base_link->camera_link y camera_link->objeto."))
    for p in np.linspace(0,1,150):
        estados.append(crear_estado(p,"2/3 · Actualizar la cadena",
            "El robot avanza y gira, y la cámara cambia su orientación relativa. Las matrices locales se recomponen para actualizar continuamente la pose global del objeto."))
    for _ in range(80):
        estados.append(crear_estado(1.0,"3/3 · Resultado global",
            "La cadena completa obtiene ^world T_obj y permite expresar en world un punto medido por la cámara. Esta es la operación matemática que TF2 encadena automáticamente."))
    return {"states":estados}

def main():
    resultado=crear_estados_demostracion()
    animador=TransformAnimator(figsize=(15.9,8.9), interval=50)
    image_path=MATRICES_DIR/"assets"/"06_composicion_inversion"/"05_cadena_frames.png"
    video_path=MATRICES_DIR/"assets"/"06_composicion_inversion"/"05_cadena_frames.webm"
    animador.animate_3d_states(
        states=resultado["states"], title="6.5. Cadena world → robot → sensor → objeto",
        limits=(-2.0,6.5,-3.0,4.5,-1.2,4.0), view=(24,-58),
        final_image_path=image_path, video_path=video_path, repeat=False, fps=20, dpi=125, show=True)

if __name__=="__main__":
    main()
