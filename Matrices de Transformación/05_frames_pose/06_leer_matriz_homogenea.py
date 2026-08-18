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
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=t; return M


def fmt_row(row):
    return "["+", ".join(f"{x:5.2f}" for x in np.asarray(row))+"]"


def crear_estado(paso, fase, mensaje):
    R=rz(np.radians(40.0))@ry(np.radians(-25.0))@rx(np.radians(15.0))
    t=np.array([2.2,-1.0,1.4])
    M=T(R,t)

    vectores=[]
    if paso>=1:
        vectores.append({"name":"t = origen B","origin":np.zeros(3),"value":t,"color":"#E07A1F","linewidth":3.0})
    if paso>=2:
        vectores.append({"name":"col 1 = x_B","origin":t,"value":R[:,0]*1.7,"color":"#C63C3C","linewidth":3.0})
    if paso>=3:
        vectores.append({"name":"col 2 = y_B","origin":t,"value":R[:,1]*1.7,"color":"#2A8F5B","linewidth":3.0})
    if paso>=4:
        vectores.append({"name":"col 3 = z_B","origin":t,"value":R[:,2]*1.7,"color":"#1F77B4","linewidth":3.0})

    frames=[{"name":"A","origin":np.zeros(3),"rotation":np.eye(3),"length":1.25,"alpha":0.30,
             "colors":("#9CA3AF","#9CA3AF","#9CA3AF")}]
    if paso>=5:
        frames.append({"name":"B","origin":t,"rotation":R,"length":1.70,"alpha":0.80})

    errores_ort=np.linalg.norm(R.T@R-np.eye(3))
    normas=[np.linalg.norm(R[:,i]) for i in range(3)]

    return {
        "frames3d":frames,
        "vectors3d":vectores,
        "points3d":[{"name":"origen B","position":t,"color":"#7B2CBF","size":72}] if paso>=1 else [],
        "message":mensaje,
        "info_title":"Leer una matriz homogénea",
        "info_lines":[
            {"text":"^A T_B", "bold":True},
            fmt_row(M[0]), fmt_row(M[1]), fmt_row(M[2]), fmt_row(M[3]),
            "",
            {"text":"LECTURA", "bold":True},
            "col 1 -> eje x_B en A",
            "col 2 -> eje y_B en A",
            "col 3 -> eje z_B en A",
            "col 4 -> origen B en A",
            "",
            {"text":"VALIDACIÓN", "bold":True},
            f"||R^T R-I|| = {errores_ort:.2e}",
            f"normas = [{normas[0]:.3f},{normas[1]:.3f},{normas[2]:.3f}]",
            f"det(R) = {np.linalg.det(R):.6f}",
            "última fila = [0,0,0,1]",
        ],
        "phase":fase,
        "info_line_height":0.0365,
        "info_fontsize":8.3,
        "legend":[
            {"kind":"line","label":"columna de traslación","color":"#E07A1F"},
            {"kind":"line","label":"columna 1 / x_B","color":"#C63C3C"},
            {"kind":"line","label":"columna 2 / y_B","color":"#2A8F5B"},
            {"kind":"line","label":"columna 3 / z_B","color":"#1F77B4"},
        ] if paso>=4 else [],
        "legend_ncol":2,
        "legend_fontsize":7.7,
    }


def crear_estados_demostracion():
    estados=[]
    fases=[
        (0,"1/6 · Matriz dada","Partimos de una matriz homogénea concreta. La leeremos geométricamente sin aplicar todavía ningún punto."),
        (1,"2/6 · Cuarta columna","La cuarta columna contiene t: la posición del origen de {B} expresada en {A}."),
        (2,"3/6 · Primera columna","La primera columna de R es el eje x_B expresado en coordenadas de {A}."),
        (3,"4/6 · Segunda columna","La segunda columna de R es el eje y_B expresado en coordenadas de {A}."),
        (4,"5/6 · Tercera columna","La tercera columna de R es el eje z_B expresado en coordenadas de {A}."),
        (5,"6/6 · Pose y validez","Las tres columnas de R y la columna t reconstruyen el frame {B}. Las comprobaciones confirman que se trata de una pose rígida válida."),
    ]
    duraciones=[30,40,40,40,40,75]
    for (paso,fase,mensaje),n in zip(fases,duraciones):
        for _ in range(n):
            estados.append(crear_estado(paso,fase,mensaje))
    return {"states":estados}


def main():
    resultado=crear_estados_demostracion()
    animador=TransformAnimator(figsize=(15.8,8.9), interval=50)
    image_path=MATRICES_DIR/"assets"/"05_frames_pose"/"06_leer_matriz_homogenea.png"
    video_path=MATRICES_DIR/"assets"/"05_frames_pose"/"06_leer_matriz_homogenea.webm"
    animador.animate_3d_states(
        states=resultado["states"], title="5.6. Cómo leer geométricamente una matriz homogénea",
        limits=(-2.3,5.0,-3.5,3.0,-1.8,4.0), view=(24.0,-58.0),
        final_image_path=image_path, video_path=video_path, repeat=False, fps=20, dpi=125, show=True)


if __name__=="__main__":
    main()
