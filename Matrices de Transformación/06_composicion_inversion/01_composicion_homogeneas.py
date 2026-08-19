from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0,0.0,0.0],[0.0,c,-s],[0.0,s,c]])

def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0.0,s],[0.0,1.0,0.0],[-s,0.0,c]])

def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])

def T(R, t):
    M = np.eye(4)
    M[:3,:3] = R
    M[:3,3] = np.asarray(t, dtype=float)
    return M

def suavizar(p):
    return 0.5 - 0.5*np.cos(np.pi*p)

def interp_pose(s, R_final, t_final, axis):
    if axis == "z":
        ang = np.arctan2(R_final[1,0], R_final[0,0])
        R = rz(s*ang)
    elif axis == "y":
        ang = np.arctan2(R_final[0,2], R_final[0,0])
        R = ry(s*ang)
    else:
        ang = np.arctan2(R_final[2,1], R_final[1,1])
        R = rx(s*ang)
    return T(R, s*np.asarray(t_final, dtype=float))

def transformar(M, p):
    return (M @ np.r_[p, 1.0])[:3]

def fmt(v):
    return "[" + ", ".join(f"{x:5.2f}" for x in np.asarray(v)) + "]"

def crear_estado(s_ab, s_bc, fase, mensaje):
    R_ab_f = rz(np.radians(32.0))
    t_ab_f = np.array([2.2, 0.7, 0.5])
    R_bc_f = ry(np.radians(-28.0))
    t_bc_f = np.array([1.35, 0.20, 0.85])

    T_ab = interp_pose(s_ab, R_ab_f, t_ab_f, "z")
    T_bc = interp_pose(s_bc, R_bc_f, t_bc_f, "y")
    T_ac = T_ab @ T_bc

    R_ab, t_ab = T_ab[:3,:3], T_ab[:3,3]
    R_bc, t_bc = T_bc[:3,:3], T_bc[:3,3]
    R_ac, t_ac = T_ac[:3,:3], T_ac[:3,3]

    # Para conectar con la fórmula de bloques de la Wiki:
    # T2 = ^A T_B, T1 = ^B T_C  ->  T2 T1 = ^A T_C
    R_bloques = R_ab @ R_bc
    t_bloques = R_ab @ t_bc + t_ab

    p_c = np.array([0.75, -0.25, 0.45])
    p_a = transformar(T_ac, p_c)

    o_a = np.zeros(3)
    o_b = t_ab
    o_c = t_ac

    return {
        "frames3d": [
            {"name":"A","origin":o_a,"rotation":np.eye(3),"length":1.35,"alpha":0.65},
            {"name":"B","origin":o_b,"rotation":R_ab,"length":1.25,"alpha":1.0},
            {"name":"C","origin":o_c,"rotation":R_ac,"length":1.15,"alpha":1.0,
             "colors":("#D97706","#0F766E","#2563EB")},
        ],
        "segments3d": [
            {"start":o_a,"end":o_b,"color":"#7B2CBF","alpha":0.55,"linestyle":"--"},
            {"start":o_b,"end":o_c,"color":"#E07A1F","alpha":0.65,"linestyle":"--"},
        ],
        "points3d": [
            {"name":"p en A","position":p_a,"color":"#7B2CBF","size":80},
        ],
        "message": mensaje,
        "info_title": "Composición homogénea",
        "info_lines": [
            {"text":"CADENA DE FRAMES","bold":True},
            "^A T_C = ^A T_B  ^B T_C",
            "",
            {"text":"DESARROLLO POR BLOQUES","bold":True},
            "T2 T1 = [R2 R1 | R2 t1+t2]",
            "",
            f"R error = {np.linalg.norm(R_ac-R_bloques):.2e}",
            f"t comp  = {fmt(t_ac)}",
            f"t bloque= {fmt(t_bloques)}",
            f"t error = {np.linalg.norm(t_ac-t_bloques):.2e}",
            "",
            f"^C p = {fmt(p_c)}",
            f"^A p = {fmt(p_a)}",
        ],
        "phase": fase,
        "info_line_height": 0.0405,
        "info_fontsize": 8.7,
        "legend": [
            {"kind":"line","label":"^A T_B","color":"#7B2CBF"},
            {"kind":"line","label":"^B T_C","color":"#E07A1F"},
            {"kind":"point","label":"p transformado","color":"#7B2CBF"},
        ],
        "legend_fontsize": 8.0,
    }

def crear_estados_demostracion():
    estados = []
    for _ in range(28):
        estados.append(crear_estado(0.0,0.0,"1/4 · Frame A",
            "Partimos del frame {A}. Construiremos dos relaciones homogéneas y las concatenaremos mediante un producto matricial."))
    for p in np.linspace(0.0,1.0,90):
        estados.append(crear_estado(suavizar(p),0.0,"2/4 · Construir ^A T_B",
            "La primera matriz sitúa {B} respecto a {A}: contiene simultáneamente su orientación y la posición de su origen."))
    for p in np.linspace(0.0,1.0,100):
        estados.append(crear_estado(1.0,suavizar(p),"3/4 · Añadir ^B T_C",
            "La segunda transformación se define localmente desde {B}. Al multiplicar ^A T_B por ^B T_C obtenemos directamente ^A T_C."))
    for _ in range(70):
        estados.append(crear_estado(1.0,1.0,"4/4 · Resultado por bloques",
            "La multiplicación 4x4 reproduce exactamente R2 R1 y R2 t1+t2. Una sola matriz contiene la transformación compuesta."))
    return {"states":estados}

def main():
    resultado = crear_estados_demostracion()
    animador = TransformAnimator(figsize=(15.8,8.9), interval=50)
    image_path = MATRICES_DIR/"assets"/"06_composicion_inversion"/"01_composicion_homogeneas.png"
    video_path = MATRICES_DIR/"assets"/"06_composicion_inversion"/"01_composicion_homogeneas.webm"
    animador.animate_3d_states(
        states=resultado["states"], title="6.1. Composición de transformaciones homogéneas",
        limits=(-1.8,5.8,-2.8,3.8,-1.5,4.0), view=(24.0,-58.0),
        final_image_path=image_path, video_path=video_path, repeat=False, fps=20, dpi=125, show=True)

if __name__ == "__main__":
    main()
