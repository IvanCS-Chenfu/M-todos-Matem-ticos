from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])

def ry(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,0.0,s],[0.0,1.0,0.0],[-s,0.0,c]])

def T3(R,t):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=np.asarray(t); return M

def invT(M):
    R=M[:3,:3]; t=M[:3,3]
    N=np.eye(4); N[:3,:3]=R.T; N[:3,3]=-R.T@t
    return N

def transformar3(M,p):
    return (M@np.r_[p,1.0])[:3]

def suavizar(p):
    return 0.5-0.5*np.cos(np.pi*p)

def fmt(v):
    return "["+", ".join(f"{x:5.2f}" for x in np.asarray(v))+"]"

def datos_2d():
    H_AB=np.array([[0.0,-1.0,2.0],[1.0,0.0,1.0],[0.0,0.0,1.0]])
    p_B=np.array([1.0,2.0,1.0])
    p_A=H_AB@p_B
    H_BA=np.linalg.inv(H_AB)
    p_B_rec=H_BA@p_A
    return H_AB,p_B,p_A,H_BA,p_B_rec

def crear_estado_2d(s, vuelta, fase, mensaje):
    H_AB,p_B,p_A,H_BA,p_B_rec=datos_2d()
    # Representamos el ejemplo 2D en el plano z=0 usando la escena 3D común.
    theta=np.pi/2
    if not vuelta:
        ang=s*theta
        t=s*np.array([2.0,1.0,0.0])
    else:
        ang=(1.0-s)*theta
        t=(1.0-s)*np.array([2.0,1.0,0.0])
    R=rz(ang)

    # El punto físico final de la Wiki está en A: [0,2].
    p_phys=np.array([p_A[0],p_A[1],0.0])
    return {
        "frames3d":[
            {"name":"A","origin":np.zeros(3),"rotation":np.eye(3),"length":1.25,"alpha":0.55},
            {"name":"B","origin":t,"rotation":R,"length":1.25,"alpha":1.0,
             "colors":("#D97706","#0F766E","#2563EB")},
        ],
        "points3d":[
            {"name":"p físico","position":p_phys,"color":"#7B2CBF","size":82},
        ],
        "segments3d":[
            {"start":t,"end":p_phys,"color":"#7B2CBF","alpha":0.55,"linestyle":"--"},
        ],
        "message":mensaje,
        "info_title":"Ejemplo 2D exacto de la Wiki",
        "info_lines":[
            {"text":"^A T_B","bold":True},
            "[ 0, -1, 2]",
            "[ 1,  0, 1]",
            "[ 0,  0, 1]",
            "",
            f"^B p = {fmt(p_B)}",
            f"^A p = {fmt(p_A)}",
            "",
            {"text":"IDA / VUELTA","bold":True},
            f"^B p recuperado = {fmt(p_B_rec)}",
            f"error = {np.linalg.norm(p_B-p_B_rec):.2e}",
        ],
        "phase":fase,
        "info_line_height":0.043,
        "info_fontsize":8.8,
    }

def crear_estado_3d(s, fase, mensaje):
    R_AB=rz(np.radians(35))
    t_AB=np.array([2.0,0.7,0.5])
    R_BC=ry(np.radians(-30))
    t_BC=np.array([1.1,-0.15,0.9])

    # Aparecen progresivamente las dos relaciones.
    T_AB=T3(rz(s*np.radians(35)), s*t_AB)
    T_BC=T3(ry(s*np.radians(-30)), s*t_BC)
    T_AC=T_AB@T_BC

    p_C=np.array([0.8,-0.3,0.45])
    p_A=transformar3(T_AC,p_C)
    p_C_rec=transformar3(invT(T_AC),p_A)

    oA=np.zeros(3); oB=T_AB[:3,3]; oC=T_AC[:3,3]
    return {
        "frames3d":[
            {"name":"A","origin":oA,"rotation":np.eye(3),"length":1.25,"alpha":0.55},
            {"name":"B","origin":oB,"rotation":T_AB[:3,:3],"length":1.20,"alpha":1.0},
            {"name":"C","origin":oC,"rotation":T_AC[:3,:3],"length":1.10,"alpha":1.0,
             "colors":("#D97706","#0F766E","#2563EB")},
        ],
        "segments3d":[
            {"start":oA,"end":oB,"color":"#7B2CBF","alpha":0.55,"linestyle":"--"},
            {"start":oB,"end":oC,"color":"#E07A1F","alpha":0.65,"linestyle":"--"},
            {"start":oC,"end":p_A,"color":"#2D7F5E","alpha":0.55,"linestyle":"--"},
        ],
        "points3d":[{"name":"p","position":p_A,"color":"#7B2CBF","size":82}],
        "message":mensaje,
        "info_title":"Extensión 3D con tres frames",
        "info_lines":[
            {"text":"CADENA","bold":True},
            "^A T_C = ^A T_B ^B T_C",
            "",
            f"^C p = {fmt(p_C)}",
            f"^A p = {fmt(p_A)}",
            f"vuelta = {fmt(p_C_rec)}",
            "",
            {"text":"COMPROBACIONES","bold":True},
            f"error ida/vuelta = {np.linalg.norm(p_C-p_C_rec):.2e}",
            f"||T_AC inv(T_AC)-I|| = {np.linalg.norm(T_AC@invT(T_AC)-np.eye(4)):.2e}",
            "",
            "La misma lógica escala",
            "de 2D a cadenas 3D.",
        ],
        "phase":fase,
        "info_line_height":0.042,
        "info_fontsize":8.7,
        "legend":[
            {"kind":"line","label":"A -> B","color":"#7B2CBF"},
            {"kind":"line","label":"B -> C","color":"#E07A1F"},
        ],
        "legend_fontsize":8.0,
    }

def crear_estados_demostracion():
    estados=[]
    for _ in range(28):
        estados.append(crear_estado_2d(0.0,False,"1/5 · Datos 2D",
            "Reproducimos exactamente la matriz ^A T_B y el punto ^B p=[1,2,1]^T del ejemplo de la Wiki."))
    for p in np.linspace(0,1,90):
        estados.append(crear_estado_2d(suavizar(p),False,"2/5 · Transformar B -> A",
            "El frame {B} alcanza una rotación de 90° y un origen (2,1). El producto homogéneo da ^A p=[0,2,1]^T."))
    for _ in range(45):
        estados.append(crear_estado_2d(1.0,False,"3/5 · Aplicar la inversa",
            "La matriz ^B T_A invierte la relación. Al aplicarla a [0,2,1]^T recuperamos exactamente [1,2,1]^T."))
    for p in np.linspace(0,1,80):
        estados.append(crear_estado_3d(suavizar(p),"4/5 · Extensión a tres frames 3D",
            "Extendemos la misma lógica a {A}, {B} y {C}: componemos dos matrices 4x4 y transformamos un punto desde C hasta A."))
    for _ in range(75):
        estados.append(crear_estado_3d(1.0,"5/5 · Verificación completa",
            "La composición, la transformación del punto y la vuelta con la inversa coinciden numéricamente. El método es idéntico para cadenas más largas."))
    return {"states":estados}

def main():
    resultado=crear_estados_demostracion()
    H_AB,p_B,p_A,H_BA,p_B_rec=datos_2d()
    print("\n=== 6.6. Ejemplo numérico 2D de la Wiki ===")
    print("^A T_B =\n", H_AB)
    print("^B p =", p_B)
    print("^A p =", p_A)
    print("^B T_A =\n", H_BA)
    print("p recuperado =", p_B_rec)

    animador=TransformAnimator(figsize=(15.8,8.9), interval=50)
    image_path=MATRICES_DIR/"assets"/"06_composicion_inversion"/"06_ejemplo_numerico_frames.png"
    video_path=MATRICES_DIR/"assets"/"06_composicion_inversion"/"06_ejemplo_numerico_frames.webm"
    animador.animate_3d_states(
        states=resultado["states"], title="6.6. Ejemplo numérico completo con varios frames",
        limits=(-2.0,5.7,-2.6,3.9,-1.3,3.8), view=(24,-58),
        final_image_path=image_path, video_path=video_path, repeat=False, fps=20, dpi=125, show=True)

if __name__=="__main__":
    main()
