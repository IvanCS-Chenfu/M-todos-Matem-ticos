from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))

from utils.transform_anim import TransformAnimator


def rx(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]],dtype=float)

def ry(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]],dtype=float)

def rz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=float)

def T(R,t):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=t; return M

def inversa_bloques(M):
    R=M[:3,:3]; t=M[:3,3]
    N=np.eye(4)
    N[:3,:3]=R.T
    N[:3,3]=-R.T@t
    return N

def suavizar(p):
    return 0.5-0.5*np.cos(np.pi*p)

def fmt(v):
    return "["+", ".join(f"{x:5.2f}" for x in np.asarray(v))+"]"

def crear_transformacion_valida():
    # Valores pseudoaleatorios reproducibles: la demo siempre produce la misma pose.
    rng=np.random.default_rng(27)
    angulos=np.radians(rng.uniform([-35,-25,-45],[35,25,45]))
    R=rz(angulos[2])@ry(angulos[1])@rx(angulos[0])
    t=rng.uniform([-2.3,-1.5,0.4],[2.3,1.5,2.0])
    return T(R,t), angulos

def crear_estado(progreso, vuelta, fase, mensaje):
    M, _=crear_transformacion_valida()
    Minv=inversa_bloques(M)
    Minv_np=np.linalg.inv(M)

    oA=np.zeros(3)
    oB=M[:3,3]
    s=suavizar(progreso)

    if vuelta:
        origen_flecha=oB
        valor_flecha=s*(oA-oB)
        nombre_flecha="B -> A"
        color="#E07A1F"
    else:
        origen_flecha=oA
        valor_flecha=s*(oB-oA)
        nombre_flecha="A -> B"
        color="#7B2CBF"

    return {
        "frames3d":[
            {"name":"A","origin":oA,"rotation":np.eye(3),"length":1.35,"alpha":0.55},
            {"name":"B","origin":oB,"rotation":M[:3,:3],"length":1.35,"alpha":1.0,
             "colors":("#D97706","#0F766E","#2563EB")},
        ],
        "segments3d":[
            {"start":oA,"end":oB,"color":"#6B7280","alpha":0.25,"linestyle":"--","linewidth":1.4},
        ],
        "vectors3d":[
            {"name":nombre_flecha,"origin":origen_flecha,"value":valor_flecha,
             "color":color,"linewidth":3.0,"show_origin":False},
        ] if np.linalg.norm(valor_flecha)>1e-9 else [],
        "message":mensaje,
        "info_title":"Inversa de una transformación rígida",
        "info_lines":[
            {"text":"FÓRMULA","bold":True},
            "T^-1 = [ R^T | -R^T t ]",
            "",
            f"t       = {fmt(M[:3,3])}",
            f"-R^T t  = {fmt(Minv[:3,3])}",
            "",
            {"text":"COMPROBACIONES","bold":True},
            f"||Tinv - np.inv(T)|| = {np.linalg.norm(Minv-Minv_np):.2e}",
            f"||T Tinv-I||         = {np.linalg.norm(M@Minv-np.eye(4)):.2e}",
            f"||Tinv T-I||         = {np.linalg.norm(Minv@M-np.eye(4)):.2e}",
            "",
            "Invertir = recorrer la",
            "relación en sentido contrario.",
        ],
        "phase":fase,
        "info_line_height":0.042,
        "info_fontsize":8.7,
        "legend":[
            {"kind":"line","label":"^A T_B","color":"#7B2CBF"},
            {"kind":"line","label":"^B T_A = (^A T_B)^-1","color":"#E07A1F"},
        ],
        "legend_fontsize":8.0,
    }

def crear_estados_demostracion():
    estados=[]
    for _ in range(28):
        estados.append(crear_estado(0.0,False,"1/4 · Dos frames",
            "Construimos una transformación rígida válida y reproducible. ^A T_B relaciona el frame {B} con {A}."))
    for p in np.linspace(0,1,100):
        estados.append(crear_estado(p,False,"2/4 · Recorrer A -> B",
            "Visualizamos el sentido asociado a ^A T_B. La matriz agrupa la orientación R y la traslación t."))
    for _ in range(45):
        estados.append(crear_estado(1.0,False,"3/4 · Construir T^-1",
            "La inversa usa R^T y no simplemente -t: la traslación inversa debe expresarse en el frame invertido como -R^T t."))
    for p in np.linspace(0,1,100):
        estados.append(crear_estado(p,True,"4/4 · Recorrer B -> A",
            "La transformación inversa recorre la misma relación en sentido contrario. T·T^-1 y T^-1·T recuperan la identidad."))
    for _ in range(55):
        estados.append(crear_estado(1.0,True,"Conclusión",
            "La fórmula por bloques coincide con numpy.linalg.inv y las dos composiciones con la inversa producen la identidad salvo error numérico."))
    return {"states":estados}

def main():
    resultado=crear_estados_demostracion()
    animador=TransformAnimator(figsize=(15.7,8.9), interval=50)
    image_path=MATRICES_DIR/"assets"/"06_composicion_inversion"/"03_inversa_transformacion_rigida.png"
    video_path=MATRICES_DIR/"assets"/"06_composicion_inversion"/"03_inversa_transformacion_rigida.webm"
    animador.animate_3d_states(
        states=resultado["states"], title="6.3. Inversa de una transformación rígida",
        limits=(-3.8,4.0,-3.1,3.2,-1.8,3.7), view=(24,-58),
        final_image_path=image_path, video_path=video_path, repeat=False, fps=20, dpi=125, show=True)

if __name__=="__main__":
    main()
