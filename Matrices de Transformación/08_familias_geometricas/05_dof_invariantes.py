from pathlib import Path
import sys
import numpy as np

CURRENT_DIR=Path(__file__).resolve().parent
MATRICES_DIR=CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator


def R2(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s],[s,c]],float)
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)
def aff(P,A): return (A@P.T).T
def hom(P,H):
    q=(H@np.c_[P,np.ones(len(P))].T).T; return q[:,:2]/q[:,2,None]


def crear_estado(progreso,fase,mensaje):
    s=suavizar(progreso)
    fig=np.array([[-0.9,-0.65],[0.9,-0.65],[1.05,0.15],[0.15,1.0],[-0.95,0.45]])
    origins=[np.array([-3.5,2.3]),np.array([3.0,2.3]),np.array([-3.5,-2.4]),np.array([3.0,-2.4])]
    # Rígida
    R=R2(s*np.radians(30)); rigid=aff(fig,R)+origins[0]
    # Similitud
    sim=(1+0.65*s)*aff(fig,R2(s*np.radians(-25)))+origins[1]
    # Afín
    A=R2(s*np.radians(15))@np.array([[1+0.4*s,1.1*s],[0,1-0.25*s]])
    afin=aff(fig,A)+origins[2]
    # Proyectiva local antes de desplazar.
    H=np.array([[1.,0.25*s,0.],[0.08*s,1.,0.],[0.30*s,-0.12*s,1.]])
    proj=hom(fig,H)+origins[3]
    polys=[
      {'points':rigid,'facecolor':'#BFDBFE','edgecolor':'#2563EB','alpha':0.38,'linewidth':2},
      {'points':sim,'facecolor':'#FED7AA','edgecolor':'#D97706','alpha':0.38,'linewidth':2},
      {'points':afin,'facecolor':'#BBF7D0','edgecolor':'#2D7F5E','alpha':0.38,'linewidth':2},
      {'points':proj,'facecolor':'#E9D5FF','edgecolor':'#7B2CBF','alpha':0.38,'linewidth':2},
    ]
    texts=[
      {'position':origins[0]+[-1.3,1.75],'text':'RÍGIDA · 3 DoF','fontweight':'bold','color':'#2563EB'},
      {'position':origins[1]+[-1.45,2.15],'text':'SIMILITUD · 4 DoF','fontweight':'bold','color':'#D97706'},
      {'position':origins[2]+[-1.25,1.75],'text':'AFÍN · 6 DoF','fontweight':'bold','color':'#2D7F5E'},
      {'position':origins[3]+[-1.55,1.75],'text':'PROYECTIVA · 8 DoF','fontweight':'bold','color':'#7B2CBF'},
    ]
    return {
      'polygons':polys,'texts':texts,'message':mensaje,'info_title':'Jerarquía e invariantes 2D',
      'info_lines':[
        {'text':'RÍGIDA · 3 DoF','bold':True},'rectas ✓  paralelas ✓','ángulos ✓ distancias ✓','',
        {'text':'SIMILITUD · 4 DoF','bold':True},'rectas ✓  paralelas ✓','ángulos ✓ distancias ×','',
        {'text':'AFÍN · 6 DoF','bold':True},'rectas ✓  paralelas ✓','ángulos × distancias ×','',
        {'text':'HOMOGRAFÍA · 8 DoF','bold':True},'rectas ✓  paralelas no siempre','ángulos × distancias ×','',
        'Rígida ⊂ Similitud ⊂ Afín ⊂ Proyectiva',
      ],'phase':fase,'info_line_height':0.0345,'info_fontsize':8.2,
    }

def crear_estados_demostracion():
    estados=[]
    for _ in range(30): estados.append(crear_estado(0,'1/2 · Misma figura','Partimos de cuatro copias idénticas. Cada familia añade grados de libertad y permite deformaciones que la anterior no puede representar.'))
    for p in np.linspace(0,1,150): estados.append(crear_estado(p,'2/2 · Aumentar generalidad','Aplicamos simultáneamente una rígida, una similitud, una afinidad y una homografía. A medida que aumenta la generalidad se conservan menos invariantes geométricos.'))
    for _ in range(95): estados.append(crear_estado(1,'Conclusión · Modelo mínimo suficiente','La jerarquía Rígida ⊂ Similitud ⊂ Afín ⊂ Proyectiva permite elegir el modelo menos general que todavía explica el problema.'))
    return {'states':estados}

def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.8,8.9),interval=50)
    a.animate_2d_states(r['states'],'8.5. Grados de libertad e invariantes geométricos',limits=(-6.0,6.3,-5.2,5.4),
      final_image_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'05_dof_invariantes.png',
      video_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'05_dof_invariantes.webm',repeat=False,fps=20,dpi=130,show=True)
if __name__=='__main__': main()
