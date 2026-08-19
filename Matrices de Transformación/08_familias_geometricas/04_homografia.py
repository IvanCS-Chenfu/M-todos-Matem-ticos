from pathlib import Path
import sys
import numpy as np

CURRENT_DIR=Path(__file__).resolve().parent
MATRICES_DIR=CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator


def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)
def H_proj(h31): return np.array([[1.,0.,0.],[0.,1.,0.],[h31,0.,1.]])
def aplicar_h(P,H):
    P=np.asarray(P,float); ph=np.c_[P,np.ones(len(P))]; q=(H@ph.T).T
    return q[:,:2]/q[:,2,None], q[:,2]
def cuadricula(xmin=0,xmax=35,ymin=-9,ymax=9,nx=8,ny=7,m=70):
    ls=[]
    for x in np.linspace(xmin,xmax,nx): ls.append(np.c_[np.full(m,x),np.linspace(ymin,ymax,m)])
    for y in np.linspace(ymin,ymax,ny): ls.append(np.c_[np.linspace(xmin,xmax,m),np.full(m,y)])
    return ls


def crear_estado(progreso,fase,mensaje):
    s=suavizar(progreso); h31=0.1*s; H=H_proj(h31)
    lines=[]
    for g in cuadricula():
        q,_=aplicar_h(g,H); lines.append({'points':q,'color':'#2563EB','linewidth':1.35,'alpha':0.58})
    p=np.array([[10.,5.]])
    qp,w=aplicar_h(p,H); ph=np.array([10.,5.,1.]); qh=H@ph
    van=np.array([10.,0.]) if h31>1e-9 else None
    points=[{'name':'p´','position':qp[0],'color':'#7B2CBF','size':92}]
    if van is not None: points.append({'name':'punto de fuga','position':van,'color':'#E07A1F','size':72})
    return {
      'polylines':lines,'points':points,
      'message':mensaje,'info_title':'Homografía y división por w',
      'info_lines':[
        {'text':'H ACTUAL','bold':True},'[1, 0, 0]','[0, 1, 0]',f'[{h31:4.2f}, 0, 1]','',
        {'text':'EJEMPLO DE LA WIKI','bold':True},'p_h = [10, 5, 1]',f'H p_h = [{qh[0]:.2f}, {qh[1]:.2f}, {qh[2]:.2f}]',f'w´ = {qh[2]:.2f}',f'p´ = [{qp[0,0]:.2f}, {qp[0,1]:.2f}]','',
        {'text':'PROYECTIVA','bold':True},'H y lambda H: misma homografía','9 entradas - 1 escala = 8 DoF','rectas -> rectas','paralelismo no garantizado',
      ],'phase':fase,'info_line_height':0.037,'info_fontsize':8.4,
      'legend':[{'kind':'line','label':'cuadrícula proyectada','color':'#2563EB'},{'kind':'point','label':'punto transformado','color':'#7B2CBF'},{'kind':'point','label':'punto de fuga','color':'#E07A1F'}],'legend_fontsize':7.9,
    }

def crear_estados_demostracion():
    estados=[]
    for _ in range(30): estados.append(crear_estado(0,'1/3 · Afinidad límite','Con h31=0, w´=1 y la transformación no introduce perspectiva. Las familias paralelas siguen paralelas.'))
    for p in np.linspace(0,1,170): estados.append(crear_estado(p,'2/3 · Introducir perspectiva','Aumentamos h31 hasta 0.1. Ahora w´=0.1x+1 depende de la posición y la división final comprime de forma distinta cada punto.'))
    for _ in range(85): estados.append(crear_estado(1,'3/3 · Ejemplo exacto y punto de fuga','Para [10,5,1]^T obtenemos [10,5,2]^T y, tras dividir por w=2, [5,2.5]^T. Las rectas horizontales convergen hacia un punto de fuga finito.'))
    return {'states':estados}

def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.8,8.8),interval=50)
    a.animate_2d_states(r['states'],'8.4. Transformación proyectiva y homografía',limits=(-1.5,11.5,-10.0,10.0),
      final_image_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'04_homografia.png',
      video_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'04_homografia.webm',repeat=False,fps=20,dpi=130,show=True)
if __name__=='__main__': main()
