from pathlib import Path
import sys
import numpy as np

CURRENT_DIR=Path(__file__).resolve().parent
MATRICES_DIR=CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator


def R2(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s],[s,c]],float)
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)
def shear(k): return np.array([[1.0,k],[0.0,1.0]])
def transformar(P,A,t=np.zeros(2)): return (A@P.T).T+t
def angulo(a,b):
    c=np.clip((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1); return np.arccos(c)

def cuadricula(xmin=-3,xmax=3,ymin=-3,ymax=3,n=7,m=45):
    lines=[]
    for x in np.linspace(xmin,xmax,n): lines.append(np.c_[np.full(m,x),np.linspace(ymin,ymax,m)])
    for y in np.linspace(ymin,ymax,n): lines.append(np.c_[np.linspace(xmin,xmax,m),np.full(m,y)])
    return lines


def crear_estado(modo,progreso,fase,mensaje):
    s=suavizar(progreso)
    if modo=='shear':
        k=2.0*s; A=shear(k)
    else:
        k=2.0
        A=R2(s*np.radians(25.0)) @ np.diag([1.0+0.45*s,1.0-0.35*s]) @ shear(k*s)
    p=np.array([1.0,3.0]); p_shear=shear(2.0)@p; pt=A@p
    grids=cuadricula()
    polylines=[{'points':transformar(g,A),'color':'#2563EB','linewidth':1.25,'alpha':0.55} for g in grids]
    # Dos direcciones inicialmente perpendiculares y dos paralelas.
    e1=np.array([1.,0.]); e2=np.array([0.,1.]); d1=A@e1; d2=A@e2
    parallel_error=abs(np.linalg.det(np.c_[A@np.array([1.,0.]),A@np.array([2.,0.])]))
    return {
      'polylines':polylines,
      'points':[{'name':'p','position':pt,'color':'#7B2CBF','size':92}],
      'message':mensaje,'info_title':'Transformación afín 2D',
      'info_lines':[
        {'text':'EJEMPLO DE CIZALLA','bold':True},'H = [[1,k],[0,1]]',f'k actual = {k*s if modo!="shear" else k:.3f}','k=2: [1,3] -> [7,3]',f'comprobación = [{p_shear[0]:.1f}, {p_shear[1]:.1f}]','',
        {'text':'AFINIDAD ACTUAL','bold':True},f'A = [{A[0,0]:5.2f}, {A[0,1]:5.2f}]',f'    [{A[1,0]:5.2f}, {A[1,1]:5.2f}]',f'det(A)={np.linalg.det(A):.3f}','',
        {'text':'INVARIANTES','bold':True},f'paralelismo error={parallel_error:.2e}',f'ángulo e1,e2: 90.00° -> {np.degrees(angulo(d1,d2)):.2f}°',f'||e1||: 1.00 -> {np.linalg.norm(d1):.3f}',
      ],'phase':fase,'info_line_height':0.0365,'info_fontsize':8.35,
      'legend':[{'kind':'line','label':'cuadrícula transformada','color':'#2563EB'},{'kind':'point','label':'punto transformado','color':'#7B2CBF'}],'legend_fontsize':8.0,
    }


def crear_estados_demostracion():
    estados=[]
    for _ in range(28): estados.append(crear_estado('shear',0,'1/4 · Identidad','Una afinidad transforma rectas en rectas. Empezamos con una cuadrícula cartesiana y un punto de referencia.'))
    for p in np.linspace(0,1,130): estados.append(crear_estado('shear',p,'2/4 · Cizalla k: 0 -> 2','Aplicamos la cizalla horizontal x´=x+ky. Con k=2, el punto [1,3]^T termina exactamente en [7,3]^T.'))
    for _ in range(40): estados.append(crear_estado('shear',1,'3/4 · Paralelas siguen paralelas','La cizalla cambia ángulos y distancias, pero líneas inicialmente paralelas permanecen paralelas.'))
    for p in np.linspace(0,1,130): estados.append(crear_estado('general',p,'4/4 · Afinidad general','Añadimos escalado no uniforme y rotación a la cizalla. La cuadrícula continúa formada por rectas y familias paralelas, aunque la métrica euclídea ya no se conserva.'))
    for _ in range(65): estados.append(crear_estado('general',1,'Conclusión · Rectas y paralelismo','Una afinidad general puede combinar rotación, traslación, escalado, reflexión y cizalla. Sus invariantes principales son rectas y paralelismo.'))
    return {'states':estados}

def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.7,8.8),interval=50)
    a.animate_2d_states(r['states'],'8.3. Transformación afín',limits=(-8.0,10.5,-5.5,6.0),
      final_image_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'03_transformacion_afin.png',
      video_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'03_transformacion_afin.webm',repeat=False,fps=20,dpi=130,show=True)
if __name__=='__main__': main()
