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
def angulo(a,b):
    c=np.clip((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1); return np.arccos(c)
def aplicar(fig,R,s,t): return (s*R@fig.T).T+t


def crear_estado(progreso,fase,mensaje):
    q=suavizar(progreso)
    theta=q*np.radians(35.0)
    R=R2(theta)
    escala=1.0+q
    fig=np.array([[-0.9,-0.6],[0.9,-0.6],[1.15,0.15],[0.0,1.0],[-1.0,0.35]])
    t_r=np.array([-3.0,0.0]); t_s=np.array([2.8,-0.15])
    Pr=aplicar(fig,R,1.0,t_r)
    Ps=aplicar(fig,R,escala,t_s)
    u0=fig[1]-fig[0]; v0=fig[4]-fig[0]
    ur=Pr[1]-Pr[0]; vr=Pr[4]-Pr[0]
    us=Ps[1]-Ps[0]; vs=Ps[4]-Ps[0]
    return {
      'polygons':[
        {'points':Pr,'facecolor':'#BFDBFE','edgecolor':'#2563EB','alpha':0.36,'linewidth':2.0},
        {'points':Ps,'facecolor':'#FED7AA','edgecolor':'#D97706','alpha':0.34,'linewidth':2.0},
      ],
      'texts':[
        {'position':t_r+np.array([-1.35,2.15]),'text':'RÍGIDA: s = 1','fontweight':'bold','color':'#2563EB'},
        {'position':t_s+np.array([-1.45,2.65]),'text':'SIMILITUD: s variable','fontweight':'bold','color':'#D97706'},
      ],
      'message':mensaje,'info_title':'Rígida frente a similitud',
      'info_lines':[
        {'text':'p´ = s R p + t','bold':True},f'theta = {np.degrees(theta):5.1f}°',f's = {escala:.3f}','',
        {'text':'LONGITUD DEL MISMO LADO','bold':True},f'original = {np.linalg.norm(u0):.3f}',f'rígida   = {np.linalg.norm(ur):.3f}',f'similitud= {np.linalg.norm(us):.3f}',f'cociente = {np.linalg.norm(us)/np.linalg.norm(u0):.3f}','',
        {'text':'ÁNGULO INTERNO','bold':True},f'original = {np.degrees(angulo(u0,v0)):.2f}°',f'rígida   = {np.degrees(angulo(ur,vr)):.2f}°',f'similitud= {np.degrees(angulo(us,vs)):.2f}°',
      ],
      'phase':fase,'info_line_height':0.040,'info_fontsize':8.6,
      'legend':[{'kind':'line','label':'rígida','color':'#2563EB'},{'kind':'line','label':'similitud','color':'#D97706'}],'legend_fontsize':8.2,
    }


def crear_estados_demostracion():
    estados=[]
    for _ in range(30): estados.append(crear_estado(0,'1/3 · Misma figura','Partimos de dos copias de la misma figura. A la izquierda solo permitimos una transformación rígida; a la derecha añadiremos escala uniforme.'))
    for p in np.linspace(0,1,150): estados.append(crear_estado(p,'2/3 · Aumentar s hasta 2','La similitud aplica la misma escala en todas las direcciones mientras rota y traslada. La forma y los ángulos permanecen iguales.'))
    for _ in range(75): estados.append(crear_estado(1,'3/3 · Invariantes de similitud','Con s=2 las longitudes se duplican y las distancias absolutas cambian, pero los ángulos y las razones de longitudes se conservan.'))
    return {'states':estados}


def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.7,8.8),interval=50)
    a.animate_2d_states(r['states'],'8.2. Transformación de similitud',limits=(-5.5,6.8,-4.0,4.7),
      final_image_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'02_similitud.png',
      video_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'02_similitud.webm',repeat=False,fps=20,dpi=130,show=True)
if __name__=='__main__': main()
