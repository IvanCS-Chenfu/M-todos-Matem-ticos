from pathlib import Path
import sys
import numpy as np
try:
    import cv2
except ImportError:
    cv2=None

CURRENT_DIR=Path(__file__).resolve().parent; MATRICES_DIR=CURRENT_DIR.parent; sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator

def hat(u):
    x,y,z=np.asarray(u,float); return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)
def rodrigues(u,theta):
    u=np.asarray(u,float); u=u/np.linalg.norm(u); K=hat(u)
    return np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)
def fmt(v): return '['+', '.join(f'{x:6.3f}' for x in np.asarray(v))+']'
def trayectoria_rotacion(u,v,theta,n=100):
    ts=np.linspace(0,theta,n); return np.array([rodrigues(u,t)@v for t in ts])

def crear_estado(theta,theta_final,u,v,fase,mensaje):
    R=rodrigues(u,theta); vr=R@v; phi=theta*u; K=hat(u)
    if cv2 is not None:
        Rcv,_=cv2.Rodrigues(phi.reshape(3,1)); err_cv=np.linalg.norm(R-Rcv)
    else: err_cv=np.nan
    traj=trayectoria_rotacion(u,v,theta,max(2,int(90*theta/max(theta_final,1e-9))+2))
    return {
      'frames3d':[{'name':'A','origin':np.zeros(3),'rotation':np.eye(3),'length':1.2,'alpha':0.18,'colors':('#9CA3AF',)*3},{'name':'R','origin':np.zeros(3),'rotation':R,'length':1.45,'alpha':0.75}],
      'vectors3d':[
        {'name':'u','origin':-2.2*u,'value':4.4*u,'color':'#E07A1F','linewidth':2.8,'show_origin':False},
        {'name':'v','origin':np.zeros(3),'value':v,'color':'#6B7280','alpha':0.30,'linewidth':2.0},
        {'name':'Rv','origin':np.zeros(3),'value':vr,'color':'#7B2CBF','linewidth':3.0},
        {'name':'phi=theta u','origin':np.zeros(3),'value':phi,'color':'#2D7F5E','linewidth':2.6},
      ],
      'polylines3d':[{'points':traj,'color':'#7B2CBF','linewidth':2.0,'alpha':0.70}] if len(traj)>1 else [],
      'message':mensaje,
      'info_title':'Eje-ángulo y Rodrigues',
      'info_lines':[
        {'text':'EJE UNITARIO','bold':True},f'u = {fmt(u)}',f'||u|| = {np.linalg.norm(u):.6f}','',
        {'text':'ÁNGULO / ROTVEC','bold':True},f'theta = {np.degrees(theta):6.1f}°',f'phi = {fmt(phi)}',f'||phi|| = {np.linalg.norm(phi):.3f}','',
        {'text':'RODRIGUES','bold':True},'R = I + sinθ K + (1-cosθ)K²',f'||R^T R-I||={np.linalg.norm(R.T@R-np.eye(3)):.2e}',f'det(R)={np.linalg.det(R):.6f}',
        f'error cv2 = {err_cv:.2e}' if cv2 is not None else 'cv2 no disponible',
      ],'phase':fase,'info_line_height':0.0385,'info_fontsize':8.5,
      'legend':[{'kind':'line','label':'eje u','color':'#E07A1F'},{'kind':'line','label':'arco de v','color':'#7B2CBF'},{'kind':'line','label':'rotvec phi','color':'#2D7F5E'}],'legend_fontsize':7.9,
    }
def crear_estados_demostracion():
    u=np.array([1.,2.,1.]); u/=np.linalg.norm(u); v=np.array([1.8,-0.4,0.7]); tf=np.radians(125)
    estados=[]
    for _ in range(30): estados.append(crear_estado(0,tf,u,v,'1/3 · Eje y vector','Toda rotación propia puede describirse mediante un eje unitario u y un ángulo theta.'))
    for p in np.linspace(0,1,160): estados.append(crear_estado(suavizar(p)*tf,tf,u,v,'2/3 · Fórmula de Rodrigues','Rodrigues convierte directamente eje y ángulo en una matriz R. El vector rota alrededor del eje u sin cambiar su norma.'))
    for _ in range(70): estados.append(crear_estado(tf,tf,u,v,'3/3 · Vector de rotación','El vector phi=theta·u reúne eje y ángulo en tres componentes y coincide con la entrada utilizada por cv2.Rodrigues.'))
    return {'states':estados}
def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.8,8.9),interval=50)
    a.animate_3d_states(r['states'],'7.2. Eje-ángulo y fórmula de Rodrigues',limits=(-3.0,3.0,-3.0,3.0,-2.7,3.2),view=(24,-58),final_image_path=MATRICES_DIR/'assets'/'07_representaciones'/'02_eje_angulo_rodrigues.png',video_path=MATRICES_DIR/'assets'/'07_representaciones'/'02_eje_angulo_rodrigues.webm',repeat=False,fps=20,dpi=125,show=True)
if __name__=='__main__': main()
