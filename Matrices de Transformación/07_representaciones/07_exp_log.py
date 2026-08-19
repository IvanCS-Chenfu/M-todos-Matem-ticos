from pathlib import Path
import sys
import numpy as np
CURRENT_DIR=Path(__file__).resolve().parent; MATRICES_DIR=CURRENT_DIR.parent; sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator

def hat(v):
    x,y,z=np.asarray(v,float); return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)
def vee(M): return np.array([M[2,1],M[0,2],M[1,0]],float)
def exp_so3(phi):
    phi=np.asarray(phi,float); th=np.linalg.norm(phi); K=hat(phi)
    if th<1e-8: return np.eye(3)+K+0.5*K@K
    return np.eye(3)+(np.sin(th)/th)*K+((1-np.cos(th))/th**2)*(K@K)
def log_so3(R):
    c=np.clip((np.trace(R)-1)/2,-1,1); th=np.arccos(c)
    if th<1e-8: return vee(0.5*(R-R.T))
    return vee((th/(2*np.sin(th)))*(R-R.T))
def J_so3(phi):
    phi=np.asarray(phi,float); th=np.linalg.norm(phi); K=hat(phi)
    if th<1e-8: return np.eye(3)+0.5*K+(1/6)*(K@K)
    return np.eye(3)+((1-np.cos(th))/th**2)*K+((th-np.sin(th))/th**3)*(K@K)
def Jinv_so3(phi):
    phi=np.asarray(phi,float); th=np.linalg.norm(phi); K=hat(phi)
    if th<1e-8: return np.eye(3)-0.5*K+(1/12)*(K@K)
    a=1/th**2-(1+np.cos(th))/(2*th*np.sin(th))
    return np.eye(3)-0.5*K+a*(K@K)
def exp_se3(xi):
    xi=np.asarray(xi,float); rho=xi[:3]; phi=xi[3:]; T=np.eye(4); T[:3,:3]=exp_so3(phi); T[:3,3]=J_so3(phi)@rho; return T
def log_se3(T):
    phi=log_so3(T[:3,:3]); rho=Jinv_so3(phi)@T[:3,3]; return np.r_[rho,phi]
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)
def fmt(v): return '['+', '.join(f'{x:6.3f}' for x in np.asarray(v))+']'

def crear_estado(k,n,xi_total,fase,mensaje):
    delta=xi_total/n
    T=np.eye(4); path=[T[:3,3].copy()]
    for _ in range(k): T=T@exp_se3(delta); path.append(T[:3,3].copy())
    xi_log=log_se3(T); Trec=exp_se3(xi_log)
    phi=xi_log[3:]
    return {
      'frames3d':[{'name':'I','origin':np.zeros(3),'rotation':np.eye(3),'length':1.15,'alpha':0.22,'colors':('#9CA3AF',)*3},{'name':'T','origin':T[:3,3],'rotation':T[:3,:3],'length':1.4,'alpha':1.0}],
      'polylines3d':[{'points':np.asarray(path),'color':'#7B2CBF','linewidth':2.5,'alpha':0.85}] if len(path)>1 else [],
      'vectors3d':[{'name':'Log(R)=phi','origin':T[:3,3],'value':phi,'color':'#E07A1F','linewidth':2.6}] if np.linalg.norm(phi)>1e-9 else [],
      'message':mensaje,'info_title':'Exp y Log en SO(3) / SE(3)',
      'info_lines':[
        {'text':'INCREMENTO LOCAL','bold':True},f'delta xi = {fmt(delta)}',f'pasos = {k:2d}/{n}','',
        {'text':'POSE ACTUAL','bold':True},f't = {fmt(T[:3,3])}',f'Log(T) = {fmt(xi_log)}','',
        {'text':'VERIFICACIONES','bold':True},f'||Exp(Log(T))-T||={np.linalg.norm(Trec-T):.2e}',f'||Exp(Log(R))-R||={np.linalg.norm(exp_so3(log_so3(T[:3,:3]))-T[:3,:3]):.2e}',
        't = J(phi) rho',
      ],'phase':fase,'info_line_height':0.040,'info_fontsize':8.5,
      'legend':[{'kind':'line','label':'trayectoria acumulada','color':'#7B2CBF'},{'kind':'line','label':'vector Log(R)','color':'#E07A1F'}],'legend_fontsize':8.0,
    }
def crear_estados_demostracion():
    xi=np.array([1.8,0.55,0.65, 0.25,-0.35,0.85]); n=28; estados=[]
    for _ in range(30): estados.append(crear_estado(0,n,xi,'1/3 · Identidad y coordenada local','Exp lleva una coordenada local del álgebra al grupo. Empezamos en la identidad y definimos un pequeño incremento se(3).'))
    for k in range(1,n+1):
        for _ in range(5): estados.append(crear_estado(k,n,xi,'2/3 · Acumular Exp(delta xi)','Aplicamos repetidamente pequeños incrementos Exp(delta xi). La pose permanece siempre en SE(3) y deja una trayectoria continua.'))
    for _ in range(80): estados.append(crear_estado(n,n,xi,'3/3 · Recuperar con Log','Log convierte la pose final de nuevo en un vector local. Exp(Log(T)) reconstruye la misma transformación salvo error numérico.'))
    return {'states':estados}
def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.8,8.9),interval=50)
    a.animate_3d_states(r['states'],'7.7. Mapas exponencial y logarítmico',limits=(-1.8,4.5,-2.5,3.2,-1.2,3.6),view=(24,-58),final_image_path=MATRICES_DIR/'assets'/'07_representaciones'/'07_exp_log.png',video_path=MATRICES_DIR/'assets'/'07_representaciones'/'07_exp_log.webm',repeat=False,fps=20,dpi=125,show=True)
if __name__=='__main__': main()
