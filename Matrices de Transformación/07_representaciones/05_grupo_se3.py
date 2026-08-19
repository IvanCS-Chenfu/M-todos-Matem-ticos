from pathlib import Path
import sys
import numpy as np
CURRENT_DIR=Path(__file__).resolve().parent; MATRICES_DIR=CURRENT_DIR.parent; sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator

def rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],float)
def ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],float)
def rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
def makeT(R,t): M=np.eye(4); M[:3,:3]=R; M[:3,3]=t; return M
def invT(T):
    R=T[:3,:3]; t=T[:3,3]; N=np.eye(4); N[:3,:3]=R.T; N[:3,3]=-R.T@t; return N
def is_so3(R,tol=1e-8): return np.asarray(R).shape==(3,3) and np.linalg.norm(R.T@R-np.eye(3))<tol and abs(np.linalg.det(R)-1)<tol
def is_se3(T,tol=1e-8):
    T=np.asarray(T,float); return T.shape==(4,4) and is_so3(T[:3,:3],tol) and np.linalg.norm(T[3]-np.array([0,0,0,1]))<tol
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)
def fmt(v): return '['+', '.join(f'{x:5.2f}' for x in v)+']'

def crear_estado(s,paso,fase,mensaje):
    R1=rz(s*np.radians(40))@ry(s*np.radians(-15)); t1=s*np.array([2.1,-0.7,0.9]); T1=makeT(R1,t1)
    R2=rx(s*np.radians(25))@rz(s*np.radians(-30)); t2=s*np.array([1.0,0.8,0.6]); T2=makeT(R2,t2)
    T12=T1@T2; T1i=invT(T1)
    Tinval=T12.copy(); Tinval[0,0]*=1.4
    o0=np.zeros(3); o1=T1[:3,3]; o2=T12[:3,3]
    return {
      'frames3d':[
        {'name':'I','origin':o0,'rotation':np.eye(3),'length':1.20,'alpha':0.25,'colors':('#9CA3AF',)*3},
        {'name':'T1','origin':o1,'rotation':R1,'length':1.15,'alpha':0.9},
        {'name':'T1T2','origin':o2,'rotation':T12[:3,:3],'length':1.25,'alpha':1.0,'colors':('#D97706','#0F766E','#2563EB')},
      ],
      'segments3d':[{'start':o0,'end':o1,'color':'#7B2CBF','alpha':0.5,'linestyle':'--'},{'start':o1,'end':o2,'color':'#E07A1F','alpha':0.65,'linestyle':'--'}],
      'message':mensaje,'info_title':'El grupo SE(3)',
      'info_lines':[
        {'text':'6 GRADOS DE LIBERTAD','bold':True},f't = {fmt(T12[:3,3])}',f'rotación: 3 DoF + traslación: 3 DoF','',
        {'text':'OPERACIÓN DE GRUPO','bold':True},f'T1 in SE(3): {is_se3(T1)}',f'T2 in SE(3): {is_se3(T2)}',f'T1T2 in SE(3): {is_se3(T12)}',f'T1^-1 in SE(3): {is_se3(T1i)}',f'||T1 T1^-1-I||={np.linalg.norm(T1@T1i-np.eye(4)):.2e}','',
        {'text':'EJEMPLO INVÁLIDO','bold':True},f'T escalada in SE(3): {is_se3(Tinval)}',
      ],'phase':fase,'info_line_height':0.0395,'info_fontsize':8.6,
      'legend':[{'kind':'line','label':'T1','color':'#7B2CBF'},{'kind':'line','label':'T2 local','color':'#E07A1F'}],'legend_fontsize':8.0,
    }
def crear_estados_demostracion():
    estados=[]
    for p in np.linspace(0,1,130): estados.append(crear_estado(suavizar(p),0,'1/3 · Una pose en SE(3)','Una transformación rígida combina tres grados de libertad de orientación y tres de traslación dentro de una matriz 4x4 restringida.'))
    for _ in range(75): estados.append(crear_estado(1,1,'2/3 · Cierre e inversa','Componer dos poses válidas produce otra pose de SE(3). La identidad pertenece al grupo y cada pose tiene una inversa rígida.'))
    for _ in range(75): estados.append(crear_estado(1,2,'3/3 · Estructura restringida','No toda matriz 4x4 pertenece a SE(3): el bloque R debe estar en SO(3) y la última fila debe conservar la estructura homogénea.'))
    return {'states':estados}
def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.8,8.9),interval=50)
    a.animate_3d_states(r['states'],'7.5. El grupo SE(3)',limits=(-2,5.8,-3,3.7,-1.5,4),view=(24,-58),final_image_path=MATRICES_DIR/'assets'/'07_representaciones'/'05_grupo_se3.png',video_path=MATRICES_DIR/'assets'/'07_representaciones'/'05_grupo_se3.webm',repeat=False,fps=20,dpi=125,show=True)
if __name__=='__main__': main()
