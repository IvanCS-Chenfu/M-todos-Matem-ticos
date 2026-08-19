from pathlib import Path
import sys
import numpy as np
CURRENT_DIR=Path(__file__).resolve().parent; MATRICES_DIR=CURRENT_DIR.parent; sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator

def rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],float)
def ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],float)
def rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
def rpy_R(r,p,y): return rz(y)@ry(p)@rx(r)
def q_normalizar(q):
    q=np.asarray(q,float); return q/np.linalg.norm(q)
def eje_angulo_q(u,theta):
    u=np.asarray(u,float); u=u/np.linalg.norm(u); return q_normalizar(np.r_[u*np.sin(theta/2),np.cos(theta/2)])
def q_R(q):
    x,y,z,w=q_normalizar(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],float)
def R_q(R):
    tr=np.trace(R)
    if tr>0:
        s=np.sqrt(tr+1)*2; w=.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    else:
        i=int(np.argmax(np.diag(R)))
        if i==0:
            s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif i==1:
            s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=.25*s
    return q_normalizar([x,y,z,w])
def q_slerp(q0,q1,t):
    q0=q_normalizar(q0); q1=q_normalizar(q1); dot=float(q0@q1)
    if dot<0: q1=-q1; dot=-dot
    if dot>0.9995: return q_normalizar((1-t)*q0+t*q1)
    th=np.arccos(np.clip(dot,-1,1)); return (np.sin((1-t)*th)/np.sin(th))*q0+(np.sin(t*th)/np.sin(th))*q1
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)
def fmt(v): return '['+', '.join(f'{x:6.3f}' for x in np.asarray(v))+']'

def crear_estado(t,fase,mensaje):
    rpy0=np.radians([-25,20,-35]); rpy1=np.radians([70,-55,145]); R0=rpy_R(*rpy0); R1=rpy_R(*rpy1)
    q0=R_q(R0); q1=R_q(R1); qs=q_slerp(q0,q1,t); Rs=q_R(qs)
    rpy_lin=(1-t)*rpy0+t*rpy1; Re=rpy_R(*rpy_lin)
    # q y -q para la orientación objetivo
    err_sign=np.linalg.norm(q_R(q1)-q_R(-q1))
    o1=np.array([-2.7,0,0.]); o2=np.array([2.7,0,0.])
    tip=np.array([1.4,0.35,0.25])
    return {
      'frames3d':[
        {'name':'SLERP','origin':o1,'rotation':Rs,'length':1.35,'alpha':1.0},
        {'name':'RPY lineal','origin':o2,'rotation':Re,'length':1.35,'alpha':1.0,'colors':('#D97706','#0F766E','#2563EB')},
      ],
      'vectors3d':[
        {'name':'v_slerp','origin':o1,'value':Rs@tip,'color':'#7B2CBF','linewidth':2.8},
        {'name':'v_rpy','origin':o2,'value':Re@tip,'color':'#E07A1F','linewidth':2.8},
      ],
      'texts3d':[
        {'position':o1+np.array([0,0,2.15]),'text':'SLERP en cuaterniones','fontweight':'bold','color':'#7B2CBF'},
        {'position':o2+np.array([0,0,2.15]),'text':'Interpolación RPY','fontweight':'bold','color':'#D97706'},
      ],
      'message':mensaje,
      'info_title':'Cuaterniones [x,y,z,w]',
      'info_lines':[
        {'text':'INTERPOLACIÓN','bold':True},f't = {t:5.2f}',f'q0 = {fmt(q0)}',f'q1 = {fmt(q1)}',f'q(t)= {fmt(qs)}',f'||q(t)|| = {np.linalg.norm(qs):.6f}','',
        {'text':'DOBLE COBERTURA','bold':True},f'||R(q)-R(-q)|| = {err_sign:.2e}','q y -q: misma orientación','',
        {'text':'CONVERSIÓN','bold':True},f'||R(q1)-R1|| = {np.linalg.norm(q_R(q1)-R1):.2e}',
      ],'phase':fase,'info_line_height':0.041,'info_fontsize':8.6,
      'legend':[{'kind':'line','label':'SLERP','color':'#7B2CBF'},{'kind':'line','label':'RPY lineal','color':'#E07A1F'}],'legend_fontsize':8.0,
    }
def crear_estados_demostracion():
    estados=[]
    for _ in range(35): estados.append(crear_estado(0,'1/3 · Dos orientaciones','Convertimos las orientaciones inicial y final a cuaterniones unitarios en orden [x,y,z,w], compatible con la convención habitual de mensajes ROS.'))
    for p in np.linspace(0,1,180): estados.append(crear_estado(suavizar(p),'2/3 · SLERP frente a RPY lineal','SLERP interpola sobre cuaterniones normalizados y produce una evolución geométrica suave. A la derecha se interpolan ingenuamente las tres componentes RPY.'))
    for _ in range(75): estados.append(crear_estado(1,'3/3 · q y -q','La orientación final también puede representarse con -q. Sus componentes cambian de signo, pero la matriz de rotación es exactamente la misma.'))
    return {'states':estados}
def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.9,8.9),interval=50)
    a.animate_3d_states(r['states'],'7.3. Cuaterniones',limits=(-5.0,5.0,-3.0,3.0,-2.0,3.4),view=(24,-58),final_image_path=MATRICES_DIR/'assets'/'07_representaciones'/'03_cuaterniones.png',video_path=MATRICES_DIR/'assets'/'07_representaciones'/'03_cuaterniones.webm',repeat=False,fps=20,dpi=125,show=True)
if __name__=='__main__': main()
