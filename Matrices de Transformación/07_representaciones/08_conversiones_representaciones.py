from pathlib import Path
import sys
import numpy as np
CURRENT_DIR=Path(__file__).resolve().parent; MATRICES_DIR=CURRENT_DIR.parent; sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator

def rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],float)
def ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],float)
def rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
def rpy_R(r,p,y): return rz(y)@ry(p)@rx(r)
def hat(v): x,y,z=np.asarray(v,float); return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)
def exp_so3(phi):
    th=np.linalg.norm(phi); K=hat(phi)
    if th<1e-9: return np.eye(3)+K
    return np.eye(3)+(np.sin(th)/th)*K+((1-np.cos(th))/th**2)*(K@K)
def log_so3(R):
    th=np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
    if th<1e-9: return np.array([0.,0.,0.])
    return (th/(2*np.sin(th)))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
def R_q(R):
    tr=np.trace(R)
    if tr>0:
        s=np.sqrt(tr+1)*2; q=np.array([(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s,.25*s])
    else:
        vals=np.diag(R); i=int(np.argmax(vals))
        if i==0:
            s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; q=np.array([.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s,(R[2,1]-R[1,2])/s])
        elif i==1:
            s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; q=np.array([(R[0,1]+R[1,0])/s,.25*s,(R[1,2]+R[2,1])/s,(R[0,2]-R[2,0])/s])
        else:
            s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; q=np.array([(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,.25*s,(R[1,0]-R[0,1])/s])
    return q/np.linalg.norm(q)
def q_R(q):
    x,y,z,w=np.asarray(q,float)/np.linalg.norm(q); return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def J(phi):
    th=np.linalg.norm(phi); K=hat(phi)
    if th<1e-9: return np.eye(3)+.5*K+(1/6)*K@K
    return np.eye(3)+(1-np.cos(th))/th**2*K+(th-np.sin(th))/th**3*(K@K)
def Jinv(phi):
    th=np.linalg.norm(phi); K=hat(phi)
    if th<1e-9: return np.eye(3)-.5*K+(1/12)*K@K
    a=1/th**2-(1+np.cos(th))/(2*th*np.sin(th)); return np.eye(3)-.5*K+a*K@K
def makeT(R,t): M=np.eye(4); M[:3,:3]=R; M[:3,3]=t; return M
def fmt(v): return '['+', '.join(f'{x:6.3f}' for x in np.asarray(v))+']'
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)

def crear_estado(s,paso,fase,mensaje):
    rpy=s*np.radians([30,-35,65]); R=rpy_R(*rpy); phi=log_so3(R); theta=np.linalg.norm(phi); u=phi/theta if theta>1e-9 else np.array([1.,0,0]); q=R_q(R)
    t=s*np.array([1.8,-0.8,1.1]); T=makeT(R,t); rho=Jinv(phi)@t; xi=np.r_[rho,phi]
    Rq=q_R(q); Rphi=exp_so3(phi)
    origins=[np.array([-3.6,-1.4,0.]),np.array([-1.2,-1.4,0.]),np.array([1.2,-1.4,0.]),np.array([3.6,-1.4,0.])]
    names=['RPY','matriz R','quaternion','eje-ángulo']
    Rs=[R,R,Rq,Rphi]
    frames=[]
    for i in range(min(paso+1,4)):
        frames.append({'name':names[i],'origin':origins[i],'rotation':Rs[i],'length':0.95,'alpha':1.0})
    # Pose global arriba
    if paso>=4: frames.append({'name':'T in SE(3)','origin':np.array([0,2.0,0])+t,'rotation':R,'length':1.15,'alpha':1.0,'colors':('#D97706','#0F766E','#2563EB')})
    return {
      'frames3d':frames,
      'segments3d':[{'start':np.array([0,2.0,0]),'end':np.array([0,2.0,0])+t,'color':'#E07A1F','alpha':0.7,'linestyle':'--'}] if paso>=4 else [],
      'message':mensaje,'info_title':'Conversiones equivalentes',
      'info_lines':[
        {'text':'ORIENTACIÓN','bold':True},f'RPY = {fmt(np.degrees(rpy))} deg',f'q[x,y,z,w] = {fmt(q)}',f'u = {fmt(u)}',f'theta = {np.degrees(theta):6.2f}°',f'rotvec phi = {fmt(phi)}','',
        {'text':'RECONSTRUCCIÓN','bold':True},f'||Rq-R||={np.linalg.norm(Rq-R):.2e}',f'||Exp(phi)-R||={np.linalg.norm(Rphi-R):.2e}','',
        {'text':'POSE','bold':True},f't = {fmt(t)}',f'xi=[rho,phi] = {fmt(xi)}',f'||J(phi)rho-t||={np.linalg.norm(J(phi)@rho-t):.2e}',
      ],'phase':fase,'info_line_height':0.0355,'info_fontsize':8.1,
    }
def crear_estados_demostracion():
    estados=[]
    for p in np.linspace(0,1,100): estados.append(crear_estado(suavizar(p),0,'1/5 · RPY -> R','Partimos de una tripleta RPY documentada y construimos una matriz R. Esta será la orientación de referencia para todas las conversiones.'))
    for _ in range(45): estados.append(crear_estado(1,1,'2/5 · Matriz R','La matriz conserva la geometría de la orientación y sirve como representación común para verificar las demás parametrizaciones.'))
    for _ in range(50): estados.append(crear_estado(1,2,'3/5 · Cuaternión','Convertimos R a un cuaternión unitario [x,y,z,w] y reconstruimos la misma matriz sin pérdida numérica significativa.'))
    for _ in range(55): estados.append(crear_estado(1,3,'4/5 · Eje-ángulo / rotvec / so(3)','Log(R) produce el rotvec phi=theta·u. Su norma da el ángulo, su dirección da el eje y Exp(phi) vuelve a la misma R.'))
    for _ in range(85): estados.append(crear_estado(1,4,'5/5 · Pose: (R,t), T y xi local','Añadimos una traslación para formar T en SE(3). La coordenada local xi=[rho,phi] se relaciona con la pose mediante la Jacobiana de SO(3).'))
    return {'states':estados}
def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(16.0,9.0),interval=50)
    a.animate_3d_states(r['states'],'7.8. Comparación y conversiones entre representaciones',limits=(-5.4,5.4,-3.2,5.0,-1.8,4.2),view=(24,-58),final_image_path=MATRICES_DIR/'assets'/'07_representaciones'/'08_conversiones_representaciones.png',video_path=MATRICES_DIR/'assets'/'07_representaciones'/'08_conversiones_representaciones.webm',repeat=False,fps=20,dpi=125,show=True)
if __name__=='__main__': main()
