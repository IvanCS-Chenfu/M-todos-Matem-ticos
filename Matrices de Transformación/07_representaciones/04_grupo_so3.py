from pathlib import Path
import sys
import numpy as np
CURRENT_DIR=Path(__file__).resolve().parent; MATRICES_DIR=CURRENT_DIR.parent; sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator

def rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],float)
def ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],float)
def rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
def is_so3(R,tol=1e-8):
    R=np.asarray(R,float); return R.shape==(3,3) and np.linalg.norm(R.T@R-np.eye(3))<tol and abs(np.linalg.det(R)-1)<tol
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)

def crear_estado(s,paso,fase,mensaje):
    R1=rz(s*np.radians(55))@rx(s*np.radians(20)); R2=ry(s*np.radians(-35))@rz(s*np.radians(25)); Rp=R1@R2
    S=R1+R2; E=2*R1
    mats=[('R1',R1,np.array([-3.2,-1.6,0.])),('R2',R2,np.array([0,-1.6,0.])),('R1R2',Rp,np.array([3.2,-1.6,0.]))]
    frames=[{'name':n,'origin':o,'rotation':R,'length':1.05,'alpha':1.0} for n,R,o in mats]
    if paso>=2:
        frames += [
          {'name':'R1+R2','origin':np.array([-1.6,2.0,0.]),'rotation':S,'length':0.72,'alpha':0.85,'colors':('#D97706','#D97706','#D97706')},
          {'name':'2R1','origin':np.array([2.0,2.0,0.]),'rotation':E,'length':0.55,'alpha':0.85,'colors':('#7B2CBF','#7B2CBF','#7B2CBF')},
        ]
    return {
      'frames3d':frames,'message':mensaje,'info_title':'El grupo SO(3)',
      'info_lines':[
        {'text':'PERTENENCIA','bold':True},f'R1 in SO(3): {is_so3(R1)}',f'R2 in SO(3): {is_so3(R2)}',f'R1R2 in SO(3): {is_so3(Rp)}','',
        {'text':'IDENTIDAD / INVERSA','bold':True},f'I in SO(3): {is_so3(np.eye(3))}',f'R1^T in SO(3): {is_so3(R1.T)}',f'||R1 R1^T-I||={np.linalg.norm(R1@R1.T-np.eye(3)):.2e}','',
        {'text':'NO ES ESPACIO VECTORIAL','bold':True},f'R1+R2 in SO(3): {is_so3(S)}',f'2 R1 in SO(3): {is_so3(E)}',f'det(R1+R2)={np.linalg.det(S):.3f}',
      ],'phase':fase,'info_line_height':0.039,'info_fontsize':8.5,
    }
def crear_estados_demostracion():
    estados=[]
    for p in np.linspace(0,1,110): estados.append(crear_estado(suavizar(p),0,'1/3 · Dos elementos de SO(3)','Construimos R1 y R2 como productos de rotaciones elementales. Ambas satisfacen ortogonalidad y determinante +1.'))
    for _ in range(60): estados.append(crear_estado(1,1,'2/3 · Operación de grupo','El producto R1R2 también pertenece a SO(3); la identidad y la inversa R1^T permanecen dentro del conjunto.'))
    for _ in range(85): estados.append(crear_estado(1,2,'3/3 · No es un espacio vectorial','La suma R1+R2 y el escalado 2R1 rompen las restricciones de una rotación. SO(3) es un grupo bajo multiplicación, no un espacio vectorial bajo suma y escalado.'))
    return {'states':estados}
def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.9,8.9),interval=50)
    a.animate_3d_states(r['states'],'7.4. El grupo SO(3)',limits=(-5,5,-3.3,4,-1.5,3.8),view=(25,-60),final_image_path=MATRICES_DIR/'assets'/'07_representaciones'/'04_grupo_so3.png',video_path=MATRICES_DIR/'assets'/'07_representaciones'/'04_grupo_so3.webm',repeat=False,fps=20,dpi=125,show=True)
if __name__=='__main__': main()
