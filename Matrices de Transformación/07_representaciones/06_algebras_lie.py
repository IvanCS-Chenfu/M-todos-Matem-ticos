from pathlib import Path
import sys
import numpy as np
CURRENT_DIR=Path(__file__).resolve().parent; MATRICES_DIR=CURRENT_DIR.parent; sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator

def hat3(v):
    x,y,z=np.asarray(v,float); return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)
def vee3(M): return np.array([M[2,1],M[0,2],M[1,0]],float)
def wedge6(xi):
    xi=np.asarray(xi,float); M=np.zeros((4,4)); M[:3,:3]=hat3(xi[3:]); M[:3,3]=xi[:3]; return M
def vee6(M): return np.r_[M[:3,3],vee3(M[:3,:3])]
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)
def fmt(v): return '['+', '.join(f'{x:6.3f}' for x in np.asarray(v))+']'
def row(M,i): return '['+', '.join(f'{x:6.3f}' for x in M[i])+']'

def crear_estado(s,paso,fase,mensaje):
    phi=s*np.array([0.25,-0.40,0.55]); rho=s*np.array([0.80,-0.30,0.45]); xi=np.r_[rho,phi]
    Ph=hat3(phi); Xh=wedge6(xi)
    vectors=[{'name':'phi','origin':np.zeros(3),'value':3*phi,'color':'#7B2CBF','linewidth':3.0}]
    if paso>=1: vectors.append({'name':'rho','origin':np.zeros(3),'value':rho,'color':'#E07A1F','linewidth':3.0})
    return {
      'frames3d':[{'name':'I','origin':np.zeros(3),'rotation':np.eye(3),'length':1.3,'alpha':0.30,'colors':('#9CA3AF',)*3}],
      'vectors3d':vectors,
      'texts3d':[{'position':np.array([-2.3,2.1,1.8]),'text':'espacio local lineal','fontweight':'bold','color':'#374151'}],
      'message':mensaje,'info_title':'Álgebras so(3) y se(3)',
      'info_lines':[
        {'text':'so(3): phi -> phi^','bold':True},f'phi = {fmt(phi)}',row(Ph,0),row(Ph,1),row(Ph,2),f'vee(phi^) error={np.linalg.norm(vee3(Ph)-phi):.2e}','',
        {'text':'se(3): xi=[rho,phi]','bold':True},f'rho = {fmt(rho)}',f'xi  = {fmt(xi)}',f'vee(xi^) error={np.linalg.norm(vee6(Xh)-xi):.2e}',
      ],'phase':fase,'info_line_height':0.040,'info_fontsize':8.4,
      'legend':[{'kind':'line','label':'parte rotacional phi','color':'#7B2CBF'},{'kind':'line','label':'parte traslacional rho','color':'#E07A1F'}] if paso>=1 else [],'legend_fontsize':8.0,
    }
def crear_estados_demostracion():
    estados=[]
    for p in np.linspace(0,1,100): estados.append(crear_estado(suavizar(p),0,'1/3 · Vector local phi','Cerca de la identidad, una pequeña rotación puede describirse con un vector ordinario phi en R³. El operador hat lo convierte en una matriz antisimétrica de so(3).'))
    for _ in range(55): estados.append(crear_estado(1,0,'2/3 · hat y vee','hat convierte el vector en una matriz antisimétrica y vee realiza la operación inversa sin perder sus tres componentes independientes.'))
    for p in np.linspace(0,1,110): estados.append(crear_estado(suavizar(p),1,'3/3 · Twist xi en se(3)','Al añadir la parte traslacional rho obtenemos xi=[rho,phi] en R⁶ y su matriz xi^ en se(3). Aquí aún no convertimos este incremento local en una pose global.'))
    for _ in range(60): estados.append(crear_estado(1,1,'Conclusión · Espacio local','so(3) y se(3) permiten expresar perturbaciones e incrementos mediante vectores lineales, mientras SO(3) y SE(3) describen orientaciones y poses globales.'))
    return {'states':estados}
def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.8,8.9),interval=50)
    a.animate_3d_states(r['states'],'7.6. Álgebras de Lie so(3) y se(3)',limits=(-3.0,3.0,-3.0,3.0,-2.0,3.2),view=(24,-58),final_image_path=MATRICES_DIR/'assets'/'07_representaciones'/'06_algebras_lie.png',video_path=MATRICES_DIR/'assets'/'07_representaciones'/'06_algebras_lie.webm',repeat=False,fps=20,dpi=125,show=True)
if __name__=='__main__': main()
