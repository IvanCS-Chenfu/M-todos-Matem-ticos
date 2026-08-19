from pathlib import Path
import sys
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator


def rx(a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],float)
def ry(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],float)
def rz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
def rpy_a_R(roll,pitch,yaw):
    # Convención del apartado: RPY extrínseco XYZ / vectores columna.
    return rz(yaw) @ ry(pitch) @ rx(roll)
def R_a_rpy(R):
    # Rama principal ZYX equivalente a RPY extrínseco XYZ.
    pitch=np.arcsin(np.clip(-R[2,0],-1.0,1.0))
    cp=np.cos(pitch)
    if abs(cp)>1e-8:
        roll=np.arctan2(R[2,1],R[2,2])
        yaw=np.arctan2(R[1,0],R[0,0])
    else:
        # En singularidad roll/yaw quedan acoplados: fijamos yaw=0 como elección.
        roll=np.arctan2(-R[0,1],R[1,1])
        yaw=0.0
    return np.array([roll,pitch,yaw])
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)
def fmt_deg(v): return '['+', '.join(f'{np.degrees(x):6.1f}°' for x in v)+']'


def crear_estado(roll,pitch,yaw,fase,mensaje):
    R_roll=rx(roll)
    R_rp=ry(pitch)@R_roll
    R=rpy_a_R(roll,pitch,yaw)
    rec=R_a_rpy(R)
    # Ejes instantáneos de roll y yaw expresados en mundo. Cerca de pitch=90° se alinean.
    eje_roll_mundo=(rz(yaw)@ry(pitch))[:,0]
    eje_yaw_mundo=np.array([0.,0.,1.])
    alineacion=abs(float(eje_roll_mundo@eje_yaw_mundo))
    o=np.zeros(3)
    return {
      'frames3d':[
        {'name':'0','origin':o,'rotation':np.eye(3),'length':1.45,'alpha':0.18,'colors':('#9CA3AF',)*3},
        {'name':'roll','origin':o,'rotation':R_roll,'length':1.20,'alpha':0.18,'colors':('#B23A48','#9CA3AF','#9CA3AF')},
        {'name':'roll+pitch','origin':o,'rotation':R_rp,'length':1.28,'alpha':0.22,'colors':('#9CA3AF','#2D7F5E','#9CA3AF')},
        {'name':'RPY','origin':o,'rotation':R,'length':1.65,'alpha':1.0},
      ],
      'vectors3d':[
        {'name':'eje roll efectivo','origin':o,'value':1.8*eje_roll_mundo,'color':'#7B2CBF','linewidth':2.8},
        {'name':'eje yaw fijo','origin':o,'value':1.8*eje_yaw_mundo,'color':'#E07A1F','linewidth':2.8},
      ],
      'message':mensaje,
      'info_title':'Euler / RPY y gimbal lock',
      'info_lines':[
        {'text':'CONVENCIÓN','bold':True},
        'R = Rz(yaw) Ry(pitch) Rx(roll)',
        '',
        f'roll  = {np.degrees(roll):6.1f}°',
        f'pitch = {np.degrees(pitch):6.1f}°',
        f'yaw   = {np.degrees(yaw):6.1f}°',
        '',
        {'text':'RECUPERACIÓN R -> RPY','bold':True},
        f'RPY rec = {fmt_deg(rec)}',
        f'error R = {np.linalg.norm(R-rpy_a_R(*rec)):.2e}',
        '',
        {'text':'CERCA DE GIMBAL LOCK','bold':True},
        f'|e_roll · e_yaw| = {alineacion:.4f}',
        '1.0 => ejes alineados',
      ],
      'phase':fase,'info_line_height':0.0395,'info_fontsize':8.7,
      'legend':[
        {'kind':'line','label':'eje roll efectivo','color':'#7B2CBF'},
        {'kind':'line','label':'eje yaw global','color':'#E07A1F'},
      ],'legend_fontsize':8.0,
    }

def crear_estados_demostracion():
    estados=[]; rf=np.radians(35); yf=np.radians(50)
    for p in np.linspace(0,1,90):
        s=suavizar(p); estados.append(crear_estado(s*rf,0,0,'1/4 · Roll','Primero aplicamos roll alrededor de X. La secuencia y la convención deben quedar fijadas antes de interpretar los tres ángulos.'))
    for p in np.linspace(0,1,110):
        s=suavizar(p); estados.append(crear_estado(rf,s*np.radians(70),0,'2/4 · Pitch','Añadimos pitch alrededor de Y fijo. La matriz total ya depende del orden de composición.'))
    for p in np.linspace(0,1,90):
        s=suavizar(p); estados.append(crear_estado(rf,np.radians(70),s*yf,'3/4 · Yaw','Añadimos yaw alrededor de Z fijo y recuperamos numéricamente la tripleta RPY desde la matriz resultante.'))
    for p in np.linspace(0,1,130):
        s=suavizar(p); pitch=np.radians(70+19.9*s); estados.append(crear_estado(rf,pitch,yf,'4/4 · Acercarse a pitch=90°','Al acercarse pitch a 90°, los ejes efectivos asociados a roll y yaw se alinean: la orientación sigue siendo válida, pero RPY pierde independencia local.'))
    for _ in range(55): estados.append(crear_estado(rf,np.radians(89.9),yf,'Conclusión · Gimbal lock','Gimbal lock es una singularidad de la parametrización RPY, no de la orientación física.'))
    return {'states':estados}

def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(15.8,8.9),interval=50)
    a.animate_3d_states(r['states'],'7.1. Ángulos de Euler y roll-pitch-yaw',limits=(-3.2,3.2,-3.2,3.2,-2.6,3.2),view=(24,-58),final_image_path=MATRICES_DIR/'assets'/'07_representaciones'/'01_euler_rpy.png',video_path=MATRICES_DIR/'assets'/'07_representaciones'/'01_euler_rpy.webm',repeat=False,fps=20,dpi=125,show=True)
if __name__=='__main__': main()
