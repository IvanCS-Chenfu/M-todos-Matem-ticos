from pathlib import Path
import sys
import numpy as np
try:
    import cv2
except ImportError:
    cv2=None

CURRENT_DIR=Path(__file__).resolve().parent
MATRICES_DIR=CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator


def rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],float)
def ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],float)
def rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
def suavizar(p): return 0.5-0.5*np.cos(np.pi*p)

def proyectar(Pw,Rcw,tcw,K):
    Pc=(Rcw@Pw.T).T+tcw
    q=(K@Pc.T).T
    uv=q[:,:2]/q[:,2,None]
    return Pc,uv

def dlt_homografia(src,dst):
    A=[]
    for (x,y),(u,v) in zip(src,dst):
        A += [[-x,-y,-1,0,0,0,u*x,u*y,u],[0,0,0,-x,-y,-1,v*x,v*y,v]]
    _,_,V=np.linalg.svd(np.asarray(A,float)); H=V[-1].reshape(3,3)
    return H/H[2,2]
def homografia_4(src,dst):
    if cv2 is not None:
        return cv2.getPerspectiveTransform(np.asarray(src,np.float32),np.asarray(dst,np.float32)).astype(float)
    return dlt_homografia(src,dst)
def aplicar_H(P,H):
    q=(H@np.c_[P,np.ones(len(P))].T).T; return q[:,:2]/q[:,2,None]
def tablero(nx=7,ny=5):
    xs=np.linspace(-1.6,1.6,nx); ys=np.linspace(-1.1,1.1,ny)
    points=np.array([[x,y,0.] for y in ys for x in xs])
    lines=[]
    for y in ys: lines.append(np.array([[x,y,0.] for x in np.linspace(xs[0],xs[-1],45)]))
    for x in xs: lines.append(np.array([[x,y,0.] for y in np.linspace(ys[0],ys[-1],45)]))
    corners=np.array([[xs[0],ys[0],0.],[xs[-1],ys[0],0.],[xs[-1],ys[-1],0.],[xs[0],ys[-1],0.]])
    return points,lines,corners

def display_uv(uv,offset=np.zeros(2),height=480.0):
    uv=np.asarray(uv,float); return np.c_[uv[:,0]+offset[0], height-uv[:,1]+offset[1]]


def crear_estado(pose_s,rect_s,fase,mensaje):
    s=suavizar(pose_s)
    # Extrínseca world -> camera. El tablero está en Z=0; t_z=6 mantiene todos los puntos delante de la cámara.
    Rcw=rx(s*np.radians(18))@ry(s*np.radians(-22))@rz(s*np.radians(8))
    tcw=np.array([0.25*s,-0.15*s,6.0])
    K=np.array([[650.,0.,320.],[0.,650.,240.],[0.,0.,1.]])
    _,lines3,corners=tablero()
    Pc_c,uv_c=proyectar(corners,Rcw,tcw,K)
    target=np.array([[780.,110.],[1180.,110.],[1180.,410.],[780.,410.]])
    Hrect=homografia_4(uv_c,target)

    polylines=[]
    # Vista proyectada a la izquierda.
    for line3 in lines3:
        _,uv=proyectar(line3,Rcw,tcw,K)
        polylines.append({'points':display_uv(uv),'color':'#2563EB','linewidth':1.35,'alpha':0.62})
    # Contorno de la imagen izquierda y área rectificada derecha.
    polylines += [
      {'points':np.array([[0,0],[640,0],[640,480],[0,480],[0,0]]),'color':'#64748B','linewidth':1.6,'alpha':0.6},
      {'points':np.array([[740,60],[1220,60],[1220,460],[740,460],[740,60]]),'color':'#64748B','linewidth':1.6,'alpha':0.6},
    ]

    # Rectificación: cada línea proyectada se transforma con H y se mezcla visualmente desde la vista inclinada trasladada a la derecha.
    r=suavizar(rect_s)
    for line3 in lines3:
        _,uv=proyectar(line3,Rcw,tcw,K)
        rect=aplicar_H(uv,Hrect)
        inicio=uv + np.array([680.,0.])
        mix=(1-r)*inicio+r*rect
        polylines.append({'points':display_uv(mix,offset=np.array([0.,0.])),'color':'#D97706','linewidth':1.25,'alpha':0.72})

    # Punto 3D seleccionado del tablero y su píxel.
    Pw=np.array([[1.0,0.55,0.]])
    Pc,uv=proyectar(Pw,Rcw,tcw,K)
    pdisp=display_uv(uv)[0]
    return {
      'polylines':polylines,
      'points':[{'name':'píxel','position':pdisp,'color':'#7B2CBF','size':85}],
      'texts':[
        {'position':np.array([200.,505.]),'text':'PROYECCIÓN DE CÁMARA','fontweight':'bold','color':'#2563EB'},
        {'position':np.array([875.,505.]),'text':'HOMOGRAFÍA / RECTIFICACIÓN','fontweight':'bold','color':'#D97706'},
      ],
      'message':mensaje,'info_title':'Pose, K y homografía',
      'info_lines':[
        {'text':'PUNTO 3D -> CÁMARA','bold':True},'P_world = [1.00, 0.55, 0.00]',f'P_cam   = [{Pc[0,0]:5.2f},{Pc[0,1]:5.2f},{Pc[0,2]:5.2f}]','',
        {'text':'CÁMARA -> PÍXEL','bold':True},'fx=fy=650, cx=320, cy=240',f'(u,v)=({uv[0,0]:6.1f},{uv[0,1]:6.1f})','s p_img = K [R|t] P_world,h','',
        {'text':'PLANO -> HOMOGRAFÍA','bold':True},f'det(H_rect)={np.linalg.det(Hrect):.3e}',f'OpenCV: {cv2 is not None}','pose [R,t] pertenece a SE(3)','H_rect es proyectiva 2D',
      ],'phase':fase,'info_line_height':0.035,'info_fontsize':8.15,
      'legend':[{'kind':'line','label':'vista en perspectiva','color':'#2563EB'},{'kind':'line','label':'rectificación','color':'#D97706'},{'kind':'point','label':'píxel seleccionado','color':'#7B2CBF'}],'legend_fontsize':7.8,
    }

def crear_estados_demostracion():
    estados=[]
    for _ in range(28): estados.append(crear_estado(0,0,'1/4 · Cámara frontal','Con la cámara frontal, el tablero plano se proyecta casi como un rectángulo. La pose [R,t] transforma puntos 3D y K los convierte en píxeles.'))
    for p in np.linspace(0,1,145): estados.append(crear_estado(p,0,'2/4 · Cambiar la pose de cámara','Inclinamos la cámara. Los puntos pasan primero de world a camera mediante una transformación rígida de SE(3) y después se proyectan con K.'))
    for _ in range(55): estados.append(crear_estado(1,0,'3/4 · Homografía entre planos','Como todos los puntos pertenecen al mismo plano, las cuatro esquinas permiten construir una homografía 2D que relaciona la vista en perspectiva con un rectángulo objetivo.'))
    for p in np.linspace(0,1,130): estados.append(crear_estado(1,p,'4/4 · Rectificar la perspectiva','Aplicamos la homografía a la cuadrícula proyectada hasta recuperar una vista frontal rectangular. H y la pose [R,t] son matrices homogéneas, pero representan objetos geométricos distintos.'))
    for _ in range(80): estados.append(crear_estado(1,1,'Conclusión · No confundir H con T','La pose pertenece a SE(3) y actúa sobre geometría 3D rígida; la homografía actúa sobre coordenadas proyectivas 2D de un plano y puede modelar perspectiva.'))
    return {'states':estados}

def main():
    r=crear_estados_demostracion(); a=TransformAnimator(figsize=(16.2,8.9),interval=50)
    a.animate_2d_states(r['states'],'8.6. Relación con visión por computador',limits=(-30,1280,-40,550),
      final_image_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'06_vision_computador.png',
      video_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'06_vision_computador.webm',repeat=False,fps=20,dpi=125,show=True)
if __name__=='__main__': main()
