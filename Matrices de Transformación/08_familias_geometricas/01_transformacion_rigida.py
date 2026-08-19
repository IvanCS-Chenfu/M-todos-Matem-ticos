from pathlib import Path
import sys
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
MATRICES_DIR = CURRENT_DIR.parent
sys.path.append(str(MATRICES_DIR))
from utils.transform_anim import TransformAnimator


def R2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def suavizar(p):
    return 0.5 - 0.5 * np.cos(np.pi * p)


def angulo(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.clip((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)
    return np.arccos(c)


def transformar_puntos(P, R, t):
    return (R @ P.T).T + t


def crear_estado(progreso, fase, mensaje):
    s = suavizar(progreso)
    theta = s * np.radians(-90.0)
    t = s * np.array([3.0, 2.0])
    R = R2(theta)

    figura = np.array([[-1.2,-0.7],[1.0,-0.7],[1.3,0.25],[0.25,1.15],[-1.1,0.55]])
    P = transformar_puntos(figura, R, t)

    # Dos lados adyacentes para medir longitud y ángulo.
    u0 = figura[1] - figura[0]
    v0 = figura[4] - figura[0]
    u1 = P[1] - P[0]
    v1 = P[4] - P[0]

    H = np.eye(3)
    H[:2,:2] = R
    H[:2,2] = t

    return {
        'polygons': [
            {'points': figura, 'facecolor':'#CBD5E1', 'edgecolor':'#64748B', 'alpha':0.18, 'linestyle':'--', 'linewidth':1.5},
            {'points': P, 'facecolor':'#93C5FD', 'edgecolor':'#1D4ED8', 'alpha':0.34, 'linewidth':2.0},
        ],
        'frames': [
            {'name':'A','origin':(0,0),'angle':0.0,'length':1.4,'alpha':0.22,
             'x_color':'#9CA3AF','y_color':'#9CA3AF'},
            {'name':'B','origin':t,'angle':theta,'length':1.35,'alpha':1.0},
        ],
        'vectors': [
            {'name':'t','origin':(0,0),'value':t,'color':'#E07A1F','linewidth':2.8},
        ] if np.linalg.norm(t)>1e-9 else [],
        'message': mensaje,
        'info_title': 'Transformación rígida 2D',
        'info_lines': [
            {'text':'EJEMPLO DE LA WIKI','bold':True},
            f'theta = {np.degrees(theta):6.1f}°',
            f't = [{t[0]:5.2f}, {t[1]:5.2f}]',
            '',
            {'text':'H = [R | t]','bold':True},
            f'[{H[0,0]:5.2f}, {H[0,1]:5.2f}, {H[0,2]:5.2f}]',
            f'[{H[1,0]:5.2f}, {H[1,1]:5.2f}, {H[1,2]:5.2f}]',
            '[ 0.00,  0.00,  1.00]',
            '',
            {'text':'INVARIANTES','bold':True},
            f'||u|| antes/después = {np.linalg.norm(u0):.3f} / {np.linalg.norm(u1):.3f}',
            f'||v|| antes/después = {np.linalg.norm(v0):.3f} / {np.linalg.norm(v1):.3f}',
            f'ángulo = {np.degrees(angulo(u0,v0)):.2f}° / {np.degrees(angulo(u1,v1)):.2f}°',
            f'det(R) = {np.linalg.det(R):.6f}',
        ],
        'phase': fase,
        'info_line_height':0.0385,
        'info_fontsize':8.5,
        'legend':[
            {'kind':'line','label':'figura inicial','color':'#64748B','linestyle':'--'},
            {'kind':'line','label':'figura transformada','color':'#1D4ED8'},
        ],
        'legend_fontsize':8.1,
    }


def crear_estados_demostracion():
    estados=[]
    for _ in range(28):
        estados.append(crear_estado(0.0,'1/3 · Configuración inicial','Una transformación rígida solo cambia la pose: la figura puede rotar y trasladarse, pero no deformarse.'))
    for p in np.linspace(0,1,150):
        estados.append(crear_estado(p,'2/3 · R(-90°) y t=[3,2]','Aplicamos progresivamente la rotación activa de -90° y la traslación [3,2]^T del ejemplo de la Wiki. Cada estado intermedio sigue siendo rígido.'))
    for _ in range(75):
        estados.append(crear_estado(1.0,'3/3 · Invariantes euclídeos','Las longitudes de los lados y el ángulo seleccionado coinciden antes y después. La transformación cambia pose, no forma.'))
    return {'states':estados}


def main():
    r=crear_estados_demostracion()
    a=TransformAnimator(figsize=(15.6,8.8),interval=50)
    a.animate_2d_states(r['states'],'8.1. Transformación euclídea o rígida',limits=(-3.0,5.5,-3.0,4.7),
        final_image_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'01_transformacion_rigida.png',
        video_path=MATRICES_DIR/'assets'/'08_familias_geometricas'/'01_transformacion_rigida.webm',
        repeat=False,fps=20,dpi=130,show=True)

if __name__=='__main__': main()
