"""
Genera videos anotados a partir de los resultados del tracking.

Dos modos:
  normal  todas las cajas, con ID y color estable por track
  foco    solo un ID, con la estela de su recorrido; el resto atenuado

Entrada: results.txt (formato MOTChallenge) + la carpeta img1/ de la
secuencia. Cualquier fuente que produzca ese mismo par sirve sin
modificar el script.

Uso:
  python -m src.tracking.visualize --config A --seq uav0000086_00000_v
  python -m src.tracking.visualize --config A --seq uav0000086_00000_v --focus-id 9
"""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2

PROJECT_ROOT=Path(__file__).parent.parent.parent
MOT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"motchallenge"
TRACKING_DIR=PROJECT_ROOT/"runs"/"tracking"
OUT_DIR=PROJECT_ROOT/"results"/"videos"

CONFIGS={
    "A":"A_yolov8n_bytetrack",
    "B":"B_yolov8n_deepsort",
    "C":"C_rtdetr_bytetrack",
    "D":"D_rtdetr_deepsort",
}

# clases en indexacion VisDrone-MOT (1-10), que es la que usan los
# results.txt tras aplicar el OFFSET_CLASE de run_tracking.py
NOMBRES_CLASE={
    1:"pedestrian",2:"people",3:"bicycle",4:"car",5:"van",
    6:"truck",7:"tricycle",8:"awning-tricycle",9:"bus",10:"motor",
}

# estela corta: en escenas densas el historial completo convierte la
# imagen en un espagueti ilegible
ESTELA_FRAMES=30


def color_por_id(track_id):
    """Color estable para cada ID: el mismo objeto siempre igual."""
    # multiplicar por primos dispersa los colores de IDs consecutivos
    r=(track_id*67)%255
    g=(track_id*113)%255
    b=(track_id*197)%255
    # se evita el negro puro, invisible sobre sombras
    return (int(max(b,60)),int(max(g,60)),int(max(r,60)))


def leer_tracks(ruta):
    """Agrupa el results.txt por frame: {n_frame: [(id,x,y,w,h,conf,cls),...]}"""
    por_frame=defaultdict(list)
    with open(ruta) as f:
        for linea in f:
            linea=linea.strip()
            if not linea:
                continue
            c=linea.split(",")
            n_frame=int(c[0])
            por_frame[n_frame].append((
                int(c[1]),float(c[2]),float(c[3]),
                float(c[4]),float(c[5]),float(c[6]),int(c[7]),
            ))
    return por_frame


def dibujar_caja(img,x,y,w,h,texto,color,grosor=2):
    x1,y1=int(x),int(y)
    x2,y2=int(x+w),int(y+h)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,grosor)

    if texto:
        escala=0.4
        (tw,th),_=cv2.getTextSize(texto,cv2.FONT_HERSHEY_SIMPLEX,escala,1)
        # fondo solido para que la etiqueta se lea sobre cualquier fondo
        cv2.rectangle(img,(x1,y1-th-4),(x1+tw+4,y1),color,-1)
        cv2.putText(img,texto,(x1+2,y1-3),cv2.FONT_HERSHEY_SIMPLEX,
                    escala,(0,0,0),1,cv2.LINE_AA)


def dibujar_estela(img,puntos,color):
    """Une los centros del objetivo; el grosor crece hacia el presente."""
    pts=list(puntos)
    for i in range(1,len(pts)):
        grosor=max(1,int(3*i/len(pts)))
        cv2.line(img,pts[i-1],pts[i],color,grosor,cv2.LINE_AA)
    if pts:
        cv2.circle(img,pts[-1],4,color,-1)


def generar(nombre_config,nombre_seq,focus_id,fps,max_frames):
    dir_config=TRACKING_DIR/CONFIGS[nombre_config]
    ruta_tracks=dir_config/"val"/f"{nombre_seq}.txt"
    if not ruta_tracks.exists():
        raise FileNotFoundError(ruta_tracks)

    dir_img=MOT_DIR/"val"/nombre_seq/"img1"
    frames=sorted(dir_img.glob("*.jpg"))
    if max_frames:
        frames=frames[:max_frames]
    if not frames:
        raise RuntimeError(f"sin frames en {dir_img}")

    por_frame=leer_tracks(ruta_tracks)

    alto,ancho=cv2.imread(str(frames[0])).shape[:2]

    sufijo=f"_foco{focus_id}" if focus_id is not None else "_normal"
    ruta_salida=OUT_DIR/f"{nombre_config}_{nombre_seq}{sufijo}.mp4"
    ruta_salida.parent.mkdir(parents=True,exist_ok=True)

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    writer=cv2.VideoWriter(str(ruta_salida),fourcc,fps,(ancho,alto))
    if not writer.isOpened():
        raise RuntimeError("no se pudo abrir el VideoWriter")

    estela=deque(maxlen=ESTELA_FRAMES)
    n_objetivo=0

    print(f"config   : {nombre_config}")
    print(f"secuencia: {nombre_seq}  {ancho}x{alto}")
    print(f"modo     : {'foco ID '+str(focus_id) if focus_id is not None else 'normal'}")
    print(f"frames   : {len(frames)} a {fps} fps\n")

    for i,ruta in enumerate(frames,start=1):
        img=cv2.imread(str(ruta))
        if img is None:
            raise RuntimeError(f"no se pudo leer {ruta}")

        tracks=por_frame.get(i,[])

        if focus_id is None:
            for tid,x,y,w,h,conf,cls in tracks:
                nombre=NOMBRES_CLASE.get(cls,"?")
                dibujar_caja(img,x,y,w,h,f"{nombre} #{tid}",color_por_id(tid))
        else:
            # primero los demas, atenuados, para que no tapen al objetivo
            for tid,x,y,w,h,conf,cls in tracks:
                if tid!=focus_id:
                    dibujar_caja(img,x,y,w,h,"",(120,120,120),1)

            objetivo=[t for t in tracks if t[0]==focus_id]
            if objetivo:
                tid,x,y,w,h,conf,cls=objetivo[0]
                estela.append((int(x+w/2),int(y+h/2)))
                color=color_por_id(tid)
                dibujar_estela(img,estela,color)
                nombre=NOMBRES_CLASE.get(cls,"?")
                dibujar_caja(img,x,y,w,h,f"{nombre} #{tid}",color,3)
                n_objetivo+=1

        # HUD: contexto minimo en la esquina
        hud=f"frame {i}/{len(frames)}  objetos: {len(tracks)}"
        if focus_id is not None:
            hud+=f"  |  foco #{focus_id}"
        cv2.putText(img,hud,(10,25),cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,255,255),2,cv2.LINE_AA)

        writer.write(img)

        if i%100==0:
            print(f"  {i}/{len(frames)}")

    writer.release()

    if focus_id is not None:
        print(f"\nel ID {focus_id} aparece en {n_objetivo}/{len(frames)} frames")
    print(f"video: {ruta_salida}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="A",choices=list(CONFIGS))
    ap.add_argument("--seq",required=True)
    ap.add_argument("--focus-id",type=int,default=None)
    ap.add_argument("--fps",type=int,default=30)
    ap.add_argument("--max-frames",type=int,default=None)
    args=ap.parse_args()

    generar(args.config,args.seq,args.focus_id,args.fps,args.max_frames)


if __name__=="__main__":
    main()