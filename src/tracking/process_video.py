"""
Fase 5 · Pasos 5.2 y 5.3 — Pipeline de la config A sobre video (o secuencia de imagenes).

    fuente (.mp4 o carpeta img1/) -> YOLOv8n (.engine o .pt) -> ByteTrack -> salidas

A diferencia de dump_detections.py + run_tracking.py (que primero detectan todo y
luego rastrean), aqui cada frame pasa por detector y tracker en el mismo ciclo:
es el sistema completo tal como se usara con los videos del dron.

Reutiliza sin cambios detectors.py (YOLOv8nDetector) y trackers.py (ByteTrackAdapter,
mismos parametros de la Fase 3), asi que con la misma fuente y los mismos pesos el
results.txt debe coincidir con el de run_tracking.py (validacion 1 del paso 5.2).

Salidas:
    runs/video/<nombre>/results.txt    formato MOTChallenge (igual que run_tracking.py)
    runs/video/<nombre>/eventos.csv    registro estructurado: 1 fila por trayectoria (paso 5.3)
    runs/video/<nombre>/resumen.json   frames, tiempos por etapa y FPS del pipeline
    results/videos/jetson/<nombre>_normal.mp4   video anotado (mismo estilo que visualize.py)

Evento = trayectoria (un ID de ByteTrack). Columnas de eventos.csv:
    track_id, clase (la mas frecuente del track), clase_id (VisDrone-MOT 1-10),
    n_frames, frame_inicio, frame_fin, t_inicio_s, t_fin_s, duracion_s,
    conf_media, conf_max, frame_mejor, x_centro, y_centro
    t_* en segundos desde el primer frame de la fuente (la hora absoluta y el GPS
    se agregan en el paso 9 con la telemetria). x/y_centro: centro de la caja en el
    frame de mayor confianza, en pixeles de la imagen original.

Uso (desde la raiz del proyecto):
    python -m src.tracking.process_video \\
        --source data/visdrone_mot/motchallenge/test-dev/uav0000119_02301_v/img1 \\
        --weights runs/yolov8n/yolov8n_mosaic10_20260606_0315/weights/best_736x1280.engine
    python -m src.tracking.process_video --source videos/vuelo1.mp4 \\
        --weights runs/yolov8n/yolov8n_mosaic10_20260606_0315/weights/best_736x1280.engine
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from src.tracking.detectors import YOLOv8nDetector
from src.tracking.trackers import ByteTrackAdapter
from src.tracking.visualize import NOMBRES_CLASE, color_por_id, dibujar_caja

PROJECT_ROOT=Path(__file__).parent.parent.parent
RUNS_DIR=PROJECT_ROOT/"runs"/"video"
VIDEOS_DIR=PROJECT_ROOT/"results"/"videos"/"jetson"
PESOS_DEF="runs/yolov8n/yolov8n_mosaic10_20260606_0315/weights/best.pt"

#desfase de indexacion: YOLO 0-9 -> VisDrone-MOT 1-10 (igual que run_tracking.py)
OFFSET_CLASE=1
EXT_IMG={".jpg",".jpeg",".png"}


class Fuente:
    """Lee frames de un video o de una carpeta de imagenes con la misma interfaz.

    Itera (n_frame, frame). En una carpeta, n_frame sale del nombre del archivo
    (000001.jpg -> 1), igual que run_tracking.py; en un video, cuenta desde 1.
    """

    def __init__(self,ruta,fps=None,max_frames=None):
        self.ruta=Path(ruta)
        self.max_frames=max_frames
        if self.ruta.is_dir():
            self.tipo="imagenes"
            self.archivos=sorted(p for p in self.ruta.iterdir() if p.suffix.lower() in EXT_IMG)
            if not self.archivos:
                raise RuntimeError(f"sin imagenes en {self.ruta}")
            self.fps=fps or self._fps_seqinfo() or 30.0
            alto,ancho=cv2.imread(str(self.archivos[0])).shape[:2]
            self.total=len(self.archivos)
        elif self.ruta.is_file():
            self.tipo="video"
            self.cap=cv2.VideoCapture(str(self.ruta))
            if not self.cap.isOpened():
                raise RuntimeError(f"OpenCV no pudo abrir {self.ruta}")
            self.fps=fps or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            ancho=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        else:
            raise FileNotFoundError(self.ruta)
        self.ancho,self.alto=ancho,alto
        if max_frames and self.total:
            self.total=min(self.total,max_frames)

    def _fps_seqinfo(self):
        #secuencias MOTChallenge: seqinfo.ini junto a img1/
        ini=self.ruta.parent/"seqinfo.ini"
        if ini.exists():
            for linea in ini.read_text().splitlines():
                if linea.startswith("frameRate="):
                    return float(linea.split("=")[1])
        return None

    def __iter__(self):
        if self.tipo=="imagenes":
            for i,p in enumerate(self.archivos,start=1):
                if self.max_frames and i>self.max_frames:
                    break
                frame=cv2.imread(str(p))
                if frame is None:
                    raise RuntimeError(f"no se pudo leer {p}")
                yield (int(p.stem) if p.stem.isdigit() else i),frame
        else:
            i=0
            while True:
                if self.max_frames and i>=self.max_frames:
                    break
                ok,frame=self.cap.read()
                if not ok:
                    break
                i+=1
                yield i,frame
            self.cap.release()


class RegistroEventos:
    """Acumula, por ID, lo necesario para una fila de eventos.csv."""

    def __init__(self):
        self.tracks={}

    def agregar(self,n_frame,tid,x1,y1,x2,y2,conf,cls_mot):
        t=self.tracks.get(tid)
        if t is None:
            t=self.tracks[tid]={"n":0,"ini":n_frame,"fin":n_frame,"suma_conf":0.0,
                               "conf_max":-1.0,"frame_mejor":n_frame,"centro":(0.0,0.0),
                               "clases":Counter()}
        t["n"]+=1
        t["fin"]=n_frame
        t["suma_conf"]+=conf
        t["clases"][cls_mot]+=1
        if conf>t["conf_max"]:
            t["conf_max"]=conf
            t["frame_mejor"]=n_frame
            t["centro"]=((x1+x2)/2,(y1+y2)/2)

    def escribir(self,ruta,fps,frame0,min_frames=1):
        filas=[]
        for tid,t in sorted(self.tracks.items()):
            if t["n"]<min_frames:
                continue
            cls_mot=t["clases"].most_common(1)[0][0]
            t_ini=(t["ini"]-frame0)/fps
            t_fin=(t["fin"]-frame0)/fps
            filas.append({
                "track_id":tid,
                "clase":NOMBRES_CLASE.get(cls_mot,"?"),
                "clase_id":cls_mot,
                "n_frames":t["n"],
                "frame_inicio":t["ini"],
                "frame_fin":t["fin"],
                "t_inicio_s":round(t_ini,3),
                "t_fin_s":round(t_fin,3),
                "duracion_s":round(t_fin-t_ini,3),
                "conf_media":round(t["suma_conf"]/t["n"],4),
                "conf_max":round(t["conf_max"],4),
                "frame_mejor":t["frame_mejor"],
                "x_centro":round(t["centro"][0],1),
                "y_centro":round(t["centro"][1],1),
            })
        ruta.parent.mkdir(parents=True,exist_ok=True)
        with open(ruta,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(filas[0].keys()) if filas else ["track_id"])
            w.writeheader()
            w.writerows(filas)
        return filas


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def procesar(args):
    fuente=Fuente(args.source,fps=args.fps,max_frames=args.max_frames)
    nombre=args.name or (fuente.ruta.parent.name if fuente.ruta.name=="img1" else fuente.ruta.stem)

    pesos=Path(args.weights)
    if not pesos.is_absolute():
        pesos=PROJECT_ROOT/pesos
    if not pesos.exists():
        raise FileNotFoundError(f"no existen los pesos: {pesos}")

    dir_salida=RUNS_DIR/nombre
    dir_salida.mkdir(parents=True,exist_ok=True)

    detector=YOLOv8nDetector(pesos,dispositivo=args.device)
    #mismos parametros que la config A de la Fase 3 (valores por defecto del adaptador)
    tracker=ByteTrackAdapter()
    eventos=RegistroEventos()

    writer=None
    ruta_video=None
    if not args.no_video:
        ancho_v=int(fuente.ancho*args.video_scale)
        alto_v=int(fuente.alto*args.video_scale)
        ruta_video=VIDEOS_DIR/f"{nombre}_normal.mp4"
        ruta_video.parent.mkdir(parents=True,exist_ok=True)
        #mp4v por CPU: la Orin Nano no tiene codificador de video por hardware (NVENC)
        writer=cv2.VideoWriter(str(ruta_video),cv2.VideoWriter_fourcc(*"mp4v"),
                               fuente.fps,(ancho_v,alto_v))
        if not writer.isOpened():
            raise RuntimeError("no se pudo abrir el VideoWriter")

    print(f"fuente : {rel(fuente.ruta)} ({fuente.tipo}, {fuente.ancho}x{fuente.alto}, "
          f"{fuente.fps:.2f} fps, {fuente.total or '?'} frames)")
    print(f"pesos  : {rel(pesos)} (imgsz {detector.imgsz})")
    print(f"salida : {rel(dir_salida)}")
    if ruta_video:
        print(f"video  : {rel(ruta_video)}")
    print()

    #tiempos por etapa (s): lectura+decodificacion, deteccion, tracking, escritura del video
    t={"lectura":0.0,"deteccion":0.0,"tracking":0.0,"video":0.0}
    lineas=[]
    n_frames=0
    frame0=None
    t_inicio=time.perf_counter()

    it=iter(fuente)
    while True:
        a=time.perf_counter()
        try:
            n_frame,frame=next(it)
        except StopIteration:
            break
        b=time.perf_counter()
        dets=detector.predict(frame)
        c=time.perf_counter()
        tracks=tracker.update(dets,frame)
        d=time.perf_counter()
        t["lectura"]+=b-a
        t["deteccion"]+=c-b
        t["tracking"]+=d-c

        if frame0 is None:
            frame0=n_frame
        n_frames+=1

        for x1,y1,x2,y2,tid,conf,cls in tracks:
            #mismo formato que run_tracking.py: left,top,w,h y clase VisDrone-MOT
            cls_mot=int(cls)+OFFSET_CLASE
            tid=int(tid)
            lineas.append(f"{n_frame},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},"
                          f"{conf:.4f},{cls_mot},-1,-1")
            eventos.agregar(n_frame,tid,float(x1),float(y1),float(x2),float(y2),float(conf),cls_mot)

        if writer is not None:
            e=time.perf_counter()
            for x1,y1,x2,y2,tid,conf,cls in tracks:
                cls_mot=int(cls)+OFFSET_CLASE
                dibujar_caja(frame,x1,y1,x2-x1,y2-y1,
                             f"{NOMBRES_CLASE.get(cls_mot,'?')} #{int(tid)}",color_por_id(int(tid)))
            hud=f"frame {n_frame}  objetos: {len(tracks)}"
            cv2.putText(frame,hud,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
            if args.video_scale!=1.0:
                frame=cv2.resize(frame,(ancho_v,alto_v),interpolation=cv2.INTER_AREA)
            writer.write(frame)
            t["video"]+=time.perf_counter()-e

        if n_frames%100==0:
            seg=time.perf_counter()-t_inicio
            print(f"  {n_frames}/{fuente.total or '?'} frames  {n_frames/seg:.1f} FPS")

    t_total=time.perf_counter()-t_inicio
    if writer is not None:
        writer.release()
    if n_frames==0:
        raise RuntimeError("la fuente no produjo frames")

    (dir_salida/"results.txt").write_text("\n".join(lineas)+"\n")
    filas=eventos.escribir(dir_salida/"eventos.csv",fuente.fps,frame0,args.min_frames)

    seg_pipeline=t["lectura"]+t["deteccion"]+t["tracking"]
    resumen={
        "fuente":rel(fuente.ruta),"tipo":fuente.tipo,
        "resolucion":[fuente.ancho,fuente.alto],"fps_fuente":round(fuente.fps,3),
        "pesos":rel(pesos),"imgsz":detector.imgsz,
        "frames":n_frames,"tracks_frame":len(lineas),
        "ids_unicos":len(eventos.tracks),"eventos":len(filas),"min_frames":args.min_frames,
        "segundos":{k:round(v,2) for k,v in t.items()}|{"total":round(t_total,2)},
        "ms_por_frame":{k:round(1000*v/n_frames,2) for k,v in t.items()},
        #FPS del sistema (lectura+deteccion+tracking) y con la escritura del video incluida
        "fps_pipeline":round(n_frames/seg_pipeline,2),
        "fps_total":round(n_frames/t_total,2),
        "video_anotado":rel(ruta_video) if ruta_video else None,
        "nota":"Tiempos de referencia del pipeline; el benchmark formal es el paso 6.",
    }
    (dir_salida/"resumen.json").write_text(json.dumps(resumen,indent=2,ensure_ascii=False))

    print(f"\ntotal: {n_frames} frames  {len(lineas)} tracks-frame  "
          f"{len(eventos.tracks)} IDs unicos  {len(filas)} eventos")
    print("ms/frame: "+"  ".join(f"{k}={v}" for k,v in resumen["ms_por_frame"].items()))
    print(f"FPS pipeline={resumen['fps_pipeline']}  FPS total={resumen['fps_total']}")
    print(f"salidas: {rel(dir_salida)}/{{results.txt,eventos.csv,resumen.json}}")
    if ruta_video:
        print(f"video  : {rel(ruta_video)}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,help="Video (.mp4, .mov...) o carpeta de imagenes (img1/)")
    ap.add_argument("--weights",default=PESOS_DEF,help="Pesos .pt o .engine (en la Jetson, best_736x1280.engine)")
    ap.add_argument("--name",default=None,help="Nombre de la salida (por defecto, el de la secuencia o del video)")
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--fps",type=float,default=None,help="Forzar FPS de la fuente (por defecto, el del video o seqinfo.ini)")
    ap.add_argument("--max-frames",type=int,default=None,help="Procesar solo los primeros N frames (pruebas)")
    ap.add_argument("--min-frames",type=int,default=1,help="Descartar de eventos.csv los tracks con menos frames")
    ap.add_argument("--no-video",action="store_true",help="No generar el video anotado")
    ap.add_argument("--video-scale",type=float,default=1.0,help="Escala del video anotado (p. ej. 0.5 para 4K)")
    procesar(ap.parse_args())


if __name__=="__main__":
    main()
