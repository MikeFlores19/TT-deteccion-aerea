"""
Ejecuta una configuracion del factorial 2x2 sobre un split.

Lee las detecciones cacheadas (.npz), las pasa por el tracker y escribe
un results.txt por secuencia en formato MOTChallenge.

Las 4 configuraciones leen los MISMOS .npz, asi que cualquier diferencia
en las metricas es atribuible unicamente al tracker o al detector, nunca
a variaciones en la deteccion.

Uso:
  python -m src.tracking.run_tracking --config A --split val
  python -m src.tracking.run_tracking --config B --split val
  python -m src.tracking.run_tracking --config C --split val
  python -m src.tracking.run_tracking --config D --split val
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.tracking.trackers import ByteTrackAdapter, DeepSortAdapter

PROJECT_ROOT=Path(__file__).parent.parent.parent
MOT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"motchallenge"
DETS_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"detections"
OUT_DIR=PROJECT_ROOT/"runs"/"tracking"

# el factorial 2x2: (detector, tracker)
CONFIGS={
    "A":("yolov8n","bytetrack"),
    "B":("yolov8n","deepsort"),
    "C":("rtdetr","bytetrack"),
    "D":("rtdetr","deepsort"),
}

# desfase de indexacion: YOLO 0-9 -> VisDrone-MOT 1-10
# VisDrone reserva el 0 para ignored-region, por eso sus objetos
# empiezan en 1 mientras que YOLO exige empezar en 0
OFFSET_CLASE=1


def construir_tracker(nombre):
    if nombre=="bytetrack":
        return ByteTrackAdapter()
    if nombre=="deepsort":
        return DeepSortAdapter()
    raise ValueError(nombre)


def rastrear_secuencia(dets_npz,dir_seq,nombre_tracker,ruta_salida):
    """Aplica el tracker a una secuencia y escribe su results.txt."""
    datos=np.load(dets_npz)
    # las claves son los nombres de frame: "000001","000002",...
    claves=sorted(datos.files)

    tracker=construir_tracker(nombre_tracker)
    dir_img=dir_seq/"img1"

    lineas=[]
    ids_vistos=set()
    t0=time.perf_counter()

    for clave in claves:
        dets=datos[clave]
        n_frame=int(clave)

        # boxmot valida que img sea ndarray aunque ByteTrack solo asocie
        # por geometria, asi que el frame se lee siempre
        frame=cv2.imread(str(dir_img/f"{clave}.jpg"))
        if frame is None:
            raise RuntimeError(f"no se pudo leer {dir_img/clave}.jpg")

        tracks=tracker.update(dets,frame)

        for x1,y1,x2,y2,tid,conf,cls in tracks:
            # el contrato usa esquinas; MOTChallenge pide left,top,w,h
            w=x2-x1
            h=y2-y1
            cls_mot=int(cls)+OFFSET_CLASE
            tid=int(tid)
            ids_vistos.add(tid)
            lineas.append(
                f"{n_frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                f"{conf:.4f},{cls_mot},-1,-1"
            )

    segundos=time.perf_counter()-t0

    ruta_salida.parent.mkdir(parents=True,exist_ok=True)
    with open(ruta_salida,"w") as f:
        f.write("\n".join(lineas)+"\n")

    return len(claves),len(lineas),len(ids_vistos),segundos


def ejecutar(nombre_config,nombre_split):
    detector,tracker=CONFIGS[nombre_config]

    dir_dets=DETS_DIR/detector/nombre_split
    if not dir_dets.exists():
        raise FileNotFoundError(
            f"faltan detecciones en {dir_dets}. "
            f"Corre antes dump_detections.py --detector {detector}"
        )

    dir_split=MOT_DIR/nombre_split
    dir_destino=OUT_DIR/f"{nombre_config}_{detector}_{tracker}"/nombre_split

    print(f"config  : {nombre_config}  ({detector} + {tracker})")
    print(f"split   : {nombre_split}")
    print(f"salida  : {dir_destino}")

    archivos=sorted(dir_dets.glob("*.npz"))
    print(f"\n=== {len(archivos)} secuencias ===")

    frames_tot=0
    lineas_tot=0
    seg_tot=0.0

    for npz in archivos:
        nombre=npz.stem
        n,lineas,ids,seg=rastrear_secuencia(
            npz,dir_split/nombre,tracker,dir_destino/f"{nombre}.txt"
        )
        frames_tot+=n
        lineas_tot+=lineas
        seg_tot+=seg
        print(f"  {nombre}: {n} frames  {lineas} tracks-frame  "
              f"{ids} IDs unicos  {seg:.1f}s")

    print(f"\ntotal: {frames_tot} frames  {lineas_tot} tracks-frame  "
          f"{seg_tot:.1f}s")
    print(f"media: {lineas_tot/frames_tot:.1f} tracks/frame")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True,choices=list(CONFIGS))
    ap.add_argument("--split",required=True,choices=["val","test-dev"])
    args=ap.parse_args()

    ejecutar(args.config,args.split)


if __name__=="__main__":
    main()