"""
Volcado de detecciones a disco (cache).

Corre un detector sobre todas las secuencias de un split y guarda las
cajas en un .npz por secuencia. Los trackers leen ese archivo en lugar
de reejecutar el detector.

Motivo doble:
  velocidad -> el detector es la parte cara; se ejecuta una sola vez
  rigor     -> las 4 configuraciones del factorial leen el MISMO archivo,
               asi cualquier diferencia en MOTA viene solo del tracker

Uso:
  python -m src.tracking.dump_detections --detector yolov8n --split val
  python -m src.tracking.dump_detections --detector rtdetr  --split val
  python -m src.tracking.dump_detections --detector yolov8n --dataset uavdt --split all

Fase 5 (Jetson) · opciones añadidas, sin cambiar el comportamiento anterior:
  --weights  otros pesos (p. ej. el best.engine de TensorRT FP16)
  --tag      nombre de la carpeta de salida en detections/ (por defecto, el del detector)
  python -m src.tracking.dump_detections --detector yolov8n --split test-dev \\
      --weights runs/yolov8n/yolov8n_mosaic10_20260606_0315/weights/best.engine --tag yolov8n_trt16
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.tracking.detectors import RTDETRDetector, YOLOv8nDetector

PROJECT_ROOT=Path(__file__).parent.parent.parent

#dataset -> carpeta base (motchallenge/ y detections/ cuelgan de ella)
#uavdt (Fase 4) solo tiene el split "all": las 50 secuencias
DATASETS={
    "visdrone":PROJECT_ROOT/"data"/"visdrone_mot",
    "uavdt":PROJECT_ROOT/"data"/"uavdt_mot",
}
SPLITS={"visdrone":["val","test-dev"], "uavdt":["all"]}

# nombre -> (clase adaptadora, ruta de pesos)
DETECTORES={
    "yolov8n":(
        YOLOv8nDetector,
        "runs/yolov8n/yolov8n_mosaic10_20260606_0315/weights/best.pt",
    ),
    "rtdetr":(
        RTDETRDetector,
        "runs/rtdetr/rtdetr_visdrone_20260606_1656/weights/best.pt",
    ),
}


def volcar_secuencia(detector,dir_seq,ruta_salida):
    """Detecta en todos los frames de una secuencia y guarda un .npz."""
    frames=sorted((dir_seq/"img1").glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"sin frames en {dir_seq}")

    # se guarda un array por frame dentro del mismo .npz, con la clave
    # "000001","000002",... asi cada frame conserva su propio (N,6)
    detecciones={}
    total_cajas=0
    t0=time.perf_counter()

    for ruta in frames:
        frame=cv2.imread(str(ruta))
        if frame is None:
            raise RuntimeError(f"no se pudo leer {ruta}")
        dets=detector.predict(frame)
        detecciones[ruta.stem]=dets
        total_cajas+=len(dets)

    segundos=time.perf_counter()-t0

    ruta_salida.parent.mkdir(parents=True,exist_ok=True)
    # compressed reduce mucho el tamano: los .npy sin comprimir de
    # RT-DETR ocupan bastante por las 300 propuestas fijas por frame
    np.savez_compressed(ruta_salida,**detecciones)

    return len(frames),total_cajas,segundos


def volcar_split(nombre_detector,nombre_split,dispositivo,dataset="visdrone",
                 pesos=None,tag=None):
    MOT_DIR=DATASETS[dataset]/"motchallenge"
    OUT_DIR=DATASETS[dataset]/"detections"
    clase,ruta_pesos=DETECTORES[nombre_detector]
    #Fase 5: pesos alternativos (.engine); rutas relativas a la raiz del proyecto
    if pesos:
        ruta_pesos=pesos
    ruta_pesos=Path(ruta_pesos)
    if not ruta_pesos.is_absolute():
        ruta_pesos=PROJECT_ROOT/ruta_pesos
    if not ruta_pesos.exists():
        raise FileNotFoundError(f"no existe {ruta_pesos}")
    #carpeta de salida: por defecto el nombre del detector (comportamiento anterior)
    tag=tag or nombre_detector

    dir_split=MOT_DIR/nombre_split
    if not dir_split.exists():
        raise FileNotFoundError(f"no existe {dir_split}")

    print(f"dataset : {dataset}")
    print(f"detector: {nombre_detector}")
    print(f"pesos   : {ruta_pesos}")
    print(f"split   : {nombre_split}")
    print(f"tag     : {tag}")

    detector=clase(ruta_pesos,dispositivo=dispositivo)

    dir_destino=OUT_DIR/tag/nombre_split
    secuencias=sorted(p for p in dir_split.iterdir() if p.is_dir())
    print(f"\n=== {len(secuencias)} secuencias ===")

    frames_tot=0
    cajas_tot=0
    seg_tot=0.0

    for dir_seq in secuencias:
        n,cajas,seg=volcar_secuencia(
            detector,dir_seq,dir_destino/f"{dir_seq.name}.npz"
        )
        frames_tot+=n
        cajas_tot+=cajas
        seg_tot+=seg
        print(f"  {dir_seq.name}: {n} frames  {cajas} cajas  "
              f"{seg:.1f}s  ({n/seg:.1f} FPS)")

    print(f"\ntotal: {frames_tot} frames  {cajas_tot} cajas  "
          f"{seg_tot:.1f}s  ({frames_tot/seg_tot:.1f} FPS)")
    print(f"media: {cajas_tot/frames_tot:.1f} cajas/frame")
    print(f"salida: {dir_destino}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--detector",required=True,choices=list(DETECTORES))
    ap.add_argument("--dataset",default="visdrone",choices=list(DATASETS))
    ap.add_argument("--split",required=True,choices=["val","test-dev","all"])
    ap.add_argument("--device",default="cuda:0")
    #Fase 5: pesos alternativos y carpeta de salida propia
    ap.add_argument("--weights",default=None,help="Pesos a usar (por defecto, el best.pt del detector)")
    ap.add_argument("--tag",default=None,help="Carpeta de salida en detections/ (por defecto, el nombre del detector)")
    args=ap.parse_args()
    if args.split not in SPLITS[args.dataset]:
        ap.error(f"{args.dataset} solo admite --split {'/'.join(SPLITS[args.dataset])}")

    volcar_split(args.detector,args.split,args.device,args.dataset,
                 pesos=args.weights,tag=args.tag)


if __name__=="__main__":
    main()
