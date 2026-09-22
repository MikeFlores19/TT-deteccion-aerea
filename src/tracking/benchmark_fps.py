"""
Mide el rendimiento real de las 4 configuraciones del factorial.

A diferencia de run_tracking.py (que lee detecciones cacheadas), aqui
el detector y el tracker corren en el MISMO bucle, como ocurriria a
bordo del dron.

Los frames se PRECARGAN en RAM antes de cronometrar: la camara IMX219
entrega el frame en memoria, no como .jpg en disco, asi que incluir la
lectura de disco mediria un costo que en vuelo no existe.

Esto mide SOLO tiempo. La calidad ya esta medida en el paso 7; el
cronometro no depende de si el sistema acierta o falla.

Uso:
  python -m src.tracking.benchmark_fps --config A
  python -m src.tracking.benchmark_fps --config A B C D
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.tracking.detectors import RTDETRDetector, YOLOv8nDetector
from src.tracking.trackers import ByteTrackAdapter, DeepSortAdapter

PROJECT_ROOT=Path(__file__).parent.parent.parent
MOT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"motchallenge"/"val"
OUT_DIR=PROJECT_ROOT/"results"/"tables"/"eficiencia"
TRACKEVAL_DIR=PROJECT_ROOT/"results"/"tables"/"tracking"

CONFIGS={
    "A":("yolov8n","bytetrack"),
    "B":("yolov8n","deepsort"),
    "C":("rtdetr","bytetrack"),
    "D":("rtdetr","deepsort"),
}

PESOS={
    "yolov8n":"runs/yolov8n/yolov8n_mosaic10_20260606_0315/weights/best.pt",
    "rtdetr":"runs/rtdetr/rtdetr_visdrone_20260606_1656/weights/best.pt",
}

CLASES_DETECTOR={"yolov8n":YOLOv8nDetector,"rtdetr":RTDETRDetector}

# dos resoluciones para ver como escala el costo con el tamano de entrada
SECUENCIAS=["uav0000086_00000_v","uav0000268_05773_v"]

# frames de calentamiento: las primeras inferencias son mas lentas por
# la inicializacion de CUDA y la asignacion de buffers
WARMUP=10

# limite de frames por secuencia: suficiente para una media estable sin
# cargar 1000 imagenes de 4K en RAM
MAX_FRAMES=200


def precargar(nombre_seq,max_frames):
    """Carga los frames en RAM antes de cronometrar."""
    dir_img=MOT_DIR/nombre_seq/"img1"
    rutas=sorted(dir_img.glob("*.jpg"))[:max_frames]
    frames=[]
    for ruta in rutas:
        f=cv2.imread(str(ruta))
        if f is None:
            raise RuntimeError(f"no se pudo leer {ruta}")
        frames.append(f)
    return frames


def construir(nombre_config):
    detector_nom,tracker_nom=CONFIGS[nombre_config]
    ruta=PROJECT_ROOT/PESOS[detector_nom]
    if not ruta.exists():
        raise FileNotFoundError(ruta)
    detector=CLASES_DETECTOR[detector_nom](ruta)
    tracker=ByteTrackAdapter() if tracker_nom=="bytetrack" else DeepSortAdapter()
    return detector,tracker,detector_nom,tracker_nom


def medir_secuencia(nombre_config,nombre_seq,frames):
    """Cronometra deteccion y seguimiento por separado sobre los frames."""
    detector,tracker,det_nom,trk_nom=construir(nombre_config)

    # calentamiento: no se cronometra
    for f in frames[:WARMUP]:
        tracker.update(detector.predict(f),f)

    # el tracker acumulo estado durante el calentamiento; se reinicia
    _,tracker,_,_=construir(nombre_config)

    t_det=[]
    t_trk=[]
    torch.cuda.synchronize()

    for f in frames:
        t0=time.perf_counter()
        dets=detector.predict(f)
        torch.cuda.synchronize()
        t1=time.perf_counter()
        tracker.update(dets,f)
        torch.cuda.synchronize()
        t2=time.perf_counter()
        t_det.append((t1-t0)*1000)
        t_trk.append((t2-t1)*1000)

    vram=torch.cuda.max_memory_allocated()/1024**2
    torch.cuda.reset_peak_memory_stats()

    det_ms=float(np.mean(t_det))
    trk_ms=float(np.mean(t_trk))
    total_ms=det_ms+trk_ms

    alto,ancho=frames[0].shape[:2]

    return {
        "config":nombre_config,
        "detector":det_nom,
        "tracker":trk_nom,
        "secuencia":nombre_seq,
        "resolucion":f"{ancho}x{alto}",
        "frames":len(frames),
        "deteccion_ms":round(det_ms,2),
        "tracking_ms":round(trk_ms,2),
        "total_ms":round(total_ms,2),
        "fps":round(1000/total_ms,2),
        "vram_mb":round(vram,1),
        "pct_tracker":round(100*trk_ms/total_ms,1),
    }


def leer_hota(nombre_config):
    """Recupera el HOTA del paso 7 para la tabla combinada."""
    detector,tracker=CONFIGS[nombre_config]
    nombre=f"{nombre_config}_{detector}_{tracker}"
    ruta=TRACKEVAL_DIR/"val"/nombre/"pedestrian_summary.txt"
    if not ruta.exists():
        return None
    lineas=ruta.read_text().strip().split("\n")
    if len(lineas)<2:
        return None
    cabeceras=lineas[0].split()
    valores=lineas[1].split()
    if "HOTA" not in cabeceras:
        return None
    return float(valores[cabeceras.index("HOTA")])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",nargs="+",default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--max-frames",type=int,default=MAX_FRAMES)
    args=ap.parse_args()

    print(f"configuraciones: {' '.join(args.config)}")
    print(f"secuencias     : {', '.join(SECUENCIAS)}")
    print(f"frames/secuencia: {args.max_frames}  (warmup {WARMUP})")
    print("frames precargados en RAM: no se cronometra lectura de disco\n")

    resultados=[]

    for nombre_seq in SECUENCIAS:
        print(f"precargando {nombre_seq}...")
        frames=precargar(nombre_seq,args.max_frames)
        alto,ancho=frames[0].shape[:2]
        print(f"  {len(frames)} frames  {ancho}x{alto}\n")

        for nombre_config in args.config:
            r=medir_secuencia(nombre_config,nombre_seq,frames)
            resultados.append(r)
            print(f"  {nombre_config} ({r['detector']}+{r['tracker']}): "
                  f"det={r['deteccion_ms']:6.2f}ms  "
                  f"trk={r['tracking_ms']:7.2f}ms  "
                  f"total={r['total_ms']:7.2f}ms  "
                  f"{r['fps']:6.2f} FPS  "
                  f"VRAM={r['vram_mb']:.0f}MB  "
                  f"tracker={r['pct_tracker']}%")
        print()

        # liberar la RAM antes de la siguiente secuencia
        del frames

    OUT_DIR.mkdir(parents=True,exist_ok=True)
    ruta_json=OUT_DIR/"fps_tracking_val.json"
    with open(ruta_json,"w") as f:
        json.dump({
            "nota":"detector+tracker en un solo bucle, frames precargados "
                   "en RAM (sin lectura de disco). Solo mide tiempo; la "
                   "calidad esta en results/tables/tracking/",
            "warmup":WARMUP,
            "resultados":resultados,
        },f,indent=2)

    # tabla combinada calidad+velocidad, con las condiciones declaradas
    ruta_csv=OUT_DIR/"resumen_calidad_velocidad.csv"
    with open(ruta_csv,"w") as f:
        f.write("config,detector,tracker,HOTA_val,secuencia,resolucion,"
                "deteccion_ms,tracking_ms,total_ms,fps,vram_mb\n")
        for r in resultados:
            hota=leer_hota(r["config"])
            hota_txt="" if hota is None else f"{hota:.2f}"
            f.write(f"{r['config']},{r['detector']},{r['tracker']},{hota_txt},"
                    f"{r['secuencia']},{r['resolucion']},{r['deteccion_ms']},"
                    f"{r['tracking_ms']},{r['total_ms']},{r['fps']},"
                    f"{r['vram_mb']}\n")

    print(f"JSON: {ruta_json}")
    print(f"CSV : {ruta_csv}")


if __name__=="__main__":
    main()