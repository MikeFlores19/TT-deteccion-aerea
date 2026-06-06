"""
Metricas de eficiencia en inferencia: latencia, FPS, VRAM y RAM.
"""

from ultralytics import YOLO
from pathlib import Path
import json
import yaml
import torch
import psutil
import os

#Paths
PROJECT_ROOT=Path(__file__).resolve().parent.parent.parent
CONFIG_PATH=PROJECT_ROOT/"configs"/"visdrone.yaml"
OUTPUT_DIR=PROJECT_ROOT/"results"/"tables"/"eficiencia"

#Modelos a medir: nombre_display -> ruta al best.pt
#Para agregar RT-DETR cuando este listo, descomentar la linea correspondiente
MODELS={
    "yolov8n_mosaic10": PROJECT_ROOT/"runs"/"yolov8n"/"yolov8n_mosaic10_20260606_0315"/"weights"/"best.pt",
    #"rtdetr_mosaic10": PROJECT_ROOT/"runs"/"rtdetr"/"rtdetr_visdrone_20260606_1656"/"weights"/"best.pt",
}

#Parametros del profiling
IMGSZ=1280 #misma resolucion que en entrenamiento
WARMUP=10  #inferencias de calentamiento que NO se cronometran
MAX_IMAGES=None #None=1610 imagenes de test completo; entero para prueba rapida


def get_test_images(config_path):
    """
    Lee el split de test desde el yaml del dataset.
    Devuelve lista de rutas de imagenes.
    """
    with open(config_path,"r") as f:
        cfg=yaml.safe_load(f)
    base=Path(cfg["path"])
    if not base.is_absolute():
        base=(PROJECT_ROOT/base).resolve()
    test_dir=base/cfg["test"]
    images=sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"})
    return images


def medir_eficiencia(weights_path,images,device):
    """
    Mide latencia, FPS, VRAM y RAM sobre la lista de imagenes.
    Protocolo: batch=1, warm-up sin cronometrar, medicion imagen por imagen.
    """
    model=YOLO(str(weights_path))

    #Warm-up: primeras inferencias que NO se cuentan
    for img in images[:WARMUP]:
        model.predict(str(img),imgsz=IMGSZ,device=device,verbose=False)

    #Reinicia contador de pico de VRAM justo antes de medir
    if device!="cpu":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    #Medicion imagen por imagen (batch=1 = condicion real de vuelo)
    total_inf_ms=0.0
    n=0
    for img in images:
        r=model.predict(str(img),imgsz=IMGSZ,device=device,verbose=False)
        total_inf_ms+=r[0].speed["inference"] #ms de inferencia pura
        n+=1

    if device!="cpu":
        torch.cuda.synchronize()
        vram_mb=torch.cuda.max_memory_allocated()/1024**2 #pico de VRAM en MB
    else:
        vram_mb=0.0

    ram_mb=psutil.Process(os.getpid()).memory_info().rss/1024**2 #RAM del proceso

    avg_inf_ms=total_inf_ms/n if n>0 else 0.0
    fps=1000/avg_inf_ms if avg_inf_ms>0 else 0.0

    return {
        "weights":str(weights_path),
        "n_images":n,
        "imgsz":IMGSZ,
        "device":str(device),
        "latency_ms":round(avg_inf_ms,2),
        "fps":round(fps,2),
        "vram_mb":round(vram_mb,1),
        "ram_mb":round(ram_mb,1),
    }


def main():
    device=0 if torch.cuda.is_available() else "cpu"
    gpu_name=torch.cuda.get_device_name(0) if device!="cpu" else "CPU"

    print()
    print("Metricas de eficiencia — VisDrone test")
    print(f"Hardware: {gpu_name}")
    print(f"imgsz={IMGSZ} · batch=1 · warmup={WARMUP}")
    print()

    #Carga imagenes de test una sola vez
    images=get_test_images(CONFIG_PATH)
    if MAX_IMAGES is not None:
        images=images[:MAX_IMAGES]
    print(f"Imagenes de test: {len(images)}")
    print()

    results={}
    for model_name,weights_path in MODELS.items():
        if not weights_path.exists():
            print(f"[!] {model_name}: no se encontro {weights_path}, se omite")
            continue
        print(f"Midiendo {model_name}...")
        results[model_name]=medir_eficiencia(weights_path,images,device)
        m=results[model_name]
        print(f"  Latencia: {m['latency_ms']} ms/img")
        print(f"  FPS:      {m['fps']}")
        print(f"  VRAM:     {m['vram_mb']} MB")
        print(f"  RAM:      {m['ram_mb']} MB")
        print()

    #Guarda JSON con nombre que incluye la GPU
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    out_path=OUTPUT_DIR/f"eficiencia_{gpu_name.replace(' ','_')}.json"
    with open(out_path,"w") as f:
        json.dump(results,f,indent=2)

    print(f"Resultados guardados en: {out_path}")
    print()


if __name__=="__main__":
    main()
