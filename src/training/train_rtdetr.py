"""
Entrenamiento RT-DETR-L sobre VisDrone-DET2019
"""

from ultralytics import YOLO
from pathlib import Path 
import json
from datetime import datetime

#Paths
PROJECT_ROOT=Path(__file__).resolve().parent.parent.parent
CONFIG_PATH=PROJECT_ROOT/"configs"/"visdrone.yaml"
RESULTS_DIR=PROJECT_ROOT/"runs"/"rtdetr"
RUN_NAME=f"rtdetr_visdrone_{datetime.now().strftime('%Y%m%d_%H%M')}"

#Hiperparametros
HYPERPARAMS={
    # Entrenamiento
    "epochs":100, #100 epocas
    "patience":20, #si despues de 20 epocas no mejora se corta
    "batch":4, #batch de 4, RT-DETR consume mas VRAM que YOLO por el transformer
    "imgsz":1280, #misma resolución que YOLOv8n para comparativa justa
    "optimizer":"AdamW",
    "lr0":0.0001, #lr mas bajo que YOLO, los transformers son sensibles a lr altos
    "lrf":0.01, #lr0xlrf para ir ajustando finamente en las ultimas epocas
    "momentum":0.937, #recordar la direccion en que venia el optimizador (carrito con inercia)
    "weight_decay":0.0001, #menor que YOLO, recomendado para arquitecturas transformer
    "warmup_epochs":3, #las primeras 3 epocas el lr sube lentamente desde casi cero hasta lr0
    "seed":0, #semilla fija para reproducibilidad (marco de experimentacion justo)

    # Augmentación 
    "mosaic":1.0,  #combina 4 imagenes, aumenta densidad y variacion de escala
    "close_mosaic":10, #apaga el mosaico en las ultimas 10 epocas para afinar con imagenes reales (sin distorsion)
    "mixup":0.15, #mezcla 2 imagenes y sus etiquetas, robustez ante variacion visual
    "hsv_h":0.015, #variacion de tono, distintas condiciones de luz
    "hsv_s":0.7, #variacion de saturacion
    "hsv_v":0.4, #variacion de brillo, simula dia/noche
    "fliplr":0.5, #flip horizontal, mas perspectivas
    "flipud":0.0, #flip vertical desactivado en perspectiva aerea
    "scale":0.5, #zoom aleatorio, simula cambios de altitud
    "translate":0.1, #desplazamiento, simula movimiento de camara

    # Hardware
    "device":0, #RTX 4050
    "workers":8, #procesos paralelos
    "amp":True, #FP16 critico para caber en 6GB
    "cache":False, #no carga todo de un jalon a la RAM
}


#Entrenamiento
def main():
    print()
    print(f"RT-DETR-L — VisDrone-DET2019")
    print(f"Run: {RUN_NAME}")
    print()

    model=YOLO("rtdetr-l.pt")

    results=model.train(
        data=str(CONFIG_PATH), #ruta al archivo .yaml que define el dataset
        project=str(RESULTS_DIR), #carpeta raiz donde se guardaran los resultados (ej: pesos)
        name=RUN_NAME, #nombre de la corrida especifica
        exist_ok=False, #si ya existe una carpeta con ese name lanza ERROR en lugar de sobreescribir
        verbose=True, #imprime los logs (progreso epoca por epoca)
        **HYPERPARAMS, #desempaca el diccionario de hiperparametros de arriba
    )

    #Guardar resumen de metricas
    run_dir=RESULTS_DIR/RUN_NAME

    """
    results.results_dict es un diccionario que Ultralytics llena
    automaticamente al terminar el entrenamiento con las metricas
    de la MEJOR epoca (la que tuvo mejor mAP50 en validacion)
    .get("clave", 0) significa: dame ese valor, y si no existe pon 0
    """

    #Metricas de precisión
    p=float(results.results_dict.get("metrics/precision(B)",0))
    r=float(results.results_dict.get("metrics/recall(B)",0))
    f1=2*p*r/(p+r) if (p+r)>0 else 0.0 #media armonica de precision y recall

    #Complejidad del modelo 
    #model.info() devuelve (capas, parametros, gradientes, GFLOPs)
    _, n_params, _, gflops=model.info()
    
    metrics={
        "run":RUN_NAME,
        "mAP50":float(results.results_dict.get("metrics/mAP50(B)",0)),
        "mAP50_95":float(results.results_dict.get("metrics/mAP50-95(B)",0)),
        "precision":p,
        "recall":r,
        "f1":f1, #nuevo: equilibrio precision-recall
        "params_M":n_params/1e6, #nuevo: millones de parametros
        "gflops":gflops, #operaciones por mil millones d epunto flotante por imagen (costo teorico de computo)
        "hyperparams":HYPERPARAMS, #guarda los hiperparametros usados
    }

    #crea un json y escribe el diccionario metrics con sangria de 2 espacios
    with open(run_dir/"metrics_summary.json","w") as f:
        json.dump(metrics,f,indent=2)

    print()
    print(f"Entrenamiento finalizado")
    print(f"mAP@0.5:{metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95:{metrics['mAP50_95']:.4f}")
    print(f"Precisión:{p:.4f}")
    print(f"Recall:{r:.4f}")
    print(f"F1-score:{f1:.4f}")
    print(f"Parámetros:{n_params/1e6:.2f}M")
    print(f"GFLOPs:{gflops:.1f}")
    print()

if __name__ == "__main__":
    main()

    