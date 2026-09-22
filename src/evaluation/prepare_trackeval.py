"""
Prepara los datos en la estructura exacta que espera TrackEval.

TrackEval (MotChallenge2DBox) tiene tres restricciones que obligan a
generar una copia adaptada en lugar de apuntarle a los datos originales:

  1) Solo admite la clase 'pedestrian' (id=1). Aborta si encuentra
     cualquier clase >1 en las predicciones. Se fuerza la clase a 1 en
     ambos lados: la evaluacion resultante es CLASS-AGNOSTIC.
  2) Espera trackers/{nombre}/data/{seq}.txt (subcarpeta 'data').
  3) Con DO_PREPROC=False sigue filtrando el gt por conf=0, que es lo
     que necesitamos (descarta las regiones ignoradas de VisDrone).

Los archivos originales NO se modifican: conservan la clase real por si
mas adelante se quiere un desglose por clase.

Uso:
  python -m src.evaluation.prepare_trackeval --split val
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT=Path(__file__).parent.parent.parent
MOT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"motchallenge"
TRACKING_DIR=PROJECT_ROOT/"runs"/"tracking"
OUT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"trackeval"

# TrackEval interpreta la columna de clase con el vocabulario de MOT17
# (1=pedestrian, 8=distractor...), incompatible con el de VisDrone.
# Forzar todo a 1 evita esa colision de significados
CLASE_UNICA=1


def forzar_clase(ruta_origen,ruta_destino,col_clase):
    """Copia un .txt cambiando la columna de clase a 1."""
    ruta_destino.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with open(ruta_origen) as f_in,open(ruta_destino,"w") as f_out:
        for linea in f_in:
            linea=linea.strip()
            if not linea:
                continue
            campos=linea.split(",")
            campos[col_clase]=str(CLASE_UNICA)
            f_out.write(",".join(campos)+"\n")
            n+=1
    return n


def preparar_gt(nombre_split):
    """Copia gt.txt y seqinfo.ini con la clase forzada a 1."""
    dir_origen=MOT_DIR/nombre_split
    dir_destino=OUT_DIR/"gt"/nombre_split

    secuencias=sorted(p for p in dir_origen.iterdir() if p.is_dir())
    print(f"\n=== ground truth: {len(secuencias)} secuencias ===")

    for dir_seq in secuencias:
        destino_seq=dir_destino/dir_seq.name
        # gt.txt: columna 7 es la clase (0-indexado)
        # frame,id,left,top,w,h,conf,cat,visibility
        n=forzar_clase(dir_seq/"gt"/"gt.txt",destino_seq/"gt"/"gt.txt",7)
        # seqinfo.ini se copia tal cual: TrackEval lo necesita para
        # conocer seqLength y las dimensiones de la secuencia
        destino_seq.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dir_seq/"seqinfo.ini",destino_seq/"seqinfo.ini")
        print(f"  {dir_seq.name}: {n} lineas")

    return [p.name for p in secuencias]


def preparar_trackers(nombre_split,secuencias):
    """Copia los results.txt de cada configuracion con la clase a 1."""
    dir_destino_base=OUT_DIR/"trackers"/nombre_split

    configs=sorted(p for p in TRACKING_DIR.iterdir() if p.is_dir())
    print(f"\n=== trackers: {len(configs)} configuraciones ===")

    nombres=[]
    for dir_config in configs:
        dir_split=dir_config/nombre_split
        if not dir_split.exists():
            print(f"  [SALTADO] {dir_config.name} no tiene {nombre_split}")
            continue

        total=0
        for nombre_seq in secuencias:
            origen=dir_split/f"{nombre_seq}.txt"
            if not origen.exists():
                print(f"  [ERROR] falta {origen}")
                continue
            # results.txt: columna 7 es la clase
            # frame,id,left,top,w,h,conf,cls,-1,-1
            total+=forzar_clase(
                origen,
                dir_destino_base/dir_config.name/"data"/f"{nombre_seq}.txt",
                7,
            )

        nombres.append(dir_config.name)
        print(f"  {dir_config.name}: {total} lineas")

    return nombres


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",default="val",choices=["val","test-dev"])
    args=ap.parse_args()

    print(f"origen gt      : {MOT_DIR/args.split}")
    print(f"origen trackers: {TRACKING_DIR}")
    print(f"destino        : {OUT_DIR}")

    secuencias=preparar_gt(args.split)
    nombres=preparar_trackers(args.split,secuencias)

    print(f"\nListo. {len(secuencias)} secuencias, {len(nombres)} configuraciones.")
    print("Configuraciones:", " ".join(nombres))


if __name__=="__main__":
    main()