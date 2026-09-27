"""
Prepara los datos en la estructura exacta que espera TrackEval.

TrackEval (MotChallenge2DBox) tiene tres restricciones que obligan a
generar una copia adaptada en lugar de apuntarle a los datos originales:

  1) Solo admite la clase 'pedestrian' (id=1). Aborta si encuentra
     cualquier clase >1 en las predicciones. Se fuerza la clase a 1 en
     ambos lados: la evaluacion resultante es CLASS-AGNOSTIC.
  2) Espera trackers/{nombre}/data/{seq}.txt (subcarpeta 'data').
  3) MotChallenge2DBox NO soporta zonas ignoradas: cablea
     gt_crowd_ignore_regions a vacio (mot_challenge_2d_box.py:273), asi
     que el filtro no se puede activar con una bandera. Se aplica aqui,
     sobre las predicciones, con el criterio del toolkit oficial de
     VisDrone (ver zonas_ignoradas.py).

Los archivos originales NO se modifican: conservan la clase real y todas
sus predicciones. El filtro vive solo en esta copia adaptada, asi que
runs/tracking/ sigue siendo la salida cruda y trazable del tracker.

Uso:
  python -m src.evaluation.prepare_trackeval --split val
"""

import argparse
import shutil
from pathlib import Path

from src.evaluation.zonas_ignoradas import (
    caja_de_linea,
    en_zona_ignorada,
    leer_zonas,
)

PROJECT_ROOT=Path(__file__).parent.parent.parent
MOT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"motchallenge"
TRACKING_DIR=PROJECT_ROOT/"runs"/"tracking"
OUT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"trackeval"

# TrackEval interpreta la columna de clase con el vocabulario de MOT17
# (1=pedestrian, 8=distractor...), incompatible con el de VisDrone.
# Forzar todo a 1 evita esa colision de significados
CLASE_UNICA=1


def forzar_clase(ruta_origen,ruta_destino,col_clase,zonas=None):
    """Copia un .txt cambiando la columna de clase a 1.

    Con zonas!=None descarta ademas las cajas que caen dentro de una zona
    ignorada. Solo se pasa para las predicciones: el toolkit oficial no
    filtra el ground truth.

    Devuelve (lineas_escritas,lineas_descartadas).
    """
    ruta_destino.parent.mkdir(parents=True,exist_ok=True)
    n=0
    descartadas=0
    with open(ruta_origen) as f_in,open(ruta_destino,"w") as f_out:
        for linea in f_in:
            linea=linea.strip()
            if not linea:
                continue
            campos=linea.split(",")
            if zonas is not None:
                frame=int(campos[0])
                if en_zona_ignorada(caja_de_linea(campos),zonas.get(frame,())):
                    descartadas+=1
                    continue
            campos[col_clase]=str(CLASE_UNICA)
            f_out.write(",".join(campos)+"\n")
            n+=1
    return n,descartadas


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
        # el gt NO se filtra por zonas ignoradas (asi lo hace el toolkit)
        n,_=forzar_clase(dir_seq/"gt"/"gt.txt",destino_seq/"gt"/"gt.txt",7)
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
        total_desc=0
        for nombre_seq in secuencias:
            origen=dir_split/f"{nombre_seq}.txt"
            if not origen.exists():
                print(f"  [ERROR] falta {origen}")
                continue
            zonas=leer_zonas(
                MOT_DIR/nombre_split/nombre_seq/"gt"/"ignore.txt"
            )
            # results.txt: columna 7 es la clase
            # frame,id,left,top,w,h,conf,cls,-1,-1
            n,desc=forzar_clase(
                origen,
                dir_destino_base/dir_config.name/"data"/f"{nombre_seq}.txt",
                7,
                zonas=zonas,
            )
            total+=n
            total_desc+=desc

        nombres.append(dir_config.name)
        pct=100*total_desc/max(1,total+total_desc)
        print(f"  {dir_config.name}: {total} lineas  "
              f"(descartadas en zona ignorada: {total_desc}, {pct:.2f}%)")

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