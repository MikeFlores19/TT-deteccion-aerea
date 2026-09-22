"""
Prepara los datos para evaluar con el protocolo oficial de VisDrone:
cinco categorias (pedestrian, car, van, truck, bus) evaluadas por
separado y luego promediadas.

Es lo que usa la literatura de VisDrone-MOT, asi que los resultados
son comparables. El toolkit oficial ignora las otras cinco clases:
'people' es ambigua con 'pedestrian' y el resto tienen pocas muestras.

Diferencia con prepare_trackeval.py: aquel fuerza TODAS las clases a 1
(evaluacion class-agnostic). Este filtra primero por clase y genera un
juego de datos independiente por cada una.

Uso:
  python -m src.evaluation.prepare_trackeval_clases --split test-dev
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT=Path(__file__).parent.parent.parent
MOT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"motchallenge"
TRACKING_DIR=PROJECT_ROOT/"runs"/"tracking"
OUT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"trackeval_clases"

# las 5 categorias del toolkit oficial, en indexacion VisDrone-MOT
CLASES={
    1:"pedestrian",
    4:"car",
    5:"van",
    6:"truck",
    9:"bus",
}

# columna de la clase en ambos formatos (0-indexado)
#   gt.txt     : frame,id,left,top,w,h,conf,cat,visibility
#   results.txt: frame,id,left,top,w,h,conf,cls,-1,-1
COL_CLASE=7

# TrackEval solo admite la clase 'pedestrian' (id=1), asi que tras
# filtrar hay que reetiquetar a 1
CLASE_UNICA=1


def filtrar_y_forzar(ruta_origen,ruta_destino,id_clase):
    """Copia solo las lineas de una clase, reetiquetandola a 1."""
    ruta_destino.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with open(ruta_origen) as f_in,open(ruta_destino,"w") as f_out:
        for linea in f_in:
            linea=linea.strip()
            if not linea:
                continue
            campos=linea.split(",")
            if int(campos[COL_CLASE])!=id_clase:
                continue
            campos[COL_CLASE]=str(CLASE_UNICA)
            f_out.write(",".join(campos)+"\n")
            n+=1
    return n


def preparar_clase(nombre_split,id_clase,nombre_clase,configs):
    """Genera gt y trackers para una sola clase."""
    dir_origen=MOT_DIR/nombre_split
    secuencias=sorted(p for p in dir_origen.iterdir() if p.is_dir())

    total_gt=0
    for dir_seq in secuencias:
        destino=OUT_DIR/nombre_clase/"gt"/nombre_split/dir_seq.name
        total_gt+=filtrar_y_forzar(
            dir_seq/"gt"/"gt.txt",destino/"gt"/"gt.txt",id_clase
        )
        destino.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dir_seq/"seqinfo.ini",destino/"seqinfo.ini")

    resumen=[]
    for nombre_config in configs:
        dir_split=TRACKING_DIR/nombre_config/nombre_split
        if not dir_split.exists():
            continue
        total_trk=0
        for dir_seq in secuencias:
            origen=dir_split/f"{dir_seq.name}.txt"
            if not origen.exists():
                print(f"  [ERROR] falta {origen}")
                continue
            total_trk+=filtrar_y_forzar(
                origen,
                OUT_DIR/nombre_clase/"trackers"/nombre_split/
                nombre_config/"data"/f"{dir_seq.name}.txt",
                id_clase,
            )
        resumen.append((nombre_config,total_trk))

    return total_gt,resumen


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",default="test-dev",choices=["val","test-dev"])
    ap.add_argument("--configs",nargs="+",default=None,
                    help="nombres de carpeta en runs/tracking; por defecto todas")
    args=ap.parse_args()

    if args.configs:
        configs=args.configs
    else:
        configs=[p.name for p in sorted(TRACKING_DIR.iterdir())
                 if (p/args.split).exists()]

    print(f"split  : {args.split}")
    print(f"configs: {' '.join(configs)}")
    print(f"destino: {OUT_DIR}\n")

    for id_clase,nombre_clase in CLASES.items():
        total_gt,resumen=preparar_clase(
            args.split,id_clase,nombre_clase,configs
        )
        detalle="  ".join(f"{c}={n}" for c,n in resumen)
        print(f"{nombre_clase:12} (cat {id_clase:2}): gt={total_gt:7}  {detalle}")

    print("\nListo.")


if __name__=="__main__":
    main()