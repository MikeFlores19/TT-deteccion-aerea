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

Igual que aquel, descarta las predicciones que caen dentro de una zona
ignorada antes de evaluar (criterio del toolkit oficial, ver
zonas_ignoradas.py). El ground truth no se filtra.

UAVDT (Fase 4, --dataset uavdt): tres categorias (car, truck, bus). UAVDT
no distingue vans, asi que las predicciones van (5) cuentan como car (4),
igual que en la evaluacion de deteccion cruzada.

Uso:
  python -m src.evaluation.prepare_trackeval_clases --split test-dev
  python -m src.evaluation.prepare_trackeval_clases --dataset uavdt --split all
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
OUT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"trackeval_clases"

#UAVDT (Fase 4): mismas carpetas pero bajo data/uavdt_mot y runs/tracking_uavdt
UAVDT={
    "mot":PROJECT_ROOT/"data"/"uavdt_mot"/"motchallenge",
    "tracking":PROJECT_ROOT/"runs"/"tracking_uavdt",
    "out":PROJECT_ROOT/"data"/"uavdt_mot"/"trackeval_clases",
}
CLASES_UAVDT={4:"car", 6:"truck", 9:"bus"}
#ids extra que cuentan como la clase en las PREDICCIONES: van (5) -> car (4)
ALIAS_UAVDT={4:{5}}

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


def filtrar_y_forzar(ruta_origen,ruta_destino,id_clase,zonas=None,alias=()):
    """Copia solo las lineas de una clase, reetiquetandola a 1.

    Con zonas!=None descarta ademas las cajas que caen dentro de una zona
    ignorada. Solo se pasa para las predicciones.
    alias: ids adicionales que tambien cuentan como id_clase (solo predicciones).

    Devuelve (lineas_escritas,lineas_descartadas_por_zona).
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
            if int(campos[COL_CLASE])!=id_clase and int(campos[COL_CLASE]) not in alias:
                continue
            if zonas is not None:
                frame=int(campos[0])
                if en_zona_ignorada(caja_de_linea(campos),zonas.get(frame,())):
                    descartadas+=1
                    continue
            campos[COL_CLASE]=str(CLASE_UNICA)
            f_out.write(",".join(campos)+"\n")
            n+=1
    return n,descartadas


def preparar_clase(nombre_split,id_clase,nombre_clase,configs,alias=()):
    """Genera gt y trackers para una sola clase."""
    dir_origen=MOT_DIR/nombre_split
    secuencias=sorted(p for p in dir_origen.iterdir() if p.is_dir())

    total_gt=0
    zonas_por_seq={}
    for dir_seq in secuencias:
        destino=OUT_DIR/nombre_clase/"gt"/nombre_split/dir_seq.name
        #el gt NO se filtra por zonas ignoradas (asi lo hace el toolkit)
        n,_=filtrar_y_forzar(
            dir_seq/"gt"/"gt.txt",destino/"gt"/"gt.txt",id_clase
        )
        total_gt+=n
        destino.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dir_seq/"seqinfo.ini",destino/"seqinfo.ini")
        #se leen una vez y se reusan para las 4 configuraciones
        zonas_por_seq[dir_seq.name]=leer_zonas(dir_seq/"gt"/"ignore.txt")

    resumen=[]
    for nombre_config in configs:
        dir_split=TRACKING_DIR/nombre_config/nombre_split
        if not dir_split.exists():
            continue
        total_trk=0
        total_desc=0
        for dir_seq in secuencias:
            origen=dir_split/f"{dir_seq.name}.txt"
            if not origen.exists():
                print(f"  [ERROR] falta {origen}")
                continue
            n,desc=filtrar_y_forzar(
                origen,
                OUT_DIR/nombre_clase/"trackers"/nombre_split/
                nombre_config/"data"/f"{dir_seq.name}.txt",
                id_clase,
                zonas=zonas_por_seq[dir_seq.name],
                alias=alias,
            )
            total_trk+=n
            total_desc+=desc
        resumen.append((nombre_config,total_trk,total_desc))

    return total_gt,resumen


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="visdrone",choices=["visdrone","uavdt"])
    ap.add_argument("--split",default="test-dev",choices=["val","test-dev","all"])
    ap.add_argument("--configs",nargs="+",default=None,
                    help="nombres de carpeta en runs/tracking; por defecto todas")
    args=ap.parse_args()

    global MOT_DIR,TRACKING_DIR,OUT_DIR
    clases,alias_por_clase=CLASES,{}
    if args.dataset=="uavdt":
        MOT_DIR,TRACKING_DIR,OUT_DIR=UAVDT["mot"],UAVDT["tracking"],UAVDT["out"]
        clases,alias_por_clase=CLASES_UAVDT,ALIAS_UAVDT
        if args.split!="all":
            ap.error("uavdt solo admite --split all")

    if args.configs:
        configs=args.configs
    else:
        configs=[p.name for p in sorted(TRACKING_DIR.iterdir())
                 if (p/args.split).exists()]

    print(f"dataset: {args.dataset}")
    print(f"split  : {args.split}")
    print(f"configs: {' '.join(configs)}")
    print(f"destino: {OUT_DIR}\n")

    for id_clase,nombre_clase in clases.items():
        total_gt,resumen=preparar_clase(
            args.split,id_clase,nombre_clase,configs,
            alias=alias_por_clase.get(id_clase,()),
        )
        detalle="  ".join(f"{c}={n}(-{d})" for c,n,d in resumen)
        print(f"{nombre_clase:12} (cat {id_clase:2}): gt={total_gt:7}  {detalle}")

    print("\nListo.")


if __name__=="__main__":
    main()
