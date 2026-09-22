"""
Evalua las configuraciones del factorial con TrackEval.

Dos protocolos:
  class-agnostic  todas las clases forzadas a 1, un solo numero global.
                  Requiere prepare_trackeval.py
  por clase       las 5 categorias del toolkit oficial de VisDrone
                  (pedestrian, car, van, truck, bus) evaluadas por
                  separado. Es lo que reporta la literatura, asi que
                  los resultados son comparables.
                  Requiere prepare_trackeval_clases.py

Configuracion elegida y por que:
  SKIP_SPLIT_FOL=True   evita la carpeta intermedia 'MOT17-train'
  DO_PREPROC=False      el preproc de MOT17 filtra 'distractores' cuyos
                        ids de clase no significan lo mismo en VisDrone;
                        ademas usa np.int, eliminado en numpy>=1.24.
                        Con False el gt SIGUE filtrando conf=0, que es
                        lo que necesitamos para las regiones ignoradas
  CLASSES_TO_EVAL=['pedestrian']  unica clase admitida por TrackEval;
                        en ambos protocolos las clases se reetiquetan
                        a 1 antes de evaluar

Uso:
  python -m src.evaluation.eval_tracking --split val
  python -m src.evaluation.eval_tracking --split test-dev --clase car
  python -m src.evaluation.eval_tracking --split test-dev --clase todas
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).parent.parent.parent
TRACKEVAL_DIR=PROJECT_ROOT/"external"/"TrackEval"
DATA_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"trackeval"
CLASES_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"trackeval_clases"
OUT_DIR=PROJECT_ROOT/"results"/"tables"/"tracking"

# las 5 categorias del toolkit oficial de VisDrone
CLASES=["pedestrian","car","van","truck","bus"]

# TrackEval no es un paquete instalable: se importa por ruta
sys.path.insert(0,str(TRACKEVAL_DIR))
import trackeval  # noqa: E402


def evaluar(nombre_split,n_cores,nombre_clase=None):
    if nombre_clase:
        base=CLASES_DIR/nombre_clase
        dir_gt=base/"gt"/nombre_split
        dir_trackers=base/"trackers"/nombre_split
        script_previo="prepare_trackeval_clases.py"
    else:
        dir_gt=DATA_DIR/"gt"/nombre_split
        dir_trackers=DATA_DIR/"trackers"/nombre_split
        script_previo="prepare_trackeval.py"

    if not dir_gt.exists():
        raise FileNotFoundError(
            f"no existe {dir_gt}. Corre antes {script_previo}"
        )

    # cada protocolo en su propia carpeta para no pisar resultados
    dir_salida=OUT_DIR/nombre_split/(nombre_clase or "class-agnostic")
    dir_salida.mkdir(parents=True,exist_ok=True)

    cfg_eval=trackeval.Evaluator.get_default_eval_config()
    cfg_eval["USE_PARALLEL"]=n_cores>1
    cfg_eval["NUM_PARALLEL_CORES"]=n_cores
    cfg_eval["PRINT_CONFIG"]=False
    cfg_eval["OUTPUT_SUMMARY"]=True
    cfg_eval["OUTPUT_DETAILED"]=True
    cfg_eval["PLOT_CURVES"]=False

    cfg_data=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    cfg_data["GT_FOLDER"]=str(dir_gt)
    cfg_data["TRACKERS_FOLDER"]=str(dir_trackers)
    cfg_data["OUTPUT_FOLDER"]=str(dir_salida)
    cfg_data["SKIP_SPLIT_FOL"]=True
    cfg_data["DO_PREPROC"]=False
    cfg_data["PRINT_CONFIG"]=False
    cfg_data["CLASSES_TO_EVAL"]=["pedestrian"]
    cfg_data["TRACKER_SUB_FOLDER"]="data"
    # sin SEQ_INFO TrackEval buscaria un seqmap; se le pasan las
    # secuencias directamente leyendolas del directorio de gt
    cfg_data["SEQ_INFO"]={
        p.name:None for p in sorted(dir_gt.iterdir()) if p.is_dir()
    }

    cfg_metricas={"THRESHOLD":0.5,"PRINT_CONFIG":False}

    print(f"protocolo : {nombre_clase or 'class-agnostic'}")
    print(f"gt        : {dir_gt}")
    print(f"trackers  : {dir_trackers}")
    print(f"salida    : {dir_salida}")
    print(f"secuencias: {len(cfg_data['SEQ_INFO'])}\n")

    evaluador=trackeval.Evaluator(cfg_eval)
    datasets=[trackeval.datasets.MotChallenge2DBox(cfg_data)]
    metricas=[
        trackeval.metrics.HOTA(cfg_metricas),
        trackeval.metrics.CLEAR(cfg_metricas),
        trackeval.metrics.Identity(cfg_metricas),
    ]

    evaluador.evaluate(datasets,metricas)
    print(f"\nResultados en: {dir_salida}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",default="val",choices=["val","test-dev"])
    ap.add_argument("--cores",type=int,default=1)
    ap.add_argument("--clase",default=None,
                    help="una de las 5 oficiales, o 'todas'; "
                         "omitir para class-agnostic")
    args=ap.parse_args()

    if args.clase=="todas":
        for c in CLASES:
            print(f"\n{'='*60}\n{c.upper()}\n{'='*60}")
            evaluar(args.split,args.cores,c)
    else:
        evaluar(args.split,args.cores,args.clase)


if __name__=="__main__":
    main()