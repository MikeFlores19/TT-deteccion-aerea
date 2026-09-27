"""
Fase 4 · Paso 8 — Evaluacion class-agnostic de vehiculos en UAVDT.

Une las tres clases ya preparadas por prepare_trackeval_clases.py
(car, truck, bus: zonas ignoradas y van->car ya aplicados) en una sola
clase "vehiculo". Sirve para separar la confusion de clase del detector
de los errores de deteccion y de asociacion.

Uso:
  python -m src.evaluation.prepare_trackeval_clases --dataset uavdt --split all
  python -m src.evaluation.merge_vehiculo_uavdt
  python -m src.evaluation.eval_tracking --dataset uavdt --split all --clase vehiculo
"""
import shutil
from pathlib import Path

PROJECT_ROOT=Path(__file__).parent.parent.parent
BASE=PROJECT_ROOT/"data"/"uavdt_mot"/"trackeval_clases"
CLASES=["car","truck","bus"]


def main():
    out=BASE/"vehiculo"
    shutil.rmtree(out,ignore_errors=True)
    n_gt=n_trk=0
    for seq in sorted((BASE/"car"/"gt"/"all").iterdir()):
        d=out/"gt"/"all"/seq.name
        (d/"gt").mkdir(parents=True)
        shutil.copy2(seq/"seqinfo.ini",d/"seqinfo.ini")
        lineas=[l for c in CLASES for l in open(BASE/c/"gt"/"all"/seq.name/"gt"/"gt.txt")]
        open(d/"gt"/"gt.txt","w").writelines(lineas)
        n_gt+=len(lineas)
    for cfg in sorted((BASE/"car"/"trackers"/"all").iterdir()):
        for f in sorted((cfg/"data").glob("*.txt")):
            d=out/"trackers"/"all"/cfg.name/"data"
            d.mkdir(parents=True,exist_ok=True)
            lineas=[l for c in CLASES for l in open(BASE/c/"trackers"/"all"/cfg.name/"data"/f.name)]
            open(d/f.name,"w").writelines(lineas)
            n_trk+=len(lineas)
    print(f"vehiculo: gt={n_gt}  predicciones={n_trk}")


if __name__=="__main__":
    main()
