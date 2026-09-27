"""
Fase 4 · Paso 7 — Análisis del domain shift VisDrone → UAVDT.

No vuelve a correr los modelos: lee las estadísticas por imagen que guardó
eval_cross.py (data/cross_eval/stats/*.npz) y recalcula las métricas por grupo
con la misma función de Ultralytics (ap_per_class).

Grupos (UAVDT, atributos oficiales por secuencia + densidad por frame):
    clima     día / noche / niebla
    altitud   baja (<30 m) / media (30-70 m) / alta (>70 m)
    vista     frontal / lateral / cenital   (una secuencia puede tener dos vistas)
    densidad  terciles de objetos por frame (baja / media / alta)

Para cada grupo: frames, secuencias, instancias, mAP@0.5, mAP@0.5:0.95, P, R, F1,
AP@0.5 de car (clase estable, presente en casi todos los frames) y degradación
relativa del mAP@0.5 contra VisDrone val (3 clases).

Salida: results/tables/cross_eval/domain_shift.csv

Uso (desde la raíz del proyecto):
    python -m src.evaluation.analyze_domain_shift
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics.utils.metrics import ap_per_class

PROJECT_ROOT=Path(__file__).resolve().parents[2]
STATS_DIR=PROJECT_ROOT/"data"/"cross_eval"/"stats"
TABLES_DIR=PROJECT_ROOT/"results"/"tables"/"cross_eval"
INDEX=PROJECT_ROOT/"data"/"cross_eval"/"uavdt"/"index.csv"
MODELS=["yolov8n", "rtdetr"]

GROUPS={
    "clima":{"día":"daylight", "noche":"night", "niebla":"fog"},
    "altitud":{"baja":"low_alt", "media":"medium_alt", "alta":"high_alt"},
    "vista":{"frontal":"front_view", "lateral":"side_view", "cenital":"bird_view"},
}


def metrics_for(z, img_mask):
    """mAP@0.5, mAP@0.5:0.95, P, R, F1 y AP50 de car sobre un subconjunto de imágenes."""
    keep_p=img_mask[z["pred_img"]]
    keep_t=img_mask[z["target_img"]]
    tcls=z["target_cls"][keep_t]
    if len(tcls)==0:
        return None
    r=ap_per_class(z["tp"][keep_p], z["conf"][keep_p], z["pred_cls"][keep_p], tcls)
    #r: tp, fp, p, r, f1, ap, unique_classes, ...
    p, rc, ap, cls=r[2], r[3], r[5], r[6]
    mp, mr=float(p.mean()), float(rc.mean())
    car=ap[list(cls).index(0), 0] if 0 in cls else np.nan
    return {"instancias":len(tcls), "mAP50":float(ap[:, 0].mean()), "mAP50-95":float(ap.mean()),
            "P":mp, "R":mr, "F1":2*mp*mr/(mp+mr) if mp+mr else 0.0, "AP50_car":float(car),
            "clases":len(cls)}


def main():
    idx=pd.read_csv(INDEX)
    #Terciles de densidad (objetos por frame), fijados sobre UAVDT
    q1, q2=np.quantile(idx["n_obj"], [1/3, 2/3])
    idx["densidad"]=np.where(idx["n_obj"]<=q1, "baja", np.where(idx["n_obj"]<=q2, "media", "alta"))
    print(f"Densidad (obj/frame): baja ≤{q1:.0f} · media ≤{q2:.0f} · alta >{q2:.0f}")

    rows=[]
    for model in MODELS:
        base=json.loads((TABLES_DIR/f"{model}__visdrone_val.json").read_text())["mAP50"]
        ref=json.loads((TABLES_DIR/f"{model}__uavdt.json").read_text())["mAP50"]
        z=dict(np.load(STATS_DIR/f"{model}__uavdt.npz"))
        files=pd.Series(z["files"])
        info=idx.set_index("image").loc[files].reset_index()

        #Control: el total recalculado debe coincidir con el paso 6
        tot=metrics_for(z, np.ones(len(files), bool))
        ok="✅" if abs(tot["mAP50"]-ref)<=1e-3 else "❌"
        print(f"\n== {model} · control total: {tot['mAP50']:.4f} vs paso 6 {ref:.4f} {ok}")

        subsets=[("total", "UAVDT completo", np.ones(len(info), bool))]
        for gname, opts in GROUPS.items():
            for label, col in opts.items():
                subsets.append((gname, label, (info[col]==1).to_numpy()))
        for label in ["baja", "media", "alta"]:
            subsets.append(("densidad", label, (info["densidad"]==label).to_numpy()))

        for gname, label, mask in subsets:
            m=metrics_for(z, mask)
            if m is None:
                continue
            rows.append({"modelo":model, "grupo":gname, "condicion":label,
                         "frames":int(mask.sum()), "secuencias":info.loc[mask, "seq"].nunique(),
                         **m, "mAP50_visdrone":base,
                         "degradacion_rel_%":(m["mAP50"]-base)/base*100})

    df=pd.DataFrame(rows)
    out=TABLES_DIR/"domain_shift.csv"
    df.round(4).to_csv(out, index=False)

    #Vista compacta: ambos modelos lado a lado
    y=df[df["modelo"]=="yolov8n"].set_index(["grupo", "condicion"])
    r=df[df["modelo"]=="rtdetr"].set_index(["grupo", "condicion"])
    view=pd.DataFrame({
        "seq":y["secuencias"], "frames":y["frames"], "inst":y["instancias"],
        "YOLO_mAP50":y["mAP50"], "YOLO_degr%":y["degradacion_rel_%"],
        "RTDETR_mAP50":r["mAP50"], "RTDETR_degr%":r["degradacion_rel_%"],
    }).round(3)
    pd.set_option("display.width", 200)
    print(f"\n{view.to_string()}")
    print(f"\nCSV:{out.relative_to(PROJECT_ROOT)}")


if __name__=="__main__":
    main()