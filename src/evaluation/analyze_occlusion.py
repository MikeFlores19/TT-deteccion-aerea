"""
Fase 4 · Paso 7b — Detección por nivel de oclusión y de fuera-de-vista (UAVDT).

La oclusión es un atributo de cada OBJETO, no del frame. Por eso aquí se
empareja cada predicción con su objeto del GT y se mide, por nivel:

    Recall   % de objetos de ese nivel detectados (conf ≥ 0.25, IoU ≥ 0.5)
    AP50     precisión promedio tratando los objetos de OTROS niveles como
             ignorados (una predicción sobre ellos no es acierto ni error)

El emparejamiento replica el de Ultralytics (match_predictions, IoU ≥ 0.5,
misma clase), así que el nivel "todos" debe reproducir el mAP@0.5 del paso 6.

Niveles oficiales de UAVDT (códigos NO ordinales, se remapean):
    oclusión        1=sin · 4=pequeña (1-30%) · 3=media (30-70%) · 2=grande (70-100%)
    fuera-de-vista  1=sin · 3=pequeña (1-30%) · 2=media (30-50%)

Entrada: data/cross_eval/stats/{modelo}__uavdt.npz (con pred_boxes) + GT oficial
Salida:  results/tables/cross_eval/occlusion.csv

Uso (desde la raíz del proyecto):
    python -m src.evaluation.analyze_occlusion
"""
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics.utils.metrics import ap_per_class

PROJECT_ROOT=Path(__file__).resolve().parents[2]
STATS_DIR=PROJECT_ROOT/"data"/"cross_eval"/"stats"
TABLES_DIR=PROJECT_ROOT/"results"/"tables"/"cross_eval"
INDEX=PROJECT_ROOT/"data"/"cross_eval"/"uavdt"/"index.csv"
GT_DIR=PROJECT_ROOT/"data"/"uavdt"/"UAV-benchmark-MOTD_v1.0"/"GT"
MODELS=["yolov8n", "rtdetr"]

UAVDT_MAP={1:0, 2:1, 3:2}  #car, truck, bus
IOU_THR=0.5
CONF_OP=0.25  #punto de operación para el recall (umbral por defecto de predicción)
LEVELS={
    "oclusion":{1:"sin", 4:"pequeña (1-30%)", 3:"media (30-70%)", 2:"grande (70-100%)"},
    "fuera_de_vista":{1:"sin", 3:"pequeña (1-30%)", 2:"media (30-50%)"},
}


def load_gt(idx):
    """GT por imagen en píxeles (mismo recorte que prepare_cross_eval): cls, xyxy, oclusión, fuera-de-vista."""
    gt={}
    for seq, g in idx.groupby("seq"):
        w, h=int(g["w"].iloc[0]), int(g["h"].iloc[0])
        raw=pd.read_csv(GT_DIR/f"{seq}_gt_whole.txt", header=None,
                        names=["frame", "id", "x", "y", "w", "h", "out", "occ", "cat"])
        raw=raw[raw["frame"].isin(g["frame"])]
        x1=raw["x"].clip(lower=0); y1=raw["y"].clip(lower=0)
        x2=(raw["x"]+raw["w"]).clip(upper=w); y2=(raw["y"]+raw["h"]).clip(upper=h)
        raw=raw.assign(x1=x1, y1=y1, x2=x2, y2=y2)
        raw=raw[(raw["x2"]>raw["x1"])&(raw["y2"]>raw["y1"])]
        for frame, r in raw.groupby("frame"):
            gt[f"{seq}_img{frame:06d}.jpg"]={
                "cls":r["cat"].map(UAVDT_MAP).to_numpy(),
                "box":r[["x1", "y1", "x2", "y2"]].to_numpy(float),
                "oclusion":r["occ"].to_numpy(), "fuera_de_vista":r["out"].to_numpy()}
    return gt


def box_iou(a, b):
    """IoU entre cajas xyxy: a (M,4) · b (N,4) → (M,N)."""
    lt=np.maximum(a[:, None, :2], b[None, :, :2])
    rb=np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter=np.clip(rb-lt, 0, None).prod(2)
    area_a=(a[:, 2:]-a[:, :2]).prod(1)
    area_b=(b[:, 2:]-b[:, :2]).prod(1)
    return inter/(area_a[:, None]+area_b[None, :]-inter+1e-9)


def match(gt_cls, gt_box, p_cls, p_box):
    """Emparejamiento de Ultralytics (match_predictions) a IoU 0.5 → índice de GT por predicción (-1 = sin pareja)."""
    out=np.full(len(p_cls), -1)
    if len(gt_cls)==0 or len(p_cls)==0:
        return out
    iou=box_iou(gt_box, p_box)*(gt_cls[:, None]==p_cls[None, :])
    m=np.array(np.nonzero(iou>=IOU_THR)).T
    if len(m):
        if len(m)>1:
            m=m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
            m=m[np.unique(m[:, 1], return_index=True)[1]]
            m=m[np.unique(m[:, 0], return_index=True)[1]]
        out[m[:, 1]]=m[:, 0]
    return out


def main():
    idx=pd.read_csv(INDEX)
    gt=load_gt(idx)
    rows=[]
    for model in MODELS:
        z=dict(np.load(STATS_DIR/f"{model}__uavdt.npz"))
        if "pred_boxes" not in z:
            raise KeyError(f"{model}__uavdt.npz no tiene pred_boxes: vuelve a correr eval_cross.py --dataset uavdt")
        files=z["files"]

        #Emparejar todas las predicciones y las de conf ≥ CONF_OP
        recs=[]
        n_bad=0
        for i, f in enumerate(files):
            g=gt.get(f, {"cls":np.zeros(0, int), "box":np.zeros((0, 4)),
                         "oclusion":np.zeros(0, int), "fuera_de_vista":np.zeros(0, int)})
            n_bad+=int(len(g["cls"])!=(z["target_img"]==i).sum())
            sel=z["pred_img"]==i
            pc, pb, cf=z["pred_cls"][sel].astype(int), z["pred_boxes"][sel], z["conf"][sel]
            recs.append({"g":g, "pc":pc, "cf":cf,
                         "m_all":match(g["cls"], g["box"], pc, pb),
                         "m_op":match(g["cls"], g["box"], pc[cf>=CONF_OP], pb[cf>=CONF_OP])})
        print(f"\n== {model} · imágenes con distinto número de objetos que el paso 6: {n_bad} (debe ser 0)")

        def evaluate(attr, level):
            tp, conf, pcls, tcls=[], [], [], []
            found, total=0, 0
            for r in recs:
                g=r["g"]
                is_lvl=np.ones(len(g["cls"]), bool) if attr is None else g[attr]==level
                #AP: acierto si su pareja es del nivel; se ignora si su pareja es de otro nivel
                m=r["m_all"]
                matched=m>=0
                if len(g["cls"]):
                    keep=~matched|is_lvl[np.clip(m, 0, None)]
                else:
                    keep=np.ones(len(m), bool)  #frame sin objetos: toda predicción es FP
                tp.append((matched&keep)[keep]); conf.append(r["cf"][keep]); pcls.append(r["pc"][keep])
                tcls.append(g["cls"][is_lvl])
                #Recall en el punto de operación
                hit=np.zeros(len(g["cls"]), bool)
                mo=r["m_op"]
                hit[mo[mo>=0]]=True
                found+=int((hit&is_lvl).sum()); total+=int(is_lvl.sum())
            tcls=np.concatenate(tcls)
            if len(tcls)==0:
                return None
            res=ap_per_class(np.concatenate(tp)[:, None], np.concatenate(conf),
                             np.concatenate(pcls), tcls)
            ap, cls=res[5][:, 0], res[6]
            car=float(ap[list(cls).index(0)]) if 0 in cls else np.nan
            return {"objetos":total, f"recall@{CONF_OP}":found/total,
                    "AP50":float(ap.mean()), "AP50_car":car}

        tot=evaluate(None, None)
        print(f"   control 'todos': AP50={tot['AP50']:.4f} (debe ≈ mAP@0.5 del paso 6)")
        rows.append({"modelo":model, "atributo":"todos", "nivel":"todos", **tot})
        for attr, lv in LEVELS.items():
            for code, name in lv.items():
                m=evaluate(attr, code)
                if m:
                    rows.append({"modelo":model, "atributo":attr, "nivel":name, **m})

    df=pd.DataFrame(rows)
    out=TABLES_DIR/"occlusion.csv"
    df.round(4).to_csv(out, index=False)

    #Vista compacta: ambos modelos lado a lado
    rc=f"recall@{CONF_OP}"
    y=df[df["modelo"]=="yolov8n"].set_index(["atributo", "nivel"])
    r=df[df["modelo"]=="rtdetr"].set_index(["atributo", "nivel"])
    view=pd.DataFrame({"objetos":y["objetos"],
                       "YOLO_recall":y[rc], "YOLO_AP50car":y["AP50_car"],
                       "RTDETR_recall":r[rc], "RTDETR_AP50car":r["AP50_car"]}).round(3)
    pd.set_option("display.width", 200)
    print(f"\n{view.to_string()}")
    print(f"\nCSV:{out.relative_to(PROJECT_ROOT)}")


if __name__=="__main__":
    main()