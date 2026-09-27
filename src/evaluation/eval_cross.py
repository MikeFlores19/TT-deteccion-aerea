"""
Fase 4 · Paso 6 — Evaluación de detección cruzada VisDrone → UAVDT.

Usa el MISMO validador de Ultralytics que produjo las métricas de Fase 2
(mismos pesos, imgsz=1280, conf=0.001, iou=0.7, max_det=300, FP32).
Solo se le añaden dos reglas mediante una subclase:

    1. Mapeo de clases (predicciones 10 → 3):
           car(3)+van(4) → car(0) · truck(5) → truck(1) · bus(8) → bus(2) · resto ✖
       Tras el mapeo se repite la NMS por clase (iou=0.7): una caja predicha
       como car y como van a la vez quedaría duplicada como car.

    2. Zonas ignoradas: se descarta toda predicción con >50% de su área
       dentro de una zona ignorada (regla de los toolkits de VisDrone/UAVDT).

Modos (--dataset):
    visdrone_val10  CALIBRACIÓN: 10 clases, sin mapeo ni filtro.
                    Debe reproducir Fase 2 (0.4755 / 0.5114).
    visdrone_val    VisDrone val, 3 clases, con zonas ignoradas → base
    uavdt           UAVDT (4,093 frames), 3 clases, con zonas ignoradas

Salidas:
    results/tables/cross_eval/{modelo}__{dataset}.json   métricas globales y por clase
    data/cross_eval/stats/{modelo}__{dataset}.npz        estadísticas por imagen (paso 7)

Uso (desde la raíz del proyecto):
    python -m src.evaluation.eval_cross --model yolov8n --dataset visdrone_val10
    python -m src.evaluation.eval_cross --model rtdetr  --dataset uavdt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from ultralytics import RTDETR, YOLO
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

PROJECT_ROOT=Path(__file__).resolve().parents[2]
CROSS_ROOT=PROJECT_ROOT/"data"/"cross_eval"

WEIGHTS={
    "yolov8n":PROJECT_ROOT/"runs/yolov8n/yolov8n_mosaic10_20260606_0315/weights/best.pt",
    "rtdetr":PROJECT_ROOT/"runs/rtdetr/rtdetr_visdrone_20260606_1656/weights/best.pt",
}
#Batch de la validación de Fase 2 (batch de entrenamiento 4 × 2). RT-DETR no usa rect,
#así que su batch no cambia el resultado; 4 para no saturar los 6 GB
DEFAULT_BATCH={"yolov8n":8, "rtdetr":4}

CLASS_NAMES_3={0:"car", 1:"truck", 2:"bus"}
VISDRONE_TO_3={3:0, 4:0, 5:1, 8:2}  #car, van→car, truck, bus
IGNORE_FRAC=0.5  #>50% del área dentro de una zona ignorada → se descarta


class CrossEvalMixin:
    """Añade mapeo de clases, zonas ignoradas y registro por imagen al validador."""
    cls_map=None       #None = sin mapeo (calibración)
    use_ignore=False
    instance=None      #último validador creado (para leer el registro)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CrossEvalMixin.instance=self
        self.records=[]
        self._pb=None

    def init_metrics(self, model):
        super().init_metrics(model)
        if self.cls_map is not None:
            #Las métricas se reportan con las 3 clases destino
            self.names=dict(CLASS_NAMES_3)
            self.nc=len(self.names)
            self.metrics.names=self.names
        #Registrar las estadísticas de cada imagen (para el desglose del paso 7)
        orig_update=self.metrics.update_stats
        def update_and_record(stat):
            self.records.append({"file":self._pb["im_file"],
                                 **{k:np.asarray(v).copy() for k, v in stat.items()}})
            orig_update(stat)
        self.metrics.update_stats=update_and_record

    def _prepare_batch(self, si, batch):
        #Se guarda el pbatch de la imagen actual; _prepare_pred lo usa a continuación
        self._pb=super()._prepare_batch(si, batch)
        return self._pb

    def _prepare_pred(self, pred):
        pred=super()._prepare_pred(pred)
        if self.cls_map is None or pred["cls"].shape[0]==0:
            return pred

        #1. Mapeo 10 → 3 clases
        lut=torch.full((256,), -1, dtype=torch.long, device=pred["cls"].device)
        for src, dst in self.cls_map.items():
            lut[src]=dst
        new_cls=lut[pred["cls"].long()]
        keep=new_cls>=0
        pred={k:v[keep] for k, v in pred.items()}
        pred["cls"]=new_cls[keep].to(pred["conf"].dtype)

        #NMS por clase tras el mapeo (car y van sobre el mismo objeto → un solo car)
        if pred["cls"].shape[0]:
            k=torchvision.ops.batched_nms(pred["bboxes"].float(), pred["conf"].float(),
                                          pred["cls"].long(), self.args.iou)
            pred={key:v[k] for key, v in pred.items()}

        #2. Zonas ignoradas
        if self.use_ignore and pred["cls"].shape[0]:
            zones=self._load_ignore(self._pb["im_file"])
            if len(zones):
                native=self._to_native(pred["bboxes"].clone().float(), self._pb).cpu().numpy()
                frac=ignored_fraction(native, zones, self._pb["ori_shape"])
                keep=torch.as_tensor(frac<=IGNORE_FRAC, device=pred["cls"].device)
                pred={k:v[keep] for k, v in pred.items()}
        return pred

    def _to_native(self, boxes, pb):
        """Cajas del espacio de evaluación → píxeles de la imagen original."""
        return ops.scale_boxes(pb["imgsz"], boxes, pb["ori_shape"], ratio_pad=pb["ratio_pad"])

    @staticmethod
    def _load_ignore(im_file):
        #.../images/X.jpg → .../ignore/X.txt (x,y,w,h en píxeles)
        p=Path(im_file)
        f=p.parent.parent/"ignore"/f"{p.stem}.txt"
        if not f.exists():
            return np.zeros((0, 4))
        rows=[list(map(float, l.split(","))) for l in f.read_text().splitlines() if l.strip()]
        return np.array(rows).reshape(-1, 4)


class CrossDetectionValidator(CrossEvalMixin, DetectionValidator):
    pass


class CrossRTDETRValidator(CrossEvalMixin, RTDETRValidator):
    def _to_native(self, boxes, pb):
        #RT-DETR estira la imagen a imgsz×imgsz (sin letterbox): escala directa
        h0, w0=pb["ori_shape"]
        h1, w1=pb["imgsz"]
        boxes[:, [0, 2]]*=w0/w1
        boxes[:, [1, 3]]*=h0/h1
        return boxes


def ignored_fraction(boxes, zones, ori_shape):
    """Fracción del área de cada caja (xyxy, px) que cae dentro de la unión de zonas."""
    h, w=int(ori_shape[0]), int(ori_shape[1])
    mask=np.zeros((h, w), dtype=np.int32)
    for x, y, zw, zh in zones:
        x1, y1=max(0, int(x)), max(0, int(y))
        x2, y2=min(w, int(np.ceil(x+zw))), min(h, int(np.ceil(y+zh)))
        mask[y1:y2, x1:x2]=1
    ii=np.pad(mask, ((1, 0), (1, 0))).cumsum(0).cumsum(1)  #imagen integral
    b=np.round(boxes).astype(int)
    x1, y1=np.clip(b[:, 0], 0, w), np.clip(b[:, 1], 0, h)
    x2, y2=np.clip(b[:, 2], 0, w), np.clip(b[:, 3], 0, h)
    inside=ii[y2, x2]-ii[y1, x2]-ii[y2, x1]+ii[y1, x1]
    area=np.maximum((x2-x1)*(y2-y1), 1)
    return inside/area


def dataset_yaml(name):
    """Crea el data.yaml de un conjunto de data/cross_eval/ (3 clases)."""
    root=CROSS_ROOT/name
    if not (root/"images").exists():
        raise FileNotFoundError(f"No existe {root}. Corre antes prepare_cross_eval.py --dataset {name}")
    cfg={"path":str(root), "train":"images", "val":"images", "names":CLASS_NAMES_3}  #train solo lo exige el formato
    f=root/"data.yaml"
    f.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(f)


def save_stats(records, path):
    """Estadísticas por imagen en un .npz plano (el paso 7 las agrupa por condición)."""
    files=[r["file"] for r in records]
    def cat(key, width=None):
        arrs=[r[key].reshape(-1, width) if width else r[key].reshape(-1) for r in records]
        return np.concatenate(arrs) if arrs else np.zeros(0)
    n_pred=[len(r["conf"]) for r in records]
    n_gt=[len(r["target_cls"]) for r in records]
    np.savez_compressed(
        path,
        files=np.array([Path(f).name for f in files]),
        tp=cat("tp", 10).astype(bool),
        conf=cat("conf").astype(np.float32),
        pred_cls=cat("pred_cls").astype(np.int16),
        pred_img=np.repeat(np.arange(len(files)), n_pred).astype(np.int32),
        target_cls=cat("target_cls").astype(np.int16),
        target_img=np.repeat(np.arange(len(files)), n_gt).astype(np.int32),
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(WEIGHTS))
    ap.add_argument("--dataset", required=True, choices=["visdrone_val10", "visdrone_val", "uavdt"])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default="0")
    args=ap.parse_args()

    #Configuración según el modo
    if args.dataset=="visdrone_val10":
        data=str(PROJECT_ROOT/"configs"/"visdrone.yaml")
        cls_map, use_ignore=None, False
    else:
        data=dataset_yaml(args.dataset)
        cls_map, use_ignore=VISDRONE_TO_3, True

    is_rtdetr=args.model=="rtdetr"
    Validator=CrossRTDETRValidator if is_rtdetr else CrossDetectionValidator
    Validator.cls_map=cls_map
    Validator.use_ignore=use_ignore

    model=(RTDETR if is_rtdetr else YOLO)(str(WEIGHTS[args.model]))
    batch=args.batch or DEFAULT_BATCH[args.model]
    tag=f"{args.model}__{args.dataset}"
    print(f"== {tag} · {data} · batch {batch}")

    t0=time.time()
    metrics=model.val(
        validator=Validator,
        data=data, split="val",
        imgsz=1280, batch=batch,
        conf=0.001, iou=0.7, max_det=300, half=False,
        device=args.device, plots=False, verbose=True,
        project=str(PROJECT_ROOT/"runs"/"cross_eval"), name=tag, exist_ok=True,
    )
    elapsed=time.time()-t0
    v=CrossEvalMixin.instance

    #Métricas globales (P y R en el umbral de F1 máximo, como Fase 2)
    mp, mr, map50, map5095=[float(x) for x in metrics.box.mean_results()]
    f1=2*mp*mr/(mp+mr) if mp+mr else 0.0
    per_class={}
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, a50, a=metrics.box.class_result(i)
        per_class[v.names[int(c)]]={"P":round(float(p), 4), "R":round(float(r), 4),
                                    "mAP50":round(float(a50), 4), "mAP50-95":round(float(a), 4),
                                    "instances":int(metrics.nt_per_class[int(c)])}

    result={
        "model":args.model, "weights":str(WEIGHTS[args.model].relative_to(PROJECT_ROOT)),
        "dataset":args.dataset, "images":v.seen, "instances":int(metrics.nt_per_class.sum()),
        "mAP50":round(map50, 4), "mAP50-95":round(map5095, 4),
        "P":round(mp, 4), "R":round(mr, 4), "F1":round(f1, 4),
        "per_class":per_class,
        "params":{"imgsz":1280, "conf":0.001, "iou":0.7, "max_det":300, "half":False, "batch":batch,
                  "class_map":cls_map, "ignore_zones":use_ignore, "ignore_frac":IGNORE_FRAC},
        "seconds":round(elapsed, 1),
    }
    out_json=PROJECT_ROOT/"results"/"tables"/"cross_eval"/f"{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    out_npz=CROSS_ROOT/"stats"/f"{tag}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_stats(v.records, out_npz)

    print(f"\n{'='*60}\n{tag}")
    print(f"Imágenes:{v.seen} · Instancias:{result['instances']:,} · Tiempo:{elapsed/60:.1f} min")
    print(f"mAP@0.5={map50:.4f} · mAP@0.5:0.95={map5095:.4f} · P={mp:.4f} · R={mr:.4f} · F1={f1:.4f}")
    if args.dataset=="visdrone_val10":
        ref={"yolov8n":0.4755, "rtdetr":0.5114}[args.model]
        diff=abs(map50-ref)
        print(f"Calibración: Fase 2={ref:.4f} · diferencia={diff:.4f} → {'✅ OK' if diff<=0.002 else '❌ REVISAR'}")
    print(f"JSON:{out_json.relative_to(PROJECT_ROOT)}")
    print(f"NPZ:{out_npz.relative_to(PROJECT_ROOT)}")


if __name__=="__main__":
    main()