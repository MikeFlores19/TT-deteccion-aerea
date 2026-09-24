"""
Fase 4 · Paso 5 — GT de 3 clases para la evaluación cruzada VisDrone → UAVDT.

Genera, para cada dataset, un conjunto listo para evaluar con las MISMAS 3 clases:
    0=car  1=truck  2=bus

UAVDT (data/uavdt/, oficial Benchmark-M)
    - 50 secuencias (train + test): el modelo nunca vio UAVDT
    - 1 de cada 10 frames: 1, 11, 21, ... (frames consecutivos son casi idénticos)
    - clases 1=car 2=truck 3=bus → 0, 1, 2
    - normaliza con la resolución de CADA secuencia (M0901 = 960x540)

VisDrone val (data/visdrone/, base de comparación)
    - labels/val en YOLO (10 clases) → {car:0, van:0, truck:1, bus:2}; el resto se descarta
    - zonas ignoradas desde labels_original_backup/val (clase 0 en la 6a columna)

Salida: data/cross_eval/{uavdt,visdrone_val}/
    images/    symlinks a las imágenes originales (no copia nada)
    labels/    YOLO 3 clases
    ignore/    zonas ignoradas en píxeles x,y,w,h (las usa el evaluador del paso 6)
    index.csv  una fila por imagen (+ atributos de condición en UAVDT)

Uso (desde la raíz del proyecto):
    python -m src.preprocessing.prepare_cross_eval --dataset uavdt
    python -m src.preprocessing.prepare_cross_eval --dataset visdrone_val
"""
import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT=Path(__file__).resolve().parents[2]
OUT_ROOT=PROJECT_ROOT/"data"/"cross_eval"

CLASS_NAMES={0:"car", 1:"truck", 2:"bus"}
STEP=10  #1 de cada 10 frames

#UAVDT oficial: 1=car 2=truck 3=bus
UAVDT_MAP={1:0, 2:1, 3:2}
#VisDrone YOLO: 3=car 4=van 5=truck 8=bus
VISDRONE_MAP={3:0, 4:0, 5:1, 8:2}

ATTR_NAMES=["daylight", "night", "fog",
            "low_alt", "medium_alt", "high_alt",
            "front_view", "side_view", "bird_view",
            "long_term"]


def reset_out(name):
    """Borra y recrea la carpeta de salida para no mezclar corridas."""
    out=OUT_ROOT/name
    if out.exists():
        shutil.rmtree(out)
    for sub in ["images", "labels", "ignore"]:
        (out/sub).mkdir(parents=True)
    return out


def to_yolo(x, y, w, h, img_w, img_h):
    """Caja en píxeles → YOLO normalizado. Recorta al borde de la imagen."""
    x1, y1=max(0, x), max(0, y)
    x2, y2=min(img_w, x+w), min(img_h, y+h)
    if x2<=x1 or y2<=y1:
        return None
    return ((x1+x2)/2/img_w, (y1+y2)/2/img_h, (x2-x1)/img_w, (y2-y1)/img_h)


def read_csv_by_frame(path):
    """Lee un GT de UAVDT y lo agrupa por frame: {frame: [[campos], ...]}."""
    by_frame={}
    if not path.exists():
        return by_frame
    for line in path.read_text().splitlines():
        parts=[int(float(p)) for p in line.strip().split(",") if p!=""]
        if parts:
            by_frame.setdefault(parts[0], []).append(parts)
    return by_frame


def write_lines(path, lines):
    path.write_text("\n".join(lines)+("\n" if lines else ""))


#-------------------------------------------------------------- UAVDT
def prepare_uavdt():
    root=PROJECT_ROOT/"data"/"uavdt"
    frames_dir=root/"UAV-benchmark-M"
    gt_dir=root/"UAV-benchmark-MOTD_v1.0"/"GT"
    attr_dir=root/"M_attr"
    out=reset_out("uavdt")

    #Split y atributos (strip por el espacio en 'M0701 _attr.txt')
    split_map, attr_map={}, {}
    for split in ["train", "test"]:
        for f in (attr_dir/split).glob("*_attr.txt"):
            seq=f.name.split("_")[0].strip()
            split_map[seq]=split
            attr_map[seq]=[int(v) for v in f.read_text().strip().split(",")]

    rows=[]
    cls_count=Counter()
    resolutions=Counter()
    n_clipped=0

    for seq_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
        seq=seq_dir.name
        jpgs=sorted(seq_dir.glob("img*.jpg"))
        img_w, img_h=Image.open(jpgs[0]).size
        gt=read_csv_by_frame(gt_dir/f"{seq}_gt_whole.txt")
        ign=read_csv_by_frame(gt_dir/f"{seq}_gt_ignore.txt")

        for jpg in jpgs:
            frame=int(jpg.stem[3:])  #img000011 → 11
            if (frame-1)%STEP!=0:
                continue
            name=f"{seq}_{jpg.stem}"
            (out/"images"/f"{name}.jpg").symlink_to(jpg.resolve())

            #Objetos → YOLO 3 clases
            labels=[]
            for _, _, x, y, w, h, _, _, cat in gt.get(frame, []):
                box=to_yolo(x, y, w, h, img_w, img_h)
                if box is None:
                    continue
                if (x, y, x+w, y+h)!=(max(0, x), max(0, y), min(img_w, x+w), min(img_h, y+h)):
                    n_clipped+=1
                cls=UAVDT_MAP[cat]
                cls_count[cls]+=1
                labels.append(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
            write_lines(out/"labels"/f"{name}.txt", labels)

            #Zonas ignoradas en píxeles
            zones=[f"{r[2]},{r[3]},{r[4]},{r[5]}" for r in ign.get(frame, [])]
            write_lines(out/"ignore"/f"{name}.txt", zones)

            rows.append([f"{name}.jpg", seq, frame, split_map.get(seq), img_w, img_h,
                         len(labels), len(zones)]+attr_map.get(seq, [None]*10))
        resolutions[f"{img_w}x{img_h}"]+=1

    header=["image", "seq", "frame", "split", "w", "h", "n_obj", "n_ignore"]+ATTR_NAMES
    with open(out/"index.csv", "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([header]+rows)

    #Verificaciones
    per_seq=Counter(r[1] for r in rows)
    print("== UAVDT")
    print(f"Secuencias:{len(per_seq)} (train {sum(v=='train' for v in split_map.values())} / test {sum(v=='test' for v in split_map.values())})")
    print(f"Frames muestreados:{len(rows):,} (1 de cada {STEP})")
    print(f"Frames por secuencia: min {min(per_seq.values())} · max {max(per_seq.values())}")
    print(f"Frames sin objetos:{sum(r[6]==0 for r in rows)}")
    print(f"Resoluciones (secuencias):{dict(resolutions)}")
    print(f"Cajas por clase:{ {CLASS_NAMES[k]: v for k, v in sorted(cls_count.items())} }")
    print(f"Cajas recortadas al borde:{n_clipped}")
    print(f"Zonas ignoradas:{sum(r[7] for r in rows):,}")
    print(f"Filas sin split:{sum(r[3] is None for r in rows)}")
    check_range(out)


#-------------------------------------------------------------- VisDrone val
def prepare_visdrone_val():
    root=PROJECT_ROOT/"data"/"visdrone"
    img_dir=root/"images"/"val"
    lbl_dir=root/"labels"/"val"
    raw_dir=root/"labels_original_backup"/"val"
    out=reset_out("visdrone_val")

    rows=[]
    before, after=Counter(), Counter()

    for jpg in sorted(img_dir.glob("*.jpg")):
        name=jpg.stem
        (out/"images"/jpg.name).symlink_to(jpg.resolve())
        img_w, img_h=Image.open(jpg).size

        #Remapeo 10 → 3 clases
        labels=[]
        lbl=lbl_dir/f"{name}.txt"
        for line in (lbl.read_text().splitlines() if lbl.exists() else []):
            parts=line.split()
            if len(parts)!=5:
                continue
            cls=int(parts[0])
            before[cls]+=1
            if cls not in VISDRONE_MAP:
                continue
            new=VISDRONE_MAP[cls]
            after[new]+=1
            labels.append(" ".join([str(new)]+parts[1:]))
        write_lines(out/"labels"/f"{name}.txt", labels)

        #Zonas ignoradas: clase 0 en la 6a columna del formato crudo
        zones=[]
        raw=raw_dir/f"{name}.txt"
        for line in (raw.read_text().splitlines() if raw.exists() else []):
            parts=[p for p in line.strip().split(",") if p!=""]
            if len(parts)>=6 and int(parts[5])==0:
                zones.append(",".join(parts[:4]))
        write_lines(out/"ignore"/f"{name}.txt", zones)

        rows.append([jpg.name, "", "", "val", img_w, img_h, len(labels), len(zones)])

    header=["image", "seq", "frame", "split", "w", "h", "n_obj", "n_ignore"]
    with open(out/"index.csv", "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([header]+rows)

    #Verificaciones
    vis_names={0:"pedestrian", 1:"people", 2:"bicycle", 3:"car", 4:"van",
               5:"truck", 6:"tricycle", 7:"awning-tricycle", 8:"bus", 9:"motor"}
    print("== VisDrone val")
    print(f"Imágenes:{len(rows)}")
    print(f"Cajas ANTES (10 clases):{ {vis_names[k]: v for k, v in sorted(before.items())} }")
    print(f"Cajas DESPUÉS (3 clases):{ {CLASS_NAMES[k]: v for k, v in sorted(after.items())} }")
    print(f"  car = car {before[3]} + van {before[4]} = {before[3]+before[4]}")
    print(f"Imágenes sin objetos (3 clases):{sum(r[6]==0 for r in rows)}")
    print(f"Zonas ignoradas:{sum(r[7] for r in rows):,}")
    check_range(out)


def check_range(out):
    """Ninguna coordenada YOLO fuera de [0,1]."""
    bad=0
    for f in (out/"labels").glob("*.txt"):
        for line in f.read_text().splitlines():
            if any(not 0<=float(v)<=1 for v in line.split()[1:]):
                bad+=1
    print(f"Coordenadas fuera de [0,1]:{bad}")
    print(f"Salida:{out.relative_to(PROJECT_ROOT)}")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["uavdt", "visdrone_val"])
    args=ap.parse_args()
    if args.dataset=="uavdt":
        prepare_uavdt()
    else:
        prepare_visdrone_val()