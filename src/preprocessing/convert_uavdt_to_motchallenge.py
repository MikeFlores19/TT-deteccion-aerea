"""
Convierte UAVDT (Benchmark-M oficial) al formato MOTChallenge que espera TrackEval.

Mismo formato de salida que convert_visdrone_mot_to_motchallenge.py, para
que el resto del pipeline de la Fase 3 (dump_detections, run_tracking,
prepare_trackeval_clases, eval_tracking) funcione igual con --dataset uavdt.

GT de tracking:
  cajas e IDs  -> {seq}_gt.txt       (GT oficial de MOT de UAVDT)
  clase        -> {seq}_gt_whole.txt  (gt.txt no trae clase; se une por frame+id)
  zonas        -> {seq}_gt_ignore.txt

Las clases se escriben en indexacion VisDrone-MOT para que coincidan con
lo que produce run_tracking.py (clase YOLO + 1):
  UAVDT car=1 -> 4 · truck=2 -> 6 · bus=3 -> 9
El van (5) de las predicciones se trata como car en prepare_trackeval_clases.py.

Estructura de salida por secuencia:
  {seq}/img1/000001.jpg ...   enlaces simbolicos a los frames originales (no copia)
  {seq}/gt/gt.txt             frame,id,left,top,w,h,conf,cat,visibility
  {seq}/gt/ignore.txt         zonas ignoradas en pixeles: frame,x,y,w,h
  {seq}/seqinfo.ini

Todas las secuencias (50) van al split "all": el detector nunca vio UAVDT.

Uso:
  python -m src.preprocessing.convert_uavdt_to_motchallenge
"""

import configparser
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT=Path(__file__).parent.parent.parent
UAVDT_DIR=PROJECT_ROOT/"data"/"uavdt"
FRAMES_DIR=UAVDT_DIR/"UAV-benchmark-M"
GT_DIR=UAVDT_DIR/"UAV-benchmark-MOTD_v1.0"/"GT"
OUT_DIR=PROJECT_ROOT/"data"/"uavdt_mot"/"motchallenge"/"all"

#UAVDT graba a 30 fps (pagina oficial del dataset)
FRAME_RATE=30

#clase UAVDT -> indexacion VisDrone-MOT (car=4, truck=6, bus=9)
CLASE_A_VISDRONE={1:4, 2:6, 3:9}

#oclusion UAVDT (codigos NO ordinales) -> visibilidad aproximada
#1=sin, 4=pequena(1-30%), 3=media(30-70%), 2=grande(70-100%)
VISIBILIDAD={1:1.0, 4:0.85, 3:0.5, 2:0.15}


def escribir_seqinfo(ruta,nombre,n_frames,ancho,alto):
    """Genera el seqinfo.ini que TrackEval necesita para leer la secuencia"""
    cfg=configparser.ConfigParser()
    cfg.optionxform=str
    cfg["Sequence"]={
        "name":nombre,
        "imDir":"img1",
        "frameRate":str(FRAME_RATE),
        "seqLength":str(n_frames),
        "imWidth":str(ancho),
        "imHeight":str(alto),
        "imExt":".jpg",
    }
    with open(ruta,"w") as f:
        cfg.write(f,space_around_delimiters=False)


def convertir_secuencia(seq):
    dir_frames=FRAMES_DIR/seq
    dir_salida=OUT_DIR/seq
    if dir_salida.exists():
        shutil.rmtree(dir_salida)
    (dir_salida/"img1").mkdir(parents=True)
    (dir_salida/"gt").mkdir(parents=True)

    #img000001.jpg -> 000001.jpg (enlace, sin copiar)
    frames=sorted(dir_frames.glob("img*.jpg"))
    for ruta in frames:
        (dir_salida/"img1"/f"{ruta.stem[3:]}.jpg").symlink_to(ruta.resolve())
    ancho,alto=Image.open(frames[0]).size
    escribir_seqinfo(dir_salida/"seqinfo.ini",seq,len(frames),ancho,alto)

    #cajas e IDs del GT de MOT + clase y oclusion del GT de deteccion
    cols=["frame","id","x","y","w","h"]
    mot=pd.read_csv(GT_DIR/f"{seq}_gt.txt",header=None,usecols=range(6),names=cols)
    whole=pd.read_csv(GT_DIR/f"{seq}_gt_whole.txt",header=None,
                      names=cols+["out","occ","cat"])
    gt=mot.merge(whole[["frame","id","occ","cat"]],on=["frame","id"],how="left")
    sin_clase=int(gt["cat"].isna().sum())
    gt=gt.dropna(subset=["cat"])
    gt["cat"]=gt["cat"].astype(int).map(CLASE_A_VISDRONE)
    gt["vis"]=gt["occ"].astype(int).map(VISIBILIDAD)
    with open(dir_salida/"gt"/"gt.txt","w") as f:
        for r in gt.itertuples(index=False):
            f.write(f"{r.frame},{r.id},{r.x},{r.y},{r.w},{r.h},1,{r.cat},{r.vis:.2f}\n")

    #zonas ignoradas: frame,id,x,y,w,h,... -> frame,x,y,w,h
    ign=pd.read_csv(GT_DIR/f"{seq}_gt_ignore.txt",header=None,usecols=range(6),names=cols)
    with open(dir_salida/"gt"/"ignore.txt","w") as f:
        for r in ign.itertuples(index=False):
            f.write(f"{r.frame},{r.x},{r.y},{r.w},{r.h}\n")

    return len(frames),ancho,alto,len(gt),sin_clase,len(ign),gt["cat"].value_counts()


def main():
    print(f"origen : {UAVDT_DIR}")
    print(f"destino: {OUT_DIR}")
    secuencias=sorted(p.name for p in FRAMES_DIR.iterdir() if p.is_dir())
    print(f"\n=== {len(secuencias)} secuencias ===")

    tot_frames=tot_cajas=tot_sin=tot_ign=0
    tot_cls=pd.Series(dtype=int)
    for seq in secuencias:
        n,anc,alt,cajas,sin,ign,cls=convertir_secuencia(seq)
        tot_frames+=n; tot_cajas+=cajas; tot_sin+=sin; tot_ign+=ign
        tot_cls=tot_cls.add(cls,fill_value=0)
        print(f"  {seq}: {n} frames  {anc}x{alt}  cajas={cajas}  zonas_ignoradas={ign}"
              +(f"  SIN_CLASE={sin}" if sin else ""))

    nombres={4:"car",6:"truck",9:"bus"}
    print(f"\ntotal: {tot_frames} frames  {tot_cajas} cajas  {tot_ign} zonas ignoradas")
    print(f"por clase: "+"  ".join(f"{nombres[int(k)]}={int(v)}" for k,v in tot_cls.sort_index().items()))
    print(f"cajas de gt.txt sin clase en gt_whole: {tot_sin} (debe ser 0)")
    print("\nListo.")


if __name__=="__main__":
    main()
