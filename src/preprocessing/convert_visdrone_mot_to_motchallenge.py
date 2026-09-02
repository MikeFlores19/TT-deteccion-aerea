"""
Convierte VisDrone-MOT al formato MOTChallenge que espera TrackEval.

VisDrone (10 cols): frame,id,left,top,w,h,score,cat,trunc,occ
MOT      (9 cols):  frame,id,left,top,w,h,conf,cat,visibility

Estructura de salida por secuencia:
  {seq}/img1/000001.jpg ... 000464.jpg
  {seq}/gt/gt.txt
  {seq}/seqinfo.ini
"""

import configparser #para escribir el seqinfo.ini
import shutil #para copiar archivos
from pathlib import Path #para manejar rutas de archivos

from PIL import Image #para leer el tamaño de las imagenes

#raiz del proyecto : src/preprocessing/archivo.py --> 3 niveles arriba
PROJECT_ROOT=Path(__file__).parent.parent.parent
RAW_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"raw"
OUT_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"motchallenge"

# VisDrone no documenta los fps por secuencia; 30 es la convencion usada
# en la literatura de este dataset. No afecta a MOTA/IDF1/HOTA/IDsw
FRAME_RATE=30

#clases descartadas 0=ignores-region, 11=others (no son objetos reales)
CLASES_EXCLUIDAS={0,11}

#nombre de la carpeta cruda -> nombre dle split de salida
SPLITS={
    "VisDrone2019-MOT-val":"val",
    "VisDrone2019-MOT-test-dev":"test-dev",
}

def convertir_anotacion(ruta_origen,ruta_destino):
    """Traduce un .txt de VisDrone-MOT al formato MOTChallenge."""

    conservadas=0
    descartadas=0

    with open (ruta_origen) as f_in, open(ruta_destino,"w") as f_out:
        for linea in f_in:
            linea=linea.strip()
            if not linea:
                continue

            campos=linea.split(",")
            frame,obj_id,left,top,w,h,score,cat,_trunc,occ=campos[:10]

            if int(cat) in CLASES_EXCLUIDAS:
                descartadas+=1
                continue

            #occ: 0=sin oclusion, 1=parcial, 2=fuerte -> visibility 1.0/0.5/0.0
            visibility=max(0.0,1.0-int(occ)*0.5) #se pone max por si hay algun valor mayor a 2, que no deberia ocurrir

            f_out.write(f"{frame},{obj_id},{left},{top},{w},{h},{score},{cat},{visibility:.1f}\n")
            conservadas+=1

    return conservadas,descartadas


def escribir_seqinfo(ruta_destino,nombre,n_frames,ancho,alto):
    """Generar el seq.info que TrackEval necesita para leer la secuencia"""

    cfg=configparser.ConfigParser() #sirve para leer y escribir archivos .ini que son como diccionarios

    #evita que confiparser convierta claves a minusculas

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

    with open(ruta_destino,"w") as f:
        cfg.write(f,space_around_delimiters=False) #space_around_delimiters=False evita que ponga espacios alrededor de los =, que es lo que espera TrackEval


def convertir_secuencia(dir_frames,ruta_anotacion,dir_salida):
    """Convierte una secuencia completa: imagenes + anotacion + seqinfo."""
    nombre=dir_frames.name
    (dir_salida/"img1").mkdir(parents=True,exist_ok=True)
    (dir_salida/"gt").mkdir(parents=True,exist_ok=True)

    frames=sorted(dir_frames.glob("*.jpg"))

    if not frames:
        raise RuntimeError(f"No se encontraron frames en {dir_frames}")

    #Visdrone usa 7 digitos, MOTChallenge usa 6 digitos, asi que se renombra
    for i,origen in enumerate(frames,start=1):
        shutil.copy2(origen,dir_salida/"img1"/f"{i:06d}.jpg")

    ancho,alto=Image.open(frames[0]).size #se toma el primero porque todas las imagenes de la secuencia tienen el mismo tamaño
    escribir_seqinfo(dir_salida/"seqinfo.ini",nombre,len(frames),ancho,alto) 

    conservadas,descartadas=convertir_anotacion(ruta_anotacion,dir_salida/"gt"/"gt.txt")
    return len(frames),ancho,alto,conservadas,descartadas

def convertir_split(nombre_crudo,nombre_split):
    """Procesa todas las secuencias de un split (val o test-dev)."""
    dir_split=RAW_DIR/nombre_crudo
    if not dir_split.exists():
        print(f"[SALTADO] no existe {dir_split}")
        return

    dir_sequences=dir_split/"sequences"
    dir_annotations=dir_split/"annotations"
    dir_destino=OUT_DIR/nombre_split

    secuencias=sorted(p for p in dir_sequences.iterdir() if p.is_dir()) #se hace sorted para que el orden sea reproducible y no dependa del orden de los archivos en el sistema de archivos
    print(f"\n=== {nombre_split}: {len(secuencias)} secuencias ===")

    for dir_frames in secuencias:
        ruta_anotacion=dir_annotations/f"{dir_frames.name}.txt"
        if not ruta_anotacion.exists():
            print(f"  [ERROR] falta anotacion de {dir_frames.name}")
            continue

        n,anc,alt,cons,desc=convertir_secuencia(
            dir_frames,ruta_anotacion,dir_destino/dir_frames.name
        )
        print(f"  {dir_frames.name}: {n} frames  {anc}x{alt}  "f"cajas={cons} descartadas={desc}")


def main():
    print(f"origen : {RAW_DIR}")
    print(f"destino: {OUT_DIR}")
    for nombre_crudo,nombre_split in SPLITS.items():
        convertir_split(nombre_crudo,nombre_split)
    print("\nListo.")


if __name__=="__main__":
    main()





            
