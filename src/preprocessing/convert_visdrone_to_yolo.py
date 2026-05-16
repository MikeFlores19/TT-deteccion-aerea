"""
convert_visdrone_to_yolo.py

Convierte las anotaciones de VisDrone-DET2019 del formato original
al formato YOLO normalizado.

Formato original VisDrone:
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion

Formato YOLO:
    class_id  x_center_norm  y_center_norm  width_norm  height_norm
"""

from pathlib import Path
from PIL import Image

#Configuración
PROJECT_ROOT=Path(__file__).resolve().parent.parent.parent #obtenemos la ruta raiz del proyecto (file es el archivo donde estamos parados y con .parent vamos subiendo)
VISDRONE_ROOT=PROJECT_ROOT/"data"/"visdrone"
SPLITS=["train","val","test"]

#Funcion dede conversion
def convert_file(label_path: Path, img_path: Path)->int:
    """
    Convierte un archivo de anotaciones VisDrone a formato YOLO.

    Args:
        label_path : ruta al archivo .txt original de VisDrone
        img_path   : ruta a la imagen correspondiente (para obtener W y H)

    Returns:
        número de detecciones válidas convertidas
    """

    #Obtener resolucion real de la imagen
    with Image.open(img_path) as img:
        W,H=img.size

    yolo_lines=[]

    with open(label_path) as f:
        for line in f:
            parts=line.strip().split(",")
            if len(parts)!=8:
                continue

            x,y,w,h,score,cat_id,_,_=[int(p) for p in parts]

            #Filtrar regiones ignoradas
            if score==0 or cat_id==0:
                continue

            #Convertir a 0-indexed
            cat_id_0=cat_id-1

            #Convertir a formato YOLO normalizado
            x_center=(x+w/2)/W
            y_center=(y+h/2)/H
            w_norm=w/W
            h_norm=h/H

            #Clampear entre 0 y 1 por seguridad
            x_center=max(0.0, min(1.0, x_center))
            y_center=max(0.0, min(1.0, y_center))
            w_norm=max(0.0, min(1.0, w_norm))
            h_norm=max(0.0, min(1.0, h_norm))

            #Agregamos los valores a la lista yolo_lines
            yolo_lines.append(
                f"{cat_id_0} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

    #Sobreescribir el archivo con el formato YOLO
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    return len(yolo_lines)

#Main
def main():
    print(f"PROJECT_ROOT:{PROJECT_ROOT}")
    print(f"VISDRONE_ROOT:{VISDRONE_ROOT}")
    print(f"Existe:{VISDRONE_ROOT.exists()}")

    #Contadores globales de todos los splits
    total_converted=0
    total_files=0

    #Recorre train, val y test
    for split in SPLITS:
        #Rutas de anotaciones e imágenes del split actual
        labels_dir=VISDRONE_ROOT/"labels"/split
        images_dir=VISDRONE_ROOT/"images"/split

        #Lista de todos los .txt del split
        label_files=sorted(labels_dir.glob("*.txt"))
        print(f"\n[{split}] Procesando {len(label_files):,} archivos...")

        #Contador de detecciones del split actual
        converted_split=0

        #Procesa cada archivo de anotaciones
        for label_path in label_files:
            #Busca la imagen correspondiente al .txt
            img_path=images_dir/(label_path.stem+".jpg")

            #Si no existe la imagen, avisa y salta
            if not img_path.exists():
                print(f"Imagen no encontrada: {img_path.name}")
                continue

            #Convierte el archivo y acumula detecciones
            n=convert_file(label_path, img_path)
            converted_split+=n

        #Resumen del split
        print(f"[{split}] {len(label_files):,} archivos · {converted_split:,} detecciones convertidas")

        #Acumula en totales globales
        total_converted+=converted_split
        total_files+=len(label_files)

    #Resumen final de todos los splits
    print()
    print(f"Total archivos procesados:{total_files:,}")
    print(f"Total detecciones válidas:{total_converted:,}")
    print(f"Conversión completada")

#Ejecuta main() solo si se corre directamente, no si se importa
if __name__=="__main__":
    main()