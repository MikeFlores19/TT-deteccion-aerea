"""
Filtro de zonas ignoradas con el criterio del toolkit oficial de VisDrone.

VisDrone marca regiones donde los objetos NO estan anotados (multitudes,
zonas borrosas, areas fuera de interes). El toolkit oficial descarta las
predicciones que caen dentro: ahi no hay ni acierto ni error, simplemente
no se evalua. Sin este filtro, una multitud dentro de una region ignorada
genera decenas de falsos positivos fantasma.

Criterio, de VisDrone2018-MOT-toolkit/eval/dropObjects.m:

    if(igrVal/(h*w)<0.5)              % conserva la prediccion
        idxLeft = cat(1, idxLeft, i);

  metrica : interseccion / area de la PREDICCION  (IoA, no IoU)
  umbral  : 0.5  -> se descarta si IoA >= 0.5
  alcance : TODAS las predicciones, no solo las no emparejadas
  zonas   : clases 0 (ignored-region) y 11 (others) del GT de VisDrone

Misma metrica y mismo umbral que TrackEval aplica en
kitti_2d_box.py:330-334 (do_ioa=True, umbral 0.5), con una diferencia de
alcance: KITTI lo limita a las predicciones no emparejadas, VisDrone lo
aplica a todas. Se sigue el de VisDrone porque es el protocolo del
dataset que se esta evaluando.

El GT NO se filtra: el toolkit oficial solo limpia el archivo de
resultados.

Comprobacion:  python -m src.evaluation.zonas_ignoradas
"""

from pathlib import Path

# se descarta la prediccion cuando al menos esta mitad de su area cae
# dentro de una zona ignorada. dropObjects.m conserva con "<0.5", asi
# que el descarte es ">=". TrackEval usa ">0.5+eps"; la diferencia solo
# aparece en el caso exacto de 0.5
UMBRAL_IOA=0.5

# columnas de la caja en MOTChallenge: frame,id,left,top,w,h,...
COLS_CAJA=slice(2,6)


def leer_zonas(ruta):
    """Lee un ignore.txt -> {frame:[(x1,y1,x2,y2),...]}.

    Devuelve un dict vacio si el archivo no existe, para que una
    secuencia sin zonas ignoradas no requiera un caso especial.
    """
    zonas={}
    ruta=Path(ruta)
    if not ruta.exists():
        return zonas
    for linea in ruta.read_text().split("\n"):
        linea=linea.strip()
        if not linea:
            continue
        campos=linea.split(",")
        frame=int(campos[0])
        x,y,w,h=(float(c) for c in campos[1:5])
        zonas.setdefault(frame,[]).append((x,y,x+w,y+h))
    return zonas


def caja_de_linea(campos):
    """Campos de una linea MOTChallenge -> (x1,y1,x2,y2)."""
    x,y,w,h=(float(c) for c in campos[COLS_CAJA])
    return (x,y,x+w,y+h)


def ioa(caja,zona):
    """Interseccion / area de la CAJA (no de la union). 0 si no se cruzan."""
    ax1,ay1,ax2,ay2=caja
    bx1,by1,bx2,by2=zona
    ancho=min(ax2,bx2)-max(ax1,bx1)
    alto=min(ay2,by2)-max(ay1,by1)
    if ancho<=0 or alto<=0:
        return 0.0
    area=(ax2-ax1)*(ay2-ay1)
    if area<=0:
        return 0.0
    return (ancho*alto)/area


def en_zona_ignorada(caja,zonas_del_frame):
    """True si la caja debe descartarse por caer en una zona ignorada."""
    return any(ioa(caja,z)>=UMBRAL_IOA for z in zonas_del_frame)


def demo():
    """Comprobacion del criterio con casos calculados a mano."""
    # zona de 100x100 en el origen
    zona=(0.0,0.0,100.0,100.0)

    # dentro por completo -> IoA 1.0 -> se descarta
    assert ioa((10,10,20,20),zona)==1.0
    assert en_zona_ignorada((10,10,20,20),[zona])

    # sin contacto -> IoA 0.0 -> se conserva
    assert ioa((200,200,210,210),zona)==0.0
    assert not en_zona_ignorada((200,200,210,210),[zona])

    # caja de 20x10 con la mitad izquierda dentro -> IoA 0.5 -> se descarta
    # (el umbral es >=, igual que el "<0.5 conserva" de dropObjects.m)
    media=ioa((90,0,110,10),zona)
    assert abs(media-0.5)<1e-9,media
    assert en_zona_ignorada((90,0,110,10),[zona])

    # un 40% dentro -> por debajo del umbral -> se conserva
    assert abs(ioa((90,0,115,10),zona)-0.4)<1e-9
    assert not en_zona_ignorada((90,0,115,10),[zona])

    # IoA no es IoU: una caja diminuta dentro de una zona enorme da IoA 1.0
    # aunque su IoU sea casi 0. Usar IoU aqui no descartaria casi nada
    assert ioa((50,50,51,51),zona)==1.0

    # basta con UNA zona que supere el umbral
    lejana=(500.0,500.0,600.0,600.0)
    assert en_zona_ignorada((10,10,20,20),[lejana,zona])
    assert not en_zona_ignorada((10,10,20,20),[lejana])

    # caja degenerada (area 0): no debe dividir por cero
    assert ioa((10,10,10,10),zona)==0.0

    # una secuencia sin ignore.txt no es un error, es un dict vacio
    assert leer_zonas("/no/existe/ignore.txt")=={}

    # el parseo va de left,top,w,h a esquinas
    assert caja_de_linea("5,1,10,20,30,40,0.9,1".split(","))==(10,20,40,60)

    print("zonas_ignoradas: todas las comprobaciones pasaron")


if __name__=="__main__":
    demo()
