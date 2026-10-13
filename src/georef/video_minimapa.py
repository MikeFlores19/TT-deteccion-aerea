"""
Fase 5 (paso 9.5, opcion B): video anotado con mini-mapa georreferenciado.

Mismas cajas e IDs que el modo normal (results.txt de la Jetson, sin inferencia
nueva) y, en la esquina, un mini-mapa satelital sincronizado con cada frame:
  - recorrido completo del dron (tenue) y su posicion actual
  - cuña blanca: hacia donde mira la camara (rumbo +- FOV/2)
  - los objetos georreferenciados en ese frame, con el MISMO color que su caja,
    y una estela corta de su recorrido
En el video, un punto en el pie de la caja marca las detecciones que si se
pudieron georreferenciar (las que estan en la zona valida).

Se arma desde el video crudo (como visualize.py) para no comprimir dos veces.

Entrada: data/dji_mini3/<clip>.mp4, runs/video/dji_<clip>/{results.txt,geo.csv.gz,dron.csv}
Salida : results/videos/jetson_minimapa/dji_<clip>_normal.mp4

Uso:
  python -m src.georef.video_minimapa
  python -m src.georef.video_minimapa --solo video29 --max-frames 300
"""

import argparse
import math
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd

from src.georef.inventario_vuelos import PROJECT_ROOT, OUT_DIR, rel
from src.georef.mapa import fondo, lonlat_a_px, ATRIBUCION
from src.georef.pose import F_PX, ANCHO
from src.tracking.visualize import NOMBRES_CLASE, color_por_id, dibujar_caja, leer_tracks

CLIPS_DIR=PROJECT_ROOT/"data"/"dji_mini3"
RUNS_DIR=PROJECT_ROOT/"runs"/"video"
OUT_VID=PROJECT_ROOT/"results"/"videos"/"jetson_minimapa"
UTC_OFFSET_H=-6
PANEL=440
MARGEN_PANEL=20
MARGEN_M=15.0
ESTELA_FRAMES=30
HFOV_DEG=2*math.degrees(math.atan((ANCHO/2)/F_PX))


def preparar_mapa(dron,geo):
    """Fondo satelital cuadrado que cubre el clip, ya escalado al panel, y su proyeccion."""
    lats=np.concatenate([dron["lat"].dropna().to_numpy(),geo["lat"].to_numpy()])
    lons=np.concatenate([dron["lon"].dropna().to_numpy(),geo["lon"].to_numpy()])
    #percentiles: unos cuantos puntos lejanos no deben encoger todo lo demas
    la_c=(np.percentile(lats,1)+np.percentile(lats,99))/2
    lo_c=(np.percentile(lons,1)+np.percentile(lons,99))/2
    m_lat=6371008.8*math.pi/180
    m_lon=m_lat*math.cos(math.radians(la_c))
    semi=max((np.percentile(lats,99)-np.percentile(lats,1))*m_lat,
             (np.percentile(lons,99)-np.percentile(lons,1))*m_lon)/2+MARGEN_M
    la0,la1=la_c-semi/m_lat,la_c+semi/m_lat
    lo0,lo1=lo_c-semi/m_lon,lo_c+semi/m_lon
    img,z,x0,y0,faltan=fondo(la0,la1,lo0,lo1)
    kx,ky=PANEL/img.shape[1],PANEL/img.shape[0]
    img=cv2.resize(img,(PANEL,PANEL),interpolation=cv2.INTER_AREA)

    def px(lon,lat):
        x,y=lonlat_a_px(lon,lat,z)
        return (x-x0)*kx,(y-y0)*ky

    mpp=156543.03392*math.cos(math.radians(la_c))/2**z/kx
    return img,px,mpp,faltan


def capa_fija(img,px,mpp,dron):
    """Fondo + recorrido completo del dron + escala + norte (se dibuja una sola vez)."""
    base=img.copy()
    x,y=px(dron["lon"].to_numpy(),dron["lat"].to_numpy())
    ok=~(np.isnan(x)|np.isnan(y))
    pts=np.stack([x[ok],y[ok]],1).astype(np.int32)
    if len(pts)>1:
        cv2.polylines(base,[pts],False,(255,255,255),1,cv2.LINE_AA)
    largo=next(v for v in (5,10,20,25,50,100,200) if v/mpp>=PANEL*0.15)
    bx,by=12,PANEL-14
    cv2.line(base,(bx,by),(bx+int(largo/mpp),by),(255,255,255),3)
    cv2.putText(base,f"{largo} m",(bx,by-6),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    cv2.arrowedLine(base,(PANEL-18,44),(PANEL-18,14),(255,255,255),2,tipLength=0.35)
    cv2.putText(base,"N",(PANEL-24,58),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(base,"Esri",(PANEL-34,PANEL-6),cv2.FONT_HERSHEY_SIMPLEX,0.35,(230,230,230),1,cv2.LINE_AA)
    return base


def generar(clip,ub_fila,max_frames=None):
    carpeta=RUNS_DIR/f"dji_{clip}"
    dron=pd.read_csv(carpeta/"dron.csv")
    geo=pd.read_csv(carpeta/"geo.csv.gz")
    tracks=leer_tracks(carpeta/"results.txt")
    geo_fr={f:g for f,g in geo.groupby("frame")}
    alcance=float(geo["dist_m"].quantile(0.95)) if len(geo) else 50.0

    img,px,mpp,faltan=preparar_mapa(dron,geo)
    base=capa_fija(img,px,mpp,dron)
    inicio=pd.to_datetime(ub_fila["inicio_utc"])+pd.Timedelta(hours=UTC_OFFSET_H)
    vuelo=int(ub_fila["vuelo"])

    cap=cv2.VideoCapture(str(CLIPS_DIR/f"{clip}.mp4"))
    fps=cap.get(cv2.CAP_PROP_FPS)
    ancho,alto=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total=min(total,max_frames)
    ruta=OUT_VID/f"dji_{clip}_normal.mp4"
    ruta.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(ruta),cv2.VideoWriter_fourcc(*"mp4v"),fps,(ancho,alto))
    estelas=defaultdict(lambda:deque(maxlen=ESTELA_FRAMES))
    x_p,y_p=ancho-PANEL-MARGEN_PANEL,alto-PANEL-MARGEN_PANEL

    n=0
    while n<total:
        ok,frame=cap.read()
        if not ok:
            break
        n+=1
        #cajas: mismo estilo que process_video (modo normal)
        tr=tracks.get(n,[])
        for tid,x,y,w,h,conf,cls in tr:
            dibujar_caja(frame,x,y,w,h,f"{NOMBRES_CLASE.get(cls,'?')} #{tid}",color_por_id(tid))
        cv2.putText(frame,f"frame {n}  objetos: {len(tr)}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,255,255),2,cv2.LINE_AA)

        panel=base.copy()
        d=dron.iloc[n-1] if n-1<len(dron) else None
        if d is not None and not np.isnan(d["lat"]):
            #cuña de la camara: rumbo +- HFOV/2 hasta el alcance tipico de la zona valida
            dx,dy=px(np.array([d["lon"]]),np.array([d["lat"]]))
            c=(int(dx[0]),int(dy[0]))
            r=alcance/mpp
            pts=[c]
            for a in (d["rumbo"]-HFOV_DEG/2,d["rumbo"]+HFOV_DEG/2):
                pts.append((int(c[0]+r*math.sin(math.radians(a))),int(c[1]-r*math.cos(math.radians(a)))))
            capa=panel.copy()
            tri=np.array(pts,np.int32)
            cv2.fillPoly(capa,[tri],(255,255,255))
            panel=cv2.addWeighted(capa,0.35,panel,0.65,0)
            #borde del angulo de vision para que se distinga sobre cualquier fondo
            cv2.polylines(panel,[tri],True,(255,255,255),2,cv2.LINE_AA)
            cv2.circle(panel,c,6,(0,0,0),-1)
            cv2.circle(panel,c,5,(255,255,255),-1)

        g=geo_fr.get(n)
        vivos=set()
        if g is not None:
            gx,gy=px(g["lon"].to_numpy(),g["lat"].to_numpy())
            for tid,u,v,xm,ym in zip(g["id"],g["u"],g["v"],gx,gy):
                color=color_por_id(int(tid))
                vivos.add(int(tid))
                estelas[int(tid)].append((int(xm),int(ym)))
                #en el video: punto en el pie de la caja = detección georreferenciada
                cv2.circle(frame,(int(u),int(v)),4,color,-1)
                cv2.circle(frame,(int(u),int(v)),4,(0,0,0),1)
        for tid in list(estelas):
            if tid not in vivos:
                del estelas[tid]
                continue
            e=list(estelas[tid])
            color=color_por_id(tid)
            if len(e)>1:
                cv2.polylines(panel,[np.array(e,np.int32)],False,color,1,cv2.LINE_AA)
            cv2.circle(panel,e[-1],4,color,-1)
            cv2.circle(panel,e[-1],4,(255,255,255),1)

        hora=(inicio+pd.Timedelta(seconds=(n-1)/fps)).strftime("%H:%M:%S")
        cv2.rectangle(panel,(0,0),(PANEL,22),(0,0,0),-1)
        #objetos en el mapa = detecciones de este frame dentro de la zona valida
        cv2.putText(panel,f"Vuelo {vuelo}  {hora}  objetos en mapa: {len(vivos)}",(6,16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(frame,(x_p-2,y_p-2),(x_p+PANEL+1,y_p+PANEL+1),(255,255,255),2)
        frame[y_p:y_p+PANEL,x_p:x_p+PANEL]=panel
        writer.write(frame)
        if n%500==0:
            print(f"    {n}/{total}")
    cap.release()
    writer.release()
    return ruta,n,faltan


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--solo",default=None)
    ap.add_argument("--max-frames",type=int,default=None)
    args=ap.parse_args()

    ub=pd.read_csv(OUT_DIR/"ubicacion_clips.csv")
    ub=ub[ub["estado"]=="OK"]
    if args.solo:
        ub=ub[ub["clip"]==args.solo]
    for _,f in ub.iterrows():
        print(f"{f['clip']}  (vuelo {int(f['vuelo'])})")
        ruta,n,faltan=generar(f["clip"],f,args.max_frames)
        print(f"  {n} frames -> {rel(ruta)}"+(f"  ({faltan} teselas sin descargar)" if faltan else ""))
    print(f"\nfondo: {ATRIBUCION}")


if __name__=="__main__":
    main()
