"""
Fase 5 (paso 9.3a): verificacion del modelo de camara y zona valida de la imagen.

Por cada clip se toma el frame de la mitad y se dibuja, con la pose del CSV
en ese instante:
  - el horizonte predicho (amarillo): si coincide con el horizonte real, el
    FOV (f=1378 px) y el GIMBAL.pitch/roll son correctos
  - curvas de distancia en el suelo (10, 25, 50, 100 m) desde la vertical del dron
  - en rojo, la zona NO georreferenciable: sobre el horizonte o con error
    estimado > ERROR_MAX_M (supuesto de suelo plano)

Salida: results/figures/georef/camara/<clip>.jpg
        results/tables/georef/zona_valida.csv

Uso:
  python -m src.georef.verificar_camara
  python -m src.georef.verificar_camara --solo video09
"""

import argparse
import subprocess

import cv2
import numpy as np
import pandas as pd

from src.georef.inventario_vuelos import PROJECT_ROOT, OUT_DIR, rel
from src.georef.pose import (PoseClip, a_suelo, error_estimado, horizonte_v,
                             ANCHO, ALTO, CX, F_PX, ERROR_MAX_M)

CLIPS_DIR=PROJECT_ROOT/"data"/"dji_mini3"
FIG_DIR=PROJECT_ROOT/"results"/"figures"/"georef"/"camara"
DISTANCIAS_M=(10,25,50,100)
#rejilla para evaluar la zona valida (px)
PASO_U,PASO_V=16,8


def frame_en(ruta,t):
    png=subprocess.run(["ffmpeg","-v","error","-ss",f"{t:.3f}","-i",str(ruta),"-frames:v","1",
                        "-f","image2pipe","-vcodec","png","-"],capture_output=True,check=True).stdout
    return cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR)


def dif_angular(a,b):
    return (np.asarray(a)-np.asarray(b)+180)%360-180


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--solo",default=None)
    ap.add_argument("--fraccion",type=float,default=0.5,help="momento del clip a revisar (0-1)")
    args=ap.parse_args()

    ub=pd.read_csv(OUT_DIR/"ubicacion_clips.csv")
    clips=[c for c in ub.loc[ub["estado"]=="OK","clip"] if not args.solo or c==args.solo]

    uu,vv=np.meshgrid(np.arange(0,ANCHO,PASO_U)+PASO_U/2,np.arange(0,ALTO,PASO_V)+PASO_V/2)
    filas=[]
    for clip in clips:
        pc=PoseClip(clip)
        t=args.fraccion*pc.duracion
        p={k:float(v[0]) for k,v in pc.en(t).items()}
        todo=pc.en(np.arange(0,pc.duracion,0.5))
        img=frame_en(CLIPS_DIR/f"{clip}.mp4",t)

        #zona valida en la rejilla
        E,N,dep=a_suelo(uu,vv,p["h"],p["pitch"],p["roll"],p["yaw"])
        err=error_estimado(uu,vv,p["h"],p["pitch"],p["roll"],p["yaw"])
        valido=~np.isnan(E)&(err<=ERROR_MAX_M)
        dist=np.hypot(E,N)

        capa=img.copy()
        for i in range(uu.shape[0]):
            for j in range(uu.shape[1]):
                if not valido[i,j]:
                    x0,y0=int(uu[i,j]-PASO_U/2),int(vv[i,j]-PASO_V/2)
                    cv2.rectangle(capa,(x0,y0),(x0+PASO_U,y0+PASO_V),(0,0,200),-1)
        img=cv2.addWeighted(capa,0.35,img,0.65,0)

        #curvas de distancia (rejilla fina en v): en cada columna, la fila mas baja que la alcanza
        uf,vf=np.meshgrid(np.arange(0,ANCHO+1,PASO_U),np.arange(ALTO))
        Ef,Nf,_=a_suelo(uf,vf,p["h"],p["pitch"],p["roll"],p["yaw"])
        distf=np.hypot(Ef,Nf)
        for d in DISTANCIAS_M:
            pts=[]
            for j in range(uf.shape[1]):
                col=distf[:,j]
                idx=np.where(~np.isnan(col)&(col>=d))[0]
                if len(idx) and idx.max()<ALTO-1:
                    pts.append((int(uf[0,j]),int(idx.max())))
            if len(pts)>1:
                cv2.polylines(img,[np.array(pts,np.int32)],False,(255,255,0),2,cv2.LINE_AA)
                cv2.putText(img,f"{d} m",(pts[len(pts)//2][0],pts[len(pts)//2][1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2,cv2.LINE_AA)

        #horizonte predicho
        xs=np.array([0,ANCHO-1])
        ys=horizonte_v(xs,p["pitch"],p["roll"],p["yaw"])
        cv2.line(img,(0,int(ys[0])),(ANCHO-1,int(ys[1])),(0,255,255),3,cv2.LINE_AA)
        cv2.putText(img,"horizonte predicho",(20,int(ys[0])-10),cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,255,255),2,cv2.LINE_AA)

        pct=100*valido.mean()
        hud=(f"{clip}  vuelo {pc.vuelo}  t={t:.1f}s  h={p['h']:.1f} m  pitch={p['pitch']:.1f}  "
             f"roll={p['roll']:.1f}  rumbo={p['yaw']:.0f}  zona valida={pct:.0f}%")
        cv2.rectangle(img,(0,ALTO-44),(ANCHO,ALTO),(0,0,0),-1)
        cv2.putText(img,hud,(12,ALTO-14),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2,cv2.LINE_AA)
        FIG_DIR.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(FIG_DIR/f"{clip}.jpg"),cv2.resize(img,(ANCHO//2,ALTO//2),interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY,90])

        #distancia al borde inferior y alcance de la zona valida en la columna central
        jc=uu.shape[1]//2
        col_ok=valido[:,jc]
        fila={
            "clip":clip,"vuelo":pc.vuelo,"t_s":round(t,2),"h_m":round(p["h"],2),
            "pitch":round(p["pitch"],1),"roll":round(p["roll"],1),"rumbo":round(p["yaw"],1),
            "horizonte_v_px":round(float(horizonte_v(CX,p["pitch"],p["roll"],p["yaw"])),1),
            "pct_zona_valida":round(pct,1),
            "dist_borde_inferior_m":round(float(dist[-1,jc]),1),
            "alcance_valido_m":round(float(np.nanmax(np.where(col_ok,dist[:,jc],np.nan))),1) if col_ok.any() else None,
            #GIMBAL.yaw vs OSD.yaw a lo largo del clip (mediana del valor absoluto)
            "dif_rumbo_gimbal_osd":round(float(np.nanmedian(np.abs(dif_angular(todo["yaw"],todo["yaw_osd"])))),1),
            "pitch_min_clip":round(float(np.nanmin(todo["pitch"])),1),
            "pitch_max_clip":round(float(np.nanmax(todo["pitch"])),1),
            "h_min_clip":round(float(np.nanmin(todo["h"])),1),
            "h_max_clip":round(float(np.nanmax(todo["h"])),1),
        }
        filas.append(fila)
        print(f"{clip}  v{pc.vuelo:<2d} h={fila['h_m']:5.1f} m  pitch={fila['pitch']:6.1f}  "
              f"horizonte_v={fila['horizonte_v_px']:6.0f}px  zona_valida={fila['pct_zona_valida']:4.0f}%  "
              f"borde_inf={fila['dist_borde_inferior_m']:5.1f} m  alcance={fila['alcance_valido_m']} m  "
              f"dif_rumbo={fila['dif_rumbo_gimbal_osd']}")

    res=pd.DataFrame(filas)
    salida=OUT_DIR/"zona_valida.csv"
    if args.solo and salida.exists():
        prev=pd.read_csv(salida)
        res=pd.concat([prev[prev["clip"]!=args.solo],res]).sort_values("clip")
    res.to_csv(salida,index=False)
    print(f"\nf={F_PX:.0f} px  error max={ERROR_MAX_M:.0f} m")
    print(f"{rel(salida)}\n{rel(FIG_DIR)}/")


if __name__=="__main__":
    main()
