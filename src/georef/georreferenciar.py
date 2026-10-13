"""
Fase 5 (paso 9.3b): georreferenciacion de las detecciones de los 27 clips.

Para cada deteccion de runs/video/dji_<clip>/results.txt (config A en la Jetson):
  1) instante: t_clip=(frame-1)/fps -> pose del dron (pose.PoseClip: A y B del 9.2)
  2) punto de apoyo en el suelo: centro del borde inferior de la caja
  3) rayo de la camara -> suelo plano -> (E,N) -> lat/lon
  4) error estimado; solo se conserva si cae en la zona valida
     (bajo el horizonte y error <= ERROR_MAX_M)

Salida por clip (runs/video/dji_<clip>/):
  geo.csv.gz  una fila por deteccion georreferenciada (comprimido: ~700k filas en total)
  dron.csv  pose del dron en cada frame (trayectoria para el mapa y el mini-mapa)
Resumen:
  results/tables/georef/objetos_geo.csv    una fila por objeto (track)
  results/tables/georef/resumen_georef.csv una fila por clip

Limitaciones (documentar): suelo plano a la altura del despegue (objetos en un
puente o en un desnivel quedan desplazados); en vehiculos el borde inferior de
la caja es el punto mas cercano, no el centro (~1-2 m); el error es estimado con
incertidumbres supuestas (pose.SIGMA_*), no medido contra puntos de control.

Uso:
  python -m src.georef.georreferenciar
  python -m src.georef.georreferenciar --solo video09
"""

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

from src.georef.inventario_vuelos import PROJECT_ROOT, OUT_DIR, rel
from src.georef.pose import (PoseClip, a_suelo, error_estimado, enu_a_latlon,
                             ERROR_MAX_M, F_PX)

RUNS_DIR=PROJECT_ROOT/"runs"/"video"
NOMBRES_CLASE={1:"pedestrian",2:"people",3:"bicycle",4:"car",5:"van",
               6:"truck",7:"tricycle",8:"awning-tricycle",9:"bus",10:"motor"}
#un objeto entra al resumen si tiene al menos esto de frames georreferenciados
MIN_FRAMES_OBJETO=3


def leer_results(ruta):
    cols=["frame","id","x","y","w","h","conf","cls","_a","_b"]
    d=pd.read_csv(ruta,header=None,names=cols)
    return d.drop(columns=["_a","_b"])


def dispersion_m(lat,lon):
    """Mediana de la distancia de cada punto a la mediana del track (m)."""
    lat0,lon0=np.median(lat),np.median(lon)
    dn=np.radians(lat-lat0)*6371008.8
    de=np.radians(lon-lon0)*6371008.8*np.cos(np.radians(lat0))
    return float(np.median(np.hypot(de,dn)))


def procesar(clip,ub_fila):
    pc=PoseClip(clip)
    fps=float(ub_fila["fps_clip"])
    inicio=pd.to_datetime(ub_fila["inicio_utc"])
    carpeta=RUNS_DIR/f"dji_{clip}"
    det=leer_results(carpeta/"results.txt")

    #pose del dron en cada frame del clip
    n_frames=int(ub_fila["frames"])
    t_fr=np.arange(n_frames)/fps
    pf=pc.en(t_fr)
    dron=pd.DataFrame({"frame":np.arange(1,n_frames+1),"t_clip_s":np.round(t_fr,3),
                       "lat":np.round(pf["lat"],7),"lon":np.round(pf["lon"],7),
                       "h_m":np.round(pf["h"],2),"pitch":np.round(pf["pitch"],2),
                       "roll":np.round(pf["roll"],2),"rumbo":np.round(pf["yaw"],2)})
    dron.to_csv(carpeta/"dron.csv",index=False)

    #pose en el instante de cada deteccion (mismo frame -> misma pose)
    k=det["frame"].to_numpy()-1
    p={c:pf[c][k] for c in ("lat","lon","h","pitch","roll","yaw")}
    u=(det["x"]+det["w"]/2).to_numpy()
    v=(det["y"]+det["h"]).to_numpy()
    E,N,dep=a_suelo(u,v,p["h"],p["pitch"],p["roll"],p["yaw"])
    err=error_estimado(u,v,p["h"],p["pitch"],p["roll"],p["yaw"])
    lat,lon=enu_a_latlon(p["lat"],p["lon"],E,N)
    valido=~np.isnan(E)&~np.isnan(lat)&(err<=ERROR_MAX_M)

    geo=pd.DataFrame({
        "frame":det["frame"],"t_clip_s":np.round(k/fps,3),"id":det["id"],
        "clase":det["cls"].map(NOMBRES_CLASE),"conf":det["conf"],
        "u":np.round(u,1),"v":np.round(v,1),
        "lat":np.round(lat,7),"lon":np.round(lon,7),
        "dist_m":np.round(np.hypot(E,N),2),"depresion_deg":np.round(dep,2),"error_m":np.round(err,2),
    })[valido]
    #gz: sin comprimir serian ~55 MB en el repo
    geo.to_csv(carpeta/"geo.csv.gz",index=False)

    #un objeto por track
    objetos=[]
    for tid,g in geo.groupby("id"):
        if len(g)<MIN_FRAMES_OBJETO:
            continue
        total=int((det["id"]==tid).sum())
        objetos.append({
            "clip":clip,"vuelo":pc.vuelo,"id":int(tid),
            "clase":g["clase"].mode().iat[0],
            "frames_georef":len(g),"frac_georef":round(len(g)/total,3),
            "lat":round(float(g["lat"].median()),7),"lon":round(float(g["lon"].median()),7),
            "error_mediano_m":round(float(g["error_m"].median()),2),
            "dist_mediana_m":round(float(g["dist_m"].median()),1),
            #si el objeto esta quieto, la dispersion mide la precision de la georreferencia
            "dispersion_m":round(dispersion_m(g["lat"].to_numpy(),g["lon"].to_numpy()),2),
            "inicio_utc":(inicio+timedelta(seconds=float(g["t_clip_s"].min()))).isoformat(timespec="milliseconds"),
            "fin_utc":(inicio+timedelta(seconds=float(g["t_clip_s"].max()))).isoformat(timespec="milliseconds"),
        })

    resumen={
        "clip":clip,"vuelo":pc.vuelo,"detecciones":len(det),
        "georreferenciadas":int(valido.sum()),
        "pct_georref":round(100*valido.mean(),1) if len(det) else 0.0,
        "ids":int(det["id"].nunique()),"objetos_georef":len(objetos),
        "error_mediano_m":round(float(geo["error_m"].median()),2) if len(geo) else None,
        "dist_mediana_m":round(float(geo["dist_m"].median()),1) if len(geo) else None,
        "h_mediana_m":round(float(np.nanmedian(pf["h"])),1),
        "pitch_mediano":round(float(np.nanmedian(pf["pitch"])),1),
    }
    return resumen,objetos,rel(carpeta)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--solo",default=None)
    args=ap.parse_args()

    ub=pd.read_csv(OUT_DIR/"ubicacion_clips.csv")
    ub=ub[ub["estado"]=="OK"]
    if args.solo:
        ub=ub[ub["clip"]==args.solo]

    resumenes,objetos=[],[]
    for _,f in ub.iterrows():
        r,o,carpeta=procesar(f["clip"],f)
        resumenes.append(r)
        objetos+=o
        print(f"{r['clip']}  v{r['vuelo']:<2d} detecciones={r['detecciones']:6d}  georref={r['pct_georref']:5.1f}%  "
              f"objetos={r['objetos_georef']:4d}/{r['ids']:<4d}  error_med={r['error_mediano_m']} m  "
              f"dist_med={r['dist_mediana_m']} m  ->  {carpeta}/geo.csv.gz")

    res=pd.DataFrame(resumenes)
    obj=pd.DataFrame(objetos)
    s_res,s_obj=OUT_DIR/"resumen_georef.csv",OUT_DIR/"objetos_geo.csv"
    if args.solo and s_res.exists():
        prev=pd.read_csv(s_res)
        res=pd.concat([prev[prev["clip"]!=args.solo],res]).sort_values("clip")
        if s_obj.exists():
            prev=pd.read_csv(s_obj)
            obj=pd.concat([prev[prev["clip"]!=args.solo],obj]).sort_values(["clip","id"])
    res.to_csv(s_res,index=False)
    obj.to_csv(s_obj,index=False)

    tot=res["detecciones"].sum()
    geo=res["georreferenciadas"].sum()
    print(f"\ntotal: {geo}/{tot} detecciones georreferenciadas ({100*geo/tot:.1f}%), "
          f"{len(obj)} objetos  (f={F_PX:.0f} px, error max={ERROR_MAX_M:.0f} m)")
    if len(obj):
        print(f"dispersion por objeto (m): p25={obj['dispersion_m'].quantile(.25):.2f}  "
              f"mediana={obj['dispersion_m'].median():.2f}  p75={obj['dispersion_m'].quantile(.75):.2f}")
    print(f"{rel(s_res)}\n{rel(s_obj)}")


if __name__=="__main__":
    main()
