"""
Fase 5 (paso 9.2a): desfase A de cada video original dentro de su registro de vuelo.

A = instante UTC del primer frame del original. Se mide comparando la
telemetria que el propio video trae en sus subtitulos (1 dato por segundo:
altura H y distancia al home D) contra el CSV del vuelo (~10 Hz), y buscando
el corrimiento que mejor las empata. creation_time e isVideo solo acotan la
busqueda (difieren unos segundos entre si).

    tiempo_vuelo = A + tiempo_original

Entrada: results/tables/georef/{asignacion_originales,tramos_grabacion}.csv (9.1g)
Salida : results/tables/georef/desfase_originales.csv
         results/figures/georef/sync_A/<original>.png  (H y D empatadas)

Uso:
  python -m src.georef.sincronizar_originales
  python -m src.georef.sincronizar_originales --solo DJI_0014.MP4
"""

import argparse
import re
import subprocess
from datetime import timedelta

import numpy as np
import pandas as pd

from src.georef.inventario_vuelos import (PROJECT_ROOT, VUELOS_DIR, OUT_DIR, PIE_A_M,
                                          inicio_utc, tiempos_utc, rel)

FIG_DIR=PROJECT_ROOT/"results"/"figures"/"georef"/"sync_A"
#ventana de busqueda alrededor del inicio de isVideo y paso de la busqueda
VENTANA_S=10.0
PASO_S=0.02
#para la nitidez: el mejor corrimiento "rival" debe estar a mas de esto del optimo
SEPARACION_S=1.0
RADIO_TIERRA_M=6371008.8


def leer_subtitulos(ruta):
    """Telemetria de los subtitulos del original: t (s), lon, lat, D (m), H (m)."""
    srt=subprocess.run(["ffmpeg","-v","error","-i",str(ruta),"-map","0:s:0","-f","srt","-"],
                       capture_output=True,text=True,check=True).stdout.replace("\r","")
    filas=[]
    for bloque in re.split(r"\n\s*\n",srt.strip()):
        m=re.search(r"(\d+):(\d+):(\d+),(\d+)\s*-->",bloque)
        if not m:
            continue
        g=re.search(r"GPS\s*\(\s*([-\d.]+),\s*([-\d.]+)",bloque)
        #(?<![\w.]) evita confundir H con H.S y D con otra etiqueta
        d=re.search(r"(?<![\w.])D\s+([-\d.]+)m",bloque)
        h=re.search(r"(?<![\w.])H\s+([-\d.]+)m",bloque)
        filas.append({
            "t":int(m[1])*3600+int(m[2])*60+int(m[3])+int(m[4])/1000,
            "lon":float(g[1]) if g else np.nan,
            "lat":float(g[2]) if g else np.nan,
            "D":float(d[1]) if d else np.nan,
            "H":float(h[1]) if h else np.nan,
        })
    return pd.DataFrame(filas)


def distancia_m(lat1,lon1,lat2,lon2):
    """Haversine en metros (acepta arreglos)."""
    p1,p2=np.radians(lat1),np.radians(lat2)
    dp=p2-p1
    dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*RADIO_TIERRA_M*np.arcsin(np.sqrt(a))


def serie_csv(ruta_csv):
    """Del CSV: t_ref (UTC de la 1a fila), segundos desde t_ref, altura (m) y distancia al home (m)."""
    d=pd.read_csv(ruta_csv,encoding="utf-8-sig",low_memory=False)
    t,_=tiempos_utc(d,inicio_utc(ruta_csv))
    t_ref=t.iloc[0]
    seg=np.array([(x-t_ref).total_seconds() for x in t])
    h=pd.to_numeric(d["OSD.height [ft]"],errors="coerce").to_numpy()*PIE_A_M
    lat=pd.to_numeric(d["OSD.latitude"],errors="coerce").to_numpy()
    lon=pd.to_numeric(d["OSD.longitude"],errors="coerce").to_numpy()
    hlat=pd.to_numeric(d["HOME.latitude"],errors="coerce").to_numpy()
    hlon=pd.to_numeric(d["HOME.longitude"],errors="coerce").to_numpy()
    #filas sin GPS o sin home valido no entran en D (no se interpola a traves de ellas)
    ok=~(np.isnan(lat)|np.isnan(lon)|np.isnan(hlat)|np.isnan(hlon)|(lat==0)|(lon==0)|(hlat==0)|(hlon==0))
    dist=np.full(len(d),np.nan)
    dist[ok]=distancia_m(lat[ok],lon[ok],hlat[ok],hlon[ok])
    return t_ref,seg,h,dist


def interp_con_huecos(x,xs,ys):
    """np.interp que devuelve NaN fuera de rango y en puntos vecinos a un NaN."""
    ok=~np.isnan(ys)
    y=np.interp(x,xs[ok],ys[ok],left=np.nan,right=np.nan)
    #si el vecino original es NaN no se confia en la interpolacion
    i=np.clip(np.searchsorted(xs,x),1,len(xs)-1)
    malo=np.isnan(ys[i])|np.isnan(ys[i-1])
    y[malo]=np.nan
    return y


def errores(A,sub,seg,h,dist):
    """Diferencias video-CSV de H y D si el primer frame cae en el segundo A del CSV."""
    x=A+sub["t"].to_numpy()
    eh=sub["H"].to_numpy()-interp_con_huecos(x,seg,h)
    ed=sub["D"].to_numpy()-interp_con_huecos(x,seg,dist)
    return eh,ed


def costo(A,sub,seg,h,dist):
    eh,ed=errores(A,sub,seg,h,dist)
    e=np.concatenate([eh,ed])
    e=e[~np.isnan(e)]
    #se exige cubrir al menos el 80% de los puntos para que el costo sea comparable
    if len(e)<0.8*2*len(sub):
        return np.inf
    return float(np.sqrt(np.mean(e**2)))


def graficar(ruta_png,nombre,A_utc,sub,seg,h,dist,A,rmse_h,rmse_d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t=sub["t"].to_numpy()
    malla=np.arange(t.min(),t.max()+0.1,0.1)
    fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
    ax[0].plot(malla,interp_con_huecos(A+malla,seg,h),lw=1,label="CSV (10 Hz)")
    ax[0].plot(t,sub["H"],"o",ms=3,label="subtitulos del video (1 Hz)")
    ax[0].set_ylabel("altura relativa H (m)")
    ax[0].set_title(f"{nombre}: primer frame = {A_utc:%Y-%m-%d %H:%M:%S.%f}"[:-4]+
                    f" UTC   RMSE H={rmse_h:.2f} m, D={rmse_d:.2f} m")
    ax[0].legend()
    ax[1].plot(malla,interp_con_huecos(A+malla,seg,dist),lw=1)
    ax[1].plot(t,sub["D"],"o",ms=3)
    ax[1].set_ylabel("distancia al home D (m)")
    ax[1].set_xlabel("tiempo del video (s)")
    fig.tight_layout()
    ruta_png.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(ruta_png,dpi=110)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--solo",default=None,help="procesar solo este original (p. ej. DJI_0014.MP4)")
    args=ap.parse_args()

    asig=pd.read_csv(OUT_DIR/"asignacion_originales.csv",parse_dates=["creation_utc"])
    tramos=pd.read_csv(OUT_DIR/"tramos_grabacion.csv",parse_dates=["inicio_utc","fin_utc"])
    asig=asig.dropna(subset=["vuelo"])
    if args.solo:
        asig=asig[asig["original"]==args.solo]

    filas=[]
    for _,o in asig.iterrows():
        tr=tramos[(tramos["vuelo"]==o["vuelo"])&(tramos["tramo"]==o["tramo"])].iloc[0]
        ruta_csv=PROJECT_ROOT/tr["csv"]
        t_ref,seg,h,dist=serie_csv(ruta_csv)
        sub=leer_subtitulos(VUELOS_DIR/"originales"/o["original"])

        #busqueda: A en segundos desde t_ref, alrededor del inicio de isVideo
        centro=(tr["inicio_utc"]-t_ref).total_seconds()
        candidatos=np.arange(centro-VENTANA_S,centro+VENTANA_S+PASO_S/2,PASO_S)
        costos=np.array([costo(A,sub,seg,h,dist) for A in candidatos])
        k=int(np.argmin(costos))
        A=float(candidatos[k])
        lejos=np.abs(candidatos-A)>SEPARACION_S
        rival=float(np.min(costos[lejos])) if lejos.any() else np.nan

        eh,ed=errores(A,sub,seg,h,dist)
        A_utc=t_ref+timedelta(seconds=A)
        fila={
            "original":o["original"],
            "vuelo":int(o["vuelo"]),
            "tramo":int(o["tramo"]),
            "n_subtitulos":len(sub),
            "A_utc":A_utc.isoformat(timespec="milliseconds"),
            "A_menos_isvideo_s":round((A_utc-tr["inicio_utc"]).total_seconds(),2),
            "A_menos_creation_s":round((A_utc-o["creation_utc"]).total_seconds(),2),
            "rmse_H_m":round(float(np.sqrt(np.nanmean(eh**2))),3),
            "rmse_D_m":round(float(np.sqrt(np.nanmean(ed**2))),3),
            "sesgo_H_m":round(float(np.nanmean(eh)),3),
            #costo del mejor corrimiento a >1 s / costo del optimo: >2 = optimo bien definido
            "nitidez":round(rival/costos[k],2) if costos[k]>0 else np.inf,
            "en_borde":bool(k==0 or k==len(candidatos)-1),
        }
        filas.append(fila)
        print(f"{fila['original']}  vuelo {fila['vuelo']:2d}.{fila['tramo']}  A={A_utc:%H:%M:%S.%f}"[:-3]+
              f"  A-isVideo={fila['A_menos_isvideo_s']:+.2f}s  A-creation={fila['A_menos_creation_s']:+.2f}s"
              f"  RMSE H={fila['rmse_H_m']:.2f} D={fila['rmse_D_m']:.2f} m  nitidez={fila['nitidez']}"
              f"{'  EN BORDE' if fila['en_borde'] else ''}")
        graficar(FIG_DIR/f"{o['original'].rsplit('.',1)[0]}.png",o["original"],A_utc,
                 sub,seg,h,dist,A,fila["rmse_H_m"],fila["rmse_D_m"])

    salida=OUT_DIR/"desfase_originales.csv"
    if args.solo and salida.exists():
        #con --solo se actualiza solo esa fila
        prev=pd.read_csv(salida)
        prev=prev[~prev["original"].isin([f["original"] for f in filas])]
        res=pd.concat([prev,pd.DataFrame(filas)]).sort_values("original")
    else:
        res=pd.DataFrame(filas)
    res.to_csv(salida,index=False)
    print(f"\n{rel(salida)}\n{rel(FIG_DIR)}/")


if __name__=="__main__":
    main()
