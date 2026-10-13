"""
Fase 5 (paso 9.1g): inventario de los vuelos del DJI Mini 3 y asignacion de
cada video original a su vuelo.

Por cada FlightRecord CSV (data/dji_mini3/vuelos/Vuelo N/):
  - tramos de grabacion (CAMERA.isVideo=True): inicio y fin en UTC, duracion
  - durante la grabacion: altura relativa (m), GIMBAL.pitch y OSD.yaw
  - calidad: filas sin GPS valido y saltos de tiempo
Por cada original (data/dji_mini3/vuelos/originales/DJI_*.MP4):
  - creation_time (UTC), duracion y fps con ffprobe
  - el tramo de grabacion mas cercano en el tiempo

Hora UTC: el nombre del CSV esta en hora local (UTC-6) y CUSTOM.updateTime
esta en UTC (verificado en el Vuelo 12). isVideo y creation_time solo sirven
para ASIGNAR; el desfase exacto (A) se mide en el 9.2 con la altura.

Salida: results/tables/georef/{tramos_grabacion.csv,asignacion_originales.csv}

Uso:
  python -m src.georef.inventario_vuelos
"""

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT=Path(__file__).parent.parent.parent
VUELOS_DIR=PROJECT_ROOT/"data"/"dji_mini3"/"vuelos"
OUT_DIR=PROJECT_ROOT/"results"/"tables"/"georef"

#Mexico (centro): UTC-6 todo el anio, sin horario de verano desde 2022
UTC_OFFSET_H=6
PIE_A_M=0.3048
#un original se asigna a un tramo si su creation_time cae a menos de esto
TOLERANCIA_S=30
#hueco entre filas que se reporta como salto de tiempo (el CSV va a ~10 Hz)
SALTO_S=1.0


def rel(p):
    return Path(p).relative_to(PROJECT_ROOT).as_posix()


def inicio_utc(ruta):
    """Hora de inicio del registro en UTC, a partir del nombre (hora local)."""
    m=re.search(r"(\d{4}-\d{2}-\d{2})_\[(\d{2})-(\d{2})-(\d{2})\]",ruta.name)
    if not m:
        raise ValueError(f"nombre sin fecha/hora: {ruta.name}")
    local=datetime.strptime(f"{m[1]} {m[2]}:{m[3]}:{m[4]}","%Y-%m-%d %H:%M:%S")
    return (local+timedelta(hours=UTC_OFFSET_H)).replace(tzinfo=timezone.utc)


def segundos_del_dia(serie):
    """'11:49:13.07 PM' -> segundos desde medianoche (float)."""
    s=serie.astype(str).str.strip()
    t=pd.to_datetime(s,format="%I:%M:%S.%f %p",errors="coerce")
    #filas sin fraccion de segundo
    faltan=t.isna()
    if faltan.any():
        t[faltan]=pd.to_datetime(s[faltan],format="%I:%M:%S %p",errors="coerce")
    return t.dt.hour*3600+t.dt.minute*60+t.dt.second+t.dt.microsecond/1e6


def tiempos_utc(d,t0):
    """Tiempo absoluto (UTC) de cada fila. Devuelve (serie, ajuste_inicial_s)."""
    seg=segundos_del_dia(d["CUSTOM.updateTime [local]"])
    #cruce de medianoche: si el reloj retrocede mas de 12 h, se suma un dia
    salto_dia=(seg.diff()<-43200).cumsum()*86400
    seg=seg+salto_dia
    seg0=seg.iloc[0]
    tod0=t0.hour*3600+t0.minute*60+t0.second
    #diferencia entre la 1a fila y el nombre del archivo (debe ser < 1 s)
    ajuste=((seg0-tod0+43200)%86400)-43200
    t=[t0+timedelta(seconds=float(ajuste+x-seg0)) for x in seg]
    return pd.Series(t,index=d.index),ajuste


def tramos_grabacion(d,t):
    """Bloques consecutivos con CAMERA.isVideo=True."""
    grabando=d["CAMERA.isVideo"].astype(str).str.strip().str.lower()=="true"
    bloque=(grabando!=grabando.shift()).cumsum()
    tramos=[]
    for _,idx in d[grabando].groupby(bloque[grabando]).groups.items():
        sub=d.loc[idx]
        h=pd.to_numeric(sub["OSD.height [ft]"],errors="coerce")*PIE_A_M
        p=pd.to_numeric(sub["GIMBAL.pitch"],errors="coerce")
        tramos.append({
            "inicio_utc":t[idx[0]],
            "fin_utc":t[idx[-1]],
            "duracion_s":(t[idx[-1]]-t[idx[0]]).total_seconds(),
            "altura_min_m":h.min(),
            "altura_med_m":h.median(),
            "altura_max_m":h.max(),
            "gimbal_pitch_min":p.min(),
            "gimbal_pitch_med":p.median(),
            "gimbal_pitch_max":p.max(),
        })
    return tramos


def info_video(ruta):
    """creation_time (UTC), duracion y fps de un original."""
    salida=subprocess.run(
        ["ffprobe","-v","error","-select_streams","v:0",
         "-show_entries","stream=r_frame_rate:format=duration:format_tags=creation_time",
         "-of","json",str(ruta)],
        capture_output=True,text=True,check=True)
    j=json.loads(salida.stdout)
    num,den=j["streams"][0]["r_frame_rate"].split("/")
    ct=j["format"]["tags"]["creation_time"]
    return {
        "creation_utc":datetime.fromisoformat(ct.replace("Z","+00:00")),
        "duracion_s":float(j["format"]["duration"]),
        "fps":float(num)/float(den),
    }


def main():
    filas_tramos=[]
    for csv in sorted(VUELOS_DIR.glob("Vuelo */FlightRecord_*.csv"),
                      key=lambda p:int(p.parent.name.split()[-1])):
        vuelo=int(csv.parent.name.split()[-1])
        d=pd.read_csv(csv,encoding="utf-8-sig",low_memory=False)
        t0=inicio_utc(csv)
        t,ajuste=tiempos_utc(d,t0)

        lat=pd.to_numeric(d["OSD.latitude"],errors="coerce")
        lon=pd.to_numeric(d["OSD.longitude"],errors="coerce")
        sin_gps=int((lat.isna()|lon.isna()|(lat==0)|(lon==0)).sum())
        dt=t.diff().dt.total_seconds()
        saltos=int((dt>SALTO_S).sum())
        retrocesos=int((dt<0).sum())

        tramos=tramos_grabacion(d,t)
        print(f"Vuelo {vuelo:2d}  {csv.name}")
        print(f"  filas={len(d)}  inicio_utc={t.iloc[0]:%Y-%m-%d %H:%M:%S}  "
              f"ajuste_nombre={ajuste:+.2f}s  sin_gps={sin_gps}  "
              f"saltos>{SALTO_S:g}s={saltos}  retrocesos={retrocesos}")
        for k,tr in enumerate(tramos,start=1):
            print(f"  tramo {k}: {tr['inicio_utc']:%H:%M:%S}-{tr['fin_utc']:%H:%M:%S} "
                  f"({tr['duracion_s']:.1f}s)  altura {tr['altura_min_m']:.1f}/"
                  f"{tr['altura_med_m']:.1f}/{tr['altura_max_m']:.1f} m  "
                  f"gimbal {tr['gimbal_pitch_min']:.1f}/{tr['gimbal_pitch_med']:.1f}/"
                  f"{tr['gimbal_pitch_max']:.1f}")
            filas_tramos.append({"vuelo":vuelo,"tramo":k,"csv":rel(csv),
                                 "sin_gps":sin_gps,"saltos":saltos,**tr})

    tramos=pd.DataFrame(filas_tramos)

    print("\nOriginales -> tramo mas cercano")
    filas_orig=[]
    for mp4 in sorted((VUELOS_DIR/"originales").glob("DJI_*.MP4")):
        v=info_video(mp4)
        dif=(tramos["inicio_utc"]-v["creation_utc"]).dt.total_seconds()
        i=dif.abs().idxmin()
        ok=abs(dif[i])<=TOLERANCIA_S
        fila={
            "original":mp4.name,
            "creation_utc":v["creation_utc"],
            "duracion_s":round(v["duracion_s"],3),
            "fps":round(v["fps"],3),
            "vuelo":int(tramos.at[i,"vuelo"]) if ok else None,
            "tramo":int(tramos.at[i,"tramo"]) if ok else None,
            #inicio del tramo menos creation_time (solo orientativo)
            "dif_inicio_s":round(dif[i],2) if ok else None,
            "dif_duracion_s":round(tramos.at[i,"duracion_s"]-v["duracion_s"],2) if ok else None,
        }
        filas_orig.append(fila)
        destino=f"Vuelo {fila['vuelo']} tramo {fila['tramo']}  dif_inicio={fila['dif_inicio_s']:+.1f}s  "\
                f"dif_dur={fila['dif_duracion_s']:+.1f}s" if ok else "SIN TRAMO (sin CSV de ese vuelo)"
        print(f"  {mp4.name}  {v['creation_utc']:%m-%d %H:%M:%S}Z  {v['duracion_s']:6.1f}s  "
              f"{v['fps']:.2f} fps  ->  {destino}")

    OUT_DIR.mkdir(parents=True,exist_ok=True)
    tramos.to_csv(OUT_DIR/"tramos_grabacion.csv",index=False)
    #Int64: vuelo/tramo quedan enteros aunque haya originales sin tramo
    pd.DataFrame(filas_orig).astype({"vuelo":"Int64","tramo":"Int64"}).to_csv(
        OUT_DIR/"asignacion_originales.csv",index=False)
    print(f"\n{rel(OUT_DIR/'tramos_grabacion.csv')}\n{rel(OUT_DIR/'asignacion_originales.csv')}")


if __name__=="__main__":
    main()
