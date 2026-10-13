"""
Fase 5 (paso 9.2e): validacion reproducible de la sincronizacion con datos
sinteticos de verdad conocida. No usa los videos reales: genera todo con
ffmpeg (lavfi) en una carpeta temporal, asi que se puede repetir aunque se
borren los originales.

Casos B (ubicar_clips.py), originales sinteticos a 47.95 fps:
  - recortes que empiezan en un frame exacto (con y sin cambio a 29.97 fps)
    -> B debe salir con error < 10 ms y estado OK
  - clip acelerado al doble -> debe salir REVISAR
  - clip con dos pedazos unidos -> debe salir REVISAR
Caso A (sincronizar_originales.py):
  - CSV sintetico a 10 Hz y video con subtitulos (H, D) desfasados una cantidad
    conocida -> A debe salir con error < 50 ms

Salida: results/tables/georef/validacion_sintetica.csv

Uso:
  python -m src.georef.validar_sincronizacion
"""

import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import src.georef.ubicar_clips as uc
import src.georef.sincronizar_originales as so
from src.georef.inventario_vuelos import OUT_DIR, inicio_utc, rel

FPS_ORIG="48000/1001"
FPS_ORIG_F=48000/1001
TOL_B_S=0.010
TOL_A_S=0.050


def ffmpeg(*args):
    subprocess.run(["ffmpeg","-v","error","-y",*args],check=True)


def estado_clip(r):
    """Mismo criterio que ubicar_clips.main()."""
    if r is None or np.isnan(r["B_s"]):
        return "REVISAR"
    vel=r["velocidad_medida"]
    vel_ok=np.isnan(vel) or abs(vel-1)<=uc.VEL_TOL
    ok=vel_ok and r["consistencia"]>=uc.CONSISTENCIA_MIN and r["frac_sin_parecido"]<=uc.SIN_PARECIDO_MAX
    return "OK" if ok else "REVISAR"


def casos_B(tmp):
    """Originales y clips sinteticos; devuelve filas de resultados."""
    orig_dir=tmp/"originales"
    orig_dir.mkdir()
    #dos originales de 20 s con contenido distinto y en movimiento
    o1,o2=orig_dir/"SINT_A.MP4",orig_dir/"SINT_B.MP4"
    ffmpeg("-f","lavfi","-i",f"testsrc2=s=640x360:r={FPS_ORIG}","-t","20","-c:v","libx264","-crf","18",str(o1))
    ffmpeg("-f","lavfi","-i",f"mandelbrot=s=640x360:r={FPS_ORIG}","-t","20","-c:v","libx264","-crf","18",str(o2))

    #(nombre, original, frame inicial, filtros extra, esperado)
    casos=[
        ("exacto_48fps",o1,200,"",           "OK"),
        ("exacto_a_30fps_A",o1,500,",fps=30000/1001","OK"),
        ("exacto_a_30fps_B",o2,333,",fps=30000/1001","OK"),
    ]
    clips=[]
    for nombre,o,f0,extra,esp in casos:
        ruta=tmp/f"{nombre}.mp4"
        ffmpeg("-i",str(o),"-vf",f"select=gte(n\\,{f0}),setpts=PTS-STARTPTS{extra}",
               "-frames:v","200","-c:v","libx264","-crf","28",str(ruta))
        clips.append((nombre,ruta,o,f0/FPS_ORIG_F,esp))
    #acelerado al doble
    ruta=tmp/"acelerado_2x.mp4"
    ffmpeg("-ss","3","-i",str(o1),"-t","8","-vf","setpts=0.5*PTS","-r","30000/1001",
           "-c:v","libx264","-crf","28",str(ruta))
    clips.append(("acelerado_2x",ruta,o1,np.nan,"REVISAR"))
    #dos pedazos unidos (1-4 s y 10-13 s)
    ruta=tmp/"dos_pedazos.mp4"
    ffmpeg("-i",str(o2),"-filter_complex",
           "[0:v]trim=1:4,setpts=PTS-STARTPTS[a];[0:v]trim=10:13,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1[o]",
           "-map","[o]","-r","30000/1001","-c:v","libx264","-crf","28",str(ruta))
    clips.append(("dos_pedazos",ruta,o2,np.nan,"REVISAR"))

    #mismo pipeline que ubicar_clips, con la cache en la carpeta temporal
    uc.DESC_DIR=tmp/"descriptores"
    originales=[{"nombre":o.name,"ruta":o,"desc":uc.descriptores(o),"fps":uc.fps_de(o)} for o in (o1,o2)]
    for o in originales:
        o["n"]=len(o["desc"])
    cl=[]
    for nombre,ruta,_,_,_ in clips:
        d=uc.descriptores(ruta)
        cl.append({"nombre":nombre,"ruta":ruta,"desc":d,"n":len(d),"fps":uc.fps_de(ruta)})
    uc.busqueda_gruesa(cl,originales)

    filas=[]
    for c,(nombre,_,o_esp,B_true,esp) in zip(cl,clips):
        k,B0,_=uc.votar(c)
        r=uc.ajuste_fino(c,originales[k],B0) if k is not None else None
        est=estado_clip(r)
        B=np.nan if r is None else r["B_s"]
        err=abs(B-B_true) if not np.isnan(B_true) and not np.isnan(B) else np.nan
        ok_orig=k is not None and originales[k]["nombre"]==o_esp.name
        pasa=(est==esp) and (np.isnan(B_true) or (ok_orig and err<TOL_B_S))
        filas.append({"prueba":"B","caso":nombre,"esperado":esp,"obtenido":est,
                      "valor_real_s":round(B_true,4) if not np.isnan(B_true) else None,
                      "valor_medido_s":round(B,4) if not np.isnan(B) else None,
                      "error_ms":round(err*1000,2) if not np.isnan(err) else None,
                      "pasa":bool(pasa)})
    return filas


def caso_A(tmp):
    """CSV a 10 Hz y video con subtitulos desfasados A_true segundos."""
    vuelo=tmp/"Vuelo 99"
    vuelo.mkdir()
    csv=vuelo/"FlightRecord_2026-10-01_[17-49-13]-aircraft.csv"
    t0=inicio_utc(csv)+timedelta(seconds=0.07)
    lat0,lon0=20.092959,-98.774998

    def estado(s):
        h=max(0.0,5+3*np.sin(s/7)+0.05*s)
        return h,lat0+0.00002*s*np.sin(s/20),lon0+0.00001*s

    filas=[]
    for i in range(1716):
        s=i*0.1
        t=t0+timedelta(seconds=s)
        h,lat,lon=estado(s)
        filas.append({"CUSTOM.updateTime [local]":t.strftime("%I:%M:%S.")+f"{t.microsecond//10000:02d} "+t.strftime("%p"),
                      #un hueco de GPS para comprobar que no se interpola a traves de el
                      "OSD.latitude":0 if 500<=i<510 else lat,"OSD.longitude":lon,
                      "OSD.height [ft]":round(h/so.PIE_A_M,1),"HOME.latitude":lat0,"HOME.longitude":lon0})
    pd.DataFrame(filas).to_csv(csv,index=False,encoding="utf-8-sig")

    A_true=9.37
    srt=[]
    for k in range(150):
        h,lat,lon=estado(A_true+k)
        D=so.distancia_m(lat0,lon0,lat,lon)
        srt.append(f"{k+1}\n00:{k//60:02d}:{k%60:02d},000 --> 00:{(k+1)//60:02d}:{(k+1)%60:02d},000\n"
                   f"F/1.7, SS 1063.56, ISO 100, EV 0, DZOOM 1.000, GPS ({lon:.4f}, {lat:.4f}, 29), "
                   f"D {D:.2f}m, H {h:.2f}m, H.S 0.00m/s, V.S -0.00m/s\n")
    (tmp/"subs.srt").write_text("\n".join(srt))
    v=tmp/"v.mp4"
    ffmpeg("-f","lavfi","-i",f"color=c=black:s=64x36:r={FPS_ORIG}","-t","150","-c:v","libx264",str(v))
    orig=tmp/"SINT_A_SUBS.MP4"
    ffmpeg("-i",str(v),"-i",str(tmp/"subs.srt"),"-c:v","copy","-c:s","mov_text",str(orig))

    t_ref,seg,h,dist=so.serie_csv(csv)
    sub=so.leer_subtitulos(orig)
    centro=8.0
    cand=np.arange(centro-so.VENTANA_S,centro+so.VENTANA_S+so.PASO_S/2,so.PASO_S)
    costos=np.array([so.costo(A,sub,seg,h,dist) for A in cand])
    A=float(cand[int(np.argmin(costos))])
    #A_true es relativo a t0 (1a fila); seg tambien
    err=abs(A-A_true)
    return [{"prueba":"A","caso":"subtitulos_desfasados","esperado":"OK",
             "obtenido":"OK" if err<TOL_A_S else "ERROR",
             "valor_real_s":A_true,"valor_medido_s":round(A,4),
             "error_ms":round(err*1000,2),"pasa":bool(err<TOL_A_S)}]


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp=Path(d)
        print("generando videos sinteticos y midiendo (B)...")
        filas=casos_B(tmp)
        print("caso A...")
        filas+=caso_A(tmp)
    res=pd.DataFrame(filas)
    print()
    print(res.to_string(index=False))
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT_DIR/"validacion_sintetica.csv",index=False)
    print(f"\n{'TODAS PASAN' if res['pasa'].all() else 'HAY FALLAS'}  ->  {rel(OUT_DIR/'validacion_sintetica.csv')}")


if __name__=="__main__":
    main()
