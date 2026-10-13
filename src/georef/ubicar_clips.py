"""
Fase 5 (paso 9.2b): desfase B de cada clip dentro de su video original.

Los clips (data/dji_mini3/videoXX.mp4) son recortes re-codificados sin
telemetria. Se ubican comparando sus frames contra los de los originales:

    tiempo_original = B + velocidad*tiempo_clip
    tiempo_UTC      = A + tiempo_original        (A del paso 9.2a)

1) Miniaturas 48x27 en gris de todos los frames (cache en data/.../descriptores/).
2) Busqueda gruesa: 1 frame del clip por segundo contra TODOS los frames de los
   18 originales (correlacion); el par (original, B) mas votado gana. No se usa
   la carpeta de la USB como prueba, solo se compara al final.
3) Ajuste fino: cada frame del clip contra el original elegido (+-0.5 s), con
   interpolacion sub-frame; recta robusta t_original=B+velocidad*t_clip.
   velocidad~1 -> se conservo la velocidad; inliers bajos -> clip con cortes.
   Frames "estaticos" (dron quieto: el frame se parece igual a todos sus vecinos)
   no aportan tiempo y no cuentan como error. Si el dron esta quieto, un error
   en B casi no mueve la posicion calculada.
   B final = B con velocidad 1 (verificada en los clips con movimiento).
4) Evidencia: frames del clip y del original al inicio, mitad y final.

Salida: results/tables/georef/ubicacion_clips.csv
        results/figures/georef/sync_B/<clip>.jpg

Uso:
  python -m src.georef.ubicar_clips
  python -m src.georef.ubicar_clips --solo video29 --sin-figuras
"""

import argparse
import re
import subprocess
from collections import Counter
from datetime import timedelta

import cv2
import numpy as np
import pandas as pd

from src.georef.inventario_vuelos import PROJECT_ROOT, VUELOS_DIR, OUT_DIR, rel

CLIPS_DIR=PROJECT_ROOT/"data"/"dji_mini3"
DESC_DIR=VUELOS_DIR/"descriptores"
FIG_DIR=PROJECT_ROOT/"results"/"figures"/"georef"/"sync_B"

#miniatura en gris: reconoce la escena y es barata de comparar
W,H=48,27
#busqueda gruesa: una muestra del clip por segundo; votos agrupados cada 0.25 s
MUESTREO_S=1.0
BIN_S=0.25
#ajuste fino: ventana alrededor de la prediccion gruesa
VENTANA_FINA_S=0.5
#correlacion minima para aceptar que dos frames son la misma escena
CORR_MIN=0.90
#pico de correlacion vs mediana de la ventana: si es menor, escena estatica (sin info de tiempo)
NITIDEZ_MIN=0.02
#tolerancias para declarar el clip OK
VEL_TOL=0.002
#inliers / frames con informacion de tiempo
CONSISTENCIA_MIN=0.90
#frames del clip sin ningun parecido cerca de la prediccion (cortes, zoom, otra fuente)
SIN_PARECIDO_MAX=0.05
#la velocidad solo se mide si los frames con movimiento abarcan al menos esto
SPAN_VEL_S=10.0
#menos frames con movimiento que esto -> nota informativa
MOVIMIENTO_NOTA=0.10


def fps_de(ruta):
    s=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
                      "stream=r_frame_rate","-of","csv=p=0",str(ruta)],
                     capture_output=True,text=True,check=True).stdout.strip()
    n,d=s.split("/")
    return float(n)/float(d)


def descriptores(ruta):
    """Miniaturas WxH en gris de todos los frames (uint8). Se guardan en cache."""
    cache=DESC_DIR/f"{ruta.stem}.npy"
    if cache.exists():
        return np.load(cache)
    crudo=subprocess.run(["ffmpeg","-v","error","-i",str(ruta),"-map","0:v:0",
                          "-vf",f"scale={W}:{H}:flags=area,format=gray",
                          "-fps_mode","passthrough","-f","rawvideo","-"],
                         capture_output=True,check=True).stdout
    m=np.frombuffer(crudo,np.uint8).reshape(-1,W*H)
    DESC_DIR.mkdir(parents=True,exist_ok=True)
    np.save(cache,m)
    return m


def normalizar(m):
    """Media 0 y norma 1 por frame: el producto punto es la correlacion de Pearson."""
    x=m.astype(np.float32)
    x-=x.mean(1,keepdims=True)
    n=np.linalg.norm(x,axis=1,keepdims=True)
    n[n<1e-6]=1.0
    return x/n


def busqueda_gruesa(clips,originales):
    """Para cada muestra de cada clip: mejor (original, t_original, corr) entre todos los originales."""
    muestras=[]
    for c in clips:
        paso=max(1,int(round(c["fps"]*MUESTREO_S)))
        idx=np.arange(0,c["n"],paso)
        c["idx_muestras"]=idx
        muestras.append(normalizar(c["desc"][idx]))
    Q=np.vstack(muestras)
    mejor_corr=np.full(len(Q),-1.0,dtype=np.float32)
    mejor_orig=np.full(len(Q),-1)
    mejor_t=np.zeros(len(Q))
    for k,o in enumerate(originales):
        on=normalizar(o["desc"])
        C=Q@on.T
        j=C.argmax(1)
        cm=C[np.arange(len(Q)),j]
        upd=cm>mejor_corr
        mejor_corr[upd]=cm[upd]
        mejor_orig[upd]=k
        mejor_t[upd]=j[upd]/o["fps"]
        print(f"  gruesa: {o['nombre']} listo")
    #repartir los resultados de vuelta a cada clip
    ini=0
    for c in clips:
        n=len(c["idx_muestras"])
        c["g_corr"]=mejor_corr[ini:ini+n]
        c["g_orig"]=mejor_orig[ini:ini+n]
        c["g_t"]=mejor_t[ini:ini+n]
        ini+=n


def votar(c):
    """(original, B) mas votado entre las muestras del clip."""
    tq=c["idx_muestras"]/c["fps"]
    B=c["g_t"]-tq
    ok=np.where(c["g_corr"]>=CORR_MIN)[0]
    if len(ok)==0:
        return None,None,0.0
    votos=Counter((int(c["g_orig"][k]),int(round(B[k]/BIN_S))) for k in ok)
    (o,b),_=votos.most_common(1)[0]
    inl=[k for k in ok if c["g_orig"][k]==o and abs(B[k]/BIN_S-b)<=1]
    return o,float(np.median(B[inl])),len(inl)/len(tq)


def ajuste_fino(c,o,B0):
    """Cada frame del clip contra el original cerca de B0; recta robusta t_o=B+v*t_c."""
    cn=normalizar(c["desc"])
    on=normalizar(o["desc"])
    fo=o["fps"]
    w=int(np.ceil(VENTANA_FINA_S*fo))
    tc_l,to_l=[],[]
    n_estaticos=0
    n_sin_parecido=0
    for i in range(c["n"]):
        tc=i/c["fps"]
        jc=int(round((B0+tc)*fo))
        lo,hi=max(0,jc-w),min(len(on),jc+w+1)
        if hi-lo<3:
            n_sin_parecido+=1
            continue
        corr=on[lo:hi]@cn[i]
        j=int(corr.argmax())
        if corr[j]<CORR_MIN:
            n_sin_parecido+=1
            continue
        if corr[j]-np.median(corr)<NITIDEZ_MIN:
            n_estaticos+=1
            continue
        #parabola por los 3 puntos del pico: posicion sub-frame
        dj=0.0
        if 0<j<len(corr)-1:
            den=corr[j-1]-2*corr[j]+corr[j+1]
            if den<0:
                dj=0.5*(corr[j-1]-corr[j+1])/den
        tc_l.append(tc)
        to_l.append((lo+j+dj)/fo)
    x,y=np.array(tc_l),np.array(to_l)
    if len(x)<10:
        return None
    #recta robusta: se descartan pares a mas de 1.5 frames del original
    keep=np.ones(len(x),bool)
    for _ in range(5):
        v,B=np.polyfit(x[keep],y[keep],1)
        nuevo=np.abs(y-(B+v*x))<=1.5/fo
        if nuevo.sum()<10 or (nuevo==keep).all():
            break
        keep=nuevo
    #inliers definitivos respecto a la recta final
    keep=np.abs(y-(B+v*x))<=1.5/fo
    if keep.sum()<10:
        #los frames con movimiento no caen sobre una recta: cortes o velocidad distinta
        return {"B_s":np.nan,"consistencia":float(keep.sum()/len(x)),
                "frac_movimiento":float(len(x)/c["n"]),"frac_estaticos":float(n_estaticos/c["n"]),
                "frac_sin_parecido":float(n_sin_parecido/c["n"])}
    res=y[keep]-(B+v*x[keep])
    #con la velocidad fija en 1: B como mediana
    B1=float(np.median(y[keep]-x[keep]))
    span=float(x[keep].max()-x[keep].min())
    return {
        "B_s":B1,
        "B_ajuste_s":float(B),
        #solo se reporta si los frames con movimiento abarcan lo suficiente para medirla
        "velocidad_medida":float(v) if span>=SPAN_VEL_S else np.nan,
        "rms_res_ms":float(np.sqrt(np.mean(res**2))*1000),
        "frac_movimiento":float(len(x)/c["n"]),
        "consistencia":float(keep.sum()/len(x)),
        "frac_estaticos":float(n_estaticos/c["n"]),
        "frac_sin_parecido":float(n_sin_parecido/c["n"]),
        "span_movimiento_s":span,
    }


def frame_en(ruta,t,ancho=480,alto=270):
    """Frame en el segundo t (seek exacto de ffmpeg), redimensionado."""
    png=subprocess.run(["ffmpeg","-v","error","-ss",f"{max(t,0):.3f}","-i",str(ruta),
                        "-frames:v","1","-vf",f"scale={ancho}:{alto}","-f","image2pipe",
                        "-vcodec","png","-"],capture_output=True,check=True).stdout
    img=cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR)
    return img if img is not None else np.zeros((alto,ancho,3),np.uint8)


def figura(c,o,r,ruta_png):
    filas=[]
    dur=c["n"]/c["fps"]
    for etiqueta,f in (("inicio",0.02),("mitad",0.5),("final",0.98)):
        tc=f*dur
        to=r["B_s"]+tc
        a=frame_en(c["ruta"],tc)
        b=frame_en(o["ruta"],to)
        for img,txt in ((a,f"{c['nombre']}  t={tc:.2f}s  ({etiqueta})"),
                        (b,f"{o['nombre']}  t={to:.2f}s")):
            cv2.rectangle(img,(0,0),(img.shape[1],22),(0,0,0),-1)
            cv2.putText(img,txt,(6,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        filas.append(cv2.hconcat([a,b]))
    ruta_png.parent.mkdir(parents=True,exist_ok=True)
    #jpg: son fotos aereas; en png pesan ~1.5 MB cada una
    cv2.imwrite(str(ruta_png),cv2.vconcat(filas),[cv2.IMWRITE_JPEG_QUALITY,90])


def vuelo_por_carpeta():
    """clips_por_vuelo.txt -> {numero_clip: vuelo} (solo como referencia)."""
    ref={}
    ruta=VUELOS_DIR/"clips_por_vuelo.txt"
    if not ruta.exists():
        return ref
    for linea in ruta.read_text().splitlines():
        m=re.match(r"Vuelo\s+(\d+):(.*)",linea)
        if m:
            for n in re.findall(r"video(\d+)",m[2]):
                ref[int(n)]=int(m[1])
    return ref


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--solo",default=None,help="procesar solo este clip (p. ej. video29)")
    ap.add_argument("--sin-figuras",action="store_true")
    args=ap.parse_args()

    rutas_clips=sorted(CLIPS_DIR.glob("video*.mp4"))
    if args.solo:
        rutas_clips=[p for p in rutas_clips if p.stem==args.solo]
    rutas_orig=sorted((VUELOS_DIR/"originales").glob("DJI_*.MP4"))

    print("miniaturas (la primera vez decodifica todo; despues usa la cache)")
    clips,originales=[],[]
    for ruta in rutas_clips:
        d=descriptores(ruta)
        clips.append({"nombre":ruta.stem,"ruta":ruta,"desc":d,"n":len(d),"fps":fps_de(ruta)})
        print(f"  {ruta.name}: {len(d)} frames")
    for ruta in rutas_orig:
        d=descriptores(ruta)
        originales.append({"nombre":ruta.name,"ruta":ruta,"desc":d,"n":len(d),"fps":fps_de(ruta)})
        print(f"  {ruta.name}: {len(d)} frames")

    print("busqueda gruesa")
    busqueda_gruesa(clips,originales)

    asig=pd.read_csv(OUT_DIR/"asignacion_originales.csv")
    vuelo_de=dict(zip(asig["original"],asig["vuelo"]))
    A=pd.read_csv(OUT_DIR/"desfase_originales.csv")
    A_de=dict(zip(A["original"],pd.to_datetime(A["A_utc"])))
    carpeta=vuelo_por_carpeta()

    print("ajuste fino")
    filas=[]
    for c in clips:
        k,B0,frac_votos=votar(c)
        num=int(re.search(r"(\d+)",c["nombre"])[1])
        fila={"clip":c["nombre"],"frames":c["n"],"fps_clip":round(c["fps"],3),
              "duracion_s":round(c["n"]/c["fps"],3),"vuelo_carpeta":carpeta.get(num)}
        if k is None:
            fila.update({"estado":"SIN COINCIDENCIA"})
            filas.append(fila)
            print(f"  {c['nombre']}: SIN COINCIDENCIA")
            continue
        o=originales[k]
        r=ajuste_fino(c,o,B0)
        vuelo=vuelo_de.get(o["nombre"])
        fila.update({"original":o["nombre"],"fps_original":round(o["fps"],3),
                     "vuelo":None if pd.isna(vuelo) else int(vuelo),
                     "frac_votos":round(frac_votos,3)})
        if r is None or np.isnan(r["B_s"]):
            if r is not None:
                fila.update({k2:round(v,4) for k2,v in r.items() if not np.isnan(v)})
            fila["estado"]="REVISAR (sin ajuste fino)"
            filas.append(fila)
            print(f"  {c['nombre']}: {o['nombre']}  REVISAR (sin ajuste fino)")
            continue
        fila.update({k2:round(v,4) for k2,v in r.items()})
        vel_ok=np.isnan(r["velocidad_medida"]) or abs(r["velocidad_medida"]-1)<=VEL_TOL
        ok=vel_ok and r["consistencia"]>=CONSISTENCIA_MIN and r["frac_sin_parecido"]<=SIN_PARECIDO_MAX
        notas=[]
        if r["frac_movimiento"]<MOVIMIENTO_NOTA:
            notas.append("pocos frames con movimiento (dron casi quieto)")
        if o["nombre"] in A_de:
            ini=A_de[o["nombre"]]+timedelta(seconds=r["B_s"])
            fin=ini+timedelta(seconds=c["n"]/c["fps"])
            fila["inicio_utc"]=ini.isoformat(timespec="milliseconds")
            fila["fin_utc"]=fin.isoformat(timespec="milliseconds")
        else:
            ok=False
            notas.append("original sin CSV (sin A)")
        fila["coincide_carpeta"]=fila["vuelo"]==fila["vuelo_carpeta"] if fila["vuelo"] else False
        fila["estado"]="OK" if ok else "REVISAR"
        fila["nota"]="; ".join(notas)
        filas.append(fila)
        vel_txt="n/a" if np.isnan(r["velocidad_medida"]) else f"{r['velocidad_medida']:.5f}"
        print(f"  {c['nombre']}: {o['nombre']} (vuelo {fila['vuelo']}, carpeta {fila['vuelo_carpeta']})"
              f"  B={r['B_s']:.3f}s  vel={vel_txt}  res={r['rms_res_ms']:.1f}ms"
              f"  mov={r['frac_movimiento']:.2f}  consist={r['consistencia']:.2f}"
              f"  sin_parecido={r['frac_sin_parecido']:.2f}  {fila['estado']}"
              f"{'  ('+fila['nota']+')' if fila['nota'] else ''}")
        if not args.sin_figuras:
            figura(c,o,r,FIG_DIR/f"{c['nombre']}.jpg")

    res=pd.DataFrame(filas)
    #solapes: dos clips que cubren el mismo tramo de un original
    solapes=[]
    if "B_s" in res:
        v=res.dropna(subset=["B_s"])
        for o,g in v.groupby("original"):
            g=g.assign(fin_o=g["B_s"]+g["duracion_s"]).sort_values("B_s")
            filas_g=list(g.itertuples())
            for i in range(len(filas_g)):
                for j in range(i+1,len(filas_g)):
                    a,b=filas_g[i],filas_g[j]
                    s=min(a.fin_o,b.fin_o)-max(a.B_s,b.B_s)
                    if s>1.0:
                        solapes.append(f"{a.clip} y {b.clip} comparten {s:.1f}s de {o}")
    res["solapes"]=res["clip"].map(lambda n:"; ".join(x for x in solapes if n+" " in x+" "))
    salida=OUT_DIR/"ubicacion_clips.csv"
    if args.solo and salida.exists():
        prev=pd.read_csv(salida)
        res=pd.concat([prev[prev["clip"]!=args.solo],res]).sort_values("clip")
    res.to_csv(salida,index=False)

    print("\nsolapes:" if solapes else "\nsin solapes entre clips")
    for s in solapes:
        print(f"  {s}")
    print(f"\n{rel(salida)}")
    if not args.sin_figuras:
        print(f"{rel(FIG_DIR)}/")


if __name__=="__main__":
    main()
