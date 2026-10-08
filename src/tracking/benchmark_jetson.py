"""
Fase 5 · Paso 6 — Benchmark de la config A (YOLOv8n + ByteTrack) en la Jetson Orin Nano.

Modos (--modo):
  reposo  Linea base: solo tegrastats durante N segundos, sin cargar torch ni el modelo.
  ram     Comparable con la Fase 3 (benchmark_fps.py): frames precargados en RAM (sin lectura
          de disco), mismas 2 secuencias de val (1344x756 y 3840x2160), detector y tracker en el
          mismo bucle. Acepta varios pesos (.pt y .engine) para aislar el efecto de TensorRT.
  video   Sistema real: .mp4 -> decodificacion (OpenCV/FFmpeg, CPU) -> detector -> ByteTrack,
          sobre una carpeta de videos (por defecto data/videos_prueba: los 17 de test-dev).

En todos los modos corre tegrastats en paralelo (si existe): potencia (VDD_IN y rieles),
temperaturas, RAM del sistema, uso de CPU/GPU y frecuencia de la GPU (sysfs).

No reemplaza a benchmark_fps.py (Fase 3) ni escribe en results/tables/eficiencia/.
Salidas en results/tables/eficiencia_jetson/:
    <modo>__<tag>.json             resumen
    <modo>__<tag>_tegrastats.csv   muestras crudas de tegrastats (t en s desde el inicio)
    video__<tag>_frames.csv        tiempos por frame (solo modo video)

Uso (desde la raiz del proyecto, en la Jetson):
    python -m src.tracking.benchmark_jetson --modo reposo --segundos 60
    python -m src.tracking.benchmark_jetson --modo ram --weights \\
        runs/yolov8n/yolov8n_mosaic10_20260606_0315/weights/best.pt \\
        runs/yolov8n/yolov8n_mosaic10_20260606_0315/weights/best_736x1280.engine
    python -m src.tracking.benchmark_jetson --modo video
"""

import argparse
import csv
import gc
import json
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

PROJECT_ROOT=Path(__file__).parent.parent.parent
OUT_DIR=PROJECT_ROOT/"results"/"tables"/"eficiencia_jetson"
VAL_DIR=PROJECT_ROOT/"data"/"visdrone_mot"/"motchallenge"/"val"
VIDEOS_DEF="data/videos_prueba"
PESOS_DEF="runs/yolov8n/yolov8n_mosaic10_20260606_0315/weights/best_736x1280.engine"

#mismas secuencias que benchmark_fps.py (Fase 3), para comparar con la laptop
SEQS_FASE3=["uav0000086_00000_v","uav0000268_05773_v"]


# ---------------------------------------------------------------- estadistica (sin numpy)

def percentil(orden,q):
    k=(len(orden)-1)*q/100
    f=int(k)
    c=min(f+1,len(orden)-1)
    return orden[f]+(orden[c]-orden[f])*(k-f)


def resumen(vals):
    vals=[float(v) for v in vals]
    if not vals:
        return None
    o=sorted(vals)
    return {"media":round(sum(o)/len(o),2),"p50":round(percentil(o,50),2),
            "p95":round(percentil(o,95),2),"min":round(o[0],2),"max":round(o[-1],2)}


def media(vals):
    return sum(vals)/len(vals) if vals else 0.0


# ---------------------------------------------------------------- tegrastats

RE_RAM=re.compile(r"RAM (\d+)/(\d+)MB")
RE_SWAP=re.compile(r"SWAP (\d+)/(\d+)MB")
RE_CPU=re.compile(r"CPU \[([^\]]*)\]")
RE_GPU=re.compile(r"GR3D_FREQ (\d+)%")
RE_TEMP=re.compile(r"(\w+)@(-?[\d.]+)C")
RE_RIEL=re.compile(r"(VDD_\w+) (\d+)mW/(\d+)mW")

#campos que se resumen (media/p50/p95/min/max) en el JSON
CAMPOS_TEG=["VDD_IN_mw","VDD_CPU_GPU_CV_mw","VDD_SOC_mw","ram_mb","swap_mb",
            "gpu_pct","cpu_pct","temp_cpu","temp_gpu","temp_tj","gpu_mhz"]


def parsear_tegrastats(linea):
    """Una linea de tegrastats -> dict. Tolera los campos que solo aparecen con sudo."""
    d={}
    m=RE_RAM.search(linea)
    if m:
        d["ram_mb"]=int(m.group(1))
        d["ram_total_mb"]=int(m.group(2))
    m=RE_SWAP.search(linea)
    if m:
        d["swap_mb"]=int(m.group(1))
    m=RE_CPU.search(linea)
    if m:
        usos=[int(c.split("%")[0]) for c in m.group(1).split(",") if "%" in c]
        if usos:
            d["cpu_pct"]=round(media(usos),1)
            d["cpu_pct_max_nucleo"]=max(usos)
    m=RE_GPU.search(linea)
    if m:
        d["gpu_pct"]=int(m.group(1))
    #temperaturas: cpu@47.3C gpu@48.5C tj@48.5C soc0@... (el % antes de @ excluye CPU/EMC/GR3D)
    for nombre,valor in RE_TEMP.findall(linea):
        d[f"temp_{nombre}"]=float(valor)
    #rieles de potencia: valor instantaneo (el segundo numero es el promedio desde el arranque)
    for nombre,inst,_ in RE_RIEL.findall(linea):
        d[f"{nombre}_mw"]=int(inst)
    return d


def ruta_freq_gpu():
    """cur_freq de la GPU en sysfs (Hz); sirve para detectar si la GPU baja de frecuencia."""
    for p in sorted(Path("/sys/class/devfreq").glob("*gpu*/cur_freq")):
        try:
            int(p.read_text())
            return p
        except (OSError,ValueError):
            continue
    return None


class Tegrastats:
    """Corre tegrastats en segundo plano y guarda (t, muestra) con el reloj de perf_counter."""

    def __init__(self,intervalo_ms=500):
        self.intervalo_ms=intervalo_ms
        self.muestras=[]
        self.proc=None
        self.freq=ruta_freq_gpu()

    def __enter__(self):
        if shutil.which("tegrastats") is None:
            print("AVISO: tegrastats no existe en este equipo; no se mide potencia ni temperatura")
            return self
        self.proc=subprocess.Popen(["tegrastats","--interval",str(self.intervalo_ms)],
                                   stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                   text=True,bufsize=1)
        self.hilo=threading.Thread(target=self._leer,daemon=True)
        self.hilo.start()
        return self

    def _leer(self):
        for linea in self.proc.stdout:
            d=parsear_tegrastats(linea)
            if self.freq is not None:
                try:
                    d["gpu_mhz"]=round(int(self.freq.read_text())/1e6)
                except (OSError,ValueError):
                    pass
            self.muestras.append((time.perf_counter(),d))

    def __exit__(self,*exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def ventana(self,t0,t1):
        return [d for t,d in self.muestras if t0<=t<=t1]

    def guardar_csv(self,ruta,t_ref):
        if not self.muestras:
            return
        claves=sorted({k for _,d in self.muestras for k in d})
        with open(ruta,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["t_s"]+claves)
            for t,d in self.muestras:
                w.writerow([round(t-t_ref,3)]+[d.get(k,"") for k in claves])


def resumir_teg(muestras):
    if not muestras:
        return None
    out={"n_muestras":len(muestras)}
    for k in CAMPOS_TEG:
        vals=[d[k] for d in muestras if k in d]
        if vals:
            out[k]=resumen(vals)
    return out


def cargar_reposo():
    """Linea base del modo reposo (si ya se corrio) para calcular deltas de potencia y RAM."""
    f=OUT_DIR/"reposo__reposo.json"
    if f.exists():
        return json.loads(f.read_text()).get("tegrastats")
    return None


def energia(teg,reposo,segundos,frames):
    """Potencia media, energia por frame y deltas contra la linea base."""
    if not teg or "VDD_IN_mw" not in teg or frames==0:
        return None
    w=teg["VDD_IN_mw"]["media"]/1000
    out={"potencia_media_w":round(w,2),"potencia_max_w":round(teg["VDD_IN_mw"]["max"]/1000,2),
         "energia_j_por_frame":round(w*segundos/frames,4)}
    if reposo and "VDD_IN_mw" in reposo:
        w0=reposo["VDD_IN_mw"]["media"]/1000
        out["potencia_reposo_w"]=round(w0,2)
        out["potencia_dinamica_w"]=round(w-w0,2)
        out["energia_dinamica_j_por_frame"]=round((w-w0)*segundos/frames,4)
    if reposo and "ram_mb" in reposo and "ram_mb" in teg:
        out["ram_delta_media_mb"]=round(teg["ram_mb"]["media"]-reposo["ram_mb"]["media"],1)
        out["ram_delta_max_mb"]=round(teg["ram_mb"]["max"]-reposo["ram_mb"]["media"],1)
    return out


# ---------------------------------------------------------------- entorno

def modo_energia():
    """Nombre del modo de nvpmodel leyendo su estado y su .conf (sin sudo)."""
    try:
        estado=Path("/var/lib/nvpmodel/status").read_text()
        idm=int(re.search(r"pmode:(\d+)",estado).group(1))
        nombres=dict(re.findall(r"< POWER_MODEL ID=(\d+) NAME=(\S+) >",Path("/etc/nvpmodel.conf").read_text()))
        return f"{nombres.get(str(idm),'?')} (ID {idm})"
    except (OSError,AttributeError,ValueError):
        return None


def entorno(con_torch=True):
    info={"host":socket.gethostname(),"nvpmodel":modo_energia()}
    try:
        info["l4t"]=Path("/etc/nv_tegra_release").read_text().splitlines()[0].strip("# ")
    except OSError:
        pass
    if con_torch:
        import torch
        import ultralytics
        info["torch"]=torch.__version__
        info["ultralytics"]=ultralytics.__version__
        info["gpu"]=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        try:
            import tensorrt
            info["tensorrt"]=tensorrt.__version__
        except ImportError:
            pass
    return info


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def resolver(p):
    p=Path(p)
    if not p.is_absolute():
        p=PROJECT_ROOT/p
    if not p.exists():
        raise FileNotFoundError(f"no existen los pesos: {p}")
    return p


def guardar(nombre,res,teg,t_ref,filas=None):
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    ruta=OUT_DIR/f"{nombre}.json"
    ruta.write_text(json.dumps(res,indent=2,ensure_ascii=False))
    teg.guardar_csv(OUT_DIR/f"{nombre}_tegrastats.csv",t_ref)
    if filas:
        with open(OUT_DIR/f"{nombre}_frames.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
    print(f"\nJSON: {rel(ruta)}")


# ---------------------------------------------------------------- modos

def modo_reposo(args,teg,t_ref):
    print(f"reposo: {args.segundos} s sin carga. No corras nada mas en la Jetson mientras tanto.")
    time.sleep(2)  #deja arrancar tegrastats
    t0=time.perf_counter()
    time.sleep(args.segundos)
    t1=time.perf_counter()
    res={"modo":"reposo","segundos":round(t1-t0,1),"entorno":entorno(con_torch=False),
         "tegrastats":resumir_teg(teg.ventana(t0,t1))}
    t=res["tegrastats"] or {}
    if "VDD_IN_mw" in t:
        print(f"VDD_IN media={t['VDD_IN_mw']['media']/1000:.2f} W  "
              f"RAM media={t['ram_mb']['media']:.0f} MB  "
              f"temp tj media={t.get('temp_tj',{}).get('media','?')} C")
    guardar(f"reposo__{args.tag or 'reposo'}",res,teg,t_ref)


def precargar(seq,max_frames,ram_mb):
    import cv2
    rutas=sorted((VAL_DIR/seq/"img1").glob("*.jpg"))
    if not rutas:
        raise FileNotFoundError(f"sin frames en {VAL_DIR/seq/'img1'} (copialos desde la laptop con rsync)")
    primero=cv2.imread(str(rutas[0]))
    mb=primero.nbytes/1024**2
    #limite por RAM: 200 frames 4K ocuparian ~5 GB de los 7.4 GB de la Jetson
    n=min(max_frames,len(rutas),max(1,int(ram_mb//mb)))
    frames=[primero]+[cv2.imread(str(r)) for r in rutas[1:n]]
    if any(f is None for f in frames):
        raise RuntimeError(f"no se pudo leer algun frame de {seq}")
    return frames


def modo_ram(args,teg,t_ref):
    import torch
    from src.tracking.detectors import YOLOv8nDetector
    from src.tracking.trackers import ByteTrackAdapter

    reposo=cargar_reposo()
    filas=[]
    print(f"secuencias: {', '.join(args.seqs)}  max {args.max_frames} frames (limite {args.ram_mb} MB)  "
          f"warmup {args.warmup}\nframes precargados en RAM: no se cronometra lectura de disco\n")

    for seq in args.seqs:
        frames=precargar(seq,args.max_frames,args.ram_mb)
        alto,ancho=frames[0].shape[:2]
        print(f"{seq}: {len(frames)} frames {ancho}x{alto}")

        for p in args.weights:
            ruta=resolver(p)
            det=YOLOv8nDetector(ruta,dispositivo=args.device)
            #calentamiento (no se cronometra); el tracker se reinicia despues
            trk=ByteTrackAdapter()
            for f in frames[:args.warmup]:
                trk.update(det.predict(f),f)
            trk=ByteTrackAdapter()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            t_det,t_trk,pre,inf,post=[],[],[],[],[]
            t0=time.perf_counter()
            for f in frames:
                a=time.perf_counter()
                dets=det.predict(f)
                torch.cuda.synchronize()
                b=time.perf_counter()
                trk.update(dets,f)
                c=time.perf_counter()
                t_det.append((b-a)*1000)
                t_trk.append((c-b)*1000)
                pre.append(det.speed["preprocess"])
                inf.append(det.speed["inference"])
                post.append(det.speed["postprocess"])
            t1=time.perf_counter()

            total=media(t_det)+media(t_trk)
            teg_v=resumir_teg(teg.ventana(t0,t1))
            fila={"secuencia":seq,"resolucion":f"{ancho}x{alto}","pesos":rel(ruta),
                  "formato":ruta.suffix.lstrip("."),"imgsz":det.imgsz,"frames":len(frames),
                  "deteccion_ms":resumen(t_det),"pre_ms":resumen(pre),"inferencia_ms":resumen(inf),
                  "post_ms":resumen(post),"tracking_ms":resumen(t_trk),
                  "total_ms":round(total,2),"fps":round(1000/total,2),
                  "fps_reloj":round(len(frames)/(t1-t0),2),
                  "pct_tracker":round(100*media(t_trk)/total,1),
                  "torch_max_mem_mb":round(torch.cuda.max_memory_allocated()/1024**2,1),
                  "energia":energia(teg_v,reposo,t1-t0,len(frames)),
                  "tegrastats":teg_v}
            filas.append(fila)
            print(f"  {ruta.name:24} det={media(t_det):6.2f} ms (pre {media(pre):5.2f} · inf {media(inf):5.2f} · "
                  f"post {media(post):5.2f})  trk={media(t_trk):6.2f} ms  total={total:6.2f} ms  "
                  f"{fila['fps']:6.2f} FPS")
            del det,trk
            gc.collect()
            torch.cuda.empty_cache()
        del frames
        gc.collect()

    res={"modo":"ram","nota":"Metodologia de benchmark_fps.py (Fase 3): frames en RAM, detector+tracker "
         "en el mismo bucle, torch.cuda.synchronize. pre/inf/post son los tiempos internos de Ultralytics.",
         "warmup":args.warmup,"entorno":entorno(),"resultados":filas}
    guardar(f"ram__{args.tag or 'ram'}",res,teg,t_ref)


def modo_video(args,teg,t_ref):
    import cv2
    import psutil
    import torch
    from src.tracking.detectors import YOLOv8nDetector
    from src.tracking.trackers import ByteTrackAdapter

    if len(args.weights)!=1:
        raise SystemExit("--modo video acepta un solo archivo de pesos")
    carpeta=Path(args.videos)
    if not carpeta.is_absolute():
        carpeta=PROJECT_ROOT/carpeta
    videos=sorted(carpeta.glob("*.mp4")) if carpeta.is_dir() else [carpeta]
    if not videos:
        raise FileNotFoundError(f"sin .mp4 en {carpeta}")

    reposo=cargar_reposo()
    ruta=resolver(args.weights[0])
    det=YOLOv8nDetector(ruta,dispositivo=args.device)
    print(f"pesos : {rel(ruta)} (imgsz {det.imgsz})\nvideos: {len(videos)} en {rel(carpeta)}  warmup {args.warmup}\n")

    #calentamiento con el primer video (no se cronometra)
    cap=cv2.VideoCapture(str(videos[0]))
    trk=ByteTrackAdapter()
    for _ in range(args.warmup):
        ok,f=cap.read()
        if not ok:
            break
        trk.update(det.predict(f),f)
    cap.release()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    filas=[]
    por_video=[]
    t0=time.perf_counter()
    for v in videos:
        cap=cv2.VideoCapture(str(v))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV no pudo abrir {v}")
        ancho=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        trk=ByteTrackAdapter()
        n=0
        tv0=time.perf_counter()
        while True:
            a=time.perf_counter()
            ok,f=cap.read()
            b=time.perf_counter()
            if not ok:
                break
            dets=det.predict(f)
            torch.cuda.synchronize()
            c=time.perf_counter()
            trk.update(dets,f)
            d=time.perf_counter()
            s=det.speed
            filas.append({"video":v.stem,"resolucion":f"{ancho}x{alto}","frame":n+1,
                          "lectura_ms":round((b-a)*1000,3),"pre_ms":round(s["preprocess"],3),
                          "inferencia_ms":round(s["inference"],3),"post_ms":round(s["postprocess"],3),
                          "deteccion_ms":round((c-b)*1000,3),"tracking_ms":round((d-c)*1000,3),
                          "total_ms":round((d-a)*1000,3),"n_dets":len(dets)})
            n+=1
        cap.release()
        tv1=time.perf_counter()
        por_video.append({"video":v.stem,"resolucion":f"{ancho}x{alto}","frames":n,
                          "segundos":round(tv1-tv0,2),"fps":round(n/(tv1-tv0),2)})
        print(f"  {v.stem}: {ancho}x{alto}  {n} frames  {n/(tv1-tv0):5.1f} FPS")
    t1=time.perf_counter()

    n_tot=len(filas)
    seg=t1-t0
    def etapas(fs):
        return {k:resumen([x[k] for x in fs]) for k in
                ["lectura_ms","pre_ms","inferencia_ms","post_ms","deteccion_ms","tracking_ms","total_ms","n_dets"]}

    #desglose por resolucion de entrada (el costo de decodificar y redimensionar escala con ella)
    por_res={}
    for r in sorted({x["resolucion"] for x in filas},key=lambda s:int(s.split("x")[0])):
        fs=[x for x in filas if x["resolucion"]==r]
        por_res[r]={"frames":len(fs),"fps":round(1000/media([x["total_ms"] for x in fs]),2),**etapas(fs)}

    teg_v=resumir_teg(teg.ventana(t0,t1))
    res={"modo":"video","pesos":rel(ruta),"imgsz":det.imgsz,"videos":len(videos),"frames":n_tot,
         "warmup":args.warmup,"segundos":round(seg,2),
         "fps_reloj":round(n_tot/seg,2),
         "fps_media_frames":round(1000/media([x["total_ms"] for x in filas]),2),
         "etapas":etapas(filas),"por_resolucion":por_res,"por_video":por_video,
         "memoria":{"torch_max_mem_mb":round(torch.cuda.max_memory_allocated()/1024**2,1),
                    "rss_proceso_mb":round(psutil.Process().memory_info().rss/1024**2,1)},
         "energia":energia(teg_v,reposo,seg,n_tot),"tegrastats":teg_v,
         "nota":"lectura_ms = espera en cap.read(); FFmpeg decodifica en hilos propios, asi que su costo de "
                "CPU aparece en cpu_pct de tegrastats y no completo en lectura_ms.",
         "entorno":entorno()}
    e=res["energia"] or {}
    print(f"\ntotal: {n_tot} frames en {seg:.1f} s -> {res['fps_reloj']} FPS")
    print("ms/frame: "+"  ".join(f"{k.replace('_ms','')}={v['media']}" for k,v in res["etapas"].items()
                                  if k.endswith("_ms")))
    if e:
        print(f"potencia media={e.get('potencia_media_w')} W  energia={e.get('energia_j_por_frame')} J/frame  "
              f"temp tj max={teg_v.get('temp_tj',{}).get('max','?')} C")
    guardar(f"video__{args.tag or ruta.stem}",res,teg,t_ref,filas)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--modo",required=True,choices=["reposo","ram","video"])
    ap.add_argument("--weights",nargs="+",default=[PESOS_DEF],help="Pesos .pt o .engine (varios en modo ram)")
    ap.add_argument("--tag",default=None,help="Sufijo del nombre de salida")
    ap.add_argument("--segundos",type=int,default=60,help="Duracion del modo reposo")
    ap.add_argument("--seqs",nargs="+",default=SEQS_FASE3,help="Secuencias de val para el modo ram")
    ap.add_argument("--max-frames",type=int,default=200,help="Frames por secuencia en modo ram (Fase 3: 200)")
    ap.add_argument("--ram-mb",type=int,default=2500,help="Limite de RAM para precargar frames en modo ram")
    ap.add_argument("--warmup",type=int,default=30,help="Frames de calentamiento (no se cronometran)")
    ap.add_argument("--videos",default=VIDEOS_DEF,help="Carpeta con .mp4 (o un solo .mp4) para el modo video")
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--intervalo-ms",type=int,default=500,help="Intervalo de muestreo de tegrastats")
    args=ap.parse_args()

    t_ref=time.perf_counter()
    with Tegrastats(args.intervalo_ms) as teg:
        {"reposo":modo_reposo,"ram":modo_ram,"video":modo_video}[args.modo](args,teg,t_ref)


if __name__=="__main__":
    main()
