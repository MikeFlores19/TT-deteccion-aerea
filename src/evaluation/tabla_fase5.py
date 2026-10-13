"""
Fase 5 (paso 11): tabla final de la Fase 5 (despliegue en la Jetson Orin Nano).

Junta, desde los archivos del repo (sin volver a medir nada), los resultados
de la config A (YOLOv8n + ByteTrack) en la laptop (RTX 4050) y en la Jetson
(TensorRT FP16 736x1280, MAXN_SUPER), mas los videos propios del DJI Mini 3
y la georreferenciacion.

Bloques: deteccion, seguimiento, velocidad, recursos, campo, georreferenciacion.
Metas del protocolo: MOTA >= 40% y detector YOLOv8n >= 30 FPS.

Salida: results/tables/fase5/tabla_final.{csv,md,tex}

Uso:
  python -m src.evaluation.tabla_fase5
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT=Path(__file__).parent.parent.parent
T=PROJECT_ROOT/"results"/"tables"
OUT_DIR=T/"fase5"
CLASES_MOT=["pedestrian","car","van","truck","bus"]
#engine del sistema (validado el 6 oct)
ENGINE="best_736x1280.engine"


def leer_json(ruta):
    return json.loads((T/ruta).read_text())


def tracking(config):
    """Promedio simple de las 5 clases oficiales en test-dev (protocolo de la Fase 3)."""
    vals={}
    for cl in CLASES_MOT:
        h,v=(T/"tracking"/"test-dev"/cl/config/"pedestrian_summary.txt").read_text().split("\n")[:2]
        vals[cl]=dict(zip(h.split(),map(float,v.split())))
    prom=lambda k:sum(vals[c][k] for c in CLASES_MOT)/len(CLASES_MOT)
    return {"MOTA":prom("MOTA"),"IDF1":prom("IDF1"),"HOTA":prom("HOTA"),
            "IDSW":sum(vals[c]["IDSW"] for c in CLASES_MOT)}


def main():
    filas=[]

    def fila(bloque,metrica,laptop="",jetson="",meta="",cumple="",fuente=""):
        filas.append({"bloque":bloque,"metrica":metrica,"laptop":laptop,"jetson":jetson,
                      "meta":meta,"cumple":cumple,"fuente":fuente})

    # ---------------------------------------------------------------- deteccion
    b="Detección (VisDrone-DET val, 548 img, 10 clases)"
    lap=leer_json("cross_eval/yolov8n__visdrone_val10.json")
    trt=leer_json("cross_eval/yolov8n__visdrone_val10__jetson_trt16.json")
    nor=leer_json("cross_eval/yolov8n__visdrone_val10__jetson_pt_norect.json")
    for k in ("mAP50","mAP50-95"):
        d=100*(trt[k]-lap[k])/lap[k]
        #el mAP de la Jetson es del engine 1280x1280 del paso 4 (validacion .pt vs TRT)
        fila(b,f"{k} (Jetson: TRT FP16 1280x1280)",f"{lap[k]:.4f}",f"{trt[k]:.4f} ({d:+.1f}%)",
             fuente="cross_eval/yolov8n__visdrone_val10*.json")
    fila(b,"mAP50 del .pt en la Jetson con el mismo letterbox que el engine","",f"{nor['mAP50']:.4f}",
         fuente="la diferencia viene del letterbox, no de FP16")

    # -------------------------------------------------------------- seguimiento
    b="Seguimiento (VisDrone-MOT test-dev, 17 sec., 5 clases; Jetson: TRT 736x1280)"
    lap=tracking("A_yolov8n_bytetrack")
    jet=tracking("A_yolov8n_bytetrack_jetson_trt16_736")
    j12=tracking("A_yolov8n_bytetrack_jetson_trt16")
    mp4=tracking("A_yolov8n_bytetrack_jetson_trt16_736_mp4")
    fila(b,"MOTA (%)",f"{lap['MOTA']:.2f}",f"{jet['MOTA']:.2f}",">= 40",
         "sí" if jet["MOTA"]>=40 else "no","tracking/test-dev/<clase>/<config>/")
    fila(b,"IDF1 (%)",f"{lap['IDF1']:.2f}",f"{jet['IDF1']:.2f}")
    fila(b,"HOTA (%)",f"{lap['HOTA']:.2f}",f"{jet['HOTA']:.2f}")
    fila(b,"IDSW (suma 5 clases)",f"{lap['IDSW']:.0f}",f"{jet['IDSW']:.0f}")
    fila(b,"MOTA con engine 1280x1280 (descartado)","",f"{j12['MOTA']:.2f}",">= 40",
         "sí" if j12["MOTA"]>=40 else "no","letterbox cuadrado: por eso se uso 736x1280")
    fila(b,"MOTA / IDF1 / HOTA leyendo .mp4 (pipeline real)","",
         f"{mp4['MOTA']:.2f} / {mp4['IDF1']:.2f} / {mp4['HOTA']:.2f}")

    # ----------------------------------------------------------------- velocidad
    b="Velocidad config A (frames en RAM; Jetson MAXN_SUPER, TRT 736x1280)"
    lap_fps={r["secuencia"]:r for r in leer_json("eficiencia/fps_tracking_val.json")["resultados"] if r["config"]=="A"}
    lap_det=leer_json("eficiencia/eficiencia_NVIDIA_GeForce_RTX_4050_Laptop_GPU.json")["yolov8n_mosaic10"]
    r6=pd.read_csv(T/"eficiencia_jetson"/"resumen_paso6.csv")
    eng=r6[(r6["pesos"]==ENGINE)&(r6["engine_fecha"]=="6oct")]
    sel=lambda modo,cond,sec=None:eng[(eng["modo"]==modo)&(eng["condicion"]==cond)&
                                     ((eng["secuencia"]==sec) if sec else True)].iloc[0]
    densa,disp="uav0000086_00000_v","uav0000268_05773_v"
    jd,jx=sel("ram","jetson_clocks",densa),sel("ram","jetson_clocks",disp)
    dd,dx=sel("ram","default",densa),sel("ram","default",disp)
    fila(b,"FPS det+track, secuencia densa (~215 obj/frame)",f"{lap_fps[densa]['fps']:.1f}",
         f"{jd['fps']:.1f} (sin jetson_clocks {dd['fps']:.1f})",fuente="eficiencia/ y eficiencia_jetson/")
    fila(b,"FPS det+track, secuencia dispersa 4K (~102 obj/frame)",f"{lap_fps[disp]['fps']:.1f}",
         f"{jx['fps']:.1f} (sin jetson_clocks {dx['fps']:.1f})")
    fps_det=1000/jd["deteccion_ms"]
    fila(b,"Detector YOLOv8n: pre+inferencia+post (ms / FPS)","",
         f"{jd['deteccion_ms']:.1f} ms / {fps_det:.0f} FPS",">= 30 FPS","sí" if fps_det>=30 else "no",
         "secuencia densa, jetson_clocks")
    fila(b,"Inferencia del modelo (ms)",f"{lap_det['latency_ms']:.2f} (.pt, 1280)",
         f"{jd['inferencia_ms']:.2f} (TRT FP16, 736x1280)",fuente="la laptop se midio en la Fase 2 (.pt)")
    fila(b,"Tracking ByteTrack, secuencia densa (ms)",f"{lap_fps[densa]['tracking_ms']:.2f}",f"{jd['tracking_ms']:.2f}")
    vc,vd=sel("video","jetson_clocks"),sel("video","default")
    #fps_reloj (frames / tiempo total), la misma cifra que usa el resto del proyecto
    fila(b,"FPS leyendo .mp4 (17 videos de test-dev)","",f"{vc['fps_reloj']:.1f} (sin jetson_clocks {vd['fps_reloj']:.1f})")

    # ------------------------------------------------------------------ recursos
    b="Recursos en la Jetson (jetson_clocks)"
    #el reposo no usa engine: se toma de la tabla completa
    rc=r6[(r6["modo"]=="reposo")&(r6["condicion"]=="jetson_clocks")].iloc[0]
    fila(b,"Potencia media (W): densa / dispersa / .mp4 / reposo","",
         f"{jd['w_media']:.1f} / {jx['w_media']:.1f} / {vc['w_media']:.1f} / {rc['w_media']:.1f}")
    fila(b,"Energía por frame (J): densa / dispersa / .mp4","",
         f"{jd['j_por_frame']:.3f} / {jx['j_por_frame']:.3f} / {vc['j_por_frame']:.3f}")
    fila(b,"Temperatura máxima Tj (°C)","",f"{max(jd['tj_max_c'],jx['tj_max_c'],vc['tj_max_c']):.1f}")
    #la VRAM de la laptop (Fase 3) es la medida defectuosa que no se presenta como consumo en operacion
    fila(b,"Memoria GPU del modelo (MB)","",f"{jd['torch_max_mem_mb']:.1f} (torch max)")
    fila(b,"RAM del sistema (MB): densa / dispersa / .mp4","",
         f"{jd['ram_sistema_media_mb']:.0f} / {jx['ram_sistema_media_mb']:.0f} / {vc['ram_sistema_media_mb']:.0f} de 8 GB")

    # --------------------------------------------------------------------- campo
    b="Campo: 27 videos propios del DJI Mini 3 (Jetson)"
    dji=pd.read_csv(T/"dji_mini3"/"resumen_videos.csv")
    tot=dji[dji["video"]=="TOTAL"].iloc[0]
    vids=dji[dji["video"]!="TOTAL"]
    cls=pd.read_csv(T/"dji_mini3"/"resumen_clases.csv")
    pct=lambda c:float(cls.loc[cls["clase"]==c,"pct"].iat[0])
    fila(b,"Videos / duración / frames","",f"{len(vids)} / {tot['duracion_s']/60:.1f} min / {int(tot['frames']):,}")
    fila(b,"Detecciones con ID (tracks-frame)","",f"{int(tot['tracks_frame']):,}")
    fila(b,"IDs únicos / eventos","",f"{int(tot['ids_unicos']):,} / {int(tot['eventos']):,} "
         f"(car {pct('car'):.1f}%, pedestrian {pct('pedestrian'):.1f}%)")
    fila(b,"FPS del pipeline det+track (media)","",f"{tot['fps_pipeline']:.1f} "
         f"({int((vids['fps_pipeline']>=30).sum())}/{len(vids)} videos >= 30)")
    fila(b,"FPS total con lectura y escritura del video anotado","",f"{tot['fps_total']:.1f}")

    # ------------------------------------------------------------ georreferencia
    b="Georreferenciación de los videos propios"
    da=pd.read_csv(T/"georef"/"desfase_originales.csv")
    ub=pd.read_csv(T/"georef"/"ubicacion_clips.csv")
    val=pd.read_csv(T/"georef"/"validacion_sintetica.csv")
    rg=pd.read_csv(T/"georef"/"resumen_georef.csv")
    ob=pd.read_csv(T/"georef"/"objetos_geo.csv")
    zv=pd.read_csv(T/"georef"/"zona_valida.csv")
    fila(b,"Sincronización A (original vs registro de vuelo)","",
         f"{len(da)} originales; |A-isVideo| <= {da['A_menos_isvideo_s'].abs().max():.2f} s; "
         f"RMSE H <= {da['rmse_H_m'].max():.2f} m")
    fila(b,"Sincronización B (clip vs original)","",
         f"{int((ub['estado']=='OK').sum())}/{len(ub)} OK; residual {ub['rms_res_ms'].min():.0f}-{ub['rms_res_ms'].max():.0f} ms; "
         f"velocidad {ub['velocidad_medida'].dropna().min():.4f}-{ub['velocidad_medida'].dropna().max():.4f}")
    fila(b,"Validación sintética (verdad conocida)","",f"{int(val['pasa'].sum())}/{len(val)} casos pasan")
    g,d=int(rg["georreferenciadas"].sum()),int(rg["detecciones"].sum())
    fila(b,"Detecciones georreferenciadas","",f"{g:,}/{d:,} ({100*g/d:.1f}%)")
    fila(b,"Objetos georreferenciados","",f"{len(ob):,}")
    #<= 10 m es el filtro de validez del 9.3 (se cumple por construccion), no una meta del protocolo
    fila(b,"Error estimado mediano por clip (m)","",f"{rg['error_mediano_m'].min():.1f}-{rg['error_mediano_m'].max():.1f}",
         fuente="filtro de validez <= 10 m por deteccion; propagacion: pitch 1 deg, rumbo 3 deg, altura 0.5 m, GPS 1.5 m")
    fila(b,"Dispersión por objeto, mediana (m)","",f"{ob['dispersion_m'].median():.2f}",fuente="repetibilidad, no exactitud")
    fila(b,"Zona válida de la imagen (%)","",f"{zv['pct_zona_valida'].min():.0f}-{zv['pct_zona_valida'].max():.0f}",
         fuente="camara inclinada (gimbal -10 a -26 deg): vista oblicua")

    res=pd.DataFrame(filas)
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    res.to_csv(OUT_DIR/"tabla_final.csv",index=False)

    #markdown: una tabla por bloque
    md=["# Fase 5 - Tabla final (config A: YOLOv8n + ByteTrack)\n",
        "Laptop: RTX 4050 (.pt). Jetson: Orin Nano 8GB, MAXN_SUPER, TensorRT FP16 736x1280.\n"]
    for bloque,g in res.groupby("bloque",sort=False):
        md.append(f"\n## {bloque}\n\n| Métrica | Laptop | Jetson | Meta | Cumple | Nota |\n|---|---|---|---|---|---|")
        for r in g.itertuples():
            md.append(f"| {r.metrica} | {r.laptop} | {r.jetson} | {r.meta} | {r.cumple} | {r.fuente} |")
    (OUT_DIR/"tabla_final.md").write_text("\n".join(md)+"\n",encoding="utf-8")

    #latex (booktabs): sin la columna de notas
    esc=lambda s:str(s).replace("\\","\\textbackslash{}").replace("%","\\%").replace("_","\\_") \
        .replace(">=","$\\geq$").replace("<=","$\\leq$").replace("&","\\&").replace("#","\\#") \
        .replace("~","$\\sim$").replace("|","$|$")
    tex=["% Requiere \\usepackage{booktabs} y \\usepackage{array}",
         "\\begin{table}[htbp]","\\centering","\\scriptsize",
         "\\caption{Resultados de la Fase 5: configuración A (YOLOv8n + ByteTrack) en la laptop (RTX 4050) "
         "y en la Jetson Orin Nano (TensorRT FP16, $736\\times1280$, MAXN\\_SUPER).}",
         "\\label{tab:fase5_final}",
         #columnas alineadas a la izquierda (sin justificar)
         "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{5.6cm}>{\\raggedright\\arraybackslash}p{2.6cm}"
         ">{\\raggedright\\arraybackslash}p{4.4cm}>{\\raggedright\\arraybackslash}p{1.6cm}}","\\toprule",
         "\\textbf{Métrica} & \\textbf{Laptop} & \\textbf{Jetson} & \\textbf{Meta} \\\\","\\midrule"]
    for i,(bloque,g) in enumerate(res.groupby("bloque",sort=False)):
        if i:
            tex.append("\\midrule")
        tex.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{esc(bloque)}}}}} \\\\")
        for r in g.itertuples():
            meta=esc(r.meta)+(f" ({esc(r.cumple)})" if r.cumple else "")
            tex.append(f"{esc(r.metrica)} & {esc(r.laptop)} & {esc(r.jetson)} & {meta} \\\\")
    tex+=["\\bottomrule","\\end{tabular}","\\end{table}"]
    (OUT_DIR/"tabla_final.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")

    print(res[["bloque","metrica","laptop","jetson","meta","cumple"]].to_string(index=False))
    print(f"\n{(OUT_DIR/'tabla_final.csv').relative_to(PROJECT_ROOT)}  (+ .md, .tex)")


if __name__=="__main__":
    main()
