"""
Fase 5 (paso 9.4): mapas de la georreferenciacion.

1) Mapa interactivo (Leaflet, un solo .html): fondo satelital u OSM, una capa
   por vuelo con la trayectoria del dron, los objetos (posicion mediana) y el
   recorrido de los que se movieron. Clic en un objeto: clip, ID, clase, hora,
   error estimado y dispersion.
2) Un mapa estatico por vuelo (.jpg) para el reporte: fondo satelital (teselas
   de Esri World Imagery, con cache local), trayectoria, objetos por clase,
   barra de escala y norte.

La revision visual de la EXACTITUD se hace aqui: los coches deben caer en las
calles y las personas en banquetas/cancha. La dispersion (9.3b) solo mide la
repetibilidad.

Entrada: results/tables/georef/{objetos_geo,ubicacion_clips}.csv,
         runs/video/dji_<clip>/{dron.csv,geo.csv.gz}
Salida : results/figures/georef/mapa_georef.html
         results/figures/georef/mapas/vuelo_NN.jpg

Uso:
  python -m src.georef.mapa
  python -m src.georef.mapa --sin-png
"""

import argparse
import json
import math
import urllib.request

import cv2
import numpy as np
import pandas as pd

from src.georef.inventario_vuelos import PROJECT_ROOT, OUT_DIR, rel

RUNS_DIR=PROJECT_ROOT/"runs"/"video"
FIG_DIR=PROJECT_ROOT/"results"/"figures"/"georef"
CACHE_TESELAS=PROJECT_ROOT/"data"/"teselas_esri"
URL_ESRI="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ATRIBUCION="Imagen satelital: Esri, Maxar, Earthstar Geographics"
UTC_OFFSET_H=-6

#mismos colores en el html y en los jpg
COLOR_CLASE={"pedestrian":"#ff3b30","people":"#ff9500","bicycle":"#ffcc00","car":"#00e5ff",
             "van":"#2f7bff","truck":"#af52de","tricycle":"#ff2d92","awning-tricycle":"#a2845e",
             "bus":"#34c759","motor":"#ff00ff"}
#un objeto "se movio" si su recorrido georreferenciado supera esto
MOVIMIENTO_M=5.0
#recorridos y trayectoria del dron submuestreados (frames)
PASO_RECORRIDO=15
PASO_DRON=15
MARGEN_M=15.0
ANCHO_PNG_PX=1600
ZOOM_MAX=19


def distancia_m(lat1,lon1,lat2,lon2):
    p=np.radians((np.asarray(lat1)+np.asarray(lat2))/2)
    dn=np.radians(np.asarray(lat2)-np.asarray(lat1))*6371008.8
    de=np.radians(np.asarray(lon2)-np.asarray(lon1))*6371008.8*np.cos(p)
    return np.hypot(de,dn)


def cargar():
    """Objetos, trayectorias del dron y recorridos de objetos en movimiento, por clip."""
    obj=pd.read_csv(OUT_DIR/"objetos_geo.csv")
    ub=pd.read_csv(OUT_DIR/"ubicacion_clips.csv")
    ub=ub[ub["estado"]=="OK"]
    dron,recorridos={},{}
    for clip in ub["clip"]:
        carpeta=RUNS_DIR/f"dji_{clip}"
        d=pd.read_csv(carpeta/"dron.csv").dropna(subset=["lat","lon"])
        dron[clip]=d.iloc[::PASO_DRON]
        g=pd.read_csv(carpeta/"geo.csv.gz")
        ids=set(obj.loc[obj["clip"]==clip,"id"])
        rec={}
        for tid,t in g[g["id"].isin(ids)].groupby("id"):
            if distancia_m(t["lat"].iat[0],t["lon"].iat[0],t["lat"].iat[-1],t["lon"].iat[-1])>=MOVIMIENTO_M:
                rec[int(tid)]=t.iloc[::PASO_RECORRIDO][["lat","lon"]].round(7).to_numpy().tolist()
        recorridos[clip]=rec
    return obj,ub,dron,recorridos


def hora_local(iso):
    t=pd.to_datetime(iso)+pd.Timedelta(hours=UTC_OFFSET_H)
    return t.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ html
def mapa_html(obj,ub,dron,recorridos,ruta):
    vuelos=[]
    for vuelo,gv in ub.groupby("vuelo"):
        capa={"vuelo":int(vuelo),"clips":[],"objetos":[],"recorridos":[],"dron":[]}
        for clip in gv["clip"]:
            capa["clips"].append(clip)
            capa["dron"].append({"clip":clip,"pts":dron[clip][["lat","lon"]].round(7).to_numpy().tolist()})
            for o in obj[obj["clip"]==clip].itertuples():
                capa["objetos"].append({
                    "lat":o.lat,"lon":o.lon,"c":o.clase,
                    "t":(f"<b>{clip} #{o.id}</b> {o.clase}<br>{hora_local(o.inicio_utc)} (hora local)"
                         f"<br>frames georref.: {o.frames_georef}<br>error estimado: {o.error_mediano_m} m"
                         f"<br>dispersion: {o.dispersion_m} m<br>distancia al dron: {o.dist_mediana_m} m")})
                if int(o.id) in recorridos[clip]:
                    capa["recorridos"].append({"c":o.clase,"pts":recorridos[clip][int(o.id)]})
        vuelos.append(capa)

    html=PLANTILLA_HTML.replace("__DATOS__",json.dumps(vuelos,separators=(",",":"))) \
                       .replace("__COLORES__",json.dumps(COLOR_CLASE)) \
                       .replace("__ATRIB__",ATRIBUCION)
    ruta.parent.mkdir(parents=True,exist_ok=True)
    ruta.write_text(html,encoding="utf-8")


PLANTILLA_HTML="""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Georreferenciación DJI Mini 3</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html,body,#m{height:100%;margin:0;font-family:system-ui,sans-serif}
.ley{background:#fff;padding:8px 10px;border-radius:6px;font-size:12px;line-height:18px;box-shadow:0 1px 4px #0004}
.ley i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;border:1px solid #fff}
</style></head><body><div id="m"></div>
<script>
const V=__DATOS__, C=__COLORES__;
const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom:21,maxNativeZoom:19,attribution:'__ATRIB__'});
const osm=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:21,maxNativeZoom:19,attribution:'&copy; OpenStreetMap'});
const m=L.map('m',{layers:[sat]});
const capas={}, limites=L.latLngBounds([]);
for(const v of V){
  const g=L.layerGroup();
  for(const d of v.dron){
    if(d.pts.length<2) continue;
    L.polyline(d.pts,{color:'#fff',weight:3,opacity:.9}).bindTooltip('dron '+d.clip).addTo(g);
    L.circleMarker(d.pts[0],{radius:5,color:'#000',fillColor:'#fff',fillOpacity:1,weight:1})
      .bindTooltip('inicio '+d.clip).addTo(g);
    d.pts.forEach(p=>limites.extend(p));
  }
  for(const r of v.recorridos)
    L.polyline(r.pts,{color:C[r.c]||'#fff',weight:1.5,opacity:.8}).addTo(g);
  for(const o of v.objetos){
    L.circleMarker([o.lat,o.lon],{radius:4,color:'#fff',weight:1,fillColor:C[o.c]||'#fff',fillOpacity:.95})
      .bindPopup(o.t).addTo(g);
    limites.extend([o.lat,o.lon]);
  }
  g.addTo(m);
  capas['Vuelo '+v.vuelo+' ('+v.clips.join(', ')+')']=g;
}
L.control.layers({'Satelite':sat,'OpenStreetMap':osm},capas,{collapsed:false}).addTo(m);
L.control.scale({imperial:false}).addTo(m);
const ley=L.control({position:'bottomright'});
ley.onAdd=()=>{const d=L.DomUtil.create('div','ley');
  d.innerHTML='<b>Clase</b><br>'+Object.entries(C).map(([k,c])=>'<i style="background:'+c+'"></i>'+k).join('<br>')+
  '<br><span style="display:inline-block;width:14px;border-top:3px solid #888;margin-right:6px"></span>dron';return d;};
ley.addTo(m);
m.fitBounds(limites.pad(0.05));
</script></body></html>
"""


# ------------------------------------------------------------------ png
def lonlat_a_px(lon,lat,z):
    """Web Mercator: grados -> pixeles globales a zoom z."""
    n=256*2**z
    x=(np.asarray(lon)+180)/360*n
    la=np.radians(np.asarray(lat))
    y=(1-np.log(np.tan(la)+1/np.cos(la))/math.pi)/2*n
    return x,y


def tesela(z,x,y):
    ruta=CACHE_TESELAS/f"{z}_{x}_{y}.jpg"
    if ruta.exists():
        return cv2.imread(str(ruta))
    try:
        req=urllib.request.Request(URL_ESRI.format(z=z,x=x,y=y),headers={"User-Agent":"TT2026-2_IA05 (tesis IPN)"})
        datos=urllib.request.urlopen(req,timeout=15).read()
        img=cv2.imdecode(np.frombuffer(datos,np.uint8),cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("tesela vacia")
        CACHE_TESELAS.mkdir(parents=True,exist_ok=True)
        ruta.write_bytes(datos)
        return img
    except Exception as e:
        print(f"  aviso: tesela {z}/{x}/{y} no disponible ({e})")
        return None


def fondo(lat_min,lat_max,lon_min,lon_max):
    """Mosaico satelital que cubre el recuadro; devuelve (img, z, x0, y0) en pixeles globales."""
    lat_c=(lat_min+lat_max)/2
    ancho_m=distancia_m(lat_c,lon_min,lat_c,lon_max)
    mpp=max(ancho_m/ANCHO_PNG_PX,1e-3)
    z=int(np.clip(math.floor(math.log2(156543.03392*math.cos(math.radians(lat_c))/mpp)),12,ZOOM_MAX))
    x0,y0=lonlat_a_px(lon_min,lat_max,z)
    x1,y1=lonlat_a_px(lon_max,lat_min,z)
    tx0,ty0,tx1,ty1=int(x0//256),int(y0//256),int(x1//256),int(y1//256)
    mosaico=np.full(((ty1-ty0+1)*256,(tx1-tx0+1)*256,3),90,np.uint8)
    faltan=0
    for ty in range(ty0,ty1+1):
        for tx in range(tx0,tx1+1):
            t=tesela(z,tx,ty)
            if t is None:
                faltan+=1
                continue
            mosaico[(ty-ty0)*256:(ty-ty0+1)*256,(tx-tx0)*256:(tx-tx0+1)*256]=t[:256,:256]
    #recorte exacto al recuadro
    ox,oy=int(x0-tx0*256),int(y0-ty0*256)
    img=mosaico[oy:oy+int(y1-y0),ox:ox+int(x1-x0)]
    return img,z,x0,y0,faltan


def mapa_png(vuelo,clips,obj,dron,recorridos,ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o=obj[obj["clip"].isin(clips)]
    lats=np.concatenate([o["lat"].to_numpy()]+[dron[c]["lat"].to_numpy() for c in clips])
    lons=np.concatenate([o["lon"].to_numpy()]+[dron[c]["lon"].to_numpy() for c in clips])
    lat_c=float(np.median(lats))
    dlat=np.degrees(MARGEN_M/6371008.8)
    dlon=dlat/math.cos(math.radians(lat_c))
    la0,la1=lats.min()-dlat,lats.max()+dlat
    lo0,lo1=lons.min()-dlon,lons.max()+dlon
    img,z,x0,y0,faltan=fondo(la0,la1,lo0,lo1)
    #zonas chicas: se amplia el fondo para que todas las figuras tengan el mismo ancho
    k=max(1.0,ANCHO_PNG_PX/img.shape[1])
    if k>1:
        img=cv2.resize(img,None,fx=k,fy=k,interpolation=cv2.INTER_CUBIC)
    px=lambda lo,la:((lonlat_a_px(lo,la,z)[0]-x0)*k,(lonlat_a_px(lo,la,z)[1]-y0)*k)

    h,w=img.shape[:2]
    fig,ax=plt.subplots(figsize=(w/150,h/150),dpi=150)
    ax.imshow(img[:,:,::-1],extent=(0,w,h,0))
    for c in clips:
        x,y=px(dron[c]["lon"].to_numpy(),dron[c]["lat"].to_numpy())
        ax.plot(x,y,color="white",lw=2.2,zorder=3)
        ax.plot(x[:1],y[:1],"o",ms=6,mfc="white",mec="black",zorder=4)
        ax.annotate(c,(x[0],y[0]),xytext=(4,4),textcoords="offset points",color="white",fontsize=7,zorder=4)
        for tid,pts in recorridos[c].items():
            if tid in set(o.loc[o["clip"]==c,"id"]):
                p=np.array(pts)
                cl=o.loc[(o["clip"]==c)&(o["id"]==tid),"clase"].iat[0]
                xr,yr=px(p[:,1],p[:,0])
                ax.plot(xr,yr,color=COLOR_CLASE.get(cl,"white"),lw=0.8,alpha=0.8,zorder=2)
    for cl,g in o.groupby("clase"):
        x,y=px(g["lon"].to_numpy(),g["lat"].to_numpy())
        ax.scatter(x,y,s=14,c=COLOR_CLASE.get(cl,"white"),edgecolors="white",linewidths=0.4,
                   label=f"{cl} ({len(g)})",zorder=5)
    ax.plot([],[],color="white",lw=2.2,label="dron")

    #barra de escala (m por pixel a esta latitud y zoom)
    mpp=156543.03392*math.cos(math.radians(lat_c))/2**z/k
    largo_m=next(v for v in (5,10,20,25,50,100,200,500) if v/mpp>=w*0.12)
    bx,by=w*0.04,h*0.95
    ax.plot([bx,bx+largo_m/mpp],[by,by],color="white",lw=4,solid_capstyle="butt",zorder=6)
    ax.text(bx+largo_m/mpp/2,by-h*0.015,f"{largo_m} m",color="white",ha="center",va="bottom",fontsize=9,zorder=6)
    ax.annotate("N",xy=(w*0.96,h*0.06),xytext=(w*0.96,h*0.14),color="white",ha="center",fontsize=11,
                arrowprops=dict(arrowstyle="-|>",color="white",lw=2),zorder=6)
    ax.legend(loc="upper left",fontsize=7,framealpha=0.85)
    ax.set_title(f"Vuelo {vuelo}  ({', '.join(clips)})  -  {len(o)} objetos georreferenciados",fontsize=10)
    ax.text(w-4,h-4,ATRIBUCION if faltan==0 else "fondo incompleto (sin conexion)",color="white",
            ha="right",va="bottom",fontsize=6,zorder=6)
    ax.set_xlim(0,w)
    ax.set_ylim(h,0)
    ax.axis("off")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(ruta,dpi=150,pil_kwargs={"quality":88})
    plt.close(fig)
    return z,faltan


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sin-png",action="store_true")
    ap.add_argument("--sin-html",action="store_true")
    args=ap.parse_args()

    obj,ub,dron,recorridos=cargar()
    if not args.sin_html:
        ruta=FIG_DIR/"mapa_georef.html"
        mapa_html(obj,ub,dron,recorridos,ruta)
        print(f"html: {rel(ruta)}  ({ruta.stat().st_size/1e6:.1f} MB, {len(obj)} objetos)")
    if not args.sin_png:
        for vuelo,gv in ub.groupby("vuelo"):
            ruta=FIG_DIR/"mapas"/f"vuelo_{int(vuelo):02d}.jpg"
            z,faltan=mapa_png(int(vuelo),list(gv["clip"]),obj,dron,recorridos,ruta)
            print(f"vuelo {int(vuelo):2d}: zoom {z}  {rel(ruta)}"+(f"  ({faltan} teselas sin descargar)" if faltan else ""))


if __name__=="__main__":
    main()
