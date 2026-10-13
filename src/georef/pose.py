"""
Fase 5 (paso 9.3): modelo de camara del DJI Mini 3 y pose del dron en cada
instante de un clip.

Camara (pinhole, sin distorsion: DJI corrige la del lente en el video):
  FOV diagonal 82.1 deg en 4:3 (ficha de DJI). El video 16:9 usa el ancho
  completo del sensor -> FOV horizontal 69.7 deg -> f = 1378 px en 1920x1080.
  Se verifica dibujando el horizonte predicho sobre frames reales (9.3a).

Pose (del CSV del vuelo, sincronizado con A y B del 9.2):
  lat/lon (OSD), altura relativa al despegue (OSD.height, m), GIMBAL.pitch
  (negativo = hacia abajo), GIMBAL.roll y rumbo GIMBAL.yaw [360].

Proyeccion: el rayo del pixel se intersecta con el plano del suelo a la
altura del despegue (supuesto de suelo plano).

Ejes: ENU local con origen en la vertical del dron (E este, N norte, U arriba).
"""

import numpy as np
import pandas as pd

from src.georef.inventario_vuelos import (PROJECT_ROOT, OUT_DIR, PIE_A_M,
                                          inicio_utc, tiempos_utc)
from src.georef.sincronizar_originales import interp_con_huecos

ANCHO,ALTO=1920,1080
FOV_DIAG_43_DEG=82.1
#4:3 -> ancho/diagonal=4/5; el 16:9 conserva el ancho
F_PX=(ANCHO/2)/(np.tan(np.radians(FOV_DIAG_43_DEG/2))*4/5)
CX,CY=ANCHO/2,ALTO/2
RADIO_TIERRA_M=6371008.8

#incertidumbres supuestas (conservadoras, equipo de consumo) para el error estimado
SIGMA_PITCH_DEG=1.0
SIGMA_YAW_DEG=3.0
SIGMA_H_M=0.5
SIGMA_GPS_M=1.5
#un punto solo se georreferencia si el rayo baja al menos esto bajo el horizonte
DEPRESION_MIN_DEG=3.0
#y si su error estimado no pasa de esto (zona valida de la imagen)
ERROR_MAX_M=10.0


def base_camara(pitch_deg,roll_deg,yaw_deg):
    """Vectores (adelante, derecha, abajo) de la camara en ENU. Acepta arreglos."""
    p,r,y=np.radians(pitch_deg),np.radians(roll_deg),np.radians(yaw_deg)
    adelante=np.stack([np.sin(y)*np.cos(p),np.cos(y)*np.cos(p),np.sin(p)],-1)
    derecha0=np.stack([np.cos(y),-np.sin(y),np.zeros_like(y)],-1)
    arriba0=np.cross(derecha0,adelante)
    #roll: giro de derecha/arriba alrededor del eje optico
    cr,sr=np.cos(r)[...,None],np.sin(r)[...,None]
    derecha=derecha0*cr-arriba0*sr
    arriba=arriba0*cr+derecha0*sr
    return adelante,derecha,-arriba


def rayo(u,v,pitch,roll,yaw,f=F_PX,cx=CX,cy=CY):
    """Direccion ENU (no unitaria) del rayo del pixel (u,v)."""
    a,d,b=base_camara(pitch,roll,yaw)
    x=((np.asarray(u,float)-cx)/f)[...,None]
    yb=((np.asarray(v,float)-cy)/f)[...,None]
    return a+x*d+yb*b


def a_suelo(u,v,h,pitch,roll,yaw,f=F_PX):
    """Pixel -> (E,N) en metros respecto a la vertical del dron, depresion (deg).

    E,N = NaN si el rayo no baja al menos DEPRESION_MIN_DEG (horizonte o cielo).
    """
    d=rayo(u,v,pitch,roll,yaw,f)
    norma=np.linalg.norm(d,axis=-1)
    depresion=np.degrees(np.arcsin(np.clip(-d[...,2]/norma,-1,1)))
    ok=(depresion>=DEPRESION_MIN_DEG)&(np.asarray(h)>0)
    t=np.where(ok,np.asarray(h)/np.where(ok,-d[...,2],1),np.nan)
    return t*d[...,0],t*d[...,1],depresion


def error_estimado(u,v,h,pitch,roll,yaw):
    """Error horizontal estimado (m): propagacion por diferencias finitas + GPS."""
    E0,N0,_=a_suelo(u,v,h,pitch,roll,yaw)
    comps=[]
    for dp,dy,dh in ((SIGMA_PITCH_DEG,0,0),(0,SIGMA_YAW_DEG,0),(0,0,SIGMA_H_M)):
        E1,N1,_=a_suelo(u,v,np.asarray(h)+dh,np.asarray(pitch)+dp,roll,np.asarray(yaw)+dy)
        comps.append((E1-E0)**2+(N1-N0)**2)
    return np.sqrt(sum(comps)+SIGMA_GPS_M**2)


def horizonte_v(u,pitch,roll,yaw,f=F_PX,cx=CX,cy=CY):
    """Fila v del horizonte (rayo horizontal) en la columna u."""
    a,d,b=base_camara(pitch,roll,yaw)
    x=(np.asarray(u,float)-cx)/f
    yb=-(a[...,2]+x*d[...,2])/b[...,2]
    return cy+f*yb


def enu_a_latlon(lat0,lon0,E,N):
    """Desplazamiento local (m) -> lat/lon (aproximacion plana, valida a cientos de metros)."""
    lat=lat0+np.degrees(np.asarray(N)/RADIO_TIERRA_M)
    lon=lon0+np.degrees(np.asarray(E)/(RADIO_TIERRA_M*np.cos(np.radians(lat0))))
    return lat,lon


class PoseClip:
    """Pose del dron en cualquier instante de un clip (segundos desde su primer frame)."""

    def __init__(self,clip):
        ub=pd.read_csv(OUT_DIR/"ubicacion_clips.csv")
        fila=ub[ub["clip"]==clip]
        if fila.empty or pd.isna(fila.iloc[0].get("inicio_utc")):
            raise ValueError(f"{clip}: sin sincronizacion en ubicacion_clips.csv")
        fila=fila.iloc[0]
        asig=pd.read_csv(OUT_DIR/"asignacion_originales.csv")
        o=asig[asig["original"]==fila["original"]].iloc[0]
        tramos=pd.read_csv(OUT_DIR/"tramos_grabacion.csv")
        tr=tramos[(tramos["vuelo"]==o["vuelo"])&(tramos["tramo"]==o["tramo"])].iloc[0]
        ruta_csv=PROJECT_ROOT/tr["csv"]

        d=pd.read_csv(ruta_csv,encoding="utf-8-sig",low_memory=False)
        t,_=tiempos_utc(d,inicio_utc(ruta_csv))
        t_ref=t.iloc[0]
        self.seg=np.array([(x-t_ref).total_seconds() for x in t])
        #copy=True: con pandas 3 to_numpy puede devolver un arreglo de solo lectura
        num=lambda c:pd.to_numeric(d[c],errors="coerce").to_numpy(dtype=float,copy=True)
        lat,lon=num("OSD.latitude"),num("OSD.longitude")
        malo=np.isnan(lat)|np.isnan(lon)|(lat==0)|(lon==0)
        lat[malo]=np.nan
        lon[malo]=np.nan
        self.lat,self.lon=lat,lon
        self.h=num("OSD.height [ft]")*PIE_A_M
        self.pitch=num("GIMBAL.pitch")
        self.roll=num("GIMBAL.roll")
        #rumbo desenrollado para interpolar sin saltos 359->0
        self.yaw_unw=np.degrees(np.unwrap(np.radians(num("GIMBAL.yaw [360]"))))
        self.yaw_osd_unw=np.degrees(np.unwrap(np.radians(num("OSD.yaw [360]"))))
        #segundo del CSV en que empieza el clip
        self.t0=(pd.to_datetime(fila["inicio_utc"])-t_ref).total_seconds()
        self.clip=clip
        self.vuelo=int(o["vuelo"])
        self.duracion=float(fila["duracion_s"])

    def en(self,t_clip):
        """Pose en t_clip (s, escalar o arreglo): dict con lat, lon, h, pitch, roll, yaw."""
        x=self.t0+np.atleast_1d(np.asarray(t_clip,float))
        it=lambda y:interp_con_huecos(x,self.seg,y)
        return {
            "lat":it(self.lat),"lon":it(self.lon),"h":it(self.h),
            "pitch":it(self.pitch),"roll":it(self.roll),
            "yaw":np.mod(it(self.yaw_unw),360),
            "yaw_osd":np.mod(it(self.yaw_osd_unw),360),
        }
