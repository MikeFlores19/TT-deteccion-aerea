"""
Adaptadores de detectores para el pipeline de seguimiento.

Ambos detectores devuelven el mismo contrato para que el tracker
no dependa de cual se use:
    ndarray (N,6) -> [x1,y1,x2,y2,conf,cls]

La clase se devuelve en indexacion YOLO (0-9). El desplazamiento a
indexacion VisDrone-MOT (1-10) se aplica al escribir resultados.
"""


import numpy as np
from ultralytics import YOLO, RTDETR

#imgsz fijo en 1280 igual que en la fase 2 , para que la comparacion entre detectores siga siendo justa
IMGSZ=1280

#confianza muy baja a porposito: BYtrack necesita las detecciones d ebaja confianza para us segunda asociacion. Cada tracker aplica despues su propio umbral
CONF_MINIMA=0.01



class DetectorBase:
    """Esta clase sirve para unificar la interfaz de los detectores YOLOv8 y RTDETR."""
    """Contrato comun: predict(frame) -> ndarray (N,6)."""

    def __init__(self,ruta_pesos,dispositivo="cuda:0"):
        self.ruta_pesos=str(ruta_pesos)
        self.dispositivo=dispositivo
        self.modelo=self._cargar()

    def _cargar(self):
        raise NotImplementedError

    def predict(self,frame):
        """Devuelve (N,6) [x1,y1,x2,y2,conf,cls] en pixeles absolutos."""
        salida=self.modelo.predict(
            frame,
            imgsz=IMGSZ,
            conf=CONF_MINIMA,
            device=self.dispositivo,
            verbose=False,
        )[0]

        cajas=salida.boxes
        if cajas is None or len(cajas)==0:
            return np.empty((0,6),dtype=np.float32)

        xyxy=cajas.xyxy.cpu().numpy()
        conf=cajas.conf.cpu().numpy().reshape(-1,1)
        cls=cajas.cls.cpu().numpy().reshape(-1,1)

        return np.hstack([xyxy,conf,cls]).astype(np.float32)


class YOLOv8nDetector(DetectorBase):
    def _cargar(self):
        return YOLO(self.ruta_pesos)


class RTDETRDetector(DetectorBase):
    def _cargar(self):
        return RTDETR(self.ruta_pesos)