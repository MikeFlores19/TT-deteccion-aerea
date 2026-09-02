"""
Adaptadores de trackers para el pipeline de seguimiento.

Ambos trackers reciben y devuelven el mismo contrato para poder
intercambiarlos sin tocar el resto del codigo:
    entrada: ndarray (N,6) -> [x1,y1,x2,y2,conf,cls]
    salida:  ndarray (M,7) -> [x1,y1,x2,y2,id,conf,cls]

Las librerias subyacentes son incompatibles entre si:
  boxmot         trabaja con ndarray y devuelve TrackResults (M,8)
  deep-sort-realtime  trabaja con listas de tuplas y devuelve objetos Track
Esa diferencia queda encapsulada aqui.
"""

import numpy as np


class TrackerBase:
    """Contrato comun: update(dets,frame) -> ndarray (M,7)."""

    def update(self,dets,frame):
        raise NotImplementedError

    @staticmethod
    def _vacio():
        """Salida sin tracks, con la forma correcta."""
        return np.empty((0,7),dtype=np.float32)


class ByteTrackAdapter(TrackerBase):
    """ByteTrack de boxmot. Kalman + IoU, sin Re-ID."""

    def __init__(self,min_conf=0.1,track_thresh=0.45,match_thresh=0.8,
                 track_buffer=25,frame_rate=30):
        from boxmot.trackers.bbox.bytetrack import ByteTrack
        self.tracker=ByteTrack(
            min_conf=min_conf,
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
        )

    def update(self,dets,frame):
        # boxmot ya acepta el contrato (N,6) tal cual
        res=self.tracker.update(dets,frame)
        if res is None or len(res)==0:
            return self._vacio()

        # TrackResults trae (M,8): x1,y1,x2,y2,id,conf,cls,det_ind
        # se descarta det_ind para quedarse con el contrato (M,7)
        return np.asarray(res,dtype=np.float32)[:,:7]


class DeepSortAdapter(TrackerBase):
    """DeepSORT de deep-sort-realtime. Kalman + IoU + Re-ID MobileNet."""

    def __init__(self,max_age=30,n_init=3,max_iou_distance=0.7,
                 max_cosine_distance=0.2,embedder="mobilenet",
                 embedder_gpu=True,conf_minima=0.5):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        # DeepSORT no tiene segunda asociacion: filtra las detecciones
        # debiles antes de entrar, a diferencia de ByteTrack
        self.conf_minima=conf_minima
        self.tracker=DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            embedder=embedder,
            embedder_gpu=embedder_gpu,
        )

    def update(self,dets,frame):
        if len(dets)==0:
            return self._vacio()

        # aplica su propio umbral sobre las mismas detecciones cacheadas
        dets=dets[dets[:,4]>=self.conf_minima]
        if len(dets)==0:
            return self._vacio()

        # contrato (N,6) -> lista de tuplas ([left,top,w,h],conf,clase)
        entrada=[]
        for x1,y1,x2,y2,conf,cls in dets:
            entrada.append(([float(x1),float(y1),float(x2-x1),float(y2-y1)],
                            float(conf),int(cls)))

        tracks=self.tracker.update_tracks(entrada,frame=frame)

        filas=[]
        for t in tracks:
            # los tentativos aun no estan confirmados por n_init frames
            if not t.is_confirmed():
                continue
            x1,y1,x2,y2=t.to_ltrb()
            conf=t.get_det_conf()
            cls=t.get_det_class()
            # un track puede predecirse sin deteccion asociada este frame
            conf=0.0 if conf is None else conf
            cls=-1 if cls is None else cls
            filas.append([x1,y1,x2,y2,int(t.track_id),conf,cls])

        if not filas:
            return self._vacio()
        return np.asarray(filas,dtype=np.float32)