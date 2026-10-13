# Fase 5 - Tabla final (config A: YOLOv8n + ByteTrack)

Laptop: RTX 4050 (.pt). Jetson: Orin Nano 8GB, MAXN_SUPER, TensorRT FP16 736x1280.


## Detección (VisDrone-DET val, 548 img, 10 clases)

| Métrica | Laptop | Jetson | Meta | Cumple | Nota |
|---|---|---|---|---|---|
| mAP50 (Jetson: TRT FP16 1280x1280) | 0.4771 | 0.4725 (-1.0%) |  |  | cross_eval/yolov8n__visdrone_val10*.json |
| mAP50-95 (Jetson: TRT FP16 1280x1280) | 0.2891 | 0.2881 (-0.3%) |  |  | cross_eval/yolov8n__visdrone_val10*.json |
| mAP50 del .pt en la Jetson con el mismo letterbox que el engine |  | 0.4727 |  |  | la diferencia viene del letterbox, no de FP16 |

## Seguimiento (VisDrone-MOT test-dev, 17 sec., 5 clases; Jetson: TRT 736x1280)

| Métrica | Laptop | Jetson | Meta | Cumple | Nota |
|---|---|---|---|---|---|
| MOTA (%) | 41.12 | 40.46 | >= 40 | sí | tracking/test-dev/<clase>/<config>/ |
| IDF1 (%) | 57.69 | 57.61 |  |  |  |
| HOTA (%) | 46.33 | 46.25 |  |  |  |
| IDSW (suma 5 clases) | 2652 | 2677 |  |  |  |
| MOTA con engine 1280x1280 (descartado) |  | 39.93 | >= 40 | no | letterbox cuadrado: por eso se uso 736x1280 |
| MOTA / IDF1 / HOTA leyendo .mp4 (pipeline real) |  | 40.79 / 57.18 / 46.05 |  |  |  |

## Velocidad config A (frames en RAM; Jetson MAXN_SUPER, TRT 736x1280)

| Métrica | Laptop | Jetson | Meta | Cumple | Nota |
|---|---|---|---|---|---|
| FPS det+track, secuencia densa (~215 obj/frame) | 55.6 | 30.2 (sin jetson_clocks 15.7) |  |  | eficiencia/ y eficiencia_jetson/ |
| FPS det+track, secuencia dispersa 4K (~102 obj/frame) | 106.0 | 46.0 (sin jetson_clocks 20.8) |  |  |  |
| Detector YOLOv8n: pre+inferencia+post (ms / FPS) |  | 16.6 ms / 60 FPS | >= 30 FPS | sí | secuencia densa, jetson_clocks |
| Inferencia del modelo (ms) | 5.95 (.pt, 1280) | 6.81 (TRT FP16, 736x1280) |  |  | la laptop se midio en la Fase 2 (.pt) |
| Tracking ByteTrack, secuencia densa (ms) | 6.14 | 16.42 |  |  |  |
| FPS leyendo .mp4 (17 videos de test-dev) |  | 22.7 (sin jetson_clocks 13.2) |  |  |  |

## Recursos en la Jetson (jetson_clocks)

| Métrica | Laptop | Jetson | Meta | Cumple | Nota |
|---|---|---|---|---|---|
| Potencia media (W): densa / dispersa / .mp4 / reposo |  | 9.0 / 10.5 / 8.7 / 5.4 |  |  |  |
| Energía por frame (J): densa / dispersa / .mp4 |  | 0.298 / 0.228 / 0.387 |  |  |  |
| Temperatura máxima Tj (°C) |  | 56.0 |  |  |  |
| Memoria GPU del modelo (MB) | 93.1 (VRAM total del proceso) | 25.5 (torch max) |  |  |  |
| RAM del sistema (MB): densa / dispersa / .mp4 |  | 2449 / 4230 / 1587 de 8 GB |  |  |  |

## Campo: 27 videos propios del DJI Mini 3 (Jetson)

| Métrica | Laptop | Jetson | Meta | Cumple | Nota |
|---|---|---|---|---|---|
| Videos / duración / frames |  | 27 / 33.1 min / 62,383 |  |  |  |
| Detecciones con ID (tracks-frame) |  | 1,143,900 |  |  |  |
| IDs únicos / eventos |  | 9,899 / 8,228 (car 49.5%, pedestrian 37.7%) |  |  |  |
| FPS del pipeline det+track (media) |  | 34.4 (18/27 videos >= 30) |  |  |  |
| FPS total con lectura y escritura del video anotado |  | 17.5 |  |  |  |

## Georreferenciación de los videos propios

| Métrica | Laptop | Jetson | Meta | Cumple | Nota |
|---|---|---|---|---|---|
| Sincronización A (original vs registro de vuelo) |  | 14 originales; |A-isVideo| <= 0.64 s; RMSE H <= 0.05 m |  |  |  |
| Sincronización B (clip vs original) |  | 27/27 OK; residual 1-6 ms; velocidad 1.0000-1.0000 |  |  |  |
| Validación sintética (verdad conocida) |  | 6/6 casos pasan |  |  |  |
| Detecciones georreferenciadas |  | 710,062/1,143,900 (62.1%) |  |  |  |
| Objetos georreferenciados |  | 5,040 |  |  |  |
| Error estimado mediano por clip (m) |  | 2.5-8.2 | <= 10 por detección | sí | propagacion: pitch 1 deg, rumbo 3 deg, altura 0.5 m, GPS 1.5 m |
| Dispersión por objeto, mediana (m) |  | 0.66 |  |  | repetibilidad, no exactitud |
| Zona válida de la imagen (%) |  | 52-98 |  |  | camara inclinada (gimbal -10 a -26 deg): vista oblicua |
