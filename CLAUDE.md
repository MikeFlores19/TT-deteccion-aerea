# CLAUDE.md — Contexto del proyecto para Claude Code

## Qué es este proyecto
Trabajo Terminal del IPN (TT2026-2_IA05). Sistema de detección y seguimiento
de objetos desde perspectiva aérea usando UAV.
Alumnos: Miguel Alejandro Flores Sotelo, Sergio de Jesús Castillo Molano.

## Stack técnico
- Python 3.12 + PyTorch 2.5.1 + CUDA 12.1
- Ultralytics 8.4.37
- Modelos: YOLOv8n (CNN) vs RT-DETR (CNN+Transformer)
- Dataset principal: VisDrone-DET2019 (10 clases)
- Dataset de validación cruzada: UAVDT (3 clases)
- Deployment: NVIDIA Jetson Orin Nano 8GB, TensorRT FP16

## Convenciones del proyecto
- Todo el código en src/ debe tener docstrings
- Los notebooks son solo para exploración y visualización
- Los scripts de src/ son el código limpio y reutilizable
- Las rutas a data/ siempre se leen desde configs/*.yaml
- Nunca hardcodear rutas absolutas

## Lo que NO debes modificar
- docs/protocolo/ (documentos LaTeX académicos)
- deployment/ sin confirmación explícita
- requirements.txt directamente (usar pip freeze)

## Clases de VisDrone (en orden)
0: pedestrian
1: people
2: bicycle
3: car
4: van
5: truck
6: tricycle
7: awning-tricycle
8: bus
9: motor

## Clases de UAVDT
0: car
1: truck
2: bus

## Notas importantes
- El .engine de TensorRT se genera en la Jetson, no en laptop
- Entrenamiento en laptop (RTX 4050, 6GB VRAM) con FP16 y batch pequeño para RT-DETR
- La carpeta data/ está en .gitignore por tamaño
- runs/ está en .gitignore, los resultados importantes van en results/
- UAVDT: el zip NO está descargado todavía. Las carpetas en data/uavdt/ existen pero están vacías.
- Labels de VisDrone están en formato ORIGINAL (no YOLO): bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion
  La conversión a formato YOLO se hace en la fase de preprocesamiento, después del EDA.

## Instrucciones para Claude Code
- Leer este archivo al inicio de cada sesión (automático)
- Trabajar un paso a la vez, nunca ejecutar nada sin explicarlo primero
- Antes de cada paso: explicar qué se va a hacer y por qué es relevante para el proyecto
- Mostrar el código propuesto y esperar confirmación del usuario antes de ejecutar
- Después de ejecutar: explicar qué significa el resultado en contexto del proyecto
- Si se toma una decisión técnica, explicar por qué y mencionar alternativas
- Si se encuentra algo inesperado en archivos o carpetas, reportarlo antes de actuar
- Al final de cada sesión, o cuando el usuario lo pida, actualizar la sección
  "Estado actual del proyecto" con lo que se hizo y el siguiente paso
- Nunca cerrar una sesión sin preguntar si se debe actualizar este archivo
- El usuario debe entender absolutamente todo lo que se hace porque debe
  defenderlo ante sinodales

## Estado actual del proyecto
- VisDrone-DET2019: organizado con splits train/val/test ✅
- VisDrone: conversión de labels a formato YOLO ✅ (convert_visdrone_to_yolo.py, verificada en figura 18)
- UAVDT: anotaciones procesadas parcialmente ✅ (distribución de clases + comparativa VisDrone↔UAVDT); zip de imágenes aún pendiente ⏳
- configs/visdrone.yaml y configs/uavdt.yaml: creados ✅
- EDA VisDrone: COMPLETO ✅ (18 figuras + 15 tablas en results/, notebook 01_eda_visdrone.ipynb)
- EDA UAVDT: iniciado ✅ (notebook 02_eda_uavdt.ipynb, distribución de clases)
- Preprocesamiento: notebook 03_preprocesamiento.ipynb ✅
- Entrenamiento YOLOv8n: COMPLETO ✅
  - 100 épocas, imgsz=1280, batch=4, AdamW, lr0=0.001, AMP
  - mAP@0.5=0.4735 · mAP@0.5:0.95=0.2859 · precisión=0.5795 · recall=0.4693
  - Mejor clase: car (0.853). Peores: bicycle (0.241), awning-tricycle (0.191)
  - Resultados en runs/yolov8n/yolov8n_visdrone_20260515_1815/
- Entrenamiento RT-DETR-L: EN CURSO ⏳
  - Script listo (src/training/train_rtdetr.py), pesos rtdetr-l.pt descargados
  - Entrenando en Google Colab (A100); OOM determinista en época 25, run no completado
  - runs/rtdetr/ aún vacío
- Evaluación y comparativa de modelos: pendiente ⏳
- Integración de tracking (DeepSORT / ByteTrack): pendiente ⏳ (src/tracking/ vacío)
- Streaming RTSP y georreferenciación con Pixhawk: pendiente ⏳
- Deployment en Jetson Orin Nano: pendiente ⏳ (deployment/ vacío)

### Hallazgos técnicos pendientes de atender
- configs/visdrone.yaml tiene un bloque `labels:` que Ultralytics IGNORA (deriva labels sustituyendo images/→labels/). Funcionó por coincidencia de estructura; conviene limpiarlo.
- Comentario erróneo en los scripts de train: copy_paste y mosaic NO balancean clases minoritarias a propósito (ver área 3 de la revisión).
- Pesos sueltos sin trackear bien: rtdetr-l.pt y yolov8n.pt en raíz, yolo26n.pt (¿de dónde salió?), y rtdetr-l.pt duplicado en src/training/.

## Último paso realizado
Sesión 2026-06-03:
- Se redactó por completo el documento de revisión de las 6 áreas + anexo:
  docs/revision_mejoras/revision_6puntos.{md,pdf} (fuente Markdown + PDF generado
  con docs/revision_mejoras/build_pdf.py, usando markdown + fpdf2).
- Decisiones tomadas en la revisión:
  - Área 3 (desbalance): OPCIÓN B — no se aplica rebalanceo artificial; el
    desbalance se trata como eje de análisis (reportar AP/F1 por clase). Verificado
    en código: YOLOv8 usa BCE (utils/loss.py); RT-DETR usa focal/varifocal por
    defecto (nn/tasks.py: use_vfl=True).
  - Área 4: YOLOv8n (3.16M params) vs RT-DETR-L (32.97M, backbone HGNetv2);
    asimetría ~10x reconocida como hallazgo. NO cambiar a ResNet-50.
  - Área 6: ya se hace fine-tuning (cargar .pt = ajuste desde COCO). Mejoras a
    aplicar: más épocas (YOLOv8n no se cortó en 100), ajuste de LR/cosine, SAHI en
    inferencia. NO ResNet-50.
  - Anexo: entrenar en RunPod (recomendado); Vast.ai más barato; Kaggle gratis para
    YOLOv8n. El OOM de RT-DETR en Colab es probablemente RAM de sistema, no VRAM.

## Siguiente paso
SESIÓN DE MAÑANA: iniciar la FASE DE IMPLEMENTACIÓN (meter mano al código), en un
solo paquete antes de reentrenar. Pendientes acordados:
  1. Montar entorno en RunPod.
  2. Instrumentar métricas en los scripts: agregar F1 (y FLOPs/params) a
     train_yolov8n.py y train_rtdetr.py (bloque de metrics dict). Ver
     memoria project_instrumentar_metricas_pendiente.
  3. Crear script de benchmark (p. ej. src/evaluation/benchmark.py) para
     latencia/FPS/VRAM/RAM en inferencia (laptop + Jetson).
  4. Aplicar mejoras de entrenamiento decididas (épocas/LR) y reentrenar ambos.
NOTA: el área 3 NO modifica los scripts de entrenamiento (opción B).
ALCANCE: UAVDT YA está descargado completo (train 1266/val 271/test 272, labels YOLO). Pero la
validación cruzada de generalización VisDrone->UAVDT se reserva para el TT 2da versión (TT2);
en esta versión el alcance llega hasta el punto 5, no se hace generalización.
Benchmark de eficiencia (latencia/FPS/VRAM/RAM): en TT1 se hace en la LAPTOP (RTX 4050) y PUEDE
ser notebook (es rápido, minutos, no como entrenar). En TT2 se hará en la Jetson, donde conviene
script en src/ (headless). Mantener la MISMA lógica de medición (warm-up, nº de imágenes,
batch=1) en ambos para que la comparación laptop<->Jetson sea justa.
Limpieza menor pendiente: quitar copy_paste=0.3 (código muerto sin máscaras en
detección) y los pesos sueltos (yolo26n.pt, duplicado de rtdetr-l.pt).