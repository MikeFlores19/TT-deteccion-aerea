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
- UAVDT: zip no descargado todavía, carpetas vacías ⏳
- configs/visdrone.yaml: creado ✅
- EDA VisDrone — Paso 1 (parseo de anotaciones → DataFrame): EJECUTADO ✅
  - 343,171 detecciones válidas en train
  - 6,470 imágenes únicas
  - 10 clases presentes
  - Pendiente: el usuario está revisando y entendiendo el código del Paso 1
- EDA VisDrone — Pasos 2-8: pendiente ⏳
- EDA UAVDT: pendiente ⏳
- Preprocesamiento y augmentación: pendiente ⏳
- Entrenamiento YOLOv8n: pendiente ⏳
- Entrenamiento RT-DETR: pendiente ⏳
- Evaluación y comparativa de modelos: pendiente ⏳
- Integración de tracking (DeepSORT / ByteTrack): pendiente ⏳
- Deployment en Jetson Orin Nano: pendiente ⏳

## Último paso realizado
Sesión 2026-05-06:
- Revisión a fondo del Protocolo TT y del template de tesis (Tesis_UPIIT.pdf)
- Confirmación del estado real de los datasets (VisDrone organizado, UAVDT vacío)
- Hallazgo clave: labels de VisDrone en formato original, no YOLO
- Creación de configs/visdrone.yaml
- Creación de notebooks/01_eda_visdrone.ipynb
- Ejecución del Paso 1 del EDA: parseo de anotaciones → DataFrame (343,171 detecciones)
- Explicación línea por línea del código del Paso 1 al usuario

## Siguiente paso
Continuar con el Paso 2 del EDA: distribución de clases.
El usuario ya entiende el Paso 1. Al iniciar la siguiente sesión, retomar desde
el Paso 2 en notebooks/01_eda_visdrone.ipynb.