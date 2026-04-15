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