# TT2026-2_IA05 — Estado del Proyecto (8 junio 2026)

## ✅ COMPLETADO

### Entrenamientos
- **YOLOv8n mosaic=1.0**: mAP50=0.4755, F1=0.5212 (4.476h en RunPod A40)
- **YOLOv8n mosaic=0.5**: mAP50=0.4708 (experimento, peor resultado)
- **RT-DETR-L mosaic=1.0**: Entrenamiento terminado (100 épocas), pesos descargados

### Eficiencia (laptop RTX 4050)
- **YOLOv8n**: 4.41ms latencia, 226.58 FPS, 72.8MB VRAM → listo para reporte técnico
- **RT-DETR**: pendiente de medir (cuando termine la presentación posterior)

### Código
- `src/evaluation/efficiency_profiling.py` → creado y funcional
- Todos los results/tables/ y runs/rtdetr/ en GitHub

## 🔄 EN PROGRESO

- RT-DETR entrenamiento terminado, pesos en GitHub

## ⏳ PENDIENTE

1. Correr efficiency_profiling.py en Jetson Orin Nano (TensorRT FP16)
2. Completar reporte técnico con métricas de YOLOv8n
3. RT-DETR a presentación posterior (no en reporte técnico actual)
4. Inference evaluation en Jetson (ByteTrack)

## Notas

- Bug Ultralytics 8.4.60 resuelto con 8.4.37
- mosaic=1.0 confirmado mejor que 0.5
- Git LFS no usado (archivos ~63MB, GitHub lo permite con warning)
