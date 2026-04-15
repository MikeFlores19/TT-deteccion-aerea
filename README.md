# TT2026 — Detección de objetos desde perspectiva aérea

**Trabajo Terminal No. TT2026-2_IA05**  
Instituto Politécnico Nacional — UPIIT

## Alumnos
- Miguel Alejandro Flores Sotelo
- Sergio de Jesús Castillo Molano

## Descripción
Sistema de detección y seguimiento de objetos desde perspectiva aérea
mediante UAV, comparando YOLOv8n (CNN) vs RT-DETR (CNN+Transformer),
entrenado sobre VisDrone y validado sobre UAVDT.
Desplegado en NVIDIA Jetson Orin Nano 8GB con TensorRT FP16.

## Stack
- Python 3.12
- PyTorch 2.5.1 + CUDA 12.1
- Ultralytics 8.4.37
- Datasets: VisDrone-DET2019, UAVDT
- Deployment: NVIDIA Jetson Orin Nano 8GB

## Setup
\```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\```

## Estructura
- `src/` — código fuente
- `notebooks/` — EDA y experimentos
- `configs/` — configuración de modelos y datasets
- `deployment/` — scripts para Jetson
- `docs/` — documento terminal y protocolo