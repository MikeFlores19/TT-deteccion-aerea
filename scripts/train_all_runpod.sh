#!/bin/bash
set -e

echo "=================================================="
echo " Entrenamiento secuencial en RunPod"
echo " 1) YOLOv8n"
echo " 2) RT-DETR"
echo "=================================================="

cd /workspace/TT-deteccion-aerea

echo "Activando entorno virtual..."
source .venv/bin/activate

echo "Verificando GPU..."
nvidia-smi

echo "=================================================="
echo " Iniciando entrenamiento YOLOv8n"
echo "=================================================="
python src/training/train_yolov8n.py

echo "=================================================="
echo " YOLOv8n terminado correctamente"
echo " Iniciando entrenamiento RT-DETR"
echo "=================================================="
python src/training/train_rtdetr.py

echo "=================================================="
echo " Entrenamientos terminados correctamente"
echo " Resultados guardados en:"
echo " runs/yolov8n/"
echo " runs/rtdetr/"
echo "=================================================="
