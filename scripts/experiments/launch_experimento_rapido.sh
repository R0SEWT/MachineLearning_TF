#!/bin/bash

# ====================================================================
# 🚀 LAUNCHER EXPERIMENTO RÁPIDO GPU
# Lanzador simple para el experimento rápido
# ====================================================================

echo "⚡🚀 LAUNCHER EXPERIMENTO RÁPIDO GPU"
echo "=================================="
echo ""

# Verificar que el script existe
SCRIPT_PATH="/home/exodia/Documentos/MachineLearning_TF/scripts/experiments/experimento_rapido_gpu.sh"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Script no encontrado en $SCRIPT_PATH"
    exit 1
fi

echo "📍 Script encontrado: $SCRIPT_PATH"
echo "⏱️ Duración estimada: 30-45 minutos"
echo "🎯 Modo: Optimización rápida con GPU"
echo ""

# Preguntar confirmación
read -p "¿Iniciar experimento rápido GPU? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Iniciando experimento rápido GPU..."
    echo "📝 Logs se guardarán en: logs/experimento_rapido_gpu_*"
    echo "📊 Resultados en: optimization_results/"
    echo ""
    
    # Ejecutar el script
    "$SCRIPT_PATH"
    
    echo ""
    echo "✅ Experimento completado!"
    echo "📄 Revisar logs y resultados en los directorios mencionados"
else
    echo "❌ Experimento cancelado"
fi
