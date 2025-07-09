#!/bin/bash

# 🔍🚀 MONITOR GPU DEL EXPERIMENTO NOCTURNO
# Script para monitorear el progreso del experimento GPU

echo "🌙🚀 MONITOR DEL EXPERIMENTO NOCTURNO GPU"
echo "========================================"

# Verificar si está ejecutándose
PROC_COUNT=$(ps aux | grep -v grep | grep experimento_nocturno_gpu | wc -l)
if [ $PROC_COUNT -gt 0 ]; then
    echo "✅ Experimento GPU ACTIVO"
    echo "🕐 Proceso iniciado: $(ps -o lstart= -p $(pgrep -f experimento_nocturno_gpu) 2>/dev/null)"
else
    echo "❌ Experimento GPU NO está ejecutándose"
fi

echo ""
echo "🚀 ESTADO ACTUAL DE GPU:"
echo "======================="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
while read line; do
    echo "   🔥 GPU: $line"
done

echo ""
echo "📈 PROCESOS USANDO GPU:"
echo "====================="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | \
while read line; do
    echo "   ⚡ $line"
done || echo "   📊 No hay procesos usando GPU actualmente"

echo ""
echo "📂 LOGS RECIENTES GPU:"
echo "===================="

# Mostrar logs más recientes
LOG_DIR="../../optimization_results/logs"
if [ -d "$LOG_DIR" ]; then
    echo "📝 Log principal GPU más reciente:"
    ls -t "$LOG_DIR"/experimento_nocturno_gpu_*.log 2>/dev/null | head -1 | xargs tail -10 2>/dev/null || echo "No hay logs GPU aún"
    
    echo ""
    echo "📊 Últimas líneas del output GPU:"
    tail -10 experimento_nocturno_gpu_output.log 2>/dev/null || echo "No hay output GPU aún"
    
    echo ""
    echo "📁 Archivos de log GPU disponibles:"
    ls -lt "$LOG_DIR"/*gpu*$(date +%Y%m%d)* 2>/dev/null | head -5 || echo "No hay logs GPU de hoy"
    
    echo ""
    echo "🎯 ÚLTIMA ACTIVIDAD GPU:"
    echo "======================="
    find "$LOG_DIR" -name "*gpu*$(date +%Y%m%d)*" -newer /tmp/gpu_check 2>/dev/null | head -3 | \
    while read file; do
        echo "   📄 $(basename "$file"): $(tail -2 "$file" 2>/dev/null | head -1)"
    done
    touch /tmp/gpu_check
else
    echo "❌ Directorio de logs no encontrado"
fi

echo ""
echo "🎯 COMANDOS ÚTILES GPU:"
echo "====================="
echo "tail -f experimento_nocturno_gpu_output.log  # Ver output GPU en tiempo real"
echo "tail -f ../../optimization_results/logs/experimento_nocturno_gpu_*.log  # Ver log principal GPU"
echo "watch -n 5 nvidia-smi  # Monitorear GPU cada 5 segundos"
echo "./monitor_experimento_gpu.sh  # Ejecutar este script de nuevo"

echo ""
echo "🛑 PARA DETENER:"
echo "==============="
echo "pkill -f experimento_nocturno_gpu  # Detener proceso GPU"

echo ""
echo "⚡ ESTADÍSTICAS RÁPIDAS GPU:"
echo "=========================="
if [ -f "../../optimization_results/logs/experimento_nocturno_gpu_"*".log" ]; then
    echo "🔍 Fases completadas:"
    grep -c "FASE.*completado" ../../optimization_results/logs/experimento_nocturno_gpu_*.log 2>/dev/null | tail -1 || echo "   🤔 Información no disponible"
    echo "🏆 Modelos optimizados:"
    grep -c "completado" ../../optimization_results/logs/experimento_nocturno_gpu_*.log 2>/dev/null | tail -1 || echo "   🤔 Información no disponible"
fi
