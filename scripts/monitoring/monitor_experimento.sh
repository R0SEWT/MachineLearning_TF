#!/bin/bash

# 🔍 MONITOR DEL EXPERIMENTO NOCTURNO
# Script para monitorear el progreso del experimento

echo "🌙 MONITOR DEL EXPERIMENTO NOCTURNO"
echo "=================================="

# Verificar si está ejecutándose
PROC_COUNT=$(ps aux | grep -v grep | grep experimento_nocturno | wc -l)
if [ $PROC_COUNT -gt 0 ]; then
    echo "✅ Experimento ACTIVO"
    echo "🕐 Proceso iniciado: $(ps -o lstart= -p $(pgrep -f experimento_nocturno))"
else
    echo "❌ Experimento NO está ejecutándose"
fi

echo ""
echo "📂 LOGS RECIENTES:"
echo "=================="

# Mostrar logs más recientes
LOG_DIR="../../optimization_results/logs"
if [ -d "$LOG_DIR" ]; then
    echo "📝 Log principal más reciente:"
    ls -t "$LOG_DIR"/experimento_nocturno_*.log 2>/dev/null | head -1 | xargs tail -10 2>/dev/null || echo "No hay logs aún"
    
    echo ""
    echo "📊 Últimas líneas del output:"
    tail -10 experimento_nocturno_output.log 2>/dev/null || echo "No hay output aún"
    
    echo ""
    echo "📁 Archivos de log disponibles:"
    ls -lt "$LOG_DIR"/*$(date +%Y%m%d)* 2>/dev/null | head -5 || echo "No hay logs de hoy"
else
    echo "❌ Directorio de logs no encontrado"
fi

echo ""
echo "🎯 PARA MÁS DETALLES:"
echo "==================="
echo "tail -f experimento_nocturno_output.log  # Ver output en tiempo real"
echo "tail -f ../../optimization_results/logs/experimento_nocturno_*.log  # Ver log principal"
echo "./monitor_experimento.sh  # Ejecutar este script de nuevo"

echo ""
echo "🛑 PARA DETENER:"
echo "==============="
echo "pkill -f experimento_nocturno  # Detener proceso"
