#!/bin/bash

# ====================================================================
# 📊 MONITOR EXPERIMENTO RÁPIDO GPU
# Script para monitorear el progreso del experimento
# ====================================================================

LOG_DIR="/home/exodia/Documentos/MachineLearning_TF/logs"
RESULTS_DIR="/home/exodia/Documentos/MachineLearning_TF/optimization_results"

clear
echo "📊 MONITOR EXPERIMENTO RÁPIDO GPU"
echo "================================="
echo ""

# Función para mostrar logs recientes
show_recent_logs() {
    echo "📝 LOGS RECIENTES:"
    echo "===================="
    ls -la "$LOG_DIR"/*rapido*gpu*.log 2>/dev/null | tail -5
    echo ""
}

# Función para mostrar estado GPU
show_gpu_status() {
    echo "🔥 ESTADO GPU:"
    echo "=============="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | head -1
    else
        echo "GPU no disponible"
    fi
    echo ""
}

# Función para mostrar progreso
show_progress() {
    echo "⚡ PROGRESO EXPERIMENTO:"
    echo "======================="
    
    # Buscar el log más reciente
    LATEST_LOG=$(ls -t "$LOG_DIR"/*rapido*gpu*.log 2>/dev/null | head -1)
    
    if [ -f "$LATEST_LOG" ]; then
        echo "📄 Último log: $(basename "$LATEST_LOG")"
        echo ""
        echo "🔍 Últimas 10 líneas:"
        tail -10 "$LATEST_LOG" | sed 's/\x1b\[[0-9;]*m//g'  # Remover colores ANSI
    else
        echo "❌ No se encontraron logs del experimento rápido"
    fi
    echo ""
}

# Función para mostrar resultados
show_results() {
    echo "📊 RESULTADOS DISPONIBLES:"
    echo "=========================="
    ls -la "$RESULTS_DIR"/*.json 2>/dev/null | tail -3
    echo ""
}

# Función para mostrar resumen si existe
show_summary() {
    SUMMARY_FILE=$(ls -t "$LOG_DIR"/resumen_experimento_rapido_gpu_*.txt 2>/dev/null | head -1)
    if [ -f "$SUMMARY_FILE" ]; then
        echo "📄 RESUMEN DISPONIBLE:"
        echo "====================="
        echo "📎 Archivo: $(basename "$SUMMARY_FILE")"
        echo ""
        echo "🔍 Contenido:"
        cat "$SUMMARY_FILE"
        echo ""
    fi
}

# Función para monitoreo continuo
monitor_continuous() {
    echo "🔄 MONITOREO CONTINUO INICIADO"
    echo "Presiona Ctrl+C para salir"
    echo ""
    
    while true; do
        clear
        echo "📊 MONITOR EXPERIMENTO RÁPIDO GPU - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"
        echo ""
        
        show_gpu_status
        show_progress
        show_recent_logs
        
        echo "🔄 Actualizando en 30 segundos..."
        sleep 30
    done
}

# Menú principal
while true; do
    echo "🎯 OPCIONES DE MONITOREO:"
    echo "1. 📝 Ver logs recientes"
    echo "2. 🔥 Estado GPU"
    echo "3. ⚡ Progreso actual"
    echo "4. 📊 Resultados"
    echo "5. 📄 Resumen (si disponible)"
    echo "6. 🔄 Monitoreo continuo"
    echo "7. 🚪 Salir"
    echo ""
    read -p "Selecciona una opción (1-7): " -n 1 -r
    echo ""
    echo ""
    
    case $REPLY in
        1)
            show_recent_logs
            ;;
        2)
            show_gpu_status
            ;;
        3)
            show_progress
            ;;
        4)
            show_results
            ;;
        5)
            show_summary
            ;;
        6)
            monitor_continuous
            ;;
        7)
            echo "👋 Saliendo del monitor..."
            exit 0
            ;;
        *)
            echo "❌ Opción no válida"
            ;;
    esac
    
    echo ""
    read -p "Presiona Enter para continuar..."
    clear
done
