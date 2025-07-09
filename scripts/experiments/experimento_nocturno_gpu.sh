#!/bin/bash

# ====================================================================
# 🌙 EXPERIMENTO NOCTURNO GPU - OPTIMIZACIÓN COMPLETA
# Sistema automático de optimización de hiperparámetros usando GPU
# ====================================================================

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../../optimization_results/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/experimento_nocturno_gpu_$TIMESTAMP.log"

# Crear directorio de logs
mkdir -p "$LOG_DIR"

# Función de logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# Función para manejar errores
handle_error() {
    log "❌ ERROR: $1"
    log "🔍 Revisar logs para más detalles"
    exit 1
}

# Función para mostrar progreso
show_progress() {
    echo "=================================================="
    echo "🚀 $1"
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
}

# Verificar GPU antes de empezar
check_gpu() {
    log "🔥 Verificando GPU disponible..."
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader,nounits >> "$MAIN_LOG"
    if [ $? -ne 0 ]; then
        handle_error "GPU no disponible o nvidia-smi falló"
    fi
    log "✅ GPU verificada y disponible"
}

# Inicio del experimento
show_progress "EXPERIMENTO NOCTURNO GPU INICIADO"
log "🌙🚀 Experimento nocturno GPU iniciado"
log "📁 Directorio: $SCRIPT_DIR"
log "📝 Log principal: $MAIN_LOG"

# Verificar GPU
check_gpu

# Activar ambiente conda
log "🔧 Activando ambiente conda ML-TF-G"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ML-TF-G || handle_error "No se pudo activar el ambiente conda"

# Test GPU rápido
show_progress "VERIFICACIÓN GPU Y SISTEMA"
log "🧪 Ejecutando tests de GPU y sistema..."
python test_gpu.py > "$LOG_DIR/gpu_test_$TIMESTAMP.log" 2>&1
if [ $? -ne 0 ]; then
    handle_error "Tests de GPU fallaron"
fi

python test_ml_system.py > "$LOG_DIR/system_test_$TIMESTAMP.log" 2>&1
if [ $? -ne 0 ]; then
    handle_error "Tests del sistema fallaron"
fi
log "✅ GPU y sistema verificados correctamente"

# ====================================================================
# FASE 1: OPTIMIZACIÓN INTENSIVA GPU (1-2 horas)
# ====================================================================
show_progress "FASE 1: OPTIMIZACIÓN INTENSIVA GPU (1-2 horas)"

log "🔥 Optimización XGBoost GPU (intensiva)..."
python quick_optimization.py --mode experimental --trials 300 --timeout 2400 > "$LOG_DIR/intensive_xgb_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ XGBoost GPU intensivo completado"
else
    log "⚠️ XGBoost GPU intensivo tuvo problemas, continuando..."
fi

log "💡 Optimización LightGBM GPU (intensiva)..."
python quick_optimization.py --mode experimental --trials 300 --timeout 2400 > "$LOG_DIR/intensive_lgb_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ LightGBM GPU intensivo completado"
else
    log "⚠️ LightGBM GPU intensivo tuvo problemas, continuando..."
fi

log "🐱 Optimización CatBoost GPU (intensiva)..."
python quick_optimization.py --mode experimental --trials 300 --timeout 2400 > "$LOG_DIR/intensive_cat_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ CatBoost GPU intensivo completado"
else
    log "⚠️ CatBoost GPU intensivo tuvo problemas, continuando..."
fi

# ====================================================================
# FASE 2: OPTIMIZACIÓN ULTRA-PROFUNDA (4-6 horas)
# ====================================================================
show_progress "FASE 2: OPTIMIZACIÓN ULTRA-PROFUNDA GPU (4-6 horas)"

log "🌟 Ejecutando optimización ultra-profunda GPU (500 trials, 3 horas por modelo)..."
python quick_optimization.py --mode experimental --trials 500 --timeout 10800 > "$LOG_DIR/ultra_deep_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Optimización ultra-profunda GPU completada"
else
    log "⚠️ Optimización ultra-profunda GPU tuvo problemas, continuando..."
fi

# ====================================================================
# FASE 3: OPTIMIZACIÓN EXTREMA NOCTURNA (6-8 horas)
# ====================================================================
show_progress "FASE 3: OPTIMIZACIÓN EXTREMA NOCTURNA GPU (6-8 horas)"

log "🚀 Ejecutando optimización extrema nocturna (1000 trials, 4 horas por modelo)..."
python quick_optimization.py --mode experimental --trials 1000 --timeout 14400 > "$LOG_DIR/extreme_nocturna_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Optimización extrema nocturna completada"
else
    log "⚠️ Optimización extrema nocturna tuvo problemas, continuando..."
fi

# ====================================================================
# FASE 4: ANÁLISIS Y PROCESAMIENTO DE RESULTADOS
# ====================================================================
show_progress "FASE 4: ANÁLISIS DE RESULTADOS GPU"

log "📊 Ejecutando análisis completo de resultados..."
python optuna_results_analyzer.py > "$LOG_DIR/results_analysis_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Análisis de resultados completado"
else
    log "⚠️ Análisis tuvo problemas, continuando..."
fi

log "🔗 Integrando mejores parámetros optimizados por GPU..."
python integrate_optimized_params.py > "$LOG_DIR/integration_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Integración completada"
else
    log "⚠️ Integración tuvo problemas, continuando..."
fi

# ====================================================================
# FASE 5: VALIDACIÓN FINAL GPU
# ====================================================================
show_progress "FASE 5: VALIDACIÓN FINAL GPU"

log "🎯 Ejecutando entrenamiento final con mejores parámetros GPU..."
timeout 3600 python crypto_ml_trainer_optimized.py > "$LOG_DIR/final_training_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Entrenamiento final GPU completado exitosamente"
else
    log "⚠️ Entrenamiento final GPU tuvo timeout o problemas"
fi

# ====================================================================
# MONITOREO GPU FINAL
# ====================================================================
show_progress "MONITOREO GPU FINAL"

log "📊 Estado final de GPU:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits >> "$MAIN_LOG"

# ====================================================================
# RESUMEN FINAL GPU
# ====================================================================
show_progress "EXPERIMENTO NOCTURNO GPU COMPLETADO"

END_TIME=$(date +"%Y%m%d_%H%M%S")
log "🎉🚀 Experimento nocturno GPU finalizado"
log "⏰ Tiempo de finalización: $(date '+%Y-%m-%d %H:%M:%S')"

# Generar resumen especializado para GPU
SUMMARY_FILE="$LOG_DIR/resumen_experimento_gpu_$TIMESTAMP.txt"
cat > "$SUMMARY_FILE" << EOF
🌙🚀 RESUMEN DEL EXPERIMENTO NOCTURNO GPU
=========================================

⏰ Inicio: $TIMESTAMP
⏰ Fin: $END_TIME

🚀 GPU UTILIZADA:
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)

📁 Logs generados:
$(ls -la "$LOG_DIR"/*gpu*$TIMESTAMP.* 2>/dev/null | tail -10)

📊 Archivos de resultados:
$(ls -la ../../optimization_results/*.json 2>/dev/null | tail -5)

🎯 VENTAJAS DE GPU OBTENIDAS:
- Optimización más rápida con datasets grandes
- Mayor paralelización en cross-validation
- Mejor exploración del espacio de hiperparámetros
- Entrenamiento acelerado de modelos complejos

📈 Para usar resultados:
1. python crypto_ml_trainer_optimized.py
2. Revisar integration_report.md
3. Los modelos ya están configurados para GPU

📝 Logs principales GPU:
- GPU Test: $LOG_DIR/gpu_test_$TIMESTAMP.log
- Intensivo GPU: $LOG_DIR/intensive_*_gpu_$TIMESTAMP.log
- Ultra-profundo: $LOG_DIR/ultra_deep_gpu_$TIMESTAMP.log
- Extremo nocturno: $LOG_DIR/extreme_nocturna_gpu_$TIMESTAMP.log
- Entrenamiento final: $LOG_DIR/final_training_gpu_$TIMESTAMP.log

🎉 Experimento GPU completado!
EOF

log "📄 Resumen GPU guardado en: $SUMMARY_FILE"

# Mostrar estadísticas finales
echo ""
echo "🌟🚀 ============================================="
echo "🌟🚀 EXPERIMENTO NOCTURNO GPU COMPLETADO"
echo "🌟🚀 ============================================="
echo ""
echo "📊 GPU Final Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
echo ""
echo "📂 Archivos generados:"
ls -la "$LOG_DIR"/*gpu*$TIMESTAMP.* 2>/dev/null || echo "No se encontraron logs GPU"
echo ""
echo "🎯 Próximos pasos:"
echo "1. Revisar: cat $SUMMARY_FILE"
echo "2. Analizar: python optuna_results_analyzer.py"
echo "3. Usar optimizados GPU: python crypto_ml_trainer_optimized.py"
echo ""
echo "✅🚀 ¡Todo listo para revisar los resultados GPU!"

log "🌟🚀 Script de experimento nocturno GPU finalizado exitosamente"
