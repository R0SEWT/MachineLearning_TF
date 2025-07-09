#!/bin/bash

# ====================================================================
# 🌙 EXPERIMENTO NOCTURNO DE OPTIMIZACIÓN COMPLETA
# Sistema automático de optimización de hiperparámetros para criptomonedas
# ====================================================================

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
LOG_DIR="$PROJECT_ROOT/optimization_results/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/experimento_nocturno_$TIMESTAMP.log"

# Crear directorio de logs
mkdir -p "$LOG_DIR"

# Cambiar al directorio raíz del proyecto
cd "$PROJECT_ROOT"

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

# Inicio del experimento
show_progress "INICIANDO EXPERIMENTO NOCTURNO DE OPTIMIZACIÓN"
log "🌙 Experimento nocturno iniciado"
log "📁 Directorio: $SCRIPT_DIR"
log "📝 Log principal: $MAIN_LOG"

# Activar ambiente conda
log "🔧 Activando ambiente conda ML-TF-G"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ML-TF-G || handle_error "No se pudo activar el ambiente conda"

# Verificar sistema antes de empezar
show_progress "VERIFICACIÓN DEL SISTEMA"
log "🧪 Ejecutando tests del sistema..."
python tests/test_ml_system.py > "$LOG_DIR/system_test_$TIMESTAMP.log" 2>&1
if [ $? -ne 0 ]; then
    handle_error "Tests del sistema fallaron"
fi
log "✅ Sistema verificado correctamente"

# ====================================================================
# FASE 1: OPTIMIZACIÓN RÁPIDA DE TODOS LOS MODELOS (30-45 min)
# ====================================================================
show_progress "FASE 1: OPTIMIZACIÓN RÁPIDA (30-45 min estimado)"

log "🔥 Optimizando XGBoost (rápido)..."
python scripts/optimization/quick_optimization.py --mode quick-xgb --trials 50 --timeout 900 > "$LOG_DIR/quick_xgb_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ XGBoost rápido completado"
else
    log "⚠️ XGBoost rápido tuvo problemas, continuando..."
fi

log "💡 Optimizando LightGBM (rápido)..."
python scripts/optimization/quick_optimization.py --mode quick-lgb --trials 50 --timeout 900 > "$LOG_DIR/quick_lgb_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ LightGBM rápido completado"
else
    log "⚠️ LightGBM rápido tuvo problemas, continuando..."
fi

log "🐱 Optimizando CatBoost (rápido)..."
python scripts/optimization/quick_optimization.py --mode quick-cat --trials 50 --timeout 900 > "$LOG_DIR/quick_cat_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ CatBoost rápido completado"
else
    log "⚠️ CatBoost rápido tuvo problemas, continuando..."
fi

# ====================================================================
# FASE 2: OPTIMIZACIÓN ESTÁNDAR (1-2 horas)
# ====================================================================
show_progress "FASE 2: OPTIMIZACIÓN ESTÁNDAR (1-2 horas estimado)"

log "🚀 Ejecutando optimización estándar (100 trials, 1 hora por modelo)..."
python scripts/optimization/quick_optimization.py --mode full --trials 100 --timeout 3600 > "$LOG_DIR/optimization_standard_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Optimización estándar completada"
else
    log "⚠️ Optimización estándar tuvo problemas, continuando..."
fi

# ====================================================================
# FASE 3: OPTIMIZACIÓN EXPERIMENTAL (3-6 horas)
# ====================================================================
show_progress "FASE 3: OPTIMIZACIÓN EXPERIMENTAL (3-6 horas estimado)"

log "🔬 Ejecutando optimización experimental (200 trials, 2 horas por modelo)..."
python quick_optimization.py --mode experimental --trials 200 --timeout 7200 > "$LOG_DIR/optimization_experimental_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Optimización experimental completada"
else
    log "⚠️ Optimización experimental tuvo problemas, continuando..."
fi

# ====================================================================
# FASE 4: OPTIMIZACIÓN ULTRA-EXTENSIVA (4-8 horas)
# ====================================================================
show_progress "FASE 4: OPTIMIZACIÓN ULTRA-EXTENSIVA (4-8 horas estimado)"

log "🌟 Ejecutando optimización ultra-extensiva (500 trials, 4 horas por modelo)..."
python quick_optimization.py --mode experimental --trials 500 --timeout 14400 > "$LOG_DIR/optimization_ultra_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Optimización ultra-extensiva completada"
else
    log "⚠️ Optimización ultra-extensiva tuvo problemas, continuando..."
fi

# ====================================================================
# FASE 5: ANÁLISIS Y PROCESAMIENTO DE RESULTADOS
# ====================================================================
show_progress "FASE 5: ANÁLISIS DE RESULTADOS"

log "📊 Ejecutando análisis completo de resultados..."
python optuna_results_analyzer.py > "$LOG_DIR/results_analysis_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Análisis de resultados completado"
else
    log "⚠️ Análisis tuvo problemas, continuando..."
fi

log "🔗 Integrando mejores parámetros..."
python integrate_optimized_params.py > "$LOG_DIR/integration_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Integración completada"
else
    log "⚠️ Integración tuvo problemas, continuando..."
fi

# ====================================================================
# FASE 6: VALIDACIÓN FINAL
# ====================================================================
show_progress "FASE 6: VALIDACIÓN FINAL"

log "🎯 Ejecutando entrenamiento con mejores parámetros..."
timeout 1800 python crypto_ml_trainer_optimized.py > "$LOG_DIR/final_training_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log "✅ Entrenamiento final completado exitosamente"
else
    log "⚠️ Entrenamiento final tuvo timeout o problemas"
fi

# ====================================================================
# RESUMEN FINAL
# ====================================================================
show_progress "EXPERIMENTO NOCTURNO COMPLETADO"

END_TIME=$(date +"%Y%m%d_%H%M%S")
log "🎉 Experimento nocturno finalizado"
log "⏰ Tiempo de finalización: $(date '+%Y-%m-%d %H:%M:%S')"

# Generar resumen
SUMMARY_FILE="$LOG_DIR/resumen_experimento_$TIMESTAMP.txt"
cat > "$SUMMARY_FILE" << EOF
🌙 RESUMEN DEL EXPERIMENTO NOCTURNO
==================================

⏰ Inicio: $TIMESTAMP
⏰ Fin: $END_TIME

📁 Logs generados:
$(ls -la "$LOG_DIR"/*_$TIMESTAMP.* 2>/dev/null | tail -10)

📊 Archivos de resultados:
$(ls -la ../../optimization_results/*.json 2>/dev/null | tail -5)

🎯 Para revisar resultados:
1. python optuna_results_analyzer.py
2. Revisar visualizaciones en: ../../optimization_results/analysis_visualizations/
3. Mejores configuraciones en: ../../optimization_results/best_configs_*.json

📈 Para usar resultados:
1. python crypto_ml_trainer_optimized.py
2. Revisar integration_report.md

📝 Logs principales:
- Sistema: $LOG_DIR/system_test_$TIMESTAMP.log
- Optimización: $LOG_DIR/optimization_*_$TIMESTAMP.log
- Análisis: $LOG_DIR/results_analysis_$TIMESTAMP.log
- Integración: $LOG_DIR/integration_$TIMESTAMP.log
- Entrenamiento final: $LOG_DIR/final_training_$TIMESTAMP.log

🎉 Experimento completado!
EOF

log "📄 Resumen guardado en: $SUMMARY_FILE"

# Mostrar estadísticas finales
echo ""
echo "🌟 ============================================="
echo "🌟 EXPERIMENTO NOCTURNO COMPLETADO EXITOSAMENTE"
echo "🌟 ============================================="
echo ""
echo "📊 Archivos generados:"
ls -la "$LOG_DIR"/*_$TIMESTAMP.* 2>/dev/null || echo "No se encontraron logs"
echo ""
echo "🎯 Próximos pasos:"
echo "1. Revisar: cat $SUMMARY_FILE"
echo "2. Analizar: python optuna_results_analyzer.py"
echo "3. Usar optimizados: python crypto_ml_trainer_optimized.py"
echo ""
echo "✅ ¡Todo listo para revisar los resultados!"

log "🌟 Script de experimento nocturno finalizado exitosamente"
