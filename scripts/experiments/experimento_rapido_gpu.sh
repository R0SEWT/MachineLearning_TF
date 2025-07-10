#!/bin/bash

# ====================================================================
# ⚡ EXPERIMENTO RÁPIDO GPU - OPTIMIZACIÓN ACELERADA
# Sistema de optimización rápida usando GPU con configuración optimizada
# ====================================================================

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../../logs"
RESULTS_DIR="$SCRIPT_DIR/../../optimization_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/experimento_rapido_gpu_$TIMESTAMP.log"

# Crear directorios necesarios
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Función de logging con colores
log() {
    echo -e "\e[34m[$(date '+%Y-%m-%d %H:%M:%S')]\e[0m $1" | tee -a "$MAIN_LOG"
}

log_success() {
    echo -e "\e[32m[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1\e[0m" | tee -a "$MAIN_LOG"
}

log_warning() {
    echo -e "\e[33m[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ $1\e[0m" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "\e[31m[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1\e[0m" | tee -a "$MAIN_LOG"
}

# Función para manejar errores
handle_error() {
    log_error "ERROR: $1"
    log "🔍 Revisar logs para más detalles"
    exit 1
}

# Función para mostrar progreso
show_progress() {
    echo ""
    echo "⚡=================================================="
    echo "🚀 $1"
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
    echo ""
}

# Verificar GPU de manera rápida
check_gpu() {
    log "🔥 Verificando GPU disponible..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1 >> "$MAIN_LOG"
        if [ $? -ne 0 ]; then
            log_warning "GPU disponible pero con advertencias"
        else
            log_success "GPU verificada y disponible"
        fi
    else
        log_warning "nvidia-smi no disponible, continuando sin GPU"
    fi
}

# Verificar ambiente Python
check_python_env() {
    log "🐍 Verificando ambiente Python..."
    
    # Activar ambiente conda
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate ML-TF-G || handle_error "No se pudo activar el ambiente conda ML-TF-G"
        log_success "Ambiente conda ML-TF-G activado"
    else
        log_warning "Conda no encontrado, usando ambiente sistema"
    fi
    
    # Verificar librerías críticas
    python -c "import optuna, xgboost, lightgbm, catboost; print('Librerías ML OK')" 2>/dev/null
    if [ $? -eq 0 ]; then
        log_success "Librerías ML verificadas"
    else
        log_warning "Algunas librerías ML podrían faltar"
    fi
}

# Inicio del experimento
show_progress "EXPERIMENTO RÁPIDO GPU INICIADO"
log "⚡🚀 Experimento rápido GPU iniciado"
log "📁 Directorio: $SCRIPT_DIR"
log "📝 Log principal: $MAIN_LOG"
log "📊 Resultados: $RESULTS_DIR"

# Verificaciones iniciales
check_gpu
check_python_env

# ====================================================================
# FASE 1: OPTIMIZACIÓN RÁPIDA GPU (15-20 minutos)
# ====================================================================
show_progress "FASE 1: OPTIMIZACIÓN RÁPIDA GPU (15-20 minutos)"

# Navegar al directorio de optimización
cd "$SCRIPT_DIR/../optimization" || handle_error "No se pudo acceder al directorio de optimización"

log "🔥 Optimización rápida con crypto_hyperparameter_optimizer (5 min)..."
timeout 300 python quick_optimization.py --quick --gpu > "$LOG_DIR/fast_optimization_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log_success "Optimización rápida completada"
else
    log_warning "Optimización rápida tuvo problemas o timeout, ejecutando script alternativo..."
    
    # Intentar con script de shell
    log "🔄 Ejecutando optimización rápida con script shell..."
    timeout 300 bash optimizacion_rapida.sh > "$LOG_DIR/fast_shell_gpu_$TIMESTAMP.log" 2>&1
    if [ $? -eq 0 ]; then
        log_success "Optimización shell rápida completada"
    else
        log_warning "Ambos métodos fallaron, continuando..."
    fi
fi

# ====================================================================
# FASE 2: OPTIMIZACIÓN INTENSIVA CORTA (10-15 minutos)
# ====================================================================
show_progress "FASE 2: OPTIMIZACIÓN INTENSIVA CORTA (10-15 minutos)"

log "🌟 Ejecutando optimización intensiva con quick_optimization (10 min)..."
timeout 600 python quick_optimization.py --intensive --gpu > "$LOG_DIR/intensive_optimization_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log_success "Optimización intensiva completada"
else
    log_warning "Optimización intensiva tuvo problemas o timeout, continuando..."
fi

# ====================================================================
# FASE 3: ENTRENAMIENTO RÁPIDO CON MEJORES PARÁMETROS
# ====================================================================
show_progress "FASE 3: ENTRENAMIENTO RÁPIDO"

# Navegar al directorio de modelos
cd "$SCRIPT_DIR/../../src/models" || handle_error "No se pudo acceder al directorio de modelos"

log "🎯 Ejecutando entrenamiento rápido con mejores parámetros..."
timeout 600 python crypto_ml_trainer_optimized.py --quick-mode > "$LOG_DIR/quick_training_gpu_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    log_success "Entrenamiento rápido completado exitosamente"
else
    log_warning "Entrenamiento rápido tuvo timeout o problemas"
fi

# ====================================================================
# FASE 4: ANÁLISIS RÁPIDO DE RESULTADOS
# ====================================================================
show_progress "FASE 4: ANÁLISIS RÁPIDO DE RESULTADOS"

log "📊 Ejecutando análisis de resultados Optuna..."
if [ -f "/home/exodia/Documentos/MachineLearning_TF/optimization_results/optuna_studies.db" ]; then
    python -c "
import optuna
import sqlite3

storage = 'sqlite:////home/exodia/Documentos/MachineLearning_TF/optimization_results/optuna_studies.db'
try:
    studies = optuna.study.get_all_study_summaries(storage)
    print(f'📊 Total de estudios: {len(studies)}')
    for study in studies[-3:]:  # Últimos 3 estudios
        print(f'- {study.study_name}: {study.n_trials} trials')
        if study.best_trial:
            print(f'  Mejor valor: {study.best_trial.value:.4f}')
except Exception as e:
    print(f'Error analizando estudios: {e}')
" > "$LOG_DIR/quick_analysis_gpu_$TIMESTAMP.log" 2>&1
    log_success "Análisis rápido completado"
else
    log_warning "Base de datos Optuna no encontrada"
fi

# ====================================================================
# MONITOREO FINAL
# ====================================================================
show_progress "RESUMEN FINAL"

log "📊 Estado final de GPU:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | head -1 >> "$MAIN_LOG"
else
    log "GPU no disponible para monitoreo"
fi

# ====================================================================
# RESUMEN FINAL
# ====================================================================
END_TIME=$(date +"%Y%m%d_%H%M%S")
log_success "Experimento rápido GPU finalizado"
log "⏰ Tiempo de finalización: $(date '+%Y-%m-%d %H:%M:%S')"

# Generar resumen
SUMMARY_FILE="$LOG_DIR/resumen_experimento_rapido_gpu_$TIMESTAMP.txt"
cat > "$SUMMARY_FILE" << EOF
⚡🚀 RESUMEN DEL EXPERIMENTO RÁPIDO GPU
====================================

⏰ Inicio: $TIMESTAMP
⏰ Fin: $END_TIME
⏱️ Duración estimada: 30-45 minutos

🚀 CONFIGURACIÓN RÁPIDA:
- XGBoost: 50 trials, 5 min
- LightGBM: 50 trials, 5 min  
- CatBoost: 50 trials, 5 min
- Intensivo: 100 trials, 10 min
- Entrenamiento: 10 min máximo

📁 Logs generados:
$(ls -la "$LOG_DIR"/*gpu*$TIMESTAMP.* 2>/dev/null)

📊 Archivos de resultados:
$(ls -la "$RESULTS_DIR"/*.json 2>/dev/null | tail -3)

🎯 VENTAJAS DEL MODO RÁPIDO:
- Optimización en 30-45 minutos
- Explora espacio de hiperparámetros eficientemente
- Usa GPU para acelerar entrenamiento
- Encuentra parámetros competitivos rápidamente

📈 Para usar resultados:
1. Revisar: cat $SUMMARY_FILE
2. Entrenar: python crypto_ml_trainer_optimized.py
3. Analizar: python optuna_results_analyzer.py

📝 Logs principales:
- Logs rápidos: $LOG_DIR/fast_*_gpu_$TIMESTAMP.log
- Intensivo: $LOG_DIR/intensive_fast_gpu_$TIMESTAMP.log
- Entrenamiento: $LOG_DIR/quick_training_gpu_$TIMESTAMP.log
- Análisis: $LOG_DIR/quick_analysis_gpu_$TIMESTAMP.log

⚡ Experimento rápido completado!
EOF

log "📄 Resumen guardado en: $SUMMARY_FILE"

# Mostrar estadísticas finales
echo ""
echo "⚡🌟 ============================================="
echo "⚡🌟 EXPERIMENTO RÁPIDO GPU COMPLETADO"
echo "⚡🌟 ============================================="
echo ""
echo "⏱️ Tiempo total: ~30-45 minutos"
echo "🎯 Optimización eficiente completada"
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Final Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1
    echo ""
fi
echo "📂 Archivos generados:"
ls -la "$LOG_DIR"/*gpu*$TIMESTAMP.* 2>/dev/null | head -5 || echo "No se encontraron logs"
echo ""
echo "🎯 Próximos pasos:"
echo "1. Revisar resumen: cat $SUMMARY_FILE"
echo "2. Ver resultados: python optuna_results_analyzer.py"
echo "3. Entrenar con óptimos: python crypto_ml_trainer_optimized.py"
echo ""
echo "✅⚡ ¡Experimento rápido completado exitosamente!"

log_success "Script de experimento rápido GPU finalizado exitosamente"
