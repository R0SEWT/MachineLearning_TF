#!/bin/bash
"""
🚀 EXPERIMENTO ESTÁNDAR - Machine Learning para Criptomonedas
================================================================

Script estándar para entrenamiento y optimización de modelos ML
- Funciona con CPU/GPU automáticamente
- Configuración adaptativa según recursos disponibles
- Pipeline completo de entrenamiento y evaluación

Autor: Sistema ML-TF
Fecha: 2025-07-09
"""

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Timestamp para logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="experimento_estandar_${TIMESTAMP}"

# Directorios
BASE_DIR="/home/exodia/Documentos/MachineLearning_TF"
LOGS_DIR="${BASE_DIR}/logs"
RESULTS_DIR="${BASE_DIR}/results"
MODELS_DIR="${BASE_DIR}/models"

# Archivos de log
MAIN_LOG="${LOGS_DIR}/${EXPERIMENT_NAME}.log"
ERROR_LOG="${LOGS_DIR}/${EXPERIMENT_NAME}_error.log"

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[${timestamp}]${NC} $message" | tee -a "$MAIN_LOG"
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${RED}[${timestamp}] ❌ ERROR: $message${NC}" | tee -a "$MAIN_LOG" "$ERROR_LOG"
}

log_success() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}[${timestamp}] ✅ $message${NC}" | tee -a "$MAIN_LOG"
}

log_warning() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[${timestamp}] ⚠️  $message${NC}" | tee -a "$MAIN_LOG"
}

create_directories() {
    log_message "📁 Creando directorios necesarios..."
    mkdir -p "$LOGS_DIR" "$RESULTS_DIR" "$MODELS_DIR"
    mkdir -p "${RESULTS_DIR}/${EXPERIMENT_NAME}"
}

check_environment() {
    log_message "🔍 Verificando ambiente de trabajo..."
    
    # Verificar directorio base
    if [[ ! -d "$BASE_DIR" ]]; then
        log_error "Directorio base no encontrado: $BASE_DIR"
        exit 1
    fi
    
    # Verificar ambiente conda
    if conda env list | grep -q "ML-TF-G"; then
        log_success "Ambiente conda ML-TF-G encontrado"
        conda activate ML-TF-G
    else
        log_warning "Ambiente ML-TF-G no encontrado, usando ambiente actual"
    fi
    
    # Verificar Python y librerías
    if python -c "import pandas, numpy, sklearn, xgboost, lightgbm, catboost" 2>/dev/null; then
        log_success "Librerías Python verificadas"
    else
        log_error "Faltan librerías Python necesarias"
        exit 1
    fi
}

detect_compute_resources() {
    log_message "🔧 Detectando recursos de cómputo..."
    
    # Detectar GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            log_success "GPU NVIDIA detectada"
            export USE_GPU=true
            export DEVICE_TYPE="gpu"
        else
            log_warning "nvidia-smi no funciona, usando CPU"
            export USE_GPU=false
            export DEVICE_TYPE="cpu"
        fi
    else
        log_message "nvidia-smi no disponible, usando CPU"
        export USE_GPU=false
        export DEVICE_TYPE="cpu"
    fi
    
    # Detectar número de cores
    export NUM_CORES=$(nproc)
    log_message "💻 Cores de CPU disponibles: $NUM_CORES"
    
    # Detectar memoria RAM
    TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
    log_message "💾 Memoria RAM total: $TOTAL_RAM"
}

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

run_data_verification() {
    log_message "📊 Verificando datasets..."
    
    cd "$BASE_DIR" || exit 1
    
    # Verificar dataset ML
    if [[ -f "data/ml_dataset.csv" ]]; then
        log_success "Dataset ML encontrado"
        
        # Verificar estructura básica
        python -c "
import pandas as pd
try:
    df = pd.read_csv('data/ml_dataset.csv')
    print(f'✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas')
    
    # Verificar target
    if 'future_ret_30d' in df.columns:
        print(f'✅ Target encontrado: {df[\"future_ret_30d\"].value_counts().to_dict()}')
    else:
        print('⚠️  Target future_ret_30d no encontrado')
        
except Exception as e:
    print(f'❌ Error cargando dataset: {e}')
    exit(1)
" | tee -a "$MAIN_LOG"
    else
        log_error "Dataset ML no encontrado en data/ml_dataset.csv"
        exit 1
    fi
}

run_quick_optimization() {
    log_message "⚡ Ejecutando optimización rápida..."
    
    cd "$BASE_DIR" || exit 1
    
    local trials=20
    local timeout=600  # 10 minutos
    
    if [[ "$USE_GPU" == "true" ]]; then
        trials=50
        timeout=1200  # 20 minutos
        log_message "🔥 Modo GPU: aumentando trials y timeout"
    fi
    
    log_message "🔧 Configuración:"
    log_message "   - Trials por modelo: $trials"
    log_message "   - Timeout: $timeout segundos"
    log_message "   - Dispositivo: $DEVICE_TYPE"
    
    # Ejecutar optimización
    python scripts/optimization/quick_optimization.py \
        --mode full \
        --trials "$trials" \
        --timeout "$timeout" \
        2>&1 | tee -a "$MAIN_LOG"
    
    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -eq 0 ]]; then
        log_success "Optimización completada exitosamente"
    else
        log_error "Optimización falló con código: $exit_code"
        return 1
    fi
}

run_model_evaluation() {
    log_message "📊 Evaluando modelos entrenados..."
    
    cd "$BASE_DIR" || exit 1
    
    # Buscar modelos recientes
    local latest_models=$(find models/ -name "*$(date +%Y%m%d)*" -type f | head -10)
    
    if [[ -n "$latest_models" ]]; then
        log_success "Modelos encontrados para evaluación:"
        echo "$latest_models" | while read -r model; do
            log_message "   📁 $model"
        done
    else
        log_warning "No se encontraron modelos recientes"
    fi
    
    # TODO: Agregar script de evaluación específico
    log_message "📈 Evaluación de modelos pendiente de implementar"
}

generate_report() {
    log_message "📋 Generando reporte del experimento..."
    
    local report_file="${RESULTS_DIR}/${EXPERIMENT_NAME}/reporte.md"
    
    cat > "$report_file" << EOF
# 📊 Reporte del Experimento Estándar

**Fecha:** $(date '+%Y-%m-%d %H:%M:%S')
**Experimento:** $EXPERIMENT_NAME
**Dispositivo:** $DEVICE_TYPE
**Cores:** $NUM_CORES

## 🔧 Configuración

- **Dataset:** data/ml_dataset.csv
- **Ambiente:** ML-TF-G
- **GPU Disponible:** $USE_GPU

## 📈 Resultados

### Optimización de Hiperparámetros
- **Estado:** $(if [[ -f "${LOGS_DIR}/${EXPERIMENT_NAME}.log" ]] && grep -q "✅.*completada" "${LOGS_DIR}/${EXPERIMENT_NAME}.log"; then echo "Completada"; else echo "Pendiente/Error"; fi)
- **Log principal:** ${MAIN_LOG}
- **Log errores:** ${ERROR_LOG}

### Modelos Generados
$(find models/ -name "*$(date +%Y%m%d)*" -type f | wc -l) modelos generados hoy

## 📂 Archivos Generados

- **Logs:** ${LOGS_DIR}/${EXPERIMENT_NAME}*
- **Modelos:** ${MODELS_DIR}/
- **Resultados:** ${RESULTS_DIR}/${EXPERIMENT_NAME}/

## 🎯 Próximos Pasos

1. Revisar métricas de modelos en logs
2. Evaluar performance en datos de validación  
3. Seleccionar mejor modelo para producción
4. Generar predicciones en datos nuevos

---
*Generado automáticamente por experimento_estandar.sh*
EOF

    log_success "Reporte generado: $report_file"
}

cleanup_and_finish() {
    log_message "🧹 Limpieza final..."
    
    # Comprimir logs antiguos si existen muchos
    local log_count=$(find "$LOGS_DIR" -name "*.log" | wc -l)
    if [[ $log_count -gt 50 ]]; then
        log_message "🗜️  Comprimiendo logs antiguos..."
        find "$LOGS_DIR" -name "*.log" -mtime +7 -exec gzip {} \;
    fi
    
    # Mostrar resumen
    log_message "📊 Resumen del experimento:"
    log_message "   - Duración total: $(($(date +%s) - START_TIME)) segundos"
    log_message "   - Log principal: $MAIN_LOG"
    log_message "   - Directorio resultados: ${RESULTS_DIR}/${EXPERIMENT_NAME}"
    
    log_success "Experimento estándar completado"
}

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

main() {
    # Banner inicial
    echo -e "${CYAN}"
    echo "=================================================="
    echo "🚀 EXPERIMENTO ESTÁNDAR ML CRIPTOMONEDAS"
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
    echo -e "${NC}"
    
    # Marcar tiempo de inicio
    START_TIME=$(date +%s)
    
    # Pipeline principal
    create_directories
    log_message "🚀 Iniciando experimento estándar: $EXPERIMENT_NAME"
    
    check_environment || exit 1
    detect_compute_resources
    run_data_verification || exit 1
    
    log_message "🔄 Iniciando pipeline de optimización..."
    if run_quick_optimization; then
        run_model_evaluation
        generate_report
    else
        log_error "Pipeline falló en optimización"
        exit 1
    fi
    
    cleanup_and_finish
    
    echo -e "${GREEN}"
    echo "=================================================="
    echo "✅ EXPERIMENTO COMPLETADO EXITOSAMENTE"
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
    echo -e "${NC}"
}

# =============================================================================
# MANEJO DE ERRORES Y SEÑALES
# =============================================================================

# Capturar errores y limpiar
trap 'log_error "Script interrumpido"; exit 1' INT TERM

# Verificar argumentos
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
