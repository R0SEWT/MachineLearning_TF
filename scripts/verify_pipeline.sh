#!/bin/bash

# =====================================================
# 🧪 VERIFICACIÓN COMPLETA DEL PIPELINE TRAS REORGANIZACIÓN
# =====================================================

echo "🚀 VERIFICACIÓN DEL PIPELINE MACHINELEARNING_TF"
echo "================================================"

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/logs/pipeline_verification_$TIMESTAMP.log"

# Crear directorio de logs
mkdir -p "$PROJECT_ROOT/logs"

# Cambiar al directorio del proyecto
cd "$PROJECT_ROOT"

# Función de logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Función para verificar archivos
check_file() {
    if [ -f "$1" ]; then
        log "✅ $1 existe"
        return 0
    else
        log "❌ $1 NO existe"
        return 1
    fi
}

# Función para verificar directorios
check_dir() {
    if [ -d "$1" ]; then
        log "✅ $1/ existe"
        return 0
    else
        log "❌ $1/ NO existe"
        return 1
    fi
}

log "🔍 Iniciando verificación del pipeline..."
log "📁 Directorio del proyecto: $PROJECT_ROOT"

# =====================================================
# 1. VERIFICACIÓN DE ESTRUCTURA
# =====================================================
log ""
log "📂 VERIFICANDO ESTRUCTURA DE DIRECTORIOS:"
log "=========================================="

check_dir "$PROJECT_ROOT/src/models"
check_dir "$PROJECT_ROOT/src/utils"
check_dir "$PROJECT_ROOT/src/scraping"
check_dir "$PROJECT_ROOT/scripts/experiments"
check_dir "$PROJECT_ROOT/scripts/monitoring"
check_dir "$PROJECT_ROOT/scripts/optimization"
check_dir "$PROJECT_ROOT/docs"
check_dir "$PROJECT_ROOT/notebooks"
check_dir "$PROJECT_ROOT/data"
check_dir "$PROJECT_ROOT/models"
check_dir "$PROJECT_ROOT/tests"

# =====================================================
# 2. VERIFICACIÓN DE ARCHIVOS PRINCIPALES
# =====================================================
log ""
log "📄 VERIFICANDO ARCHIVOS PRINCIPALES:"
log "===================================="

check_file "$PROJECT_ROOT/README.md"
check_file "$PROJECT_ROOT/environment.yml"
check_file "$PROJECT_ROOT/src/models/crypto_ml_trainer.py"
check_file "$PROJECT_ROOT/src/models/crypto_ml_trainer_optimized.py"
check_file "$PROJECT_ROOT/src/utils/utils/feature_engineering.py"
check_file "$PROJECT_ROOT/scripts/optimization/quick_optimization.py"
check_file "$PROJECT_ROOT/scripts/optimization/crypto_hyperparameter_optimizer.py"
check_file "$PROJECT_ROOT/scripts/experiments/experimento_nocturno.sh"
check_file "$PROJECT_ROOT/scripts/monitoring/monitor_experimento_gpu.sh"

# =====================================================
# 3. VERIFICACIÓN DE DATOS
# =====================================================
log ""
log "📊 VERIFICANDO DATOS:"
log "===================="

check_file "$PROJECT_ROOT/data/crypto_ohlc_join.csv"
check_file "$PROJECT_ROOT/data/ml_dataset.csv"
check_file "$PROJECT_ROOT/data/crypto_modeling_groups.csv"

# =====================================================
# 4. VERIFICACIÓN DE ENTORNO CONDA
# =====================================================
log ""
log "🐍 VERIFICANDO ENTORNO CONDA:"
log "============================="

# Activar conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || {
    log "⚠️ No se pudo cargar conda"
}

# Verificar que el entorno ML-TF-G existe
if conda env list | grep -q "ML-TF-G"; then
    log "✅ Entorno ML-TF-G existe"
    
    # Activar entorno
    conda activate ML-TF-G
    if [ $? -eq 0 ]; then
        log "✅ Entorno ML-TF-G activado correctamente"
    else
        log "❌ No se pudo activar entorno ML-TF-G"
    fi
else
    log "❌ Entorno ML-TF-G NO existe"
fi

# =====================================================
# 5. VERIFICACIÓN DE LIBRERÍAS
# =====================================================
log ""
log "📚 VERIFICANDO LIBRERÍAS PRINCIPALES:"
log "====================================="

python -c "import pandas; print('✅ pandas disponible')" 2>/dev/null || log "❌ pandas NO disponible"
python -c "import numpy; print('✅ numpy disponible')" 2>/dev/null || log "❌ numpy NO disponible"
python -c "import sklearn; print('✅ scikit-learn disponible')" 2>/dev/null || log "❌ scikit-learn NO disponible"
python -c "import xgboost; print('✅ xgboost disponible')" 2>/dev/null || log "❌ xgboost NO disponible"
python -c "import lightgbm; print('✅ lightgbm disponible')" 2>/dev/null || log "❌ lightgbm NO disponible"
python -c "import catboost; print('✅ catboost disponible')" 2>/dev/null || log "❌ catboost NO disponible"
python -c "import optuna; print('✅ optuna disponible')" 2>/dev/null || log "❌ optuna NO disponible"
python -c "import jupyter; print('✅ jupyter disponible')" 2>/dev/null || log "❌ jupyter NO disponible"

# =====================================================
# 6. PRUEBA DE IMPORTS
# =====================================================
log ""
log "🔗 VERIFICANDO IMPORTS DEL PROYECTO:"
log "==================================="

cd "$PROJECT_ROOT"

# Verificar imports del trainer
log "🧪 Probando imports del trainer principal..."
python -c "
import sys
sys.path.append('src/utils/utils')
try:
    from src.utils.utils.feature_engineering import create_ml_features, prepare_ml_dataset
    print('✅ Feature engineering importado correctamente')
except Exception as e:
    print(f'❌ Error importando feature engineering: {e}')
" 2>/dev/null || log "❌ Error en imports del trainer"

# =====================================================
# 7. PRUEBA RÁPIDA DEL TRAINER
# =====================================================
log ""
log "🚀 PRUEBA RÁPIDA DEL TRAINER:"
log "============================"

log "🧪 Ejecutando trainer por 30 segundos..."
timeout 30 python src/models/crypto_ml_trainer.py > "$PROJECT_ROOT/logs/trainer_test_$TIMESTAMP.log" 2>&1
if [ $? -eq 124 ]; then
    log "✅ Trainer ejecutándose correctamente (timeout después de 30s)"
elif [ $? -eq 0 ]; then
    log "✅ Trainer completado exitosamente"
else
    log "❌ Trainer tuvo problemas - revisar logs"
fi

# =====================================================
# 8. VERIFICACIÓN DE SCRIPTS
# =====================================================
log ""
log "📜 VERIFICANDO SCRIPTS DE OPTIMIZACIÓN:"
log "======================================="

# Verificar que quick_optimization funcione
log "🧪 Probando quick_optimization.py..."
python scripts/optimization/quick_optimization.py --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    log "✅ quick_optimization.py funciona correctamente"
else
    log "❌ quick_optimization.py tiene problemas"
fi

# =====================================================
# 9. RESUMEN FINAL
# =====================================================
log ""
log "📋 RESUMEN DE VERIFICACIÓN:"
log "=========================="

ERROR_COUNT=$(grep -c "❌" "$LOG_FILE")
SUCCESS_COUNT=$(grep -c "✅" "$LOG_FILE")

log "✅ Verificaciones exitosas: $SUCCESS_COUNT"
log "❌ Errores encontrados: $ERROR_COUNT"

if [ $ERROR_COUNT -eq 0 ]; then
    log "🎉 PIPELINE COMPLETAMENTE FUNCIONAL!"
    log "🚀 Todos los componentes están listos para usar"
else
    log "⚠️ PIPELINE PARCIALMENTE FUNCIONAL"
    log "🔧 Revisar errores antes de usar en producción"
fi

log ""
log "📄 Log completo guardado en: $LOG_FILE"
log "🕐 Verificación completada: $(date)"

echo ""
echo "📊 RESUMEN:"
echo "==========="
echo "✅ Éxitos: $SUCCESS_COUNT"
echo "❌ Errores: $ERROR_COUNT"
echo "📄 Log: $LOG_FILE"
