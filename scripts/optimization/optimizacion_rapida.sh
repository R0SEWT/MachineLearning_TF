#!/bin/bash

# ====================================================================
# ⚡ OPTIMIZACIÓN RÁPIDA AUTOMATIZADA
# Para experimentos de desarrollo y pruebas rápidas
# ====================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../../optimization_results/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/optimization_rapida_$TIMESTAMP.log"

mkdir -p "$LOG_DIR"

# Función de logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

echo "⚡ ============================================="
echo "⚡ OPTIMIZACIÓN RÁPIDA AUTOMATIZADA"
echo "⚡ Tiempo estimado: 30-60 minutos"
echo "⚡ ============================================="

# Activar ambiente
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ML-TF-G

log "🚀 Iniciando optimización rápida automatizada"

# Tests del sistema
log "🧪 Verificando sistema..."
python test_ml_system.py > "$LOG_DIR/test_rapido_$TIMESTAMP.log" 2>&1

# XGBoost rápido
log "🔥 XGBoost (30 trials, 10 min)..."
python quick_optimization.py --mode quick-xgb --trials 30 --timeout 600 > "$LOG_DIR/xgb_rapido_$TIMESTAMP.log" 2>&1

# LightGBM rápido  
log "💡 LightGBM (30 trials, 10 min)..."
python quick_optimization.py --mode quick-lgb --trials 30 --timeout 600 > "$LOG_DIR/lgb_rapido_$TIMESTAMP.log" 2>&1

# CatBoost rápido
log "🐱 CatBoost (30 trials, 10 min)..."
python quick_optimization.py --mode quick-cat --trials 30 --timeout 600 > "$LOG_DIR/cat_rapido_$TIMESTAMP.log" 2>&1

# Análisis
log "📊 Analizando resultados..."
python optuna_results_analyzer.py > "$LOG_DIR/analisis_rapido_$TIMESTAMP.log" 2>&1

# Integración
log "🔗 Integrando mejores parámetros..."
python integrate_optimized_params.py > "$LOG_DIR/integracion_rapida_$TIMESTAMP.log" 2>&1

# Validación
log "🎯 Validación con trainer optimizado..."
timeout 600 python crypto_ml_trainer_optimized.py > "$LOG_DIR/validacion_rapida_$TIMESTAMP.log" 2>&1

echo ""
echo "✅ ============================================="
echo "✅ OPTIMIZACIÓN RÁPIDA COMPLETADA"
echo "✅ ============================================="
echo ""
echo "📊 Para revisar resultados:"
echo "   python optuna_results_analyzer.py"
echo ""
echo "🎯 Para usar optimizados:"
echo "   python crypto_ml_trainer_optimized.py"
echo ""
echo "📝 Logs en: $LOG_DIR/"

log "✅ Optimización rápida completada exitosamente"
