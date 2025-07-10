#!/bin/bash
#
# ⚙️ CONFIGURACIÓN RÁPIDA PARA EXPERIMENTO ESTÁNDAR
# =================================================
#
# Script para verificar y configurar todo lo necesario
# antes de ejecutar el experimento estándar.
#
# Verificaciones:
# - Ambiente conda
# - Librerías Python
# - Datasets
# - Configuraciones
# - Permisos
#

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="/home/exodia/Documentos/MachineLearning_TF"

echo -e "${CYAN}"
echo "=================================================="
echo "⚙️ CONFIGURACIÓN RÁPIDA - EXPERIMENTO ESTÁNDAR"
echo "=================================================="
echo -e "${NC}"

# 1. Verificar estructura de directorios
echo -e "${BLUE}📁 Verificando estructura de directorios...${NC}"

required_dirs=(
    "data"
    "logs" 
    "models"
    "results"
    "scripts/experiments"
    "scripts/optimization"
)

for dir in "${required_dirs[@]}"; do
    if [[ -d "${BASE_DIR}/${dir}" ]]; then
        echo -e "   ✅ ${dir}"
    else
        echo -e "   ❌ ${dir} - Creando..."
        mkdir -p "${BASE_DIR}/${dir}"
    fi
done

# 2. Verificar dataset ML
echo -e "\n${BLUE}📊 Verificando dataset ML...${NC}"
if [[ -f "${BASE_DIR}/data/ml_dataset.csv" ]]; then
    echo -e "   ✅ ml_dataset.csv encontrado"
    
    # Activar ambiente antes de verificar
    if conda env list | grep -q "ML-TF-G"; then
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate ML-TF-G 2>/dev/null
    fi
    
    # Verificar estructura básica
    cd "${BASE_DIR}" && python -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('data/ml_dataset.csv')
    print(f'   📊 Shape: {df.shape[0]} filas, {df.shape[1]} columnas')
    
    # Verificar columnas clave
    required_cols = ['future_ret_30d', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f'   ⚠️  Columnas faltantes: {missing_cols}')
    else:
        print('   ✅ Columnas clave presentes')
        
    # Verificar distribución del target
    if 'future_ret_30d' in df.columns:
        target_dist = df['future_ret_30d'].value_counts()
        print(f'   🎯 Distribución target: {target_dist.to_dict()}')
        
except Exception as e:
    print(f'   ❌ Error verificando dataset: {e}')
    sys.exit(1)
" 2>/dev/null || echo -e "   ⚠️  No se pudo verificar estructura del dataset"
else
    echo -e "   ❌ ml_dataset.csv NO encontrado"
    echo -e "   💡 Necesitas generar el dataset ML primero"
fi

# 3. Verificar ambiente conda
echo -e "\n${BLUE}🐍 Verificando ambiente conda...${NC}"
if command -v conda &> /dev/null; then
    echo -e "   ✅ conda instalado"
    
    if conda env list | grep -q "ML-TF-G"; then
        echo -e "   ✅ Ambiente ML-TF-G encontrado"
        echo -e "   Activando ambiente..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate ML-TF-G 2>/dev/null
        if [[ $? -eq 0 ]]; then
            echo -e "   ✅ Ambiente activado"
        else
            echo -e "   ❌ Error al activar el ambiente ML-TF-G"
        fi
    else
        echo -e "   ⚠️  Ambiente ML-TF-G no encontrado"
        echo -e "   💡 Crear con: conda create -n ML-TF-G python=3.9"
    fi
else
    echo -e "   ❌ conda no instalado"
fi

# 4. Verificar librerías Python
echo -e "\n${BLUE}📦 Verificando librerías Python...${NC}"

# Activar ambiente si existe
if conda env list | grep -q "ML-TF-G"; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ML-TF-G 2>/dev/null
fi

required_libs=(
    "pandas"
    "numpy" 
    "sklearn"
    "xgboost"
    "lightgbm"
    "catboost"
    "optuna"
)

missing_libs=()

for lib in "${required_libs[@]}"; do
    if python -c "import $lib" 2>/dev/null; then
        echo -e "   ✅ $lib"
    else
        echo -e "   ❌ $lib"
        missing_libs+=("$lib")
    fi
done

if [[ ${#missing_libs[@]} -gt 0 ]]; then
    echo -e "\n${YELLOW}💡 Para instalar librerías faltantes:${NC}"
    echo -e "   pip install ${missing_libs[*]}"
fi

# 5. Verificar configuraciones
echo -e "\n${BLUE}⚙️ Verificando configuraciones...${NC}"

config_files=(
    "scripts/optimization/config/optimization_config.py"
    "scripts/optimization/quick_optimization.py"
    "scripts/optimization/crypto_hyperparameter_optimizer.py"
)

for file in "${config_files[@]}"; do
    if [[ -f "${BASE_DIR}/${file}" ]]; then
        echo -e "   ✅ $(basename "$file")"
    else
        echo -e "   ❌ $(basename "$file")"
    fi
done

# 6. Verificar permisos de scripts
echo -e "\n${BLUE}🔐 Verificando permisos de scripts...${NC}"

scripts=(
    "scripts/experiments/experimento_estandar.sh"
    "scripts/experiments/monitor_experimento_estandar.sh"
)

for script in "${scripts[@]}"; do
    if [[ -x "${BASE_DIR}/${script}" ]]; then
        echo -e "   ✅ $(basename "$script") - ejecutable"
    else
        echo -e "   ⚠️  $(basename "$script") - sin permisos"
        chmod +x "${BASE_DIR}/${script}" 2>/dev/null && echo -e "   ✅ Permisos corregidos"
    fi
done

# 7. Verificar recursos del sistema
echo -e "\n${BLUE}💻 Verificando recursos del sistema...${NC}"

# CPU
cpu_cores=$(nproc)
echo -e "   💻 CPU cores: $cpu_cores"

# RAM
total_ram=$(free -h | awk '/^Mem:/ {print $2}')
echo -e "   💾 RAM total: $total_ram"

# GPU
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        echo -e "   🔥 GPU: $gpu_info"
    else
        echo -e "   ⚠️  GPU: nvidia-smi no funciona"
    fi
else
    echo -e "   💻 GPU: No disponible (usando CPU)"
fi

# Espacio en disco
disk_space=$(df -h "${BASE_DIR}" | awk 'NR==2 {print $4}')
echo -e "   💽 Espacio disponible: $disk_space"

# 8. Resumen final
echo -e "\n${CYAN}📋 RESUMEN DE CONFIGURACIÓN${NC}"
echo "============================"

all_ready=true

# Verificar requisitos críticos
if [[ ! -f "${BASE_DIR}/data/ml_dataset.csv" ]]; then
    echo -e "${RED}❌ Dataset ML faltante${NC}"
    all_ready=false
fi

if [[ ${#missing_libs[@]} -gt 0 ]]; then
    echo -e "${RED}❌ Librerías faltantes: ${missing_libs[*]}${NC}"
    all_ready=false
fi

if [[ ! -f "${BASE_DIR}/scripts/experiments/experimento_estandar.sh" ]]; then
    echo -e "${RED}❌ Script principal faltante${NC}"
    all_ready=false
fi

if $all_ready; then
    echo -e "${GREEN}✅ CONFIGURACIÓN COMPLETA${NC}"
    echo -e "${GREEN}🚀 Listo para ejecutar experimento estándar${NC}"
    echo ""
    echo -e "${YELLOW}Para ejecutar:${NC}"
    echo -e "   cd ${BASE_DIR}"
    echo -e "   ./scripts/experiments/experimento_estandar.sh"
    echo ""
    echo -e "${YELLOW}Para monitorear:${NC}"
    echo -e "   ./scripts/experiments/monitor_experimento_estandar.sh"
else
    echo -e "${RED}❌ CONFIGURACIÓN INCOMPLETA${NC}"
    echo -e "${YELLOW}💡 Revisa los elementos marcados arriba${NC}"
fi

echo ""
