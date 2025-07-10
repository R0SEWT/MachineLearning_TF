#!/bin/bash
"""
📊 MONITOR EXPERIMENTO ESTÁNDAR
===============================

Script para monitorear el progreso del experimento estándar
- Muestra logs en tiempo real
- Reporta métricas clave
- Permite cancelar experimento de forma segura

Uso: ./monitor_experimento_estandar.sh [EXPERIMENT_NAME]
"""

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuración
BASE_DIR="/home/exodia/Documentos/MachineLearning_TF"
LOGS_DIR="${BASE_DIR}/logs"

# Función para mostrar ayuda
show_help() {
    echo -e "${CYAN}📊 Monitor Experimento Estándar${NC}"
    echo -e "${BLUE}===============================${NC}"
    echo ""
    echo "Uso: $0 [EXPERIMENT_NAME]"
    echo ""
    echo "Opciones:"
    echo "  -h, --help     Mostrar esta ayuda"
    echo "  -l, --list     Listar experimentos disponibles"
    echo "  -f, --follow   Seguir logs en tiempo real (por defecto)"
    echo "  -s, --summary  Mostrar solo resumen"
    echo ""
    echo "Ejemplos:"
    echo "  $0                           # Monitorear experimento más reciente"
    echo "  $0 experimento_estandar_*    # Monitorear experimento específico"
    echo "  $0 --list                    # Ver experimentos disponibles"
}

# Función para listar experimentos
list_experiments() {
    echo -e "${CYAN}📋 Experimentos Disponibles:${NC}"
    echo "=============================="
    
    if [[ -d "$LOGS_DIR" ]]; then
        local experiments=$(find "$LOGS_DIR" -name "experimento_estandar_*.log" -exec basename {} .log \; | sort -r | head -10)
        
        if [[ -n "$experiments" ]]; then
            echo "$experiments" | while read -r exp; do
                local log_file="${LOGS_DIR}/${exp}.log"
                local size=$(du -h "$log_file" 2>/dev/null | cut -f1)
                local date=$(date -r "$log_file" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
                
                echo -e "${GREEN}📁 $exp${NC}"
                echo -e "   📅 $date | 📊 $size"
                echo ""
            done
        else
            echo -e "${YELLOW}⚠️  No se encontraron experimentos${NC}"
        fi
    else
        echo -e "${RED}❌ Directorio de logs no encontrado: $LOGS_DIR${NC}"
    fi
}

# Función para obtener estadísticas del experimento
get_experiment_stats() {
    local log_file="$1"
    
    if [[ ! -f "$log_file" ]]; then
        echo -e "${RED}❌ Archivo de log no encontrado: $log_file${NC}"
        return 1
    fi
    
    echo -e "${CYAN}📊 Estadísticas del Experimento:${NC}"
    echo "================================"
    
    # Información básica
    local start_time=$(grep "🚀 Iniciando experimento" "$log_file" | head -1 | cut -d']' -f1 | cut -d'[' -f2)
    local current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${BLUE}⏰ Inicio:${NC} $start_time"
    echo -e "${BLUE}⏰ Ahora:${NC} $current_time"
    
    # Estado actual
    local last_message=$(tail -1 "$log_file")
    echo -e "${BLUE}📝 Último mensaje:${NC} $last_message"
    
    # Contadores
    local total_lines=$(wc -l < "$log_file")
    local errors=$(grep -c "❌" "$log_file" 2>/dev/null || echo "0")
    local warnings=$(grep -c "⚠️" "$log_file" 2>/dev/null || echo "0")
    local success=$(grep -c "✅" "$log_file" 2>/dev/null || echo "0")
    
    echo ""
    echo -e "${BLUE}📈 Contadores:${NC}"
    echo -e "   📄 Total líneas: $total_lines"
    echo -e "   ✅ Éxitos: $success"
    echo -e "   ⚠️  Advertencias: $warnings"
    echo -e "   ❌ Errores: $errors"
    
    # Progreso de optimización
    if grep -q "Optimización completada" "$log_file"; then
        echo -e "${GREEN}🎯 Estado: Optimización completada${NC}"
    elif grep -q "Ejecutando optimización" "$log_file"; then
        echo -e "${YELLOW}🔄 Estado: Optimización en progreso${NC}"
    elif grep -q "Verificando datasets" "$log_file"; then
        echo -e "${BLUE}📊 Estado: Verificación de datos${NC}"
    else
        echo -e "${YELLOW}⚠️  Estado: Iniciando o desconocido${NC}"
    fi
    
    echo ""
}

# Función para seguir logs en tiempo real
follow_logs() {
    local log_file="$1"
    
    if [[ ! -f "$log_file" ]]; then
        echo -e "${RED}❌ Archivo de log no encontrado: $log_file${NC}"
        return 1
    fi
    
    echo -e "${CYAN}📡 Siguiendo logs en tiempo real...${NC}"
    echo -e "${YELLOW}   (Presiona Ctrl+C para salir)${NC}"
    echo "================================"
    
    # Mostrar estadísticas iniciales
    get_experiment_stats "$log_file"
    echo ""
    echo -e "${PURPLE}📜 Log Stream:${NC}"
    echo "=============="
    
    # Seguir logs con colores
    tail -f "$log_file" | while read -r line; do
        if [[ "$line" == *"❌"* ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ "$line" == *"✅"* ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ "$line" == *"⚠️"* ]]; then
            echo -e "${YELLOW}$line${NC}"
        elif [[ "$line" == *"🚀"* ]] || [[ "$line" == *"🔄"* ]]; then
            echo -e "${BLUE}$line${NC}"
        else
            echo "$line"
        fi
    done
}

# Función para mostrar resumen
show_summary() {
    local log_file="$1"
    
    get_experiment_stats "$log_file"
    
    echo ""
    echo -e "${CYAN}📋 Resumen de Actividad:${NC}"
    echo "========================"
    
    # Mostrar últimas 10 líneas importantes
    grep -E "(🚀|✅|❌|⚠️|🔄|📊)" "$log_file" | tail -10 | while read -r line; do
        if [[ "$line" == *"❌"* ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ "$line" == *"✅"* ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ "$line" == *"⚠️"* ]]; then
            echo -e "${YELLOW}$line${NC}"
        else
            echo -e "${BLUE}$line${NC}"
        fi
    done
}

# Función principal
main() {
    local experiment_name=""
    local mode="follow"
    
    # Procesar argumentos
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -l|--list)
                list_experiments
                exit 0
                ;;
            -f|--follow)
                mode="follow"
                shift
                ;;
            -s|--summary)
                mode="summary"
                shift
                ;;
            *)
                experiment_name="$1"
                shift
                ;;
        esac
    done
    
    # Si no se especifica experimento, usar el más reciente
    if [[ -z "$experiment_name" ]]; then
        experiment_name=$(find "$LOGS_DIR" -name "experimento_estandar_*.log" -exec basename {} .log \; | sort -r | head -1)
        
        if [[ -z "$experiment_name" ]]; then
            echo -e "${RED}❌ No se encontraron experimentos${NC}"
            echo -e "${YELLOW}💡 Ejecuta primero: ./experimento_estandar.sh${NC}"
            exit 1
        fi
        
        echo -e "${YELLOW}📍 Usando experimento más reciente: $experiment_name${NC}"
        echo ""
    fi
    
    local log_file="${LOGS_DIR}/${experiment_name}.log"
    
    # Ejecutar según modo
    case $mode in
        "follow")
            follow_logs "$log_file"
            ;;
        "summary")
            show_summary "$log_file"
            ;;
        *)
            echo -e "${RED}❌ Modo desconocido: $mode${NC}"
            exit 1
            ;;
    esac
}

# Verificar si se ejecuta directamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
