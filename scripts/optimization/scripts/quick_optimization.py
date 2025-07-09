#!/usr/bin/env python3
"""
🚀 Script de Optimización Rápida - Fase 5
==========================================

Script mejorado y enterprise-ready para optimización rápida de hiperparámetros
que reemplaza los scripts dispersos del sistema anterior.

Autor: Sistema de Optimización IA
Fecha: 2025-01-09 (Fase 5 - Organización)
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional
import json

# Agregar el directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent))

from config.optimization_config import get_quick_config, get_production_config, get_gpu_config, get_cpu_config
from utils.logging_setup import setup_logging, get_logger
from utils.import_manager import diagnose_imports, test_critical_imports
from core.optimizer import HyperparameterOptimizer, quick_optimization, production_optimization


def setup_argument_parser() -> argparse.ArgumentParser:
    """Configurar parser de argumentos"""
    parser = argparse.ArgumentParser(
        description="🚀 Optimizador de Hiperparámetros - Fase 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s data.csv                           # Optimización rápida con configuración por defecto
  %(prog)s data.csv --mode production         # Optimización completa de producción
  %(prog)s data.csv --models xgboost lightgbm # Solo XGBoost y LightGBM
  %(prog)s data.csv --trials 200              # 200 trials por modelo
  %(prog)s data.csv --gpu                     # Optimización para GPU
  %(prog)s data.csv --output /tmp/results     # Directorio de salida personalizado
  %(prog)s --diagnose                         # Diagnóstico del sistema
        """
    )
    
    # Argumentos principales
    parser.add_argument(
        'data_path',
        nargs='?',
        help='Ruta al archivo de datos (CSV, Parquet, Excel)'
    )
    
    # Configuraciones predefinidas
    parser.add_argument(
        '--mode',
        choices=['quick', 'production', 'gpu', 'cpu'],
        default='quick',
        help='Modo de optimización (default: quick)'
    )
    
    # Modelos específicos
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['xgboost', 'lightgbm', 'catboost'],
        help='Modelos específicos a optimizar'
    )
    
    # Parámetros de optimización
    parser.add_argument(
        '--trials',
        type=int,
        help='Número de trials por modelo'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        help='Timeout en segundos por modelo'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Número de folds para validación cruzada (default: 5)'
    )
    
    # Configuración de salida
    parser.add_argument(
        '--output',
        default='./results',
        help='Directorio de salida para resultados (default: ./results)'
    )
    
    parser.add_argument(
        '--experiment-id',
        help='ID personalizado para el experimento'
    )
    
    # Configuración de sistema
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Forzar uso de GPU'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Forzar uso de CPU'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=float,
        help='Límite de memoria en GB'
    )
    
    # Configuración de logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Nivel de logging (default: INFO)'
    )
    
    parser.add_argument(
        '--log-dir',
        default='./logs',
        help='Directorio de logs (default: ./logs)'
    )
    
    # Funciones de diagnóstico
    parser.add_argument(
        '--diagnose',
        action='store_true',
        help='Ejecutar diagnóstico del sistema'
    )
    
    parser.add_argument(
        '--test-imports',
        action='store_true',
        help='Probar imports críticos'
    )
    
    # Configuración de cache
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Deshabilitar cache de datos'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Limpiar cache antes de ejecutar'
    )
    
    return parser


def get_config_from_args(args) -> any:
    """Crear configuración basada en argumentos"""
    # Configuración base según modo
    if args.mode == 'quick':
        config = get_quick_config()
    elif args.mode == 'production':
        config = get_production_config()
    elif args.mode == 'gpu':
        config = get_gpu_config()
    elif args.mode == 'cpu':
        config = get_cpu_config()
    else:
        config = get_quick_config()
    
    # Aplicar modificaciones específicas
    if args.models:
        config.enabled_models = args.models
    
    if args.trials:
        for model in config.enabled_models:
            config.model_trials[model] = args.trials
    
    if args.timeout:
        config.optimization_timeout = args.timeout
    
    if args.cv_folds:
        config.cv_folds = args.cv_folds
    
    if args.gpu:
        config.enable_gpu = True
    
    if args.cpu:
        config.enable_gpu = False
    
    if args.memory_limit:
        config.max_memory_usage_gb = args.memory_limit
    
    if args.no_cache:
        config.enable_cache = False
    
    # Configurar directorios
    config.log_dir = args.log_dir
    config.results_dir = args.output
    
    return config


def run_diagnostics():
    """Ejecutar diagnósticos del sistema"""
    print("🔍 Diagnóstico del Sistema de Optimización")
    print("=" * 50)
    
    # Diagnóstico de imports
    diagnose_imports()
    
    # Test de imports críticos
    print("\n" + "=" * 50)
    if test_critical_imports():
        print("✅ Todos los imports críticos están disponibles")
        return True
    else:
        print("❌ Algunos imports críticos faltan")
        return False


def main():
    """Función principal"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Configurar logging
    logging_config = {
        "level": args.log_level,
        "log_dir": args.log_dir,
        "enable_file_logging": True,
        "enable_console_logging": True
    }
    setup_logging(logging_config)
    logger = get_logger("main")
    
    # Ejecutar diagnósticos si se solicita
    if args.diagnose:
        success = run_diagnostics()
        sys.exit(0 if success else 1)
    
    if args.test_imports:
        success = test_critical_imports()
        sys.exit(0 if success else 1)
    
    # Verificar que se proporcionó ruta de datos
    if not args.data_path:
        parser.error("Se requiere ruta de datos o usar --diagnose")
    
    # Verificar que el archivo existe
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"❌ Archivo de datos no encontrado: {data_path}")
        sys.exit(1)
    
    # Crear configuración
    config = get_config_from_args(args)
    
    # Mostrar configuración
    logger.info("🚀 Iniciando Optimización de Hiperparámetros - Fase 5")
    logger.info(f"📊 Archivo de datos: {data_path}")
    logger.info(f"🎯 Modo: {args.mode}")
    logger.info(f"🤖 Modelos: {config.enabled_models}")
    logger.info(f"🔄 Trials por modelo: {config.model_trials}")
    logger.info(f"⏱️  Timeout: {config.optimization_timeout}s")
    logger.info(f"🖥️  GPU habilitada: {config.enable_gpu}")
    logger.info(f"💾 Cache habilitado: {config.enable_cache}")
    logger.info(f"📁 Directorio de resultados: {config.results_dir}")
    
    try:
        # Limpiar cache si se solicita
        if args.clear_cache:
            from core.data_manager import DataManager
            data_manager = DataManager({'cache_dir': './cache'})
            data_manager.clear_cache()
            logger.info("🗑️  Cache limpiado")
        
        # Crear optimizador
        optimizer = HyperparameterOptimizer(config)
        
        # Ejecutar optimización
        logger.info("🏁 Iniciando optimización...")
        result = optimizer.optimize_all_models(
            str(data_path),
            experiment_id=args.experiment_id
        )
        
        # Guardar resultados
        optimizer.save_results(result, args.output)
        
        # Mostrar resumen
        logger.info("🎉 Optimización completada exitosamente!")
        logger.info(f"🏆 Mejor modelo: {result.best_model}")
        logger.info(f"📊 Mejor score: {result.best_score:.4f}")
        logger.info(f"⏱️  Tiempo total: {result.total_time:.2f}s")
        
        # Mostrar resultados por modelo
        print("\n📋 Resumen de Resultados:")
        print("=" * 50)
        for model_name, model_result in result.model_results.items():
            print(f"🤖 {model_name.upper()}:")
            print(f"   Score: {model_result.best_score:.4f}")
            print(f"   Trials: {model_result.n_trials}")
            print(f"   Tiempo: {model_result.optimization_time:.2f}s")
            print(f"   CV Score: {np.mean(model_result.cv_scores):.4f} ± {np.std(model_result.cv_scores):.4f}")
            print()
        
        # Mostrar información de datos
        print("📊 Información de Datos:")
        print(f"   Shape: {result.data_info.shape}")
        print(f"   Memoria: {result.data_info.memory_usage_mb:.2f} MB")
        print(f"   Distribución target: {result.data_info.target_distribution}")
        
        logger.info(f"💾 Resultados guardados en: {args.output}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Optimización interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error durante optimización: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import numpy as np  # Para estadísticas de CV
    main()
