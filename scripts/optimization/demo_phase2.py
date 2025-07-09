#!/usr/bin/env python3
"""
Demostración Completa del Optimizador de Hiperparámetros - Fase 2
Muestra todas las capacidades avanzadas implementadas
"""

import sys
import os
import time
from pathlib import Path
import argparse

# Agregar paths necesarios
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def demo_quick_optimization():
    """Demostración de optimización rápida"""
    print("🚀 DEMO: Optimización Rápida (5 minutos)")
    print("="*60)
    
    try:
        from crypto_hyperparameter_optimizer import CryptoHyperparameterOptimizer
        
        # Configurar paths
        data_path = "/home/exodia/Documentos/MachineLearning_TF/data/ml_dataset.csv"
        results_path = Path("/tmp/demo_quick_optimization")
        results_path.mkdir(exist_ok=True)
        
        # Crear optimizador
        optimizer = CryptoHyperparameterOptimizer(
            data_path=data_path,
            results_path=str(results_path)
        )
        
        # Cargar datos
        print("📁 Cargando datos...")
        optimizer.load_and_prepare_data()
        
        # Optimización rápida de XGBoost
        print("🔥 Optimizando XGBoost (estrategia rápida)...")
        start_time = time.time()
        
        study = optimizer.optimize_xgboost(
            n_trials=20,
            timeout=300,  # 5 minutos
            use_temporal_cv=True,
            optimization_strategy='quick'
        )
        
        duration = time.time() - start_time
        
        print(f"\n✅ Optimización completada en {duration:.1f}s")
        print(f"🏆 Mejor AUC: {study.best_value:.4f}")
        print(f"🎯 Trials ejecutados: {len(study.trials)}")
        
        # Mostrar información de convergencia
        if optimizer.convergence_history.get('xgboost'):
            conv_info = optimizer.convergence_history['xgboost']
            print(f"📊 Early stopping: {conv_info.get('stopped', False)}")
            if conv_info.get('stopped'):
                print(f"🛑 Razón: {conv_info.get('stop_reason', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demostración rápida: {e}")
        return False

def demo_balanced_optimization():
    """Demostración de optimización balanceada"""
    print("\n⚖️ DEMO: Optimización Balanceada (15 minutos)")
    print("="*60)
    
    try:
        from crypto_hyperparameter_optimizer import CryptoHyperparameterOptimizer
        
        # Configurar paths
        data_path = "/home/exodia/Documentos/MachineLearning_TF/data/ml_dataset.csv"
        results_path = Path("/tmp/demo_balanced_optimization")
        results_path.mkdir(exist_ok=True)
        
        # Crear optimizador
        optimizer = CryptoHyperparameterOptimizer(
            data_path=data_path,
            results_path=str(results_path)
        )
        
        # Cargar datos
        print("📁 Cargando datos...")
        optimizer.load_and_prepare_data()
        
        # Optimización balanceada de todos los modelos
        print("🔥 Optimizando todos los modelos (estrategia balanceada)...")
        start_time = time.time()
        
        optimizer.optimize_all_models(
            n_trials=50,
            timeout_per_model=300,  # 5 minutos por modelo
            use_temporal_cv=True,
            optimization_strategy='balanced'
        )
        
        duration = time.time() - start_time
        
        print(f"\n✅ Optimización completa en {duration:.1f}s")
        
        # Mostrar resultados de todos los modelos
        print("\n📊 RESUMEN DE RESULTADOS:")
        print("-" * 40)
        
        for model_name in optimizer.best_scores.keys():
            score = optimizer.best_scores[model_name]
            print(f"🏆 {model_name.upper()}: AUC = {score:.4f}")
            
            # Información de convergencia
            if optimizer.convergence_history.get(model_name):
                conv_info = optimizer.convergence_history[model_name]
                if conv_info.get('stopped'):
                    print(f"   🛑 Early stopping: {conv_info.get('stop_reason', 'Unknown')}")
        
        # Evaluar mejores modelos
        print("\n📈 Evaluando mejores modelos...")
        optimizer.evaluate_best_models()
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demostración balanceada: {e}")
        return False

def demo_advanced_features():
    """Demostración de características avanzadas"""
    print("\n🎯 DEMO: Características Avanzadas")
    print("="*60)
    
    try:
        # 1. Probar samplers avanzados
        print("🎲 Probando samplers avanzados...")
        from config.optuna_config import SAMPLER_FACTORY
        
        samplers_to_test = ['tpe', 'cmaes', 'random', 'nsga2']
        for sampler_type in samplers_to_test:
            try:
                sampler = SAMPLER_FACTORY.create_sampler(sampler_type, {})
                print(f"   ✅ {sampler_type.upper()}: {type(sampler).__name__}")
            except Exception as e:
                print(f"   ⚠️  {sampler_type.upper()}: {e}")
        
        # 2. Probar pruners avanzados
        print("\n✂️  Probando pruners avanzados...")
        from config.optuna_config import PRUNER_FACTORY
        
        pruners_to_test = ['median', 'successive_halving', 'hyperband', 'percentile']
        for pruner_type in pruners_to_test:
            try:
                pruner = PRUNER_FACTORY.create_pruner(pruner_type, {})
                print(f"   ✅ {pruner_type.upper()}: {type(pruner).__name__}")
            except Exception as e:
                print(f"   ⚠️  {pruner_type.upper()}: {e}")
        
        # 3. Probar validación temporal
        print("\n📅 Probando validación temporal...")
        from utils.temporal_validator import TEMPORAL_VALIDATOR
        print(f"   ✅ Validador temporal: {type(TEMPORAL_VALIDATOR).__name__}")
        
        # 4. Probar early stopping
        print("\n🛑 Probando early stopping...")
        from utils.early_stopping import ADAPTIVE_CONTROLLER
        monitor = ADAPTIVE_CONTROLLER.get_monitor('test_model')
        print(f"   ✅ Controlador adaptativo: {type(ADAPTIVE_CONTROLLER).__name__}")
        print(f"   ✅ Monitor creado: {type(monitor).__name__}")
        
        # 5. Probar optimización multi-objetivo
        print("\n🎯 Probando optimización multi-objetivo...")
        from utils.multi_objective import MULTI_OBJECTIVE_OPTIMIZER
        
        # Crear estudio multi-objetivo
        study = MULTI_OBJECTIVE_OPTIMIZER.create_multi_objective_study("demo_study")
        print(f"   ✅ Estudio multi-objetivo: {len(study.directions)} objetivos")
        print(f"   ✅ Sampler: {type(study.sampler).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demostración de características avanzadas: {e}")
        return False

def demo_visualization():
    """Demostración de visualizaciones (si están disponibles)"""
    print("\n📊 DEMO: Visualizaciones")
    print("="*60)
    
    try:
        # Intentar importar matplotlib
        import matplotlib.pyplot as plt
        import optuna.visualization as vis
        
        print("✅ Capacidades de visualización disponibles:")
        print("   📈 Optuna visualization")
        print("   📊 Matplotlib")
        
        # Crear un estudio dummy para demostración
        import optuna
        study = optuna.create_study(direction='maximize')
        
        # Agregar algunos trials dummy
        for i in range(10):
            trial = study.ask()
            score = 0.8 + 0.1 * (i / 10) + 0.05 * ((-1) ** i)
            study.tell(trial, score)
        
        print(f"   📊 Estudio demo creado con {len(study.trials)} trials")
        print("   💡 Visualizaciones disponibles:")
        print("      - plot_optimization_history")
        print("      - plot_param_importances")
        print("      - plot_contour")
        print("      - plot_slice")
        print("      - plot_parallel_coordinate")
        
        return True
        
    except ImportError:
        print("⚠️  Matplotlib no disponible - visualizaciones limitadas")
        return False

def main():
    """Función principal de demostración"""
    parser = argparse.ArgumentParser(description="Demostración del Optimizador - Fase 2")
    parser.add_argument("--mode", choices=['quick', 'balanced', 'advanced', 'all'], 
                      default='quick', help="Modo de demostración")
    parser.add_argument("--data-path", type=str, 
                      default="/home/exodia/Documentos/MachineLearning_TF/data/ml_dataset.csv",
                      help="Ruta a los datos")
    
    args = parser.parse_args()
    
    print("🎭======================================================================")
    print("🎭 DEMOSTRACIÓN COMPLETA - OPTIMIZADOR FASE 2")
    print("🎭======================================================================")
    print(f"   🎯 Modo: {args.mode}")
    print(f"   📁 Datos: {args.data_path}")
    
    # Verificar que existen los datos
    if not os.path.exists(args.data_path):
        print(f"❌ Archivo de datos no encontrado: {args.data_path}")
        print("💡 Usa datos sintéticos o especifica --data-path")
        return False
    
    success = True
    
    if args.mode == 'quick' or args.mode == 'all':
        success &= demo_quick_optimization()
    
    if args.mode == 'balanced' or args.mode == 'all':
        success &= demo_balanced_optimization()
    
    if args.mode == 'advanced' or args.mode == 'all':
        success &= demo_advanced_features()
        success &= demo_visualization()
    
    # Resumen final
    print("\n🎉======================================================================")
    if success:
        print("🎉 DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
        print("🎉======================================================================")
        print("\n✅ Capacidades demostradas:")
        print("   🚀 Optimización con mejoras de Fase 2")
        print("   🎯 Samplers y pruners avanzados")
        print("   📅 Validación cruzada temporal")
        print("   🛑 Early stopping inteligente")
        print("   🎯 Optimización multi-objetivo")
        print("   📊 Métricas múltiples y análisis")
        print("\n📖 Para más información:")
        print("   📄 README_PHASE2.md")
        print("   🧪 test_phase2_improvements.py")
    else:
        print("❌ DEMOSTRACIÓN COMPLETADA CON ERRORES")
        print("❌======================================================================")
        print("\n🔍 Revisar:")
        print("   📦 Dependencias instaladas")
        print("   📁 Rutas de datos correctas")
        print("   🔧 Configuración del entorno")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
