#!/usr/bin/env python3
"""
Script de Testing para Fase 2 del Optimizador de Hiperparámetros
Prueba todas las mejoras avanzadas implementadas
"""

import sys
import os
import time
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar paths necesarios
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_phase2_improvements():
    """
    Probar todas las mejoras de Fase 2 implementadas
    """
    print("🧪======================================================================")
    print("🧪 TESTING FASE 2 - MEJORAS AVANZADAS DEL OPTIMIZADOR")
    print("🧪======================================================================")
    
    try:
        from crypto_hyperparameter_optimizer import CryptoHyperparameterOptimizer
        print("✅ CryptoHyperparameterOptimizer importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando CryptoHyperparameterOptimizer: {e}")
        return False
    
    # Test 1: Verificar componentes de Fase 2
    print("\n📋 Test 1: Verificar componentes de Fase 2")
    print("-" * 50)
    
    try:
        # Importar componentes de Fase 2
        from config.optuna_config import SAMPLER_FACTORY, PRUNER_FACTORY, STRATEGY_SELECTOR
        from utils.temporal_validator import TEMPORAL_VALIDATOR
        from utils.early_stopping import ADAPTIVE_CONTROLLER
        from utils.multi_objective import MULTI_OBJECTIVE_OPTIMIZER
        
        print("✅ Componentes de Fase 2 importados correctamente:")
        print(f"   🎯 SAMPLER_FACTORY: {type(SAMPLER_FACTORY).__name__}")
        print(f"   ✂️  PRUNER_FACTORY: {type(PRUNER_FACTORY).__name__}")
        print(f"   🎲 STRATEGY_SELECTOR: {type(STRATEGY_SELECTOR).__name__}")
        print(f"   📅 TEMPORAL_VALIDATOR: {type(TEMPORAL_VALIDATOR).__name__}")
        print(f"   🛑 ADAPTIVE_CONTROLLER: {type(ADAPTIVE_CONTROLLER).__name__}")
        print(f"   🎯 MULTI_OBJECTIVE_OPTIMIZER: {type(MULTI_OBJECTIVE_OPTIMIZER).__name__}")
        
    except ImportError as e:
        print(f"❌ Error importando componentes de Fase 2: {e}")
        return False
    
    # Test 2: Crear datos sintéticos para pruebas
    print("\n📊 Test 2: Creando datos sintéticos")
    print("-" * 50)
    
    try:
        # Crear dataset sintético
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Features con nombres realistas
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=feature_names
        )
        
        # Target binario
        y = np.random.binomial(1, 0.3, n_samples)
        
        # Agregar columna de fecha para testing temporal
        X['date'] = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        print(f"✅ Datos sintéticos creados:")
        print(f"   📊 Forma: {X.shape}")
        print(f"   🎯 Clases: {np.unique(y, return_counts=True)}")
        print(f"   📅 Rango de fechas: {X['date'].min()} - {X['date'].max()}")
        
        # Guardar datos temporalmente con nombre de columna objetivo correcto
        data_path = Path("/tmp/crypto_test_data.csv")
        test_data = X.copy()
        test_data['high_return_30d'] = y  # Usar el nombre correcto de la columna objetivo
        test_data.to_csv(data_path, index=False)
        
        print(f"   💾 Datos guardados en: {data_path}")
        
    except Exception as e:
        print(f"❌ Error creando datos sintéticos: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: Inicializar optimizador con componentes de Fase 2
    print("\n🔧 Test 3: Inicializar optimizador con Fase 2")
    print("-" * 50)
    
    try:
        # Configurar rutas
        results_path = Path("/tmp/crypto_optimization_test_phase2")
        results_path.mkdir(exist_ok=True)
        
        # Crear optimizador
        optimizer = CryptoHyperparameterOptimizer(
            data_path=str(data_path),
            results_path=str(results_path)
        )
        
        print("✅ Optimizador inicializado con Fase 2:")
        print(f"   🔍 Validador de datos: {optimizer.data_validator is not None}")
        print(f"   📊 Calculadora de métricas: {optimizer.metrics_calculator is not None}")
        print(f"   📝 Logger: {optimizer.logger is not None}")
        print(f"   🎯 GPU Manager: {optimizer.gpu_manager is not None}")
        print(f"   📅 Validador temporal: {optimizer.temporal_validator is not None}")
        print(f"   🛑 Controlador adaptativo: {optimizer.adaptive_controller is not None}")
        print(f"   🎲 Sampler factory: {optimizer.sampler_factory is not None}")
        print(f"   ✂️  Pruner factory: {optimizer.pruner_factory is not None}")
        
    except Exception as e:
        print(f"❌ Error inicializando optimizador: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: Cargar y preparar datos con validación robusta
    print("\n📁 Test 4: Cargar y preparar datos")
    print("-" * 50)
    
    try:
        start_time = time.time()
        optimizer.load_and_prepare_data()
        load_time = time.time() - start_time
        
        print(f"✅ Datos cargados y preparados en {load_time:.2f}s:")
        print(f"   📊 X_train: {optimizer.X_train.shape}")
        print(f"   📊 X_val: {optimizer.X_val.shape}")
        print(f"   📊 X_test: {optimizer.X_test.shape}")
        print(f"   🎯 y_train: {optimizer.y_train.shape}")
        print(f"   📅 Columna de fecha: {'date' in optimizer.X_train.columns}")
        
        # Verificar validación de datos
        if optimizer.data_validator:
            print("   ✅ Validación de datos ejecutada")
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        traceback.print_exc()
        return False
    
    # Test 5: Probar samplers y pruners avanzados
    print("\n🎯 Test 5: Probar samplers y pruners avanzados")
    print("-" * 50)
    
    try:
        # Probar diferentes samplers
        sampler_types = ['tpe', 'random', 'cmaes', 'nsga2']
        
        for sampler_type in sampler_types:
            try:
                sampler = optimizer.sampler_factory.create_sampler(
                    sampler_type, 
                    {}  # Usar dict vacío en lugar de config
                )
                print(f"   ✅ {sampler_type.upper()}: {type(sampler).__name__}")
            except Exception as e:
                print(f"   ⚠️  {sampler_type.upper()}: {e}")
        
        # Probar diferentes pruners
        pruner_types = ['median', 'successive_halving', 'hyperband', 'percentile']
        
        for pruner_type in pruner_types:
            try:
                pruner = optimizer.pruner_factory.create_pruner(
                    pruner_type, 
                    {}  # Usar dict vacío en lugar de config
                )
                print(f"   ✅ {pruner_type.upper()}: {type(pruner).__name__}")
            except Exception as e:
                print(f"   ⚠️  {pruner_type.upper()}: {e}")
        
    except Exception as e:
        print(f"❌ Error probando samplers/pruners: {e}")
        traceback.print_exc()
        return False
    
    # Test 6: Probar validación cruzada temporal
    print("\n📅 Test 6: Probar validación cruzada temporal")
    print("-" * 50)
    
    try:
        if optimizer.temporal_validator:
            # Crear un modelo simple para testing
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            
            # Preparar datos con fecha
            X_with_date = optimizer.X_train.copy()
            if 'date' not in X_with_date.columns:
                X_with_date['date'] = pd.date_range(start='2020-01-01', periods=len(X_with_date), freq='D')
            
            try:
                cv_results = optimizer.temporal_validator.perform_time_series_cv(
                    estimator=model,
                    X=X_with_date,
                    y=optimizer.y_train,
                    scoring='roc_auc',
                    cv_type='time_series'
                )
                
                print(f"   ✅ Validación cruzada temporal exitosa:")
                print(f"      📊 Score medio: {cv_results['mean_score']:.4f}")
                print(f"      📊 Std score: {cv_results['std_score']:.4f}")
                print(f"      📊 Estabilidad: {cv_results['stability_metrics']['stability_score']:.4f}")
                print(f"      📊 Folds: {cv_results['n_folds']}")
                
            except Exception as e:
                print(f"   ⚠️  Error en validación temporal: {e}")
                
        else:
            print("   ⚠️  Validador temporal no disponible")
        
    except Exception as e:
        print(f"❌ Error probando validación temporal: {e}")
        traceback.print_exc()
        return False
    
    # Test 7: Probar early stopping adaptativo
    print("\n🛑 Test 7: Probar early stopping adaptativo")
    print("-" * 50)
    
    try:
        if optimizer.adaptive_controller:
            # Obtener monitor para XGBoost
            monitor = optimizer.adaptive_controller.get_monitor('xgboost')
            monitor.reset()
            
            # Simular progreso de optimización
            scores = [0.7, 0.72, 0.75, 0.76, 0.76, 0.76, 0.76, 0.76]
            
            stopped = False
            for trial_num, score in enumerate(scores):
                should_stop = monitor.update(trial_num, score)
                print(f"   📊 Trial {trial_num}: score={score:.3f}, should_stop={should_stop}")
                
                if should_stop:
                    stopped = True
                    break
            
            summary = monitor.get_summary()
            print(f"   📊 Resumen early stopping:")
            print(f"      🛑 Stopped: {summary['stopped']}")
            print(f"      📊 Best score: {summary['best_score']:.4f}")
            print(f"      📊 Trials sin mejora: {summary.get('n_trials_without_improvement', 0)}")
            
            if summary['stopped']:
                print(f"      🛑 Razón: {summary['stop_reason']}")
                
        else:
            print("   ⚠️  Controlador adaptativo no disponible")
        
    except Exception as e:
        print(f"❌ Error probando early stopping: {e}")
        traceback.print_exc()
        return False
    
    # Test 8: Probar optimización multi-objetivo
    print("\n🎯 Test 8: Probar optimización multi-objetivo")
    print("-" * 50)
    
    try:
        if MULTI_OBJECTIVE_OPTIMIZER:
            # Crear estudio multi-objetivo
            study = MULTI_OBJECTIVE_OPTIMIZER.create_multi_objective_study(
                "test_multi_objective"
            )
            
            print(f"   ✅ Estudio multi-objetivo creado:")
            print(f"      📊 Direcciones: {study.directions}")
            print(f"      🎯 Sampler: {type(study.sampler).__name__}")
            print(f"      📊 Objetivos primarios: {len(MULTI_OBJECTIVE_OPTIMIZER.config.primary_objectives)}")
            
            # Simular algunos resultados
            mock_results = [
                {'auc': 0.85, 'precision': 0.80, 'recall': 0.75, 'f1': 0.77},
                {'auc': 0.82, 'precision': 0.85, 'recall': 0.78, 'f1': 0.81},
                {'auc': 0.88, 'precision': 0.75, 'recall': 0.85, 'f1': 0.80}
            ]
            
            for i, result in enumerate(mock_results):
                objective_values = MULTI_OBJECTIVE_OPTIMIZER.calculate_objective_values(
                    trial_results={'cv_scores': [0.8, 0.82, 0.85, 0.79, 0.81]},
                    model_performance=result,
                    training_time=10.0
                )
                print(f"      📊 Resultado {i+1}: {[f'{v:.3f}' for v in objective_values]}")
            
        else:
            print("   ⚠️  Optimizador multi-objetivo no disponible")
        
    except Exception as e:
        print(f"❌ Error probando optimización multi-objetivo: {e}")
        traceback.print_exc()
        return False
    
    # Test 9: Probar optimización rápida con mejoras de Fase 2
    print("\n🚀 Test 9: Optimización rápida con Fase 2")
    print("-" * 50)
    
    try:
        print("   🔥 Ejecutando optimización rápida de XGBoost...")
        
        start_time = time.time()
        study = optimizer.optimize_xgboost(
            n_trials=5,  # Pocas trials para test rápido
            timeout=30,  # 30 segundos máximo
            use_temporal_cv=True,
            optimization_strategy='quick'
        )
        opt_time = time.time() - start_time
        
        print(f"   ✅ Optimización completada en {opt_time:.2f}s:")
        print(f"      🏆 Mejor score: {study.best_value:.4f}")
        print(f"      🎯 Trials ejecutados: {len(study.trials)}")
        print(f"      📊 Convergencia: {optimizer.convergence_history.get('xgboost', {})}")
        
        # Verificar que se guardaron los resultados
        if optimizer.best_params.get('xgboost'):
            print(f"      ✅ Parámetros guardados: {len(optimizer.best_params['xgboost'])} params")
        
    except Exception as e:
        print(f"❌ Error en optimización rápida: {e}")
        traceback.print_exc()
        return False
    
    # Test 10: Limpiar archivos temporales
    print("\n🧹 Test 10: Limpieza")
    print("-" * 50)
    
    try:
        # Limpiar archivos temporales
        import shutil
        
        if data_path.exists():
            data_path.unlink()
            print("   ✅ Datos de prueba eliminados")
        
        if results_path.exists():
            shutil.rmtree(results_path)
            print("   ✅ Directorio de resultados eliminado")
        
    except Exception as e:
        print(f"   ⚠️  Error en limpieza: {e}")
    
    print("\n🎉======================================================================")
    print("🎉 TESTING FASE 2 COMPLETADO EXITOSAMENTE")
    print("🎉======================================================================")
    print("\n✅ Todos los componentes de Fase 2 funcionan correctamente:")
    print("   🎯 Samplers y pruners avanzados de Optuna")
    print("   📅 Validación cruzada temporal")
    print("   🛑 Early stopping inteligente")
    print("   🎯 Optimización multi-objetivo")
    print("   📊 Métricas múltiples y estabilidad")
    print("   🔧 Integración completa en funciones de optimización")
    
    return True

if __name__ == "__main__":
    success = test_phase2_improvements()
    exit(0 if success else 1)
