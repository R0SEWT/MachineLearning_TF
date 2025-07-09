#!/usr/bin/env python3
"""
Test de las mejoras de Fase 1 del optimizador de hiperparámetros
"""

import sys
import os
from pathlib import Path

# Agregar paths para imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Test de importación de componentes"""
    print("🧪 TESTING IMPORTACIONES DE FASE 1")
    print("=" * 50)
    
    try:
        from config.optimization_config import CONFIG, MODEL_CONFIG
        print("✅ Configuración importada correctamente")
        print(f"   - Métrica primaria: {CONFIG.primary_metric}")
        print(f"   - CV folds: {CONFIG.cv_folds}")
        print(f"   - GPU preferido: {CONFIG.prefer_gpu}")
    except ImportError as e:
        print(f"❌ Error importando configuración: {e}")
        return False
    
    try:
        from utils.gpu_manager import GPUManager, GPU_MANAGER
        print("✅ GPU Manager importado correctamente")
        print(f"   - CUDA disponible: {GPU_MANAGER.cuda_available}")
        print(f"   - GPU info: {len(GPU_MANAGER.gpu_info)} componentes")
    except ImportError as e:
        print(f"❌ Error importando GPU Manager: {e}")
        return False
    
    try:
        from utils.data_validator import DataValidator, DataValidationError
        print("✅ Data Validator importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Data Validator: {e}")
        return False
    
    try:
        from utils.metrics_calculator import MetricsCalculator, MetricsResult
        print("✅ Metrics Calculator importado correctamente")
        calc = MetricsCalculator()
        print(f"   - Métricas disponibles: {len(calc.metrics_registry)}")
    except ImportError as e:
        print(f"❌ Error importando Metrics Calculator: {e}")
        return False
    
    try:
        from utils.optimization_logger import OptimizationLogger, get_optimization_logger
        print("✅ Optimization Logger importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Optimization Logger: {e}")
        return False
    
    return True

def test_gpu_manager():
    """Test del GPU Manager"""
    print("\n🎮 TESTING GPU MANAGER")
    print("=" * 50)
    
    try:
        from utils.gpu_manager import GPU_MANAGER
        
        # Test detección de hardware
        GPU_MANAGER.print_hardware_summary()
        
        # Test configuraciones
        print("\n🔧 Configuraciones de GPU:")
        try:
            xgb_config = GPU_MANAGER.get_xgboost_config()
            print(f"   XGBoost: {xgb_config}")
        except Exception as e:
            print(f"   XGBoost error: {e}")
        
        try:
            lgb_config = GPU_MANAGER.get_lightgbm_config()
            print(f"   LightGBM: {lgb_config}")
        except Exception as e:
            print(f"   LightGBM error: {e}")
        
        try:
            cb_config = GPU_MANAGER.get_catboost_config()
            print(f"   CatBoost: {cb_config}")
        except Exception as e:
            print(f"   CatBoost error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en GPU Manager: {e}")
        return False

def test_data_validator():
    """Test del Data Validator"""
    print("\n📊 TESTING DATA VALIDATOR")
    print("=" * 50)
    
    try:
        from utils.data_validator import DataValidator
        import numpy as np
        import pandas as pd
        
        validator = DataValidator()
        
        # Test con datos sintéticos
        print("Creando datos sintéticos para testing...")
        np.random.seed(42)
        
        # DataFrame de prueba
        test_data = {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        }
        df_test = pd.DataFrame(test_data)
        
        # Test validación de estructura
        df_info = validator.validate_dataframe_structure(df_test)
        print(f"✅ Estructura validada: {df_info['shape']}")
        
        # Test validación de target
        target_info = validator.validate_target_variable(df_test['target'])
        print(f"✅ Target validado: {target_info['value_counts']}")
        
        # Test validación de features
        X_test = df_test[['feature1', 'feature2', 'category']]
        feature_info = validator.validate_features(X_test)
        print(f"✅ Features validadas: {feature_info['n_features']} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en Data Validator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_calculator():
    """Test del Metrics Calculator"""
    print("\n📈 TESTING METRICS CALCULATOR")
    print("=" * 50)
    
    try:
        from utils.metrics_calculator import MetricsCalculator
        import numpy as np
        
        calc = MetricsCalculator()
        
        # Datos sintéticos
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100, p=[0.6, 0.4])
        y_pred = np.random.choice([0, 1], 100, p=[0.5, 0.5])
        y_proba = np.random.random(100)
        cv_scores = [0.75, 0.78, 0.72, 0.76, 0.74]
        
        # Test cálculo de métricas
        results = calc.calculate_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            cv_scores=cv_scores
        )
        
        print(f"✅ Métricas calculadas:")
        print(f"   - Score primario: {results.primary_score:.4f}")
        print(f"   - Score compuesto: {results.composite_score:.4f}")
        print(f"   - Métricas secundarias: {len(results.secondary_scores)}")
        
        # Test resumen
        summary = calc.get_metrics_summary(results)
        print(f"✅ Resumen generado: {len(summary)} caracteres")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en Metrics Calculator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_logger():
    """Test del Optimization Logger"""
    print("\n📝 TESTING OPTIMIZATION LOGGER")
    print("=" * 50)
    
    try:
        from utils.optimization_logger import get_optimization_logger
        
        # Crear logger de prueba
        logger = get_optimization_logger(
            log_dir="test_logs",
            log_level="INFO",
            enable_file_logging=True
        )
        
        print(f"✅ Logger creado: sesión {logger.session_id}")
        
        # Test diferentes tipos de logs
        logger.log_info("Test mensaje info", {"test": True})
        logger.log_warning("Test mensaje warning")
        logger.log_debug("Test mensaje debug")
        
        # Test logs específicos de optimización
        logger.log_optimization_start({"test_config": True})
        logger.log_trial_start(1, "test_model", {"param1": 0.5})
        logger.log_trial_complete(1, "test_model", 0.85, 10.5)
        
        # Test resumen
        summary = logger.get_session_summary()
        print(f"✅ Resumen de sesión: {summary['total_trials']} trials")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en Optimization Logger: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test de integración de componentes"""
    print("\n🔗 TESTING INTEGRACIÓN")
    print("=" * 50)
    
    try:
        # Test del optimizador principal
        from crypto_hyperparameter_optimizer import CryptoHyperparameterOptimizer
        
        # Crear optimizador sin datos reales
        optimizer = CryptoHyperparameterOptimizer(
            data_path="dummy_path.csv",  # No existe, solo para test
            results_path="test_results"
        )
        
        print("✅ Optimizador inicializado con componentes de Fase 1")
        print(f"   - Logger: {'✅' if optimizer.logger else '❌'}")
        print(f"   - GPU Manager: {'✅' if optimizer.gpu_manager else '❌'}")
        print(f"   - Data Validator: {'✅' if optimizer.data_validator else '❌'}")
        print(f"   - Metrics Calculator: {'✅' if optimizer.metrics_calculator else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en integración: {e}")
        return False

def main():
    """Ejecutar todos los tests"""
    print("🚀 TESTING COMPLETO DE FASE 1")
    print("🚀 OPTIMIZADOR DE HIPERPARÁMETROS")
    print("🚀" + "=" * 48)
    
    tests = [
        ("Importaciones", test_imports),
        ("GPU Manager", test_gpu_manager),
        ("Data Validator", test_data_validator),
        ("Metrics Calculator", test_metrics_calculator),
        ("Optimization Logger", test_optimization_logger),
        ("Integración", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 ERROR CRÍTICO en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print("\n🏁 RESUMEN DE TESTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{len(tests)} tests pasaron")
    
    if passed == len(tests):
        print("🎉 TODOS LOS TESTS DE FASE 1 PASARON!")
        return True
    else:
        print("⚠️ Algunos tests fallaron, revisar implementación")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
