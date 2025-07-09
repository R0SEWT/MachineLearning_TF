"""
🚀 Sistema de Testing Completo - Fase 5
=======================================

Suite completa de tests enterprise-ready que reemplaza el testing limitado
del sistema anterior con tests unitarios, de integración y de performance.

Autor: Sistema de Optimización IA
Fecha: 2025-01-09 (Fase 5 - Organización)
"""

import unittest
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil
import json
import pandas as pd
import numpy as np

# Agregar path para imports locales
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.optimization_config import OptimizationConfig, get_quick_config
from utils.logging_setup import setup_logging, get_logger
from utils.import_manager import ImportManager, test_critical_imports
from core.data_manager import DataManager
from core.optimizer import HyperparameterOptimizer
from analysis.results_analyzer import ResultsAnalyzer


class TestOptimizationConfig(unittest.TestCase):
    """Tests para configuración de optimización"""
    
    def setUp(self):
        """Configuración inicial de tests"""
        self.config = OptimizationConfig()
    
    def test_config_creation(self):
        """Test creación de configuración básica"""
        self.assertIsInstance(self.config, OptimizationConfig)
        self.assertIsInstance(self.config.enabled_models, list)
        self.assertGreater(len(self.config.enabled_models), 0)
    
    def test_config_validation(self):
        """Test validación de configuración"""
        # Test configuración válida
        valid_config = OptimizationConfig(test_size=0.2, validation_size=0.15)
        # Debería pasar sin errores
        
        # Test configuración inválida
        with self.assertRaises(ValueError):
            OptimizationConfig(test_size=0.8, validation_size=0.8)  # Suma > 1.0
    
    def test_model_config_generation(self):
        """Test generación de configuración por modelo"""
        for model_name in self.config.enabled_models:
            model_config = self.config.get_model_config(model_name)
            self.assertIsInstance(model_config, dict)
            self.assertIn("random_state", model_config)
    
    def test_config_serialization(self):
        """Test serialización/deserialización de configuración"""
        config_dict = self.config.to_dict()
        self.assertIsInstance(config_dict, dict)
        
        # Test deserialización
        restored_config = OptimizationConfig.from_dict(config_dict)
        self.assertEqual(self.config.enabled_models, restored_config.enabled_models)
    
    def test_predefined_configs(self):
        """Test configuraciones predefinidas"""
        quick_config = get_quick_config()
        self.assertIsInstance(quick_config, OptimizationConfig)
        
        # Quick config debe tener menos trials
        self.assertLessEqual(max(quick_config.model_trials.values()), 50)


class TestImportManager(unittest.TestCase):
    """Tests para gestor de imports"""
    
    def setUp(self):
        """Configuración inicial"""
        self.import_manager = ImportManager()
    
    def test_critical_imports(self):
        """Test imports críticos del sistema"""
        # Este test puede fallar si no están instaladas las dependencias
        # pero debería al menos ejecutar sin errores
        result = test_critical_imports()
        self.assertIsInstance(result, bool)
    
    def test_safe_import(self):
        """Test import seguro"""
        # Test import existente
        os_module = self.import_manager.safe_import("os")
        self.assertIsNotNone(os_module)
        
        # Test import inexistente
        fake_module = self.import_manager.safe_import("fake_module_that_does_not_exist")
        self.assertIsNone(fake_module)
    
    def test_import_status(self):
        """Test estado de imports"""
        status = self.import_manager.get_import_status()
        self.assertIsInstance(status, dict)
        self.assertIn("cached_modules", status)
        self.assertIn("failed_imports", status)
        self.assertIn("search_paths", status)


class TestDataManager(unittest.TestCase):
    """Tests para gestor de datos"""
    
    def setUp(self):
        """Configuración inicial"""
        self.data_manager = DataManager()
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data_path = self._create_sample_data()
    
    def tearDown(self):
        """Limpieza después de tests"""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_data(self) -> str:
        """Crear datos de muestra para testing"""
        np.random.seed(42)
        n_samples, n_features = 1000, 10
        
        # Crear features sintéticas
        features = np.random.randn(n_samples, n_features)
        target = np.random.randint(0, 2, n_samples)
        
        # Crear DataFrame
        df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
        df["target_next_close_positive"] = target
        
        # Guardar CSV
        data_path = Path(self.temp_dir) / "sample_data.csv"
        df.to_csv(data_path, index=False)
        
        return str(data_path)
    
    def test_data_loading(self):
        """Test carga básica de datos"""
        features, target, info = self.data_manager.load_data(self.sample_data_path)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(target, pd.Series)
        self.assertGreater(len(features), 0)
        self.assertEqual(len(features), len(target))
    
    def test_data_preprocessing(self):
        """Test preprocesamiento de datos"""
        features, target, info = self.data_manager.load_data(
            self.sample_data_path,
            normalize_features=True,
            handle_missing="drop"
        )
        
        # Verificar que no hay valores faltantes
        self.assertEqual(features.isnull().sum().sum(), 0)
        self.assertEqual(target.isnull().sum(), 0)
    
    def test_train_test_split(self):
        """Test división de datos"""
        features, target, _ = self.data_manager.load_data(self.sample_data_path)
        
        splits = self.data_manager.get_train_val_test_split(features, target)
        
        required_keys = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
        for key in required_keys:
            self.assertIn(key, splits)
        
        # Verificar tamaños
        total_samples = len(features)
        split_total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
        self.assertEqual(total_samples, split_total)
    
    def test_cache_functionality(self):
        """Test funcionalidad de cache"""
        # Primera carga (sin cache)
        start_time = time.time()
        features1, target1, info1 = self.data_manager.load_data(self.sample_data_path)
        first_load_time = time.time() - start_time
        
        # Segunda carga (con cache)
        start_time = time.time()
        features2, target2, info2 = self.data_manager.load_data(self.sample_data_path)
        second_load_time = time.time() - start_time
        
        # Cache debería ser más rápido
        self.assertLess(second_load_time, first_load_time)
        
        # Datos deberían ser idénticos
        pd.testing.assert_frame_equal(features1, features2)
        pd.testing.assert_series_equal(target1, target2)


class TestOptimizer(unittest.TestCase):
    """Tests para optimizador principal"""
    
    def setUp(self):
        """Configuración inicial"""
        # Configuración mínima para tests rápidos
        self.config = get_quick_config()
        self.config.model_trials = {"xgboost": 3}  # Muy pocos trials para tests
        self.config.optimization_timeout = 30  # 30 segundos máximo
        self.config.enabled_models = ["xgboost"]  # Solo un modelo para tests
        
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data_path = self._create_sample_data()
    
    def tearDown(self):
        """Limpieza después de tests"""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_data(self) -> str:
        """Crear datos de muestra"""
        np.random.seed(42)
        n_samples, n_features = 500, 5  # Dataset pequeño para tests rápidos
        
        features = np.random.randn(n_samples, n_features)
        target = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
        df["target_next_close_positive"] = target
        
        data_path = Path(self.temp_dir) / "test_data.csv"
        df.to_csv(data_path, index=False)
        
        return str(data_path)
    
    def test_optimizer_initialization(self):
        """Test inicialización del optimizador"""
        try:
            optimizer = HyperparameterOptimizer(self.config)
            self.assertIsInstance(optimizer, HyperparameterOptimizer)
            self.assertGreater(len(optimizer.model_handlers), 0)
        except ImportError:
            self.skipTest("Dependencias de ML no disponibles")
    
    @unittest.skipIf(not test_critical_imports(), "Dependencias críticas no disponibles")
    def test_single_model_optimization(self):
        """Test optimización de un solo modelo"""
        try:
            optimizer = HyperparameterOptimizer(self.config)
            
            # Cargar datos
            data_manager = DataManager()
            features, target, _ = data_manager.load_data(self.sample_data_path)
            
            # Optimizar
            result = optimizer.optimize_single_model(
                "xgboost", features, target, "test_experiment"
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result.model_name, "xgboost")
            self.assertGreater(result.best_score, 0)
            self.assertIsInstance(result.best_params, dict)
            
        except ImportError:
            self.skipTest("XGBoost no disponible")
    
    @unittest.skipIf(not test_critical_imports(), "Dependencias críticas no disponibles")
    def test_full_optimization(self):
        """Test optimización completa (solo si hay dependencias)"""
        try:
            optimizer = HyperparameterOptimizer(self.config)
            
            result = optimizer.optimize_all_models(
                self.sample_data_path, "test_full_experiment"
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result.experiment_id, "test_full_experiment")
            self.assertGreater(len(result.model_results), 0)
            self.assertIsNotNone(result.best_model)
            
        except ImportError:
            self.skipTest("Dependencias de ML no disponibles")


class TestResultsAnalyzer(unittest.TestCase):
    """Tests para analizador de resultados"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = ResultsAnalyzer(self.temp_dir)
        self._create_sample_results()
    
    def tearDown(self):
        """Limpieza"""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_results(self):
        """Crear resultados de muestra"""
        sample_result = {
            "experiment_id": "test_experiment_001",
            "timestamp": "2025-01-09T10:00:00",
            "total_time": 120.5,
            "best_model": "xgboost",
            "best_score": 0.85,
            "data_info": {
                "shape": [1000, 10],
                "memory_usage_mb": 5.2
            },
            "model_results": {
                "xgboost": {
                    "best_score": 0.85,
                    "best_params": {"n_estimators": 100, "max_depth": 6},
                    "n_trials": 50,
                    "optimization_time": 60.2,
                    "cv_scores": [0.82, 0.85, 0.83, 0.86, 0.84],
                    "feature_importance": {"feature_0": 0.3, "feature_1": 0.2}
                }
            }
        }
        
        results_file = Path(self.temp_dir) / "results_test_experiment_001_20250109_100000.json"
        with open(results_file, 'w') as f:
            json.dump(sample_result, f)
    
    def test_load_experiments(self):
        """Test carga de experimentos"""
        count = self.analyzer.load_experiments()
        self.assertEqual(count, 1)
        self.assertIn("test_experiment_001", self.analyzer.experiments)
    
    def test_experiment_summary(self):
        """Test resumen de experimentos"""
        self.analyzer.load_experiments()
        summary = self.analyzer.get_experiment_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
        self.assertIn("experiment_id", summary.columns)
        self.assertIn("best_score", summary.columns)
    
    def test_model_comparison(self):
        """Test comparación de modelos"""
        self.analyzer.load_experiments()
        comparison = self.analyzer.get_model_comparison()
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertGreater(len(comparison), 0)
        self.assertIn("model_name", comparison.columns)
    
    def test_best_parameters(self):
        """Test obtención de mejores parámetros"""
        self.analyzer.load_experiments()
        best_params = self.analyzer.get_best_parameters("xgboost")
        
        self.assertIsInstance(best_params, dict)
        self.assertIn("n_estimators", best_params)
        self.assertIn("_score", best_params)
    
    def test_statistics(self):
        """Test estadísticas generales"""
        self.analyzer.load_experiments()
        stats = self.analyzer.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_experiments", stats)
        self.assertIn("score_statistics", stats)
        self.assertIn("time_statistics", stats)


class TestPerformance(unittest.TestCase):
    """Tests de performance del sistema"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpieza"""
        shutil.rmtree(self.temp_dir)
    
    def test_memory_usage(self):
        """Test uso de memoria"""
        # Crear dataset grande para test de memoria
        n_samples = 10000
        n_features = 50
        
        np.random.seed(42)
        features = np.random.randn(n_samples, n_features)
        target = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
        df["target_next_close_positive"] = target
        
        data_path = Path(self.temp_dir) / "large_data.csv"
        df.to_csv(data_path, index=False)
        
        # Test carga con monitoring de memoria
        import psutil
        process = psutil.Process()
        
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        data_manager = DataManager()
        features, target, info = data_manager.load_data(str(data_path))
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # La carga no debería usar más de 100MB adicionales
        self.assertLess(memory_increase, 100)
        
        # Limpiar memoria
        del features, target, info
        gc.collect()
    
    def test_loading_speed(self):
        """Test velocidad de carga"""
        # Crear dataset mediano
        n_samples = 5000
        n_features = 20
        
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        df["target_next_close_positive"] = np.random.randint(0, 2, n_samples)
        
        data_path = Path(self.temp_dir) / "speed_test_data.csv"
        df.to_csv(data_path, index=False)
        
        # Test velocidad de primera carga
        data_manager = DataManager()
        start_time = time.time()
        features, target, info = data_manager.load_data(str(data_path))
        first_load_time = time.time() - start_time
        
        # Test velocidad de segunda carga (con cache)
        start_time = time.time()
        features2, target2, info2 = data_manager.load_data(str(data_path))
        second_load_time = time.time() - start_time
        
        # Primera carga debería ser < 5 segundos
        self.assertLess(first_load_time, 5.0)
        
        # Segunda carga (cache) debería ser < 1 segundo
        self.assertLess(second_load_time, 1.0)
        
        # Cache debería ser significativamente más rápido
        self.assertLess(second_load_time, first_load_time / 2)


def run_test_suite(verbosity: int = 2, specific_test: Optional[str] = None) -> bool:
    """
    Ejecutar suite completa de tests.
    
    Args:
        verbosity: Nivel de verbosidad (0-2)
        specific_test: Test específico a ejecutar
        
    Returns:
        True si todos los tests pasan
    """
    # Configurar logging para tests
    setup_logging({"level": "WARNING"})  # Reducir noise durante tests
    
    # Crear test loader
    loader = unittest.TestLoader()
    
    if specific_test:
        # Ejecutar test específico
        suite = loader.loadTestsFromName(specific_test)
    else:
        # Ejecutar todos los tests
        test_classes = [
            TestOptimizationConfig,
            TestImportManager,
            TestDataManager,
            TestOptimizer,
            TestResultsAnalyzer,
            TestPerformance
        ]
        
        suite = unittest.TestSuite()
        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Mostrar resumen
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    print(f"\n{'='*50}")
    print(f"📊 Resumen de Tests:")
    print(f"   Total ejecutados: {total_tests}")
    print(f"   ✅ Exitosos: {total_tests - failures - errors}")
    print(f"   ❌ Fallidos: {failures}")
    print(f"   💥 Errores: {errors}")
    print(f"   ⏭️  Saltados: {skipped}")
    print(f"{'='*50}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="🧪 Suite de Tests - Sistema de Optimización")
    parser.add_argument("--verbose", "-v", action="count", default=1, help="Nivel de verbosidad")
    parser.add_argument("--test", "-t", help="Test específico a ejecutar")
    parser.add_argument("--quick", "-q", action="store_true", help="Solo tests rápidos")
    
    args = parser.parse_args()
    
    print("🧪 Suite de Tests del Sistema de Optimización - Fase 5")
    print("=" * 55)
    
    if args.quick:
        # Solo tests que no requieren dependencias pesadas
        specific_classes = [
            "TestOptimizationConfig",
            "TestImportManager",
            "TestDataManager"
        ]
        print("⚡ Ejecutando tests rápidos...")
        
        success = True
        for test_class in specific_classes:
            print(f"\n🔄 Ejecutando {test_class}...")
            success &= run_test_suite(args.verbose, test_class)
    else:
        success = run_test_suite(args.verbose, args.test)
    
    sys.exit(0 if success else 1)
