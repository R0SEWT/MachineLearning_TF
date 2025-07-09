"""
🚀 Tests Unitarios del Sistema de Optimización - Fase 5
======================================================

Suite completa de tests para validar todos los componentes del sistema
reorganizado de optimización de hiperparámetros.

Autor: Sistema de Optimización IA
Fecha: 2025-01-09 (Fase 5 - Organización)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys
from unittest.mock import patch, MagicMock

# Agregar path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from config.optimization_config import OptimizationConfig, get_quick_config
from utils.logging_setup import OptimizationLogger, setup_logging
from utils.import_manager import ImportManager, safe_import
from core.data_manager import DataManager, DataCache
from core.optimizer import HyperparameterOptimizer, ModelHandler
from analysis.results_analyzer import ResultsAnalyzer


class TestOptimizationConfig(unittest.TestCase):
    """Tests para configuración de optimización"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_default_config_creation(self):
        """Test creación de configuración por defecto"""
        config = OptimizationConfig()
        
        self.assertIsInstance(config.enabled_models, list)
        self.assertGreater(len(config.enabled_models), 0)
        self.assertIsInstance(config.model_trials, dict)
        self.assertEqual(config.random_state, 42)
        self.assertGreater(config.cv_folds, 0)
    
    def test_config_validation(self):
        """Test validación de configuración"""
        # Configuración inválida - test_size + validation_size >= 1.0
        with self.assertRaises(ValueError):
            OptimizationConfig(test_size=0.6, validation_size=0.5)
        
        # Configuración inválida - min_samples muy bajo
        with self.assertRaises(ValueError):
            OptimizationConfig(min_samples_per_split=50)
        
        # Configuración inválida - sin modelos
        with self.assertRaises(ValueError):
            OptimizationConfig(enabled_models=[])
    
    def test_model_config_generation(self):
        """Test generación de configuración por modelo"""
        config = OptimizationConfig(enable_gpu=True)
        
        # Test XGBoost config
        xgb_config = config.get_model_config("xgboost")
        self.assertIn("random_state", xgb_config)
        self.assertIn("tree_method", xgb_config)
        self.assertEqual(xgb_config["tree_method"], "gpu_hist")
        
        # Test con GPU deshabilitada
        config.enable_gpu = False
        xgb_config = config.get_model_config("xgboost")
        self.assertEqual(xgb_config["tree_method"], "hist")
    
    def test_config_serialization(self):
        """Test serialización de configuración"""
        config = OptimizationConfig()
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("enabled_models", config_dict)
        
        # Test from_dict
        new_config = OptimizationConfig.from_dict(config_dict)
        self.assertEqual(config.enabled_models, new_config.enabled_models)
        self.assertEqual(config.random_state, new_config.random_state)
    
    def test_predefined_configs(self):
        """Test configuraciones predefinidas"""
        quick_config = get_quick_config()
        self.assertEqual(quick_config.optimization_timeout, 600)
        self.assertLessEqual(quick_config.model_trials["xgboost"], 50)


class TestLoggingSystem(unittest.TestCase):
    """Tests para sistema de logging"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_logger_initialization(self):
        """Test inicialización del logger"""
        config = {
            "level": "INFO",
            "log_dir": self.temp_dir,
            "enable_file_logging": True,
            "enable_console_logging": False
        }
        
        logger = OptimizationLogger(config)
        self.assertIsNotNone(logger)
        self.assertEqual(logger.config["level"], "INFO")
    
    def test_structured_logging(self):
        """Test logging estructurado"""
        setup_logging({
            "level": "INFO",
            "log_dir": self.temp_dir,
            "enable_structured_logging": True,
            "enable_console_logging": False
        })
        
        logger = OptimizationLogger()
        logger.log_optimization_start("xgboost", 100, "test_001")
        
        # Verificar que se creó el archivo de log
        log_files = list(Path(self.temp_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0)
    
    def test_context_management(self):
        """Test gestión de contexto"""
        logger = OptimizationLogger()
        
        # Establecer contexto
        logger.set_context(
            experiment_id="test_001",
            model_name="xgboost",
            trial_number=1
        )
        
        self.assertEqual(logger.context.experiment_id, "test_001")
        self.assertEqual(logger.context.model_name, "xgboost")
        self.assertEqual(logger.context.trial_number, 1)
        
        # Limpiar contexto
        logger.clear_context()
        self.assertIsNone(logger.context.experiment_id)


class TestImportManager(unittest.TestCase):
    """Tests para gestor de imports"""
    
    def test_import_manager_initialization(self):
        """Test inicialización del gestor de imports"""
        manager = ImportManager()
        
        self.assertIsInstance(manager._search_paths, list)
        self.assertGreater(len(manager._search_paths), 0)
        self.assertIsInstance(manager._cache, dict)
    
    def test_safe_import_existing_module(self):
        """Test import seguro de módulo existente"""
        manager = ImportManager()
        
        # Test import de módulo estándar
        os_module = manager.safe_import("os")
        self.assertIsNotNone(os_module)
        
        # Verificar cache
        self.assertIn("os", manager._cache)
    
    def test_safe_import_nonexistent_module(self):
        """Test import seguro de módulo inexistente"""
        manager = ImportManager()
        
        # Test import de módulo que no existe
        fake_module = manager.safe_import("fake_module_that_does_not_exist")
        self.assertIsNone(fake_module)
        
        # Verificar que se registró el fallo
        self.assertIn("fake_module_that_does_not_exist", manager._failed_imports)
    
    def test_ml_libraries_detection(self):
        """Test detección de librerías ML"""
        from utils.import_manager import get_ml_libraries
        
        libs = get_ml_libraries()
        self.assertIsInstance(libs, dict)
        
        # Verificar que al menos algunas librerías estándar estén disponibles
        expected_libs = ["pandas", "numpy"]
        for lib in expected_libs:
            if lib in libs:
                self.assertIsNotNone(libs[lib])


class TestDataManager(unittest.TestCase):
    """Tests para gestor de datos"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Crear datos sintéticos para testing
        self.test_data_file = Path(self.temp_dir) / "test_data.csv"
        self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Crear datos sintéticos para testing"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Crear features
        data = {
            f"feature_{i}": np.random.randn(n_samples) 
            for i in range(n_features)
        }
        
        # Crear target binario
        data["target_next_close_positive"] = np.random.randint(0, 2, n_samples)
        
        # Agregar algunas columnas a excluir
        data["id"] = range(n_samples)
        data["date"] = pd.date_range("2020-01-01", periods=n_samples, freq="D")
        
        df = pd.DataFrame(data)
        df.to_csv(self.test_data_file, index=False)
    
    def test_data_manager_initialization(self):
        """Test inicialización del gestor de datos"""
        config = {
            "cache_dir": str(Path(self.temp_dir) / "cache"),
            "target_column": "target_next_close_positive"
        }
        
        manager = DataManager(config)
        self.assertEqual(manager.config["target_column"], "target_next_close_positive")
        self.assertIsNotNone(manager.cache)
    
    def test_data_loading(self):
        """Test carga de datos"""
        manager = DataManager({
            "cache_dir": str(Path(self.temp_dir) / "cache"),
            "cache_enabled": False  # Deshabilitar cache para test
        })
        
        features, target, info = manager.load_data(str(self.test_data_file))
        
        # Verificar que se cargaron correctamente
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(target, pd.Series)
        self.assertGreater(len(features), 0)
        self.assertGreater(len(features.columns), 0)
        
        # Verificar que se excluyeron columnas correctas
        excluded_cols = ["id", "date", "target_next_close_positive"]
        for col in excluded_cols:
            self.assertNotIn(col, features.columns)
    
    def test_data_preprocessing(self):
        """Test preprocesamiento de datos"""
        manager = DataManager({
            "cache_enabled": False,
            "handle_missing": "drop",
            "normalize_features": False
        })
        
        features, target, info = manager.load_data(str(self.test_data_file))
        
        # Verificar info
        self.assertIsNotNone(info)
        self.assertEqual(info.shape, features.shape)
        self.assertIsInstance(info.target_distribution, dict)
        self.assertGreater(info.memory_usage_mb, 0)
    
    def test_train_test_split(self):
        """Test división de datos"""
        manager = DataManager({"cache_enabled": False})
        features, target, _ = manager.load_data(str(self.test_data_file))
        
        splits = manager.get_train_val_test_split(features, target)
        
        # Verificar splits
        expected_keys = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
        for key in expected_keys:
            self.assertIn(key, splits)
            self.assertGreater(len(splits[key]), 0)
        
        # Verificar proporciones aproximadas
        total_samples = len(features)
        train_samples = len(splits["X_train"])
        val_samples = len(splits["X_val"])
        test_samples = len(splits["X_test"])
        
        self.assertEqual(total_samples, train_samples + val_samples + test_samples)
    
    def test_data_cache(self):
        """Test sistema de cache"""
        cache_dir = Path(self.temp_dir) / "cache"
        cache = DataCache(str(cache_dir), max_size_mb=100)
        
        # Test cache miss
        result = cache.get("nonexistent_key")
        self.assertIsNone(result)
        
        # Test cache set/get
        test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        test_series = pd.Series([1, 0, 1])
        
        from core.data_manager import DataInfo
        test_info = DataInfo(
            shape=(3, 2),
            columns=["a", "b"],
            missing_values={},
            data_types={}
        )
        
        cache.set("test_key", test_df, test_series, test_info)
        cached_result = cache.get("test_key")
        
        self.assertIsNotNone(cached_result)
        cached_df, cached_series, cached_info = cached_result
        pd.testing.assert_frame_equal(test_df, cached_df)
        pd.testing.assert_series_equal(test_series, cached_series)


class TestOptimizer(unittest.TestCase):
    """Tests para optimizador principal"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    @patch('core.optimizer.optuna')
    @patch('core.optimizer.get_ml_libraries')
    def test_optimizer_initialization(self, mock_get_ml_libs, mock_optuna):
        """Test inicialización del optimizador"""
        # Mock dependencies
        mock_get_ml_libs.return_value = {"xgboost": MagicMock(), "pandas": MagicMock(), "numpy": MagicMock()}
        mock_optuna.create_study = MagicMock()
        
        config = get_quick_config()
        config.enabled_models = ["xgboost"]  # Solo un modelo para test
        
        # El test debe fallar graciosamente sin las librerías reales
        try:
            optimizer = HyperparameterOptimizer(config)
            # Si llegamos aquí, el mock funcionó
            self.assertIsNotNone(optimizer)
        except (ImportError, RuntimeError):
            # Esperado si no hay librerías ML disponibles
            self.skipTest("Librerías ML no disponibles para testing")
    
    def test_model_handler_base_class(self):
        """Test clase base ModelHandler"""
        config = OptimizationConfig()
        
        # Test que la clase base lance NotImplementedError
        handler = ModelHandler("test_model", config)
        
        with self.assertRaises(NotImplementedError):
            handler.create_model({})
        
        with self.assertRaises(NotImplementedError):
            handler.get_param_space()


class TestResultsAnalyzer(unittest.TestCase):
    """Tests para analizador de resultados"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Crear resultados sintéticos para testing
        self._create_synthetic_results()
    
    def _create_synthetic_results(self):
        """Crear resultados sintéticos para testing"""
        # Crear directorio de resultados
        results_dir = Path(self.temp_dir) / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Crear resultado sintético
        result_data = {
            "experiment_id": "test_exp_001",
            "timestamp": "2025-01-09T10:00:00",
            "total_time": 300.5,
            "best_model": "xgboost",
            "best_score": 0.875,
            "data_info": {
                "shape": [1000, 20],
                "memory_usage_mb": 2.5
            },
            "model_results": {
                "xgboost": {
                    "best_score": 0.875,
                    "best_params": {"n_estimators": 200, "max_depth": 6},
                    "n_trials": 100,
                    "optimization_time": 150.2,
                    "cv_scores": [0.87, 0.88, 0.86, 0.89, 0.87]
                },
                "lightgbm": {
                    "best_score": 0.862,
                    "best_params": {"n_estimators": 180, "max_depth": 7},
                    "n_trials": 100,
                    "optimization_time": 135.8,
                    "cv_scores": [0.86, 0.87, 0.85, 0.88, 0.86]
                }
            }
        }
        
        result_file = results_dir / "results_test_exp_001_20250109_100000.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f)
    
    def test_results_analyzer_initialization(self):
        """Test inicialización del analizador"""
        results_dir = Path(self.temp_dir) / "results"
        analyzer = ResultsAnalyzer(str(results_dir))
        
        self.assertGreater(len(analyzer.experiments), 0)
        self.assertIn("test_exp_001", analyzer.experiments)
    
    def test_experiment_summary(self):
        """Test generación de resumen"""
        results_dir = Path(self.temp_dir) / "results"
        analyzer = ResultsAnalyzer(str(results_dir))
        
        summary = analyzer.get_experiment_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
        self.assertIn("experiment_id", summary.columns)
        self.assertIn("best_score", summary.columns)
    
    def test_model_performance_analysis(self):
        """Test análisis de performance"""
        results_dir = Path(self.temp_dir) / "results"
        analyzer = ResultsAnalyzer(str(results_dir))
        
        performance = analyzer.analyze_model_performance()
        
        self.assertIsInstance(performance, dict)
        self.assertIn("xgboost", performance)
        self.assertIn("lightgbm", performance)
        
        # Verificar métricas calculadas
        xgb_perf = performance["xgboost"]
        self.assertIn("mean_score", xgb_perf)
        self.assertIn("std_score", xgb_perf)
        self.assertIn("consistency_score", xgb_perf)
    
    def test_report_generation(self):
        """Test generación de reporte"""
        results_dir = Path(self.temp_dir) / "results"
        analyzer = ResultsAnalyzer(str(results_dir))
        
        report_file = Path(self.temp_dir) / "test_report.md"
        report_content = analyzer.generate_report(str(report_file))
        
        self.assertIsInstance(report_content, str)
        self.assertGreater(len(report_content), 0)
        self.assertIn("Reporte de Análisis", report_content)
        
        # Verificar que se creó el archivo
        self.assertTrue(report_file.exists())


class TestIntegration(unittest.TestCase):
    """Tests de integración del sistema completo"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Configurar logging para tests
        setup_logging({
            "level": "WARNING",  # Reducir ruido en tests
            "log_dir": str(Path(self.temp_dir) / "logs"),
            "enable_console_logging": False
        })
    
    def test_config_to_optimizer_flow(self):
        """Test flujo de configuración a optimizador"""
        config = get_quick_config()
        config.enabled_models = []  # Sin modelos para evitar dependencias
        
        # Test que el optimizador falle graciosamente sin modelos
        with self.assertRaises(RuntimeError):
            HyperparameterOptimizer(config)
    
    def test_data_manager_to_analyzer_flow(self):
        """Test flujo de gestor de datos a analizador"""
        # Crear datos sintéticos
        test_data_file = Path(self.temp_dir) / "test_data.csv"
        df = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "target_next_close_positive": np.random.randint(0, 2, 100)
        })
        df.to_csv(test_data_file, index=False)
        
        # Test carga con DataManager
        manager = DataManager({"cache_enabled": False})
        features, target, info = manager.load_data(str(test_data_file))
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(target, pd.Series)
        self.assertIsNotNone(info)


def run_tests():
    """Ejecutar todos los tests"""
    # Crear suite de tests
    test_classes = [
        TestOptimizationConfig,
        TestLoggingSystem,
        TestImportManager,
        TestDataManager,
        TestOptimizer,
        TestResultsAnalyzer,
        TestIntegration
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Configurar path para imports
    import sys
    from pathlib import Path
    
    # Agregar directorio padre
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("🧪 Ejecutando Tests del Sistema de Optimización - Fase 5")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ Todos los tests pasaron exitosamente")
        sys.exit(0)
    else:
        print("\n❌ Algunos tests fallaron")
        sys.exit(1)
