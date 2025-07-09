"""
🔬 Tests para el módulo data_analysis
=====================================

Tests completos y robustos para todas las funciones del módulo data_analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .test_utils import TestResult, create_test_data, measure_execution_time, safe_import
from typing import List, Tuple

class TestDataAnalysis:
    """Clase de tests para data_analysis"""
    
    def __init__(self):
        self.test_data = create_test_data()
        self.results: List[TestResult] = []
    
    def test_imports(self) -> TestResult:
        """Test de importaciones del módulo"""
        start_time = time.time()
        
        try:
            from utils.data_analysis import (
                calculate_basic_metrics, evaluate_data_quality,
                calculate_market_dominance, generate_summary_report
            )
            
            execution_time = time.time() - start_time
            result = TestResult(
                "test_imports", 
                True, 
                execution_time,
                "Todas las funciones principales importadas correctamente"
            )
        except ImportError as e:
            execution_time = time.time() - start_time
            result = TestResult(
                "test_imports", 
                False, 
                execution_time,
                f"Error de importación: {e}",
                e
            )
        
        self.results.append(result)
        return result
    
    def test_calculate_basic_metrics(self) -> TestResult:
        """Test para calculate_basic_metrics"""
        import time
        start_time = time.time()
        
        try:
            from utils.data_analysis import calculate_basic_metrics
            
            metrics = calculate_basic_metrics(self.test_data)
            
            # Validaciones
            required_keys = ['total_observations', 'total_tokens', 'total_narratives']
            missing_keys = [key for key in required_keys if key not in metrics]
            
            if missing_keys:
                raise ValueError(f"Faltan claves requeridas: {missing_keys}")
            
            if metrics['total_observations'] != len(self.test_data):
                raise ValueError("total_observations no coincide con el tamaño del DataFrame")
            
            execution_time = time.time() - start_time
            result = TestResult(
                "test_calculate_basic_metrics",
                True,
                execution_time,
                f"Métricas calculadas: {metrics['total_observations']} obs, {metrics['total_tokens']} tokens"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                "test_calculate_basic_metrics",
                False,
                execution_time,
                f"Error: {e}",
                e
            )
        
        self.results.append(result)
        return result
    
    def test_outlier_detection(self) -> TestResult:
        """Test para detección de outliers"""
        import time
        start_time = time.time()
        
        try:
            # Intentar diferentes funciones de outliers
            outlier_func = None
            
            # Buscar función de outliers disponible
            outlier_functions = [
                'detect_outliers',
                'detect_outliers_iqr', 
                'find_outliers',
                'get_outliers'
            ]
            
            for func_name in outlier_functions:
                func = safe_import('utils.data_analysis', func_name)
                if func:
                    outlier_func = func
                    break
            
            if not outlier_func:
                raise ValueError("No se encontró función de detección de outliers")
            
            # Ejecutar función (adaptarse a diferentes signatures)
            try:
                outliers = outlier_func(self.test_data, 'close')
            except TypeError:
                # Intentar solo con DataFrame
                outliers = outlier_func(self.test_data)
            
            outlier_count = len(outliers) if hasattr(outliers, '__len__') else 0
            pct = (outlier_count / len(self.test_data)) * 100
            
            execution_time = time.time() - start_time
            result = TestResult(
                "test_outlier_detection",
                True,
                execution_time,
                f"Outliers detectados: {outlier_count} ({pct:.1f}%)"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                "test_outlier_detection",
                False,
                execution_time,
                f"Error: {e}",
                e
            )
        
        self.results.append(result)
        return result
    
    def test_quality_evaluation(self) -> TestResult:
        """Test para evaluación de calidad"""
        import time
        start_time = time.time()
        
        try:
            from utils.data_analysis import calculate_basic_metrics, evaluate_data_quality
            from utils.config import QUALITY_THRESHOLDS
            
            metrics = calculate_basic_metrics(self.test_data)
            quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
            
            if not isinstance(quality, dict):
                raise ValueError("evaluate_data_quality debe retornar un diccionario")
            
            if 'overall_status' not in quality:
                raise ValueError("Falta 'overall_status' en evaluación de calidad")
            
            execution_time = time.time() - start_time
            result = TestResult(
                "test_quality_evaluation",
                True,
                execution_time,
                f"Calidad evaluada: {quality['overall_status']}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                "test_quality_evaluation",
                False,
                execution_time,
                f"Error: {e}",
                e
            )
        
        self.results.append(result)
        return result
    
    def test_market_dominance(self) -> TestResult:
        """Test para cálculo de dominancia de mercado"""
        import time
        start_time = time.time()
        
        try:
            from utils.data_analysis import calculate_market_dominance
            
            dominance = calculate_market_dominance(self.test_data)
            
            if not hasattr(dominance, '__len__'):
                raise ValueError("calculate_market_dominance debe retornar un objeto con longitud")
            
            if len(dominance) == 0:
                raise ValueError("No se calculó dominancia para ningún grupo")
            
            execution_time = time.time() - start_time
            result = TestResult(
                "test_market_dominance",
                True,
                execution_time,
                f"Dominancia calculada para {len(dominance)} grupos"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                "test_market_dominance",
                False,
                execution_time,
                f"Error: {e}",
                e
            )
        
        self.results.append(result)
        return result
    
    def test_summary_report(self) -> TestResult:
        """Test para generación de reporte resumen"""
        import time
        start_time = time.time()
        
        try:
            from utils.data_analysis import calculate_basic_metrics, evaluate_data_quality, generate_summary_report
            from utils.config import QUALITY_THRESHOLDS
            
            metrics = calculate_basic_metrics(self.test_data)
            quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
            report = generate_summary_report(metrics, quality)
            
            if not isinstance(report, str):
                raise ValueError("generate_summary_report debe retornar un string")
            
            if len(report) < 50:
                raise ValueError("Reporte demasiado corto")
            
            execution_time = time.time() - start_time
            result = TestResult(
                "test_summary_report",
                True,
                execution_time,
                f"Reporte generado ({len(report)} caracteres)"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = TestResult(
                "test_summary_report",
                False,
                execution_time,
                f"Error: {e}",
                e
            )
        
        self.results.append(result)
        return result
    
    def run_all_tests(self) -> Tuple[int, int]:
        """Ejecutar todos los tests del módulo"""
        print("🔬 === Testing data_analysis ===")
        
        # Ejecutar todos los tests
        test_methods = [
            self.test_imports,
            self.test_calculate_basic_metrics,
            self.test_outlier_detection,
            self.test_quality_evaluation,
            self.test_market_dominance,
            self.test_summary_report
        ]
        
        for test_method in test_methods:
            result = test_method()
            status = "✅" if result.passed else "❌"
            print(f"   {status} {result.name}: {result.details}")
        
        # Calcular estadísticas
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"   📊 data_analysis: {passed}/{total} ({success_rate:.1f}%)")
        
        return passed, total

def run_data_analysis_tests() -> Tuple[int, int]:
    """Función de conveniencia para ejecutar tests de data_analysis"""
    tester = TestDataAnalysis()
    return tester.run_all_tests()
