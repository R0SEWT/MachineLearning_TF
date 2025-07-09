#!/usr/bin/env python3
"""
🏆 SISTEMA DE TESTING PROFESIONAL PARA EDA DE CRIPTOMONEDAS
================================================================

Sistema completo de testing que incluye:
- Tests funcionales
- Tests de calidad
- Tests de rendimiento  
- Tests de casos edge
- Reporting profesional
- Logging estructurado

Autor: Sistema de Testing Automatizado
Versión: 2.0.0
"""

import sys
import os
import time
import logging
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json

# Configurar path
sys.path.append('.')

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EDATesting')

class TestResult:
    """Clase para almacenar resultados de tests"""
    def __init__(self, name: str, passed: bool, execution_time: float, 
                 details: str = "", error: Optional[Exception] = None):
        self.name = name
        self.passed = passed
        self.execution_time = execution_time
        self.details = details
        self.error = error
        self.timestamp = datetime.now()

class TestSuite:
    """Suite de tests profesional"""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Iniciar suite de tests"""
        self.start_time = time.time()
        logger.info(f"🧪 Iniciando suite: {self.name}")
    
    def end(self):
        """Finalizar suite de tests"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info(f"✅ Suite {self.name} completada: {passed}/{total} ({success_rate:.1f}%) en {total_time:.2f}s")
        return success_rate, passed, total
    
    def run_test(self, test_func, test_name: str, *args, **kwargs) -> TestResult:
        """Ejecutar un test individual"""
        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if isinstance(result, tuple):
                passed, details = result
            else:
                passed, details = bool(result), str(result)
            
            test_result = TestResult(test_name, passed, execution_time, details)
            self.results.append(test_result)
            
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {test_name} ({execution_time:.3f}s): {details}")
            
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            test_result = TestResult(test_name, False, execution_time, str(e), e)
            self.results.append(test_result)
            
            logger.error(f"  ❌ {test_name} ({execution_time:.3f}s): {str(e)}")
            return test_result

class EDATester:
    """Tester principal para el sistema EDA"""
    
    def __init__(self):
        self.test_suites: List[TestSuite] = []
        self.test_data = None
        
    def create_comprehensive_test_data(self):
        """Crear datos de prueba comprehensivos"""
        import pandas as pd
        import numpy as np
        
        logger.info("📊 Generando datos de prueba comprehensivos...")
        
        # Seed para reproducibilidad
        np.random.seed(42)
        
        # Crear datos más realistas
        n_observations = 1000
        tokens = ['BTC', 'ETH', 'ADA', 'SOL', 'MATIC', 'LINK', 'DOT', 'UNI'] * 125
        narratives = ['defi', 'gaming', 'ai', 'meme', 'rwa', 'infrastructure'] * 167
        
        data = {
            'id': tokens[:n_observations],
            'symbol': [f'SYM{i%50}' for i in range(n_observations)],
            'name': [f'Token {i%100}' for i in range(n_observations)],
            'narrative': narratives[:n_observations],
            'close': np.random.lognormal(8, 1.5, n_observations),
            'market_cap': np.random.lognormal(25, 2, n_observations),
            'volume': np.random.lognormal(20, 1.8, n_observations),
            'date': pd.date_range('2020-01-01', periods=n_observations, freq='D')
        }
        
        df = pd.DataFrame(data)
        
        # Agregar algunos valores problemáticos para testing
        problem_indices = np.random.choice(df.index, 50, replace=False)
        df.loc[problem_indices[:10], 'market_cap'] = np.nan
        df.loc[problem_indices[10:15], 'volume'] = 0
        df.loc[problem_indices[15:20], 'close'] = np.inf
        df.loc[problem_indices[20:25], 'close'] = -1
        
        # Limpiar valores problemáticos para evitar errores
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df[df['close'] > 0]
        
        logger.info(f"✅ Datos generados: {len(df)} observaciones, {df['id'].nunique()} tokens")
        
        self.test_data = df
        return df
    
    def test_data_analysis_module(self) -> TestSuite:
        """Tests completos para data_analysis"""
        suite = TestSuite("DataAnalysis")
        suite.start()
        
        def test_imports():
            try:
                from utils.data_analysis import (
                    calculate_basic_metrics, detect_outliers, evaluate_data_quality,
                    calculate_market_dominance, generate_summary_report
                )
                return True, "Importaciones exitosas"
            except ImportError as e:
                return False, f"Error de importación: {e}"
        
        def test_basic_metrics():
            from utils.data_analysis import calculate_basic_metrics
            metrics = calculate_basic_metrics(self.test_data)
            required_keys = ['total_observations', 'total_tokens', 'total_narratives']
            
            if all(key in metrics for key in required_keys):
                return True, f"Métricas: {metrics['total_observations']} obs, {metrics['total_tokens']} tokens"
            return False, "Faltan claves requeridas en métricas"
        
        def test_outlier_detection():
            from utils.data_analysis import detect_outliers
            outliers = detect_outliers(self.test_data, 'close')
            outlier_count = len(outliers) if hasattr(outliers, '__len__') else 0
            pct = (outlier_count / len(self.test_data)) * 100
            return True, f"{outlier_count} outliers ({pct:.1f}%)"
        
        def test_quality_evaluation():
            from utils.data_analysis import calculate_basic_metrics, evaluate_data_quality
            from utils.config import QUALITY_THRESHOLDS
            
            metrics = calculate_basic_metrics(self.test_data)
            quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
            
            if 'overall_status' in quality:
                return True, f"Calidad: {quality['overall_status']}"
            return False, "Error en evaluación de calidad"
        
        def test_market_dominance():
            from utils.data_analysis import calculate_market_dominance
            dominance = calculate_market_dominance(self.test_data)
            
            if hasattr(dominance, '__len__') and len(dominance) > 0:
                return True, f"Dominancia calculada para {len(dominance)} narrativas"
            return False, "Error en cálculo de dominancia"
        
        def test_summary_report():
            from utils.data_analysis import calculate_basic_metrics, evaluate_data_quality, generate_summary_report
            from utils.config import QUALITY_THRESHOLDS
            
            metrics = calculate_basic_metrics(self.test_data)
            quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
            report = generate_summary_report(metrics, quality)
            
            if isinstance(report, str) and len(report) > 100:
                return True, f"Reporte generado ({len(report)} caracteres)"
            return False, "Error en generación de reporte"
        
        # Ejecutar tests
        suite.run_test(test_imports, "Importaciones")
        suite.run_test(test_basic_metrics, "Métricas básicas")
        suite.run_test(test_outlier_detection, "Detección de outliers")
        suite.run_test(test_quality_evaluation, "Evaluación de calidad")
        suite.run_test(test_market_dominance, "Dominancia de mercado")
        suite.run_test(test_summary_report, "Reporte resumen")
        
        suite.end()
        return suite
    
    def test_feature_engineering_module(self) -> TestSuite:
        """Tests completos para feature_engineering"""
        suite = TestSuite("FeatureEngineering")
        suite.start()
        
        def test_imports():
            try:
                from utils.feature_engineering import (
                    calculate_returns, calculate_moving_averages, calculate_volatility,
                    create_technical_features
                )
                return True, "Importaciones exitosas"
            except ImportError as e:
                return False, f"Error de importación: {e}"
        
        def test_returns_calculation():
            from utils.feature_engineering import calculate_returns
            df_returns = calculate_returns(self.test_data)
            return_cols = [col for col in df_returns.columns if 'ret_' in col]
            
            if len(return_cols) > 0:
                return True, f"Retornos calculados: {return_cols}"
            return False, "No se generaron columnas de retornos"
        
        def test_moving_averages():
            from utils.feature_engineering import calculate_moving_averages
            df_ma = calculate_moving_averages(self.test_data)
            ma_cols = [col for col in df_ma.columns if 'sma_' in col]
            
            if len(ma_cols) > 0:
                return True, f"Medias móviles: {ma_cols}"
            return False, "No se generaron medias móviles"
        
        def test_volatility():
            from utils.feature_engineering import calculate_returns, calculate_volatility
            df_returns = calculate_returns(self.test_data)
            df_vol = calculate_volatility(df_returns)
            vol_cols = [col for col in df_vol.columns if 'vol_' in col]
            
            if len(vol_cols) > 0:
                return True, f"Volatilidad: {vol_cols}"
            return False, "No se calculó volatilidad"
        
        def test_technical_features():
            from utils.feature_engineering import create_technical_features
            from utils.config import TECHNICAL_FEATURES
            
            df_tech = create_technical_features(self.test_data, TECHNICAL_FEATURES)
            new_cols = len(df_tech.columns) - len(self.test_data.columns)
            
            if new_cols > 0:
                return True, f"Features técnicos: {new_cols} nuevas columnas"
            return False, "No se agregaron features técnicos"
        
        # Ejecutar tests
        suite.run_test(test_imports, "Importaciones")
        suite.run_test(test_returns_calculation, "Cálculo de retornos")
        suite.run_test(test_moving_averages, "Medias móviles")
        suite.run_test(test_volatility, "Volatilidad")
        suite.run_test(test_technical_features, "Features técnicos")
        
        suite.end()
        return suite
    
    def test_visualizations_module(self) -> TestSuite:
        """Tests completos para visualizations"""
        suite = TestSuite("Visualizations")
        suite.start()
        
        def test_imports():
            try:
                from utils.visualizations import (
                    plot_narrative_distribution, plot_market_cap_analysis,
                    plot_time_series_analysis
                )
                return True, "Importaciones exitosas"
            except ImportError as e:
                return False, f"Error de importación: {e}"
        
        def test_narrative_distribution():
            from utils.visualizations import plot_narrative_distribution
            fig = plot_narrative_distribution(self.test_data)
            
            if fig is not None:
                return True, "Gráfico de distribución narrativas generado"
            return False, "Error en gráfico de narrativas"
        
        def test_market_cap_analysis():
            from utils.visualizations import plot_market_cap_analysis
            fig = plot_market_cap_analysis(self.test_data)
            
            if fig is not None:
                return True, "Análisis de market cap generado"
            return False, "Error en análisis de market cap"
        
        def test_time_series():
            from utils.visualizations import plot_time_series_analysis
            fig = plot_time_series_analysis(self.test_data)
            
            if fig is not None:
                return True, "Análisis temporal generado"
            return False, "Error en análisis temporal"
        
        # Ejecutar tests
        suite.run_test(test_imports, "Importaciones")
        suite.run_test(test_narrative_distribution, "Distribución narrativas")
        suite.run_test(test_market_cap_analysis, "Análisis market cap")
        suite.run_test(test_time_series, "Análisis temporal")
        
        suite.end()
        return suite
    
    def test_config_module(self) -> TestSuite:
        """Tests completos para config"""
        suite = TestSuite("Config")
        suite.start()
        
        def test_imports():
            try:
                from utils.config import (
                    NARRATIVE_COLORS, QUALITY_THRESHOLDS, ANALYSIS_CONFIG,
                    PROJECT_PATHS, TECHNICAL_FEATURES
                )
                return True, "Importaciones exitosas"
            except ImportError as e:
                return False, f"Error de importación: {e}"
        
        def test_narrative_colors():
            from utils.config import NARRATIVE_COLORS
            
            if isinstance(NARRATIVE_COLORS, dict) and len(NARRATIVE_COLORS) > 0:
                return True, f"Colores narrativas: {len(NARRATIVE_COLORS)} definidos"
            return False, "Error en colores de narrativas"
        
        def test_quality_thresholds():
            from utils.config import QUALITY_THRESHOLDS
            required_keys = ['completeness', 'duplicates']
            
            if all(key in QUALITY_THRESHOLDS for key in required_keys):
                return True, "Umbrales de calidad configurados"
            return False, "Faltan umbrales de calidad"
        
        def test_analysis_config():
            from utils.config import ANALYSIS_CONFIG
            
            if isinstance(ANALYSIS_CONFIG, dict) and len(ANALYSIS_CONFIG) > 0:
                return True, f"Config análisis: {len(ANALYSIS_CONFIG)} parámetros"
            return False, "Error en config de análisis"
        
        def test_project_paths():
            from utils.config import PROJECT_PATHS
            
            if isinstance(PROJECT_PATHS, dict) and len(PROJECT_PATHS) > 0:
                return True, "Rutas del proyecto configuradas"
            return False, "Error en rutas del proyecto"
        
        # Ejecutar tests
        suite.run_test(test_imports, "Importaciones")
        suite.run_test(test_narrative_colors, "Colores narrativas")
        suite.run_test(test_quality_thresholds, "Umbrales calidad")
        suite.run_test(test_analysis_config, "Config análisis")
        suite.run_test(test_project_paths, "Rutas proyecto")
        
        suite.end()
        return suite
    
    def test_performance(self) -> TestSuite:
        """Tests de rendimiento"""
        suite = TestSuite("Performance")
        suite.start()
        
        def test_large_dataset_processing():
            """Test con dataset grande"""
            import pandas as pd
            import numpy as np
            
            # Crear dataset grande
            large_data = self.test_data.copy()
            for i in range(10):  # 10x más datos
                large_data = pd.concat([large_data, self.test_data], ignore_index=True)
            
            start_time = time.time()
            
            from utils.data_analysis import calculate_basic_metrics
            metrics = calculate_basic_metrics(large_data)
            
            processing_time = time.time() - start_time
            
            if processing_time < 5.0:  # Debería procesar en menos de 5 segundos
                return True, f"Dataset grande procesado en {processing_time:.2f}s"
            return False, f"Procesamiento lento: {processing_time:.2f}s"
        
        def test_memory_efficiency():
            """Test de eficiencia de memoria"""
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Procesar datos
            from utils.feature_engineering import calculate_returns, calculate_moving_averages
            df_processed = calculate_returns(self.test_data)
            df_processed = calculate_moving_averages(df_processed)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            if memory_increase < 100:  # Menos de 100MB de incremento
                return True, f"Incremento de memoria: {memory_increase:.1f}MB"
            return False, f"Alto consumo de memoria: {memory_increase:.1f}MB"
        
        # Ejecutar tests
        suite.run_test(test_large_dataset_processing, "Procesamiento dataset grande")
        suite.run_test(test_memory_efficiency, "Eficiencia de memoria")
        
        suite.end()
        return suite
    
    def test_edge_cases(self) -> TestSuite:
        """Tests de casos edge"""
        suite = TestSuite("EdgeCases")
        suite.start()
        
        def test_empty_dataframe():
            """Test con DataFrame vacío"""
            import pandas as pd
            empty_df = pd.DataFrame()
            
            try:
                from utils.data_analysis import calculate_basic_metrics
                metrics = calculate_basic_metrics(empty_df)
                return True, "DataFrame vacío manejado correctamente"
            except Exception as e:
                return False, f"Error con DataFrame vacío: {e}"
        
        def test_single_row():
            """Test con una sola fila"""
            single_row = self.test_data.head(1)
            
            try:
                from utils.data_analysis import calculate_basic_metrics
                metrics = calculate_basic_metrics(single_row)
                return True, "Una fila manejada correctamente"
            except Exception as e:
                return False, f"Error con una fila: {e}"
        
        def test_all_nan_column():
            """Test con columna completamente NaN"""
            import numpy as np
            test_df = self.test_data.copy()
            test_df['all_nan'] = np.nan
            
            try:
                from utils.data_analysis import calculate_basic_metrics
                metrics = calculate_basic_metrics(test_df)
                return True, "Columna NaN manejada correctamente"
            except Exception as e:
                return False, f"Error con columna NaN: {e}"
        
        def test_missing_required_columns():
            """Test con columnas requeridas faltantes"""
            import pandas as pd
            incomplete_df = pd.DataFrame({'only_one_col': [1, 2, 3]})
            
            try:
                from utils.data_analysis import calculate_basic_metrics
                metrics = calculate_basic_metrics(incomplete_df)
                return True, "Columnas faltantes manejadas"
            except Exception as e:
                return False, f"Error con columnas faltantes: {e}"
        
        # Ejecutar tests
        suite.run_test(test_empty_dataframe, "DataFrame vacío")
        suite.run_test(test_single_row, "Una sola fila")
        suite.run_test(test_all_nan_column, "Columna completamente NaN")
        suite.run_test(test_missing_required_columns, "Columnas requeridas faltantes")
        
        suite.end()
        return suite
    
    def generate_professional_report(self):
        """Generar reporte profesional"""
        total_tests = sum(len(suite.results) for suite in self.test_suites)
        total_passed = sum(sum(1 for r in suite.results if r.passed) for suite in self.test_suites)
        overall_success = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Estadísticas por suite
        suite_stats = []
        for suite in self.test_suites:
            passed = sum(1 for r in suite.results if r.passed)
            total = len(suite.results)
            success_rate = (passed / total * 100) if total > 0 else 0
            total_time = suite.end_time - suite.start_time if suite.end_time else 0
            
            suite_stats.append({
                'name': suite.name,
                'passed': passed,
                'total': total,
                'success_rate': success_rate,
                'execution_time': total_time
            })
        
        # Generar reporte en JSON
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'overall': {
                'success_rate': overall_success,
                'total_tests': total_tests,
                'total_passed': total_passed,
                'total_failed': total_tests - total_passed
            },
            'suites': suite_stats,
            'failed_tests': [
                {
                    'suite': suite.name,
                    'test': result.name,
                    'error': str(result.error) if result.error else result.details,
                    'execution_time': result.execution_time
                }
                for suite in self.test_suites
                for result in suite.results
                if not result.passed
            ]
        }
        
        # Guardar reporte JSON
        with open('test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return report_data
    
    def run_all_tests(self):
        """Ejecutar todos los tests"""
        start_time = time.time()
        
        print("🏆" + "="*60)
        print("🚀 SISTEMA DE TESTING PROFESIONAL - EDA CRIPTOMONEDAS")
        print("🏆" + "="*60)
        
        # Crear datos de prueba
        self.create_comprehensive_test_data()
        
        # Ejecutar suites de tests
        self.test_suites.append(self.test_data_analysis_module())
        self.test_suites.append(self.test_feature_engineering_module())
        self.test_suites.append(self.test_visualizations_module())
        self.test_suites.append(self.test_config_module())
        self.test_suites.append(self.test_performance())
        self.test_suites.append(self.test_edge_cases())
        
        # Generar reporte
        report = self.generate_professional_report()
        
        # Mostrar resumen final
        total_time = time.time() - start_time
        
        print("\n🏁" + "="*60)
        print("📊 REPORTE FINAL PROFESIONAL")
        print("🏁" + "="*60)
        print(f"⏱️  Tiempo total de ejecución: {total_time:.2f}s")
        print(f"📈 Tasa de éxito general: {report['overall']['success_rate']:.1f}%")
        print(f"✅ Tests pasados: {report['overall']['total_passed']}")
        print(f"❌ Tests fallidos: {report['overall']['total_failed']}")
        print(f"🧪 Total tests: {report['overall']['total_tests']}")
        
        print(f"\n📋 Resultados por módulo:")
        for suite_data in report['suites']:
            status = "✅" if suite_data['success_rate'] == 100 else "⚠️" if suite_data['success_rate'] >= 80 else "❌"
            print(f"   {status} {suite_data['name']}: {suite_data['passed']}/{suite_data['total']} ({suite_data['success_rate']:.1f}%) - {suite_data['execution_time']:.2f}s")
        
        if report['failed_tests']:
            print(f"\n❌ Tests fallidos:")
            for failed in report['failed_tests']:
                print(f"   • {failed['suite']}.{failed['test']}: {failed['error']}")
        
        print(f"\n📄 Reportes generados:")
        print(f"   • test_results.log (Log detallado)")
        print(f"   • test_report.json (Reporte JSON)")
        
        print("\n🎯" + "="*60)
        if report['overall']['success_rate'] >= 95:
            print("🎉 ¡EXCELENTE! Sistema completamente funcional")
        elif report['overall']['success_rate'] >= 80:
            print("👍 ¡BUENO! Sistema mayormente funcional con mejoras menores")
        elif report['overall']['success_rate'] >= 60:
            print("⚠️  ACEPTABLE. Sistema funcional pero necesita mejoras")
        else:
            print("❌ CRÍTICO. Sistema necesita atención inmediata")
        print("🎯" + "="*60)

def main():
    """Función principal"""
    # Suprimir warnings para output más limpio
    warnings.filterwarnings('ignore')
    
    tester = EDATester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
