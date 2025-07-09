"""
🏃‍♂️ Test Runner Principal
=========================

Ejecutor principal que coordina todos los tests del sistema EDA.
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Configurar path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_utils import TestResult, create_test_data, format_test_summary

class TestRunner:
    """Ejecutor principal de tests"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.test_results = {}
        self.test_data = None
    
    def setup(self):
        """Configuración inicial"""
        self.start_time = time.time()
        self.test_data = create_test_data()
        print("🧪" + "="*60)
        print("🚀 SISTEMA DE TESTING PROFESIONAL - EDA CRIPTOMONEDAS")
        print("🧪" + "="*60)
        print(f"📊 Datos de prueba: {len(self.test_data)} observaciones")
    
    def test_data_analysis_module(self) -> Tuple[int, int]:
        """Tests para data_analysis"""
        print("\n🔬 === Testing data_analysis ===")
        
        passed = 0
        total = 0
        
        # Test 1: Importaciones
        total += 1
        try:
            from utils.data_analysis import (
                calculate_basic_metrics, evaluate_data_quality,
                calculate_market_dominance, generate_summary_report
            )
            print("   ✅ Test 1: Importaciones exitosas")
            passed += 1
        except ImportError as e:
            print(f"   ❌ Test 1: Error en importaciones - {e}")
        
        # Test 2: Métricas básicas
        total += 1
        try:
            from utils.data_analysis import calculate_basic_metrics
            metrics = calculate_basic_metrics(self.test_data)
            
            if isinstance(metrics, dict) and 'total_observations' in metrics:
                print(f"   ✅ Test 2: Métricas básicas - {metrics['total_observations']} obs, {metrics.get('total_tokens', 'N/A')} tokens")
                passed += 1
            else:
                print("   ❌ Test 2: Error en estructura de métricas")
        except Exception as e:
            print(f"   ❌ Test 2: Error en métricas - {e}")
        
        # Test 3: Outliers (adaptativo)
        total += 1
        try:
            # Intentar diferentes funciones de outliers
            outlier_func = None
            outlier_functions = ['detect_outliers', 'detect_outliers_iqr', 'find_outliers']
            
            for func_name in outlier_functions:
                try:
                    module = __import__('utils.data_analysis', fromlist=[func_name])
                    outlier_func = getattr(module, func_name, None)
                    if outlier_func:
                        break
                except:
                    continue
            
            if outlier_func:
                try:
                    outliers = outlier_func(self.test_data, 'close')
                except TypeError:
                    outliers = outlier_func(self.test_data)
                
                outlier_count = len(outliers) if hasattr(outliers, '__len__') else 0
                pct = (outlier_count / len(self.test_data)) * 100
                print(f"   ✅ Test 3: Outliers detectados - {outlier_count} ({pct:.1f}%)")
                passed += 1
            else:
                print("   ⚠️  Test 3: No hay funciones de outliers disponibles")
        except Exception as e:
            print(f"   ❌ Test 3: Error en outliers - {e}")
        
        # Test 4: Evaluación de calidad
        total += 1
        try:
            from utils.data_analysis import calculate_basic_metrics, evaluate_data_quality
            from utils.config import QUALITY_THRESHOLDS
            
            metrics = calculate_basic_metrics(self.test_data)
            quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
            
            if isinstance(quality, dict) and 'overall_status' in quality:
                print(f"   ✅ Test 4: Calidad evaluada - {quality['overall_status']}")
                passed += 1
            else:
                print("   ❌ Test 4: Error en evaluación de calidad")
        except Exception as e:
            print(f"   ❌ Test 4: Error en calidad - {e}")
        
        # Test 5: Dominancia de mercado
        total += 1
        try:
            from utils.data_analysis import calculate_market_dominance
            dominance = calculate_market_dominance(self.test_data)
            
            if hasattr(dominance, '__len__') and len(dominance) > 0:
                print(f"   ✅ Test 5: Dominancia calculada - {len(dominance)} grupos")
                passed += 1
            else:
                print("   ❌ Test 5: Error en dominancia")
        except Exception as e:
            print(f"   ❌ Test 5: Error en dominancia - {e}")
        
        # Test 6: Reporte resumen
        total += 1
        try:
            from utils.data_analysis import calculate_basic_metrics, evaluate_data_quality, generate_summary_report
            from utils.config import QUALITY_THRESHOLDS
            
            metrics = calculate_basic_metrics(self.test_data)
            quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
            report = generate_summary_report(metrics, quality)
            
            if isinstance(report, str) and len(report) > 50:
                print(f"   ✅ Test 6: Reporte generado ({len(report)} chars)")
                passed += 1
            else:
                print("   ❌ Test 6: Error en reporte")
        except Exception as e:
            print(f"   ❌ Test 6: Error en reporte - {e}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"   📊 data_analysis: {passed}/{total} ({success_rate:.1f}%)")
        
        return passed, total
    
    def test_feature_engineering_module(self) -> Tuple[int, int]:
        """Tests para feature_engineering"""
        print("\n🔧 === Testing feature_engineering ===")
        
        passed = 0
        total = 0
        
        # Test 1: Importaciones
        total += 1
        try:
            from utils.feature_engineering import (
                calculate_returns, calculate_moving_averages, 
                calculate_volatility, create_technical_features
            )
            print("   ✅ Test 1: Importaciones exitosas")
            passed += 1
        except ImportError as e:
            print(f"   ❌ Test 1: Error en importaciones - {e}")
        
        # Test 2: Retornos
        total += 1
        try:
            from utils.feature_engineering import calculate_returns
            df_returns = calculate_returns(self.test_data)
            return_cols = [col for col in df_returns.columns if 'ret_' in col]
            
            if len(return_cols) > 0:
                print(f"   ✅ Test 2: Retornos calculados - {return_cols}")
                passed += 1
            else:
                print("   ❌ Test 2: No se generaron retornos")
        except Exception as e:
            print(f"   ❌ Test 2: Error en retornos - {e}")
        
        # Test 3: Medias móviles
        total += 1
        try:
            from utils.feature_engineering import calculate_moving_averages
            df_ma = calculate_moving_averages(self.test_data)
            ma_cols = [col for col in df_ma.columns if 'sma_' in col or 'ma_' in col]
            
            if len(ma_cols) > 0:
                print(f"   ✅ Test 3: Medias móviles - {ma_cols}")
                passed += 1
            else:
                print("   ❌ Test 3: No se generaron medias móviles")
        except Exception as e:
            print(f"   ❌ Test 3: Error en medias móviles - {e}")
        
        # Test 4: Volatilidad
        total += 1
        try:
            from utils.feature_engineering import calculate_returns, calculate_volatility
            df_returns = calculate_returns(self.test_data)
            df_vol = calculate_volatility(df_returns)
            vol_cols = [col for col in df_vol.columns if 'vol_' in col]
            
            if len(vol_cols) > 0:
                print(f"   ✅ Test 4: Volatilidad calculada - {vol_cols}")
                passed += 1
            else:
                print("   ❌ Test 4: No se calculó volatilidad")
        except Exception as e:
            print(f"   ❌ Test 4: Error en volatilidad - {e}")
        
        # Test 5: Features técnicos
        total += 1
        try:
            from utils.feature_engineering import create_technical_features
            from utils.config import TECHNICAL_FEATURES
            
            df_tech = create_technical_features(self.test_data, TECHNICAL_FEATURES)
            new_cols = len(df_tech.columns) - len(self.test_data.columns)
            
            if new_cols > 0:
                print(f"   ✅ Test 5: Features técnicos - {new_cols} nuevas columnas")
                passed += 1
            else:
                print("   ❌ Test 5: No se agregaron features")
        except Exception as e:
            print(f"   ❌ Test 5: Error en features - {e}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"   📊 feature_engineering: {passed}/{total} ({success_rate:.1f}%)")
        
        return passed, total
    
    def test_visualizations_module(self) -> Tuple[int, int]:
        """Tests para visualizations"""
        print("\n📊 === Testing visualizations ===")
        
        passed = 0
        total = 0
        
        # Test 1: Importaciones
        total += 1
        try:
            from utils.visualizations import (
                plot_narrative_distribution, plot_market_cap_analysis
            )
            print("   ✅ Test 1: Importaciones exitosas")
            passed += 1
        except ImportError as e:
            print(f"   ❌ Test 1: Error en importaciones - {e}")
        
        # Tests de visualizaciones (con manejo de errores)
        visualization_tests = [
            ('plot_narrative_distribution', 'Distribución narrativas'),
            ('plot_market_cap_analysis', 'Análisis market cap'),
            ('plot_temporal_analysis', 'Análisis temporal'),
            ('plot_returns_analysis', 'Análisis retornos')
        ]
        
        for func_name, description in visualization_tests:
            total += 1
            try:
                import utils.visualizations as viz_module
                from utils.config import NARRATIVE_COLORS
                
                # Obtener función
                viz_func = getattr(viz_module, func_name, None)
                
                if viz_func:
                    # Intentar con diferentes parámetros
                    try:
                        fig = viz_func(self.test_data, NARRATIVE_COLORS)
                    except TypeError:
                        fig = viz_func(self.test_data)
                    
                    if fig is not None:
                        print(f"   ✅ Test {total}: {description} generado")
                        passed += 1
                    else:
                        print(f"   ❌ Test {total}: {description} retornó None")
                else:
                    print(f"   ⚠️  Test {total}: {func_name} no disponible")
            except Exception as e:
                print(f"   ❌ Test {total}: Error en {description} - {str(e)[:50]}...")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"   📊 visualizations: {passed}/{total} ({success_rate:.1f}%)")
        
        return passed, total
    
    def test_config_module(self) -> Tuple[int, int]:
        """Tests para config"""
        print("\n⚙️ === Testing config ===")
        
        passed = 0
        total = 0
        
        # Test importación
        total += 1
        try:
            import utils.config as config
            print("   ✅ Test 1: Módulo config importado")
            passed += 1
        except ImportError as e:
            print(f"   ❌ Test 1: Error importando config - {e}")
        
        # Tests de variables de configuración
        config_vars = [
            ('NARRATIVE_COLORS', 'Colores narrativas'),
            ('QUALITY_THRESHOLDS', 'Umbrales calidad'),
            ('ANALYSIS_CONFIG', 'Config análisis'),
            ('TECHNICAL_FEATURES', 'Features técnicos')
        ]
        
        for var_name, description in config_vars:
            total += 1
            try:
                import utils.config as config
                var_value = getattr(config, var_name, None)
                
                if var_value is not None:
                    if isinstance(var_value, dict) and len(var_value) > 0:
                        print(f"   ✅ Test {total}: {description} - {len(var_value)} elementos")
                        passed += 1
                    elif isinstance(var_value, (list, tuple)) and len(var_value) > 0:
                        print(f"   ✅ Test {total}: {description} - {len(var_value)} elementos")
                        passed += 1
                    else:
                        print(f"   ⚠️  Test {total}: {description} existe pero vacío")
                else:
                    print(f"   ⚠️  Test {total}: {description} no encontrado")
            except Exception as e:
                print(f"   ❌ Test {total}: Error en {description} - {e}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"   📊 config: {passed}/{total} ({success_rate:.1f}%)")
        
        return passed, total
    
    def generate_report(self, results: List[Tuple[int, int]], module_names: List[str]):
        """Generar reporte final"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        total_passed = sum(r[0] for r in results)
        total_tests = sum(r[1] for r in results)
        overall_success = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n🏁" + "="*60)
        print("📊 REPORTE FINAL")
        print("🏁" + "="*60)
        print(f"⏱️  Tiempo total: {total_time:.2f}s")
        print(f"📈 Tasa de éxito: {overall_success:.1f}%")
        print(f"✅ Tests pasados: {total_passed}")
        print(f"❌ Tests fallidos: {total_tests - total_passed}")
        print(f"🧪 Total tests: {total_tests}")
        
        print(f"\n📋 Resultados por módulo:")
        for i, (passed, total) in enumerate(results):
            success_rate = (passed / total * 100) if total > 0 else 0
            status = "✅" if success_rate == 100 else "⚠️" if success_rate >= 80 else "❌"
            print(f"   {status} {module_names[i]}: {passed}/{total} ({success_rate:.1f}%)")
        
        # Guardar reporte JSON
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': total_time,
            'overall_success_rate': overall_success,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'modules': {
                module_names[i]: {
                    'passed': results[i][0],
                    'total': results[i][1],
                    'success_rate': (results[i][0] / results[i][1] * 100) if results[i][1] > 0 else 0
                }
                for i in range(len(module_names))
            }
        }
        
        report_path = os.path.join(os.path.dirname(__file__), 'reports', 'test_report.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Reporte guardado en: {report_path}")
        
        print("\n🎯" + "="*60)
        if overall_success >= 95:
            print("🎉 ¡EXCELENTE! Sistema completamente funcional")
        elif overall_success >= 80:
            print("👍 ¡BUENO! Sistema funcional con mejoras menores")
        elif overall_success >= 60:
            print("✅ ACEPTABLE. Sistema funcional")
        else:
            print("⚠️  NECESITA MEJORAS. Sistema básico")
        print("🎯" + "="*60)
        
        return report_data
    
    def run_all_tests(self):
        """Ejecutar todos los tests"""
        self.setup()
        
        results = []
        module_names = []
        
        # Ejecutar tests por módulo
        results.append(self.test_data_analysis_module())
        module_names.append('data_analysis')
        
        results.append(self.test_feature_engineering_module())
        module_names.append('feature_engineering')
        
        results.append(self.test_visualizations_module())
        module_names.append('visualizations')
        
        results.append(self.test_config_module())
        module_names.append('config')
        
        # Generar reporte final
        report = self.generate_report(results, module_names)
        
        return report

def run_all_tests():
    """Función de conveniencia para ejecutar todos los tests"""
    runner = TestRunner()
    return runner.run_all_tests()
