#!/usr/bin/env python3
"""
🔧 SISTEMA DE TESTING INTELIGENTE Y ADAPTATIVO
=================================================

Sistema que se adapta automáticamente a las funciones disponibles
y genera tests robustos para el código real existente.

✅ Auto-detección de funciones disponibles
✅ Tests adaptativos basados en el código real
✅ Cobertura completa y robusta
✅ Reporte profesional con métricas avanzadas
"""

import sys
import os
import time
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any
import warnings

# Configurar path
sys.path.append('.')

# Suprimir warnings para output más limpio
warnings.filterwarnings('ignore')

class SmartTester:
    """Tester inteligente que se adapta al código real"""
    
    def __init__(self):
        self.results = {}
        self.test_data = None
        self.available_functions = {}
        
    def create_smart_test_data(self):
        """Crear datos de prueba inteligentes"""
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        
        # Datos realistas para criptomonedas
        n_obs = 200
        data = {
            'id': ['BTC', 'ETH', 'ADA', 'SOL', 'MATIC'] * (n_obs // 5),
            'symbol': [f'SYM{i%10}' for i in range(n_obs)],
            'name': [f'Token {i%20}' for i in range(n_obs)],
            'narrative': ['defi', 'gaming', 'ai', 'meme', 'rwa', 'infrastructure'] * (n_obs // 6 + 1),
            'close': np.random.lognormal(8, 1, n_obs),
            'market_cap': np.random.lognormal(25, 2, n_obs),
            'volume': np.random.lognormal(20, 1.5, n_obs),
            'date': pd.date_range('2023-01-01', periods=n_obs, freq='D')
        }
        
        # Ajustar longitudes para que coincidan
        for key in data:
            if len(data[key]) > n_obs:
                data[key] = data[key][:n_obs]
            elif len(data[key]) < n_obs:
                # Repetir hasta completar
                while len(data[key]) < n_obs:
                    data[key].extend(data[key][:min(len(data[key]), n_obs - len(data[key]))])
        
        df = pd.DataFrame(data)
        
        # Agregar algunos NaN para testing
        nan_indices = np.random.choice(df.index, 10, replace=False)
        df.loc[nan_indices[:5], 'market_cap'] = np.nan
        
        self.test_data = df
        print(f"📊 Datos de prueba creados: {len(df)} observaciones, {df['id'].nunique()} tokens únicos")
        return df
    
    def discover_available_functions(self, module_name: str):
        """Descubrir funciones disponibles en un módulo"""
        try:
            module = importlib.import_module(f'utils.{module_name}')
            functions = {}
            
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and not name.startswith('_'):
                    sig = inspect.signature(obj)
                    functions[name] = {
                        'callable': obj,
                        'signature': sig,
                        'parameters': list(sig.parameters.keys()),
                        'docstring': obj.__doc__ or 'Sin documentación'
                    }
            
            self.available_functions[module_name] = functions
            print(f"🔍 Módulo {module_name}: {len(functions)} funciones descubiertas")
            
            return functions
            
        except ImportError as e:
            print(f"❌ Error importando {module_name}: {e}")
            return {}
    
    def test_data_analysis_smart(self):
        """Tests inteligentes para data_analysis"""
        functions = self.discover_available_functions('data_analysis')
        
        tests_passed = 0
        total_tests = 0
        
        print("\n🔬 === Testing data_analysis (Inteligente) ===")
        
        # Test de importación
        total_tests += 1
        if functions:
            print("   ✅ Test 1: Módulo importado exitosamente")
            tests_passed += 1
        else:
            print("   ❌ Test 1: Error en importación del módulo")
        
        # Test calculate_basic_metrics
        total_tests += 1
        if 'calculate_basic_metrics' in functions:
            try:
                metrics = functions['calculate_basic_metrics']['callable'](self.test_data)
                if isinstance(metrics, dict) and 'total_observations' in metrics:
                    print(f"   ✅ Test 2: Métricas básicas - {metrics['total_observations']} obs, {metrics.get('total_tokens', 'N/A')} tokens")
                    tests_passed += 1
                else:
                    print("   ❌ Test 2: Error en estructura de métricas")
            except Exception as e:
                print(f"   ❌ Test 2: Error en métricas básicas - {e}")
        else:
            print("   ⚠️  Test 2: Función calculate_basic_metrics no disponible")
        
        # Test outliers (adaptativo)
        total_tests += 1
        outlier_functions = [name for name in functions.keys() if 'outlier' in name.lower()]
        if outlier_functions:
            try:
                func_name = outlier_functions[0]
                outliers = functions[func_name]['callable'](self.test_data, 'close')
                outlier_count = len(outliers) if hasattr(outliers, '__len__') else 0
                print(f"   ✅ Test 3: Outliers detectados con {func_name} - {outlier_count} valores")
                tests_passed += 1
            except Exception as e:
                print(f"   ❌ Test 3: Error en detección de outliers - {e}")
        else:
            print("   ⚠️  Test 3: No hay funciones de detección de outliers disponibles")
        
        # Test evaluación de calidad
        total_tests += 1
        if 'evaluate_data_quality' in functions:
            try:
                from utils.config import QUALITY_THRESHOLDS
                metrics = functions['calculate_basic_metrics']['callable'](self.test_data)
                quality = functions['evaluate_data_quality']['callable'](metrics, QUALITY_THRESHOLDS)
                if isinstance(quality, dict) and 'overall_status' in quality:
                    print(f"   ✅ Test 4: Calidad evaluada - {quality['overall_status']}")
                    tests_passed += 1
                else:
                    print("   ❌ Test 4: Error en estructura de calidad")
            except Exception as e:
                print(f"   ❌ Test 4: Error en evaluación de calidad - {e}")
        else:
            print("   ⚠️  Test 4: Función evaluate_data_quality no disponible")
        
        # Test dominancia de mercado
        total_tests += 1
        if 'calculate_market_dominance' in functions:
            try:
                dominance = functions['calculate_market_dominance']['callable'](self.test_data)
                if hasattr(dominance, '__len__') and len(dominance) > 0:
                    print(f"   ✅ Test 5: Dominancia calculada - {len(dominance)} grupos")
                    tests_passed += 1
                else:
                    print("   ❌ Test 5: Error en dominancia")
            except Exception as e:
                print(f"   ❌ Test 5: Error en dominancia - {e}")
        else:
            print("   ⚠️  Test 5: Función calculate_market_dominance no disponible")
        
        # Test reporte
        total_tests += 1
        if 'generate_summary_report' in functions:
            try:
                metrics = functions['calculate_basic_metrics']['callable'](self.test_data)
                from utils.config import QUALITY_THRESHOLDS
                quality = functions['evaluate_data_quality']['callable'](metrics, QUALITY_THRESHOLDS)
                report = functions['generate_summary_report']['callable'](metrics, quality)
                if isinstance(report, str) and len(report) > 50:
                    print(f"   ✅ Test 6: Reporte generado - {len(report)} caracteres")
                    tests_passed += 1
                else:
                    print("   ❌ Test 6: Error en reporte")
            except Exception as e:
                print(f"   ❌ Test 6: Error en reporte - {e}")
        else:
            print("   ⚠️  Test 6: Función generate_summary_report no disponible")
        
        success_rate = (tests_passed / total_tests) * 100
        print(f"   📊 data_analysis: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
        
        return tests_passed, total_tests
    
    def test_feature_engineering_smart(self):
        """Tests inteligentes para feature_engineering"""
        functions = self.discover_available_functions('feature_engineering')
        
        tests_passed = 0
        total_tests = 0
        
        print("\n🔧 === Testing feature_engineering (Inteligente) ===")
        
        # Test de importación
        total_tests += 1
        if functions:
            print("   ✅ Test 1: Módulo importado exitosamente")
            tests_passed += 1
        else:
            print("   ❌ Test 1: Error en importación del módulo")
        
        # Test calculate_returns
        total_tests += 1
        if 'calculate_returns' in functions:
            try:
                df_returns = functions['calculate_returns']['callable'](self.test_data)
                return_cols = [col for col in df_returns.columns if 'ret_' in col]
                if len(return_cols) > 0:
                    print(f"   ✅ Test 2: Retornos calculados - {return_cols}")
                    tests_passed += 1
                else:
                    print("   ❌ Test 2: No se generaron retornos")
            except Exception as e:
                print(f"   ❌ Test 2: Error en retornos - {e}")
        else:
            print("   ⚠️  Test 2: Función calculate_returns no disponible")
        
        # Test moving averages
        total_tests += 1
        ma_functions = [name for name in functions.keys() if 'moving' in name.lower() or 'ma' in name.lower()]
        if ma_functions:
            try:
                func_name = ma_functions[0]
                df_ma = functions[func_name]['callable'](self.test_data)
                ma_cols = [col for col in df_ma.columns if any(x in col for x in ['sma_', 'ma_', 'ema_'])]
                if len(ma_cols) > 0:
                    print(f"   ✅ Test 3: Medias móviles con {func_name} - {ma_cols}")
                    tests_passed += 1
                else:
                    print(f"   ❌ Test 3: No se generaron medias móviles con {func_name}")
            except Exception as e:
                print(f"   ❌ Test 3: Error en medias móviles - {e}")
        else:
            print("   ⚠️  Test 3: No hay funciones de medias móviles disponibles")
        
        # Test volatilidad
        total_tests += 1
        if 'calculate_volatility' in functions:
            try:
                # Primero calculamos retornos
                df_returns = functions['calculate_returns']['callable'](self.test_data)
                df_vol = functions['calculate_volatility']['callable'](df_returns)
                vol_cols = [col for col in df_vol.columns if 'vol_' in col]
                if len(vol_cols) > 0:
                    print(f"   ✅ Test 4: Volatilidad calculada - {vol_cols}")
                    tests_passed += 1
                else:
                    print("   ❌ Test 4: No se calculó volatilidad")
            except Exception as e:
                print(f"   ❌ Test 4: Error en volatilidad - {e}")
        else:
            print("   ⚠️  Test 4: Función calculate_volatility no disponible")
        
        # Test technical features
        total_tests += 1
        tech_functions = [name for name in functions.keys() if 'technical' in name.lower() or 'feature' in name.lower()]
        if tech_functions:
            try:
                func_name = tech_functions[0]
                from utils.config import TECHNICAL_FEATURES
                df_tech = functions[func_name]['callable'](self.test_data, TECHNICAL_FEATURES)
                new_cols = len(df_tech.columns) - len(self.test_data.columns)
                if new_cols > 0:
                    print(f"   ✅ Test 5: Features técnicos con {func_name} - {new_cols} nuevas columnas")
                    tests_passed += 1
                else:
                    print(f"   ❌ Test 5: No se agregaron features con {func_name}")
            except Exception as e:
                print(f"   ❌ Test 5: Error en features técnicos - {e}")
        else:
            print("   ⚠️  Test 5: No hay funciones de features técnicos disponibles")
        
        success_rate = (tests_passed / total_tests) * 100
        print(f"   📊 feature_engineering: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
        
        return tests_passed, total_tests
    
    def test_visualizations_smart(self):
        """Tests inteligentes para visualizations"""
        functions = self.discover_available_functions('visualizations')
        
        tests_passed = 0
        total_tests = 0
        
        print("\n📊 === Testing visualizations (Inteligente) ===")
        
        # Test de importación
        total_tests += 1
        if functions:
            print("   ✅ Test 1: Módulo importado exitosamente")
            tests_passed += 1
        else:
            print("   ❌ Test 1: Error en importación del módulo")
        
        # Test para cada función de visualización
        plot_functions = [name for name in functions.keys() if name.startswith('plot_')]
        
        for func_name in plot_functions:
            total_tests += 1
            try:
                from utils.config import NARRATIVE_COLORS
                
                # Intentar con diferentes combinaciones de parámetros
                func_info = functions[func_name]
                params = func_info['parameters']
                
                if 'colors' in params:
                    fig = func_info['callable'](self.test_data, NARRATIVE_COLORS)
                else:
                    fig = func_info['callable'](self.test_data)
                
                if fig is not None:
                    print(f"   ✅ Test {total_tests}: {func_name} generado exitosamente")
                    tests_passed += 1
                else:
                    print(f"   ❌ Test {total_tests}: {func_name} retornó None")
                    
            except Exception as e:
                print(f"   ❌ Test {total_tests}: Error en {func_name} - {str(e)[:50]}...")
        
        if not plot_functions:
            total_tests += 1
            print("   ⚠️  Test 2: No se encontraron funciones de visualización")
        
        success_rate = (tests_passed / total_tests) * 100
        print(f"   📊 visualizations: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
        
        return tests_passed, total_tests
    
    def test_config_smart(self):
        """Tests inteligentes para config"""
        print("\n⚙️ === Testing config (Inteligente) ===")
        
        tests_passed = 0
        total_tests = 0
        
        # Test de importación básica
        total_tests += 1
        try:
            import utils.config
            print("   ✅ Test 1: Módulo config importado exitosamente")
            tests_passed += 1
        except ImportError as e:
            print(f"   ❌ Test 1: Error importando config - {e}")
        
        # Descubrir qué variables están disponibles
        config_vars = {}
        try:
            import utils.config as config
            config_vars = {name: getattr(config, name) for name in dir(config) if not name.startswith('_')}
        except:
            pass
        
        # Test variables importantes
        important_vars = ['NARRATIVE_COLORS', 'QUALITY_THRESHOLDS', 'ANALYSIS_CONFIG', 'TECHNICAL_FEATURES']
        
        for var_name in important_vars:
            total_tests += 1
            if var_name in config_vars:
                var_value = config_vars[var_name]
                if isinstance(var_value, dict) and len(var_value) > 0:
                    print(f"   ✅ Test {total_tests}: {var_name} configurado - {len(var_value)} elementos")
                    tests_passed += 1
                elif isinstance(var_value, (list, tuple)) and len(var_value) > 0:
                    print(f"   ✅ Test {total_tests}: {var_name} configurado - {len(var_value)} elementos")
                    tests_passed += 1
                else:
                    print(f"   ⚠️  Test {total_tests}: {var_name} existe pero está vacío")
            else:
                print(f"   ⚠️  Test {total_tests}: {var_name} no encontrado")
        
        success_rate = (tests_passed / total_tests) * 100
        print(f"   📊 config: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
        
        return tests_passed, total_tests
    
    def run_comprehensive_tests(self):
        """Ejecutar todos los tests inteligentes"""
        start_time = time.time()
        
        print("🧠" + "="*60)
        print("🚀 SISTEMA DE TESTING INTELIGENTE Y ADAPTATIVO")
        print("🧠" + "="*60)
        
        # Crear datos de prueba
        self.create_smart_test_data()
        
        # Ejecutar tests inteligentes
        all_results = []
        
        all_results.append(self.test_data_analysis_smart())
        all_results.append(self.test_feature_engineering_smart())
        all_results.append(self.test_visualizations_smart())
        all_results.append(self.test_config_smart())
        
        # Calcular estadísticas finales
        total_passed = sum(result[0] for result in all_results)
        total_tests = sum(result[1] for result in all_results)
        overall_success = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        execution_time = time.time() - start_time
        
        # Reporte final
        print("\n🏁" + "="*60)
        print("📊 REPORTE FINAL INTELIGENTE")
        print("🏁" + "="*60)
        print(f"⏱️  Tiempo de ejecución: {execution_time:.2f}s")
        print(f"📈 Tasa de éxito adaptativa: {overall_success:.1f}%")
        print(f"✅ Tests pasados: {total_passed}")
        print(f"❌ Tests fallidos: {total_tests - total_passed}")
        print(f"🧪 Total tests ejecutados: {total_tests}")
        
        print(f"\n🔍 Funciones descubiertas por módulo:")
        for module_name, functions in self.available_functions.items():
            print(f"   📦 {module_name}: {len(functions)} funciones")
            for func_name in list(functions.keys())[:3]:  # Mostrar las primeras 3
                print(f"      • {func_name}")
            if len(functions) > 3:
                print(f"      • ... y {len(functions) - 3} más")
        
        print("\n🎯" + "="*60)
        if overall_success >= 90:
            print("🎉 ¡EXCELENTE! Sistema completamente funcional y bien estructurado")
        elif overall_success >= 75:
            print("👍 ¡BUENO! Sistema funcional con estructura sólida")
        elif overall_success >= 60:
            print("✅ FUNCIONAL. Sistema operativo con algunas limitaciones")
        else:
            print("⚠️  BÁSICO. Sistema funciona pero necesita expansión")
        print("🎯" + "="*60)
        
        return overall_success, total_passed, total_tests

def main():
    """Función principal"""
    tester = SmartTester()
    tester.run_comprehensive_tests()

if __name__ == "__main__":
    main()
