#!/usr/bin/env python3
"""
🧪 Sistema de Testing Funcional - Compatible con el código real
"""

import sys
import os
import traceback
from pathlib import Path

# Configurar path
sys.path.append('.')

def create_test_data():
    """Crear datos de prueba"""
    import pandas as pd
    import numpy as np
    
    # Datos de prueba realistas
    np.random.seed(42)
    
    data = {
        'id': ['BTC', 'ETH', 'ADA', 'SOL', 'MATIC'] * 20,
        'narrative': ['defi', 'gaming', 'ai', 'meme', 'rwa'] * 20,
        'close': np.random.lognormal(8, 1, 100),
        'market_cap': np.random.lognormal(25, 2, 100),
        'volume': np.random.lognormal(20, 1.5, 100),
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'symbol': [f'SYM{i%10}' for i in range(100)],
        'name': [f'Token {i%20}' for i in range(100)]
    }
    
    df = pd.DataFrame(data)
    
    # Agregar algunos NaN para testing
    df.loc[np.random.choice(df.index, 5), 'market_cap'] = np.nan
    
    return df

def test_data_analysis():
    """Test del módulo data_analysis"""
    print("🔬 === Testing data_analysis ===")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        # Test 1: Importaciones
        total_tests += 1
        from utils.data_analysis import (
            calculate_basic_metrics,
            get_basic_metrics,
            evaluate_data_quality,
            detect_outliers_iqr,
            calculate_market_dominance,
            generate_summary_report
        )
        print("   ✅ Test 1: Importaciones exitosas")
        tests_passed += 1
        
        # Crear datos
        df = create_test_data()
        
        # Test 2: Métricas básicas
        total_tests += 1
        metrics = calculate_basic_metrics(df)
        required_keys = ['total_observations', 'total_tokens', 'total_narratives', 'completeness']
        if all(key in metrics for key in required_keys):
            print(f"   ✅ Test 2: Métricas básicas - {metrics['total_observations']} obs, {metrics['total_tokens']} tokens")
            tests_passed += 1
        else:
            print("   ❌ Test 2: Faltan claves en métricas")
        
        # Test 3: Detección de outliers
        total_tests += 1
        outliers_count, outliers_pct = detect_outliers_iqr(df['close'])
        if isinstance(outliers_count, int) and isinstance(outliers_pct, float):
            print(f"   ✅ Test 3: Outliers detectados - {outliers_count} ({outliers_pct:.1f}%)")
            tests_passed += 1
        else:
            print("   ❌ Test 3: Error en detección de outliers")
        
        # Test 4: Evaluación de calidad
        total_tests += 1
        from utils.config import QUALITY_THRESHOLDS
        quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
        if 'overall_status' in quality:
            print(f"   ✅ Test 4: Calidad evaluada - {quality['overall_status']}")
            tests_passed += 1
        else:
            print("   ❌ Test 4: Error en evaluación de calidad")
        
        # Test 5: Dominancia de mercado
        total_tests += 1
        try:
            dominance = calculate_market_dominance(df)
            if hasattr(dominance, '__len__') and len(dominance) > 0:
                print(f"   ✅ Test 5: Dominancia calculada - {len(dominance)} narrativas")
                tests_passed += 1
            else:
                print("   ❌ Test 5: Error en dominancia")
        except Exception as e:
            print(f"   ❌ Test 5: Error en dominancia - {e}")
        
        # Test 6: Reporte resumen
        total_tests += 1
        try:
            quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
            report = generate_summary_report(metrics, quality)
            if isinstance(report, str) and len(report) > 0:
                print(f"   ✅ Test 6: Reporte generado")
                tests_passed += 1
            else:
                print("   ❌ Test 6: Error en reporte")
        except Exception as e:
            print(f"   ❌ Test 6: Reporte falló - {e}")
        
        success_rate = (tests_passed / total_tests) * 100
        print(f"   📊 data_analysis: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
        return tests_passed, total_tests
        
    except Exception as e:
        print(f"   ❌ Error general en data_analysis: {e}")
        return 0, total_tests

def test_feature_engineering():
    """Test del módulo feature_engineering"""
    print("🔧 === Testing feature_engineering ===")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        # Test 1: Importaciones
        total_tests += 1
        from utils.feature_engineering import (
            calculate_returns,
            calculate_moving_averages,
            calculate_volatility,
            create_technical_features
        )
        print("   ✅ Test 1: Importaciones exitosas")
        tests_passed += 1
        
        # Crear datos
        df = create_test_data()
        df = df.sort_values(['id', 'date']).reset_index(drop=True)
        
        # Test 2: Retornos
        total_tests += 1
        try:
            df_returns = calculate_returns(df)
            return_cols = [col for col in df_returns.columns if 'ret_' in col]
            if len(return_cols) > 0:
                print(f"   ✅ Test 2: Retornos calculados - {len(return_cols)} columnas")
                tests_passed += 1
            else:
                print("   ❌ Test 2: No se generaron columnas de retornos")
        except Exception as e:
            print(f"   ⚠️  Test 2: Error en retornos - {e}")
        
        # Test 3: Medias móviles
        total_tests += 1
        try:
            df_ma = calculate_moving_averages(df)
            ma_cols = [col for col in df_ma.columns if 'ma_' in col]
            if len(ma_cols) > 0:
                print(f"   ✅ Test 3: Medias móviles - {len(ma_cols)} columnas")
                tests_passed += 1
            else:
                print("   ❌ Test 3: No se generaron medias móviles")
        except Exception as e:
            print(f"   ⚠️  Test 3: Error en medias móviles - {e}")
        
        # Test 4: Volatilidad
        total_tests += 1
        try:
            # Primero calculamos retornos para tener la columna ret_1d
            df_with_returns = calculate_returns(df)
            df_vol = calculate_volatility(df_with_returns)
            vol_cols = [col for col in df_vol.columns if 'vol_' in col]
            if len(vol_cols) > 0:
                print(f"   ✅ Test 4: Volatilidad calculada - {len(vol_cols)} columnas")
                tests_passed += 1
            else:
                print("   ❌ Test 4: No se calculó volatilidad")
        except Exception as e:
            print(f"   ❌ Test 4: Error en volatilidad - {e}")
        
        # Test 5: Features técnicos
        total_tests += 1
        try:
            from utils.config import TECHNICAL_FEATURES
            df_tech = create_technical_features(df, TECHNICAL_FEATURES)
            if len(df_tech.columns) > len(df.columns):
                print(f"   ✅ Test 5: Features técnicos - {len(df_tech.columns) - len(df.columns)} nuevas columnas")
                tests_passed += 1
            else:
                print("   ❌ Test 5: No se agregaron features técnicos")
        except Exception as e:
            print(f"   ⚠️  Test 5: Error en features técnicos - {e}")
        
        success_rate = (tests_passed / total_tests) * 100
        print(f"   📊 feature_engineering: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
        return tests_passed, total_tests
        
    except Exception as e:
        print(f"   ❌ Error general en feature_engineering: {e}")
        return 0, total_tests

def test_visualizations():
    """Test del módulo visualizations"""
    print("📊 === Testing visualizations ===")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        # Test 1: Importaciones
        total_tests += 1
        from utils.visualizations import (
            plot_narrative_distribution,
            plot_market_cap_analysis,
            plot_temporal_analysis,
            plot_returns_analysis
        )
        print("   ✅ Test 1: Importaciones exitosas")
        tests_passed += 1
        
        # Crear datos
        df = create_test_data()
        from utils.config import NARRATIVE_COLORS
        
        # Test 2: Distribución narrativas
        total_tests += 1
        try:
            fig = plot_narrative_distribution(df, NARRATIVE_COLORS)
            if fig is not None:
                print("   ✅ Test 2: Gráfico distribución narrativas")
                tests_passed += 1
            else:
                print("   ❌ Test 2: No se generó gráfico")
        except Exception as e:
            print(f"   ⚠️  Test 2: Error en distribución - {e}")
        
        # Test 3: Market cap analysis
        total_tests += 1
        try:
            fig = plot_market_cap_analysis(df, NARRATIVE_COLORS)
            if fig is not None:
                print("   ✅ Test 3: Análisis market cap")
                tests_passed += 1
            else:
                print("   ❌ Test 3: No se generó análisis")
        except Exception as e:
            print(f"   ⚠️  Test 3: Error en market cap - {e}")
        
        # Test 4: Análisis temporal
        total_tests += 1
        try:
            fig = plot_temporal_analysis(df, NARRATIVE_COLORS)
            if fig is not None:
                print("   ✅ Test 4: Análisis temporal")
                tests_passed += 1
            else:
                print("   ❌ Test 4: No se generó análisis temporal")
        except Exception as e:
            print(f"   ⚠️  Test 4: Error en temporal - {e}")
        
        success_rate = (tests_passed / total_tests) * 100
        print(f"   📊 visualizations: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
        return tests_passed, total_tests
        
    except Exception as e:
        print(f"   ❌ Error general en visualizations: {e}")
        return 0, total_tests

def test_config():
    """Test del módulo config"""
    print("⚙️ === Testing config ===")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        # Test 1: Importaciones
        total_tests += 1
        from utils.config import (
            NARRATIVE_COLORS,
            QUALITY_THRESHOLDS,
            ANALYSIS_CONFIG,
            get_project_paths,
            setup_plotting_style
        )
        print("   ✅ Test 1: Importaciones exitosas")
        tests_passed += 1
        
        # Test 2: Colores narrativas
        total_tests += 1
        if isinstance(NARRATIVE_COLORS, dict) and len(NARRATIVE_COLORS) > 0:
            print(f"   ✅ Test 2: Colores narrativas - {len(NARRATIVE_COLORS)} definidos")
            tests_passed += 1
        else:
            print("   ❌ Test 2: Error en colores")
        
        # Test 3: Umbrales calidad
        total_tests += 1
        if isinstance(QUALITY_THRESHOLDS, dict) and 'excellent' in QUALITY_THRESHOLDS:
            print("   ✅ Test 3: Umbrales de calidad configurados")
            tests_passed += 1
        else:
            print("   ❌ Test 3: Error en umbrales")
        
        # Test 4: Config análisis
        total_tests += 1
        if isinstance(ANALYSIS_CONFIG, dict) and len(ANALYSIS_CONFIG) > 0:
            print(f"   ✅ Test 4: Config análisis - {len(ANALYSIS_CONFIG)} parámetros")
            tests_passed += 1
        else:
            print("   ❌ Test 4: Error en config")
        
        # Test 5: Rutas proyecto
        total_tests += 1
        try:
            paths = get_project_paths()
            if isinstance(paths, dict) and 'root' in paths:
                print("   ✅ Test 5: Rutas del proyecto configuradas")
                tests_passed += 1
            else:
                print("   ❌ Test 5: Error en rutas")
        except Exception as e:
            print(f"   ⚠️  Test 5: Error en rutas - {e}")
        
        success_rate = (tests_passed / total_tests) * 100
        print(f"   📊 config: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
        return tests_passed, total_tests
        
    except Exception as e:
        print(f"   ❌ Error general en config: {e}")
        return 0, total_tests

def main():
    """Ejecutar todos los tests"""
    print("🧪" + "="*60)
    print("🚀 SISTEMA DE TESTING FUNCIONAL")
    print("🧪" + "="*60)
    print()
    
    # Ejecutar tests
    modules = {
        'data_analysis': test_data_analysis,
        'feature_engineering': test_feature_engineering,
        'visualizations': test_visualizations,
        'config': test_config
    }
    
    total_passed = 0
    total_tests = 0
    results = {}
    
    for module_name, test_func in modules.items():
        try:
            passed, tests = test_func()
            results[module_name] = (passed, tests)
            total_passed += passed
            total_tests += tests
        except Exception as e:
            print(f"❌ Error ejecutando {module_name}: {e}")
            results[module_name] = (0, 1)
            total_tests += 1
    
    # Reporte final
    print("\n" + "🏁" + "="*60)
    print("📊 REPORTE FINAL")
    print("🏁" + "="*60)
    
    overall_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📈 Tasa de éxito general: {overall_rate:.1f}%")
    print(f"✅ Tests pasados: {total_passed}")
    print(f"❌ Tests fallidos: {total_tests - total_passed}")
    print(f"🧪 Total tests: {total_tests}")
    print()
    
    print("📋 Resultados por módulo:")
    for module, (passed, total) in results.items():
        rate = (passed / total * 100) if total > 0 else 0
        status = "✅" if rate >= 80 else "⚠️" if rate >= 60 else "❌"
        print(f"   {status} {module}: {passed}/{total} ({rate:.1f}%)")
    
    print("\n" + "🎯" + "="*60)
    
    if overall_rate >= 80:
        print("🎉 ¡EXCELENTE! El sistema está funcionando correctamente")
        return True
    elif overall_rate >= 60:
        print("⚠️  BUENO: El sistema funciona con algunas mejoras necesarias")
        return True
    else:
        print("❌ NECESITA ATENCIÓN: Hay problemas que requieren corrección")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
