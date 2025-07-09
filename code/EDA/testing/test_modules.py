#!/usr/bin/env python3
"""
Script de prueba para verificar todos los módulos de utils
"""

import sys
import os
sys.path.append('.')

def test_data_analysis():
    """Prueba el módulo data_analysis"""
    print("=== Probando módulo data_analysis ===")
    
    try:
        from utils.data_analysis import (
            get_basic_metrics, 
            calculate_basic_metrics,
            evaluate_data_quality,
            detect_outliers_iqr,
            calculate_distribution_stats,
            calculate_market_dominance,
            generate_summary_report
        )
        print("✅ Todas las funciones importadas correctamente")
        
        # Crear datos de prueba
        import pandas as pd
        import numpy as np
        
        test_data = {
            'id': ['BTC', 'ETH', 'ADA'] * 100,
            'narrative': ['defi', 'defi', 'gaming'] * 100,
            'close': np.random.normal(50000, 5000, 300),
            'market_cap': np.random.normal(1e12, 1e11, 300),
            'date': pd.date_range('2024-01-01', periods=300)
        }
        df = pd.DataFrame(test_data)
        
        # Probar funciones
        metrics = get_basic_metrics(df)
        print(f"✅ get_basic_metrics: {len(metrics)} métricas calculadas")
        
        outliers_count, outliers_pct = detect_outliers_iqr(df['close'])
        print(f"✅ detect_outliers_iqr: {outliers_count} outliers ({outliers_pct:.1f}%)")
        
        dominance = calculate_market_dominance(df)
        print(f"✅ calculate_market_dominance: {len(dominance)} narrativas analizadas")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en data_analysis: {e}")
        return False

def test_config():
    """Prueba el módulo config"""
    print("\n=== Probando módulo config ===")
    
    try:
        from utils.config import NARRATIVE_COLORS, QUALITY_THRESHOLDS, PATHS
        print("✅ Configuración importada correctamente")
        print(f"✅ Colores de narrativa: {len(NARRATIVE_COLORS)} definidos")
        print(f"✅ Umbrales de calidad: {len(QUALITY_THRESHOLDS)} definidos")
        return True
        
    except Exception as e:
        print(f"❌ Error en config: {e}")
        return False

def test_visualizations():
    """Prueba el módulo visualizations"""
    print("\n=== Probando módulo visualizations ===")
    
    try:
        from utils.visualizations import (
            plot_narrative_distribution,
            plot_market_cap_analysis,
            plot_temporal_patterns,
            create_dashboard
        )
        print("✅ Funciones de visualización importadas correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en visualizations: {e}")
        return False

def test_feature_engineering():
    """Prueba el módulo feature_engineering"""
    print("\n=== Probando módulo feature_engineering ===")
    
    try:
        from utils.feature_engineering import (
            calculate_returns,
            calculate_technical_indicators,
            create_features,
            apply_clustering
        )
        print("✅ Funciones de feature engineering importadas correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en feature_engineering: {e}")
        return False

def main():
    """Función principal de prueba"""
    print("🧪 INICIANDO PRUEBAS DE MÓDULOS UTILS")
    print("=" * 50)
    
    results = []
    results.append(test_data_analysis())
    results.append(test_config())
    results.append(test_visualizations())
    results.append(test_feature_engineering())
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS:")
    
    if all(results):
        print("🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
        print("✅ Los módulos están correctamente modularizados y funcionando")
    else:
        print("⚠️ Algunas pruebas fallaron:")
        modules = ['data_analysis', 'config', 'visualizations', 'feature_engineering']
        for i, result in enumerate(results):
            status = "✅" if result else "❌"
            print(f"   {status} {modules[i]}")

if __name__ == "__main__":
    main()
