#!/usr/bin/env python3
"""
📊 TESTS PARA DATA ANALYSIS MODULE
==================================

Tests específicos para el módulo utils.data_analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from test_utils import TestSuite, create_test_data, safe_import


def test_data_analysis_imports():
    """Test de importaciones del módulo data_analysis"""
    try:
        from utils.data_analysis import (
            calculate_basic_metrics,
            evaluate_data_quality, 
            calculate_market_dominance,
            generate_summary_report
        )
        return True, "Todas las importaciones exitosas"
    except ImportError as e:
        return False, f"Error de importación: {e}"


def test_calculate_basic_metrics():
    """Test para calculate_basic_metrics"""
    try:
        from utils.data_analysis import calculate_basic_metrics
        
        df = create_test_data(100)
        metrics = calculate_basic_metrics(df)
        
        # Verificar estructura básica
        required_keys = ['total_observations', 'total_tokens', 'total_narratives']
        if not all(key in metrics for key in required_keys):
            return False, f"Faltan claves requeridas: {required_keys}"
        
        # Verificar valores lógicos
        if metrics['total_observations'] != len(df):
            return False, f"Observaciones incorrectas: {metrics['total_observations']} vs {len(df)}"
        
        return True, f"Métricas: {metrics['total_observations']} obs, {metrics['total_tokens']} tokens"
        
    except Exception as e:
        return False, f"Error: {e}"


def test_evaluate_data_quality():
    """Test para evaluate_data_quality"""
    try:
        from utils.data_analysis import calculate_basic_metrics, evaluate_data_quality
        from utils.config import QUALITY_THRESHOLDS
        
        df = create_test_data(100)
        metrics = calculate_basic_metrics(df)
        quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
        
        # Verificar estructura
        if not isinstance(quality, dict):
            return False, "La función debe retornar un diccionario"
        
        if 'overall_status' not in quality:
            return False, "Falta 'overall_status' en el resultado"
        
        return True, f"Calidad evaluada: {quality['overall_status']}"
        
    except Exception as e:
        return False, f"Error: {e}"


def test_calculate_market_dominance():
    """Test para calculate_market_dominance"""
    try:
        from utils.data_analysis import calculate_market_dominance
        
        df = create_test_data(100)
        dominance = calculate_market_dominance(df)
        
        # Verificar que retorna algo válido
        if not hasattr(dominance, '__len__'):
            return False, "La función debe retornar una estructura con longitud"
        
        if len(dominance) == 0:
            return False, "No se calculó dominancia para ningún grupo"
        
        return True, f"Dominancia calculada para {len(dominance)} grupos"
        
    except Exception as e:
        return False, f"Error: {e}"


def test_generate_summary_report():
    """Test para generate_summary_report"""
    try:
        from utils.data_analysis import calculate_basic_metrics, evaluate_data_quality, generate_summary_report
        from utils.config import QUALITY_THRESHOLDS
        
        df = create_test_data(100)
        metrics = calculate_basic_metrics(df)
        quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
        report = generate_summary_report(metrics, quality)
        
        # Verificar que es un string no vacío
        if not isinstance(report, str):
            return False, "El reporte debe ser un string"
        
        if len(report) < 100:
            return False, f"Reporte muy corto: {len(report)} caracteres"
        
        return True, f"Reporte generado: {len(report)} caracteres"
        
    except Exception as e:
        return False, f"Error: {e}"


def test_outlier_detection():
    """Test para funciones de detección de outliers"""
    try:
        # Intentar importar cualquier función de outliers disponible
        outlier_functions = []
        
        try:
            from utils.data_analysis import detect_outliers
            outlier_functions.append(('detect_outliers', detect_outliers))
        except ImportError:
            pass
        
        try:
            from utils.data_analysis import detect_outliers_iqr
            outlier_functions.append(('detect_outliers_iqr', detect_outliers_iqr))
        except ImportError:
            pass
        
        if not outlier_functions:
            return False, "No hay funciones de detección de outliers disponibles"
        
        df = create_test_data(100)
        
        for func_name, func in outlier_functions:
            try:
                # Probar con diferentes firmas de función
                if func_name == 'detect_outliers_iqr':
                    # Esta función probablemente solo toma una serie
                    outliers = func(df['close'])
                else:
                    # Esta función probablemente toma df y columna
                    outliers = func(df, 'close')
                
                outlier_count = len(outliers) if hasattr(outliers, '__len__') else 0
                return True, f"Outliers detectados con {func_name}: {outlier_count}"
                
            except Exception as e:
                continue
        
        return False, "Ninguna función de outliers funcionó correctamente"
        
    except Exception as e:
        return False, f"Error general: {e}"


def run_data_analysis_tests():
    """Ejecutar todos los tests de data_analysis"""
    suite = TestSuite("data_analysis")
    suite.start()
    
    # Ejecutar tests
    suite.run_test(test_data_analysis_imports, "Importaciones")
    suite.run_test(test_calculate_basic_metrics, "Métricas básicas")
    suite.run_test(test_evaluate_data_quality, "Evaluación calidad")
    suite.run_test(test_calculate_market_dominance, "Dominancia mercado")
    suite.run_test(test_generate_summary_report, "Reporte resumen")
    suite.run_test(test_outlier_detection, "Detección outliers")
    
    return suite.end()


if __name__ == "__main__":
    run_data_analysis_tests()
