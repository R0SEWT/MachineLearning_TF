"""
🧪 SUITE DE TESTING PARA EDA CRIPTOMONEDAS
==========================================

Estructura de testing profesional organizada en módulos:
- test_data_analysis.py: Tests para análisis de datos
- test_feature_engineering.py: Tests para feature engineering
- test_visualizations.py: Tests para visualizaciones  
- test_config.py: Tests para configuración
- test_integration.py: Tests de integración
- test_performance.py: Tests de rendimiento

Utilidades:
- test_utils.py: Utilidades comunes para testing
- conftest.py: Configuración pytest
- run_all_tests.py: Ejecutor principal
"""

__version__ = "1.0.0"
__author__ = "EDA Testing Suite"

# Importaciones principales para la suite de testing
from .test_utils import create_test_data, TestResult, TestSuite
from .run_all_tests import run_complete_test_suite

__all__ = [
    'create_test_data',
    'TestResult', 
    'TestSuite',
    'run_complete_test_suite'
]
