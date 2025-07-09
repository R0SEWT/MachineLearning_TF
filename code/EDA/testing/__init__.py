"""
🧪 Sistema de Testing Profesional para EDA de Criptomonedas
============================================================

Este paquete contiene todos los tests, utilities y herramientas de testing
para el sistema EDA de análisis de criptomonedas.

Estructura:
- test_*.py: Tests individuales por módulo
- fixtures/: Datos de prueba reutilizables
- reports/: Reportes de testing generados
- test_runner.py: Ejecutor principal de tests

Uso:
    from testing import run_all_tests
    run_all_tests()

Autor: Sistema EDA Testing
Versión: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Sistema EDA Testing"

# Imports principales
from .test_runner import TestRunner, run_all_tests
from .test_utils import create_test_data, TestResult

__all__ = [
    'TestRunner',
    'run_all_tests', 
    'create_test_data',
    'TestResult'
]
