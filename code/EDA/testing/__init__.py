"""
🧪 Sistema de Testing Profesional para EDA de Criptomonedas
============================================================

Este paquete contiene todos los tests, utilities y herramientas de testing
para el sistema EDA de análisis de criptomonedas.

Estructura:
- test_*.py: Tests individuales y sistemas completos
- fixtures/: Datos de prueba reutilizables
- reports/: Reportes de testing generados
- run_tests.py: Ejecutor principal de tests

Uso principal:
    python testing/test_functional.py    # Tests 100% funcionales
    python testing/test_smart.py         # Tests inteligentes adaptivos
    python testing/run_tests.py          # Sistema completo

Autor: Sistema EDA Testing
Versión: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Sistema EDA Testing"

# Tests disponibles
AVAILABLE_TESTS = [
    'test_functional.py',      # Tests básicos 100% compatibles
    'test_smart.py',           # Tests inteligentes auto-adaptivos  
    'test_professional.py',    # Suite completa con edge cases
    'test_definitive.py',      # Tests definitivos
    'run_tests.py'             # Ejecutor principal
]

def list_available_tests():
    """Listar tests disponibles"""
    print("🧪 Tests Disponibles:")
    for test in AVAILABLE_TESTS:
        print(f"   • {test}")

__all__ = ['AVAILABLE_TESTS', 'list_available_tests']
