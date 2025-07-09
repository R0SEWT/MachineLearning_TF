#!/usr/bin/env python3
"""
🚀 Ejecutor Principal de Tests
==============================

Script principal para ejecutar todos los tests del sistema EDA.

Uso:
    python run_tests.py
    
    O desde la carpeta principal:
    python testing/run_tests.py
"""

import sys
import os
import warnings

# Configurar path y suprimir warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

def main():
    """Función principal"""
    print("🧪 Iniciando sistema de testing profesional...")
    
    try:
        from test_runner import run_all_tests
        result = run_all_tests()
        
        # Mostrar información adicional
        print(f"\n📈 Resumen ejecutivo:")
        print(f"   • Tasa de éxito: {result['overall_success_rate']:.1f}%")
        print(f"   • Tiempo de ejecución: {result['execution_time']:.2f}s")
        print(f"   • Tests ejecutados: {result['total_tests']}")
        
        # Código de salida basado en el éxito
        exit_code = 0 if result['overall_success_rate'] >= 80 else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ Error ejecutando tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
