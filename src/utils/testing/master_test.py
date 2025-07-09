#!/usr/bin/env python3
"""
🚀 MAESTRO DE TESTS - Ejecutor Universal
========================================

Script maestro para ejecutar cualquier test del sistema EDA.
Detecta automáticamente qué tests están disponibles y permite
ejecutarlos de forma individual o grupal.

Uso:
    python testing/master_test.py                    # Menú interactivo
    python testing/master_test.py --functional       # Test funcional
    python testing/master_test.py --smart           # Test inteligente
    python testing/master_test.py --all             # Todos los tests
    python testing/master_test.py --list            # Listar disponibles
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Configurar path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def print_banner():
    """Banner principal"""
    print("🚀" + "="*70)
    print("🧪 MAESTRO DE TESTS - SISTEMA EDA CRIPTOMONEDAS")
    print("🚀" + "="*70)

def detect_available_tests():
    """Detectar tests disponibles"""
    test_dir = Path(__file__).parent
    available_tests = {}
    
    # Tests principales recomendados
    priority_tests = {
        'test_functional.py': '✅ Tests funcionales (100% compatibles)',
        'test_smart.py': '🧠 Tests inteligentes (auto-adaptivos)',
        'test_professional.py': '🏆 Tests profesionales (suite completa)',
        'test_definitive.py': '🎯 Tests definitivos',
    }
    
    # Verificar cuáles existen
    for test_file, description in priority_tests.items():
        test_path = test_dir / test_file
        if test_path.exists():
            available_tests[test_file] = description
    
    # Buscar otros tests
    other_tests = list(test_dir.glob('test_*.py'))
    for test_path in other_tests:
        test_file = test_path.name
        if test_file not in available_tests and test_file != 'test_utils.py':
            available_tests[test_file] = f'📋 {test_file} (test adicional)'
    
    return available_tests

def list_available_tests():
    """Listar tests disponibles"""
    tests = detect_available_tests()
    
    print("\n🧪 Tests Disponibles:")
    print("-" * 50)
    
    for test_file, description in tests.items():
        print(f"   {description}")
        print(f"      Ejecutar: python testing/{test_file}")
    
    print(f"\n📊 Total: {len(tests)} tests disponibles")

def run_specific_test(test_name):
    """Ejecutar un test específico"""
    test_dir = Path(__file__).parent
    test_path = test_dir / test_name
    
    if not test_path.exists():
        print(f"❌ Error: {test_name} no encontrado")
        return False
    
    print(f"\n🔥 Ejecutando {test_name}...")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Cambiar al directorio correcto
        original_cwd = os.getcwd()
        os.chdir(test_dir)
        
        # Ejecutar el test
        exit_code = os.system(f"python {test_name}")
        
        # Restaurar directorio
        os.chdir(original_cwd)
        
        execution_time = time.time() - start_time
        
        if exit_code == 0:
            print(f"\n✅ {test_name} completado exitosamente ({execution_time:.2f}s)")
            return True
        else:
            print(f"\n❌ {test_name} terminó con errores ({execution_time:.2f}s)")
            return False
            
    except Exception as e:
        print(f"\n💥 Error ejecutando {test_name}: {e}")
        return False

def run_all_tests():
    """Ejecutar todos los tests recomendados"""
    print("\n🎯 Ejecutando TODOS los tests principales...")
    
    priority_tests = [
        'test_functional.py',
        'test_smart.py', 
        'test_professional.py',
        'test_definitive.py'
    ]
    
    results = {}
    total_time = 0
    
    for test_name in priority_tests:
        test_path = Path(__file__).parent / test_name
        if test_path.exists():
            print(f"\n{'='*50}")
            start_time = time.time()
            success = run_specific_test(test_name)
            exec_time = time.time() - start_time
            total_time += exec_time
            results[test_name] = success
        else:
            print(f"\n⚠️  {test_name} no encontrado, saltando...")
    
    # Reporte final
    print("\n" + "🏁" + "="*50)
    print("📊 REPORTE FINAL DE TODOS LOS TESTS")
    print("🏁" + "="*50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"⏱️  Tiempo total: {total_time:.2f}s")
    print(f"📈 Tests exitosos: {passed}/{total} ({success_rate:.1f}%)")
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}")
    
    print("\n🎯" + "="*50)
    if success_rate >= 75:
        print("🎉 ¡EXCELENTE! La mayoría de tests pasaron")
    elif success_rate >= 50:
        print("👍 ACEPTABLE. Algunos tests necesitan atención")
    else:
        print("⚠️  CRÍTICO. Muchos tests fallando")
    print("🎯" + "="*50)

def interactive_menu():
    """Menú interactivo"""
    tests = detect_available_tests()
    
    while True:
        print("\n🎮 MENÚ INTERACTIVO")
        print("-" * 30)
        print("1. 📋 Listar tests disponibles")
        print("2. ✅ Ejecutar test funcional")
        print("3. 🧠 Ejecutar test inteligente")
        print("4. 🏆 Ejecutar test profesional")
        print("5. 🎯 Ejecutar todos los tests")
        print("6. 🔧 Ejecutar test personalizado")
        print("0. 🚪 Salir")
        
        choice = input("\n👉 Selecciona una opción: ").strip()
        
        if choice == "0":
            print("👋 ¡Hasta luego!")
            break
        elif choice == "1":
            list_available_tests()
        elif choice == "2":
            run_specific_test("test_functional.py")
        elif choice == "3":
            run_specific_test("test_smart.py")
        elif choice == "4":
            run_specific_test("test_professional.py")
        elif choice == "5":
            run_all_tests()
        elif choice == "6":
            print("\n📋 Tests disponibles:")
            for i, test_name in enumerate(tests.keys(), 1):
                print(f"   {i}. {test_name}")
            
            try:
                test_idx = int(input("\n👉 Número del test: ")) - 1
                test_names = list(tests.keys())
                if 0 <= test_idx < len(test_names):
                    run_specific_test(test_names[test_idx])
                else:
                    print("❌ Número inválido")
            except ValueError:
                print("❌ Por favor ingresa un número")
        else:
            print("❌ Opción inválida")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Maestro de Tests EDA")
    parser.add_argument('--functional', action='store_true', help='Ejecutar test funcional')
    parser.add_argument('--smart', action='store_true', help='Ejecutar test inteligente')
    parser.add_argument('--professional', action='store_true', help='Ejecutar test profesional')
    parser.add_argument('--definitive', action='store_true', help='Ejecutar test definitivo')
    parser.add_argument('--all', action='store_true', help='Ejecutar todos los tests')
    parser.add_argument('--list', action='store_true', help='Listar tests disponibles')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.list:
        list_available_tests()
    elif args.functional:
        run_specific_test('test_functional.py')
    elif args.smart:
        run_specific_test('test_smart.py')
    elif args.professional:
        run_specific_test('test_professional.py')
    elif args.definitive:
        run_specific_test('test_definitive.py')
    elif args.all:
        run_all_tests()
    else:
        # Menú interactivo por defecto
        interactive_menu()

if __name__ == "__main__":
    main()
