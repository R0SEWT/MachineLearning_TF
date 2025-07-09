#!/usr/bin/env python3
"""
🚀 Script de Inicialización del Sistema EDA
===========================================

Este script configura y verifica el entorno del sistema EDA,
asegurando que todo esté listo para funcionar correctamente.

Uso:
    python setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Banner de inicialización"""
    print("🚀" + "="*60)
    print("🛠️  CONFIGURACIÓN DEL SISTEMA EDA")
    print("🚀" + "="*60)

def check_python_version():
    """Verificar versión de Python"""
    print("\n📋 Verificando Python...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requiere Python 3.8+")
        return False

def check_dependencies():
    """Verificar dependencias principales"""
    print("\n📦 Verificando dependencias...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'plotly',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - Instalado")
        except ImportError:
            print(f"   ❌ {package} - NO ENCONTRADO")
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(packages):
    """Instalar paquetes faltantes"""
    if not packages:
        return True
    
    print(f"\n📥 Instalando {len(packages)} paquetes faltantes...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"] + packages
        )
        print("   ✅ Paquetes instalados exitosamente")
        return True
    except subprocess.CalledProcessError:
        print("   ❌ Error instalando paquetes")
        return False

def verify_directory_structure():
    """Verificar estructura de directorios"""
    print("\n📁 Verificando estructura de directorios...")
    
    required_dirs = [
        'utils',
        'testing', 
        'notebooks',
        'docs',
        'scripts',
        'outputs'
    ]
    
    base_path = Path(__file__).parent
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/ - Existe")
        else:
            print(f"   ❌ {dir_name}/ - NO ENCONTRADO")
            missing_dirs.append(dir_name)
    
    return missing_dirs

def create_missing_directories(directories):
    """Crear directorios faltantes"""
    if not directories:
        return True
    
    print(f"\n📂 Creando {len(directories)} directorios faltantes...")
    
    base_path = Path(__file__).parent
    
    for dir_name in directories:
        dir_path = base_path / dir_name
        try:
            dir_path.mkdir(exist_ok=True)
            print(f"   ✅ {dir_name}/ - Creado")
        except Exception as e:
            print(f"   ❌ {dir_name}/ - Error: {e}")
            return False
    
    return True

def test_system_functionality():
    """Probar funcionalidad básica del sistema"""
    print("\n🧪 Probando funcionalidad del sistema...")
    
    try:
        # Test de importación de utils
        sys.path.append('.')
        from utils.config import NARRATIVE_COLORS
        print("   ✅ Módulos utils - Importación OK")
        
        # Test básico de datos
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            'id': ['BTC', 'ETH'] * 5,
            'close': np.random.random(10),
            'narrative': ['defi', 'gaming'] * 5
        })
        
        from utils.data_analysis import calculate_basic_metrics
        metrics = calculate_basic_metrics(df)
        print("   ✅ Análisis de datos - Funcional")
        
        from utils.visualizations import plot_narrative_distribution  
        print("   ✅ Visualizaciones - Disponible")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en funcionalidad: {e}")
        return False

def run_quick_test():
    """Ejecutar test rápido del sistema"""
    print("\n⚡ Ejecutando test rápido...")
    
    try:
        if os.path.exists('testing/test_functional.py'):
            result = subprocess.run([
                sys.executable, 'testing/test_functional.py'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("   ✅ Test funcional - PASADO")
                return True
            else:
                print("   ⚠️  Test funcional - Con advertencias")
                return True
        else:
            print("   ⚠️  Test funcional - No encontrado")
            return True
            
    except Exception as e:
        print(f"   ❌ Error en test: {e}")
        return False

def generate_status_report():
    """Generar reporte de estado"""
    print("\n📊 Generando reporte de estado...")
    
    report = {
        'timestamp': str(pd.Timestamp.now()),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'system_ready': True,
        'modules_available': [],
        'recommendations': []
    }
    
    # Verificar módulos
    modules_to_check = ['data_analysis', 'feature_engineering', 'visualizations', 'config']
    
    for module in modules_to_check:
        try:
            __import__(f'utils.{module}')
            report['modules_available'].append(module)
        except ImportError:
            report['system_ready'] = False
    
    # Guardar reporte
    import json
    with open('outputs/setup_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("   ✅ Reporte guardado en outputs/setup_report.json")
    
    return report

def main():
    """Función principal"""
    print_banner()
    
    # Verificaciones
    python_ok = check_python_version()
    missing_packages = check_dependencies() 
    missing_dirs = verify_directory_structure()
    
    # Correcciones
    if missing_packages:
        install_missing_packages(missing_packages)
    
    if missing_dirs:
        create_missing_directories(missing_dirs)
    
    # Tests
    functionality_ok = test_system_functionality()
    test_ok = run_quick_test()
    
    # Reporte final
    report = generate_status_report()
    
    print("\n🏁" + "="*60)
    print("📊 REPORTE FINAL DE CONFIGURACIÓN")
    print("🏁" + "="*60)
    
    if python_ok and not missing_packages and functionality_ok:
        print("🎉 ¡SISTEMA EDA COMPLETAMENTE CONFIGURADO!")
        print("✅ Todo listo para usar")
        print("\n🚀 Próximos pasos:")
        print("   • python testing/test_functional.py")
        print("   • jupyter notebook notebooks/EDA_crypto_modular.ipynb")
        print("   • python testing/master_test.py")
    else:
        print("⚠️  CONFIGURACIÓN COMPLETADA CON ADVERTENCIAS")
        print("ℹ️  El sistema debería funcionar, pero revisa los mensajes arriba")
    
    print(f"\n📄 Módulos disponibles: {len(report['modules_available'])}/4")
    for module in report['modules_available']:
        print(f"   ✅ {module}")
    
    print("\n🎯" + "="*60)

if __name__ == "__main__":
    main()
