"""
🚀 Gestor de Imports Inteligente - Fase 5
=========================================

Sistema inteligente de gestión de imports que resuelve automáticamente
las rutas de imports problemáticos que existían en el sistema anterior.

Autor: Sistema de Optimización IA
Fecha: 2025-01-09 (Fase 5 - Organización)
"""

import importlib
import importlib.util
import sys
import os
from typing import Any, Optional, Dict, List, Union
from pathlib import Path
import warnings


class ImportManager:
    """
    Gestor inteligente de imports que resuelve automáticamente
    dependencias y rutas problemáticas.
    """
    
    def __init__(self):
        """Inicializar el gestor de imports"""
        self._cache: Dict[str, Any] = {}
        self._search_paths: List[str] = []
        self._failed_imports: List[str] = []
        self._setup_search_paths()
    
    def _setup_search_paths(self):
        """Configurar rutas de búsqueda automática"""
        # Directorio actual
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Rutas base del proyecto
        base_paths = [
            current_dir,
            os.path.join(current_dir, ".."),
            os.path.join(current_dir, "..", ".."),
            os.path.join(current_dir, "..", "..", "src"),
            os.path.join(current_dir, "..", "..", "src", "utils"),
            os.path.join(current_dir, "..", "utils"),
            "/home/exodia/Documentos/MachineLearning_TF/src",
            "/home/exodia/Documentos/MachineLearning_TF/src/utils",
            "/home/exodia/Documentos/MachineLearning_TF/scripts/optimization"
        ]
        
        # Agregar rutas que existen
        for path in base_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and abs_path not in self._search_paths:
                self._search_paths.append(abs_path)
                if abs_path not in sys.path:
                    sys.path.insert(0, abs_path)
    
    def safe_import(self, module_name: str, fallback_paths: Optional[List[str]] = None) -> Optional[Any]:
        """
        Importar módulo de forma segura con múltiples intentos.
        
        Args:
            module_name: Nombre del módulo a importar
            fallback_paths: Rutas alternativas a intentar
            
        Returns:
            Módulo importado o None si falla
        """
        # Verificar cache primero
        if module_name in self._cache:
            return self._cache[module_name]
        
        # Lista de intentos de import
        import_attempts = [module_name]
        
        # Agregar rutas de fallback
        if fallback_paths:
            import_attempts.extend(fallback_paths)
        
        # Intentar cada ruta
        for attempt in import_attempts:
            try:
                module = importlib.import_module(attempt)
                self._cache[module_name] = module
                return module
            except ImportError as e:
                continue
        
        # Si falla, intentar importar desde rutas de búsqueda
        for search_path in self._search_paths:
            try:
                # Buscar archivo en el directorio
                for file_ext in ['.py', '/__init__.py']:
                    module_file = os.path.join(search_path, module_name.replace('.', '/') + file_ext)
                    if os.path.exists(module_file):
                        spec = importlib.util.spec_from_file_location(module_name, module_file)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            self._cache[module_name] = module
                            return module
            except Exception:
                continue
        
        # Registrar fallo
        if module_name not in self._failed_imports:
            self._failed_imports.append(module_name)
            warnings.warn(f"No se pudo importar {module_name}")
        
        return None
    
    def get_feature_engineering_module(self):
        """Obtener módulo de feature engineering con múltiples intentos"""
        attempts = [
            "feature_engineering",
            "utils.feature_engineering",
            "utils.utils.feature_engineering",
            "src.utils.utils.feature_engineering",
            "optimization.utils.feature_engineering"
        ]
        
        return self.safe_import("feature_engineering", attempts)
    
    def get_crypto_ml_trainer_module(self):
        """Obtener módulo de crypto ML trainer con múltiples intentos"""
        attempts = [
            "crypto_ml_trainer",
            "utils.crypto_ml_trainer",
            "optimization.crypto_ml_trainer",
            "advanced_crypto_hyperparameter_optimizer"
        ]
        
        return self.safe_import("crypto_ml_trainer", attempts)
    
    def get_data_processing_module(self):
        """Obtener módulo de procesamiento de datos"""
        attempts = [
            "data_processing",
            "src.data_processing",
            "utils.data_processing"
        ]
        
        return self.safe_import("data_processing", attempts)
    
    def import_with_fallback(self, primary: str, fallbacks: List[str]) -> Optional[Any]:
        """
        Importar con lista de fallbacks específicos.
        
        Args:
            primary: Import primario a intentar
            fallbacks: Lista de imports de fallback
            
        Returns:
            Módulo importado o None
        """
        # Intentar import primario
        module = self.safe_import(primary)
        if module:
            return module
        
        # Intentar fallbacks
        for fallback in fallbacks:
            module = self.safe_import(fallback)
            if module:
                return module
        
        return None
    
    def clear_cache(self):
        """Limpiar cache de imports"""
        self._cache.clear()
    
    def get_import_status(self) -> Dict[str, Any]:
        """Obtener estado de imports"""
        return {
            "cached_modules": list(self._cache.keys()),
            "failed_imports": self._failed_imports,
            "search_paths": self._search_paths
        }


# ==================== FUNCIONES DE CONVENIENCIA ====================

# Instancia global del gestor
_import_manager = ImportManager()

def safe_import(module_name: str, fallback_paths: Optional[List[str]] = None) -> Optional[Any]:
    """Función de conveniencia para import seguro"""
    return _import_manager.safe_import(module_name, fallback_paths)

def get_feature_engineering_module():
    """Obtener módulo de feature engineering de forma segura"""
    return _import_manager.get_feature_engineering_module()

def get_crypto_ml_trainer_module():
    """Obtener módulo de crypto ML trainer de forma segura"""
    return _import_manager.get_crypto_ml_trainer_module()

def get_data_processing_module():
    """Obtener módulo de procesamiento de datos de forma segura"""
    return _import_manager.get_data_processing_module()

def import_with_fallback(primary: str, fallbacks: List[str]) -> Optional[Any]:
    """Importar con fallbacks específicos"""
    return _import_manager.import_with_fallback(primary, fallbacks)


# ==================== IMPORTS ESPECÍFICOS DEL SISTEMA ====================

def get_ml_libraries():
    """Obtener librerías de ML con imports seguros"""
    libraries = {}
    
    # XGBoost
    xgb = safe_import("xgboost")
    if xgb:
        libraries["xgboost"] = xgb
    
    # LightGBM
    lgb = safe_import("lightgbm")
    if lgb:
        libraries["lightgbm"] = lgb
    
    # CatBoost
    catboost = safe_import("catboost")
    if catboost:
        libraries["catboost"] = catboost
    
    # Scikit-learn
    sklearn = safe_import("sklearn")
    if sklearn:
        libraries["sklearn"] = sklearn
    
    # Pandas
    pandas = safe_import("pandas")
    if pandas:
        libraries["pandas"] = pandas
    
    # NumPy
    numpy = safe_import("numpy")
    if numpy:
        libraries["numpy"] = numpy
    
    # Optuna
    optuna = safe_import("optuna")
    if optuna:
        libraries["optuna"] = optuna
    
    return libraries

def get_visualization_libraries():
    """Obtener librerías de visualización con imports seguros"""
    libraries = {}
    
    # Matplotlib
    matplotlib = safe_import("matplotlib.pyplot")
    if matplotlib:
        libraries["matplotlib"] = matplotlib
    
    # Seaborn
    seaborn = safe_import("seaborn")
    if seaborn:
        libraries["seaborn"] = seaborn
    
    # Plotly
    plotly = safe_import("plotly")
    if plotly:
        libraries["plotly"] = plotly
    
    return libraries

def get_system_libraries():
    """Obtener librerías del sistema con imports seguros"""
    libraries = {}
    
    # GPU Utils
    gputil = safe_import("GPUtil")
    if gputil:
        libraries["gputil"] = gputil
    
    # PSUtil
    psutil = safe_import("psutil")
    if psutil:
        libraries["psutil"] = psutil
    
    # TensorFlow
    tensorflow = safe_import("tensorflow")
    if tensorflow:
        libraries["tensorflow"] = tensorflow
    
    return libraries


# ==================== DECORADORES DE IMPORT ====================

def require_module(module_name: str, fallbacks: Optional[List[str]] = None):
    """
    Decorador que requiere que un módulo esté disponible.
    
    Args:
        module_name: Nombre del módulo requerido
        fallbacks: Módulos de fallback opcionales
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            module = safe_import(module_name, fallbacks)
            if module is None:
                raise ImportError(f"Módulo requerido '{module_name}' no disponible")
            
            # Agregar módulo como primer argumento
            return func(module, *args, **kwargs)
        
        return wrapper
    return decorator

def optional_module(module_name: str, fallbacks: Optional[List[str]] = None):
    """
    Decorador que proporciona un módulo opcional.
    
    Args:
        module_name: Nombre del módulo opcional
        fallbacks: Módulos de fallback opcionales
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            module = safe_import(module_name, fallbacks)
            # Agregar módulo como primer argumento (puede ser None)
            return func(module, *args, **kwargs)
        
        return wrapper
    return decorator


# ==================== UTILIDADES DE DIAGNÓSTICO ====================

def diagnose_imports() -> Dict[str, Any]:
    """Diagnosticar estado de imports del sistema"""
    print("🔍 Diagnóstico de Imports - Sistema de Optimización")
    print("=" * 50)
    
    # ML Libraries
    print("\n📊 Librerías de Machine Learning:")
    ml_libs = get_ml_libraries()
    for name, lib in ml_libs.items():
        version = getattr(lib, "__version__", "Unknown")
        print(f"  ✅ {name}: {version}")
    
    # Missing ML libraries
    expected_ml = ["xgboost", "lightgbm", "catboost", "sklearn", "pandas", "numpy", "optuna"]
    missing_ml = [lib for lib in expected_ml if lib not in ml_libs]
    if missing_ml:
        print(f"  ❌ Faltantes: {', '.join(missing_ml)}")
    
    # Visualization Libraries
    print("\n📈 Librerías de Visualización:")
    vis_libs = get_visualization_libraries()
    for name, lib in vis_libs.items():
        version = getattr(lib, "__version__", "Unknown")
        print(f"  ✅ {name}: {version}")
    
    # System Libraries
    print("\n🖥️  Librerías del Sistema:")
    sys_libs = get_system_libraries()
    for name, lib in sys_libs.items():
        version = getattr(lib, "__version__", "Unknown")
        print(f"  ✅ {name}: {version}")
    
    # Import status
    status = _import_manager.get_import_status()
    print(f"\n📦 Estado de Imports:")
    print(f"  - Módulos en cache: {len(status['cached_modules'])}")
    print(f"  - Imports fallidos: {len(status['failed_imports'])}")
    print(f"  - Rutas de búsqueda: {len(status['search_paths'])}")
    
    if status['failed_imports']:
        print(f"  ❌ Fallidos: {', '.join(status['failed_imports'])}")
    
    return {
        "ml_libraries": ml_libs,
        "visualization_libraries": vis_libs,
        "system_libraries": sys_libs,
        "import_status": status
    }

def test_critical_imports() -> bool:
    """Probar imports críticos del sistema"""
    critical_modules = [
        "pandas",
        "numpy",
        "sklearn",
        "optuna"
    ]
    
    all_available = True
    print("🧪 Probando imports críticos...")
    
    for module in critical_modules:
        lib = safe_import(module)
        if lib:
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module}")
            all_available = False
    
    return all_available


if __name__ == "__main__":
    # Demo del gestor de imports
    print("🚀 Gestor de Imports Inteligente - Fase 5")
    print("========================================")
    
    # Diagnóstico completo
    diagnose_imports()
    
    # Probar imports críticos
    print("\n" + "=" * 50)
    if test_critical_imports():
        print("✅ Todos los imports críticos disponibles")
    else:
        print("❌ Algunos imports críticos faltan")
    
    # Probar imports específicos del sistema
    print("\n🧪 Probando imports específicos del sistema:")
    
    fe_module = get_feature_engineering_module()
    if fe_module:
        print("  ✅ Feature Engineering disponible")
    else:
        print("  ⚠️  Feature Engineering no encontrado")
    
    trainer_module = get_crypto_ml_trainer_module()
    if trainer_module:
        print("  ✅ Crypto ML Trainer disponible")
    else:
        print("  ⚠️  Crypto ML Trainer no encontrado")
