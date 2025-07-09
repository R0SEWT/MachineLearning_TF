#!/usr/bin/env python3
"""
🛠️ UTILIDADES PARA TESTING EDA
===============================

Módulo con utilidades comunes para todos los tests del sistema EDA.
Incluye generación de datos de prueba, clases helper y funciones auxiliares.
"""

import sys
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configurar path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

# Suprimir warnings para output más limpio
warnings.filterwarnings('ignore')

class TestResult:
    """Clase para almacenar resultados de un test individual"""
    
    def __init__(self, name: str, passed: bool, execution_time: float = 0.0, 
                 details: str = "", error: Optional[Exception] = None):
        self.name = name
        self.passed = passed
        self.execution_time = execution_time
        self.details = details
        self.error = error
        self.timestamp = datetime.now()
    
    def __str__(self):
        status = "✅" if self.passed else "❌"
        return f"{status} {self.name} ({self.execution_time:.3f}s): {self.details}"

class TestSuite:
    """Suite de tests con reporting y estadísticas"""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Iniciar la suite de tests"""
        self.start_time = time.time()
        print(f"\n🧪 === Testing {self.name} ===")
    
    def run_test(self, test_func, test_name: str, *args, **kwargs) -> TestResult:
        """Ejecutar un test individual y registrar el resultado"""
        start_time = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Interpretar resultado
            if isinstance(result, tuple):
                passed, details = result
            elif isinstance(result, bool):
                passed, details = result, "Test completado"
            else:
                passed, details = bool(result), str(result)
            
            test_result = TestResult(test_name, passed, execution_time, details)
            self.results.append(test_result)
            
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name} ({execution_time:.3f}s): {details}")
            
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            test_result = TestResult(test_name, False, execution_time, str(e), e)
            self.results.append(test_result)
            
            print(f"   ❌ {test_name} ({execution_time:.3f}s): ERROR - {str(e)}")
            return test_result
    
    def end(self):
        """Finalizar la suite y mostrar estadísticas"""
        self.end_time = time.time()
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        total_time = self.end_time - self.start_time if self.start_time else 0
        
        print(f"   📊 {self.name}: {passed_tests}/{total_tests} ({success_rate:.1f}%) en {total_time:.2f}s")
        
        return success_rate, passed_tests, total_tests

def create_test_data(size: int = 100, seed: int = 42):
    """
    Crear datos de prueba realistas para criptomonedas
    
    Args:
        size: Número de observaciones
        seed: Semilla para reproducibilidad
        
    Returns:
        DataFrame con datos de prueba
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(seed)
    
    # Tokens y narrativas realistas
    tokens = ['BTC', 'ETH', 'ADA', 'SOL', 'MATIC', 'LINK', 'DOT', 'UNI', 'AVAX', 'ATOM']
    narratives = ['defi', 'gaming', 'ai', 'meme', 'rwa', 'infrastructure']
    
    data = {
        'id': np.random.choice(tokens, size),
        'symbol': [f'SYM{i%20}' for i in range(size)],
        'name': [f'Token {i%30}' for i in range(size)],
        'narrative': np.random.choice(narratives, size),
        'close': np.random.lognormal(8, 1.2, size),
        'market_cap': np.random.lognormal(25, 1.8, size),
        'volume': np.random.lognormal(20, 1.5, size),
        'date': pd.date_range('2023-01-01', periods=size, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Agregar algunos valores faltantes de manera realista
    missing_indices = np.random.choice(df.index, min(size // 20, 10), replace=False)
    if len(missing_indices) > 0:
        df.loc[missing_indices[:len(missing_indices)//2], 'market_cap'] = np.nan
        df.loc[missing_indices[len(missing_indices)//2:], 'volume'] = np.nan
    
    return df

def safe_import(module_name: str, from_name: str = None):
    """
    Importar módulo de manera segura
    
    Args:
        module_name: Nombre del módulo
        from_name: Función/clase específica a importar
        
    Returns:
        Módulo importado o None si falla
    """
    try:
        if from_name:
            module = __import__(module_name, fromlist=[from_name])
            return getattr(module, from_name)
        else:
            return __import__(module_name)
    except ImportError as e:
        print(f"⚠️  Warning: No se pudo importar {module_name}.{from_name or ''} - {e}")
        return None

def check_function_signature(func, expected_params: List[str]) -> bool:
    """
    Verificar que una función tenga los parámetros esperados
    
    Args:
        func: Función a verificar
        expected_params: Lista de parámetros esperados
        
    Returns:
        True si la función tiene los parámetros esperados
    """
    import inspect
    
    try:
        sig = inspect.signature(func)
        func_params = list(sig.parameters.keys())
        
        # Verificar que todos los parámetros esperados estén presentes
        return all(param in func_params for param in expected_params)
    except Exception:
        return False

def measure_execution_time(func, *args, **kwargs):
    """
    Medir tiempo de ejecución de una función
    
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales
        **kwargs: Argumentos con nombre
        
    Returns:
        Tuple (resultado, tiempo_ejecución)
    """
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        return None, execution_time

def validate_dataframe_output(df, expected_cols: List[str] = None, min_rows: int = 1) -> bool:
    """
    Validar que un DataFrame tenga la estructura esperada
    
    Args:
        df: DataFrame a validar
        expected_cols: Columnas esperadas
        min_rows: Número mínimo de filas
        
    Returns:
        True si el DataFrame es válido
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        return False
    
    if len(df) < min_rows:
        return False
    
    if expected_cols:
        return all(col in df.columns for col in expected_cols)
    
    return True

def create_mock_config():
    """Crear configuración mock para tests"""
    return {
        'NARRATIVE_COLORS': {
            'defi': '#1f77b4',
            'gaming': '#ff7f0e', 
            'ai': '#2ca02c',
            'meme': '#d62728',
            'rwa': '#9467bd',
            'infrastructure': '#8c564b'
        },
        'QUALITY_THRESHOLDS': {
            'completeness': {'excellent': 95, 'good': 85, 'acceptable': 70},
            'duplicates': {'excellent': 1, 'good': 5, 'acceptable': 10}
        },
        'ANALYSIS_CONFIG': {
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'correlation_threshold': 0.7,
            'missing_threshold': 0.1,
            'volatility_window': 30,
            'return_periods': [1, 7, 30],
            'ma_windows': [7, 30],
            'bollinger_window': 20
        },
        'TECHNICAL_FEATURES': {
            'returns': {'periods': [1, 7, 30]},
            'moving_averages': {'windows': [7, 30]},
            'volatility': {'window': 30},
            'bollinger_bands': {'window': 20, 'std_dev': 2}
        }
    }

class TestLogger:
    """Logger simple para tests"""
    
    def __init__(self, log_file: str = "test_results.log"):
        self.log_file = log_file
        self.logs = []
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {level} - {message}"
        self.logs.append(log_entry)
        
        # Escribir al archivo
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception:
            pass  # Fallar silenciosamente si no se puede escribir
    
    def get_logs(self) -> List[str]:
        return self.logs.copy()

# Singleton logger global
test_logger = TestLogger()

def log_test_info(message: str):
    """Log información de test"""
    test_logger.log(message, "INFO")

def log_test_error(message: str):
    """Log error de test"""
    test_logger.log(message, "ERROR")

def log_test_warning(message: str):
    """Log warning de test"""
    test_logger.log(message, "WARNING")
