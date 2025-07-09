"""
🔧 Utilidades de Testing
========================

Funciones y clases de utilidad para el sistema de testing.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sys
import os

# Configurar path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestResult:
    """Clase para almacenar resultados de tests individuales"""
    
    def __init__(self, name: str, passed: bool, execution_time: float, 
                 details: str = "", error: Optional[Exception] = None):
        self.name = name
        self.passed = passed
        self.execution_time = execution_time
        self.details = details
        self.error = error
        self.timestamp = datetime.now()
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name} ({self.execution_time:.3f}s): {self.details}"

def create_test_data(n_observations: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Crear datos de prueba realistas para criptomonedas
    
    Args:
        n_observations: Número de observaciones
        seed: Semilla para reproducibilidad
        
    Returns:
        DataFrame con datos de prueba
    """
    np.random.seed(seed)
    
    # Tokens y narrativas realistas
    tokens = ['BTC', 'ETH', 'ADA', 'SOL', 'MATIC', 'LINK', 'DOT', 'UNI']
    narratives = ['defi', 'gaming', 'ai', 'meme', 'rwa', 'infrastructure']
    
    data = {
        'id': np.random.choice(tokens, n_observations),
        'symbol': [f'SYM{i%50}' for i in range(n_observations)],
        'name': [f'Token {i%100}' for i in range(n_observations)],
        'narrative': np.random.choice(narratives, n_observations),
        'close': np.random.lognormal(8, 1.5, n_observations),
        'market_cap': np.random.lognormal(25, 2, n_observations),
        'volume': np.random.lognormal(20, 1.8, n_observations),
        'date': pd.date_range('2020-01-01', periods=n_observations, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Agregar algunos valores problemáticos para testing robusto
    problem_indices = np.random.choice(df.index, min(20, n_observations//10), replace=False)
    df.loc[problem_indices[:5], 'market_cap'] = np.nan
    df.loc[problem_indices[5:8], 'volume'] = 0
    
    # Limpiar valores infinitos y negativos
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df['close'] > 0]
    
    return df

def create_empty_test_data() -> pd.DataFrame:
    """Crear DataFrame vacío para testing edge cases"""
    return pd.DataFrame()

def create_single_row_test_data() -> pd.DataFrame:
    """Crear DataFrame con una sola fila para testing"""
    return create_test_data(n_observations=1)

def create_large_test_data() -> pd.DataFrame:
    """Crear dataset grande para testing de performance"""
    return create_test_data(n_observations=10000)

def measure_execution_time(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    Medir tiempo de ejecución de una función
    
    Args:
        func: Función a ejecutar
        *args, **kwargs: Argumentos para la función
        
    Returns:
        Tuple con (resultado, tiempo_ejecución)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time

def safe_import(module_path: str, function_name: str):
    """
    Importar una función de forma segura
    
    Args:
        module_path: Ruta del módulo (ej: 'utils.data_analysis')
        function_name: Nombre de la función
        
    Returns:
        Función importada o None si falla
    """
    try:
        module = __import__(module_path, fromlist=[function_name])
        return getattr(module, function_name, None)
    except (ImportError, AttributeError):
        return None

def get_available_functions(module_path: str) -> List[str]:
    """
    Obtener lista de funciones disponibles en un módulo
    
    Args:
        module_path: Ruta del módulo
        
    Returns:
        Lista de nombres de funciones
    """
    try:
        module = __import__(module_path, fromlist=[''])
        functions = [name for name, obj in vars(module).items() 
                    if callable(obj) and not name.startswith('_')]
        return functions
    except ImportError:
        return []

def validate_dataframe_output(df: pd.DataFrame, expected_columns: List[str] = None) -> bool:
    """
    Validar que el output sea un DataFrame válido
    
    Args:
        df: DataFrame a validar
        expected_columns: Lista de columnas esperadas
        
    Returns:
        True si es válido, False si no
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            return False
    
    return True

def count_new_columns(df_original: pd.DataFrame, df_new: pd.DataFrame) -> int:
    """
    Contar cuántas columnas nuevas se agregaron
    
    Args:
        df_original: DataFrame original
        df_new: DataFrame después del procesamiento
        
    Returns:
        Número de columnas nuevas
    """
    if not isinstance(df_new, pd.DataFrame):
        return 0
    
    return len(df_new.columns) - len(df_original.columns)

def format_test_summary(passed: int, total: int, module_name: str) -> str:
    """
    Formatear resumen de tests para un módulo
    
    Args:
        passed: Número de tests pasados
        total: Número total de tests
        module_name: Nombre del módulo
        
    Returns:
        String formateado
    """
    success_rate = (passed / total * 100) if total > 0 else 0
    status_icon = "✅" if success_rate == 100 else "⚠️" if success_rate >= 80 else "❌"
    
    return f"   {status_icon} {module_name}: {passed}/{total} ({success_rate:.1f}%)"

def save_test_data_sample(df: pd.DataFrame, filename: str = "test_data_sample.csv"):
    """
    Guardar una muestra de los datos de prueba
    
    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo
    """
    filepath = os.path.join(os.path.dirname(__file__), 'fixtures', filename)
    df.head(10).to_csv(filepath, index=False)
    print(f"📁 Muestra de datos guardada en: {filepath}")

class TestDataGenerator:
    """Generador de diferentes tipos de datos de prueba"""
    
    @staticmethod
    def crypto_data(n_obs: int = 200) -> pd.DataFrame:
        """Generar datos de criptomonedas"""
        return create_test_data(n_obs)
    
    @staticmethod
    def edge_case_data() -> Dict[str, pd.DataFrame]:
        """Generar casos edge para testing"""
        return {
            'empty': create_empty_test_data(),
            'single_row': create_single_row_test_data(),
            'large': create_large_test_data()
        }
    
    @staticmethod
    def problematic_data() -> pd.DataFrame:
        """Generar datos con problemas comunes"""
        df = create_test_data(100)
        
        # Agregar problemas específicos
        df.loc[0:5, 'close'] = np.nan
        df.loc[10:12, 'volume'] = np.inf
        df.loc[20:22, 'market_cap'] = -1
        
        return df
