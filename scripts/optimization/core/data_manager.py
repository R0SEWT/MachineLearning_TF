"""
🚀 Gestor de Datos Centralizado - Fase 5
========================================

Sistema centralizado de gestión de datos que reemplaza la carga de datos
dispersa y inconsistente del sistema anterior.

Autor: Sistema de Optimización IA
Fecha: 2025-01-09 (Fase 5 - Organización)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import pickle
import hashlib
from datetime import datetime, timedelta
import warnings

# Import local modules
from .import_manager import safe_import
from .logging_setup import get_logger

# Safe imports de scikit-learn
sklearn_model_selection = safe_import("sklearn.model_selection")
sklearn_preprocessing = safe_import("sklearn.preprocessing")


@dataclass
class DataInfo:
    """Información sobre un dataset"""
    shape: Tuple[int, int]
    columns: List[str]
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    target_distribution: Optional[Dict[str, int]] = None
    memory_usage_mb: float = 0.0
    load_time: float = 0.0


class DataCache:
    """Sistema de cache para datos procesados"""
    
    def __init__(self, cache_dir: str = "./cache", max_size_mb: int = 1024):
        """
        Inicializar cache de datos.
        
        Args:
            cache_dir: Directorio de cache
            max_size_mb: Tamaño máximo de cache en MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.logger = get_logger("data_cache")
    
    def _get_cache_key(self, data_path: str, preprocessing_params: Dict[str, Any]) -> str:
        """Generar clave de cache basada en datos y parámetros"""
        # Incluir timestamp del archivo para invalidar cache si cambia
        file_mtime = Path(data_path).stat().st_mtime if Path(data_path).exists() else 0
        
        cache_string = f"{data_path}_{file_mtime}_{str(preprocessing_params)}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Tuple[pd.DataFrame, pd.Series, DataInfo]]:
        """Obtener datos del cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.logger.info(f"📥 Datos cargados desde cache: {cache_key[:8]}...")
                    return cached_data
            except Exception as e:
                self.logger.warning(f"Error cargando cache {cache_key}: {e}")
                cache_file.unlink(missing_ok=True)  # Eliminar cache corrupto
        
        return None
    
    def set(self, cache_key: str, features: pd.DataFrame, target: pd.Series, info: DataInfo):
        """Guardar datos en cache"""
        try:
            # Verificar espacio disponible
            if not self._has_space_for_cache():
                self._cleanup_old_cache()
            
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cached_data = (features, target, info)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            self.logger.info(f"📤 Datos guardados en cache: {cache_key[:8]}...")
            
        except Exception as e:
            self.logger.error(f"Error guardando en cache {cache_key}: {e}")
    
    def _has_space_for_cache(self) -> bool:
        """Verificar si hay espacio disponible en cache"""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        return (total_size / (1024 * 1024)) < self.max_size_mb
    
    def _cleanup_old_cache(self):
        """Limpiar cache antiguo"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        # Ordenar por fecha de modificación (más antiguos primero)
        cache_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Eliminar 25% de archivos más antiguos
        files_to_remove = len(cache_files) // 4
        for cache_file in cache_files[:files_to_remove]:
            cache_file.unlink(missing_ok=True)
            self.logger.info(f"🗑️  Cache eliminado: {cache_file.name}")


class DataManager:
    """
    Gestor centralizado de datos para el sistema de optimización.
    
    Proporciona carga, preprocesamiento y gestión de datos unificada.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar gestor de datos.
        
        Args:
            config: Configuración del gestor de datos
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger("data_manager")
        self.cache = DataCache(
            cache_dir=self.config.get("cache_dir", "./cache"),
            max_size_mb=self.config.get("max_cache_size_mb", 1024)
        )
        self._data_info: Optional[DataInfo] = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuración por defecto"""
        return {
            "cache_enabled": True,
            "cache_dir": "./cache",
            "max_cache_size_mb": 1024,
            "target_column": "target_next_close_positive",
            "exclude_columns": ['id', 'date', 'name', 'symbol', 'cmc_id', 'price'],
            "handle_missing": "drop",
            "normalize_features": False,
            "feature_selection": False,
            "min_samples": 1000
        }
    
    def load_data(self, data_path: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series, DataInfo]:
        """
        Cargar datos con preprocesamiento automático.
        
        Args:
            data_path: Ruta al archivo de datos
            **kwargs: Parámetros adicionales de preprocesamiento
            
        Returns:
            Tuple de (features, target, info)
        """
        start_time = datetime.now()
        
        # Combinar configuración con parámetros pasados
        preprocessing_params = {**self.config, **kwargs}
        
        # Verificar cache si está habilitado
        if self.config.get("cache_enabled", True):
            cache_key = self.cache._get_cache_key(data_path, preprocessing_params)
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                features, target, info = cached_data
                self._data_info = info
                return features, target, info
        
        # Cargar datos desde archivo
        self.logger.info(f"📊 Cargando datos desde: {data_path}")
        
        try:
            # Detectar formato y cargar
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            elif data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"Formato de archivo no soportado: {data_path}")
            
            self.logger.info(f"📋 Datos cargados: {df.shape[0]:,} filas, {df.shape[1]:,} columnas")
            
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            raise
        
        # Preprocesar datos
        features, target, info = self._preprocess_data(df, preprocessing_params)
        
        # Calcular tiempo de carga
        load_time = (datetime.now() - start_time).total_seconds()
        info.load_time = load_time
        
        # Guardar en cache si está habilitado
        if self.config.get("cache_enabled", True):
            self.cache.set(cache_key, features, target, info)
        
        self._data_info = info
        self.logger.info(f"✅ Datos procesados en {load_time:.2f}s")
        
        return features, target, info
    
    def _preprocess_data(self, df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, DataInfo]:
        """
        Preprocesar datos según configuración.
        
        Args:
            df: DataFrame original
            params: Parámetros de preprocesamiento
            
        Returns:
            Tuple de (features, target, info)
        """
        self.logger.info("🔧 Iniciando preprocesamiento de datos")
        
        # Información inicial
        initial_shape = df.shape
        missing_before = df.isnull().sum().to_dict()
        
        # Extraer target
        target_column = params.get("target_column", "target_next_close_positive")
        if target_column not in df.columns:
            available_targets = [col for col in df.columns if 'target' in col.lower()]
            if available_targets:
                target_column = available_targets[0]
                self.logger.warning(f"Target column no encontrada, usando: {target_column}")
            else:
                raise ValueError(f"Target column '{target_column}' no encontrada")
        
        target = df[target_column].copy()
        
        # Excluir columnas
        exclude_columns = params.get("exclude_columns", [])
        exclude_columns.append(target_column)  # Excluir target de features
        
        features = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        
        self.logger.info(f"📊 Features: {features.shape[1]:,} columnas")
        self.logger.info(f"🎯 Target: {target_column}")
        
        # Manejo de valores faltantes
        features, target = self._handle_missing_values(features, target, params)
        
        # Filtrar por mínimo de muestras
        min_samples = params.get("min_samples", 1000)
        if len(features) < min_samples:
            self.logger.warning(f"Pocas muestras disponibles: {len(features)} < {min_samples}")
        
        # Selección de features (opcional)
        if params.get("feature_selection", False):
            features = self._select_features(features, target, params)
        
        # Normalización (opcional)
        if params.get("normalize_features", False):
            features = self._normalize_features(features)
        
        # Crear información de datos
        info = self._create_data_info(features, target, initial_shape)
        
        self.logger.info(f"✅ Preprocesamiento completado: {features.shape[0]:,} x {features.shape[1]:,}")
        
        return features, target, info
    
    def _handle_missing_values(self, features: pd.DataFrame, target: pd.Series, 
                              params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Manejo de valores faltantes"""
        strategy = params.get("handle_missing", "drop")
        
        if strategy == "drop":
            # Eliminar filas con valores faltantes
            valid_indices = features.dropna().index.intersection(target.dropna().index)
            features = features.loc[valid_indices]
            target = target.loc[valid_indices]
            
            self.logger.info(f"🗑️  Eliminadas {len(features.index) - len(valid_indices):,} filas con valores faltantes")
            
        elif strategy == "fill":
            # Rellenar valores faltantes
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            categorical_columns = features.select_dtypes(include=['object']).columns
            
            # Rellenar numéricos con mediana
            features[numeric_columns] = features[numeric_columns].fillna(features[numeric_columns].median())
            
            # Rellenar categóricos con moda
            for col in categorical_columns:
                features[col] = features[col].fillna(features[col].mode().iloc[0] if len(features[col].mode()) > 0 else "unknown")
            
            # Rellenar target con moda
            if target.isnull().any():
                target = target.fillna(target.mode().iloc[0] if len(target.mode()) > 0 else 0)
            
            self.logger.info(f"🔧 Valores faltantes rellenados")
        
        return features, target
    
    def _select_features(self, features: pd.DataFrame, target: pd.Series, 
                        params: Dict[str, Any]) -> pd.DataFrame:
        """Selección automática de features"""
        if sklearn_model_selection is None:
            self.logger.warning("Scikit-learn no disponible, saltando selección de features")
            return features
        
        try:
            from sklearn.feature_selection import SelectKBest, f_classif
            
            # Seleccionar solo columnas numéricas
            numeric_features = features.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) == 0:
                self.logger.warning("No hay features numéricas para selección")
                return features
            
            # Seleccionar top K features
            k = min(params.get("max_features", 50), len(numeric_features.columns))
            selector = SelectKBest(score_func=f_classif, k=k)
            
            selected_features = selector.fit_transform(numeric_features, target)
            selected_columns = numeric_features.columns[selector.get_support()]
            
            # Agregar features categóricas de vuelta
            categorical_features = features.select_dtypes(include=['object'])
            if len(categorical_features.columns) > 0:
                features_selected = pd.concat([
                    pd.DataFrame(selected_features, columns=selected_columns, index=features.index),
                    categorical_features
                ], axis=1)
            else:
                features_selected = pd.DataFrame(selected_features, columns=selected_columns, index=features.index)
            
            self.logger.info(f"🎯 Seleccionadas {len(selected_columns):,} features numéricas")
            return features_selected
            
        except Exception as e:
            self.logger.warning(f"Error en selección de features: {e}")
            return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalizar features numéricas"""
        if sklearn_preprocessing is None:
            self.logger.warning("Scikit-learn no disponible, saltando normalización")
            return features
        
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Normalizar solo columnas numéricas
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return features
            
            scaler = StandardScaler()
            features_normalized = features.copy()
            features_normalized[numeric_columns] = scaler.fit_transform(features[numeric_columns])
            
            self.logger.info(f"📏 Normalizadas {len(numeric_columns):,} features numéricas")
            return features_normalized
            
        except Exception as e:
            self.logger.warning(f"Error en normalización: {e}")
            return features
    
    def _create_data_info(self, features: pd.DataFrame, target: pd.Series, 
                         initial_shape: Tuple[int, int]) -> DataInfo:
        """Crear información detallada de los datos"""
        # Distribución del target
        if target.dtype in ['int64', 'int32', 'bool']:
            target_dist = target.value_counts().to_dict()
        else:
            # Para targets continuos, crear bins
            target_dist = {"continuous": len(target)}
        
        # Uso de memoria
        memory_usage = (features.memory_usage(deep=True).sum() + target.memory_usage(deep=True)) / (1024 * 1024)
        
        return DataInfo(
            shape=features.shape,
            columns=features.columns.tolist(),
            missing_values=features.isnull().sum().to_dict(),
            data_types=features.dtypes.astype(str).to_dict(),
            target_distribution=target_dist,
            memory_usage_mb=memory_usage
        )
    
    def get_train_val_test_split(self, features: pd.DataFrame, target: pd.Series,
                                test_size: float = 0.2, val_size: float = 0.15,
                                random_state: int = 42) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        División temporal de datos para ML.
        
        Args:
            features: Features del dataset
            target: Variable objetivo
            test_size: Proporción para test
            val_size: Proporción para validación
            random_state: Semilla aleatoria
            
        Returns:
            Diccionario con splits de datos
        """
        if sklearn_model_selection is None:
            raise ImportError("Scikit-learn requerido para split de datos")
        
        from sklearn.model_selection import train_test_split
        
        # Split train/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state, stratify=target
        )
        
        # Split train/val del conjunto temporal
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        self.logger.info(f"📊 Split de datos:")
        self.logger.info(f"  - Train: {X_train.shape[0]:,} muestras")
        self.logger.info(f"  - Val:   {X_val.shape[0]:,} muestras")
        self.logger.info(f"  - Test:  {X_test.shape[0]:,} muestras")
        
        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }
    
    def get_data_info(self) -> Optional[DataInfo]:
        """Obtener información de los datos cargados"""
        return self._data_info
    
    def clear_cache(self):
        """Limpiar cache de datos"""
        cache_files = list(self.cache.cache_dir.glob("*.pkl"))
        for cache_file in cache_files:
            cache_file.unlink()
        
        self.logger.info(f"🗑️  Cache limpiado: {len(cache_files)} archivos eliminados")


# ==================== FUNCIONES DE CONVENIENCIA ====================

def load_crypto_data(data_path: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series, DataInfo]:
    """
    Función de conveniencia para cargar datos de crypto.
    
    Args:
        data_path: Ruta a los datos
        **kwargs: Parámetros adicionales
        
    Returns:
        Tuple de (features, target, info)
    """
    manager = DataManager()
    return manager.load_data(data_path, **kwargs)

def get_train_test_split(features: pd.DataFrame, target: pd.Series, 
                        **kwargs) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Función de conveniencia para split de datos.
    
    Args:
        features: Features del dataset
        target: Variable objetivo
        **kwargs: Parámetros de split
        
    Returns:
        Diccionario con splits de datos
    """
    manager = DataManager()
    return manager.get_train_val_test_split(features, target, **kwargs)


if __name__ == "__main__":
    # Demo del gestor de datos
    print("🚀 Gestor de Datos Centralizado - Fase 5")
    print("=======================================")
    
    # Configurar logging
    from .logging_setup import setup_logging
    setup_logging({"level": "INFO"})
    
    # Ejemplo de uso
    data_path = "../../../data/crypto_ohlc_join.csv"
    
    if Path(data_path).exists():
        print(f"📊 Cargando datos de ejemplo: {data_path}")
        
        try:
            # Cargar datos
            manager = DataManager()
            features, target, info = manager.load_data(data_path)
            
            print(f"✅ Datos cargados exitosamente:")
            print(f"  - Shape: {info.shape}")
            print(f"  - Memoria: {info.memory_usage_mb:.2f} MB")
            print(f"  - Tiempo de carga: {info.load_time:.2f}s")
            print(f"  - Distribución target: {info.target_distribution}")
            
            # Split de datos
            splits = manager.get_train_val_test_split(features, target)
            print(f"📊 Split completado")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"⚠️  Archivo de datos no encontrado: {data_path}")
        print("   Creando ejemplo con datos sintéticos...")
        
        # Crear datos sintéticos para demo
        np.random.seed(42)
        n_samples, n_features = 1000, 20
        
        features_synthetic = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        target_synthetic = pd.Series(np.random.randint(0, 2, n_samples))
        
        print(f"✅ Datos sintéticos creados: {features_synthetic.shape}")
        
        # Test split
        manager = DataManager()
        splits = manager.get_train_val_test_split(features_synthetic, target_synthetic)
        print(f"📊 Split de datos sintéticos completado")
