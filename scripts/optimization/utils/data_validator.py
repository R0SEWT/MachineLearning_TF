#!/usr/bin/env python3
"""
Validador de datos robusto para el optimizador de hiperparámetros
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Excepción personalizada para errores de validación de datos"""
    pass

class DataValidator:
    """Validador robusto de datos para optimización"""
    
    def __init__(self, config: Any = None):
        self.config = config
        self.validation_results = {}
    
    def validate_file_path(self, file_path: str) -> bool:
        """Validar que el archivo existe y es accesible"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise DataValidationError(f"Archivo no encontrado: {file_path}")
            
            if not path.is_file():
                raise DataValidationError(f"La ruta no es un archivo: {file_path}")
            
            if not os.access(file_path, os.R_OK):
                raise DataValidationError(f"Sin permisos de lectura: {file_path}")
            
            # Verificar tamaño del archivo
            file_size = path.stat().st_size
            if file_size == 0:
                raise DataValidationError(f"Archivo vacío: {file_path}")
            
            logger.info(f"Archivo validado: {file_path} ({file_size / 1e6:.1f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Error validando archivo {file_path}: {e}")
            raise DataValidationError(f"Error validando archivo: {e}")
    
    def validate_dataframe_structure(self, df: pd.DataFrame, 
                                   required_columns: List[str] = None) -> Dict[str, Any]:
        """Validar estructura del DataFrame"""
        validation_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Verificar que no esté vacío
        if df.empty:
            raise DataValidationError("DataFrame está vacío")
        
        # Verificar columnas requeridas
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise DataValidationError(f"Columnas faltantes: {missing_cols}")
        
        # Verificar valores faltantes excesivos
        missing_pct = (df.isnull().sum() / len(df)) * 100
        problematic_cols = missing_pct[missing_pct > 80].index.tolist()
        if problematic_cols:
            logger.warning(f"Columnas con >80% valores faltantes: {problematic_cols}")
        
        # Verificar duplicados
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Registros duplicados encontrados: {duplicates}")
            validation_info['duplicates'] = duplicates
        
        logger.info(f"DataFrame validado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return validation_info
    
    def validate_target_variable(self, y: pd.Series, min_class_size: int = 50) -> Dict[str, Any]:
        """Validar variable objetivo para clasificación"""
        target_info = {
            'length': len(y),
            'unique_values': y.nunique(),
            'value_counts': y.value_counts().to_dict(),
            'missing_values': y.isnull().sum()
        }
        
        # Verificar valores faltantes en target
        if y.isnull().sum() > 0:
            raise DataValidationError(f"Variable objetivo tiene {y.isnull().sum()} valores faltantes")
        
        # Verificar que sea clasificación binaria
        unique_vals = y.nunique()
        if unique_vals != 2:
            raise DataValidationError(f"Se esperan 2 clases, encontradas: {unique_vals}")
        
        # Verificar balance de clases
        value_counts = y.value_counts()
        min_class_count = value_counts.min()
        max_class_count = value_counts.max()
        
        if min_class_count < min_class_size:
            raise DataValidationError(f"Clase minoritaria tiene solo {min_class_count} muestras (mínimo: {min_class_size})")
        
        # Calcular desbalance
        imbalance_ratio = max_class_count / min_class_count
        target_info['imbalance_ratio'] = imbalance_ratio
        
        if imbalance_ratio > 10:
            logger.warning(f"Clases muy desbalanceadas (ratio: {imbalance_ratio:.2f})")
        
        logger.info(f"Variable objetivo validada: {value_counts.to_dict()}")
        return target_info
    
    def validate_features(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Validar matriz de features"""
        feature_info = {
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'feature_types': {},
            'problematic_features': []
        }
        
        for col in X.columns:
            col_info = self._analyze_feature(X[col])
            feature_info['feature_types'][col] = col_info
            
            # Identificar features problemáticas
            if col_info['problematic']:
                feature_info['problematic_features'].append(col)
        
        # Verificar multicolinealidad básica
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                corr_matrix = X[numeric_cols].corr().abs()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.95:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                
                if high_corr_pairs:
                    logger.warning(f"Features altamente correlacionadas: {len(high_corr_pairs)} pares")
                    feature_info['high_correlation_pairs'] = high_corr_pairs[:10]  # Top 10
            except Exception as e:
                logger.warning(f"Error calculando correlaciones: {e}")
        
        logger.info(f"Features validadas: {X.shape[1]} features, {len(feature_info['problematic_features'])} problemáticas")
        return feature_info
    
    def _analyze_feature(self, series: pd.Series) -> Dict[str, Any]:
        """Analizar una feature individual"""
        info = {
            'dtype': str(series.dtype),
            'missing_pct': (series.isnull().sum() / len(series)) * 100,
            'unique_values': series.nunique(),
            'problematic': False,
            'issues': []
        }
        
        # Verificar valores infinitos
        if pd.api.types.is_numeric_dtype(series):
            inf_count = np.isinf(series).sum()
            if inf_count > 0:
                info['issues'].append(f"Valores infinitos: {inf_count}")
                info['problematic'] = True
            
            # Verificar varianza
            if series.var() == 0:
                info['issues'].append("Varianza cero")
                info['problematic'] = True
            
            # Verificar outliers extremos
            if series.nunique() > 10:  # Solo para variables continuas
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((series < (q1 - 3 * iqr)) | (series > (q3 + 3 * iqr))).sum()
                outlier_pct = (outliers / len(series)) * 100
                if outlier_pct > 10:
                    info['issues'].append(f"Outliers extremos: {outlier_pct:.1f}%")
        
        # Verificar valores faltantes excesivos
        if info['missing_pct'] > 50:
            info['issues'].append(f"Muchos valores faltantes: {info['missing_pct']:.1f}%")
            info['problematic'] = True
        
        return info
    
    def validate_data_splits(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                           X_test: pd.DataFrame, y_train: pd.Series, 
                           y_val: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Validar splits de datos"""
        split_info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_ratio': len(X_train) / (len(X_train) + len(X_val) + len(X_test)),
            'val_ratio': len(X_val) / (len(X_train) + len(X_val) + len(X_test)),
            'test_ratio': len(X_test) / (len(X_train) + len(X_val) + len(X_test))
        }
        
        # Verificar que las features coincidan
        if not (X_train.columns.equals(X_val.columns) and X_train.columns.equals(X_test.columns)):
            raise DataValidationError("Las columnas no coinciden entre splits")
        
        # Verificar distribución de clases en cada split
        for split_name, y_split in [('train', y_train), ('val', y_val), ('test', y_test)]:
            class_dist = y_split.value_counts(normalize=True)
            split_info[f'{split_name}_class_distribution'] = class_dist.to_dict()
        
        # Verificar que no haya leakage temporal (si hay columna de fecha)
        if 'date' in X_train.columns:
            try:
                max_train_date = pd.to_datetime(X_train['date']).max()
                min_val_date = pd.to_datetime(X_val['date']).min()
                min_test_date = pd.to_datetime(X_test['date']).min()
                
                if min_val_date <= max_train_date:
                    logger.warning("Posible leakage temporal entre train y validation")
                if min_test_date <= max_train_date:
                    logger.warning("Posible leakage temporal entre train y test")
            except Exception as e:
                logger.warning(f"Error verificando leakage temporal: {e}")
        
        logger.info(f"Splits validados - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return split_info
    
    def validate_memory_requirements(self, X: pd.DataFrame, 
                                   safety_factor: float = 2.0) -> Dict[str, Any]:
        """Validar requerimientos de memoria"""
        try:
            import psutil
            
            # Memoria del DataFrame
            df_memory = X.memory_usage(deep=True).sum()
            
            # Memoria del sistema
            system_memory = psutil.virtual_memory()
            available_memory = system_memory.available
            
            # Estimación de memoria requerida (considerando modelos y CV)
            estimated_memory_needed = df_memory * safety_factor
            
            memory_info = {
                'dataframe_memory_mb': df_memory / 1e6,
                'available_memory_mb': available_memory / 1e6,
                'estimated_needed_mb': estimated_memory_needed / 1e6,
                'memory_sufficient': estimated_memory_needed < available_memory,
                'memory_usage_pct': (estimated_memory_needed / available_memory) * 100
            }
            
            if not memory_info['memory_sufficient']:
                logger.warning(f"Memoria insuficiente: necesita {estimated_memory_needed/1e6:.1f}MB, disponible {available_memory/1e6:.1f}MB")
            
            return memory_info
            
        except ImportError:
            logger.warning("psutil no disponible para verificación de memoria")
            return {'memory_check': 'unavailable'}
    
    def run_full_validation(self, data_path: str, target_column: str,
                           exclude_columns: List[str] = None) -> Dict[str, Any]:
        """Ejecutar validación completa"""
        logger.info("Iniciando validación completa de datos")
        
        validation_results = {}
        
        try:
            # 1. Validar archivo
            self.validate_file_path(data_path)
            validation_results['file_validation'] = True
            
            # 2. Cargar y validar DataFrame
            df = pd.read_csv(data_path)
            validation_results['dataframe_info'] = self.validate_dataframe_structure(df)
            
            # 3. Validar variable objetivo
            if target_column in df.columns:
                validation_results['target_info'] = self.validate_target_variable(df[target_column])
            else:
                raise DataValidationError(f"Columna objetivo '{target_column}' no encontrada")
            
            # 4. Preparar features
            if exclude_columns:
                feature_columns = [col for col in df.columns if col not in exclude_columns]
            else:
                feature_columns = [col for col in df.columns if col != target_column]
            
            X = df[feature_columns]
            validation_results['feature_info'] = self.validate_features(X)
            
            # 5. Validar memoria
            validation_results['memory_info'] = self.validate_memory_requirements(X)
            
            logger.info("Validación completa exitosa")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error en validación completa: {e}")
            raise DataValidationError(f"Validación fallida: {e}")
