#!/usr/bin/env python3
"""
Validación cruzada temporal avanzada para datos de series temporales
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Iterator, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesValidationConfig:
    """Configuración para validación cruzada temporal"""
    
    # Configuración TimeSeriesSplit
    n_splits: int = 5
    max_train_size: Optional[int] = None
    test_size: Optional[int] = None
    gap: int = 0  # Gap entre train y test para evitar leakage
    
    # Configuración Walk-Forward
    walk_forward_window: int = 30  # Días de ventana
    walk_forward_step: int = 7     # Días de paso
    min_train_size: int = 100      # Mínimo tamaño de entrenamiento
    
    # Configuración Purged CV
    purged_window: int = 3         # Días de purga
    embargo_window: int = 1        # Días de embargo
    
    # Configuración de estabilidad
    stability_checks: bool = True
    min_samples_per_fold: int = 50
    max_variance_ratio: float = 0.3  # Máxima varianza entre folds

class TimeSeriesValidator:
    """Validador avanzado para datos de series temporales"""
    
    def __init__(self, config: TimeSeriesValidationConfig = None):
        self.config = config or TimeSeriesValidationConfig()
        self.validation_results = {}
    
    def get_time_series_splits(self, X: pd.DataFrame, y: pd.Series, 
                              date_column: str = 'date') -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera splits de validación cruzada temporal
        """
        if date_column not in X.columns:
            raise ValueError(f"Columna de fecha '{date_column}' no encontrada")
        
        # Asegurar que los datos estén ordenados por fecha
        sort_idx = X[date_column].argsort()
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]
        
        n_samples = len(X_sorted)
        n_splits = self.config.n_splits
        
        # Calcular tamaños de splits
        if self.config.test_size:
            test_size = self.config.test_size
        else:
            test_size = n_samples // (n_splits + 1)
        
        if self.config.max_train_size:
            max_train_size = self.config.max_train_size
        else:
            max_train_size = n_samples
        
        for i in range(n_splits):
            # Calcular índices de test
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            # Calcular índices de train con gap
            train_end = test_start - self.config.gap
            train_start = max(0, train_end - max_train_size)
            
            if train_end - train_start < self.config.min_train_size:
                continue
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))
            
            yield train_idx, test_idx
    
    def get_walk_forward_splits(self, X: pd.DataFrame, y: pd.Series,
                               date_column: str = 'date') -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera splits de validación walk-forward
        """
        if date_column not in X.columns:
            raise ValueError(f"Columna de fecha '{date_column}' no encontrada")
        
        # Convertir fechas
        dates = pd.to_datetime(X[date_column])
        min_date = dates.min()
        max_date = dates.max()
        
        current_date = min_date + timedelta(days=self.config.min_train_size)
        
        while current_date + timedelta(days=self.config.walk_forward_window) <= max_date:
            # Definir ventana de entrenamiento
            train_end_date = current_date
            train_start_date = train_end_date - timedelta(days=self.config.walk_forward_window)
            
            # Definir ventana de test
            test_start_date = current_date + timedelta(days=1)
            test_end_date = test_start_date + timedelta(days=self.config.walk_forward_step)
            
            # Obtener índices
            train_mask = (dates >= train_start_date) & (dates <= train_end_date)
            test_mask = (dates >= test_start_date) & (dates < test_end_date)
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            if len(train_idx) >= self.config.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx
            
            # Avanzar ventana
            current_date += timedelta(days=self.config.walk_forward_step)
    
    def get_purged_cv_splits(self, X: pd.DataFrame, y: pd.Series,
                            date_column: str = 'date') -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera splits con purga para evitar leakage temporal
        """
        if date_column not in X.columns:
            raise ValueError(f"Columna de fecha '{date_column}' no encontrada")
        
        dates = pd.to_datetime(X[date_column])
        n_samples = len(X)
        n_splits = self.config.n_splits
        
        # Calcular tamaño de cada fold
        fold_size = n_samples // n_splits
        
        for i in range(n_splits):
            # Test set
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            
            # Purge window (remover samples cercanos al test set)
            purge_start = max(0, test_start - self.config.purged_window)
            purge_end = min(n_samples, test_end + self.config.purged_window)
            
            # Train set (todo excepto test y purge)
            train_idx = np.concatenate([
                np.arange(0, purge_start),
                np.arange(purge_end, n_samples)
            ])
            
            test_idx = np.arange(test_start, test_end)
            
            # Embargo (remover samples muy recientes del train)
            if self.config.embargo_window > 0:
                embargo_cutoff = test_start - self.config.embargo_window
                train_idx = train_idx[train_idx < embargo_cutoff]
            
            if len(train_idx) >= self.config.min_train_size:
                yield train_idx, test_idx
    
    def calculate_stability_metrics(self, cv_scores: List[float]) -> Dict[str, float]:
        """
        Calcular métricas de estabilidad de CV
        """
        if len(cv_scores) < 2:
            return {'stability_score': 0.0, 'cv_std': 0.0, 'cv_var': 0.0}
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_var = np.var(cv_scores)
        
        # Coeficiente de variación
        cv_coefficient = cv_std / cv_mean if cv_mean > 0 else float('inf')
        
        # Score de estabilidad (inverso del coeficiente de variación)
        stability_score = 1 / (1 + cv_coefficient)
        
        return {
            'stability_score': stability_score,
            'cv_std': cv_std,
            'cv_var': cv_var,
            'cv_coefficient': cv_coefficient,
            'cv_mean': cv_mean
        }
    
    def validate_temporal_consistency(self, X: pd.DataFrame, y: pd.Series,
                                    date_column: str = 'date') -> Dict[str, Any]:
        """
        Validar consistencia temporal de los datos
        """
        dates = pd.to_datetime(X[date_column])
        
        # Verificar orden temporal
        is_sorted = dates.is_monotonic_increasing
        
        # Detectar gaps temporales
        date_diffs = dates.diff().dt.days
        gaps = date_diffs[date_diffs > 7]  # Gaps > 7 días
        
        # Verificar distribución temporal de clases
        df_temp = pd.DataFrame({'date': dates, 'target': y})
        df_temp['year_month'] = df_temp['date'].dt.to_period('M')
        
        temporal_distribution = df_temp.groupby('year_month')['target'].agg([
            'count', 'mean', 'std'
        ]).fillna(0)
        
        # Detectar drift temporal en la distribución de target
        monthly_means = temporal_distribution['mean']
        temporal_variance = monthly_means.var()
        
        return {
            'is_temporally_sorted': is_sorted,
            'n_temporal_gaps': len(gaps),
            'max_gap_days': gaps.max() if len(gaps) > 0 else 0,
            'temporal_target_variance': temporal_variance,
            'temporal_distribution': temporal_distribution.to_dict(),
            'date_range': {
                'start': dates.min(),
                'end': dates.max(),
                'total_days': (dates.max() - dates.min()).days
            }
        }
    
    def perform_time_series_cv(self, estimator, X: pd.DataFrame, y: pd.Series,
                              scoring: str = 'roc_auc',
                              cv_type: str = 'time_series',
                              date_column: str = 'date') -> Dict[str, Any]:
        """
        Realizar validación cruzada temporal completa
        """
        try:
            from sklearn.metrics import get_scorer
            scorer = get_scorer(scoring)
        except ImportError:
            # Fallback manual
            if scoring == 'roc_auc':
                from sklearn.metrics import roc_auc_score
                scorer = lambda est, X, y: roc_auc_score(y, est.predict_proba(X)[:, 1])
            else:
                raise ValueError(f"Scoring '{scoring}' no soportado sin sklearn")
        
        # Seleccionar tipo de CV
        if cv_type == 'time_series':
            cv_splits = self.get_time_series_splits(X, y, date_column)
        elif cv_type == 'walk_forward':
            cv_splits = self.get_walk_forward_splits(X, y, date_column)
        elif cv_type == 'purged':
            cv_splits = self.get_purged_cv_splits(X, y, date_column)
        else:
            raise ValueError(f"Tipo de CV '{cv_type}' no soportado")
        
        scores = []
        fold_info = []
        
        # Preparar features (remover columna de fecha)
        feature_columns = [col for col in X.columns if col != date_column]
        X_features = X[feature_columns]
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            try:
                # Dividir datos
                X_train_fold = X_features.iloc[train_idx]
                X_test_fold = X_features.iloc[test_idx]
                y_train_fold = y.iloc[train_idx]
                y_test_fold = y.iloc[test_idx]
                
                # Entrenar modelo
                estimator.fit(X_train_fold, y_train_fold)
                
                # Evaluar
                score = scorer(estimator, X_test_fold, y_test_fold)
                scores.append(score)
                
                # Información del fold
                fold_info.append({
                    'fold': fold_idx,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'score': score,
                    'train_period': {
                        'start': X.iloc[train_idx][date_column].min(),
                        'end': X.iloc[train_idx][date_column].max()
                    },
                    'test_period': {
                        'start': X.iloc[test_idx][date_column].min(),
                        'end': X.iloc[test_idx][date_column].max()
                    }
                })
                
                logger.info(f"Fold {fold_idx}: Score={score:.4f}, Train={len(train_idx)}, Test={len(test_idx)}")
                
            except Exception as e:
                logger.warning(f"Error en fold {fold_idx}: {e}")
                continue
        
        if not scores:
            raise ValueError("No se pudo completar ningún fold de CV")
        
        # Calcular métricas de estabilidad
        stability_metrics = self.calculate_stability_metrics(scores)
        
        # Validar consistencia temporal
        temporal_validation = self.validate_temporal_consistency(X, y, date_column)
        
        results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'fold_info': fold_info,
            'n_folds': len(scores),
            'cv_type': cv_type,
            'stability_metrics': stability_metrics,
            'temporal_validation': temporal_validation
        }
        
        # Verificar estabilidad
        if self.config.stability_checks:
            self._check_stability_warnings(results)
        
        return results
    
    def _check_stability_warnings(self, results: Dict[str, Any]):
        """Verificar y emitir warnings sobre estabilidad"""
        
        # Verificar varianza entre folds
        cv_coefficient = results['stability_metrics']['cv_coefficient']
        if cv_coefficient > self.config.max_variance_ratio:
            logger.warning(f"Alta varianza entre folds: CV={cv_coefficient:.3f}")
        
        # Verificar tamaño mínimo de folds
        min_fold_size = min(fold['test_size'] for fold in results['fold_info'])
        if min_fold_size < self.config.min_samples_per_fold:
            logger.warning(f"Fold muy pequeño: {min_fold_size} < {self.config.min_samples_per_fold}")
        
        # Verificar gaps temporales
        temporal_val = results['temporal_validation']
        if temporal_val['n_temporal_gaps'] > 0:
            logger.warning(f"Gaps temporales detectados: {temporal_val['n_temporal_gaps']}")
        
        # Verificar drift temporal
        if temporal_val['temporal_target_variance'] > 0.1:
            logger.warning(f"Posible drift temporal: varianza={temporal_val['temporal_target_variance']:.3f}")

class TimeSeriesFeatureValidator:
    """Validador específico para features en series temporales"""
    
    @staticmethod
    def detect_look_ahead_bias(X: pd.DataFrame, date_column: str = 'date') -> List[str]:
        """
        Detectar variables que podrían tener look-ahead bias
        """
        suspicious_features = []
        
        # Buscar nombres de columnas sospechosos
        lookhead_patterns = ['future_', 'next_', 'forward_', 'ahead_', 'tomorrow_']
        
        for col in X.columns:
            if col == date_column:
                continue
                
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in lookhead_patterns):
                suspicious_features.append(col)
        
        return suspicious_features
    
    @staticmethod
    def validate_feature_timing(X: pd.DataFrame, y: pd.Series, 
                               date_column: str = 'date') -> Dict[str, Any]:
        """
        Validar el timing de las features vs target
        """
        dates = pd.to_datetime(X[date_column])
        
        # Verificar que las features no sean del futuro
        validation_results = {
            'look_ahead_features': TimeSeriesFeatureValidator.detect_look_ahead_bias(X, date_column),
            'temporal_alignment': True,  # Placeholder para verificaciones más complejas
            'feature_lag_analysis': {}   # Análisis de lag de features
        }
        
        return validation_results

# Instancia global del validador temporal
TEMPORAL_VALIDATOR = TimeSeriesValidator()
