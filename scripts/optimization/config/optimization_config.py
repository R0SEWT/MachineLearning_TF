#!/usr/bin/env python3
"""
Configuración centralizada para el optimizador de hiperparámetros
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import os

@dataclass
class OptimizationConfig:
    """Configuración principal para optimización"""
    
    # Configuración de datos
    data_path: str = "/home/exodia/Documentos/MachineLearning_TF/data/crypto_ohlc_join.csv"
    results_path: str = "../../optimization_results"
    target_period: int = 30
    min_market_cap: float = 0
    max_market_cap: float = 10_000_000
    
    # Configuración de validación cruzada
    cv_folds: int = 3
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    
    # Configuración de optimización
    default_n_trials: int = 100
    default_timeout_per_model: int = 1800  # 30 minutos
    max_trials_no_improvement: int = 20
    
    # Configuración de GPU
    prefer_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    fallback_to_cpu: bool = True
    
    # Configuración de logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Métricas a calcular
    primary_metric: str = "roc_auc"
    secondary_metrics: List[str] = None
    
    # Columnas a excluir en feature engineering
    exclude_columns: List[str] = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        if self.secondary_metrics is None:
            self.secondary_metrics = [
                "precision", "recall", "f1", "accuracy",
                "sharpe_ratio", "max_drawdown", "stability_score"
            ]
        
        if self.exclude_columns is None:
            self.exclude_columns = [
                'id', 'date', 'name', 'symbol', 'cmc_id', 'price'
            ]
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Obtener configuración de GPU"""
        return {
            "prefer_gpu": self.prefer_gpu,
            "memory_fraction": self.gpu_memory_fraction,
            "fallback_to_cpu": self.fallback_to_cpu
        }
    
    def get_data_split_config(self) -> Dict[str, float]:
        """Obtener configuración de split de datos"""
        train_size = 1.0 - self.test_size - self.val_size
        return {
            "train_size": train_size,
            "val_size": self.val_size,
            "test_size": self.test_size
        }

@dataclass
class ModelConfig:
    """Configuración específica por modelo"""
    
    # Rangos de hiperparámetros para XGBoost
    xgboost_params = {
        'n_estimators': {'type': 'int', 'low': 100, 'high': 1000, 'step': 50},
        'max_depth': {'type': 'int', 'low': 3, 'high': 12},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
        'reg_lambda': {'type': 'float', 'low': 0, 'high': 10},
        'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
        'gamma': {'type': 'float', 'low': 0, 'high': 5}
    }
    
    # Rangos de hiperparámetros para LightGBM
    lightgbm_params = {
        'n_estimators': {'type': 'int', 'low': 100, 'high': 1000, 'step': 50},
        'max_depth': {'type': 'int', 'low': 3, 'high': 12},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
        'reg_lambda': {'type': 'float', 'low': 0, 'high': 10},
        'min_child_samples': {'type': 'int', 'low': 5, 'high': 100},
        'num_leaves': {'type': 'int', 'low': 10, 'high': 300}
    }
    
    # Rangos de hiperparámetros para CatBoost
    catboost_params = {
        'iterations': {'type': 'int', 'low': 100, 'high': 1000, 'step': 50},
        'depth': {'type': 'int', 'low': 3, 'high': 10},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bylevel': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'l2_leaf_reg': {'type': 'float', 'low': 1, 'high': 10},
        'min_data_in_leaf': {'type': 'int', 'low': 1, 'high': 100},
        'bootstrap_type': {'type': 'categorical', 'choices': ['Bayesian', 'Bernoulli']}
    }

# Instancia global de configuración
CONFIG = OptimizationConfig()
MODEL_CONFIG = ModelConfig()
