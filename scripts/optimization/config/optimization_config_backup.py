"""
🚀 Configuración Centralizada del Sistema de Optimización - Fase 5
================================================================

Configuración enterprise-ready centralizada que reemplaza todas las configuraciones
hardcodeadas distribuidas por el sistema.

Autor: Sistema de Optimización IA
Fecha: 2025-01-09 (Fase 5 - Organización)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

@dataclass
class OptimizationConfig:
    """
    Configuración centralizada del sistema de optimización de hiperparámetros.
    
    Esta clase centraliza todas las configuraciones que anteriormente estaban
    dispersas por el código, proporcionando un punto único de verdad.
    """
    
    # ==================== CONFIGURACIÓN DE DATOS ====================
    data_path: str = "../../../data/crypto_ohlc_join.csv"
    target_column: str = "target_next_close_positive"
    test_size: float = 0.2
    validation_size: float = 0.15
    min_samples_per_split: int = 1000
    
    # ==================== CONFIGURACIÓN DE MODELOS ====================
    enabled_models: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost"
    ])
    
    # Configuración de trials por modelo
    model_trials: Dict[str, int] = field(default_factory=lambda: {
        "xgboost": 100,
        "lightgbm": 100,
        "catboost": 100
    })
    
    # ==================== CONFIGURACIÓN DE OPTIMIZACIÓN ====================
    optimization_timeout: Optional[int] = 3600  # 1 hora en segundos
    n_jobs: int = -1  # Usar todos los cores disponibles
    random_state: int = 42
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"
    
    # ==================== CONFIGURACIÓN DE GPU ====================
    enable_gpu: bool = True
    gpu_device_id: int = 0
    gpu_memory_limit: Optional[int] = None  # MB, None = sin límite
    
    # ==================== CONFIGURACIÓN DE CACHE ====================
    enable_cache: bool = True
    cache_dir: str = "./cache"
    max_cache_size_mb: int = 1024  # 1GB
    cache_expiry_hours: int = 24
    
    # ==================== CONFIGURACIÓN DE LOGGING ====================
    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_file_prefix: str = "optimization"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ==================== CONFIGURACIÓN DE RESULTADOS ====================
    results_dir: str = "./results"
    save_models: bool = True
    save_feature_importance: bool = True
    save_optimization_history: bool = True
    
    # ==================== CONFIGURACIÓN DE FEATURES ====================
    feature_engineering_enabled: bool = True
    technical_indicators: List[str] = field(default_factory=lambda: [
        "rsi", "macd", "bollinger_bands", "moving_averages"
    ])
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    
    # ==================== CONFIGURACIÓN DE MEMORIA ====================
    max_memory_usage_gb: float = 8.0
    garbage_collection_frequency: int = 10  # Cada N trials
    
    def __post_init__(self):
        """Validaciones y configuraciones automáticas post-inicialización"""
        self._validate_config()
        self._setup_directories()
        self._detect_gpu()
    
    def _validate_config(self):
        """Validar configuración para detectar inconsistencias"""
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("test_size + validation_size debe ser < 1.0")
        
        if self.min_samples_per_split < 100:
            raise ValueError("min_samples_per_split debe ser >= 100")
        
        if not self.enabled_models:
            raise ValueError("Debe haber al menos un modelo habilitado")
        
        # Validar que todos los modelos habilitados tengan configuración de trials
        for model in self.enabled_models:
            if model not in self.model_trials:
                self.model_trials[model] = 100  # Default
    
    def _setup_directories(self):
        """Crear directorios necesarios si no existen"""
        directories = [
            self.cache_dir,
            self.log_dir,
            self.results_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _detect_gpu(self):
        """Detectar automáticamente disponibilidad de GPU"""
        if self.enable_gpu:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if not gpus:
                    print("⚠️  GPU solicitada pero no detectada. Deshabilitando GPU.")
                    self.enable_gpu = False
                else:
                    print(f"✅ GPU detectada: {gpus[0].name}")
            except ImportError:
                try:
                    import tensorflow as tf
                    if tf.config.list_physical_devices('GPU'):
                        print("✅ GPU detectada via TensorFlow")
                    else:
                        print("⚠️  GPU no detectada. Deshabilitando GPU.")
                        self.enable_gpu = False
                except ImportError:
                    print("⚠️  No se puede detectar GPU. Deshabilitando GPU.")
                    self.enable_gpu = False
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Obtener configuración específica para un modelo"""
        base_config = {
            "random_state": self.random_state,
            "n_jobs": self.n_jobs if model_name != "catboost" else 1,
            "enable_gpu": self.enable_gpu
        }
        
        # Configuraciones específicas por modelo
        if model_name == "xgboost":
            base_config.update({
                "tree_method": "gpu_hist" if self.enable_gpu else "hist",
                "gpu_id": self.gpu_device_id if self.enable_gpu else None
            })
        elif model_name == "lightgbm":
            base_config.update({
                "device": "gpu" if self.enable_gpu else "cpu",
                "gpu_device_id": self.gpu_device_id if self.enable_gpu else None
            })
        elif model_name == "catboost":
            base_config.update({
                "task_type": "GPU" if self.enable_gpu else "CPU",
                "devices": f"0:{self.gpu_device_id}" if self.enable_gpu else None
            })
        
        return base_config
    
    def get_data_path(self) -> str:
        """Obtener ruta absoluta de datos"""
        if os.path.isabs(self.data_path):
            return self.data_path
        
        # Ruta relativa desde el directorio actual
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, self.data_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario para serialización"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizationConfig':
        """Crear configuración desde diccionario"""
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        """Guardar configuración a archivo JSON"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'OptimizationConfig':
        """Cargar configuración desde archivo JSON"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# ==================== CONFIGURACIONES PREDEFINIDAS ====================

def get_quick_config() -> OptimizationConfig:
    """Configuración para optimización rápida (desarrollo/testing)"""
    return OptimizationConfig(
        model_trials={"xgboost": 20, "lightgbm": 20, "catboost": 20},
        optimization_timeout=600,  # 10 minutos
        cache_expiry_hours=1,
        max_memory_usage_gb=4.0
    )

def get_production_config() -> OptimizationConfig:
    """Configuración para optimización completa (producción)"""
    return OptimizationConfig(
        model_trials={"xgboost": 200, "lightgbm": 200, "catboost": 200},
        optimization_timeout=7200,  # 2 horas
        cache_expiry_hours=48,
        max_memory_usage_gb=16.0,
        cv_folds=10
    )

def get_gpu_config() -> OptimizationConfig:
    """Configuración optimizada para GPU"""
    return OptimizationConfig(
        enable_gpu=True,
        model_trials={"xgboost": 300, "lightgbm": 300, "catboost": 150},
        optimization_timeout=3600,  # 1 hora
        max_memory_usage_gb=12.0,
        n_jobs=1  # GPU maneja paralelización internamente
    )

def get_cpu_config() -> OptimizationConfig:
    """Configuración optimizada para CPU"""
    return OptimizationConfig(
        enable_gpu=False,
        model_trials={"xgboost": 150, "lightgbm": 150, "catboost": 100},
        optimization_timeout=5400,  # 1.5 horas
        max_memory_usage_gb=8.0,
        n_jobs=-1  # Usar todos los cores
    )


# ==================== INSTANCIA GLOBAL ====================

# Configuración por defecto del sistema
DEFAULT_CONFIG = OptimizationConfig()

# Función de conveniencia para obtener la configuración
def get_config() -> OptimizationConfig:
    """Obtener la configuración global del sistema"""
    return DEFAULT_CONFIG

def set_config(config: OptimizationConfig):
    """Establecer una nueva configuración global"""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config


if __name__ == "__main__":
    # Demo de configuraciones
    print("🚀 Configuración Centralizada - Fase 5")
    print("=====================================")
    
    # Configuración por defecto
    config = OptimizationConfig()
    print(f"✅ Configuración por defecto creada")
    print(f"   - Modelos habilitados: {config.enabled_models}")
    print(f"   - GPU habilitada: {config.enable_gpu}")
    print(f"   - Ruta de datos: {config.get_data_path()}")
    
    # Configuraciones predefinidas
    quick_config = get_quick_config()
    print(f"\n⚡ Configuración rápida:")
    print(f"   - Trials XGBoost: {quick_config.model_trials['xgboost']}")
    print(f"   - Timeout: {quick_config.optimization_timeout}s")
    
    production_config = get_production_config()
    print(f"\n🏭 Configuración producción:")
    print(f"   - Trials XGBoost: {production_config.model_trials['xgboost']}")
    print(f"   - CV Folds: {production_config.cv_folds}")
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
