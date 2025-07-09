"""
🚀 Sistema de Logging Estructurado - Fase 5
===========================================

Sistema de logging enterprise-ready que reemplaza el logging inconsistente
del sistema anterior, proporcionando logging estructurado, contextual y escalable.

Autor: Sistema de Optimización IA
Fecha: 2025-01-09 (Fase 5 - Organización)
"""

import logging
import logging.handlers
from typing import Dict, Any, Optional, Union
import json
import time
from datetime import datetime
from pathlib import Path
import threading
from dataclasses import dataclass
import sys
import traceback


@dataclass
class LogContext:
    """Contexto de logging para seguimiento de experimentos"""
    experiment_id: Optional[str] = None
    model_name: Optional[str] = None
    trial_number: Optional[int] = None
    phase: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    """Formatter personalizado para logging estructurado"""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear el registro de log con estructura JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Agregar contexto si está disponible
        if self.include_context and hasattr(record, 'context'):
            log_entry["context"] = record.context
        
        # Agregar información de excepción si existe
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Agregar campos personalizados
        for key, value in record.__dict__.items():
            if key not in {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'context'}:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class OptimizationLogger:
    """
    Sistema de logging centralizado para optimización de hiperparámetros.
    
    Proporciona logging estructurado, contextual y escalable para todo el sistema.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton para asegurar una sola instancia de logger"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el sistema de logging.
        
        Args:
            config: Configuración de logging. Si es None, usa configuración por defecto.
        """
        if hasattr(self, '_initialized'):
            return
        
        self.config = config or self._get_default_config()
        self.context = LogContext()
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
        self._initialized = True
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuración por defecto del logging"""
        return {
            "level": "INFO",
            "log_dir": "./logs",
            "log_file_prefix": "optimization",
            "max_file_size_mb": 100,
            "backup_count": 5,
            "enable_file_logging": True,
            "enable_console_logging": True,
            "enable_structured_logging": True,
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    
    def _setup_logging(self):
        """Configurar el sistema de logging"""
        # Crear directorio de logs si no existe
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar nivel de logging
        level = getattr(logging, self.config["level"].upper())
        
        # Configurar logging root
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Limpiar handlers existentes
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Configurar formatters
        if self.config["enable_structured_logging"]:
            formatter = StructuredFormatter()
            console_formatter = logging.Formatter(self.config["log_format"])
        else:
            formatter = logging.Formatter(self.config["log_format"])
            console_formatter = formatter
        
        # Handler para archivo
        if self.config["enable_file_logging"]:
            log_file = log_dir / f"{self.config['log_file_prefix']}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config["max_file_size_mb"] * 1024 * 1024,
                backupCount=self.config["backup_count"],
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Handler para consola
        if self.config["enable_console_logging"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Obtener un logger específico.
        
        Args:
            name: Nombre del logger
            
        Returns:
            Logger configurado
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def set_context(self, **kwargs):
        """
        Establecer contexto global para logging.
        
        Args:
            **kwargs: Campos de contexto (experiment_id, model_name, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
    
    def clear_context(self):
        """Limpiar el contexto de logging"""
        self.context = LogContext()
    
    def _add_context_to_record(self, record):
        """Agregar contexto al registro de log"""
        record.context = {
            "experiment_id": self.context.experiment_id,
            "model_name": self.context.model_name,
            "trial_number": self.context.trial_number,
            "phase": self.context.phase,
            "user_id": self.context.user_id,
            "session_id": self.context.session_id
        }
        return record
    
    # ==================== MÉTODOS DE LOGGING ESPECÍFICOS ====================
    
    def log_system_info(self, logger_name: str = "system"):
        """Log información del sistema"""
        logger = self.get_logger(logger_name)
        import platform
        import psutil
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_free_gb": psutil.disk_usage('.').free / (1024**3)
        }
        
        logger.info("🖥️  Información del sistema", extra={"system_info": system_info})
    
    def log_optimization_start(self, model_name: str, trials: int, 
                              experiment_id: str, logger_name: str = "optimization"):
        """Log inicio de optimización"""
        logger = self.get_logger(logger_name)
        self.set_context(experiment_id=experiment_id, model_name=model_name, phase="optimization")
        
        logger.info(f"🚀 Iniciando optimización {model_name}", extra={
            "model_name": model_name,
            "total_trials": trials,
            "experiment_id": experiment_id,
            "event_type": "optimization_start"
        })
    
    def log_trial_result(self, trial_number: int, score: float, params: Dict[str, Any],
                        duration: float, logger_name: str = "optimization"):
        """Log resultado de trial individual"""
        logger = self.get_logger(logger_name)
        self.set_context(trial_number=trial_number)
        
        logger.info(f"📊 Trial {trial_number} completado", extra={
            "trial_number": trial_number,
            "score": score,
            "parameters": params,
            "duration_seconds": duration,
            "event_type": "trial_complete"
        })
    
    def log_optimization_complete(self, model_name: str, best_score: float, 
                                 total_duration: float, total_trials: int,
                                 logger_name: str = "optimization"):
        """Log finalización de optimización"""
        logger = self.get_logger(logger_name)
        
        logger.info(f"✅ Optimización {model_name} completada", extra={
            "model_name": model_name,
            "best_score": best_score,
            "total_duration_seconds": total_duration,
            "total_trials": total_trials,
            "trials_per_second": total_trials / total_duration if total_duration > 0 else 0,
            "event_type": "optimization_complete"
        })
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                  logger_name: str = "error"):
        """Log error con contexto completo"""
        logger = self.get_logger(logger_name)
        
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "event_type": "error"
        }
        
        logger.error(f"❌ Error: {error}", extra=error_info, exc_info=True)
    
    def log_performance_metrics(self, metrics: Dict[str, float], 
                               logger_name: str = "performance"):
        """Log métricas de performance"""
        logger = self.get_logger(logger_name)
        
        logger.info("📈 Métricas de performance", extra={
            "metrics": metrics,
            "event_type": "performance_metrics"
        })
    
    def log_memory_usage(self, component: str, logger_name: str = "memory"):
        """Log uso de memoria"""
        logger = self.get_logger(logger_name)
        import psutil
        
        memory_info = {
            "component": component,
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "event_type": "memory_usage"
        }
        
        logger.info(f"💾 Uso de memoria - {component}", extra=memory_info)
    
    def log_data_info(self, data_shape: tuple, features_count: int, 
                      target_distribution: Dict[str, int], 
                      logger_name: str = "data"):
        """Log información de datos"""
        logger = self.get_logger(logger_name)
        
        logger.info("📋 Información de datos", extra={
            "data_shape": data_shape,
            "features_count": features_count,
            "target_distribution": target_distribution,
            "event_type": "data_info"
        })


# ==================== DECORADORES DE LOGGING ====================

def log_execution_time(logger_name: str = "performance"):
    """Decorador para logear tiempo de ejecución de funciones"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = OptimizationLogger().get_logger(logger_name)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(f"⏱️  {func.__name__} ejecutado", extra={
                    "function_name": func.__name__,
                    "duration_seconds": duration,
                    "event_type": "function_execution"
                })
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ Error en {func.__name__}", extra={
                    "function_name": func.__name__,
                    "duration_seconds": duration,
                    "error": str(e),
                    "event_type": "function_error"
                }, exc_info=True)
                raise
        
        return wrapper
    return decorator


def log_memory_usage_decorator(component: str, logger_name: str = "memory"):
    """Decorador para logear uso de memoria antes y después de ejecución"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = OptimizationLogger().get_logger(logger_name)
            import psutil
            
            # Memoria antes
            memory_before = psutil.virtual_memory().percent
            
            result = func(*args, **kwargs)
            
            # Memoria después
            memory_after = psutil.virtual_memory().percent
            memory_delta = memory_after - memory_before
            
            logger.info(f"💾 Uso de memoria - {component}", extra={
                "component": component,
                "function_name": func.__name__,
                "memory_before_percent": memory_before,
                "memory_after_percent": memory_after,
                "memory_delta_percent": memory_delta,
                "event_type": "memory_tracking"
            })
            
            return result
        
        return wrapper
    return decorator


# ==================== INSTANCIA GLOBAL ====================

# Instancia global del logger
_global_logger = None

def get_logger(name: str = "optimization") -> logging.Logger:
    """
    Obtener logger global del sistema.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = OptimizationLogger()
    
    return _global_logger.get_logger(name)

def setup_logging(config: Optional[Dict[str, Any]] = None):
    """
    Configurar logging global del sistema.
    
    Args:
        config: Configuración de logging
    """
    global _global_logger
    _global_logger = OptimizationLogger(config)

def set_logging_context(**kwargs):
    """
    Establecer contexto global de logging.
    
    Args:
        **kwargs: Campos de contexto
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = OptimizationLogger()
    
    _global_logger.set_context(**kwargs)


if __name__ == "__main__":
    # Demo del sistema de logging
    print("🚀 Sistema de Logging Estructurado - Fase 5")
    print("==========================================")
    
    # Configurar logging
    config = {
        "level": "INFO",
        "log_dir": "./demo_logs",
        "enable_structured_logging": True
    }
    
    setup_logging(config)
    
    # Obtener logger
    logger = get_logger("demo")
    
    # Establecer contexto
    set_logging_context(
        experiment_id="demo_001",
        model_name="xgboost",
        phase="testing"
    )
    
    # Ejemplos de logging
    logger.info("🚀 Demo iniciado")
    
    # Simular optimización
    optimization_logger = OptimizationLogger()
    optimization_logger.log_optimization_start("xgboost", 100, "demo_001")
    optimization_logger.log_trial_result(1, 0.85, {"n_estimators": 100}, 5.2)
    optimization_logger.log_optimization_complete("xgboost", 0.87, 520.5, 100)
    
    # Log de sistema
    optimization_logger.log_system_info()
    
    print("✅ Demo completado. Revisar logs en ./demo_logs/")
