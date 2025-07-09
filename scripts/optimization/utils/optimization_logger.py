#!/usr/bin/env python3
"""
Sistema de logging estructurado para optimización de hiperparámetros
"""

import logging
import logging.handlers
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

@dataclass
class LogEntry:
    """Entrada estructurada de log"""
    timestamp: str
    level: str
    component: str
    message: str
    context: Dict[str, Any]
    trial_id: Optional[int] = None
    model_name: Optional[str] = None
    metric_value: Optional[float] = None

class OptimizationLogger:
    """Logger especializado para optimización de hiperparámetros"""
    
    def __init__(self, log_dir: str = "logs", 
                 log_level: str = "INFO",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        
        # Crear timestamp para esta sesión
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configurar loggers
        self.main_logger = self._setup_main_logger(log_level)
        self.trial_logger = self._setup_trial_logger()
        self.metrics_logger = self._setup_metrics_logger()
        
        # Contadores y estadísticas
        self.stats = {
            'total_trials': 0,
            'successful_trials': 0,
            'failed_trials': 0,
            'best_score': None,
            'start_time': datetime.now()
        }
        
        self.log_info("OptimizationLogger inicializado", {
            'session_id': self.session_id,
            'log_dir': str(self.log_dir),
            'file_logging': enable_file_logging,
            'console_logging': enable_console_logging
        })
    
    def _setup_main_logger(self, log_level: str) -> logging.Logger:
        """Configurar logger principal"""
        logger = logging.getLogger(f'optimization_main_{self.session_id}')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Evitar duplicación de handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Formatter para logs estructurados
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
        )
        
        # Handler para consola
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Handler para archivo
        if self.enable_file_logging:
            log_file = self.log_dir / f"optimization_{self.session_id}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=50*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_trial_logger(self) -> logging.Logger:
        """Configurar logger específico para trials"""
        logger = logging.getLogger(f'optimization_trials_{self.session_id}')
        logger.setLevel(logging.DEBUG)
        
        if logger.handlers:
            logger.handlers.clear()
        
        if self.enable_file_logging:
            log_file = self.log_dir / f"trials_{self.session_id}.jsonl"
            file_handler = logging.FileHandler(log_file)
            
            # Formatter JSON para trials
            json_formatter = JsonFormatter()
            file_handler.setFormatter(json_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_metrics_logger(self) -> logging.Logger:
        """Configurar logger específico para métricas"""
        logger = logging.getLogger(f'optimization_metrics_{self.session_id}')
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        if self.enable_file_logging:
            log_file = self.log_dir / f"metrics_{self.session_id}.csv"
            file_handler = logging.FileHandler(log_file)
            
            # Formatter CSV para métricas
            csv_formatter = CsvFormatter()
            file_handler.setFormatter(csv_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    # ==================== MÉTODOS DE LOGGING PRINCIPALES ====================
    
    def log_debug(self, message: str, context: Dict[str, Any] = None):
        """Log nivel DEBUG"""
        self._log(logging.DEBUG, message, context)
    
    def log_info(self, message: str, context: Dict[str, Any] = None):
        """Log nivel INFO"""
        self._log(logging.INFO, message, context)
    
    def log_warning(self, message: str, context: Dict[str, Any] = None):
        """Log nivel WARNING"""
        self._log(logging.WARNING, message, context)
    
    def log_error(self, message: str, context: Dict[str, Any] = None, exception: Exception = None):
        """Log nivel ERROR"""
        if exception:
            if context is None:
                context = {}
            context.update({
                'exception_type': type(exception).__name__,
                'exception_message': str(exception)
            })
        self._log(logging.ERROR, message, context)
    
    def _log(self, level: int, message: str, context: Dict[str, Any] = None):
        """Método interno de logging"""
        if context is None:
            context = {}
        
        # Añadir contexto automático
        context.update({
            'session_id': self.session_id,
            'timestamp_iso': datetime.now().isoformat()
        })
        
        # Log al logger principal
        if context:
            formatted_message = f"{message} | Context: {json.dumps(context, default=str)}"
        else:
            formatted_message = message
        
        self.main_logger.log(level, formatted_message)
    
    # ==================== MÉTODOS ESPECÍFICOS DE OPTIMIZACIÓN ====================
    
    def log_optimization_start(self, config: Dict[str, Any]):
        """Log inicio de optimización"""
        self.log_info("🚀 OPTIMIZACIÓN INICIADA", {
            'config': config,
            'component': 'optimization'
        })
    
    def log_optimization_complete(self, results: Dict[str, Any]):
        """Log finalización de optimización"""
        duration = datetime.now() - self.stats['start_time']
        
        self.log_info("✅ OPTIMIZACIÓN COMPLETADA", {
            'duration_seconds': duration.total_seconds(),
            'total_trials': self.stats['total_trials'],
            'successful_trials': self.stats['successful_trials'],
            'failed_trials': self.stats['failed_trials'],
            'success_rate': self.stats['successful_trials'] / max(self.stats['total_trials'], 1),
            'best_score': self.stats['best_score'],
            'results': results,
            'component': 'optimization'
        })
    
    def log_model_optimization_start(self, model_name: str, n_trials: int, config: Dict[str, Any]):
        """Log inicio de optimización de un modelo específico"""
        self.log_info(f"🔧 OPTIMIZACIÓN {model_name.upper()} INICIADA", {
            'model_name': model_name,
            'n_trials': n_trials,
            'config': config,
            'component': 'model_optimization'
        })
    
    def log_model_optimization_complete(self, model_name: str, best_score: float, 
                                      best_params: Dict[str, Any], duration: float):
        """Log finalización de optimización de un modelo"""
        self.log_info(f"✅ OPTIMIZACIÓN {model_name.upper()} COMPLETADA", {
            'model_name': model_name,
            'best_score': best_score,
            'best_params': best_params,
            'duration_seconds': duration,
            'component': 'model_optimization'
        })
        
        # Actualizar estadísticas
        if self.stats['best_score'] is None or best_score > self.stats['best_score']:
            self.stats['best_score'] = best_score
    
    def log_trial_start(self, trial_id: int, model_name: str, params: Dict[str, Any]):
        """Log inicio de trial"""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level="INFO",
            component="trial",
            message="Trial iniciado",
            context={
                'params': params,
                'model_name': model_name
            },
            trial_id=trial_id,
            model_name=model_name
        )
        
        self.trial_logger.info(json.dumps(asdict(log_entry), default=str))
        self.stats['total_trials'] += 1
    
    def log_trial_complete(self, trial_id: int, model_name: str, score: float, 
                          duration: float, status: str = "success"):
        """Log finalización de trial"""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level="INFO" if status == "success" else "ERROR",
            component="trial",
            message=f"Trial {status}",
            context={
                'duration_seconds': duration,
                'status': status
            },
            trial_id=trial_id,
            model_name=model_name,
            metric_value=score
        )
        
        self.trial_logger.info(json.dumps(asdict(log_entry), default=str))
        
        if status == "success":
            self.stats['successful_trials'] += 1
        else:
            self.stats['failed_trials'] += 1
    
    def log_trial_pruned(self, trial_id: int, model_name: str, reason: str):
        """Log trial podado por early stopping"""
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level="INFO",
            component="trial",
            message="Trial podado",
            context={
                'reason': reason,
                'status': 'pruned'
            },
            trial_id=trial_id,
            model_name=model_name
        )
        
        self.trial_logger.info(json.dumps(asdict(log_entry), default=str))
    
    def log_metrics(self, trial_id: int, model_name: str, metrics: Dict[str, float]):
        """Log métricas de evaluación"""
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'trial_id': trial_id,
            'model_name': model_name,
            **metrics
        }
        
        # CSV format log
        if not hasattr(self, '_metrics_header_written'):
            header = ','.join(metrics_entry.keys())
            self.metrics_logger.info(header)
            self._metrics_header_written = True
        
        values = ','.join(str(v) for v in metrics_entry.values())
        self.metrics_logger.info(values)
    
    def log_gpu_info(self, gpu_info: Dict[str, Any]):
        """Log información de GPU"""
        self.log_info("🎮 INFORMACIÓN DE GPU", {
            'gpu_info': gpu_info,
            'component': 'hardware'
        })
    
    def log_data_info(self, data_info: Dict[str, Any]):
        """Log información de datos"""
        self.log_info("📊 INFORMACIÓN DE DATOS", {
            'data_info': data_info,
            'component': 'data'
        })
    
    def log_memory_usage(self, memory_info: Dict[str, Any]):
        """Log uso de memoria"""
        self.log_debug("💾 USO DE MEMORIA", {
            'memory_info': memory_info,
            'component': 'performance'
        })
    
    def log_progress(self, current_trial: int, total_trials: int, 
                    current_best: float, model_name: str):
        """Log progreso de optimización"""
        progress_pct = (current_trial / total_trials) * 100
        
        self.log_info(f"📈 PROGRESO {model_name.upper()}: {current_trial}/{total_trials} ({progress_pct:.1f}%)", {
            'current_trial': current_trial,
            'total_trials': total_trials,
            'progress_percent': progress_pct,
            'current_best_score': current_best,
            'model_name': model_name,
            'component': 'progress'
        })
    
    # ==================== UTILIDADES ====================
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Obtener resumen de la sesión actual"""
        duration = datetime.now() - self.stats['start_time']
        
        return {
            'session_id': self.session_id,
            'duration_seconds': duration.total_seconds(),
            'total_trials': self.stats['total_trials'],
            'successful_trials': self.stats['successful_trials'],
            'failed_trials': self.stats['failed_trials'],
            'success_rate': self.stats['successful_trials'] / max(self.stats['total_trials'], 1),
            'best_score': self.stats['best_score'],
            'start_time': self.stats['start_time'].isoformat(),
            'end_time': datetime.now().isoformat()
        }
    
    def export_logs(self, output_dir: str = None) -> Dict[str, str]:
        """Exportar logs a directorio específico"""
        if output_dir is None:
            output_dir = self.log_dir / "exports" / self.session_id
        
        export_dir = Path(output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Copiar archivos de log
        log_files = list(self.log_dir.glob(f"*{self.session_id}*"))
        exported_files = {}
        
        for log_file in log_files:
            dest_file = export_dir / log_file.name
            dest_file.write_text(log_file.read_text())
            exported_files[log_file.stem] = str(dest_file)
        
        # Crear resumen
        summary_file = export_dir / "session_summary.json"
        summary_file.write_text(json.dumps(self.get_session_summary(), indent=2, default=str))
        exported_files['summary'] = str(summary_file)
        
        self.log_info("📁 LOGS EXPORTADOS", {
            'export_dir': str(export_dir),
            'exported_files': list(exported_files.keys()),
            'component': 'export'
        })
        
        return exported_files

class JsonFormatter(logging.Formatter):
    """Formatter para logs en formato JSON"""
    
    def format(self, record):
        return record.getMessage()

class CsvFormatter(logging.Formatter):
    """Formatter para logs en formato CSV"""
    
    def format(self, record):
        return record.getMessage()

# Instancia global del logger
def get_optimization_logger(log_dir: str = "logs", **kwargs) -> OptimizationLogger:
    """Factory function para obtener logger de optimización"""
    return OptimizationLogger(log_dir=log_dir, **kwargs)
