#!/usr/bin/env python3
"""
Sistema de paralelización avanzado para optimización de hiperparámetros
Incluye multiple workers, distributed optimization y queue management
"""

import multiprocessing as mp
import threading
import queue
import time
import logging
import os
import pickle
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ParallelizationConfig:
    """Configuración para paralelización"""
    
    # Configuración básica
    n_workers: int = None  # None = auto-detect
    max_workers: int = None  # Límite máximo de workers
    worker_type: str = 'process'  # 'process' or 'thread'
    
    # Configuración de cola
    queue_size: int = 1000
    batch_size: int = 10
    timeout: int = 30
    
    # Configuración distribuida
    distributed_mode: bool = False
    master_port: int = 5000
    nodes: List[str] = field(default_factory=list)
    
    # Configuración de recursos
    memory_limit_mb: int = 8192  # 8GB por worker
    cpu_limit_percent: int = 80
    
    # Configuración de retry
    max_retries: int = 3
    retry_delay: int = 1
    
    # Configuración de monitoring
    monitor_interval: int = 10
    log_worker_stats: bool = True

@dataclass
class WorkerStats:
    """Estadísticas de worker"""
    
    worker_id: str
    trials_completed: int = 0
    trials_failed: int = 0
    avg_trial_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_activity: float = 0.0
    status: str = 'idle'  # 'idle', 'running', 'error', 'stopped'

class WorkerManager:
    """Gestor de workers para optimización paralela"""
    
    def __init__(self, config: ParallelizationConfig):
        self.config = config
        self.workers_stats: Dict[str, WorkerStats] = {}
        self.task_queue = queue.Queue(maxsize=config.queue_size)
        self.result_queue = queue.Queue()
        self.workers = []
        self.is_running = False
        self.monitor_thread = None
        
        # Auto-detectar número de workers
        if config.n_workers is None:
            self.config.n_workers = min(mp.cpu_count(), 8)
        
        # Aplicar límite máximo
        if config.max_workers:
            self.config.n_workers = min(self.config.n_workers, config.max_workers)
        
        logger.info(f"WorkerManager inicializado con {self.config.n_workers} workers")
    
    def start_workers(self):
        """Iniciar workers para procesamiento paralelo"""
        if self.is_running:
            logger.warning("Workers ya están ejecutándose")
            return
        
        self.is_running = True
        
        # Crear workers según tipo
        if self.config.worker_type == 'process':
            self.executor = ProcessPoolExecutor(max_workers=self.config.n_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.config.n_workers)
        
        # Inicializar estadísticas de workers
        for i in range(self.config.n_workers):
            worker_id = f"worker_{i}"
            self.workers_stats[worker_id] = WorkerStats(worker_id=worker_id)
        
        # Iniciar thread de monitoreo
        if self.config.log_worker_stats:
            self.monitor_thread = threading.Thread(target=self._monitor_workers)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
        
        logger.info(f"Workers iniciados: {self.config.n_workers} {self.config.worker_type}s")
    
    def stop_workers(self):
        """Detener workers"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Detener executor
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # Actualizar estadísticas
        for stats in self.workers_stats.values():
            stats.status = 'stopped'
        
        logger.info("Workers detenidos")
    
    def submit_task(self, task_func: Callable, *args, **kwargs) -> Any:
        """Enviar tarea para procesamiento paralelo"""
        if not self.is_running:
            raise RuntimeError("Workers no están ejecutándose")
        
        # Enviar tarea al executor
        future = self.executor.submit(task_func, *args, **kwargs)
        return future
    
    def submit_batch(self, tasks: List[Tuple[Callable, tuple, dict]]) -> List[Any]:
        """Enviar lote de tareas para procesamiento paralelo"""
        if not self.is_running:
            raise RuntimeError("Workers no están ejecutándose")
        
        # Enviar todas las tareas
        futures = []
        for task_func, args, kwargs in tasks:
            future = self.executor.submit(task_func, *args, **kwargs)
            futures.append(future)
        
        # Recopilar resultados
        results = []
        for future in as_completed(futures, timeout=self.config.timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error en tarea: {e}")
                results.append(None)
        
        return results
    
    def _monitor_workers(self):
        """Monitorear estadísticas de workers"""
        while self.is_running:
            try:
                # Simular obtención de estadísticas
                # En implementación real, obtener métricas reales
                for worker_id, stats in self.workers_stats.items():
                    stats.memory_usage_mb = self._get_memory_usage()
                    stats.cpu_usage_percent = self._get_cpu_usage()
                    stats.last_activity = time.time()
                
                # Log estadísticas cada cierto tiempo
                if self.config.log_worker_stats:
                    self._log_worker_stats()
                
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error en monitoreo de workers: {e}")
    
    def _get_memory_usage(self) -> float:
        """Obtener uso de memoria (simulado)"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Obtener uso de CPU (simulado)"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def _log_worker_stats(self):
        """Log estadísticas de workers"""
        total_trials = sum(stats.trials_completed for stats in self.workers_stats.values())
        total_failures = sum(stats.trials_failed for stats in self.workers_stats.values())
        avg_memory = sum(stats.memory_usage_mb for stats in self.workers_stats.values()) / len(self.workers_stats)
        avg_cpu = sum(stats.cpu_usage_percent for stats in self.workers_stats.values()) / len(self.workers_stats)
        
        logger.info(f"Worker Stats - Trials: {total_trials}, Failures: {total_failures}, "
                   f"Avg Memory: {avg_memory:.1f}MB, Avg CPU: {avg_cpu:.1f}%")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas generales"""
        return {
            'n_workers': self.config.n_workers,
            'worker_type': self.config.worker_type,
            'is_running': self.is_running,
            'workers_stats': {k: v.__dict__ for k, v in self.workers_stats.items()},
            'queue_size': self.task_queue.qsize(),
            'total_trials': sum(stats.trials_completed for stats in self.workers_stats.values()),
            'total_failures': sum(stats.trials_failed for stats in self.workers_stats.values())
        }

class DistributedOptimizer:
    """Optimizador distribuido para múltiples nodos"""
    
    def __init__(self, config: ParallelizationConfig):
        self.config = config
        self.is_master = True  # Simplificado para esta implementación
        self.worker_manager = WorkerManager(config)
        self.node_stats = {}
        
    def setup_distributed_mode(self):
        """Configurar modo distribuido"""
        if not self.config.distributed_mode:
            return
        
        logger.info("Configurando modo distribuido...")
        
        # En implementación real, configurar comunicación entre nodos
        # Por ahora, simulamos con configuración local
        for i, node in enumerate(self.config.nodes):
            self.node_stats[node] = {
                'status': 'connected',
                'workers': self.config.n_workers,
                'trials_completed': 0,
                'last_heartbeat': time.time()
            }
        
        logger.info(f"Modo distribuido configurado con {len(self.config.nodes)} nodos")
    
    def distribute_trials(self, trials: List[Any]) -> List[Any]:
        """Distribuir trials entre nodos"""
        if not self.config.distributed_mode:
            # Modo local
            return self.worker_manager.submit_batch(trials)
        
        # Modo distribuido (simulado)
        results = []
        chunk_size = len(trials) // len(self.config.nodes)
        
        for i, node in enumerate(self.config.nodes):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < len(self.config.nodes) - 1 else len(trials)
            node_trials = trials[start_idx:end_idx]
            
            # En implementación real, enviar trials al nodo
            # Por ahora, procesamos localmente
            node_results = self.worker_manager.submit_batch(node_trials)
            results.extend(node_results)
        
        return results
    
    def start(self):
        """Iniciar optimizador distribuido"""
        self.setup_distributed_mode()
        self.worker_manager.start_workers()
        logger.info("Optimizador distribuido iniciado")
    
    def stop(self):
        """Detener optimizador distribuido"""
        self.worker_manager.stop_workers()
        logger.info("Optimizador distribuido detenido")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cluster"""
        return {
            'master_stats': self.worker_manager.get_stats(),
            'node_stats': self.node_stats,
            'distributed_mode': self.config.distributed_mode,
            'total_nodes': len(self.config.nodes) + 1  # +1 para nodo master
        }

class ParallelTrialExecutor:
    """Executor para trials paralelos de Optuna"""
    
    def __init__(self, config: ParallelizationConfig):
        self.config = config
        self.distributed_optimizer = DistributedOptimizer(config)
        self.execution_history = []
        
    def execute_parallel_trials(self, objective_func: Callable, 
                              study: Any, n_trials: int) -> Dict[str, Any]:
        """Ejecutar trials en paralelo"""
        start_time = time.time()
        
        # Iniciar optimizador distribuido
        self.distributed_optimizer.start()
        
        try:
            # Preparar tasks para ejecución paralela
            tasks = []
            for i in range(n_trials):
                tasks.append((self._execute_single_trial, (objective_func, study, i), {}))
            
            # Ejecutar trials distribuidos
            results = self.distributed_optimizer.distribute_trials(tasks)
            
            # Procesar resultados
            successful_trials = [r for r in results if r is not None]
            failed_trials = len(results) - len(successful_trials)
            
            execution_time = time.time() - start_time
            
            # Guardar historial
            execution_info = {
                'n_trials': n_trials,
                'successful_trials': len(successful_trials),
                'failed_trials': failed_trials,
                'execution_time': execution_time,
                'trials_per_second': len(successful_trials) / execution_time if execution_time > 0 else 0,
                'worker_stats': self.distributed_optimizer.worker_manager.get_stats(),
                'timestamp': time.time()
            }
            
            self.execution_history.append(execution_info)
            
            return execution_info
            
        finally:
            # Detener optimizador distribuido
            self.distributed_optimizer.stop()
    
    def _execute_single_trial(self, objective_func: Callable, study: Any, trial_id: int) -> Any:
        """Ejecutar un trial individual"""
        try:
            # Simular ejecución de trial
            # En implementación real, ejecutar objective_func con study
            trial_start = time.time()
            
            # Placeholder para ejecución real
            result = {
                'trial_id': trial_id,
                'value': 0.5 + 0.3 * (trial_id % 10) / 10,  # Valor simulado
                'duration': time.time() - trial_start,
                'status': 'completed'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error en trial {trial_id}: {e}")
            return None
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de ejecuciones"""
        return self.execution_history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento"""
        if not self.execution_history:
            return {}
        
        latest = self.execution_history[-1]
        all_trials = sum(h['successful_trials'] for h in self.execution_history)
        all_time = sum(h['execution_time'] for h in self.execution_history)
        
        return {
            'latest_execution': latest,
            'total_trials': all_trials,
            'total_time': all_time,
            'avg_trials_per_second': all_trials / all_time if all_time > 0 else 0,
            'total_executions': len(self.execution_history)
        }

# Configuraciones predefinidas
DEFAULT_PARALLELIZATION_CONFIG = ParallelizationConfig(
    n_workers=None,  # Auto-detect
    worker_type='process',
    queue_size=1000,
    batch_size=10,
    timeout=30,
    memory_limit_mb=8192,
    cpu_limit_percent=80,
    max_retries=3,
    monitor_interval=10,
    log_worker_stats=True
)

DISTRIBUTED_CONFIG = ParallelizationConfig(
    n_workers=4,
    worker_type='process',
    distributed_mode=True,
    master_port=5000,
    nodes=['node1', 'node2', 'node3'],
    queue_size=2000,
    batch_size=20,
    timeout=60,
    memory_limit_mb=16384,
    cpu_limit_percent=90,
    max_retries=5,
    monitor_interval=15,
    log_worker_stats=True
)

# Instancias globales
WORKER_MANAGER = WorkerManager(DEFAULT_PARALLELIZATION_CONFIG)
DISTRIBUTED_OPTIMIZER = DistributedOptimizer(DEFAULT_PARALLELIZATION_CONFIG)
PARALLEL_TRIAL_EXECUTOR = ParallelTrialExecutor(DEFAULT_PARALLELIZATION_CONFIG)

if __name__ == "__main__":
    # Test básico
    print("🚀 Sistema de Paralelización inicializado")
    print(f"   👥 Workers: {DEFAULT_PARALLELIZATION_CONFIG.n_workers}")
    print(f"   🔄 Tipo: {DEFAULT_PARALLELIZATION_CONFIG.worker_type}")
    print(f"   📊 Queue size: {DEFAULT_PARALLELIZATION_CONFIG.queue_size}")
    print(f"   🌐 Modo distribuido: {DEFAULT_PARALLELIZATION_CONFIG.distributed_mode}")
    
    # Test de worker manager
    WORKER_MANAGER.start_workers()
    time.sleep(1)
    stats = WORKER_MANAGER.get_stats()
    print(f"   📈 Workers iniciados: {stats['n_workers']}")
    WORKER_MANAGER.stop_workers()
