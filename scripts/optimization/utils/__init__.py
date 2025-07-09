#!/usr/bin/env python3
"""
Módulo de utilidades para optimización de hiperparámetros
Incluye mejoras de Fase 1 y Fase 2
"""

# Componentes de Fase 1 (Fundamentos críticos)
try:
    from .gpu_manager import GPU_MANAGER, GPUManager
    from .data_validator import DataValidator, DataValidationError
    from .metrics_calculator import MetricsCalculator, MetricsResult
    from .optimization_logger import OptimizationLogger, get_optimization_logger
    print("✅ Componentes de Fase 1 importados correctamente")
except ImportError as e:
    print(f"⚠️ Error importando componentes de Fase 1: {e}")

# Componentes de Fase 2 (Optimización avanzada)
try:
    from .temporal_validator import (
        TimeSeriesValidator, TimeSeriesValidationConfig, 
        TEMPORAL_VALIDATOR
    )
    from .early_stopping import (
        EarlyStoppingMonitor, EarlyStoppingConfig, 
        AdaptiveOptimizationController, ADAPTIVE_CONTROLLER
    )
    from .multi_objective import (
        MultiObjectiveOptimizer, MultiObjectiveConfig,
        OptimizationObjective, MULTI_OBJECTIVE_OPTIMIZER
    )
    print("✅ Componentes de Fase 2 importados correctamente")
except ImportError as e:
    print(f"⚠️ Error importando componentes de Fase 2: {e}")

# Componentes de Fase 3 (Eficiencia y escalabilidad)
try:
    from .parallelization import (
        ParallelizationConfig, WorkerManager, DistributedOptimizer,
        ParallelTrialExecutor, WORKER_MANAGER, DISTRIBUTED_OPTIMIZER,
        PARALLEL_TRIAL_EXECUTOR
    )
    from .memory_manager import (
        MemoryConfig, MemoryManager, MemoryMonitor, GarbageCollector,
        DataChunkProcessor, CacheManager, PersistenceManager, MEMORY_MANAGER
    )
    print("✅ Componentes de Fase 3 importados correctamente")
except ImportError as e:
    print(f"⚠️ Error importando componentes de Fase 3: {e}")

# Metadatos del módulo
__version__ = "3.0.0"
__author__ = "Crypto ML Team"
__description__ = "Utilidades avanzadas para optimización de hiperparámetros con paralelización y gestión de memoria"

# Exportaciones principales
__all__ = [
    # Fase 1
    'GPU_MANAGER',
    'GPUManager',
    'DataValidator',
    'DataValidationError',
    'MetricsCalculator',
    'MetricsResult',
    'OptimizationLogger',
    'get_optimization_logger',
    
    # Fase 2
    'TimeSeriesValidator',
    'TimeSeriesValidationConfig',
    'TEMPORAL_VALIDATOR',
    'EarlyStoppingMonitor',
    'EarlyStoppingConfig',
    'AdaptiveOptimizationController',
    'ADAPTIVE_CONTROLLER',
    'MultiObjectiveOptimizer',
    'MultiObjectiveConfig',
    'OptimizationObjective',
    'MULTI_OBJECTIVE_OPTIMIZER',
    
    # Fase 3
    'ParallelizationConfig',
    'WorkerManager',
    'DistributedOptimizer',
    'ParallelTrialExecutor',
    'WORKER_MANAGER',
    'DISTRIBUTED_OPTIMIZER',
    'PARALLEL_TRIAL_EXECUTOR',
    'MemoryConfig',
    'MemoryManager',
    'MemoryMonitor',
    'GarbageCollector',
    'DataChunkProcessor',
    'CacheManager',
    'PersistenceManager',
    'MEMORY_MANAGER'
]

def get_phase_info():
    """Obtener información sobre las fases implementadas"""
    return {
        'version': __version__,
        'fase_1_disponible': 'DataValidator' in globals(),
        'fase_2_disponible': 'TimeSeriesValidator' in globals(),
        'fase_3_disponible': 'ParallelizationConfig' in globals() and 'MemoryManager' in globals(),
        'componentes_fase_1': [
            'GPU_MANAGER', 'DataValidator', 'MetricsCalculator', 
            'OptimizationLogger'
        ],
        'componentes_fase_2': [
            'TimeSeriesValidator', 'EarlyStoppingMonitor', 
            'AdaptiveOptimizationController', 'MultiObjectiveOptimizer'
        ],
        'componentes_fase_3': [
            'ParallelizationConfig', 'WorkerManager', 'DistributedOptimizer',
            'ParallelTrialExecutor', 'MemoryManager', 'CacheManager'
        ]
    }

if __name__ == "__main__":
    print("🔧 Módulo de utilidades de optimización")
    info = get_phase_info()
    print(f"   📖 Versión: {info['version']}")
    print(f"   ✅ Fase 1 disponible: {info['fase_1_disponible']}")
    print(f"   ✅ Fase 2 disponible: {info['fase_2_disponible']}")
    print(f"   ✅ Fase 3 disponible: {info['fase_3_disponible']}")
    print(f"   📊 Componentes Fase 1: {len(info['componentes_fase_1'])}")
    print(f"   📊 Componentes Fase 2: {len(info['componentes_fase_2'])}")
    print(f"   📊 Componentes Fase 3: {len(info['componentes_fase_3'])}")
