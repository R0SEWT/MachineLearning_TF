#!/usr/bin/env python3
"""
Configuración avanzada para samplers y pruners de Optuna
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner, ThresholdPruner
from optuna.study import StudyDirection

@dataclass
class SamplerConfig:
    """Configuración para samplers de Optuna"""
    
    # Configuración TPESampler
    tpe_config = {
        'n_startup_trials': 20,  # Trials aleatorios antes de usar TPE
        'n_ei_candidates': 24,   # Candidatos para Expected Improvement
        'gamma': 0.25,           # Ratio de muestras para modelar
        'multivariate': True,    # Considerar correlaciones entre parámetros
        'constant_liar': True,   # Paralelización mejorada
        'warn_independent_sampling': True,
        'seed': 42
    }
    
    # Configuración CmaEsSampler
    cmaes_config = {
        'n_startup_trials': 10,  # Trials aleatorios antes de usar CMA-ES
        'independent_sampler': None,  # RandomSampler por defecto
        'warn_independent_sampling': True,
        'seed': 42
    }
    
    # Configuración RandomSampler
    random_config = {
        'seed': 42
    }
    
    # Estrategia de selección de sampler
    sampler_strategy = {
        'default': 'tpe',
        'continuous_heavy': 'cmaes',  # Para espacios con muchos parámetros continuos
        'mixed_space': 'tpe',         # Para espacios mixtos
        'high_dimensional': 'tpe',    # Para espacios de alta dimensión
        'small_budget': 'random'      # Para pocos trials
    }

@dataclass
class PrunerConfig:
    """Configuración para pruners de Optuna"""
    
    # Configuración MedianPruner
    median_config = {
        'n_startup_trials': 5,      # Trials antes de empezar pruning
        'n_warmup_steps': 10,       # Steps de warmup
        'interval_steps': 1,        # Intervalo de evaluación
        'n_min_trials': 10          # Mínimo de trials para estadísticas
    }
    
    # Configuración HyperbandPruner
    hyperband_config = {
        'min_resource': 1,          # Recurso mínimo (epochs, iterations)
        'max_resource': 'auto',     # Recurso máximo
        'reduction_factor': 3,      # Factor de reducción
        'bootstrap_count': 0        # Trials antes de empezar
    }
    
    # Configuración ThresholdPruner
    threshold_config = {
        'lower': 0.5,               # Umbral inferior
        'upper': 1.0,               # Umbral superior
        'n_warmup_steps': 5         # Steps de warmup
    }
    
    # Estrategia de selección de pruner
    pruner_strategy = {
        'default': 'median',
        'aggressive': 'hyperband',
        'conservative': 'threshold',
        'adaptive': 'median'
    }

@dataclass
class MultiObjectiveConfig:
    """Configuración para optimización multi-objetivo"""
    
    # Objetivos disponibles
    objectives = {
        'primary': {'name': 'roc_auc', 'direction': 'maximize', 'weight': 0.6},
        'stability': {'name': 'stability_score', 'direction': 'maximize', 'weight': 0.2},
        'efficiency': {'name': 'training_time', 'direction': 'minimize', 'weight': 0.1},
        'complexity': {'name': 'model_complexity', 'direction': 'minimize', 'weight': 0.1}
    }
    
    # Configuración de estudio multi-objetivo
    study_config = {
        'directions': ['maximize', 'maximize'],  # Para objectives primarios
        'sampler': 'nsga2',  # NSGA-II para multi-objetivo
        'pruner': 'hyperband'
    }
    
    # Configuración NSGA-II
    nsga2_config = {
        'population_size': 50,
        'mutation_prob': 0.1,
        'crossover_prob': 0.9,
        'swapping_prob': 0.5,
        'seed': 42
    }

@dataclass
class ConvergenceConfig:
    """Configuración para detección de convergencia"""
    
    # Criterios de convergencia
    convergence_criteria = {
        'patience': 20,              # Trials sin mejora
        'min_improvement': 0.001,    # Mejora mínima considerada
        'stability_window': 10,      # Ventana para calcular estabilidad
        'convergence_threshold': 0.99, # Umbral de convergencia
        'max_trials_factor': 2.0     # Factor máximo de trials
    }
    
    # Configuración de early stopping adaptativo
    adaptive_stopping = {
        'enabled': True,
        'warmup_trials': 30,         # Trials antes de considerar stopping
        'check_interval': 5,         # Intervalo de verificación
        'improvement_threshold': 0.005, # Umbral de mejora
        'plateau_patience': 15       # Paciencia para plateau
    }

@dataclass
class OptimizationStrategy:
    """Estrategia completa de optimización"""
    
    # Estrategias por tipo de problema
    strategies = {
        'quick_exploration': {
            'sampler': 'random',
            'pruner': 'median',
            'trials': 50,
            'timeout': 300,
            'convergence': 'standard'
        },
        'balanced_optimization': {
            'sampler': 'tpe',
            'pruner': 'median',
            'trials': 200,
            'timeout': 1800,
            'convergence': 'adaptive'
        },
        'thorough_search': {
            'sampler': 'tpe',
            'pruner': 'hyperband',
            'trials': 500,
            'timeout': 3600,
            'convergence': 'strict'
        },
        'continuous_optimization': {
            'sampler': 'cmaes',
            'pruner': 'median',
            'trials': 300,
            'timeout': 2400,
            'convergence': 'adaptive'
        }
    }

class OptunaSamplerFactory:
    """Factory para crear samplers de Optuna"""
    
    @staticmethod
    def create_sampler(sampler_type: str, config: SamplerConfig, **kwargs) -> optuna.samplers.BaseSampler:
        """Crear sampler según tipo y configuración"""
        
        if sampler_type == 'tpe':
            return TPESampler(
                n_startup_trials=config.tpe_config['n_startup_trials'],
                n_ei_candidates=config.tpe_config['n_ei_candidates'],
                gamma=config.tpe_config['gamma'],
                multivariate=config.tpe_config['multivariate'],
                constant_liar=config.tpe_config['constant_liar'],
                warn_independent_sampling=config.tpe_config['warn_independent_sampling'],
                seed=config.tpe_config['seed'],
                **kwargs
            )
        
        elif sampler_type == 'cmaes':
            return CmaEsSampler(
                n_startup_trials=config.cmaes_config['n_startup_trials'],
                warn_independent_sampling=config.cmaes_config['warn_independent_sampling'],
                seed=config.cmaes_config['seed'],
                **kwargs
            )
        
        elif sampler_type == 'random':
            return RandomSampler(
                seed=config.random_config['seed'],
                **kwargs
            )
        
        elif sampler_type == 'nsga2':
            try:
                from optuna.samplers import NSGAIISampler
                return NSGAIISampler(
                    population_size=kwargs.get('population_size', 50),
                    mutation_prob=kwargs.get('mutation_prob', 0.1),
                    crossover_prob=kwargs.get('crossover_prob', 0.9),
                    swapping_prob=kwargs.get('swapping_prob', 0.5),
                    seed=kwargs.get('seed', 42)
                )
            except ImportError:
                # Fallback si NSGA-II no está disponible
                return TPESampler(seed=42)
        
        else:
            raise ValueError(f"Sampler tipo '{sampler_type}' no soportado")

class OptunaPrunerFactory:
    """Factory para crear pruners de Optuna"""
    
    @staticmethod
    def create_pruner(pruner_type: str, config: PrunerConfig, **kwargs) -> optuna.pruners.BasePruner:
        """Crear pruner según tipo y configuración"""
        
        if pruner_type == 'median':
            return MedianPruner(
                n_startup_trials=config.median_config['n_startup_trials'],
                n_warmup_steps=config.median_config['n_warmup_steps'],
                interval_steps=config.median_config['interval_steps'],
                n_min_trials=config.median_config['n_min_trials'],
                **kwargs
            )
        
        elif pruner_type == 'hyperband':
            return HyperbandPruner(
                min_resource=config.hyperband_config['min_resource'],
                max_resource=config.hyperband_config['max_resource'],
                reduction_factor=config.hyperband_config['reduction_factor'],
                bootstrap_count=config.hyperband_config['bootstrap_count'],
                **kwargs
            )
        
        elif pruner_type == 'threshold':
            return ThresholdPruner(
                lower=config.threshold_config['lower'],
                upper=config.threshold_config['upper'],
                n_warmup_steps=config.threshold_config['n_warmup_steps'],
                **kwargs
            )
        
        elif pruner_type == 'none':
            return optuna.pruners.NopPruner()
        
        else:
            raise ValueError(f"Pruner tipo '{pruner_type}' no soportado")

class OptimizationStrategySelector:
    """Selector inteligente de estrategia de optimización"""
    
    def __init__(self, config: OptimizationStrategy):
        self.config = config
    
    def select_strategy(self, 
                       n_trials: int, 
                       timeout: Optional[int] = None,
                       problem_type: str = 'balanced',
                       space_characteristics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Seleccionar estrategia basada en restricciones y características del problema
        """
        
        # Determinar estrategia base
        if n_trials < 100:
            base_strategy = 'quick_exploration'
        elif n_trials < 300:
            base_strategy = 'balanced_optimization'
        else:
            base_strategy = 'thorough_search'
        
        # Ajustar según características del espacio
        if space_characteristics:
            continuous_ratio = space_characteristics.get('continuous_ratio', 0.5)
            n_params = space_characteristics.get('n_params', 10)
            
            if continuous_ratio > 0.8 and n_params < 15:
                base_strategy = 'continuous_optimization'
        
        # Ajustar según timeout
        if timeout and timeout < 600:  # Menos de 10 minutos
            base_strategy = 'quick_exploration'
        
        strategy = self.config.strategies[base_strategy].copy()
        
        # Override con parámetros específicos
        strategy['trials'] = n_trials
        if timeout:
            strategy['timeout'] = timeout
        
        return strategy

# Instancias globales de configuración
SAMPLER_CONFIG = SamplerConfig()
PRUNER_CONFIG = PrunerConfig()
MULTI_OBJECTIVE_CONFIG = MultiObjectiveConfig()
CONVERGENCE_CONFIG = ConvergenceConfig()
OPTIMIZATION_STRATEGY = OptimizationStrategy()

# Factories globales
SAMPLER_FACTORY = OptunaSamplerFactory()
PRUNER_FACTORY = OptunaPrunerFactory()
STRATEGY_SELECTOR = OptimizationStrategySelector(OPTIMIZATION_STRATEGY)
