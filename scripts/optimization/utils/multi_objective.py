#!/usr/bin/env python3
"""
Optimización Multi-Objetivo para Modelos de Criptomonedas
Implementa NSGA-II y métricas múltiples para optimización avanzada
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import optuna
from optuna.samplers import NSGAIISampler
import warnings
from enum import Enum

warnings.filterwarnings('ignore')

class OptimizationObjective(Enum):
    """Objetivos de optimización disponibles"""
    MAXIMIZE_AUC = "maximize_auc"
    MINIMIZE_OVERFITTING = "minimize_overfitting"
    MAXIMIZE_STABILITY = "maximize_stability"
    MINIMIZE_TRAINING_TIME = "minimize_training_time"
    MAXIMIZE_PRECISION = "maximize_precision"
    MAXIMIZE_RECALL = "maximize_recall"
    MAXIMIZE_F1 = "maximize_f1"
    MINIMIZE_FALSE_POSITIVES = "minimize_false_positives"
    MINIMIZE_FALSE_NEGATIVES = "minimize_false_negatives"

@dataclass
class MultiObjectiveConfig:
    """Configuración para optimización multi-objetivo"""
    
    # Objetivos primarios y secundarios
    primary_objectives: List[OptimizationObjective]
    secondary_objectives: List[OptimizationObjective]
    
    # Pesos para combinación de objetivos
    objective_weights: Dict[OptimizationObjective, float]
    
    # Configuración NSGA-II
    population_size: int = 50
    mutation_prob: float = 0.1
    crossover_prob: float = 0.9
    
    # Criterios de Pareto
    pareto_front_size: int = 10
    convergence_tolerance: float = 1e-6
    max_generations_without_improvement: int = 20
    
    # Preferencias del usuario
    risk_tolerance: float = 0.5  # 0 = conservador, 1 = agresivo
    stability_preference: float = 0.7  # Importancia de estabilidad
    performance_preference: float = 0.8  # Importancia de rendimiento

class MultiObjectiveOptimizer:
    """
    Optimizador multi-objetivo con NSGA-II y análisis de trade-offs
    """
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.pareto_solutions = []
        self.convergence_history = []
        self.trade_off_analysis = {}
        
    def create_nsga2_sampler(self) -> NSGAIISampler:
        """Crear sampler NSGA-II configurado"""
        return NSGAIISampler(
            population_size=self.config.population_size,
            mutation_prob=self.config.mutation_prob,
            crossover_prob=self.config.crossover_prob
        )
    
    def create_multi_objective_study(self, study_name: str, storage_url: str = None) -> optuna.Study:
        """
        Crear estudio multi-objetivo con NSGA-II
        
        Args:
            study_name: Nombre del estudio
            storage_url: URL de almacenamiento (opcional)
            
        Returns:
            Estudio Optuna configurado para multi-objetivo
        """
        # Configurar direcciones según objetivos
        directions = []
        for obj in self.config.primary_objectives:
            if "maximize" in obj.value:
                directions.append("maximize")
            else:
                directions.append("minimize")
        
        # Crear sampler NSGA-II
        sampler = self.create_nsga2_sampler()
        
        # Configurar estudio
        study_kwargs = {
            'directions': directions,
            'study_name': study_name,
            'sampler': sampler
        }
        
        if storage_url:
            study_kwargs['storage'] = storage_url
            study_kwargs['load_if_exists'] = True
        
        return optuna.create_study(**study_kwargs)
    
    def calculate_objective_values(self, 
                                  trial_results: Dict[str, Any],
                                  model_performance: Dict[str, float],
                                  training_time: float) -> List[float]:
        """
        Calcular valores de objetivos para un trial
        
        Args:
            trial_results: Resultados del trial
            model_performance: Métricas de rendimiento del modelo
            training_time: Tiempo de entrenamiento
            
        Returns:
            Lista de valores de objetivos
        """
        objective_values = []
        
        for obj in self.config.primary_objectives:
            value = self._calculate_single_objective(
                obj, trial_results, model_performance, training_time
            )
            objective_values.append(value)
        
        return objective_values
    
    def _calculate_single_objective(self,
                                   objective: OptimizationObjective,
                                   trial_results: Dict[str, Any],
                                   model_performance: Dict[str, float],
                                   training_time: float) -> float:
        """Calcular valor de un objetivo específico"""
        
        if objective == OptimizationObjective.MAXIMIZE_AUC:
            return model_performance.get('auc', 0.0)
        
        elif objective == OptimizationObjective.MINIMIZE_OVERFITTING:
            train_auc = trial_results.get('train_auc', 0.0)
            val_auc = model_performance.get('auc', 0.0)
            return abs(train_auc - val_auc)  # Diferencia entre train y val
        
        elif objective == OptimizationObjective.MAXIMIZE_STABILITY:
            cv_scores = trial_results.get('cv_scores', [])
            if len(cv_scores) > 1:
                return 1.0 - np.std(cv_scores)  # Menor desviación = mayor estabilidad
            return 0.0
        
        elif objective == OptimizationObjective.MINIMIZE_TRAINING_TIME:
            return training_time
        
        elif objective == OptimizationObjective.MAXIMIZE_PRECISION:
            return model_performance.get('precision', 0.0)
        
        elif objective == OptimizationObjective.MAXIMIZE_RECALL:
            return model_performance.get('recall', 0.0)
        
        elif objective == OptimizationObjective.MAXIMIZE_F1:
            return model_performance.get('f1', 0.0)
        
        elif objective == OptimizationObjective.MINIMIZE_FALSE_POSITIVES:
            return model_performance.get('false_positive_rate', 1.0)
        
        elif objective == OptimizationObjective.MINIMIZE_FALSE_NEGATIVES:
            return model_performance.get('false_negative_rate', 1.0)
        
        return 0.0
    
    def analyze_pareto_front(self, study: optuna.Study) -> Dict[str, Any]:
        """
        Analizar el frente de Pareto y encontrar soluciones óptimas
        
        Args:
            study: Estudio Optuna completado
            
        Returns:
            Análisis del frente de Pareto
        """
        if not study.trials:
            return {'error': 'No hay trials disponibles'}
        
        # Obtener trials no podados
        completed_trials = [trial for trial in study.trials 
                          if trial.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            return {'error': 'No hay trials completados'}
        
        # Extraer valores de objetivos
        objective_values = []
        trial_params = []
        trial_numbers = []
        
        for trial in completed_trials:
            if trial.values:
                objective_values.append(trial.values)
                trial_params.append(trial.params)
                trial_numbers.append(trial.number)
        
        if not objective_values:
            return {'error': 'No hay valores de objetivos disponibles'}
        
        objective_values = np.array(objective_values)
        
        # Encontrar frente de Pareto
        pareto_indices = self._find_pareto_front(objective_values)
        
        # Seleccionar mejores soluciones del frente de Pareto
        pareto_solutions = []
        for idx in pareto_indices[:self.config.pareto_front_size]:
            pareto_solutions.append({
                'trial_number': trial_numbers[idx],
                'params': trial_params[idx],
                'objectives': objective_values[idx].tolist(),
                'dominance_count': len([i for i in pareto_indices if i != idx])
            })
        
        # Análisis de trade-offs
        trade_offs = self._analyze_trade_offs(objective_values[pareto_indices])
        
        # Recomendar solución basada en preferencias
        recommended_solution = self._recommend_solution(pareto_solutions)
        
        return {
            'pareto_front_size': len(pareto_indices),
            'pareto_solutions': pareto_solutions,
            'trade_offs': trade_offs,
            'recommended_solution': recommended_solution,
            'convergence_metrics': self._calculate_convergence_metrics(objective_values),
            'diversity_metrics': self._calculate_diversity_metrics(objective_values[pareto_indices])
        }
    
    def _find_pareto_front(self, objective_values: np.ndarray) -> List[int]:
        """Encontrar índices del frente de Pareto"""
        n_points = len(objective_values)
        pareto_indices = []
        
        # Ajustar direcciones (convertir minimización a maximización)
        adjusted_values = objective_values.copy()
        for i, obj in enumerate(self.config.primary_objectives):
            if "minimize" in obj.value:
                adjusted_values[:, i] = -adjusted_values[:, i]
        
        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i != j:
                    # Verificar si j domina a i
                    if all(adjusted_values[j] >= adjusted_values[i]) and \
                       any(adjusted_values[j] > adjusted_values[i]):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _analyze_trade_offs(self, pareto_values: np.ndarray) -> Dict[str, Any]:
        """Analizar trade-offs entre objetivos"""
        if len(pareto_values) < 2:
            return {'error': 'Insuficientes puntos en el frente de Pareto'}
        
        n_objectives = len(self.config.primary_objectives)
        correlations = np.corrcoef(pareto_values.T)
        
        trade_offs = {}
        for i in range(n_objectives):
            for j in range(i + 1, n_objectives):
                obj1 = self.config.primary_objectives[i].value
                obj2 = self.config.primary_objectives[j].value
                correlation = correlations[i, j]
                
                trade_offs[f"{obj1}_vs_{obj2}"] = {
                    'correlation': correlation,
                    'trade_off_strength': abs(correlation),
                    'relationship': 'negative' if correlation < -0.1 else 'positive' if correlation > 0.1 else 'neutral'
                }
        
        return trade_offs
    
    def _recommend_solution(self, pareto_solutions: List[Dict]) -> Dict[str, Any]:
        """Recomendar solución basada en preferencias del usuario"""
        if not pareto_solutions:
            return {'error': 'No hay soluciones en el frente de Pareto'}
        
        # Calcular scores basados en preferencias
        best_solution = None
        best_score = -np.inf
        
        for solution in pareto_solutions:
            score = self._calculate_preference_score(solution['objectives'])
            
            if score > best_score:
                best_score = score
                best_solution = solution
        
        if best_solution:
            best_solution['preference_score'] = best_score
            best_solution['recommendation_reason'] = self._get_recommendation_reason(best_solution)
        
        return best_solution or pareto_solutions[0]
    
    def _calculate_preference_score(self, objective_values: List[float]) -> float:
        """Calcular score basado en preferencias del usuario"""
        score = 0.0
        
        for i, (obj, value) in enumerate(zip(self.config.primary_objectives, objective_values)):
            weight = self.config.objective_weights.get(obj, 1.0)
            
            # Normalizar valor (asumiendo rango 0-1 para la mayoría de métricas)
            normalized_value = min(max(value, 0.0), 1.0)
            
            # Ajustar por preferencias
            if obj == OptimizationObjective.MAXIMIZE_STABILITY:
                normalized_value *= self.config.stability_preference
            elif obj in [OptimizationObjective.MAXIMIZE_AUC, OptimizationObjective.MAXIMIZE_F1]:
                normalized_value *= self.config.performance_preference
            
            score += weight * normalized_value
        
        return score
    
    def _get_recommendation_reason(self, solution: Dict) -> str:
        """Generar razón para la recomendación"""
        objectives = solution['objectives']
        reasons = []
        
        for i, (obj, value) in enumerate(zip(self.config.primary_objectives, objectives)):
            if obj == OptimizationObjective.MAXIMIZE_AUC and value > 0.8:
                reasons.append(f"Alto AUC ({value:.3f})")
            elif obj == OptimizationObjective.MAXIMIZE_STABILITY and value > 0.9:
                reasons.append(f"Alta estabilidad ({value:.3f})")
            elif obj == OptimizationObjective.MINIMIZE_OVERFITTING and value < 0.1:
                reasons.append(f"Bajo overfitting ({value:.3f})")
        
        return "; ".join(reasons) if reasons else "Mejor balance general"
    
    def _calculate_convergence_metrics(self, objective_values: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de convergencia"""
        if len(objective_values) < 10:
            return {'error': 'Insuficientes datos para análisis de convergencia'}
        
        # Hypervolume approximation
        reference_point = np.min(objective_values, axis=0) - 0.1
        hypervolume = self._approximate_hypervolume(objective_values, reference_point)
        
        return {
            'hypervolume': hypervolume,
            'spread': np.mean(np.std(objective_values, axis=0)),
            'uniformity': self._calculate_uniformity(objective_values)
        }
    
    def _calculate_diversity_metrics(self, pareto_values: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de diversidad del frente de Pareto"""
        if len(pareto_values) < 2:
            return {'diversity': 0.0}
        
        # Calcular distancias entre puntos
        distances = []
        for i in range(len(pareto_values)):
            for j in range(i + 1, len(pareto_values)):
                dist = np.linalg.norm(pareto_values[i] - pareto_values[j])
                distances.append(dist)
        
        return {
            'diversity': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'distance_std': np.std(distances)
        }
    
    def _approximate_hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        """Aproximación simple del hypervolume"""
        # Implementación simplificada - en producción usar pygmo o deap
        volumes = []
        for point in points:
            volume = np.prod(np.maximum(point - reference_point, 0))
            volumes.append(volume)
        return np.sum(volumes)
    
    def _calculate_uniformity(self, objective_values: np.ndarray) -> float:
        """Calcular uniformidad de la distribución"""
        if len(objective_values) < 3:
            return 0.0
        
        # Calcular distancias al vecino más cercano
        nearest_distances = []
        for i, point in enumerate(objective_values):
            distances = [np.linalg.norm(point - other) 
                        for j, other in enumerate(objective_values) if i != j]
            if distances:
                nearest_distances.append(min(distances))
        
        if not nearest_distances:
            return 0.0
        
        # Uniformidad basada en la desviación estándar de las distancias
        return 1.0 / (1.0 + np.std(nearest_distances))

# Configuraciones predefinidas
DEFAULT_MULTI_OBJECTIVE_CONFIG = MultiObjectiveConfig(
    primary_objectives=[
        OptimizationObjective.MAXIMIZE_AUC,
        OptimizationObjective.MINIMIZE_OVERFITTING,
        OptimizationObjective.MAXIMIZE_STABILITY
    ],
    secondary_objectives=[
        OptimizationObjective.MINIMIZE_TRAINING_TIME,
        OptimizationObjective.MAXIMIZE_F1
    ],
    objective_weights={
        OptimizationObjective.MAXIMIZE_AUC: 1.0,
        OptimizationObjective.MINIMIZE_OVERFITTING: 0.8,
        OptimizationObjective.MAXIMIZE_STABILITY: 0.7,
        OptimizationObjective.MINIMIZE_TRAINING_TIME: 0.3,
        OptimizationObjective.MAXIMIZE_F1: 0.6
    }
)

# Instancia global
MULTI_OBJECTIVE_OPTIMIZER = MultiObjectiveOptimizer(DEFAULT_MULTI_OBJECTIVE_CONFIG)

if __name__ == "__main__":
    # Test básico
    print("🎯 Multi-Objective Optimizer inicializado")
    print(f"   📊 Objetivos primarios: {len(DEFAULT_MULTI_OBJECTIVE_CONFIG.primary_objectives)}")
    print(f"   📈 Objetivos secundarios: {len(DEFAULT_MULTI_OBJECTIVE_CONFIG.secondary_objectives)}")
    print(f"   🧬 Población NSGA-II: {DEFAULT_MULTI_OBJECTIVE_CONFIG.population_size}")
