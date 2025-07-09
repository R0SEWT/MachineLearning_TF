#!/usr/bin/env python3
"""
Sistema de early stopping inteligente y detección de convergencia
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import time
import math

logger = logging.getLogger(__name__)

@dataclass
class EarlyStoppingConfig:
    """Configuración para early stopping"""
    
    # Configuración básica
    patience: int = 20                    # Trials sin mejora
    min_improvement: float = 0.001        # Mejora mínima considerada
    warmup_trials: int = 30              # Trials antes de considerar stopping
    
    # Configuración adaptativa
    adaptive_patience: bool = True        # Paciencia adaptativa
    patience_factor: float = 1.5         # Factor de incremento de paciencia
    max_patience: int = 50               # Máxima paciencia
    
    # Configuración de convergencia
    convergence_window: int = 10         # Ventana para detectar convergencia
    convergence_threshold: float = 0.995 # Umbral de convergencia (R²)
    plateau_threshold: float = 0.0005    # Umbral para detectar plateau
    
    # Configuración de estabilidad
    stability_window: int = 15           # Ventana para calcular estabilidad
    min_stability: float = 0.8           # Estabilidad mínima requerida
    instability_patience: int = 10       # Paciencia para inestabilidad

@dataclass
class ConvergenceMetrics:
    """Métricas de convergencia"""
    
    is_converged: bool = False
    convergence_score: float = 0.0
    plateau_detected: bool = False
    stability_score: float = 0.0
    trend_score: float = 0.0
    n_trials_since_improvement: int = 0
    best_score: float = -np.inf
    best_trial: int = -1

class EarlyStoppingMonitor:
    """Monitor de early stopping con capacidades avanzadas"""
    
    def __init__(self, config: EarlyStoppingConfig = None):
        self.config = config or EarlyStoppingConfig()
        self.reset()
    
    def reset(self):
        """Resetear el monitor"""
        self.trial_scores = []
        self.trial_times = []
        self.best_score = -np.inf
        self.best_trial = -1
        self.trials_since_improvement = 0
        self.current_patience = self.config.patience
        self.convergence_history = deque(maxlen=self.config.convergence_window)
        self.stability_history = deque(maxlen=self.config.stability_window)
        self.stopped = False
        self.stop_reason = None
        self.start_time = time.time()
    
    def update(self, trial_number: int, score: float, additional_metrics: Dict[str, float] = None) -> bool:
        """
        Actualizar monitor con nuevo score y determinar si debe parar
        
        Returns:
            True si debe parar, False si debe continuar
        """
        current_time = time.time()
        self.trial_scores.append(score)
        self.trial_times.append(current_time)
        
        # Actualizar mejor score
        if score > self.best_score:
            self.best_score = score
            self.best_trial = trial_number
            self.trials_since_improvement = 0
            
            # Adaptar paciencia si está habilitado
            if self.config.adaptive_patience:
                self._adapt_patience()
        else:
            self.trials_since_improvement += 1
        
        # Esperar warmup
        if trial_number < self.config.warmup_trials:
            return False
        
        # Verificar diferentes criterios de stopping
        should_stop, reason = self._check_stopping_criteria(trial_number)
        
        if should_stop:
            self.stopped = True
            self.stop_reason = reason
            logger.info(f"Early stopping activado: {reason}")
        
        return should_stop
    
    def _adapt_patience(self):
        """Adaptar paciencia basado en mejoras"""
        if self.config.adaptive_patience:
            self.current_patience = min(
                int(self.current_patience * self.config.patience_factor),
                self.config.max_patience
            )
            logger.debug(f"Paciencia adaptada a: {self.current_patience}")
    
    def _check_stopping_criteria(self, trial_number: int) -> Tuple[bool, Optional[str]]:
        """Verificar todos los criterios de stopping"""
        
        # 1. Patience básico
        if self.trials_since_improvement >= self.current_patience:
            return True, f"Patience excedido: {self.trials_since_improvement} trials sin mejora"
        
        # 2. Convergencia
        convergence_metrics = self.calculate_convergence_metrics()
        if convergence_metrics.is_converged:
            return True, f"Convergencia detectada: score={convergence_metrics.convergence_score:.4f}"
        
        # 3. Plateau
        if convergence_metrics.plateau_detected:
            return True, f"Plateau detectado: estabilidad={convergence_metrics.stability_score:.4f}"
        
        # 4. Inestabilidad
        if (convergence_metrics.stability_score < self.config.min_stability and 
            trial_number > self.config.warmup_trials + self.config.instability_patience):
            return True, f"Inestabilidad detectada: {convergence_metrics.stability_score:.4f}"
        
        return False, None
    
    def calculate_convergence_metrics(self) -> ConvergenceMetrics:
        """Calcular métricas de convergencia"""
        metrics = ConvergenceMetrics()
        
        if len(self.trial_scores) < self.config.convergence_window:
            return metrics
        
        recent_scores = self.trial_scores[-self.config.convergence_window:]
        
        # 1. Detectar convergencia usando R²
        convergence_score = self._calculate_convergence_score(recent_scores)
        metrics.convergence_score = convergence_score
        metrics.is_converged = convergence_score > self.config.convergence_threshold
        
        # 2. Detectar plateau
        plateau_detected = self._detect_plateau(recent_scores)
        metrics.plateau_detected = plateau_detected
        
        # 3. Calcular estabilidad
        stability_score = self._calculate_stability_score()
        metrics.stability_score = stability_score
        
        # 4. Calcular tendencia
        trend_score = self._calculate_trend_score(recent_scores)
        metrics.trend_score = trend_score
        
        # Actualizar información general
        metrics.n_trials_since_improvement = self.trials_since_improvement
        metrics.best_score = self.best_score
        metrics.best_trial = self.best_trial
        
        return metrics
    
    def _calculate_convergence_score(self, scores: List[float]) -> float:
        """Calcular score de convergencia usando R²"""
        if len(scores) < 3:
            return 0.0
        
        try:
            # Usar regresión lineal para detectar convergencia
            x = np.arange(len(scores))
            y = np.array(scores)
            
            # Calcular R²
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            
            # Fit línea
            slope = np.sum((x - np.mean(x)) * (y - y_mean)) / np.sum((x - np.mean(x)) ** 2)
            intercept = y_mean - slope * np.mean(x)
            y_pred = slope * x + intercept
            
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Convergencia si R² es alto y slope es bajo
            slope_factor = 1 / (1 + abs(slope) * 1000)  # Penalizar pendiente alta
            
            return r_squared * slope_factor
            
        except Exception:
            return 0.0
    
    def _detect_plateau(self, scores: List[float]) -> bool:
        """Detectar plateau en los scores"""
        if len(scores) < self.config.convergence_window:
            return False
        
        # Calcular varianza de los scores recientes
        variance = np.var(scores)
        
        # Comparar con threshold
        return variance < self.config.plateau_threshold
    
    def _calculate_stability_score(self) -> float:
        """Calcular score de estabilidad"""
        if len(self.trial_scores) < self.config.stability_window:
            return 1.0
        
        recent_scores = self.trial_scores[-self.config.stability_window:]
        
        # Coeficiente de variación
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        if mean_score <= 0:
            return 0.0
        
        cv = std_score / mean_score
        stability = 1 / (1 + cv)
        
        return stability
    
    def _calculate_trend_score(self, scores: List[float]) -> float:
        """Calcular score de tendencia (positiva, negativa, plana)"""
        if len(scores) < 3:
            return 0.0
        
        # Calcular pendiente de regresión lineal
        x = np.arange(len(scores))
        y = np.array(scores)
        
        slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
        
        # Normalizar slope
        return np.tanh(slope * 100)  # Tanh para mantener en [-1, 1]
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen del monitor"""
        duration = time.time() - self.start_time
        convergence_metrics = self.calculate_convergence_metrics()
        
        return {
            'stopped': self.stopped,
            'stop_reason': self.stop_reason,
            'best_score': self.best_score,
            'best_trial': self.best_trial,
            'total_trials': len(self.trial_scores),
            'trials_since_improvement': self.trials_since_improvement,
            'current_patience': self.current_patience,
            'duration_seconds': duration,
            'convergence_metrics': {
                'is_converged': convergence_metrics.is_converged,
                'convergence_score': convergence_metrics.convergence_score,
                'plateau_detected': convergence_metrics.plateau_detected,
                'stability_score': convergence_metrics.stability_score,
                'trend_score': convergence_metrics.trend_score
            }
        }

class TrialPruner:
    """Pruner personalizado para trials individuales"""
    
    def __init__(self, config: EarlyStoppingConfig = None):
        self.config = config or EarlyStoppingConfig()
    
    def should_prune(self, trial_scores: List[float], current_score: float,
                    trial_number: int) -> Tuple[bool, str]:
        """
        Determinar si un trial debe ser podado
        
        Args:
            trial_scores: Scores de trials anteriores
            current_score: Score actual del trial
            trial_number: Número del trial actual
            
        Returns:
            (should_prune, reason)
        """
        
        if trial_number < self.config.warmup_trials:
            return False, "Warmup period"
        
        if len(trial_scores) < 5:
            return False, "Insufficient history"
        
        # 1. Pruning por percentil
        percentile_25 = np.percentile(trial_scores, 25)
        if current_score < percentile_25 * 0.8:  # 20% por debajo del Q1
            return True, f"Score muy bajo: {current_score:.4f} < {percentile_25*0.8:.4f}"
        
        # 2. Pruning por trending
        recent_best = max(trial_scores[-10:]) if len(trial_scores) >= 10 else max(trial_scores)
        if current_score < recent_best * 0.9:  # 10% por debajo del mejor reciente
            return True, f"Score por debajo del trend: {current_score:.4f} < {recent_best*0.9:.4f}"
        
        return False, "No pruning"

class AdaptiveOptimizationController:
    """Controlador adaptativo para toda la optimización"""
    
    def __init__(self, config: EarlyStoppingConfig = None):
        self.config = config or EarlyStoppingConfig()
        self.monitors = {}  # Un monitor por modelo
        self.pruner = TrialPruner(config)
        self.global_start_time = time.time()
    
    def get_monitor(self, model_name: str) -> EarlyStoppingMonitor:
        """Obtener monitor para un modelo específico"""
        if model_name not in self.monitors:
            self.monitors[model_name] = EarlyStoppingMonitor(self.config)
        return self.monitors[model_name]
    
    def should_stop_model(self, model_name: str, trial_number: int, 
                         score: float, additional_metrics: Dict[str, float] = None) -> bool:
        """Verificar si debe parar la optimización de un modelo"""
        monitor = self.get_monitor(model_name)
        return monitor.update(trial_number, score, additional_metrics)
    
    def should_prune_trial(self, model_name: str, trial_scores: List[float], 
                          current_score: float, trial_number: int) -> Tuple[bool, str]:
        """Verificar si debe podar un trial específico"""
        return self.pruner.should_prune(trial_scores, current_score, trial_number)
    
    def get_global_summary(self) -> Dict[str, Any]:
        """Obtener resumen global de la optimización"""
        total_duration = time.time() - self.global_start_time
        
        summary = {
            'total_duration_seconds': total_duration,
            'models': {}
        }
        
        for model_name, monitor in self.monitors.items():
            summary['models'][model_name] = monitor.get_summary()
        
        # Estadísticas globales
        all_trials = sum(len(m.trial_scores) for m in self.monitors.values())
        best_scores = {name: m.best_score for name, m in self.monitors.items()}
        
        summary['global_stats'] = {
            'total_trials': all_trials,
            'best_scores': best_scores,
            'best_overall': max(best_scores.values()) if best_scores else 0.0
        }
        
        return summary
    
    def adapt_strategy(self, model_name: str, current_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Adaptar estrategia de optimización basada en performance actual
        """
        monitor = self.get_monitor(model_name)
        convergence_metrics = monitor.calculate_convergence_metrics()
        
        recommendations = {
            'continue_optimization': True,
            'suggested_trials': self.config.patience,
            'suggested_patience': self.config.patience,
            'strategy_changes': []
        }
        
        # Adaptar basado en convergencia
        if convergence_metrics.convergence_score > 0.8:
            recommendations['suggested_trials'] = max(10, self.config.patience // 2)
            recommendations['strategy_changes'].append("Reducir trials por alta convergencia")
        
        # Adaptar basado en estabilidad
        if convergence_metrics.stability_score < 0.5:
            recommendations['suggested_patience'] = int(self.config.patience * 1.5)
            recommendations['strategy_changes'].append("Aumentar patience por inestabilidad")
        
        # Adaptar basado en tendencia
        if abs(convergence_metrics.trend_score) < 0.1:  # Tendencia plana
            recommendations['suggested_trials'] = max(5, self.config.patience // 3)
            recommendations['strategy_changes'].append("Reducir trials por tendencia plana")
        
        return recommendations

# Instancia global del controlador
ADAPTIVE_CONTROLLER = AdaptiveOptimizationController()
