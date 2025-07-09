#!/usr/bin/env python3
"""
Calculadora de métricas múltiples para evaluación completa de modelos
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MetricsResult:
    """Resultado de cálculo de métricas"""
    primary_score: float
    secondary_scores: Dict[str, float]
    composite_score: float
    metadata: Dict[str, Any]

class MetricsCalculator:
    """Calculadora avanzada de métricas para modelos de ML"""
    
    def __init__(self, primary_metric: str = "roc_auc"):
        self.primary_metric = primary_metric
        self.metrics_registry = self._build_metrics_registry()
    
    def _build_metrics_registry(self) -> Dict[str, callable]:
        """Construir registro de métricas disponibles"""
        return {
            # Métricas de clasificación estándar
            'roc_auc': self._calculate_roc_auc,
            'precision': self._calculate_precision,
            'recall': self._calculate_recall,
            'f1': self._calculate_f1,
            'accuracy': self._calculate_accuracy,
            'balanced_accuracy': self._calculate_balanced_accuracy,
            'log_loss': self._calculate_log_loss,
            
            # Métricas específicas de trading
            'sharpe_ratio': self._calculate_sharpe_ratio,
            'max_drawdown': self._calculate_max_drawdown,
            'profit_factor': self._calculate_profit_factor,
            'win_rate': self._calculate_win_rate,
            'avg_win_loss_ratio': self._calculate_avg_win_loss_ratio,
            
            # Métricas de estabilidad
            'stability_score': self._calculate_stability_score,
            'consistency_score': self._calculate_consistency_score,
            'volatility_score': self._calculate_volatility_score
        }
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_proba: Optional[np.ndarray] = None,
                            cv_scores: Optional[List[float]] = None,
                            returns_data: Optional[np.ndarray] = None,
                            metrics_to_calculate: Optional[List[str]] = None) -> MetricsResult:
        """
        Calcular todas las métricas especificadas
        
        Args:
            y_true: Valores reales
            y_pred: Predicciones binarias
            y_proba: Probabilidades (para métricas que las requieren)
            cv_scores: Scores de cross-validation para métricas de estabilidad
            returns_data: Datos de retornos para métricas de trading
            metrics_to_calculate: Lista de métricas a calcular
        """
        if metrics_to_calculate is None:
            metrics_to_calculate = [
                'roc_auc', 'precision', 'recall', 'f1', 'accuracy',
                'sharpe_ratio', 'stability_score'
            ]
        
        # Validar entradas
        self._validate_inputs(y_true, y_pred, y_proba)
        
        # Calcular métricas
        scores = {}
        metadata = {
            'n_samples': len(y_true),
            'n_positive': np.sum(y_true),
            'n_negative': np.sum(1 - y_true),
            'positive_rate': np.mean(y_true)
        }
        
        for metric_name in metrics_to_calculate:
            if metric_name in self.metrics_registry:
                try:
                    score = self._calculate_metric(
                        metric_name, y_true, y_pred, y_proba, 
                        cv_scores, returns_data
                    )
                    scores[metric_name] = score
                    logger.debug(f"Métrica {metric_name}: {score:.4f}")
                except Exception as e:
                    logger.warning(f"Error calculando métrica {metric_name}: {e}")
                    scores[metric_name] = np.nan
            else:
                logger.warning(f"Métrica desconocida: {metric_name}")
        
        # Calcular score primario
        primary_score = scores.get(self.primary_metric, np.nan)
        
        # Calcular score compuesto
        composite_score = self._calculate_composite_score(scores)
        
        return MetricsResult(
            primary_score=primary_score,
            secondary_scores=scores,
            composite_score=composite_score,
            metadata=metadata
        )
    
    def _calculate_metric(self, metric_name: str, y_true: np.ndarray, 
                         y_pred: np.ndarray, y_proba: Optional[np.ndarray],
                         cv_scores: Optional[List[float]], 
                         returns_data: Optional[np.ndarray]) -> float:
        """Calcular una métrica específica"""
        metric_func = self.metrics_registry[metric_name]
        
        # Determinar qué argumentos necesita la función
        if metric_name in ['roc_auc', 'log_loss']:
            if y_proba is None:
                raise ValueError(f"Métrica {metric_name} requiere probabilidades")
            return metric_func(y_true, y_proba)
        elif metric_name in ['sharpe_ratio', 'max_drawdown', 'profit_factor', 
                           'win_rate', 'avg_win_loss_ratio']:
            if returns_data is None:
                logger.warning(f"Métrica {metric_name} requiere datos de retornos, usando predicciones")
                return metric_func(y_true, y_pred)
            return metric_func(y_true, returns_data)
        elif metric_name in ['stability_score', 'consistency_score']:
            if cv_scores is None:
                logger.warning(f"Métrica {metric_name} requiere CV scores")
                return np.nan
            return metric_func(cv_scores)
        else:
            return metric_func(y_true, y_pred)
    
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_proba: Optional[np.ndarray]):
        """Validar entradas de métricas"""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true y y_pred deben tener la misma longitud")
        
        if y_proba is not None and len(y_true) != len(y_proba):
            raise ValueError("y_true y y_proba deben tener la misma longitud")
        
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true debe contener solo 0s y 1s")
        
        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("y_pred debe contener solo 0s y 1s")
    
    # ==================== MÉTRICAS DE CLASIFICACIÓN ====================
    
    def _calculate_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calcular ROC AUC"""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_proba)
        except ImportError:
            # Implementación manual básica
            return self._manual_roc_auc(y_true, y_proba)
    
    def _manual_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Implementación manual de ROC AUC"""
        # Ordenar por probabilidad descendente
        order = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[order]
        
        # Calcular TPR y FPR
        tpr = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
        fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)
        
        # Calcular AUC usando regla del trapecio
        return np.trapz(tpr, fpr)
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular Precision"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular Recall (Sensitivity)"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular F1-Score"""
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular Accuracy"""
        return np.mean(y_true == y_pred)
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular Balanced Accuracy"""
        sensitivity = self._calculate_recall(y_true, y_pred)
        specificity = self._calculate_specificity(y_true, y_pred)
        return (sensitivity + specificity) / 2
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular Specificity"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_log_loss(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calcular Log Loss"""
        # Evitar log(0)
        epsilon = 1e-15
        y_proba = np.clip(y_proba, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba))
    
    # ==================== MÉTRICAS DE TRADING ====================
    
    def _calculate_sharpe_ratio(self, y_true: np.ndarray, returns: np.ndarray,
                               risk_free_rate: float = 0.0) -> float:
        """Calcular Sharpe Ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Anualizado
    
    def _calculate_max_drawdown(self, y_true: np.ndarray, returns: np.ndarray) -> float:
        """Calcular Maximum Drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_profit_factor(self, y_true: np.ndarray, returns: np.ndarray) -> float:
        """Calcular Profit Factor"""
        if len(returns) == 0:
            return 0.0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf') if len(positive_returns) > 0 else 0.0
        
        gross_profit = np.sum(positive_returns)
        gross_loss = abs(np.sum(negative_returns))
        
        return gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    def _calculate_win_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcular Win Rate (porcentaje de predicciones correctas positivas)"""
        positive_predictions = np.sum(y_pred == 1)
        if positive_predictions == 0:
            return 0.0
        
        correct_positive = np.sum((y_true == 1) & (y_pred == 1))
        return correct_positive / positive_predictions
    
    def _calculate_avg_win_loss_ratio(self, y_true: np.ndarray, returns: np.ndarray) -> float:
        """Calcular ratio promedio win/loss"""
        if len(returns) == 0:
            return 0.0
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        return avg_win / avg_loss if avg_loss > 0 else 0.0
    
    # ==================== MÉTRICAS DE ESTABILIDAD ====================
    
    def _calculate_stability_score(self, cv_scores: List[float]) -> float:
        """Calcular score de estabilidad basado en varianza de CV"""
        if len(cv_scores) < 2:
            return 0.0
        
        cv_std = np.std(cv_scores)
        cv_mean = np.mean(cv_scores)
        
        # Score inverso a la variabilidad relativa
        coefficient_of_variation = cv_std / cv_mean if cv_mean > 0 else float('inf')
        stability = 1 / (1 + coefficient_of_variation)
        
        return stability
    
    def _calculate_consistency_score(self, cv_scores: List[float]) -> float:
        """Calcular score de consistencia"""
        if len(cv_scores) < 2:
            return 0.0
        
        # Porcentaje de scores que están dentro de 1 desviación estándar de la media
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        within_std = np.sum(np.abs(np.array(cv_scores) - mean_score) <= std_score)
        return within_std / len(cv_scores)
    
    def _calculate_volatility_score(self, cv_scores: List[float]) -> float:
        """Calcular score de volatilidad (inverso de la desviación estándar)"""
        if len(cv_scores) < 2:
            return 0.0
        
        volatility = np.std(cv_scores)
        return 1 / (1 + volatility)  # Score inverso a la volatilidad
    
    # ==================== SCORE COMPUESTO ====================
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calcular score compuesto ponderado"""
        # Pesos para diferentes tipos de métricas
        weights = {
            'roc_auc': 0.3,
            'f1': 0.2,
            'precision': 0.15,
            'recall': 0.15,
            'sharpe_ratio': 0.1,
            'stability_score': 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            if metric in weights and not np.isnan(score):
                # Normalizar scores negativos (como max_drawdown)
                if metric in ['max_drawdown', 'log_loss']:
                    normalized_score = 1 / (1 + abs(score))
                else:
                    normalized_score = score
                
                weighted_sum += weights[metric] * normalized_score
                total_weight += weights[metric]
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_metrics_summary(self, results: MetricsResult) -> str:
        """Generar resumen de métricas en texto"""
        summary = f"📊 RESUMEN DE MÉTRICAS\n"
        summary += f"{'='*50}\n"
        summary += f"Score Primario ({self.primary_metric}): {results.primary_score:.4f}\n"
        summary += f"Score Compuesto: {results.composite_score:.4f}\n\n"
        
        summary += f"📈 MÉTRICAS DETALLADAS:\n"
        for metric, score in results.secondary_scores.items():
            if not np.isnan(score):
                summary += f"   {metric:20}: {score:.4f}\n"
        
        summary += f"\n📋 METADATA:\n"
        for key, value in results.metadata.items():
            summary += f"   {key:20}: {value}\n"
        
        return summary
