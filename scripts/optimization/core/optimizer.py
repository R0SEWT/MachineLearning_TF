"""
🚀 Optimizador Principal Refactorizado - Fase 5
===============================================

Optimizador enterprise-ready que centraliza y mejora toda la lógica de
optimización de hiperparámetros que estaba dispersa en el sistema anterior.

Autor: Sistema de Optimización IA
Fecha: 2025-01-09 (Fase 5 - Organización)
"""

import time
import gc
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Local imports
from ..config.optimization_config import OptimizationConfig, get_config
from ..utils.logging_setup import get_logger, OptimizationLogger, log_execution_time
from ..utils.import_manager import safe_import, get_ml_libraries
from .data_manager import DataManager, DataInfo

# Safe imports
optuna = safe_import("optuna")
sklearn_metrics = safe_import("sklearn.metrics")
sklearn_model_selection = safe_import("sklearn.model_selection")


@dataclass
class OptimizationResult:
    """Resultado de una optimización"""
    model_name: str
    best_score: float
    best_params: Dict[str, Any]
    n_trials: int
    optimization_time: float
    cv_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None
    model_object: Optional[Any] = None
    study_object: Optional[Any] = None


@dataclass
class ExperimentResult:
    """Resultado completo de un experimento"""
    experiment_id: str
    config: OptimizationConfig
    data_info: DataInfo
    model_results: Dict[str, OptimizationResult] = field(default_factory=dict)
    total_time: float = 0.0
    best_model: Optional[str] = None
    best_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ModelHandler:
    """Handler base para modelos de ML"""
    
    def __init__(self, model_name: str, config: OptimizationConfig):
        """
        Inicializar handler de modelo.
        
        Args:
            model_name: Nombre del modelo
            config: Configuración de optimización
        """
        self.model_name = model_name
        self.config = config
        self.logger = get_logger(f"model_{model_name}")
    
    def create_model(self, params: Dict[str, Any]) -> Any:
        """Crear instancia del modelo con parámetros dados"""
        raise NotImplementedError("Subclases deben implementar create_model")
    
    def get_param_space(self) -> Dict[str, Any]:
        """Obtener espacio de parámetros para optimización"""
        raise NotImplementedError("Subclases deben implementar get_param_space")
    
    def extract_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extraer importancia de features del modelo"""
        return None


class XGBoostHandler(ModelHandler):
    """Handler para XGBoost"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__("xgboost", config)
        self.xgb = safe_import("xgboost")
        if self.xgb is None:
            raise ImportError("XGBoost no disponible")
    
    def create_model(self, params: Dict[str, Any]) -> Any:
        """Crear modelo XGBoost"""
        model_config = self.config.get_model_config("xgboost")
        
        # Combinar parámetros de optimización con configuración base
        final_params = {**model_config, **params}
        
        return self.xgb.XGBClassifier(**final_params)
    
    def get_param_space(self) -> Dict[str, Any]:
        """Espacio de parámetros para XGBoost"""
        return {
            'n_estimators': ('int', 100, 1000),
            'max_depth': ('int', 3, 12),
            'learning_rate': ('float', 0.01, 0.3, True),  # log=True
            'subsample': ('float', 0.6, 1.0),
            'colsample_bytree': ('float', 0.6, 1.0),
            'reg_alpha': ('float', 0, 10),
            'reg_lambda': ('float', 0, 10),
            'min_child_weight': ('int', 1, 10),
            'gamma': ('float', 0, 5)
        }
    
    def extract_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extraer importancia de features de XGBoost"""
        try:
            importance = model.feature_importances_
            feature_names = getattr(model, 'feature_names_in_', 
                                  [f"feature_{i}" for i in range(len(importance))])
            return dict(zip(feature_names, importance.tolist()))
        except Exception:
            return None


class LightGBMHandler(ModelHandler):
    """Handler para LightGBM"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__("lightgbm", config)
        self.lgb = safe_import("lightgbm")
        if self.lgb is None:
            raise ImportError("LightGBM no disponible")
    
    def create_model(self, params: Dict[str, Any]) -> Any:
        """Crear modelo LightGBM"""
        model_config = self.config.get_model_config("lightgbm")
        final_params = {**model_config, **params}
        
        # Ajustar parámetros específicos de LightGBM
        if 'device' in final_params and final_params['device'] == 'gpu':
            final_params['objective'] = 'binary'
        
        return self.lgb.LGBMClassifier(**final_params)
    
    def get_param_space(self) -> Dict[str, Any]:
        """Espacio de parámetros para LightGBM"""
        return {
            'n_estimators': ('int', 100, 1000),
            'max_depth': ('int', 3, 12),
            'learning_rate': ('float', 0.01, 0.3, True),
            'num_leaves': ('int', 20, 300),
            'feature_fraction': ('float', 0.4, 1.0),
            'bagging_fraction': ('float', 0.4, 1.0),
            'bagging_freq': ('int', 1, 7),
            'min_child_samples': ('int', 5, 100),
            'reg_alpha': ('float', 0, 10),
            'reg_lambda': ('float', 0, 10)
        }
    
    def extract_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extraer importancia de features de LightGBM"""
        try:
            importance = model.feature_importances_
            feature_names = getattr(model, 'feature_name_', 
                                  [f"feature_{i}" for i in range(len(importance))])
            return dict(zip(feature_names, importance.tolist()))
        except Exception:
            return None


class CatBoostHandler(ModelHandler):
    """Handler para CatBoost"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__("catboost", config)
        self.cb = safe_import("catboost")
        if self.cb is None:
            raise ImportError("CatBoost no disponible")
    
    def create_model(self, params: Dict[str, Any]) -> Any:
        """Crear modelo CatBoost"""
        model_config = self.config.get_model_config("catboost")
        final_params = {**model_config, **params}
        
        # CatBoost parámetros específicos
        final_params['verbose'] = False  # Reducir output
        final_params['allow_writing_files'] = False  # No escribir archivos automáticamente
        
        return self.cb.CatBoostClassifier(**final_params)
    
    def get_param_space(self) -> Dict[str, Any]:
        """Espacio de parámetros para CatBoost"""
        return {
            'iterations': ('int', 100, 1000),
            'depth': ('int', 4, 10),
            'learning_rate': ('float', 0.01, 0.3, True),
            'l2_leaf_reg': ('float', 1, 10),
            'border_count': ('int', 32, 255),
            'bagging_temperature': ('float', 0, 1),
            'random_strength': ('float', 0, 10)
        }
    
    def extract_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extraer importancia de features de CatBoost"""
        try:
            importance = model.get_feature_importance()
            feature_names = getattr(model, 'feature_names_', 
                                  [f"feature_{i}" for i in range(len(importance))])
            return dict(zip(feature_names, importance.tolist()))
        except Exception:
            return None


class HyperparameterOptimizer:
    """
    Optimizador principal de hiperparámetros enterprise-ready.
    
    Centraliza toda la lógica de optimización con soporte para múltiples
    modelos, caching inteligente, logging estructurado y manejo robusto de errores.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Inicializar optimizador.
        
        Args:
            config: Configuración de optimización. Si None, usa configuración por defecto.
        """
        self.config = config or get_config()
        self.logger = get_logger("optimizer")
        self.optimization_logger = OptimizationLogger()
        self.data_manager = DataManager()
        
        # Inicializar handlers de modelos
        self.model_handlers: Dict[str, ModelHandler] = {}
        self._initialize_model_handlers()
        
        # Verificar dependencias críticas
        self._verify_dependencies()
        
        self.logger.info("🚀 Optimizador inicializado")
    
    def _initialize_model_handlers(self):
        """Inicializar handlers de modelos disponibles"""
        ml_libs = get_ml_libraries()
        
        if "xgboost" in ml_libs and "xgboost" in self.config.enabled_models:
            try:
                self.model_handlers["xgboost"] = XGBoostHandler(self.config)
                self.logger.info("✅ XGBoost handler inicializado")
            except Exception as e:
                self.logger.warning(f"❌ Error inicializando XGBoost: {e}")
        
        if "lightgbm" in ml_libs and "lightgbm" in self.config.enabled_models:
            try:
                self.model_handlers["lightgbm"] = LightGBMHandler(self.config)
                self.logger.info("✅ LightGBM handler inicializado")
            except Exception as e:
                self.logger.warning(f"❌ Error inicializando LightGBM: {e}")
        
        if "catboost" in ml_libs and "catboost" in self.config.enabled_models:
            try:
                self.model_handlers["catboost"] = CatBoostHandler(self.config)
                self.logger.info("✅ CatBoost handler inicializado")
            except Exception as e:
                self.logger.warning(f"❌ Error inicializando CatBoost: {e}")
        
        if not self.model_handlers:
            raise RuntimeError("Ningún modelo disponible para optimización")
    
    def _verify_dependencies(self):
        """Verificar dependencias críticas"""
        if optuna is None:
            raise ImportError("Optuna es requerido para optimización")
        
        if sklearn_metrics is None or sklearn_model_selection is None:
            raise ImportError("Scikit-learn es requerido para métricas y validación")
        
        self.logger.info("✅ Dependencias verificadas")
    
    @log_execution_time("optimization")
    def optimize_single_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                             experiment_id: str) -> OptimizationResult:
        """
        Optimizar un modelo específico.
        
        Args:
            model_name: Nombre del modelo a optimizar
            X: Features
            y: Target
            experiment_id: ID del experimento
            
        Returns:
            Resultado de optimización
        """
        if model_name not in self.model_handlers:
            raise ValueError(f"Modelo {model_name} no disponible")
        
        handler = self.model_handlers[model_name]
        n_trials = self.config.model_trials.get(model_name, 100)
        
        # Log inicio
        self.optimization_logger.log_optimization_start(
            model_name, n_trials, experiment_id
        )
        
        start_time = time.time()
        
        # Crear estudio Optuna
        study = optuna.create_study(
            direction='maximize',
            study_name=f"{experiment_id}_{model_name}",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # Función objetivo
        def objective(trial):
            return self._objective_function(trial, handler, X, y)
        
        # Optimización
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=self.config.optimization_timeout,
                callbacks=[self._trial_callback],
                catch=(Exception,)
            )
        except KeyboardInterrupt:
            self.logger.warning(f"⚠️  Optimización {model_name} interrumpida por usuario")
        except Exception as e:
            self.logger.error(f"❌ Error en optimización {model_name}: {e}")
            raise
        
        optimization_time = time.time() - start_time
        
        # Obtener mejores resultados
        best_params = study.best_params
        best_score = study.best_value
        
        # Entrenar modelo final con mejores parámetros
        final_model = handler.create_model(best_params)
        final_model.fit(X, y)
        
        # Validación cruzada del mejor modelo
        cv_scores = self._cross_validate_model(final_model, X, y)
        
        # Extraer importancia de features
        feature_importance = handler.extract_feature_importance(final_model)
        
        # Log finalización
        self.optimization_logger.log_optimization_complete(
            model_name, best_score, optimization_time, len(study.trials)
        )
        
        # Crear resultado
        result = OptimizationResult(
            model_name=model_name,
            best_score=best_score,
            best_params=best_params,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            model_object=final_model,
            study_object=study
        )
        
        # Limpiar memoria
        gc.collect()
        
        return result
    
    def _objective_function(self, trial, handler: ModelHandler, X: pd.DataFrame, y: pd.Series) -> float:
        """Función objetivo para Optuna"""
        try:
            # Generar parámetros del trial
            params = {}
            param_space = handler.get_param_space()
            
            for param_name, param_config in param_space.items():
                if param_config[0] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
                elif param_config[0] == 'float':
                    log_scale = len(param_config) > 3 and param_config[3]
                    params[param_name] = trial.suggest_float(
                        param_name, param_config[1], param_config[2], log=log_scale
                    )
                elif param_config[0] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config[1])
            
            # Crear y entrenar modelo
            model = handler.create_model(params)
            
            # Validación cruzada
            cv_scores = self._cross_validate_model(model, X, y)
            mean_score = np.mean(cv_scores)
            
            # Log resultado del trial
            self.optimization_logger.log_trial_result(
                trial.number, mean_score, params, time.time() - trial.datetime_start.timestamp()
            )
            
            return mean_score
            
        except Exception as e:
            self.logger.warning(f"Trial {trial.number} falló: {e}")
            return 0.0  # Retornar score bajo para trials fallidos
    
    def _cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Realizar validación cruzada"""
        from sklearn.model_selection import cross_val_score
        
        try:
            scores = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric,
                n_jobs=1  # Evitar conflictos de paralelización
            )
            return scores.tolist()
        except Exception as e:
            self.logger.warning(f"Error en validación cruzada: {e}")
            return [0.0] * self.config.cv_folds
    
    def _trial_callback(self, study, trial):
        """Callback para cada trial de Optuna"""
        # Limpiar memoria cada N trials
        if trial.number % self.config.garbage_collection_frequency == 0:
            gc.collect()
        
        # Log de progreso cada 10 trials
        if trial.number % 10 == 0:
            self.logger.info(f"🔄 Trial {trial.number}: mejor score = {study.best_value:.4f}")
    
    @log_execution_time("full_optimization")
    def optimize_all_models(self, data_path: str, experiment_id: Optional[str] = None) -> ExperimentResult:
        """
        Optimizar todos los modelos configurados.
        
        Args:
            data_path: Ruta a los datos
            experiment_id: ID del experimento (generado automáticamente si None)
            
        Returns:
            Resultado completo del experimento
        """
        if experiment_id is None:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"🚀 Iniciando optimización completa - Experimento: {experiment_id}")
        
        start_time = time.time()
        
        # Cargar datos
        features, target, data_info = self.data_manager.load_data(data_path)
        
        # Log información de datos
        self.optimization_logger.log_data_info(
            features.shape, len(features.columns), 
            data_info.target_distribution or {}
        )
        
        # Crear resultado del experimento
        experiment_result = ExperimentResult(
            experiment_id=experiment_id,
            config=self.config,
            data_info=data_info
        )
        
        # Optimizar cada modelo
        for model_name in self.model_handlers.keys():
            try:
                self.logger.info(f"🔄 Optimizando {model_name}...")
                
                # Optimizar modelo
                model_result = self.optimize_single_model(
                    model_name, features, target, experiment_id
                )
                
                experiment_result.model_results[model_name] = model_result
                
                self.logger.info(f"✅ {model_name} completado: {model_result.best_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ Error optimizando {model_name}: {e}")
                # Continuar con otros modelos
                continue
        
        # Determinar mejor modelo
        if experiment_result.model_results:
            best_model_name = max(
                experiment_result.model_results.keys(),
                key=lambda m: experiment_result.model_results[m].best_score
            )
            experiment_result.best_model = best_model_name
            experiment_result.best_score = experiment_result.model_results[best_model_name].best_score
        
        experiment_result.total_time = time.time() - start_time
        
        self.logger.info(f"✅ Optimización completa finalizada en {experiment_result.total_time:.2f}s")
        self.logger.info(f"🏆 Mejor modelo: {experiment_result.best_model} ({experiment_result.best_score:.4f})")
        
        return experiment_result
    
    def save_results(self, result: ExperimentResult, output_dir: str = "./results"):
        """
        Guardar resultados del experimento.
        
        Args:
            result: Resultado del experimento
            output_dir: Directorio de salida
        """
        from pathlib import Path
        import pickle
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Guardar configuración
        config_file = output_path / f"config_{result.experiment_id}_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(result.config.to_dict(), f, indent=2, default=str)
        
        # Guardar resultados
        results_file = output_path / f"results_{result.experiment_id}_{timestamp}.json"
        results_data = {
            "experiment_id": result.experiment_id,
            "total_time": result.total_time,
            "best_model": result.best_model,
            "best_score": result.best_score,
            "timestamp": result.timestamp.isoformat(),
            "data_info": {
                "shape": result.data_info.shape,
                "memory_usage_mb": result.data_info.memory_usage_mb,
                "target_distribution": result.data_info.target_distribution
            },
            "model_results": {
                name: {
                    "best_score": res.best_score,
                    "best_params": res.best_params,
                    "n_trials": res.n_trials,
                    "optimization_time": res.optimization_time,
                    "cv_scores": res.cv_scores,
                    "feature_importance": res.feature_importance
                }
                for name, res in result.model_results.items()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Guardar modelos si está habilitado
        if self.config.save_models:
            for name, model_result in result.model_results.items():
                model_file = output_path / f"model_{name}_{result.experiment_id}_{timestamp}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model_result.model_object, f)
        
        self.logger.info(f"💾 Resultados guardados en: {output_path}")


# ==================== FUNCIONES DE CONVENIENCIA ====================

def quick_optimization(data_path: str, models: Optional[List[str]] = None) -> ExperimentResult:
    """
    Función de conveniencia para optimización rápida.
    
    Args:
        data_path: Ruta a los datos
        models: Lista de modelos a optimizar (None = todos)
        
    Returns:
        Resultado del experimento
    """
    from ..config.optimization_config import get_quick_config
    
    config = get_quick_config()
    if models:
        config.enabled_models = models
    
    optimizer = HyperparameterOptimizer(config)
    return optimizer.optimize_all_models(data_path)

def production_optimization(data_path: str, models: Optional[List[str]] = None) -> ExperimentResult:
    """
    Función de conveniencia para optimización de producción.
    
    Args:
        data_path: Ruta a los datos
        models: Lista de modelos a optimizar (None = todos)
        
    Returns:
        Resultado del experimento
    """
    from ..config.optimization_config import get_production_config
    
    config = get_production_config()
    if models:
        config.enabled_models = models
    
    optimizer = HyperparameterOptimizer(config)
    return optimizer.optimize_all_models(data_path)


if __name__ == "__main__":
    # Demo del optimizador
    print("🚀 Optimizador Principal Refactorizado - Fase 5")
    print("===============================================")
    
    # Configurar logging
    from ..utils.logging_setup import setup_logging
    setup_logging({"level": "INFO"})
    
    # Configuración de demo
    from ..config.optimization_config import get_quick_config
    
    config = get_quick_config()
    config.enabled_models = ["xgboost"]  # Solo XGBoost para demo
    config.model_trials = {"xgboost": 10}  # Pocos trials para demo
    
    print(f"✅ Configuración de demo creada")
    print(f"   - Modelos: {config.enabled_models}")
    print(f"   - Trials: {config.model_trials}")
    
    # Verificar dependencias
    try:
        optimizer = HyperparameterOptimizer(config)
        print(f"✅ Optimizador inicializado correctamente")
        print(f"   - Handlers disponibles: {list(optimizer.model_handlers.keys())}")
    except Exception as e:
        print(f"❌ Error inicializando optimizador: {e}")
    
    print("\n🎯 Para ejecutar optimización real, usar:")
    print("   result = quick_optimization('path/to/data.csv')")
