#!/usr/bin/env python3
"""
Sistema de Optimización de Hiperparámetros con Optuna para Criptomonedas
Optimización automática de XGBoost, LightGBM y CatBoost
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
import json
import pickle
from typing import Dict, Any, List, Tuple, Optional
import time

# Suprimir warnings
warnings.filterwarnings('ignore')

# Importar nuevos componentes de Fase 1
try:
    from config.optimization_config import CONFIG, MODEL_CONFIG
    from utils.gpu_manager import GPU_MANAGER
    from utils.data_validator import DataValidator, DataValidationError
    from utils.metrics_calculator import MetricsCalculator, MetricsResult
    from utils.optimization_logger import get_optimization_logger
    print("✅ Nuevos componentes de Fase 1 importados correctamente")
except ImportError as e:
    print(f"⚠️ Error importando componentes de Fase 1: {e}")
    print("Continuando con funcionalidad básica...")

# Agregar paths necesarios
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'utils', 'utils'))

# Intentar importar feature engineering
try:
    from src.utils.utils.feature_engineering import create_ml_features, prepare_ml_dataset
    print("✅ Feature engineering importado desde src.utils.utils")
except ImportError:
    try:
        from utils.utils.feature_engineering import create_ml_features, prepare_ml_dataset
        print("✅ Feature engineering importado desde utils.utils")
    except ImportError:
        try:
            from feature_engineering import create_ml_features, prepare_ml_dataset
            print("✅ Feature engineering importado desde feature_engineering")
        except ImportError:
            print("❌ No se pudo importar feature_engineering")
            sys.exit(1)

# Imports de ML
try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score, classification_report
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
    print("✅ Todas las librerías importadas correctamente")
except ImportError as e:
    print(f"❌ Error importando librerías: {e}")
    sys.exit(1)

class CryptoHyperparameterOptimizer:
    """
    Sistema completo de optimización de hiperparámetros para modelos de criptomonedas
    Incluye mejoras de Fase 1: validación robusta, GPU inteligente, métricas múltiples y logging
    """
    
    def __init__(self, data_path: str = None, results_path: str = None, config: Any = None):
        """
        Inicializar el optimizador con mejoras de Fase 1
        
        Args:
            data_path: Ruta a los datos (opcional, usa CONFIG si no se especifica)
            results_path: Ruta donde guardar resultados (opcional)
            config: Configuración personalizada (opcional)
        """
        # Usar configuración global o personalizada
        self.config = config if config is not None else CONFIG
        
        # Rutas de datos y resultados
        self.data_path = data_path or self.config.data_path
        self.results_path = Path(results_path or self.config.results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Inicializar componentes de Fase 1
        try:
            # Logger estructurado
            self.logger = get_optimization_logger(
                log_dir=str(self.results_path / "logs"),
                log_level=self.config.log_level,
                enable_file_logging=self.config.log_to_file
            )
            
            # GPU Manager
            self.gpu_manager = GPU_MANAGER
            
            # Validador de datos
            self.data_validator = DataValidator(self.config)
            
            # Calculadora de métricas
            self.metrics_calculator = MetricsCalculator(
                primary_metric=self.config.primary_metric
            )
            
            print("✅ Componentes de Fase 1 inicializados correctamente")
            
        except Exception as e:
            print(f"⚠️ Error inicializando componentes de Fase 1: {e}")
            print("Continuando con funcionalidad básica...")
            self.logger = None
            self.gpu_manager = None
            self.data_validator = None
            self.metrics_calculator = None
        
        # Datasets
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Estudios de Optuna
        self.studies = {}
        self.best_params = {}
        self.best_scores = {}
        self.detailed_results = {}  # Para métricas múltiples
        
        # Configuración desde CONFIG
        self.cv_folds = self.config.cv_folds
        self.random_state = self.config.random_state
        
        # Log inicio
        if self.logger:
            self.logger.log_optimization_start({
                'data_path': self.data_path,
                'results_path': str(self.results_path),
                'cv_folds': self.cv_folds,
                'random_state': self.random_state
            })
            
            # Log información de hardware
            if self.gpu_manager:
                self.gpu_manager.print_hardware_summary()
                self.logger.log_gpu_info(self.gpu_manager.gpu_info)
        
        print("🔧 CryptoHyperparameterOptimizer inicializado")
        print(f"   📁 Datos: {self.data_path}")
        print(f"   💾 Resultados: {self.results_path}")
        print(f"   🎯 Métrica primaria: {self.config.primary_metric}")
        print(f"   🔄 CV folds: {self.cv_folds}")
    
    def load_and_prepare_data(self, target_period: int = None, min_market_cap: float = None, 
                             max_market_cap: float = None):
        """
        Cargar y preparar datos con validación robusta (Fase 1)
        """
        # Usar valores de configuración si no se especifican
        target_period = target_period or self.config.target_period
        min_market_cap = min_market_cap or self.config.min_market_cap
        max_market_cap = max_market_cap or self.config.max_market_cap
        
        print("🚀======================================================================")
        print("📊 CARGANDO Y PREPARANDO DATOS CON VALIDACIÓN ROBUSTA")
        print("🚀======================================================================")
        
        # FASE 1: VALIDACIÓN ROBUSTA
        try:
            if self.data_validator:
                target_col = f'high_return_{target_period}d'
                
                # Validación completa de datos
                validation_results = self.data_validator.run_full_validation(
                    data_path=self.data_path,
                    target_column=target_col,
                    exclude_columns=self.config.exclude_columns
                )
                
                if self.logger:
                    self.logger.log_data_info(validation_results)
                
                print("✅ Validación de datos completada exitosamente")
                
            else:
                print("⚠️ Validador no disponible, usando validación básica")
                
        except DataValidationError as e:
            error_msg = f"Error de validación de datos: {e}"
            if self.logger:
                self.logger.log_error(error_msg, exception=e)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error inesperado en validación: {e}"
            if self.logger:
                self.logger.log_error(error_msg, exception=e)
            print(f"⚠️ {error_msg}")
        
        # Cargar datos
        print(f"📁 Cargando datos desde: {self.data_path}")
        if not os.path.exists(self.data_path):
            error_msg = f"No se encontró el archivo: {self.data_path}"
            if self.logger:
                self.logger.log_error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            df = pd.read_csv(self.data_path)
            print(f"   📊 Datos cargados: {df.shape}")
            
        except Exception as e:
            error_msg = f"Error cargando datos: {e}"
            if self.logger:
                self.logger.log_error(error_msg, exception=e)
            raise RuntimeError(error_msg)
        
        # Filtrar por market cap con validación
        try:
            df_filtered = df[(df['market_cap'] >= min_market_cap) & 
                            (df['market_cap'] <= max_market_cap)].copy()
            print(f"   💰 Filtrado por market cap: {df_filtered.shape}")
            
            if len(df_filtered) == 0:
                raise ValueError("No quedan datos después del filtro de market cap")
                
        except Exception as e:
            error_msg = f"Error filtrando por market cap: {e}"
            if self.logger:
                self.logger.log_error(error_msg, exception=e)
            raise RuntimeError(error_msg)
        
        # Crear features con manejo de errores
        try:
            print("🔧 Creando features avanzadas...")
            df_features = create_ml_features(df_filtered, include_targets=True)
            
        except Exception as e:
            error_msg = f"Error creando features: {e}"
            if self.logger:
                self.logger.log_error(error_msg, exception=e)
            raise RuntimeError(error_msg)
        
        # Preparar dataset con validación
        target_col = f'high_return_{target_period}d'
        print(f"🎯 Variable objetivo: {target_col}")
        
        if target_col not in df_features.columns:
            error_msg = f"Columna objetivo '{target_col}' no encontrada"
            if self.logger:
                self.logger.log_error(error_msg)
            raise ValueError(error_msg)
        
        # Split temporal con configuración
        try:
            df_clean = df_features.dropna(subset=[target_col]).sort_values('date')
            
            # Usar configuración para splits
            split_config = self.config.get_data_split_config()
            n_total = len(df_clean)
            
            train_end = int(split_config['train_size'] * n_total)
            val_end = int((split_config['train_size'] + split_config['val_size']) * n_total)
            
            df_train = df_clean.iloc[:train_end]
            df_val = df_clean.iloc[train_end:val_end]
            df_test = df_clean.iloc[val_end:]
            
            print(f"   📊 Train: {len(df_train)} ({len(df_train)/n_total:.1%})")
            print(f"   📊 Validation: {len(df_val)} ({len(df_val)/n_total:.1%})")
            print(f"   📊 Test: {len(df_test)} ({len(df_test)/n_total:.1%})")
            
        except Exception as e:
            error_msg = f"Error en split de datos: {e}"
            if self.logger:
                self.logger.log_error(error_msg, exception=e)
            raise RuntimeError(error_msg)
        
        # Preparar features con manejo de errores robusto
        try:
            exclude_cols = self.config.exclude_columns + \
                           [col for col in df_clean.columns if col.startswith('future_') or 
                            col.startswith('high_return_') or col.startswith('return_category_') or
                            col.startswith('extreme_return_')]
            
            feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
            
            def prepare_features(df_subset):
                """Preparar features para un subset con validación"""
                X = df_subset[feature_cols].copy()
                
                # Manejar variables categóricas
                categorical_cols = []
                for col in X.columns:
                    if X[col].dtype == 'object' or col in ['narrative', 'cluster_id']:
                        categorical_cols.append(col)
                
                if categorical_cols:
                    try:
                        from sklearn.preprocessing import LabelEncoder
                        for col in categorical_cols:
                            if col in X.columns:
                                le = LabelEncoder()
                                # Fit con todos los datos para consistencia
                                all_values = pd.concat([df_train[col], df_val[col], df_test[col]]).astype(str)
                                le.fit(all_values)
                                X[col] = le.transform(X[col].astype(str))
                    except ImportError:
                        if self.logger:
                            self.logger.log_warning("sklearn no disponible para LabelEncoder")
                        # Encoding manual básico
                        for col in categorical_cols:
                            if col in X.columns:
                                X[col] = pd.Categorical(X[col]).codes
                
                # Convertir a numérico y limpiar con validación
                for col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                
                X = X.fillna(0)
                
                # Limpiar infinitos
                X = X.replace([np.inf, -np.inf], 0)
                
                return X
            
            # Preparar cada subset
            self.X_train = prepare_features(df_train)
            self.X_val = prepare_features(df_val)
            self.X_test = prepare_features(df_test)
            
            self.y_train = df_train[target_col]
            self.y_val = df_val[target_col]
            self.y_test = df_test[target_col]
            
        except Exception as e:
            error_msg = f"Error preparando features: {e}"
            if self.logger:
                self.logger.log_error(error_msg, exception=e)
            raise RuntimeError(error_msg)
        
        # Validación final de splits usando DataValidator
        try:
            if self.data_validator:
                split_validation = self.data_validator.validate_data_splits(
                    self.X_train, self.X_val, self.X_test,
                    self.y_train, self.y_val, self.y_test
                )
                
                if self.logger:
                    self.logger.log_info("Validación de splits completada", split_validation)
                
        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Error en validación de splits: {e}")
            print(f"⚠️ Error en validación de splits: {e}")
        
        # Información final
        print(f"   🔧 Features utilizadas: {len(feature_cols)}")
        print(f"   🎯 Distribución train: {self.y_train.value_counts().to_dict()}")
        print(f"   🎯 Distribución val: {self.y_val.value_counts().to_dict()}")
        print(f"   🎯 Distribución test: {self.y_test.value_counts().to_dict()}")
        
        # Log memoria si está disponible
        if self.logger and self.data_validator:
            try:
                memory_info = self.data_validator.validate_memory_requirements(self.X_train)
                self.logger.log_memory_usage(memory_info)
            except Exception as e:
                self.logger.log_warning(f"Error verificando memoria: {e}")
        
        if self.logger:
            self.logger.log_info("Preparación de datos completada exitosamente", {
                'train_size': len(self.X_train),
                'val_size': len(self.X_val),
                'test_size': len(self.X_test),
                'n_features': len(feature_cols),
                'target_column': target_col
            })
        
        return self
    
    def optimize_xgboost(self, n_trials: int = None, timeout: Optional[int] = None):
        """
        Optimizar hiperparámetros de XGBoost con mejoras de Fase 1
        """
        n_trials = n_trials or self.config.default_n_trials
        timeout = timeout or self.config.default_timeout_per_model
        
        print("\n🔥======================================================================")
        print("🔥 OPTIMIZANDO XGBOOST CON MEJORAS DE FASE 1")
        print("🔥======================================================================")
        
        # Log inicio de optimización del modelo
        if self.logger:
            self.logger.log_model_optimization_start('xgboost', n_trials, {
                'timeout': timeout,
                'cv_folds': self.cv_folds
            })
        
        model_start_time = time.time()
        
        def objective(trial):
            """Función objetivo para XGBoost con mejoras de Fase 1"""
            trial_start_time = time.time()
            
            # Configuración base con GPU Manager
            base_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.random_state,
                'verbosity': 0,
            }
            
            # Configuración GPU/CPU inteligente
            if self.gpu_manager:
                try:
                    gpu_config = self.gpu_manager.get_xgboost_config(
                        fallback_to_cpu=self.config.fallback_to_cpu
                    )
                    base_params.update(gpu_config)
                except Exception as e:
                    if self.logger:
                        self.logger.log_warning(f"Error configurando GPU para XGBoost: {e}")
                    # Fallback a CPU
                    base_params.update({'tree_method': 'hist', 'n_jobs': -1})
            else:
                # Configuración CPU por defecto
                base_params.update({'tree_method': 'hist', 'n_jobs': -1})
            
            # Hiperparámetros a optimizar usando MODEL_CONFIG
            xgb_config = MODEL_CONFIG.xgboost_params
            params = base_params.copy()
            
            for param_name, param_config in xgb_config.items():
                if param_config['type'] == 'int':
                    if 'step' in param_config:
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], 
                            param_config['high'], step=param_config['step']
                        )
                    else:
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], param_config['high']
                        )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], 
                        param_config['high'], log=param_config.get('log', False)
                    )
            
            # Log inicio del trial
            if self.logger:
                self.logger.log_trial_start(trial.number, 'xgboost', params)
            
            try:
                # Entrenar modelo
                model = xgb.XGBClassifier(**params)
                
                # Cross-validation en datos de entrenamiento
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                primary_score = cv_scores.mean()
                
                # Calcular métricas múltiples si está disponible
                if self.metrics_calculator:
                    try:
                        # Entrenar en datos completos para evaluación
                        model.fit(self.X_train, self.y_train)
                        y_pred = model.predict(self.X_val)
                        y_proba = model.predict_proba(self.X_val)[:, 1]
                        
                        # Calcular todas las métricas
                        metrics_result = self.metrics_calculator.calculate_all_metrics(
                            y_true=self.y_val.values,
                            y_pred=y_pred,
                            y_proba=y_proba,
                            cv_scores=cv_scores.tolist(),
                            metrics_to_calculate=self.config.secondary_metrics
                        )
                        
                        # Log métricas
                        if self.logger:
                            self.logger.log_metrics(trial.number, 'xgboost', metrics_result.secondary_scores)
                        
                    except Exception as e:
                        if self.logger:
                            self.logger.log_warning(f"Error calculando métricas múltiples: {e}")
                
                # Log trial exitoso
                trial_duration = time.time() - trial_start_time
                if self.logger:
                    self.logger.log_trial_complete(trial.number, 'xgboost', primary_score, trial_duration)
                
                return primary_score
                
            except Exception as e:
                # Log trial fallido
                trial_duration = time.time() - trial_start_time
                if self.logger:
                    self.logger.log_trial_complete(trial.number, 'xgboost', 0.0, trial_duration, "failed")
                    self.logger.log_error(f"Error en trial XGBoost {trial.number}", {'params': params}, e)
                
                raise optuna.TrialPruned()
        
        # Crear y ejecutar estudio
        study_name = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f'sqlite:///{self.results_path}/optuna_studies.db',
            load_if_exists=True
        )
        
        print(f"   🎯 Ejecutando {n_trials} trials...")
        if timeout:
            print(f"   ⏰ Timeout: {timeout} segundos")
        
        # Log progreso periódico
        def progress_callback(study, trial):
            if trial.number % 10 == 0 and self.logger:
                current_best = study.best_value if study.best_value else 0.0
                self.logger.log_progress(trial.number, n_trials, current_best, 'xgboost')
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[progress_callback])
        
        # Guardar resultados
        model_duration = time.time() - model_start_time
        self.studies['xgboost'] = study
        self.best_params['xgboost'] = study.best_params
        self.best_scores['xgboost'] = study.best_value
        
        print(f"   ✅ Optimización completada!")
        print(f"   🏆 Mejor AUC: {study.best_value:.4f}")
        print(f"   🔧 Mejores parámetros: {study.best_params}")
        print(f"   ⏰ Tiempo total: {model_duration:.1f}s")
        
        # Log finalización
        if self.logger:
            self.logger.log_model_optimization_complete(
                'xgboost', study.best_value, study.best_params, model_duration
            )
        
        return study
    
    def optimize_lightgbm(self, n_trials: int = 100, timeout: Optional[int] = None):
        """
        Optimizar hiperparámetros de LightGBM
        """
        print("\n💡======================================================================")
        print("💡 OPTIMIZANDO LIGHTGBM CON OPTUNA")
        print("💡======================================================================")
        
        def objective(trial):
            """Función objetivo para LightGBM"""
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'device': 'gpu',          # 🚀 Usar GPU
                'gpu_platform_id': 0,    # 🚀 GPU Platform ID
                'gpu_device_id': 0,      # 🚀 GPU Device ID
                'random_state': self.random_state,
                'verbosity': -1,
                
                # Hiperparámetros a optimizar
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300)
            }
            
            # Entrenar modelo
            model = lgb.LGBMClassifier(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Crear y ejecutar estudio
        study_name = f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f'sqlite:///{self.results_path}/optuna_studies.db',
            load_if_exists=True
        )
        
        print(f"   🎯 Ejecutando {n_trials} trials...")
        if timeout:
            print(f"   ⏰ Timeout: {timeout} segundos")
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Guardar resultados
        self.studies['lightgbm'] = study
        self.best_params['lightgbm'] = study.best_params
        self.best_scores['lightgbm'] = study.best_value
        
        print(f"   ✅ Optimización completada!")
        print(f"   🏆 Mejor AUC: {study.best_value:.4f}")
        print(f"   🔧 Mejores parámetros: {study.best_params}")
        
        return study
    
    def optimize_catboost(self, n_trials: int = 100, timeout: Optional[int] = None):
        """
        Optimizar hiperparámetros de CatBoost
        """
        print("\n🐱======================================================================")
        print("🐱 OPTIMIZANDO CATBOOST CON OPTUNA")
        print("🐱======================================================================")
        
        def objective(trial):
            """Función objetivo para CatBoost"""
            params = {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'task_type': 'GPU',       # 🚀 Usar GPU
                'devices': '0',           # 🚀 GPU Device ID
                'random_state': self.random_state,
                'verbose': False,
                'allow_writing_files': False,
                
                # Hiperparámetros a optimizar
                'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli'])
            }
            
            # Parámetros específicos para bootstrap
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
            else:
                params['subsample'] = trial.suggest_float('subsample_bernoulli', 0.6, 1.0)
            
            # Entrenar modelo
            model = cb.CatBoostClassifier(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Crear y ejecutar estudio
        study_name = f"catboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f'sqlite:///{self.results_path}/optuna_studies.db',
            load_if_exists=True
        )
        
        print(f"   🎯 Ejecutando {n_trials} trials...")
        if timeout:
            print(f"   ⏰ Timeout: {timeout} segundos")
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Guardar resultados
        self.studies['catboost'] = study
        self.best_params['catboost'] = study.best_params
        self.best_scores['catboost'] = study.best_value
        
        print(f"   ✅ Optimización completada!")
        print(f"   🏆 Mejor AUC: {study.best_value:.4f}")
        print(f"   🔧 Mejores parámetros: {study.best_params}")
        
        return study
    
    def optimize_all_models(self, n_trials: int = 100, timeout_per_model: Optional[int] = None):
        """
        Optimizar todos los modelos secuencialmente
        """
        print("🚀======================================================================")
        print("🚀 OPTIMIZACIÓN COMPLETA DE TODOS LOS MODELOS")
        print("🚀======================================================================")
        
        # Lista de modelos a optimizar
        models_to_optimize = [
            ('XGBoost', self.optimize_xgboost),
            ('LightGBM', self.optimize_lightgbm),
            ('CatBoost', self.optimize_catboost)
        ]
        
        start_time = datetime.now()
        
        for model_name, optimize_func in models_to_optimize:
            print(f"\n🎯 Iniciando optimización de {model_name}...")
            model_start = datetime.now()
            
            try:
                optimize_func(n_trials=n_trials, timeout=timeout_per_model)
                model_time = datetime.now() - model_start
                print(f"   ⏰ Tiempo {model_name}: {model_time}")
                
            except Exception as e:
                print(f"   ❌ Error optimizando {model_name}: {e}")
                continue
        
        total_time = datetime.now() - start_time
        print(f"\n⏰ Tiempo total de optimización: {total_time}")
        
        # Guardar resumen de resultados
        self.save_optimization_summary()
        
        return self
    
    def evaluate_best_models(self):
        """
        Evaluar los mejores modelos encontrados en el conjunto de validación
        """
        print("\n📊======================================================================")
        print("📊 EVALUANDO MEJORES MODELOS EN VALIDACIÓN")
        print("📊======================================================================")
        
        evaluation_results = {}
        
        for model_name in self.best_params.keys():
            print(f"\n🔍 Evaluando {model_name.upper()}...")
            
            try:
                # Crear modelo con mejores parámetros
                if model_name == 'xgboost':
                    model = xgb.XGBClassifier(**self.best_params[model_name])
                elif model_name == 'lightgbm':
                    model = lgb.LGBMClassifier(**self.best_params[model_name])
                elif model_name == 'catboost':
                    model = cb.CatBoostClassifier(**self.best_params[model_name])
                
                # Entrenar en datos de entrenamiento
                model.fit(self.X_train, self.y_train)
                
                # Evaluar en validación
                y_pred_proba = model.predict_proba(self.X_val)[:, 1]
                val_auc = roc_auc_score(self.y_val, y_pred_proba)
                
                # Evaluar en test
                y_test_pred_proba = model.predict_proba(self.X_test)[:, 1]
                test_auc = roc_auc_score(self.y_test, y_test_pred_proba)
                
                evaluation_results[model_name] = {
                    'cv_score': self.best_scores[model_name],
                    'val_auc': val_auc,
                    'test_auc': test_auc,
                    'params': self.best_params[model_name]
                }
                
                print(f"   📊 CV Score: {self.best_scores[model_name]:.4f}")
                print(f"   📊 Validation AUC: {val_auc:.4f}")
                print(f"   📊 Test AUC: {test_auc:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error evaluando {model_name}: {e}")
                continue
        
        # Guardar resultados de evaluación
        results_file = self.results_path / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print(f"\n💾 Resultados guardados en: {results_file}")
        
        return evaluation_results
    
    def save_optimization_summary(self):
        """
        Guardar resumen completo de la optimización
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Resumen de mejores parámetros
        summary = {
            'timestamp': timestamp,
            'best_params': self.best_params,
            'best_scores': self.best_scores,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state
        }
        
        # Guardar como JSON
        summary_file = self.results_path / f"optimization_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Guardar estudios como pickle
        studies_file = self.results_path / f"optuna_studies_{timestamp}.pkl"
        with open(studies_file, 'wb') as f:
            pickle.dump(self.studies, f)
        
        print(f"\n💾 Resumen guardado en: {summary_file}")
        print(f"💾 Estudios guardados en: {studies_file}")
        
        return summary_file, studies_file
    
    def generate_visualizations(self):
        """
        Generar visualizaciones de los estudios de optimización
        """
        print("\n📈======================================================================")
        print("📈 GENERANDO VISUALIZACIONES")
        print("📈======================================================================")
        
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8')
        
        for model_name, study in self.studies.items():
            print(f"\n📊 Generando gráficos para {model_name}...")
            
            # Crear directorio para visualizaciones
            viz_dir = self.results_path / "visualizations" / model_name
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 1. Historia de optimización
                fig1 = plot_optimization_history(study)
                fig1.write_html(viz_dir / "optimization_history.html")
                
                # 2. Importancia de parámetros
                fig2 = plot_param_importances(study)
                fig2.write_html(viz_dir / "param_importances.html")
                
                # 3. Gráfico de contorno (solo para los 2 parámetros más importantes)
                if len(study.best_params) >= 2:
                    param_names = list(study.best_params.keys())[:2]
                    fig3 = plot_contour(study, params=param_names)
                    fig3.write_html(viz_dir / f"contour_{param_names[0]}_{param_names[1]}.html")
                
                print(f"   ✅ Visualizaciones guardadas en: {viz_dir}")
                
            except Exception as e:
                print(f"   ⚠️  Error generando visualizaciones para {model_name}: {e}")
    
    def print_final_summary(self):
        """
        Imprimir resumen final de la optimización
        """
        print("\n🏆======================================================================")
        print("🏆 RESUMEN FINAL DE OPTIMIZACIÓN")
        print("🏆======================================================================")
        
        if not self.best_scores:
            print("   ❌ No hay resultados de optimización disponibles")
            return
        
        # Ordenar modelos por performance
        sorted_models = sorted(self.best_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🏅 RANKING DE MODELOS (por CV Score):")
        for i, (model_name, score) in enumerate(sorted_models, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"   {medal} {i}. {model_name.upper():12} AUC: {score:.4f}")
        
        print("\n🔧 MEJORES HIPERPARÁMETROS:")
        for model_name in self.best_params:
            print(f"\n   🔹 {model_name.upper()}:")
            for param, value in self.best_params[model_name].items():
                print(f"      {param}: {value}")
        
        print(f"\n📁 Resultados guardados en: {self.results_path}")

def main():
    """
    Función principal para ejecutar optimización completa con mejoras de Fase 1
    """
    print("🚀 SISTEMA DE OPTIMIZACIÓN DE HIPERPARÁMETROS - FASE 1")
    print("🚀 CRIPTOMONEDAS DE BAJA CAPITALIZACIÓN")
    print("🚀 MEJORAS: Validación robusta, GPU inteligente, métricas múltiples, logging")
    print("🚀======================================================================")
    
    # Inicializar optimizador con configuración mejorada
    try:
        optimizer = CryptoHyperparameterOptimizer()
        
        # Cargar y preparar datos con validación robusta
        optimizer.load_and_prepare_data()
        
        # Configuración de optimización desde CONFIG
        N_TRIALS = optimizer.config.default_n_trials
        TIMEOUT_PER_MODEL = optimizer.config.default_timeout_per_model
        
        print(f"\n⚙️  CONFIGURACIÓN DE OPTIMIZACIÓN (FASE 1):")
        print(f"   🔢 Trials por modelo: {N_TRIALS}")
        print(f"   ⏰ Timeout por modelo: {TIMEOUT_PER_MODEL} segundos")
        print(f"   🔄 CV folds: {optimizer.cv_folds}")
        print(f"   🎯 Métrica primaria: {optimizer.config.primary_metric}")
        print(f"   📊 Métricas secundarias: {len(optimizer.config.secondary_metrics)}")
        print(f"   🎮 GPU disponible: {'✅' if optimizer.gpu_manager and optimizer.gpu_manager.cuda_available else '❌'}")
        
        # Ejecutar optimización completa
        optimizer.optimize_all_models(
            n_trials=N_TRIALS,
            timeout_per_model=TIMEOUT_PER_MODEL
        )
        
        # Evaluar mejores modelos
        optimizer.evaluate_best_models()
        
        # Generar visualizaciones
        optimizer.generate_visualizations()
        
        # Resumen final
        optimizer.print_final_summary()
        
        # Finalizar logging
        if optimizer.logger:
            optimizer.logger.log_optimization_complete({
                'best_scores': optimizer.best_scores,
                'best_params': optimizer.best_params
            })
        
        print("\n✅ OPTIMIZACIÓN FASE 1 COMPLETADA EXITOSAMENTE!")
        
    except Exception as e:
        print(f"\n❌ ERROR EN OPTIMIZACIÓN: {e}")
        import traceback
        traceback.print_exc()
        
        # Log error si está disponible
        if 'optimizer' in locals() and hasattr(optimizer, 'logger') and optimizer.logger:
            optimizer.logger.log_error("Error crítico en optimización", exception=e)
        
        sys.exit(1)

if __name__ == "__main__":
    main()
