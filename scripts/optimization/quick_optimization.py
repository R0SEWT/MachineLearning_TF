#!/usr/bin/env python3
"""
Script de Optimización Rápida con Optuna - REAL ML
Optimización rápida usando datos reales de criptomonedas
"""

import sys
import os
import argparse
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def load_crypto_data():
    """Carga los datos de criptomonedas procesados"""
    data_path = "/home/exodia/Documentos/MachineLearning_TF/data/ml_dataset.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ No se encontró el dataset en {data_path}")
        return None, None
    
    try:
        df = pd.read_csv(data_path)
        print(f"📊 Dataset cargado: {df.shape}")
        
        # Asumiendo que la última columna es el target
        target_cols = ['high_return_30d', 'target', 'label', 'y']
        target_col = None
        
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            print(f"❌ No se encontró columna objetivo en: {df.columns.tolist()}")
            return None, None
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Limpiar datos
        X = X.select_dtypes(include=[np.number])  # Solo columnas numéricas
        
        # Remover NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"✅ Datos limpios: X{X.shape}, y{y.shape}")
        print(f"🎯 Distribución target: {y.value_counts().to_dict()}")
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None, None

def objective_xgboost(trial, X, y):
    """Función objetivo para XGBoost con datos reales"""
    try:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist' if hasattr(xgb, 'gpu') else 'hist'
        }
        
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return np.mean(scores)
        
    except Exception as e:
        print(f"❌ Error en XGBoost trial: {e}")
        return 0.5  # Score neutro

def objective_lightgbm(trial, X, y):
    """Función objetivo para LightGBM con datos reales"""
    try:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'random_state': 42,
            'metric': 'auc',
            'objective': 'binary',
            'verbose': -1,
            'device': 'gpu' if hasattr(lgb, 'gpu') else 'cpu'
        }
        
        model = lgb.LGBMClassifier(**params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return np.mean(scores)
        
    except Exception as e:
        print(f"❌ Error en LightGBM trial: {e}")
        return 0.5

def objective_catboost(trial, X, y):
    """Función objetivo para CatBoost con datos reales"""
    try:
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 128),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'task_type': 'GPU' if hasattr(cb, 'cuda') else 'CPU',
            'random_seed': 42,
            'verbose': False,
            'eval_metric': 'AUC',
            'loss_function': 'Logloss'
        }
        
        model = cb.CatBoostClassifier(**params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return np.mean(scores)
        
    except Exception as e:
        print(f"❌ Error en CatBoost trial: {e}")
        return 0.5

def quick_optimize_real(trials=50, timeout=300, algorithm='all'):
    """Ejecuta optimización real con datos de criptomonedas"""
    print(f"🚀 Iniciando optimización REAL: {trials} trials, {timeout}s timeout")
    
    # Cargar datos
    X, y = load_crypto_data()
    if X is None or y is None:
        print("❌ No se pudieron cargar los datos")
        return False
    
    # Configurar base de datos
    db_path = "/home/exodia/Documentos/MachineLearning_TF/optimization_results/optuna_studies.db"
    storage = f"sqlite:///{db_path}"
    
    algorithms = ['xgboost', 'lightgbm', 'catboost'] if algorithm == 'all' else [algorithm]
    
    results = {}
    
    for algo in algorithms:
        try:
            study_name = f"real_{algo}_optimization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction='maximize',
                load_if_exists=True
            )
            
            print(f"📊 Optimizando {algo.upper()}: {study_name}")
            
            # Seleccionar función objetivo
            if algo == 'xgboost':
                objective_func = lambda trial: objective_xgboost(trial, X, y)
            elif algo == 'lightgbm':
                objective_func = lambda trial: objective_lightgbm(trial, X, y)
            elif algo == 'catboost':
                objective_func = lambda trial: objective_catboost(trial, X, y)
            
            # Ejecutar optimización
            study.optimize(objective_func, n_trials=trials, timeout=timeout)
            
            results[algo] = {
                'best_score': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials)
            }
            
            print(f"✅ {algo.upper()} completado")
            print(f"🏆 Mejor AUC: {study.best_value:.4f}")
            print(f"📊 Trials: {len(study.trials)}")
            
        except Exception as e:
            print(f"❌ Error optimizando {algo}: {e}")
            continue
    
    # Mostrar resumen
    print(f"\n🎯 RESUMEN DE OPTIMIZACIÓN REAL:")
    print("=" * 40)
    for algo, result in results.items():
        print(f"{algo.upper()}: AUC={result['best_score']:.4f} ({result['n_trials']} trials)")
    
    return len(results) > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimización rápida REAL con Optuna')
    parser.add_argument('--trials', type=int, default=50, help='Número de trials')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout en segundos')
    parser.add_argument('--quick', action='store_true', help='Modo rápido (25 trials, 2 min)')
    parser.add_argument('--intensive', action='store_true', help='Modo intensivo (100 trials, 10 min)')
    parser.add_argument('--gpu', action='store_true', help='Usar GPU')
    parser.add_argument('--algorithm', type=str, default='all', 
                        choices=['all', 'xgboost', 'lightgbm', 'catboost'],
                        help='Algoritmo a optimizar')
    
    args = parser.parse_args()
    
    # Ajustar parámetros según modo
    if args.quick:
        trials, timeout = 25, 120
    elif args.intensive:
        trials, timeout = 100, 600
    else:
        trials, timeout = args.trials, args.timeout
    
    print(f"⚡ Modo: {'Quick' if args.quick else 'Intensive' if args.intensive else 'Custom'}")
    print(f"🔥 GPU: {'Habilitado' if args.gpu else 'Deshabilitado'}")
    print(f"🎯 Algoritmo: {args.algorithm}")
    
    success = quick_optimize_real(trials, timeout, args.algorithm)
    sys.exit(0 if success else 1)
