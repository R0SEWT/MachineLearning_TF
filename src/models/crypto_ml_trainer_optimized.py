#!/usr/bin/env python3
"""
Sistema de Entrenamiento de Modelos ML para Criptomonedas de Baja Capitalización
Basado en el informe de estrategia de modelado

Objetivo: Identificar tokens con potencial de retorno >100% en 30 días
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
import json

# Suprimir warnings
warnings.filterwarnings('ignore')

# Agregar path del EDA
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EDA', 'utils'))
from feature_engineering import create_ml_features, prepare_ml_dataset

# Imports de ML
try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import xgboost as xgb
    import lightgbm as lgb
    print("✅ Todas las librerías de ML importadas correctamente")
except ImportError as e:
    print(f"❌ Error importando librerías de ML: {e}")
    print("💡 Ejecutar: pip install xgboost lightgbm scikit-learn")
    sys.exit(1)

# Imports opcionales
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    print("✅ CatBoost disponible")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost no disponible, usando solo XGBoost y LightGBM")

try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("✅ Optuna disponible para optimización")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna no disponible, usando parámetros por defecto")

class CryptoMLTrainer:
    """
    Clase principal para entrenar modelos de ML para criptomonedas
    """
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            # Construir ruta absoluta al archivo de datos
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(current_dir, "..", "..", "data", "crypto_ohlc_join.csv")
        else:
            self.data_path = data_path
            
        self.models = {}
        self.feature_importance = {}
        self.results = {}
        self.scaler = StandardScaler()
        
        # Configuración de modelos
        self.model_configs = {
            'xgboost': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'gpu_hist',  # 🚀 Usar GPU
                'gpu_id': 0,               # 🚀 GPU ID
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'n_estimators': 1000,
                'random_state': 42,
                'early_stopping_rounds': 100
            },
            'lightgbm': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'device': 'gpu',          # 🚀 Usar GPU
                'gpu_platform_id': 0,    # 🚀 GPU Platform ID
                'gpu_device_id': 0,      # 🚀 GPU Device ID
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_estimators': 1000
            }
        }
        
        if CATBOOST_AVAILABLE:
            self.model_configs['catboost'] = {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'task_type': 'GPU',       # 🚀 Usar GPU
                'devices': '0',           # 🚀 GPU Device ID
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_state': 42,
                'verbose': False
            }
    
        # ======= CONFIGURACIONES OPTIMIZADAS CON OPTUNA =======
        # Fecha optimización: 2025-07-09 08:10:06
        # Mejores configuraciones encontradas automáticamente

        # LIGHTGBM - AUC: 0.9964
        self.model_configs['lightgbm'] = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbosity': -1,
            'n_estimators': 400,
            'max_depth': 8,
            'learning_rate': 0.028057329508564436,
            'subsample': 0.6066387374597534,
            'colsample_bytree': 0.8674210641290667,
            'reg_alpha': 0.6526109553222303,
            'reg_lambda': 4.001387239264024,
            'min_child_samples': 27,
            'num_leaves': 154,
        }

        # XGBOOST - AUC: 0.9970
        self.model_configs['xgboost'] = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42,
            'verbosity': 0,
            'n_estimators': 900,
            'max_depth': 11,
            'learning_rate': 0.011016324241359387,
            'subsample': 0.8541267258182842,
            'colsample_bytree': 0.6347652705269241,
            'reg_alpha': 0.5039395467781004,
            'reg_lambda': 0.24738607636418342,
            'min_child_weight': 1,
            'gamma': 0.028335681879852387,
        }


    def load_and_prepare_data(self, target_period: int = 30, min_market_cap: float = 0, 
                             max_market_cap: float = 10_000_000):
        """
        Cargar y preparar datos para entrenamiento
        
        Args:
            target_period: Período de predicción en días
            min_market_cap: Market cap mínimo
            max_market_cap: Market cap máximo (para low-cap)
        """
        print("🚀======================================================================")
        print("📊 CARGANDO Y PREPARANDO DATOS")
        print("🚀======================================================================")
        
        # Cargar datos
        print(f"📁 Cargando datos desde: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"No se encontró el archivo: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"   📊 Datos cargados: {df.shape}")
        print(f"   📅 Período: {df['date'].min()} a {df['date'].max()}")
        print(f"   🪙 Tokens únicos: {df['id'].nunique()}")
        
        # Filtrar por market cap (enfoque en low-cap)
        df_filtered = df[(df['market_cap'] >= min_market_cap) & 
                        (df['market_cap'] <= max_market_cap)].copy()
        
        print(f"   💰 Filtrado por market cap ${min_market_cap:,.0f} - ${max_market_cap:,.0f}")
        print(f"   📊 Datos filtrados: {df_filtered.shape}")
        print(f"   🪙 Tokens low-cap: {df_filtered['id'].nunique()}")
        
        # Mostrar distribución por narrativa
        if 'narrative' in df_filtered.columns:
            narrative_dist = df_filtered['narrative'].value_counts()
            print(f"   🎯 Distribución por narrativa:")
            for narrative, count in narrative_dist.items():
                print(f"      {narrative}: {count:,} observaciones")
        
        # Crear features de ML
        print("\n🔧 Creando features avanzadas...")
        df_features = create_ml_features(df_filtered, include_targets=True)
        
        # Preparar dataset para ML
        target_col = f'high_return_{target_period}d'
        print(f"\n🎯 Variable objetivo: {target_col}")
        
        X_train, X_test, y_train, y_test = prepare_ml_dataset(
            df_features, 
            target_col=target_col,
            min_history_days=60,
            test_size=0.2
        )
        
        # Guardar datasets
        self.X_train = X_train
        self.X_test = X_test  
        self.y_train = y_train
        self.y_test = y_test
        self.df_features = df_features
        self.target_col = target_col
        
        print("\n✅ Datos preparados exitosamente!")
        return self
    
    def train_xgboost(self, optimize: bool = False):
        """Entrenar modelo XGBoost"""
        print("\n🔥 Entrenando XGBoost...")
        
        if optimize and OPTUNA_AVAILABLE:
            print("   🎯 Optimizando hiperparámetros con Optuna...")
            study = optuna.create_study(direction='maximize')
            study.optimize(self._optimize_xgboost, n_trials=50)
            best_params = study.best_params
            print(f"   ✅ Mejores parámetros: {best_params}")
            
            # Actualizar configuración
            self.model_configs['xgboost'].update(best_params)
        
        # Entrenar modelo final
        model = xgb.XGBClassifier(**self.model_configs['xgboost'])
        
        # Fit con validación
        eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        model.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=False)
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = model.feature_importances_
        
        print("   ✅ XGBoost entrenado!")
        return model
    
    def train_lightgbm(self, optimize: bool = False):
        """Entrenar modelo LightGBM"""
        print("\n💡 Entrenando LightGBM...")
        
        if optimize and OPTUNA_AVAILABLE:
            print("   🎯 Optimizando hiperparámetros con Optuna...")
            study = optuna.create_study(direction='maximize')
            study.optimize(self._optimize_lightgbm, n_trials=50)
            best_params = study.best_params
            print(f"   ✅ Mejores parámetros: {best_params}")
            
            # Actualizar configuración
            self.model_configs['lightgbm'].update(best_params)
        
        # Entrenar modelo final
        model = lgb.LGBMClassifier(**self.model_configs['lightgbm'])
        
        # Fit con validación
        eval_set = [(self.X_test, self.y_test)]
        model.fit(self.X_train, self.y_train, eval_set=eval_set, callbacks=[lgb.early_stopping(100)])
        
        self.models['lightgbm'] = model
        self.feature_importance['lightgbm'] = model.feature_importances_
        
        print("   ✅ LightGBM entrenado!")
        return model
    
    def train_catboost(self, optimize: bool = False):
        """Entrenar modelo CatBoost (si está disponible)"""
        if not CATBOOST_AVAILABLE:
            print("   ⚠️  CatBoost no disponible, saltando...")
            return None
            
        print("\n🐱 Entrenando CatBoost...")
        
        # No usar features categóricas ya que todo fue convertido a numérico
        categorical_features = []
        
        # Verificar si hay columnas que podrían ser categóricas
        potential_cat_features = []
        for col in self.X_train.columns:
            # Solo considerar categóricas si tienen pocos valores únicos y son enteros
            unique_vals = self.X_train[col].nunique()
            if unique_vals <= 10 and self.X_train[col].dtype in ['int64', 'int32']:
                # Verificar si son enteros (no decimales)
                if (self.X_train[col] % 1 == 0).all():
                    potential_cat_features.append(col)
        
        print(f"   🔤 Features potencialmente categóricas: {len(potential_cat_features)}")
        if len(potential_cat_features) > 0:
            print(f"      {potential_cat_features}")
        
        if optimize and OPTUNA_AVAILABLE:
            print("   🎯 Optimizando hiperparámetros con Optuna...")
            study = optuna.create_study(direction='maximize')
            study.optimize(self._optimize_catboost, n_trials=50)
            best_params = study.best_params
            print(f"   ✅ Mejores parámetros: {best_params}")
            
            # Actualizar configuración
            self.model_configs['catboost'].update(best_params)
        
        # Entrenar modelo final
        model = cb.CatBoostClassifier(**self.model_configs['catboost'])
        
        # Fit con validación (sin features categóricas para evitar errores)
        model.fit(
            self.X_train, self.y_train,
            eval_set=(self.X_test, self.y_test),
            use_best_model=True
        )
        
        self.models['catboost'] = model
        self.feature_importance['catboost'] = model.feature_importances_
        
        print("   ✅ CatBoost entrenado!")
        return model
    
    def create_ensemble(self):
        """Crear ensemble de modelos"""
        print("\n🎭 Creando ensemble de modelos...")
        
        # Crear versiones simplificadas de los modelos sin early stopping para el ensemble
        ensemble_models = []
        
        if 'xgboost' in self.models and self.models['xgboost'] is not None:
            xgb_simple = xgb.XGBClassifier(
                tree_method='gpu_hist',  # 🚀 Usar GPU
                gpu_id=0,               # 🚀 GPU ID
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42,
                verbosity=0
            )
            ensemble_models.append(('xgboost', xgb_simple))
        
        if 'lightgbm' in self.models and self.models['lightgbm'] is not None:
            lgb_simple = lgb.LGBMClassifier(
                device='gpu',          # 🚀 Usar GPU
                gpu_platform_id=0,    # 🚀 GPU Platform ID
                gpu_device_id=0,      # 🚀 GPU Device ID
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42,
                verbosity=-1
            )
            ensemble_models.append(('lightgbm', lgb_simple))
        
        if 'catboost' in self.models and self.models['catboost'] is not None:
            cb_simple = cb.CatBoostClassifier(
                task_type='GPU',       # 🚀 Usar GPU
                devices='0',           # 🚀 GPU Device ID
                depth=6,
                learning_rate=0.1,
                iterations=100,
                random_state=42,
                verbose=False
            )
            ensemble_models.append(('catboost', cb_simple))
        
        if len(ensemble_models) < 2:
            print("   ⚠️  Se necesitan al menos 2 modelos para ensemble")
            return None
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        
        ensemble.fit(self.X_train, self.y_train)
        self.models['ensemble'] = ensemble
        
        print(f"   ✅ Ensemble creado con {len(ensemble_models)} modelos!")
        return ensemble
    
    def evaluate_models(self):
        """Evaluar todos los modelos entrenados"""
        print("\n🚀======================================================================")
        print("📊 EVALUACIÓN DE MODELOS")
        print("🚀======================================================================")
        
        for name, model in self.models.items():
            if model is None:
                continue
                
            print(f"\n🔍 Evaluando {name.upper()}...")
            
            # Predicciones
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Métricas
            auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
            
            print(f"   📊 Classification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['No High Return', 'High Return']))
            
            if auc:
                print(f"   🎯 AUC-ROC Score: {auc:.4f}")
            
            # Guardar resultados
            self.results[name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc': auc,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
            }
        
        # Mostrar ranking de modelos
        self._show_model_ranking()
        
        return self.results
    
    def save_models(self, output_dir: str = "../../models"):
        """Guardar modelos entrenados"""
        print(f"\n💾 Guardando modelos en: {output_dir}")
        
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            if model is None:
                continue
                
            model_path = f"{output_dir}/{name}_crypto_ml_{timestamp}"
            
            if name == 'xgboost':
                model.save_model(f"{model_path}.model")
                print(f"   ✅ {name} guardado: {model_path}.model")
                
            elif name == 'lightgbm':
                model.booster_.save_model(f"{model_path}.txt")
                print(f"   ✅ {name} guardado: {model_path}.txt")
                
            elif name == 'catboost' and CATBOOST_AVAILABLE:
                model.save_model(f"{model_path}.cbm")
                print(f"   ✅ {name} guardado: {model_path}.cbm")
            
            # Guardar configuración
            config_path = f"{model_path}_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.model_configs.get(name, {}), f, indent=2)
        
        # Guardar feature importance
        importance_path = f"{output_dir}/feature_importance_{timestamp}.json"
        with open(importance_path, 'w') as f:
            # Convertir numpy arrays a listas para JSON
            importance_dict = {}
            for name, importance in self.feature_importance.items():
                importance_dict[name] = {
                    'features': self.X_train.columns.tolist(),
                    'importance': importance.tolist() if hasattr(importance, 'tolist') else importance
                }
            json.dump(importance_dict, f, indent=2)
        
        print(f"   📊 Feature importance guardado: {importance_path}")
        
        return timestamp
    
    def get_feature_importance(self, top_n: int = 20):
        """Mostrar features más importantes"""
        print(f"\n🚀======================================================================")
        print(f"📊 TOP {top_n} FEATURES MÁS IMPORTANTES")
        print("🚀======================================================================")
        
        for name, importance in self.feature_importance.items():
            if importance is None:
                continue
                
            print(f"\n🔍 {name.upper()}:")
            
            # Crear DataFrame con importancia
            feature_df = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Mostrar top features
            for i, (_, row) in enumerate(feature_df.head(top_n).iterrows()):
                print(f"   {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    def predict_opportunities(self, df_current: pd.DataFrame = None, 
                            model_name: str = 'ensemble', threshold: float = 0.75):
        """
        Predecir oportunidades actuales
        
        Args:
            df_current: DataFrame con datos actuales (opcional, usa test por defecto)
            model_name: Nombre del modelo a usar
            threshold: Umbral de probabilidad para oportunidades
            
        Returns:
            DataFrame con oportunidades detectadas
        """
        print(f"\n🔍 Detectando oportunidades con {model_name}...")
        
        if model_name not in self.models or self.models[model_name] is None:
            print(f"   ❌ Modelo {model_name} no disponible")
            return None
        
        # Usar datos de test si no se proveen datos actuales
        if df_current is None:
            X_current = self.X_test
            print("   📊 Usando datos de test para demostración")
        else:
            # Procesar datos actuales (sería el flujo en producción)
            df_features = create_ml_features(df_current, include_targets=False)
            feature_cols = [col for col in self.X_train.columns if col in df_features.columns]
            X_current = df_features[feature_cols]
        
        # Predicciones
        model = self.models[model_name]
        probabilities = model.predict_proba(X_current)[:, 1]
        predictions = model.predict(X_current)
        
        # Crear DataFrame de oportunidades
        opportunities = pd.DataFrame({
            'index': X_current.index,
            'prediction': predictions,
            'probability': probabilities,
            'high_confidence': (probabilities >= threshold).astype(int)
        })
        
        # Filtrar oportunidades de alta confianza
        high_conf_opps = opportunities[opportunities['high_confidence'] == 1]
        
        print(f"   🎯 Oportunidades detectadas: {len(high_conf_opps)} de {len(opportunities)}")
        print(f"   📊 Probabilidad promedio: {probabilities.mean():.3f}")
        print(f"   🎯 Probabilidad máxima: {probabilities.max():.3f}")
        
        return opportunities.sort_values('probability', ascending=False)
    
    def _optimize_xgboost(self, trial):
        """Función objetivo para optimización XGBoost con Optuna"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        }
        
        params.update(self.model_configs['xgboost'])
        params['n_estimators'] = 100  # Menos iteraciones para optimización
        
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='roc_auc')
        return scores.mean()
    
    def _optimize_lightgbm(self, trial):
        """Función objetivo para optimización LightGBM con Optuna"""
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        
        params.update(self.model_configs['lightgbm'])
        params['n_estimators'] = 100  # Menos iteraciones para optimización
        
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='roc_auc')
        return scores.mean()
    
    def _optimize_catboost(self, trial):
        """Función objetivo para optimización CatBoost con Optuna"""
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        }
        
        params.update(self.model_configs['catboost'])
        params['iterations'] = 100  # Menos iteraciones para optimización
        
        model = cb.CatBoostClassifier(**params)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='roc_auc')
        return scores.mean()
    
    def _show_model_ranking(self):
        """Mostrar ranking de modelos por AUC"""
        print(f"\n🏆 RANKING DE MODELOS:")
        
        model_aucs = [(name, results.get('auc', 0)) for name, results in self.results.items() 
                     if results.get('auc') is not None]
        model_aucs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, auc) in enumerate(model_aucs):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "   "
            print(f"   {medal} {i+1}. {name.upper():<15} AUC: {auc:.4f}")

def run_training_pipeline(optimize_models: bool = True, max_market_cap: float = 10_000_000):
    """
    Ejecutar pipeline completo de entrenamiento
    
    Args:
        optimize_models: Si optimizar hiperparámetros
        max_market_cap: Market cap máximo para filtrar (low-cap)
    """
    print("🚀======================================================================")
    print("🤖 SISTEMA DE ML PARA CRIPTOMONEDAS DE BAJA CAPITALIZACIÓN")
    print("🚀======================================================================")
    print(f"🎯 Objetivo: Identificar tokens con retorno >100% en 30 días")
    print(f"💰 Enfoque: Market cap < ${max_market_cap:,.0f}")
    print(f"⚙️  Optimización: {'Activada' if optimize_models else 'Desactivada'}")
    print("🚀======================================================================")
    
    # Inicializar trainer
    trainer = CryptoMLTrainer()
    
    # Cargar y preparar datos
    trainer.load_and_prepare_data(
        target_period=30,
        max_market_cap=max_market_cap
    )
    
    # Entrenar modelos
    trainer.train_xgboost(optimize=optimize_models)
    trainer.train_lightgbm(optimize=optimize_models)
    
    if CATBOOST_AVAILABLE:
        trainer.train_catboost(optimize=optimize_models)
    
    # Crear ensemble
    trainer.create_ensemble()
    
    # Evaluar modelos
    trainer.evaluate_models()
    
    # Mostrar feature importance
    trainer.get_feature_importance(top_n=15)
    
    # Detectar oportunidades
    opportunities = trainer.predict_opportunities(threshold=0.75)
    
    # Guardar modelos
    timestamp = trainer.save_models()
    
    print("\n🚀======================================================================")
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("🚀======================================================================")
    print(f"📊 Modelos entrenados: {len(trainer.models)}")
    print(f"💾 Modelos guardados con timestamp: {timestamp}")
    print(f"🎯 Oportunidades detectadas: {len(opportunities[opportunities['high_confidence']==1])}")
    print("🚀======================================================================")
    
    return trainer, opportunities

if __name__ == "__main__":
    # Ejecutar pipeline principal
    trainer, opportunities = run_training_pipeline(
        optimize_models=False,  # Cambiar a True para optimización con Optuna
        max_market_cap=10_000_000  # 10M para low-cap
    )
    
    print("\n🎯 Top 10 oportunidades detectadas:")
    top_opportunities = opportunities.head(10)
    for i, (_, row) in enumerate(top_opportunities.iterrows()):
        print(f"   {i+1:2d}. Índice {int(row['index']):4d}: {row['probability']:.3f} probabilidad")
