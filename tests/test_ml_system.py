#!/usr/bin/env python3
"""
Script de prueba para el sistema de ML de criptomonedas
Verificar que todos los componentes funcionan correctamente
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar paths necesarios
sys.path.append('/home/exodia/Documentos/MachineLearning_TF/code/EDA/utils')
sys.path.append('.')

def test_data_loading():
    """Probar carga de datos"""
    print("📊 Probando carga de datos...")
    
    data_path = "/home/exodia/Documentos/MachineLearning_TF/data/crypto_ohlc_join.csv"
    if not os.path.exists(data_path):
        print(f"❌ No se encontró el archivo: {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    print(f"   ✅ Datos cargados: {df.shape}")
    print(f"   📊 Columnas: {list(df.columns)}")
    print(f"   🪙 Tokens únicos: {df['id'].nunique()}")
    
    # Verificar datos de low-cap
    low_cap = df[df['market_cap'] < 10_000_000]
    print(f"   💰 Observaciones low-cap (<10M): {len(low_cap)}")
    print(f"   🪙 Tokens low-cap: {low_cap['id'].nunique()}")
    
    return True

def test_feature_engineering():
    """Probar feature engineering"""
    print("\n🔧 Probando feature engineering...")
    
    try:
        from feature_engineering import create_ml_features, create_target_variables
        print("   ✅ Imports de feature engineering exitosos")
    except ImportError as e:
        print(f"   ❌ Error importando feature engineering: {e}")
        return False
    
    # Crear datos de prueba
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    test_data = []
    
    for token_id in ['BTC', 'ETH', 'TEST1']:
        for date in dates:
            test_data.append({
                'id': token_id,
                'date': date,
                'close': np.random.uniform(100, 1000),
                'volume': np.random.uniform(1000, 100000),
                'market_cap': np.random.uniform(1_000_000, 50_000_000),
                'narrative': np.random.choice(['meme', 'ai', 'gaming'])
            })
    
    df_test = pd.DataFrame(test_data)
    print(f"   📊 Dataset de prueba creado: {df_test.shape}")
    
    # Probar creación de targets
    try:
        df_with_targets = create_target_variables(df_test)
        target_cols = [col for col in df_with_targets.columns if 'future_return' in col or 'high_return' in col]
        print(f"   🎯 Variables objetivo creadas: {len(target_cols)}")
        print(f"   📊 Targets: {target_cols}")
        return True
    except Exception as e:
        print(f"   ❌ Error creando targets: {e}")
        return False

def test_ml_libraries():
    """Probar disponibilidad de librerías de ML"""
    print("\n🤖 Probando librerías de ML...")
    
    # Probar scikit-learn
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        print("   ✅ scikit-learn disponible")
    except ImportError:
        print("   ❌ scikit-learn no disponible")
        return False
    
    # Probar XGBoost
    try:
        import xgboost as xgb
        print("   ✅ XGBoost disponible")
    except ImportError:
        print("   ⚠️  XGBoost no disponible")
    
    # Probar LightGBM
    try:
        import lightgbm as lgb
        print("   ✅ LightGBM disponible")
    except ImportError:
        print("   ⚠️  LightGBM no disponible")
    
    # Probar CatBoost
    try:
        import catboost as cb
        print("   ✅ CatBoost disponible")
    except ImportError:
        print("   ⚠️  CatBoost no disponible")
    
    # Probar Optuna
    try:
        import optuna
        print("   ✅ Optuna disponible")
    except ImportError:
        print("   ⚠️  Optuna no disponible")
    
    return True

def test_simple_ml_pipeline():
    """Probar pipeline simple de ML"""
    print("\n🧪 Probando pipeline simple de ML...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        
        # Crear datos sintéticos
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        # Crear target con cierta correlación
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo simple
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        print("   📊 Resultados del modelo simple:")
        print(classification_report(y_test, y_pred))
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en pipeline simple: {e}")
        return False

def test_model_trainer():
    """Probar el trainer principal (sin datos reales)"""
    print("\n🎯 Probando CryptoMLTrainer...")
    
    try:
        from crypto_ml_trainer import CryptoMLTrainer
        print("   ✅ Import de CryptoMLTrainer exitoso")
        
        # Solo probar inicialización
        trainer = CryptoMLTrainer()
        print("   ✅ Inicialización exitosa")
        print(f"   📊 Configuraciones de modelos: {list(trainer.model_configs.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Error importando CryptoMLTrainer: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error inicializando CryptoMLTrainer: {e}")
        return False

def run_all_tests():
    """Ejecutar todas las pruebas"""
    print("🚀======================================================================")
    print("🧪 PRUEBAS DEL SISTEMA DE ML PARA CRIPTOMONEDAS")
    print("🚀======================================================================")
    
    tests = [
        ("Carga de datos", test_data_loading),
        ("Feature engineering", test_feature_engineering),
        ("Librerías de ML", test_ml_libraries),
        ("Pipeline simple", test_simple_ml_pipeline),
        ("Model trainer", test_model_trainer)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Error en {test_name}: {e}")
            results[test_name] = False
    
    # Resumen de resultados
    print("\n🚀======================================================================")
    print("📊 RESUMEN DE PRUEBAS")
    print("🚀======================================================================")
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado final: {passed}/{len(tests)} pruebas pasaron")
    
    if passed == len(tests):
        print("🎉 ¡Todos los componentes están listos!")
        print("🚀 Puedes ejecutar el entrenamiento completo")
    else:
        print("⚠️  Algunos componentes necesitan atención")
        print("💡 Revisa las dependencias e instalaciones")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎯 Para ejecutar el entrenamiento completo:")
        print("   python crypto_ml_trainer.py")
    else:
        print("\n🔧 Para instalar dependencias:")
        print("   pip install xgboost lightgbm catboost optuna")
        print("   conda install -c conda-forge scikit-learn pandas numpy")
