#!/usr/bin/env python3
"""
Test de GPU para Modelos de Machine Learning
Verificar que XGBoost, LightGBM y CatBoost pueden usar GPU
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("🚀 PROBANDO CAPACIDADES DE GPU PARA ML")
print("=" * 50)

# Verificar CUDA
print("\n1. 🔍 VERIFICANDO CUDA...")
try:
    import torch
    print(f"   ✅ PyTorch disponible: {torch.__version__}")
    print(f"   🎯 CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   🔥 GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"   💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print("   ⚠️ PyTorch no disponible")

# Test XGBoost GPU
print("\n2. 🔥 PROBANDO XGBOOST GPU...")
try:
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Crear datos sintéticos
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test GPU
    print("   🚀 Creando modelo XGBoost con GPU...")
    model_gpu = xgb.XGBClassifier(
        tree_method='gpu_hist',
        gpu_id=0,
        n_estimators=100,
        random_state=42,
        verbosity=0
    )
    
    import time
    start = time.time()
    model_gpu.fit(X_train, y_train)
    gpu_time = time.time() - start
    print(f"   ✅ XGBoost GPU completado en {gpu_time:.2f}s")
    
    # Test CPU para comparar
    print("   🖥️ Comparando con CPU...")
    model_cpu = xgb.XGBClassifier(
        tree_method='hist',
        n_estimators=100,
        random_state=42,
        verbosity=0
    )
    
    start = time.time()
    model_cpu.fit(X_train, y_train)
    cpu_time = time.time() - start
    print(f"   🖥️ XGBoost CPU completado en {cpu_time:.2f}s")
    print(f"   ⚡ Aceleración GPU: {cpu_time/gpu_time:.1f}x más rápido")
    
except Exception as e:
    print(f"   ❌ Error con XGBoost: {e}")

# Test LightGBM GPU
print("\n3. 💡 PROBANDO LIGHTGBM GPU...")
try:
    import lightgbm as lgb
    
    print("   🚀 Creando modelo LightGBM con GPU...")
    model_gpu = lgb.LGBMClassifier(
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
    
    start = time.time()
    model_gpu.fit(X_train, y_train)
    gpu_time = time.time() - start
    print(f"   ✅ LightGBM GPU completado en {gpu_time:.2f}s")
    
    # Test CPU
    model_cpu = lgb.LGBMClassifier(
        device='cpu',
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
    
    start = time.time()
    model_cpu.fit(X_train, y_train)
    cpu_time = time.time() - start
    print(f"   🖥️ LightGBM CPU completado en {cpu_time:.2f}s")
    print(f"   ⚡ Aceleración GPU: {cpu_time/gpu_time:.1f}x más rápido")
    
except Exception as e:
    print(f"   ❌ Error con LightGBM: {e}")

# Test CatBoost GPU
print("\n4. 🐱 PROBANDO CATBOOST GPU...")
try:
    import catboost as cb
    
    print("   🚀 Creando modelo CatBoost con GPU...")
    model_gpu = cb.CatBoostClassifier(
        task_type='GPU',
        devices='0',
        iterations=100,
        random_state=42,
        verbose=False
    )
    
    start = time.time()
    model_gpu.fit(X_train, y_train)
    gpu_time = time.time() - start
    print(f"   ✅ CatBoost GPU completado en {gpu_time:.2f}s")
    
    # Test CPU
    model_cpu = cb.CatBoostClassifier(
        task_type='CPU',
        iterations=100,
        random_state=42,
        verbose=False
    )
    
    start = time.time()
    model_cpu.fit(X_train, y_train)
    cpu_time = time.time() - start
    print(f"   🖥️ CatBoost CPU completado en {cpu_time:.2f}s")
    print(f"   ⚡ Aceleración GPU: {cpu_time/gpu_time:.1f}x más rápido")
    
except Exception as e:
    print(f"   ❌ Error con CatBoost: {e}")

print("\n🎉 RESUMEN DE PRUEBAS GPU COMPLETADO")
print("=" * 50)
print("Si ves errores arriba, algunos modelos pueden no estar")
print("configurados para GPU o necesitas drivers/dependencias.")
print("Si funcionan, ¡estás listo para entrenar con GPU! 🚀")
