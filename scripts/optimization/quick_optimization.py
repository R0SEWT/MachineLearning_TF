#!/usr/bin/env python3
"""
Script de configuración rápida para optimización de hiperparámetros
Permite ejecutar optimizaciones específicas y personalizadas
"""

import argparse
import sys
import os
from datetime import datetime

# Agregar paths necesarios
sys.path.append('/home/exodia/Documentos/MachineLearning_TF/code/EDA/utils')

def quick_xgboost_optimization(trials=30, timeout=600):
    """
    Optimización rápida solo para XGBoost
    """
    from crypto_hyperparameter_optimizer import CryptoHyperparameterOptimizer
    
    print("🔥 OPTIMIZACIÓN RÁPIDA - XGBOOST")
    print("=" * 50)
    
    optimizer = CryptoHyperparameterOptimizer()
    optimizer.load_and_prepare_data()
    
    study = optimizer.optimize_xgboost(n_trials=trials, timeout=timeout)
    optimizer.evaluate_best_models()
    optimizer.save_optimization_summary()
    
    print(f"\n✅ XGBoost optimizado!")
    print(f"🏆 Mejor AUC: {study.best_value:.4f}")
    print(f"🔧 Mejores parámetros: {study.best_params}")
    
    return study

def quick_lightgbm_optimization(trials=30, timeout=600):
    """
    Optimización rápida solo para LightGBM
    """
    from crypto_hyperparameter_optimizer import CryptoHyperparameterOptimizer
    
    print("💡 OPTIMIZACIÓN RÁPIDA - LIGHTGBM")
    print("=" * 50)
    
    optimizer = CryptoHyperparameterOptimizer()
    optimizer.load_and_prepare_data()
    
    study = optimizer.optimize_lightgbm(n_trials=trials, timeout=timeout)
    optimizer.evaluate_best_models()
    optimizer.save_optimization_summary()
    
    print(f"\n✅ LightGBM optimizado!")
    print(f"🏆 Mejor AUC: {study.best_value:.4f}")
    print(f"🔧 Mejores parámetros: {study.best_params}")
    
    return study

def quick_catboost_optimization(trials=30, timeout=600):
    """
    Optimización rápida solo para CatBoost
    """
    from crypto_hyperparameter_optimizer import CryptoHyperparameterOptimizer
    
    print("🐱 OPTIMIZACIÓN RÁPIDA - CATBOOST")
    print("=" * 50)
    
    optimizer = CryptoHyperparameterOptimizer()
    optimizer.load_and_prepare_data()
    
    study = optimizer.optimize_catboost(n_trials=trials, timeout=timeout)
    optimizer.evaluate_best_models()
    optimizer.save_optimization_summary()
    
    print(f"\n✅ CatBoost optimizado!")
    print(f"🏆 Mejor AUC: {study.best_value:.4f}")
    print(f"🔧 Mejores parámetros: {study.best_params}")
    
    return study

def full_optimization(trials=50, timeout_per_model=1800):
    """
    Optimización completa de todos los modelos
    """
    from crypto_hyperparameter_optimizer import CryptoHyperparameterOptimizer
    
    print("🚀 OPTIMIZACIÓN COMPLETA - TODOS LOS MODELOS")
    print("=" * 60)
    
    optimizer = CryptoHyperparameterOptimizer()
    optimizer.load_and_prepare_data()
    
    optimizer.optimize_all_models(
        n_trials=trials,
        timeout_per_model=timeout_per_model
    )
    
    optimizer.evaluate_best_models()
    optimizer.generate_visualizations()
    optimizer.print_final_summary()
    
    return optimizer

def experimental_optimization(trials=100, timeout_per_model=3600):
    """
    Optimización experimental con más trials y tiempo
    Para encontrar los mejores hiperparámetros posibles
    """
    from crypto_hyperparameter_optimizer import CryptoHyperparameterOptimizer
    
    print("🧪 OPTIMIZACIÓN EXPERIMENTAL - BÚSQUEDA EXTENSIVA")
    print("=" * 70)
    print(f"⚠️  Esta optimización puede tomar varias horas")
    print(f"🔢 Trials por modelo: {trials}")
    print(f"⏰ Timeout por modelo: {timeout_per_model/60:.0f} minutos")
    
    optimizer = CryptoHyperparameterOptimizer()
    optimizer.load_and_prepare_data()
    
    optimizer.optimize_all_models(
        n_trials=trials,
        timeout_per_model=timeout_per_model
    )
    
    optimizer.evaluate_best_models()
    optimizer.generate_visualizations()
    optimizer.print_final_summary()
    
    return optimizer

def load_and_compare_studies():
    """
    Cargar estudios previos y compararlos
    """
    import pickle
    import json
    from pathlib import Path
    
    results_path = Path("../../optimization_results")
    
    print("📊 COMPARANDO ESTUDIOS PREVIOS")
    print("=" * 40)
    
    # Buscar archivos de resumen
    summary_files = list(results_path.glob("optimization_summary_*.json"))
    
    if not summary_files:
        print("❌ No se encontraron estudios previos")
        return
    
    print(f"📁 Encontrados {len(summary_files)} estudios:")
    
    all_results = []
    for summary_file in sorted(summary_files):
        with open(summary_file, 'r') as f:
            data = json.load(f)
            all_results.append(data)
        
        print(f"\n📅 {data['timestamp']}:")
        for model, score in data['best_scores'].items():
            print(f"   {model}: {score:.4f}")
    
    # Encontrar el mejor resultado general
    best_overall = None
    best_score = 0
    
    for result in all_results:
        for model, score in result['best_scores'].items():
            if score > best_score:
                best_score = score
                best_overall = (result['timestamp'], model, score, result['best_params'][model])
    
    if best_overall:
        timestamp, model, score, params = best_overall
        print(f"\n🏆 MEJOR RESULTADO HISTÓRICO:")
        print(f"   📅 Fecha: {timestamp}")
        print(f"   🤖 Modelo: {model}")
        print(f"   📊 AUC: {score:.4f}")
        print(f"   🔧 Parámetros: {params}")

def main():
    parser = argparse.ArgumentParser(description='Optimización de hiperparámetros para criptomonedas')
    parser.add_argument('--mode', choices=['quick-xgb', 'quick-lgb', 'quick-cat', 'full', 'experimental', 'compare'], 
                       default='full', help='Modo de optimización')
    parser.add_argument('--trials', type=int, default=50, help='Número de trials por modelo')
    parser.add_argument('--timeout', type=int, default=1800, help='Timeout en segundos por modelo')
    
    args = parser.parse_args()
    
    print(f"🚀 Iniciando optimización en modo: {args.mode}")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.mode == 'quick-xgb':
            quick_xgboost_optimization(args.trials, args.timeout)
        elif args.mode == 'quick-lgb':
            quick_lightgbm_optimization(args.trials, args.timeout)
        elif args.mode == 'quick-cat':
            quick_catboost_optimization(args.trials, args.timeout)
        elif args.mode == 'full':
            full_optimization(args.trials, args.timeout)
        elif args.mode == 'experimental':
            experimental_optimization(args.trials, args.timeout)
        elif args.mode == 'compare':
            load_and_compare_studies()
            
    except KeyboardInterrupt:
        print("\n⚠️  Optimización interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la optimización: {e}")
        raise

if __name__ == "__main__":
    main()
