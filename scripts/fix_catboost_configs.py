#!/usr/bin/env python3
"""
Script para corregir configuraciones inconsistentes de CatBoost
Detecta y estandariza las configuraciones de modelos guardados
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

def analyze_config(config_path):
    """Analizar una configuración y detectar inconsistencias"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        analysis = {
            'path': config_path,
            'has_gpu_config': 'task_type' in config and config.get('task_type') == 'GPU',
            'has_devices': 'devices' in config,
            'has_metadata': '_metadata' in config,
            'config': config
        }
        
        return analysis
    except Exception as e:
        print(f"❌ Error leyendo {config_path}: {e}")
        return None

def standardize_config(config_path, gpu_available=True):
    """Estandarizar una configuración de CatBoost"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Determinar si se debe usar GPU basado en el nombre del archivo
        # Los archivos con timestamp 081050 parecen haber usado GPU
        use_gpu = gpu_available and ('081050' in str(config_path) or 'gpu' in str(config_path).lower())
        
        # Crear configuración estandarizada
        if use_gpu:
            # Configuración con GPU
            standard_config = {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'task_type': 'GPU',
                'devices': '0',
                'iterations': config.get('iterations', 1000),
                'learning_rate': config.get('learning_rate', 0.05),
                'depth': config.get('depth', 6),
                'l2_leaf_reg': config.get('l2_leaf_reg', 3),
                'random_state': config.get('random_state', 42),
                'verbose': config.get('verbose', False)
            }
        else:
            # Configuración con CPU
            standard_config = {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'task_type': 'CPU',
                'iterations': config.get('iterations', 1000),
                'learning_rate': config.get('learning_rate', 0.05),
                'depth': config.get('depth', 6),
                'l2_leaf_reg': config.get('l2_leaf_reg', 3),
                'random_state': config.get('random_state', 42),
                'verbose': config.get('verbose', False)
            }
        
        # Añadir metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        standard_config['_metadata'] = {
            'corrected_timestamp': timestamp,
            'original_config': config,
            'correction_reason': 'Standardization for consistency',
            'gpu_used': use_gpu,
            'corrected_by': 'config_standardizer_script'
        }
        
        return standard_config
        
    except Exception as e:
        print(f"❌ Error procesando {config_path}: {e}")
        return None

def main():
    print("🔧 CORRECTOR DE CONFIGURACIONES CATBOOST")
    print("=" * 50)
    
    # Buscar archivos de configuración de CatBoost
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Directorio 'models' no encontrado")
        return
    
    config_files = list(models_dir.glob("catboost_*_config.json"))
    
    if not config_files:
        print("❌ No se encontraron archivos de configuración de CatBoost")
        return
    
    print(f"📁 Encontrados {len(config_files)} archivos de configuración de CatBoost")
    
    # Analizar configuraciones existentes
    print("\n📊 ANÁLISIS DE CONFIGURACIONES EXISTENTES:")
    print("-" * 50)
    
    analyses = []
    for config_file in config_files:
        analysis = analyze_config(config_file)
        if analysis:
            analyses.append(analysis)
            gpu_status = "✅ GPU" if analysis['has_gpu_config'] else "🖥️ CPU"
            devices_status = "✅" if analysis['has_devices'] else "❌"
            metadata_status = "✅" if analysis['has_metadata'] else "❌"
            
            print(f"📄 {config_file.name}")
            print(f"   {gpu_status} | Devices: {devices_status} | Metadata: {metadata_status}")
    
    # Detectar inconsistencias
    gpu_configs = [a for a in analyses if a['has_gpu_config']]
    cpu_configs = [a for a in analyses if not a['has_gpu_config']]
    
    print(f"\n🔍 INCONSISTENCIAS DETECTADAS:")
    print(f"   📊 Configuraciones con GPU: {len(gpu_configs)}")
    print(f"   📊 Configuraciones con CPU: {len(cpu_configs)}")
    
    if len(gpu_configs) > 0 and len(cpu_configs) > 0:
        print("   ⚠️  INCONSISTENCIA: Mezcla de configuraciones GPU/CPU")
    
    # Preguntar si corregir
    print(f"\n🔧 ¿Deseas estandarizar las configuraciones? (y/N): ", end="")
    response = input().strip().lower()
    
    if response in ['y', 'yes', 'sí', 's']:
        print("\n🚀 ESTANDARIZANDO CONFIGURACIONES...")
        
        # Crear backups
        backup_dir = models_dir / "config_backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for config_file in config_files:
            # Crear backup
            backup_path = backup_dir / f"{config_file.stem}_backup_{timestamp}.json"
            backup_path.write_text(config_file.read_text())
            print(f"   💾 Backup: {backup_path}")
            
            # Estandarizar configuración
            new_config = standardize_config(config_file)
            if new_config:
                with open(config_file, 'w') as f:
                    json.dump(new_config, f, indent=2)
                print(f"   ✅ Estandarizado: {config_file}")
        
        print(f"\n✅ Configuraciones estandarizadas!")
        print(f"💾 Backups guardados en: {backup_dir}")
        
    else:
        print("❌ Operación cancelada")
    
    print("\n📋 RECOMENDACIONES:")
    print("   1. Usar el trainer actualizado que detecta GPU automáticamente")
    print("   2. Verificar que las configuraciones futuras sean consistentes")
    print("   3. Revisar logs de entrenamiento para confirmar el uso de GPU/CPU")

if __name__ == "__main__":
    main()
