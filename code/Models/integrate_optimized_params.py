#!/usr/bin/env python3
"""
Integrador de hiperparámetros optimizados
Actualiza automáticamente el trainer principal con los mejores parámetros encontrados
"""

import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

def load_best_configs(results_path: str = "../../optimization_results") -> Dict[str, Any]:
    """
    Cargar las mejores configuraciones encontradas
    
    Args:
        results_path: Ruta donde están los resultados
        
    Returns:
        Diccionario con las mejores configuraciones por modelo
    """
    results_dir = Path(results_path)
    
    # Buscar el archivo de mejores configuraciones más reciente
    config_files = list(results_dir.glob("best_configs_*.json"))
    
    if not config_files:
        print("❌ No se encontraron archivos de configuración")
        return {}
    
    # Tomar el más reciente
    latest_config = max(config_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📁 Cargando configuraciones desde: {latest_config.name}")
    
    with open(latest_config, 'r') as f:
        configs = json.load(f)
    
    print(f"✅ Configuraciones cargadas para {len(configs)} modelos")
    return configs

def backup_trainer(trainer_path: str = "crypto_ml_trainer.py") -> str:
    """
    Crear backup del trainer actual
    
    Args:
        trainer_path: Ruta al trainer
        
    Returns:
        Ruta del backup creado
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"crypto_ml_trainer_backup_{timestamp}.py"
    
    shutil.copy2(trainer_path, backup_path)
    print(f"💾 Backup creado: {backup_path}")
    
    return backup_path

def update_trainer_configs(configs: Dict[str, Any], trainer_path: str = "crypto_ml_trainer.py"):
    """
    Actualizar las configuraciones del trainer principal
    
    Args:
        configs: Configuraciones optimizadas
        trainer_path: Ruta al trainer
    """
    print("\n🔧======================================================================")
    print("🔧 ACTUALIZANDO CONFIGURACIONES DEL TRAINER")
    print("🔧======================================================================")
    
    # Leer archivo actual
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Crear nuevas configuraciones
    new_configs = {}
    
    for model_name, config in configs.items():
        params = config['parameters']
        
        if model_name == 'xgboost':
            new_configs['xgboost'] = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'random_state': 42,
                'verbosity': 0,
                **params
            }
        
        elif model_name == 'lightgbm':
            new_configs['lightgbm'] = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'random_state': 42,
                'verbosity': -1,
                **params
            }
        
        elif model_name == 'catboost':
            new_configs['catboost'] = {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'random_state': 42,
                'verbose': False,
                'allow_writing_files': False,
                **params
            }
    
    # Generar código de configuración
    config_code = "        # ======= CONFIGURACIONES OPTIMIZADAS CON OPTUNA =======\n"
    config_code += f"        # Fecha optimización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    config_code += "        # Mejores configuraciones encontradas automáticamente\n\n"
    
    for model_name, model_config in new_configs.items():
        config_code += f"        # {model_name.upper()} - AUC: {configs[model_name]['cv_score']:.4f}\n"
        config_code += f"        self.model_configs['{model_name}'] = {{\n"
        
        for param, value in model_config.items():
            if isinstance(value, str):
                config_code += f"            '{param}': '{value}',\n"
            else:
                config_code += f"            '{param}': {value},\n"
        
        config_code += "        }\n\n"
    
    # Buscar y reemplazar la sección de configuraciones
    import re
    
    # Patrón para encontrar las configuraciones actuales
    pattern = r"(        # Configuraciones por defecto.*?)(        self\.model_configs\['xgboost'\] = \{.*?\}.*?self\.model_configs\['catboost'\] = \{.*?\})"
    
    if re.search(pattern, content, re.DOTALL):
        # Reemplazar configuraciones existentes
        new_content = re.sub(
            pattern,
            lambda m: config_code.rstrip() + "\n",
            content,
            flags=re.DOTALL
        )
    else:
        # Si no encuentra el patrón, agregar al final del __init__
        init_pattern = r"(def __init__\(self.*?\n)(.*?)(\n    def )"
        new_content = re.sub(
            init_pattern,
            lambda m: m.group(1) + m.group(2) + "\n" + config_code + m.group(3),
            content,
            flags=re.DOTALL
        )
    
    # Escribir archivo actualizado
    with open(trainer_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Configuraciones actualizadas en el trainer!")
    
    # Mostrar resumen de cambios
    print("\n📊 RESUMEN DE CAMBIOS:")
    for model_name, config in configs.items():
        print(f"   🔹 {model_name.upper()}:")
        print(f"      📊 CV Score: {config['cv_score']:.4f}")
        if config.get('test_auc'):
            print(f"      📊 Test AUC: {config['test_auc']:.4f}")
        
        # Mostrar parámetros principales
        params = config['parameters']
        key_params = ['n_estimators', 'iterations', 'max_depth', 'depth', 'learning_rate']
        for param in key_params:
            if param in params:
                value = params[param]
                print(f"      🔧 {param}: {value}")

def create_optimized_trainer(configs: Dict[str, Any], 
                           output_name: str = "crypto_ml_trainer_optimized.py"):
    """
    Crear una versión optimizada del trainer sin modificar el original
    
    Args:
        configs: Configuraciones optimizadas
        output_name: Nombre del archivo de salida
    """
    print(f"\n📝 Creando trainer optimizado: {output_name}")
    
    # Copiar trainer original
    shutil.copy2("crypto_ml_trainer.py", output_name)
    
    # Actualizar configuraciones en la copia
    update_trainer_configs(configs, output_name)
    
    print(f"✅ Trainer optimizado creado: {output_name}")
    return output_name

def compare_configurations(configs: Dict[str, Any]):
    """
    Comparar configuraciones optimizadas con las por defecto
    """
    print("\n📊======================================================================")
    print("📊 COMPARACIÓN CON CONFIGURACIONES POR DEFECTO")
    print("📊======================================================================")
    
    # Configuraciones por defecto (del trainer original)
    default_configs = {
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1
        },
        'lightgbm': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 0
        },
        'catboost': {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'l2_leaf_reg': 3
        }
    }
    
    for model_name, optimized_config in configs.items():
        if model_name in default_configs:
            print(f"\n🔹 {model_name.upper()}:")
            print(f"   📊 Mejora de AUC: {optimized_config['cv_score']:.4f}")
            
            default_params = default_configs[model_name]
            optimized_params = optimized_config['parameters']
            
            print("   🔧 Cambios en parámetros:")
            
            for param in default_params:
                if param in optimized_params:
                    default_val = default_params[param]
                    optimized_val = optimized_params[param]
                    
                    if default_val != optimized_val:
                        change = ((optimized_val - default_val) / default_val * 100) if default_val != 0 else float('inf')
                        print(f"      {param}: {default_val} → {optimized_val:.4f} ({change:+.1f}%)")
                    else:
                        print(f"      {param}: {default_val} (sin cambio)")

def generate_integration_report(configs: Dict[str, Any], 
                              report_path: str = "integration_report.md"):
    """
    Generar reporte de integración
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# 🔧 Reporte de Integración de Hiperparámetros Optimizados

**Fecha:** {timestamp}
**Modelos optimizados:** {len(configs)}

## 📊 Resumen de Mejoras

"""
    
    for model_name, config in configs.items():
        report += f"""### {model_name.upper()}
- **CV Score:** {config['cv_score']:.4f}
- **Validation AUC:** {config.get('val_auc', 'N/A')}
- **Test AUC:** {config.get('test_auc', 'N/A')}
- **Fecha optimización:** {config.get('timestamp', 'N/A')}

**Parámetros optimizados:**
```json
{json.dumps(config['parameters'], indent=2)}
```

"""
    
    report += f"""
## 🚀 Próximos Pasos

1. **Validar configuraciones:** Ejecutar entrenamiento con nuevos parámetros
2. **Comparar performance:** Verificar mejoras en métricas de test
3. **Monitorear estabilidad:** Ejecutar múltiples entrenamientos
4. **Actualizar producción:** Implementar en sistema principal

---
*Reporte generado automáticamente por el sistema de optimización*
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Reporte generado: {report_path}")

def main():
    """
    Función principal de integración
    """
    print("🔧 INTEGRADOR DE HIPERPARÁMETROS OPTIMIZADOS")
    print("🔧 ACTUALIZACIÓN AUTOMÁTICA DEL TRAINER")
    print("🔧======================================================================")
    
    # Cargar mejores configuraciones
    configs = load_best_configs()
    
    if not configs:
        print("❌ No hay configuraciones para integrar")
        return
    
    # Mostrar configuraciones encontradas
    print(f"\n📋 Configuraciones encontradas:")
    for model_name, config in configs.items():
        print(f"   🤖 {model_name}: AUC {config['cv_score']:.4f}")
    
    # Crear backup del trainer original
    backup_path = backup_trainer()
    
    # Comparar configuraciones
    compare_configurations(configs)
    
    # Opción 1: Crear trainer optimizado (recomendado)
    optimized_trainer = create_optimized_trainer(configs)
    
    # Opción 2: Actualizar trainer original (comentado por seguridad)
    # update_trainer_configs(configs)
    
    # Generar reporte
    generate_integration_report(configs)
    
    print(f"\n✅======================================================================")
    print(f"✅ INTEGRACIÓN COMPLETADA")
    print(f"✅======================================================================")
    print(f"📝 Trainer optimizado: {optimized_trainer}")
    print(f"💾 Backup original: {backup_path}")
    print(f"📄 Reporte: integration_report.md")
    print(f"\n🚀 Para usar el trainer optimizado:")
    print(f"   python {optimized_trainer}")

if __name__ == "__main__":
    main()
