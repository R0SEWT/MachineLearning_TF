#!/usr/bin/env python3
"""
🚀 Script de Optimización por Lotes - Fase 5
============================================

Script para optimización automática por lotes de múltiples configuraciones
y datasets, con paralelización y gestión avanzada de recursos.

Autor: Sistema de Optimización IA
Fecha: 2025-01-09 (Fase 5 - Organización)
"""

import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import concurrent.futures
from dataclasses import dataclass, asdict
import time

# Agregar directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from config.optimization_config import OptimizationConfig, get_quick_config, get_production_config
from core.optimizer import HyperparameterOptimizer
from analysis.results_analyzer import ResultsAnalyzer
from utils.logging_setup import setup_logging, get_logger
from utils.import_manager import test_critical_imports


@dataclass
class BatchExperiment:
    """Configuración de un experimento en lote"""
    name: str
    data_path: str
    config_name: str
    enabled_models: List[str]
    custom_params: Dict[str, Any]
    priority: int = 1
    max_retries: int = 3


class BatchOptimizer:
    """
    Optimizador por lotes para múltiples experimentos.
    
    Permite ejecutar múltiples configuraciones de optimización
    de forma automática, con paralelización y gestión de recursos.
    """
    
    def __init__(self, output_dir: str = "./batch_results", max_workers: int = 2):
        """
        Inicializar optimizador por lotes.
        
        Args:
            output_dir: Directorio base para resultados
            max_workers: Número máximo de workers paralelos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        # Configurar logging
        self.logger = get_logger("batch_optimizer")
        
        # Estado del batch
        self.experiments: List[BatchExperiment] = []
        self.results: Dict[str, Dict[str, Any]] = {}
        self.failed_experiments: List[str] = []
        
        self.logger.info(f"🚀 BatchOptimizer inicializado: {output_dir}")
    
    def load_experiments_from_config(self, config_file: str):
        """
        Cargar experimentos desde archivo de configuración.
        
        Args:
            config_file: Archivo YAML o JSON con configuraciones
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_file}")
        
        # Cargar según extensión
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Formato de archivo no soportado: {config_path.suffix}")
        
        # Convertir a experimentos
        experiments_data = config_data.get('experiments', [])
        
        for exp_data in experiments_data:
            experiment = BatchExperiment(
                name=exp_data['name'],
                data_path=exp_data['data_path'],
                config_name=exp_data.get('config', 'quick'),
                enabled_models=exp_data.get('models', ['xgboost', 'lightgbm', 'catboost']),
                custom_params=exp_data.get('params', {}),
                priority=exp_data.get('priority', 1),
                max_retries=exp_data.get('max_retries', 3)
            )
            
            self.experiments.append(experiment)
        
        self.logger.info(f"📋 Cargados {len(self.experiments)} experimentos desde {config_file}")
    
    def add_experiment(self, experiment: BatchExperiment):
        """Agregar experimento individual"""
        self.experiments.append(experiment)
        self.logger.info(f"➕ Experimento agregado: {experiment.name}")
    
    def _create_config_for_experiment(self, experiment: BatchExperiment) -> OptimizationConfig:
        """Crear configuración para un experimento específico"""
        # Configuración base
        if experiment.config_name == "quick":
            config = get_quick_config()
        elif experiment.config_name == "production":
            config = get_production_config()
        else:
            config = get_quick_config()
        
        # Aplicar modelos específicos
        config.enabled_models = experiment.enabled_models
        
        # Aplicar parámetros personalizados
        for key, value in experiment.custom_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Configurar directorios específicos del experimento
        exp_output_dir = self.output_dir / experiment.name
        config.results_dir = str(exp_output_dir)
        config.log_dir = str(exp_output_dir / "logs")
        config.cache_dir = str(exp_output_dir / "cache")
        
        return config
    
    def _run_single_experiment(self, experiment: BatchExperiment) -> Dict[str, Any]:
        """
        Ejecutar un experimento individual.
        
        Args:
            experiment: Configuración del experimento
            
        Returns:
            Resultado del experimento
        """
        experiment_logger = get_logger(f"exp_{experiment.name}")
        experiment_logger.info(f"🚀 Iniciando experimento: {experiment.name}")
        
        start_time = time.time()
        result = {
            "experiment_name": experiment.name,
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "data_path": experiment.data_path,
            "config_name": experiment.config_name,
            "enabled_models": experiment.enabled_models,
            "error": None,
            "optimization_result": None
        }
        
        try:
            # Verificar archivo de datos
            if not Path(experiment.data_path).exists():
                raise FileNotFoundError(f"Archivo de datos no encontrado: {experiment.data_path}")
            
            # Crear configuración
            config = self._create_config_for_experiment(experiment)
            
            # Crear optimizador
            optimizer = HyperparameterOptimizer(config)
            
            # Ejecutar optimización
            experiment_id = f"batch_{experiment.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            opt_result = optimizer.optimize_all_models(experiment.data_path, experiment_id)
            
            # Guardar resultados
            optimizer.save_results(opt_result, config.results_dir)
            
            # Actualizar resultado
            result.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "duration": time.time() - start_time,
                "experiment_id": experiment_id,
                "best_model": opt_result.best_model,
                "best_score": opt_result.best_score,
                "total_time": opt_result.total_time,
                "n_models": len(opt_result.model_results),
                "optimization_result": {
                    "experiment_id": opt_result.experiment_id,
                    "best_model": opt_result.best_model,
                    "best_score": opt_result.best_score,
                    "total_time": opt_result.total_time,
                    "model_results": {
                        name: {
                            "best_score": res.best_score,
                            "best_params": res.best_params,
                            "n_trials": res.n_trials,
                            "optimization_time": res.optimization_time
                        }
                        for name, res in opt_result.model_results.items()
                    }
                }
            })
            
            experiment_logger.info(f"✅ Experimento completado: {experiment.name}")
            experiment_logger.info(f"🏆 Mejor modelo: {opt_result.best_model} ({opt_result.best_score:.4f})")
            
        except Exception as e:
            result.update({
                "status": "failed",
                "end_time": datetime.now().isoformat(),
                "duration": time.time() - start_time,
                "error": str(e)
            })
            
            experiment_logger.error(f"❌ Experimento falló: {experiment.name} - {e}")
            self.failed_experiments.append(experiment.name)
        
        return result
    
    def run_batch_sequential(self) -> Dict[str, Dict[str, Any]]:
        """
        Ejecutar experimentos de forma secuencial.
        
        Returns:
            Diccionario con resultados de todos los experimentos
        """
        self.logger.info(f"🔄 Ejecutando {len(self.experiments)} experimentos secuencialmente")
        
        # Ordenar por prioridad
        sorted_experiments = sorted(self.experiments, key=lambda x: x.priority, reverse=True)
        
        for i, experiment in enumerate(sorted_experiments, 1):
            self.logger.info(f"📊 Progreso: {i}/{len(sorted_experiments)} - {experiment.name}")
            
            # Ejecutar con reintentos
            for attempt in range(experiment.max_retries):
                try:
                    result = self._run_single_experiment(experiment)
                    self.results[experiment.name] = result
                    
                    if result["status"] == "completed":
                        break
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  Intento {attempt + 1} falló para {experiment.name}: {e}")
                    
                    if attempt == experiment.max_retries - 1:
                        self.results[experiment.name] = {
                            "experiment_name": experiment.name,
                            "status": "failed_all_retries",
                            "error": str(e),
                            "attempts": experiment.max_retries
                        }
        
        return self.results
    
    def run_batch_parallel(self) -> Dict[str, Dict[str, Any]]:
        """
        Ejecutar experimentos en paralelo.
        
        Returns:
            Diccionario con resultados de todos los experimentos
        """
        self.logger.info(f"⚡ Ejecutando {len(self.experiments)} experimentos en paralelo ({self.max_workers} workers)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Enviar todos los experimentos
            future_to_experiment = {
                executor.submit(self._run_single_experiment, exp): exp 
                for exp in self.experiments
            }
            
            # Recopilar resultados
            completed = 0
            for future in concurrent.futures.as_completed(future_to_experiment):
                experiment = future_to_experiment[future]
                completed += 1
                
                try:
                    result = future.result()
                    self.results[experiment.name] = result
                    
                    self.logger.info(f"✅ Completado ({completed}/{len(self.experiments)}): {experiment.name}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error en {experiment.name}: {e}")
                    self.results[experiment.name] = {
                        "experiment_name": experiment.name,
                        "status": "failed",
                        "error": str(e)
                    }
                    self.failed_experiments.append(experiment.name)
        
        return self.results
    
    def save_batch_results(self, filename: str = "batch_results.json"):
        """Guardar resultados del batch"""
        results_file = self.output_dir / filename
        
        batch_summary = {
            "batch_execution": {
                "timestamp": datetime.now().isoformat(),
                "total_experiments": len(self.experiments),
                "completed_experiments": len([r for r in self.results.values() if r.get("status") == "completed"]),
                "failed_experiments": len(self.failed_experiments),
                "output_directory": str(self.output_dir)
            },
            "experiments": self.results,
            "failed_experiments": self.failed_experiments
        }
        
        with open(results_file, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        self.logger.info(f"💾 Resultados del batch guardados: {results_file}")
    
    def generate_batch_report(self) -> str:
        """Generar reporte del batch"""
        if not self.results:
            return "No hay resultados para reportar."
        
        completed = [r for r in self.results.values() if r.get("status") == "completed"]
        failed = [r for r in self.results.values() if r.get("status") != "completed"]
        
        report_lines = [
            "# 📊 Reporte de Optimización por Lotes",
            "",
            f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total de experimentos:** {len(self.experiments)}",
            f"**Completados:** {len(completed)}",
            f"**Fallidos:** {len(failed)}",
            f"**Tasa de éxito:** {len(completed)/len(self.experiments)*100:.1f}%",
            "",
            "## 🏆 Experimentos Completados",
            ""
        ]
        
        if completed:
            report_lines.append("| Experimento | Mejor Modelo | Score | Tiempo (min) |")
            report_lines.append("|-------------|--------------|-------|--------------|")
            
            for result in completed:
                opt_result = result.get("optimization_result", {})
                report_lines.append(
                    f"| {result['experiment_name']} | "
                    f"{opt_result.get('best_model', 'N/A')} | "
                    f"{opt_result.get('best_score', 0):.4f} | "
                    f"{result.get('duration', 0)/60:.1f} |"
                )
            
            report_lines.append("")
        
        if failed:
            report_lines.extend([
                "## ❌ Experimentos Fallidos",
                ""
            ])
            
            for result in failed:
                report_lines.append(f"- **{result['experiment_name']}**: {result.get('error', 'Error desconocido')}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


def create_sample_config():
    """Crear archivo de configuración de ejemplo"""
    sample_config = {
        "experiments": [
            {
                "name": "quick_test",
                "data_path": "../../../data/crypto_ohlc_join.csv",
                "config": "quick",
                "models": ["xgboost", "lightgbm"],
                "params": {
                    "model_trials": {"xgboost": 50, "lightgbm": 50}
                },
                "priority": 1
            },
            {
                "name": "production_test",
                "data_path": "../../../data/crypto_ohlc_join.csv",
                "config": "production",
                "models": ["xgboost", "lightgbm", "catboost"],
                "params": {
                    "optimization_timeout": 3600
                },
                "priority": 2
            }
        ]
    }
    
    with open("batch_config_sample.yaml", 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    print("✅ Archivo de configuración de ejemplo creado: batch_config_sample.yaml")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="🚀 Optimización por Lotes - Fase 5",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Archivo de configuración YAML/JSON con experimentos"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./batch_results",
        help="Directorio de salida (default: ./batch_results)"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Ejecutar experimentos en paralelo"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="Número de workers paralelos (default: 2)"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Crear archivo de configuración de ejemplo"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Logging verbose"
    )
    
    args = parser.parse_args()
    
    # Crear ejemplo
    if args.create_sample:
        create_sample_config()
        return 0
    
    # Verificar configuración
    if not args.config:
        print("❌ Archivo de configuración requerido")
        print("💡 Crear ejemplo: python batch_optimization.py --create-sample")
        print("💡 Ejecutar: python batch_optimization.py --config batch_config.yaml")
        return 1
    
    # Configurar logging
    log_config = {
        "level": "DEBUG" if args.verbose else "INFO",
        "log_dir": str(Path(args.output) / "logs"),
        "enable_console_logging": True,
        "enable_file_logging": True
    }
    setup_logging(log_config)
    
    # Verificar dependencias
    if not test_critical_imports():
        print("❌ Faltan dependencias críticas")
        return 1
    
    # Crear optimizador por lotes
    try:
        batch_optimizer = BatchOptimizer(args.output, args.workers)
        batch_optimizer.load_experiments_from_config(args.config)
        
        # Ejecutar experimentos
        start_time = time.time()
        
        if args.parallel:
            results = batch_optimizer.run_batch_parallel()
        else:
            results = batch_optimizer.run_batch_sequential()
        
        total_time = time.time() - start_time
        
        # Guardar resultados
        batch_optimizer.save_batch_results()
        
        # Generar reporte
        report = batch_optimizer.generate_batch_report()
        report_file = Path(args.output) / "batch_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Resumen final
        completed = len([r for r in results.values() if r.get("status") == "completed"])
        failed = len(batch_optimizer.failed_experiments)
        
        print(f"\n🎯 Batch completado en {total_time/60:.1f} minutos")
        print(f"✅ Experimentos completados: {completed}")
        print(f"❌ Experimentos fallidos: {failed}")
        print(f"📁 Resultados en: {args.output}")
        print(f"📄 Reporte en: {report_file}")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        print(f"❌ Error ejecutando batch: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
