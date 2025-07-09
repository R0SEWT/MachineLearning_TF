"""
🚀 Analizador de Resultados Mejorado - Fase 5
============================================

Sistema enterprise-ready de análisis y visualización de resultados que
reemplaza el analizador disperso del sistema anterior.

Autor: Sistema de Optimización IA
Fecha: 2025-01-09 (Fase 5 - Organización)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

# Local imports
from ..utils.logging_setup import get_logger
from ..utils.import_manager import safe_import, get_visualization_libraries

# Safe imports para visualización
matplotlib = safe_import("matplotlib.pyplot")
seaborn = safe_import("seaborn")
plotly = safe_import("plotly.graph_objects")
plotly_express = safe_import("plotly.express")


@dataclass
class ModelComparison:
    """Comparación entre modelos"""
    model_name: str
    best_score: float
    mean_cv_score: float
    std_cv_score: float
    n_trials: int
    optimization_time: float
    trials_per_second: float
    best_params: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class ExperimentAnalysis:
    """Análisis completo de un experimento"""
    experiment_id: str
    timestamp: datetime
    total_time: float
    best_model: str
    best_score: float
    model_comparisons: List[ModelComparison]
    data_info: Dict[str, Any]
    config_summary: Dict[str, Any]


class ResultsAnalyzer:
    """
    Analizador enterprise-ready de resultados de optimización.
    
    Proporciona análisis profundo, visualizaciones y comparaciones
    de experimentos de optimización de hiperparámetros.
    """
    
    def __init__(self, results_dir: str = "./results"):
        """
        Inicializar analizador de resultados.
        
        Args:
            results_dir: Directorio donde buscar resultados
        """
        self.results_dir = Path(results_dir)
        self.logger = get_logger("results_analyzer")
        self.experiments: Dict[str, ExperimentAnalysis] = {}
        
        # Verificar disponibilidad de librerías de visualización
        self.viz_libs = get_visualization_libraries()
        
        if not self.viz_libs:
            self.logger.warning("⚠️  No hay librerías de visualización disponibles")
    
    def load_experiments(self, pattern: str = "results_*.json") -> int:
        """
        Cargar experimentos desde archivos de resultados.
        
        Args:
            pattern: Patrón de archivos a cargar
            
        Returns:
            Número de experimentos cargados
        """
        if not self.results_dir.exists():
            self.logger.warning(f"Directorio de resultados no existe: {self.results_dir}")
            return 0
        
        result_files = list(self.results_dir.glob(pattern))
        loaded_count = 0
        
        for result_file in result_files:
            try:
                experiment = self._load_experiment_from_file(result_file)
                self.experiments[experiment.experiment_id] = experiment
                loaded_count += 1
                self.logger.info(f"📊 Cargado experimento: {experiment.experiment_id}")
            except Exception as e:
                self.logger.warning(f"❌ Error cargando {result_file}: {e}")
        
        self.logger.info(f"✅ Cargados {loaded_count} experimentos")
        return loaded_count
    
    def _load_experiment_from_file(self, file_path: Path) -> ExperimentAnalysis:
        """Cargar experimento desde archivo JSON"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Crear comparaciones de modelos
        model_comparisons = []
        for model_name, model_data in data.get("model_results", {}).items():
            cv_scores = model_data.get("cv_scores", [])
            comparison = ModelComparison(
                model_name=model_name,
                best_score=model_data.get("best_score", 0.0),
                mean_cv_score=np.mean(cv_scores) if cv_scores else 0.0,
                std_cv_score=np.std(cv_scores) if cv_scores else 0.0,
                n_trials=model_data.get("n_trials", 0),
                optimization_time=model_data.get("optimization_time", 0.0),
                trials_per_second=model_data.get("n_trials", 0) / max(model_data.get("optimization_time", 1), 1),
                best_params=model_data.get("best_params", {}),
                feature_importance=model_data.get("feature_importance")
            )
            model_comparisons.append(comparison)
        
        # Crear análisis del experimento
        return ExperimentAnalysis(
            experiment_id=data.get("experiment_id", "unknown"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            total_time=data.get("total_time", 0.0),
            best_model=data.get("best_model", "unknown"),
            best_score=data.get("best_score", 0.0),
            model_comparisons=model_comparisons,
            data_info=data.get("data_info", {}),
            config_summary=data.get("config_summary", {})
        )
    
    def get_experiment_summary(self, experiment_id: Optional[str] = None) -> pd.DataFrame:
        """
        Obtener resumen de experimentos.
        
        Args:
            experiment_id: ID específico de experimento (None = todos)
            
        Returns:
            DataFrame con resumen de experimentos
        """
        if not self.experiments:
            self.logger.warning("No hay experimentos cargados")
            return pd.DataFrame()
        
        experiments_to_analyze = (
            [self.experiments[experiment_id]] if experiment_id and experiment_id in self.experiments
            else list(self.experiments.values())
        )
        
        summary_data = []
        for exp in experiments_to_analyze:
            summary_data.append({
                "experiment_id": exp.experiment_id,
                "timestamp": exp.timestamp,
                "best_model": exp.best_model,
                "best_score": exp.best_score,
                "total_time": exp.total_time,
                "num_models": len(exp.model_comparisons),
                "data_samples": exp.data_info.get("shape", [0, 0])[0],
                "data_features": exp.data_info.get("shape", [0, 0])[1],
                "memory_usage_mb": exp.data_info.get("memory_usage_mb", 0.0)
            })
        
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.sort_values("timestamp", ascending=False)
        
        return df
    
    def get_model_comparison(self, experiment_id: Optional[str] = None) -> pd.DataFrame:
        """
        Obtener comparación detallada de modelos.
        
        Args:
            experiment_id: ID específico de experimento (None = todos)
            
        Returns:
            DataFrame con comparación de modelos
        """
        if not self.experiments:
            return pd.DataFrame()
        
        experiments_to_analyze = (
            [self.experiments[experiment_id]] if experiment_id and experiment_id in self.experiments
            else list(self.experiments.values())
        )
        
        comparison_data = []
        for exp in experiments_to_analyze:
            for model_comp in exp.model_comparisons:
                comparison_data.append({
                    "experiment_id": exp.experiment_id,
                    "model_name": model_comp.model_name,
                    "best_score": model_comp.best_score,
                    "mean_cv_score": model_comp.mean_cv_score,
                    "std_cv_score": model_comp.std_cv_score,
                    "n_trials": model_comp.n_trials,
                    "optimization_time": model_comp.optimization_time,
                    "trials_per_second": model_comp.trials_per_second,
                    "timestamp": exp.timestamp
                })
        
        df = pd.DataFrame(comparison_data)
        if not df.empty:
            df = df.sort_values(["experiment_id", "best_score"], ascending=[True, False])
        
        return df
    
    def get_best_parameters(self, model_name: str, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtener mejores parámetros para un modelo específico.
        
        Args:
            model_name: Nombre del modelo
            experiment_id: ID específico de experimento (None = mejor de todos)
            
        Returns:
            Diccionario con mejores parámetros
        """
        experiments_to_analyze = (
            [self.experiments[experiment_id]] if experiment_id and experiment_id in self.experiments
            else list(self.experiments.values())
        )
        
        best_params = {}
        best_score = 0.0
        
        for exp in experiments_to_analyze:
            for model_comp in exp.model_comparisons:
                if model_comp.model_name == model_name and model_comp.best_score > best_score:
                    best_score = model_comp.best_score
                    best_params = model_comp.best_params.copy()
                    best_params["_score"] = best_score
                    best_params["_experiment_id"] = exp.experiment_id
        
        return best_params
    
    def analyze_feature_importance(self, experiment_id: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Analizar importancia de features por modelo.
        
        Args:
            experiment_id: ID específico de experimento (None = último)
            
        Returns:
            Diccionario con DataFrames de importancia por modelo
        """
        if not self.experiments:
            return {}
        
        # Usar experimento específico o el más reciente
        if experiment_id and experiment_id in self.experiments:
            exp = self.experiments[experiment_id]
        else:
            exp = max(self.experiments.values(), key=lambda x: x.timestamp)
        
        importance_data = {}
        
        for model_comp in exp.model_comparisons:
            if model_comp.feature_importance:
                df = pd.DataFrame([
                    {"feature": feature, "importance": importance}
                    for feature, importance in model_comp.feature_importance.items()
                ])
                df = df.sort_values("importance", ascending=False)
                importance_data[model_comp.model_name] = df
        
        return importance_data
    
    def plot_model_comparison(self, experiment_id: Optional[str] = None, 
                             save_path: Optional[str] = None) -> Optional[Any]:
        """
        Crear gráfico de comparación de modelos.
        
        Args:
            experiment_id: ID específico de experimento
            save_path: Ruta para guardar gráfico
            
        Returns:
            Figura de matplotlib/plotly o None
        """
        if "matplotlib" not in self.viz_libs:
            self.logger.warning("Matplotlib no disponible para visualización")
            return None
        
        df = self.get_model_comparison(experiment_id)
        if df.empty:
            self.logger.warning("No hay datos para visualizar")
            return None
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Comparación de Modelos - {experiment_id or "Todos los Experimentos"}', fontsize=16)
        
        # Score comparison
        axes[0, 0].bar(df['model_name'], df['best_score'])
        axes[0, 0].set_title('Mejor Score por Modelo')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # CV Score with error bars
        axes[0, 1].errorbar(df['model_name'], df['mean_cv_score'], yerr=df['std_cv_score'], fmt='o')
        axes[0, 1].set_title('CV Score (Media ± Std)')
        axes[0, 1].set_ylabel('CV Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Optimization time
        axes[1, 0].bar(df['model_name'], df['optimization_time'])
        axes[1, 0].set_title('Tiempo de Optimización')
        axes[1, 0].set_ylabel('Tiempo (segundos)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Trials per second
        axes[1, 1].bar(df['model_name'], df['trials_per_second'])
        axes[1, 1].set_title('Eficiencia (Trials/segundo)')
        axes[1, 1].set_ylabel('Trials/segundo')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Gráfico guardado en: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, model_name: str, experiment_id: Optional[str] = None,
                               top_n: int = 20, save_path: Optional[str] = None) -> Optional[Any]:
        """
        Crear gráfico de importancia de features.
        
        Args:
            model_name: Nombre del modelo
            experiment_id: ID específico de experimento
            top_n: Número de features top a mostrar
            save_path: Ruta para guardar gráfico
            
        Returns:
            Figura de matplotlib o None
        """
        if "matplotlib" not in self.viz_libs:
            self.logger.warning("Matplotlib no disponible para visualización")
            return None
        
        importance_data = self.analyze_feature_importance(experiment_id)
        
        if model_name not in importance_data:
            self.logger.warning(f"No hay datos de importancia para {model_name}")
            return None
        
        import matplotlib.pyplot as plt
        
        df = importance_data[model_name].head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(df['feature'], df['importance'])
        ax.set_title(f'Importancia de Features - {model_name}')
        ax.set_xlabel('Importancia')
        
        # Colorear barras por importancia
        colors = plt.cm.viridis(df['importance'] / df['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Gráfico de importancia guardado en: {save_path}")
        
        return fig
    
    def plot_optimization_history(self, experiment_id: str, model_name: str,
                                 save_path: Optional[str] = None) -> Optional[Any]:
        """
        Crear gráfico de historia de optimización (requiere datos de Optuna).
        
        Args:
            experiment_id: ID del experimento
            model_name: Nombre del modelo
            save_path: Ruta para guardar gráfico
            
        Returns:
            Figura de plotly o None
        """
        # Este método requeriría acceso a los estudios de Optuna guardados
        # Por ahora, retornamos None con un mensaje informativo
        self.logger.info("📊 Historia de optimización requiere estudios de Optuna guardados")
        return None
    
    def export_report(self, experiment_id: Optional[str] = None, 
                     output_path: str = "./report.html") -> str:
        """
        Exportar reporte completo en HTML.
        
        Args:
            experiment_id: ID específico de experimento (None = todos)
            output_path: Ruta del archivo de reporte
            
        Returns:
            Ruta del archivo generado
        """
        # Generar datos para el reporte
        summary_df = self.get_experiment_summary(experiment_id)
        comparison_df = self.get_model_comparison(experiment_id)
        
        # Crear HTML básico
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Optimización - {experiment_id or 'Todos los Experimentos'}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; }}
            </style>
        </head>
        <body>
            <h1>🚀 Reporte de Optimización de Hiperparámetros</h1>
            <p><strong>Generado:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>📊 Resumen de Experimentos</h2>
            {summary_df.to_html(classes='summary-table', table_id='summary') if not summary_df.empty else '<p>No hay datos disponibles</p>'}
            
            <h2>🤖 Comparación de Modelos</h2>
            {comparison_df.to_html(classes='comparison-table', table_id='comparison') if not comparison_df.empty else '<p>No hay datos disponibles</p>'}
            
            <h2>🏆 Mejores Parámetros por Modelo</h2>
        """
        
        # Agregar mejores parámetros para cada modelo
        if not comparison_df.empty:
            for model_name in comparison_df['model_name'].unique():
                best_params = self.get_best_parameters(model_name, experiment_id)
                if best_params:
                    html_content += f"<h3>{model_name.upper()}</h3><ul>"
                    for param, value in best_params.items():
                        if not param.startswith('_'):
                            html_content += f"<li><strong>{param}:</strong> {value}</li>"
                    html_content += f"<li class='metric'>Score: {best_params.get('_score', 'N/A')}</li>"
                    html_content += "</ul>"
        
        html_content += """
        </body>
        </html>
        """
        
        # Guardar archivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"📄 Reporte exportado a: {output_path}")
        return output_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas generales de todos los experimentos"""
        if not self.experiments:
            return {}
        
        all_scores = []
        all_times = []
        model_counts = {}
        
        for exp in self.experiments.values():
            all_times.append(exp.total_time)
            for model_comp in exp.model_comparisons:
                all_scores.append(model_comp.best_score)
                model_counts[model_comp.model_name] = model_counts.get(model_comp.model_name, 0) + 1
        
        return {
            "total_experiments": len(self.experiments),
            "total_models_tested": sum(model_counts.values()),
            "unique_models": list(model_counts.keys()),
            "model_frequency": model_counts,
            "score_statistics": {
                "mean": np.mean(all_scores) if all_scores else 0,
                "std": np.std(all_scores) if all_scores else 0,
                "min": np.min(all_scores) if all_scores else 0,
                "max": np.max(all_scores) if all_scores else 0
            },
            "time_statistics": {
                "mean": np.mean(all_times) if all_times else 0,
                "total": np.sum(all_times) if all_times else 0,
                "min": np.min(all_times) if all_times else 0,
                "max": np.max(all_times) if all_times else 0
            }
        }


# ==================== FUNCIONES DE CONVENIENCIA ====================

def analyze_latest_experiment(results_dir: str = "./results") -> Optional[ExperimentAnalysis]:
    """
    Analizar el experimento más reciente.
    
    Args:
        results_dir: Directorio de resultados
        
    Returns:
        Análisis del experimento más reciente o None
    """
    analyzer = ResultsAnalyzer(results_dir)
    count = analyzer.load_experiments()
    
    if count == 0:
        return None
    
    latest_exp = max(analyzer.experiments.values(), key=lambda x: x.timestamp)
    return latest_exp

def compare_all_experiments(results_dir: str = "./results") -> pd.DataFrame:
    """
    Comparar todos los experimentos disponibles.
    
    Args:
        results_dir: Directorio de resultados
        
    Returns:
        DataFrame con comparación de experimentos
    """
    analyzer = ResultsAnalyzer(results_dir)
    analyzer.load_experiments()
    return analyzer.get_experiment_summary()

def get_best_model_params(model_name: str, results_dir: str = "./results") -> Dict[str, Any]:
    """
    Obtener los mejores parámetros históricos para un modelo.
    
    Args:
        model_name: Nombre del modelo
        results_dir: Directorio de resultados
        
    Returns:
        Mejores parámetros encontrados
    """
    analyzer = ResultsAnalyzer(results_dir)
    analyzer.load_experiments()
    return analyzer.get_best_parameters(model_name)


if __name__ == "__main__":
    # Demo del analizador
    print("🚀 Analizador de Resultados Mejorado - Fase 5")
    print("============================================")
    
    # Crear analizador
    analyzer = ResultsAnalyzer("./results")
    
    # Cargar experimentos
    count = analyzer.load_experiments()
    print(f"📊 Experimentos cargados: {count}")
    
    if count > 0:
        # Mostrar resumen
        summary = analyzer.get_experiment_summary()
        print("\n📋 Resumen de Experimentos:")
        print(summary.to_string(index=False))
        
        # Mostrar estadísticas
        stats = analyzer.get_statistics()
        print(f"\n📈 Estadísticas Generales:")
        print(f"   - Total experimentos: {stats['total_experiments']}")
        print(f"   - Modelos únicos: {len(stats['unique_models'])}")
        print(f"   - Score promedio: {stats['score_statistics']['mean']:.4f}")
        print(f"   - Tiempo total: {stats['time_statistics']['total']:.2f}s")
        
        # Generar reporte
        report_path = analyzer.export_report()
        print(f"\n📄 Reporte generado: {report_path}")
    else:
        print("⚠️  No se encontraron experimentos para analizar")
        print("   Ejecutar optimizaciones primero con quick_optimization.py")
