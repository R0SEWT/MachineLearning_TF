#!/usr/bin/env python3
"""
Analizador y visualizador de resultados de optimización Optuna
Genera reportes detallados y gráficos interactivos
"""

import sys
import os
import pandas as pd
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Agregar paths necesarios
sys.path.append('/home/exodia/Documentos/MachineLearning_TF/code/EDA/utils')

try:
    import optuna
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTTING_AVAILABLE = True
except ImportError:
    print("⚠️  Librerías de visualización no disponibles")
    PLOTTING_AVAILABLE = False

class OptunaResultsAnalyzer:
    """
    Analizador completo de resultados de optimización Optuna
    """
    
    def __init__(self, results_path: str = "../../optimization_results"):
        """
        Inicializar analizador
        
        Args:
            results_path: Ruta donde están los resultados
        """
        self.results_path = Path(results_path)
        self.studies = {}
        self.summaries = []
        self.evaluations = []
        
        print(f"🔍 OptunaResultsAnalyzer inicializado")
        print(f"   📁 Buscando resultados en: {self.results_path}")
        
        self.load_all_results()
    
    def load_all_results(self):
        """
        Cargar todos los resultados disponibles
        """
        print("\n📂 Cargando resultados...")
        
        # Cargar estudios de Optuna
        study_files = list(self.results_path.glob("optuna_studies_*.pkl"))
        for study_file in study_files:
            try:
                with open(study_file, 'rb') as f:
                    studies = pickle.load(f)
                    timestamp = study_file.stem.split('_')[-2:]  # Extraer timestamp
                    self.studies[f"{'_'.join(timestamp)}"] = studies
                print(f"   ✅ Estudios cargados: {study_file.name}")
            except Exception as e:
                print(f"   ❌ Error cargando {study_file.name}: {e}")
        
        # Cargar resúmenes
        summary_files = list(self.results_path.glob("optimization_summary_*.json"))
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    self.summaries.append(summary)
                print(f"   ✅ Resumen cargado: {summary_file.name}")
            except Exception as e:
                print(f"   ❌ Error cargando {summary_file.name}: {e}")
        
        # Cargar evaluaciones
        eval_files = list(self.results_path.glob("evaluation_results_*.json"))
        for eval_file in eval_files:
            try:
                with open(eval_file, 'r') as f:
                    evaluation = json.load(f)
                    self.evaluations.append(evaluation)
                print(f"   ✅ Evaluación cargada: {eval_file.name}")
            except Exception as e:
                print(f"   ❌ Error cargando {eval_file.name}: {e}")
        
        print(f"\n📊 Resultados cargados:")
        print(f"   📈 Estudios: {len(self.studies)}")
        print(f"   📋 Resúmenes: {len(self.summaries)}")
        print(f"   🧪 Evaluaciones: {len(self.evaluations)}")
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """
        Crear DataFrame con comparación de todos los resultados
        """
        comparison_data = []
        
        for summary in self.summaries:
            timestamp = summary['timestamp']
            
            for model_name, score in summary['best_scores'].items():
                params = summary['best_params'][model_name]
                
                # Encontrar evaluación correspondiente
                eval_data = None
                for evaluation in self.evaluations:
                    if model_name in evaluation:
                        eval_data = evaluation[model_name]
                        break
                
                row = {
                    'timestamp': timestamp,
                    'model': model_name,
                    'cv_score': score,
                    'val_auc': eval_data.get('val_auc') if eval_data else None,
                    'test_auc': eval_data.get('test_auc') if eval_data else None,
                    'n_estimators': params.get('n_estimators'),
                    'max_depth': params.get('max_depth', params.get('depth')),
                    'learning_rate': params.get('learning_rate'),
                }
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_performance_report(self):
        """
        Generar reporte detallado de performance
        """
        print("\n📊======================================================================")
        print("📊 REPORTE DE PERFORMANCE DETALLADO")
        print("📊======================================================================")
        
        if not self.summaries:
            print("❌ No hay datos para generar reporte")
            return
        
        df = self.create_comparison_dataframe()
        
        if df.empty:
            print("❌ No se pudo crear DataFrame de comparación")
            return
        
        # Estadísticas generales
        print("\n📈 ESTADÍSTICAS GENERALES:")
        print(f"   🔢 Total de experimentos: {len(df)}")
        print(f"   🤖 Modelos únicos: {df['model'].nunique()}")
        print(f"   📅 Período de experimentos: {df['timestamp'].min()} - {df['timestamp'].max()}")
        
        # Mejores resultados por modelo
        print("\n🏆 MEJORES RESULTADOS POR MODELO:")
        best_by_model = df.loc[df.groupby('model')['cv_score'].idxmax()]
        
        for _, row in best_by_model.iterrows():
            print(f"\n   🔹 {row['model'].upper()}:")
            print(f"      📊 CV Score: {row['cv_score']:.4f}")
            if pd.notna(row['val_auc']):
                print(f"      📊 Validation AUC: {row['val_auc']:.4f}")
            if pd.notna(row['test_auc']):
                print(f"      📊 Test AUC: {row['test_auc']:.4f}")
            print(f"      📅 Fecha: {row['timestamp']}")
            
            # Parámetros principales
            if pd.notna(row['n_estimators']):
                print(f"      🔧 n_estimators: {row['n_estimators']}")
            if pd.notna(row['max_depth']):
                print(f"      🔧 max_depth: {row['max_depth']}")
            if pd.notna(row['learning_rate']):
                print(f"      🔧 learning_rate: {row['learning_rate']:.4f}")
        
        # Modelo global mejor
        best_overall = df.loc[df['cv_score'].idxmax()]
        print(f"\n🥇 MEJOR MODELO GLOBAL:")
        print(f"   🤖 Modelo: {best_overall['model'].upper()}")
        print(f"   📊 CV Score: {best_overall['cv_score']:.4f}")
        print(f"   📅 Fecha: {best_overall['timestamp']}")
        
        # Evolución temporal
        print(f"\n📈 EVOLUCIÓN TEMPORAL:")
        df_sorted = df.sort_values('timestamp')
        for model in df['model'].unique():
            model_data = df_sorted[df_sorted['model'] == model]
            if len(model_data) > 1:
                improvement = model_data['cv_score'].iloc[-1] - model_data['cv_score'].iloc[0]
                print(f"   📊 {model}: {improvement:+.4f} (primer vs último experimento)")
        
        return df
    
    def analyze_hyperparameter_importance(self):
        """
        Analizar importancia de hiperparámetros
        """
        print("\n🔧======================================================================")
        print("🔧 ANÁLISIS DE IMPORTANCIA DE HIPERPARÁMETROS")
        print("🔧======================================================================")
        
        for timestamp, studies in self.studies.items():
            print(f"\n📅 Estudios del {timestamp}:")
            
            for model_name, study in studies.items():
                if hasattr(study, 'trials') and len(study.trials) > 0:
                    print(f"\n   🤖 {model_name.upper()}:")
                    print(f"      🔢 Total trials: {len(study.trials)}")
                    print(f"      🏆 Mejor valor: {study.best_value:.4f}")
                    
                    # Analizar correlaciones de parámetros con performance
                    trial_data = []
                    for trial in study.trials:
                        if trial.state == optuna.trial.TrialState.COMPLETE:
                            row = {'value': trial.value}
                            row.update(trial.params)
                            trial_data.append(row)
                    
                    if trial_data:
                        trial_df = pd.DataFrame(trial_data)
                        
                        # Calcular correlaciones
                        numeric_cols = trial_df.select_dtypes(include=[float, int]).columns
                        if 'value' in numeric_cols and len(numeric_cols) > 1:
                            correlations = trial_df[numeric_cols].corr()['value'].abs().sort_values(ascending=False)
                            
                            print(f"      📈 Parámetros más influyentes:")
                            for param, corr in correlations.items():
                                if param != 'value' and corr > 0.1:
                                    print(f"         {param}: {corr:.3f}")
    
    def generate_interactive_visualizations(self):
        """
        Generar visualizaciones interactivas
        """
        if not PLOTTING_AVAILABLE:
            print("⚠️  Visualizaciones no disponibles (falta plotly)")
            return
        
        print("\n📈======================================================================")
        print("📈 GENERANDO VISUALIZACIONES INTERACTIVAS")
        print("📈======================================================================")
        
        viz_dir = self.results_path / "analysis_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Comparación de modelos
        df = self.create_comparison_dataframe()
        if not df.empty:
            self._create_model_comparison_plot(df, viz_dir)
            self._create_parameter_analysis_plots(df, viz_dir)
            self._create_evolution_plot(df, viz_dir)
        
        # 2. Análisis de estudios individuales
        for timestamp, studies in self.studies.items():
            study_dir = viz_dir / f"studies_{timestamp}"
            study_dir.mkdir(exist_ok=True)
            
            for model_name, study in studies.items():
                self._create_study_visualizations(study, model_name, study_dir)
        
        print(f"   ✅ Visualizaciones guardadas en: {viz_dir}")
    
    def _create_model_comparison_plot(self, df: pd.DataFrame, output_dir: Path):
        """Crear gráfico de comparación de modelos"""
        try:
            fig = go.Figure()
            
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                
                fig.add_trace(go.Scatter(
                    x=model_data['timestamp'],
                    y=model_data['cv_score'],
                    mode='markers+lines',
                    name=model.upper(),
                    marker=dict(size=10),
                    hovertemplate="<b>%{fullData.name}</b><br>" +
                                "Fecha: %{x}<br>" +
                                "CV Score: %{y:.4f}<extra></extra>"
                ))
            
            fig.update_layout(
                title="Evolución de Performance por Modelo",
                xaxis_title="Timestamp",
                yaxis_title="CV Score (AUC)",
                hovermode='closest',
                height=500
            )
            
            fig.write_html(output_dir / "model_comparison.html")
            print(f"   ✅ Comparación de modelos: model_comparison.html")
            
        except Exception as e:
            print(f"   ❌ Error creando comparación de modelos: {e}")
    
    def _create_parameter_analysis_plots(self, df: pd.DataFrame, output_dir: Path):
        """Crear gráficos de análisis de parámetros"""
        try:
            # Gráfico de learning rate vs performance
            fig_lr = px.scatter(
                df, 
                x='learning_rate', 
                y='cv_score',
                color='model',
                title="Learning Rate vs Performance",
                hover_data=['max_depth', 'n_estimators']
            )
            fig_lr.write_html(output_dir / "learning_rate_analysis.html")
            
            # Gráfico de max_depth vs performance
            fig_depth = px.scatter(
                df,
                x='max_depth',
                y='cv_score', 
                color='model',
                title="Max Depth vs Performance",
                hover_data=['learning_rate', 'n_estimators']
            )
            fig_depth.write_html(output_dir / "max_depth_analysis.html")
            
            print(f"   ✅ Análisis de parámetros: learning_rate_analysis.html, max_depth_analysis.html")
            
        except Exception as e:
            print(f"   ❌ Error creando análisis de parámetros: {e}")
    
    def _create_evolution_plot(self, df: pd.DataFrame, output_dir: Path):
        """Crear gráfico de evolución temporal"""
        try:
            fig = px.line(
                df,
                x='timestamp',
                y='cv_score',
                color='model',
                title="Evolución Temporal de Performance",
                markers=True
            )
            
            fig.update_layout(
                height=500,
                xaxis_title="Timestamp",
                yaxis_title="CV Score (AUC)"
            )
            
            fig.write_html(output_dir / "temporal_evolution.html")
            print(f"   ✅ Evolución temporal: temporal_evolution.html")
            
        except Exception as e:
            print(f"   ❌ Error creando evolución temporal: {e}")
    
    def _create_study_visualizations(self, study, model_name: str, output_dir: Path):
        """Crear visualizaciones para un estudio específico"""
        try:
            if not hasattr(study, 'trials') or len(study.trials) == 0:
                return
            
            # Historia de optimización
            from optuna.visualization import plot_optimization_history
            fig_history = plot_optimization_history(study)
            fig_history.write_html(output_dir / f"{model_name}_optimization_history.html")
            
            # Importancia de parámetros
            from optuna.visualization import plot_param_importances
            fig_importance = plot_param_importances(study)
            fig_importance.write_html(output_dir / f"{model_name}_param_importances.html")
            
            print(f"   ✅ Visualizaciones de {model_name}: {output_dir.name}")
            
        except Exception as e:
            print(f"   ❌ Error creando visualizaciones de {model_name}: {e}")
    
    def export_best_configs(self, output_file: Optional[str] = None):
        """
        Exportar las mejores configuraciones encontradas
        """
        print("\n💾======================================================================")
        print("💾 EXPORTANDO MEJORES CONFIGURACIONES")
        print("💾======================================================================")
        
        if output_file is None:
            output_file = self.results_path / f"best_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        best_configs = {}
        
        # Encontrar la mejor configuración para cada modelo
        df = self.create_comparison_dataframe()
        if not df.empty:
            best_by_model = df.loc[df.groupby('model')['cv_score'].idxmax()]
            
            for _, row in best_by_model.iterrows():
                model = row['model']
                timestamp = row['timestamp']
                
                # Buscar los parámetros completos
                for summary in self.summaries:
                    if summary['timestamp'] == timestamp and model in summary['best_params']:
                        best_configs[model] = {
                            'cv_score': row['cv_score'],
                            'val_auc': row['val_auc'] if pd.notna(row['val_auc']) else None,
                            'test_auc': row['test_auc'] if pd.notna(row['test_auc']) else None,
                            'timestamp': timestamp,
                            'parameters': summary['best_params'][model]
                        }
                        break
        
        # Guardar configuraciones
        with open(output_file, 'w') as f:
            json.dump(best_configs, f, indent=2, default=str)
        
        print(f"✅ Mejores configuraciones exportadas a: {output_file}")
        
        # Mostrar resumen
        for model, config in best_configs.items():
            print(f"\n🔹 {model.upper()}:")
            print(f"   📊 CV Score: {config['cv_score']:.4f}")
            if config['val_auc']:
                print(f"   📊 Validation AUC: {config['val_auc']:.4f}")
            if config['test_auc']:
                print(f"   📊 Test AUC: {config['test_auc']:.4f}")
            print(f"   📅 Fecha: {config['timestamp']}")
        
        return best_configs
    
    def generate_full_report(self):
        """
        Generar reporte completo de análisis
        """
        print("🚀======================================================================")
        print("🚀 GENERANDO REPORTE COMPLETO DE ANÁLISIS")
        print("🚀======================================================================")
        
        # 1. Reporte de performance
        df = self.generate_performance_report()
        
        # 2. Análisis de hiperparámetros
        self.analyze_hyperparameter_importance()
        
        # 3. Visualizaciones
        self.generate_interactive_visualizations()
        
        # 4. Exportar mejores configs
        best_configs = self.export_best_configs()
        
        # 5. Resumen final
        print("\n✅======================================================================")
        print("✅ ANÁLISIS COMPLETADO")
        print("✅======================================================================")
        print(f"📁 Todos los resultados en: {self.results_path}")
        print(f"📈 Visualizaciones en: {self.results_path}/analysis_visualizations")
        
        return {
            'performance_df': df,
            'best_configs': best_configs,
            'studies_count': len(self.studies),
            'summaries_count': len(self.summaries)
        }

def main():
    """
    Función principal para análisis completo
    """
    print("🔍 ANALIZADOR DE RESULTADOS OPTUNA")
    print("🔍 OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("🔍======================================================================")
    
    analyzer = OptunaResultsAnalyzer()
    results = analyzer.generate_full_report()
    
    print(f"\n🎯 Análisis completado exitosamente!")
    return results

if __name__ == "__main__":
    main()
