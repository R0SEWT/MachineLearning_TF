"""
Módulo de visualizaciones para el análisis EDA de criptomonedas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

def plot_narrative_distribution(df: pd.DataFrame, colors: Dict[str, str], 
                               figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Crea gráficos de distribución por narrativa
    
    Args:
        df: DataFrame con los datos
        colors: Diccionario de colores por narrativa
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Distribución por narrativa
    narrative_dist = df['narrative'].value_counts()
    plot_colors = [colors.get(n, f'C{i}') for i, n in enumerate(narrative_dist.index)]
    
    # Gráfico de pastel
    wedges, texts, autotexts = ax1.pie(narrative_dist.values, labels=narrative_dist.index,
                                       autopct='%1.1f%%', colors=plot_colors, startangle=90)
    ax1.set_title('Distribución por Narrativa', fontweight='bold')
    
    # Gráfico de barras
    bars = ax2.bar(narrative_dist.index, narrative_dist.values, color=plot_colors)
    ax2.set_title('Número de Observaciones por Narrativa', fontweight='bold')
    ax2.set_ylabel('Número de Observaciones')
    ax2.tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for bar, value in zip(bars, narrative_dist.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(narrative_dist.values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_market_cap_analysis(df: pd.DataFrame, colors: Dict[str, str], 
                            figsize: Tuple[int, int] = (15, 6)) -> plt.Figure:
    """
    Crea análisis visual del market cap por narrativa
    
    Args:
        df: DataFrame con los datos
        colors: Diccionario de colores por narrativa
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    if 'market_cap' not in df.columns:
        fig.suptitle('Market Cap no disponible en el dataset', fontsize=16)
        return fig
    
    # Market cap total por narrativa
    market_cap_by_narrative = df.groupby('narrative')['market_cap'].sum().sort_values(ascending=False)
    total_market_cap = market_cap_by_narrative.sum()
    
    # Gráfico de barras
    plot_colors = [colors.get(n, f'C{i}') for i, n in enumerate(market_cap_by_narrative.index)]
    bars = ax1.bar(market_cap_by_narrative.index, market_cap_by_narrative.values/1e9, color=plot_colors)
    ax1.set_title('Market Cap Total por Narrativa', fontweight='bold')
    ax1.set_ylabel('Market Cap (Billones USD)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}B'))
    
    # Gráfico de pastel para dominancia
    wedges, texts, autotexts = ax2.pie(market_cap_by_narrative.values, 
                                       labels=market_cap_by_narrative.index,
                                       autopct='%1.1f%%', colors=plot_colors, startangle=90)
    ax2.set_title('Dominancia de Market Cap', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_temporal_analysis(df: pd.DataFrame, colors: Dict[str, str], 
                          figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Crea análisis temporal avanzado
    
    Args:
        df: DataFrame con los datos
        colors: Diccionario de colores por narrativa
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Análisis de Patrones Temporales', fontsize=16, fontweight='bold')
    
    if 'date' not in df.columns:
        fig.text(0.5, 0.5, 'Datos temporales no disponibles', ha='center', va='center', fontsize=20)
        return fig
    
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['date'])
    
    # 1. Evolución del market cap por narrativa
    ax1 = axes[0, 0]
    if 'market_cap' in df_temp.columns:
        temporal_data = df_temp.groupby(['date', 'narrative'])['market_cap'].sum().reset_index()
        pivot_temporal = temporal_data.pivot(index='date', columns='narrative', values='market_cap').fillna(0)
        
        for narrative in pivot_temporal.columns:
            ax1.plot(pivot_temporal.index, pivot_temporal[narrative]/1e9, 
                    label=narrative, linewidth=2, 
                    color=colors.get(narrative, 'gray'))
        
        ax1.set_title('Evolución Market Cap por Narrativa', fontweight='bold')
        ax1.set_ylabel('Market Cap (Billones USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Tokens activos por día
    ax2 = axes[0, 1]
    active_tokens = df_temp.groupby(['date', 'narrative'])['id'].nunique().reset_index()
    pivot_tokens = active_tokens.pivot(index='date', columns='narrative', values='id').fillna(0)
    
    for narrative in pivot_tokens.columns:
        ax2.plot(pivot_tokens.index, pivot_tokens[narrative], 
                label=narrative, linewidth=2,
                color=colors.get(narrative, 'gray'))
    
    ax2.set_title('Tokens Activos por Día', fontweight='bold')
    ax2.set_ylabel('Número de Tokens')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Actividad por día de la semana
    ax3 = axes[1, 0]
    df_temp['day_of_week'] = df_temp['date'].dt.day_name()
    activity_heatmap = df_temp.groupby(['day_of_week', 'narrative']).size().unstack(fill_value=0)
    
    # Reordenar días de la semana
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    activity_heatmap = activity_heatmap.reindex([d for d in day_order if d in activity_heatmap.index])
    
    sns.heatmap(activity_heatmap, annot=True, fmt='d', cmap='YlOrRd', ax=ax3)
    ax3.set_title('Actividad por Día de la Semana', fontweight='bold')
    ax3.set_ylabel('Día de la Semana')
    
    # 4. Distribución mensual
    ax4 = axes[1, 1]
    df_temp['year_month'] = df_temp['date'].dt.to_period('M')
    monthly_counts = df_temp['year_month'].value_counts().sort_index()
    
    ax4.plot(range(len(monthly_counts)), monthly_counts.values, marker='o', linewidth=2)
    ax4.set_title('Registros por Mes', fontweight='bold')
    ax4.set_ylabel('Número de Registros')
    ax4.set_xlabel('Período')
    ax4.grid(True, alpha=0.3)
    
    # Etiquetas del eje x (mostrar algunos meses)
    if len(monthly_counts) > 0:
        n_labels = min(6, len(monthly_counts))
        indices = np.linspace(0, len(monthly_counts)-1, n_labels, dtype=int)
        ax4.set_xticks(indices)
        ax4.set_xticklabels([str(monthly_counts.index[i]) for i in indices], rotation=45)
    
    plt.tight_layout()
    return fig

def plot_returns_analysis(df: pd.DataFrame, colors: Dict[str, str], 
                         figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Crea análisis de distribuciones de retornos
    
    Args:
        df: DataFrame con los datos
        colors: Diccionario de colores por narrativa
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Análisis de Retornos por Narrativa', fontsize=16, fontweight='bold')
    
    # Verificar columnas de retornos
    return_cols = [col for col in ['ret_1d', 'ret_7d', 'ret_30d'] if col in df.columns]
    
    if not return_cols:
        fig.text(0.5, 0.5, 'Datos de retornos no disponibles', ha='center', va='center', fontsize=20)
        return fig
    
    # 1. Histograma de retornos por narrativa
    ax1 = axes[0, 0]
    if 'ret_1d' in df.columns:
        for i, narrative in enumerate(df['narrative'].unique()):
            data = df[df['narrative'] == narrative]['ret_1d'].dropna()
            if len(data) > 0:
                # Filtrar outliers extremos
                data_filtered = data[(data > data.quantile(0.01)) & (data < data.quantile(0.99))]
                ax1.hist(data_filtered, bins=30, alpha=0.6, 
                        label=f'{narrative} (n={len(data)})',
                        color=colors.get(narrative, f'C{i}'))
        
        ax1.set_title('Distribución Retornos 1 Día', fontweight='bold')
        ax1.set_xlabel('Retorno')
        ax1.set_ylabel('Frecuencia')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Box plot de retornos por narrativa
    ax2 = axes[0, 1]
    if 'ret_7d' in df.columns:
        boxplot_data = []
        labels = []
        
        for narrative in df['narrative'].unique():
            data = df[df['narrative'] == narrative]['ret_7d'].dropna()
            if len(data) > 10:
                # Filtrar outliers extremos
                q1, q99 = data.quantile([0.05, 0.95])
                filtered_data = data[(data >= q1) & (data <= q99)]
                boxplot_data.append(filtered_data)
                labels.append(narrative)
        
        if boxplot_data:
            bp = ax2.boxplot(boxplot_data, labels=labels, patch_artist=True)
            for patch, narrative in zip(bp['boxes'], labels):
                patch.set_facecolor(colors.get(narrative, 'lightblue'))
        
        ax2.set_title('Retornos 7 Días por Narrativa', fontweight='bold')
        ax2.set_ylabel('Retorno')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # 3. Volatilidad por narrativa
    ax3 = axes[0, 2]
    if 'ret_1d' in df.columns:
        volatility_by_narrative = df.groupby('narrative')['ret_1d'].std().sort_values(ascending=True)
        bars = ax3.barh(range(len(volatility_by_narrative)), volatility_by_narrative.values)
        ax3.set_yticks(range(len(volatility_by_narrative)))
        ax3.set_yticklabels(volatility_by_narrative.index)
        ax3.set_title('Volatilidad por Narrativa', fontweight='bold')
        ax3.set_xlabel('Volatilidad (Desv. Std.)')
        
        # Colorear barras
        for i, (bar, narrative) in enumerate(zip(bars, volatility_by_narrative.index)):
            bar.set_color(colors.get(narrative, f'C{i}'))
        
        ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Correlación entre horizontes de retornos
    ax4 = axes[1, 0]
    available_returns = [col for col in return_cols if col in df.columns]
    if len(available_returns) > 1:
        returns_corr = df[available_returns].corr()
        im = ax4.imshow(returns_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(returns_corr.columns)))
        ax4.set_yticks(range(len(returns_corr.columns)))
        ax4.set_xticklabels([col.replace('ret_', '') + 'D' for col in returns_corr.columns])
        ax4.set_yticklabels([col.replace('ret_', '') + 'D' for col in returns_corr.columns])
        ax4.set_title('Correlación entre Horizontes', fontweight='bold')
        
        # Añadir valores de correlación
        for i in range(len(returns_corr)):
            for j in range(len(returns_corr.columns)):
                ax4.text(j, i, f'{returns_corr.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontweight='bold')
    
    # 5. Análisis de asimetría por narrativa
    ax5 = axes[1, 1]
    if 'ret_1d' in df.columns:
        skew_data = []
        kurt_data = []
        narrative_names = []
        
        for narrative in df['narrative'].unique():
            data = df[df['narrative'] == narrative]['ret_1d'].dropna()
            if len(data) > 30:
                skew_data.append(data.skew())
                kurt_data.append(data.kurtosis())
                narrative_names.append(narrative)
        
        if narrative_names:
            scatter_colors = [colors.get(n, 'gray') for n in narrative_names]
            ax5.scatter(skew_data, kurt_data, c=scatter_colors, s=100, alpha=0.7)
            
            for i, narrative in enumerate(narrative_names):
                ax5.annotate(narrative, (skew_data[i], kurt_data[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax5.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax5.set_xlabel('Asimetría (Skewness)')
            ax5.set_ylabel('Curtosis')
            ax5.set_title('Asimetría vs Curtosis', fontweight='bold')
            ax5.grid(True, alpha=0.3)
    
    # 6. Estadísticas resumen
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Crear tabla resumen
    if 'ret_1d' in df.columns:
        summary_stats = []
        for narrative in df['narrative'].unique():
            data = df[df['narrative'] == narrative]['ret_1d'].dropna()
            if len(data) > 10:
                stats = {
                    'Narrativa': narrative,
                    'Media': f"{data.mean():.4f}",
                    'Std': f"{data.std():.4f}",
                    'Skew': f"{data.skew():.2f}",
                    'N': len(data)
                }
                summary_stats.append(stats)
        
        if summary_stats:
            summary_text = "📊 ESTADÍSTICAS RESUMEN\n\n"
            for stat in summary_stats:
                summary_text += f"{stat['Narrativa']:8}: μ={stat['Media']}, σ={stat['Std']}\n"
                summary_text += f"{'':10} Skew={stat['Skew']}, N={stat['N']}\n\n"
            
            ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_quality_dashboard(metrics: Dict[str, Any], quality_eval: Dict[str, Any], 
                          df: pd.DataFrame, colors: Dict[str, str],
                          figsize: Tuple[int, int] = (18, 10)) -> plt.Figure:
    """
    Crea dashboard de calidad consolidado
    
    Args:
        metrics: Métricas del dataset
        quality_eval: Evaluación de calidad
        df: DataFrame con los datos
        colors: Diccionario de colores por narrativa
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('📊 Dashboard de Calidad del Dataset', fontsize=16, fontweight='bold')
    
    # 1. Distribución por narrativa
    ax1 = axes[0, 0]
    narrative_dist = df['narrative'].value_counts()
    plot_colors = [colors.get(n, f'C{i}') for i, n in enumerate(narrative_dist.index)]
    wedges, texts, autotexts = ax1.pie(narrative_dist.values, labels=narrative_dist.index,
                                       autopct='%1.1f%%', colors=plot_colors, startangle=90)
    ax1.set_title('Distribución por Narrativa', fontweight='bold')
    
    # 2. Market Cap por narrativa
    ax2 = axes[0, 1]
    if 'market_cap' in df.columns:
        market_cap_by_narrative = df.groupby('narrative')['market_cap'].sum().sort_values(ascending=True)
        bars = ax2.barh(range(len(market_cap_by_narrative)), market_cap_by_narrative.values/1e9)
        ax2.set_yticks(range(len(market_cap_by_narrative)))
        ax2.set_yticklabels(market_cap_by_narrative.index)
        ax2.set_xlabel('Market Cap (Billones USD)')
        ax2.set_title('Market Cap por Narrativa', fontweight='bold')
        
        # Colorear barras
        for i, (bar, narrative) in enumerate(zip(bars, market_cap_by_narrative.index)):
            bar.set_color(colors.get(narrative, f'C{i}'))
    
    # 3. Completitud de datos
    ax3 = axes[0, 2]
    missing_by_var = df.isnull().sum().sort_values(ascending=False)
    completeness_by_var = (1 - missing_by_var / len(df)) * 100
    
    missing_vars = completeness_by_var[completeness_by_var < 100]
    if len(missing_vars) > 0:
        ax3.barh(range(len(missing_vars)), missing_vars.values, color='orange', alpha=0.7)
        ax3.set_yticks(range(len(missing_vars)))
        ax3.set_yticklabels(missing_vars.index)
        ax3.set_xlabel('% Completitud')
        ax3.set_title('Completitud por Variable', fontweight='bold')
        ax3.axvline(x=95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, '✅ Todas las variables\ntienen 100% completitud',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12, fontweight='bold')
        ax3.set_title('Completitud por Variable', fontweight='bold')
    
    # 4. Evolución temporal
    ax4 = axes[1, 0]
    if 'date' in df.columns:
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        daily_tokens = df_temp.groupby('date')['id'].nunique()
        ax4.plot(daily_tokens.index, daily_tokens.values, linewidth=2, color='blue')
        ax4.set_ylabel('Tokens únicos por día')
        ax4.set_xlabel('Fecha')
        ax4.set_title('Cobertura Temporal', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Sin información\ntemporal disponible',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Cobertura Temporal', fontweight='bold')
    
    # 5. Distribución de retornos
    ax5 = axes[1, 1]
    if 'ret_1d' in df.columns:
        returns_clean = df['ret_1d'].dropna()
        returns_filtered = returns_clean[(returns_clean > returns_clean.quantile(0.01)) &
                                       (returns_clean < returns_clean.quantile(0.99))]
        ax5.hist(returns_filtered, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax5.set_xlabel('Retornos 1 día')
        ax5.set_ylabel('Frecuencia')
        ax5.set_title('Distribución de Retornos', fontweight='bold')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Retornos no\ncalculados aún',
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Distribución de Retornos', fontweight='bold')
    
    # 6. Semáforo de calidad
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    semaforo_text = f"""🎯 EVALUACIÓN DE CALIDAD

{quality_eval['vol_score'][0]} Volumen: {metrics['total_observations']:,} obs
{quality_eval['comp_score'][0]} Completitud: {metrics['completeness']:.1f}%
{quality_eval['div_score'][0]} Narrativas: {metrics['total_narratives']}
{quality_eval['temp_score'][0]} Temporal: {metrics['date_range']} días

📊 SCORE GENERAL: {quality_eval['readiness_percentage']:.0f}%
{quality_eval['overall_status']}

{'✅ LISTO PARA ML' if quality_eval['readiness_percentage'] >= 75 else '⚠️ MEJORAS NECESARIAS' if quality_eval['readiness_percentage'] >= 50 else '❌ TRABAJO REQUERIDO'}
"""
    
    ax6.text(0.05, 0.95, semaforo_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    return fig
