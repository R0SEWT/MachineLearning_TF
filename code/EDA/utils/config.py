"""
Módulo de configuración para análisis EDA de criptomonedas
Contiene configuraciones, colores y constantes utilizadas en el análisis
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de estilo visual
def setup_plotting_style():
    """Configura el estilo visual para los gráficos"""
    plt.style.use("classic")
    sns.set_context("talk")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

# Colores para narrativas
NARRATIVE_COLORS = {
    'meme': '#ff6b6b',
    'rwa': '#4ecdc4', 
    'gaming': '#45b7d1',
    'ai': '#96ceb4',
    'DeFi': '#ffa726',
    'Infrastructure': '#ab47bc'
}

# Configuración de rutas
def get_project_paths():
    """Obtiene las rutas principales del proyecto"""
    ROOT = Path.cwd().parents[1] if 'code' in str(Path.cwd()) else Path.cwd()
    DATA_PATH = ROOT / "data" / "crypto_ohlc_join.csv"
    OUT_PATH = ROOT / "data" / "ml_dataset.csv"
    REPORTS_PATH = ROOT / "reports"
    
    return {
        'root': ROOT,
        'data': DATA_PATH,
        'output': OUT_PATH,
        'reports': REPORTS_PATH
    }

# Configuración de análisis
ANALYSIS_CONFIG = {
    'outlier_contamination': 0.05,
    'min_history_days': 60,
    'volatility_window': 30,
    'correlation_window': 90,
    'quantile_filter': (0.01, 0.99),
    'n_clusters': 4,
    'test_size': 0.2,
    'random_state': 42
}

# Umbrales para evaluación de calidad
QUALITY_THRESHOLDS = {
    'excellent': {
        'observations': 10000,
        'completeness': 95,
        'narratives': 3,
        'temporal_days': 180,
        'outlier_rate': 10
    },
    'good': {
        'observations': 1000,
        'completeness': 85,
        'narratives': 2,
        'temporal_days': 90,
        'outlier_rate': 15
    }
}

# Columnas esperadas en el dataset
EXPECTED_COLUMNS = {
    'required': ['close', 'date', 'id', 'narrative'],
    'optional': ['market_cap', 'volume', 'price', 'symbol', 'name'],
    'calculated': ['ret_1d', 'ret_7d', 'ret_30d', 'vol_30d', 'future_ret_30d']
}

# Configuración de features técnicos
TECHNICAL_FEATURES = {
    'returns': [1, 7, 30],  # días para calcular retornos
    'moving_averages': [7, 30],  # ventanas para promedios móviles
    'volatility_window': 30,
    'bollinger_window': 20
}
