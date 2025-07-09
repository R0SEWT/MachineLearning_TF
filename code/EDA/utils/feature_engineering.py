"""
Módulo de feature engineering para análisis de criptomonedas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

def calculate_returns(df: pd.DataFrame, periods: List[int] = [1, 7, 30], 
                     price_col: str = 'close', id_col: str = 'id') -> pd.DataFrame:
    """
    Calcula retornos para diferentes períodos
    
    Args:
        df: DataFrame con los datos
        periods: Lista de períodos para calcular retornos
        price_col: Columna de precios
        id_col: Columna de identificador
        
    Returns:
        DataFrame con columnas de retornos añadidas
    """
    df_result = df.copy()
    
    for period in periods:
        col_name = f'ret_{period}d'
        df_result[col_name] = df_result.groupby(id_col)[price_col].pct_change(period)
    
    return df_result

def calculate_moving_averages(df: pd.DataFrame, windows: List[int] = [7, 30],
                             price_col: str = 'close', id_col: str = 'id') -> pd.DataFrame:
    """
    Calcula promedios móviles para diferentes ventanas
    
    Args:
        df: DataFrame con los datos
        windows: Lista de ventanas para promedios móviles
        price_col: Columna de precios
        id_col: Columna de identificador
        
    Returns:
        DataFrame con columnas de SMA añadidas
    """
    df_result = df.copy()
    
    for window in windows:
        col_name = f'sma_{window}'
        df_result[col_name] = df_result.groupby(id_col)[price_col].transform(
            lambda x: x.rolling(window).mean()
        )
    
    return df_result

def calculate_volatility(df: pd.DataFrame, window: int = 30, 
                        return_col: str = 'ret_1d', id_col: str = 'id') -> pd.DataFrame:
    """
    Calcula volatilidad móvil
    
    Args:
        df: DataFrame con los datos
        window: Ventana para cálculo de volatilidad
        return_col: Columna de retornos para calcular volatilidad
        id_col: Columna de identificador
        
    Returns:
        DataFrame con columna de volatilidad añadida
    """
    df_result = df.copy()
    
    vol_col_name = f'vol_{window}d'
    df_result[vol_col_name] = df_result.groupby(id_col)[return_col].transform(
        lambda x: x.rolling(window).std()
    )
    
    return df_result

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2,
                             price_col: str = 'close', id_col: str = 'id') -> pd.DataFrame:
    """
    Calcula bandas de Bollinger
    
    Args:
        df: DataFrame con los datos
        window: Ventana para cálculo
        num_std: Número de desviaciones estándar
        price_col: Columna de precios
        id_col: Columna de identificador
        
    Returns:
        DataFrame con bandas de Bollinger añadidas
    """
    df_result = df.copy()
    
    # Calcular SMA y std
    sma = df_result.groupby(id_col)[price_col].transform(lambda x: x.rolling(window).mean())
    std = df_result.groupby(id_col)[price_col].transform(lambda x: x.rolling(window).std())
    
    # Calcular bandas
    df_result[f'bb_upper_{window}'] = sma + (num_std * std)
    df_result[f'bb_lower_{window}'] = sma - (num_std * std)
    df_result[f'bb_width_{window}'] = (df_result[f'bb_upper_{window}'] - df_result[f'bb_lower_{window}']) / sma
    df_result[f'bb_position_{window}'] = (df_result[price_col] - df_result[f'bb_lower_{window}']) / (df_result[f'bb_upper_{window}'] - df_result[f'bb_lower_{window}'])
    
    return df_result

def calculate_future_returns(df: pd.DataFrame, periods: List[int] = [30],
                           price_col: str = 'close', id_col: str = 'id') -> pd.DataFrame:
    """
    Calcula retornos futuros (targets para ML)
    
    Args:
        df: DataFrame con los datos
        periods: Lista de períodos futuros
        price_col: Columna de precios
        id_col: Columna de identificador
        
    Returns:
        DataFrame con columnas de retornos futuros
    """
    df_result = df.copy()
    
    for period in periods:
        col_name = f'future_ret_{period}d'
        df_result[col_name] = df_result.groupby(id_col)[price_col].pct_change(-period)
    
    return df_result

def create_technical_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea todas las features técnicas basándose en configuración
    
    Args:
        df: DataFrame con los datos
        config: Diccionario de configuración
        
    Returns:
        DataFrame con todas las features técnicas
    """
    df_features = df.copy()
    
    # Calcular retornos
    if 'returns' in config:
        df_features = calculate_returns(df_features, config['returns'])
    
    # Calcular promedios móviles
    if 'moving_averages' in config:
        df_features = calculate_moving_averages(df_features, config['moving_averages'])
    
    # Calcular volatilidad
    if 'volatility_window' in config and 'ret_1d' in df_features.columns:
        df_features = calculate_volatility(df_features, config['volatility_window'])
    
    # Calcular bandas de Bollinger
    if 'bollinger_window' in config:
        df_features = calculate_bollinger_bands(df_features, config['bollinger_window'])
    
    return df_features

def filter_tokens_by_history(df: pd.DataFrame, min_days: int = 60, 
                           date_col: str = 'date', id_col: str = 'id') -> pd.DataFrame:
    """
    Filtra tokens con histórico mínimo suficiente
    
    Args:
        df: DataFrame con los datos
        min_days: Mínimo número de días requeridos
        date_col: Columna de fecha
        id_col: Columna de identificador
        
    Returns:
        DataFrame filtrado
    """
    lengths = df.groupby(id_col)[date_col].nunique().reset_index(name='n_days')
    good_ids = lengths[lengths['n_days'] >= min_days][id_col]
    
    return df[df[id_col].isin(good_ids)].copy()

def prepare_ml_dataset(df: pd.DataFrame, target_col: str = 'future_ret_30d',
                      categorical_cols: List[str] = ['narrative'],
                      drop_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara el dataset final para machine learning
    
    Args:
        df: DataFrame con todas las features
        target_col: Columna objetivo
        categorical_cols: Columnas categóricas para encoding
        drop_cols: Columnas a eliminar
        
    Returns:
        Tuple con (X, y) preparados para ML
    """
    if drop_cols is None:
        drop_cols = ['date', 'id', 'name', 'symbol']
    
    # Copiar y limpiar datos
    ml_df = df.copy()
    
    # Eliminar filas con target faltante
    ml_df = ml_df.dropna(subset=[target_col])
    
    # Seleccionar columnas válidas
    available_cols = [col for col in ml_df.columns if col not in drop_cols]
    ml_df = ml_df[available_cols]
    
    # Separar X e y
    y = ml_df[target_col]
    X = ml_df.drop(columns=[target_col])
    
    # One-hot encoding para variables categóricas
    X_encoded = pd.get_dummies(X, columns=[col for col in categorical_cols if col in X.columns], 
                              drop_first=True, dummy_na=False)
    
    # Eliminar filas con valores faltantes
    valid_indices = X_encoded.dropna().index.intersection(y.dropna().index)
    X_final = X_encoded.loc[valid_indices]
    y_final = y.loc[valid_indices]
    
    return X_final, y_final

def add_clustering_features(df: pd.DataFrame, feature_cols: List[str], 
                          n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """
    Añade features de clustering al dataset
    
    Args:
        df: DataFrame con los datos
        feature_cols: Columnas para usar en clustering
        n_clusters: Número de clusters
        random_state: Semilla aleatoria
        
    Returns:
        DataFrame con columna de cluster añadida
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    df_result = df.copy()
    
    # Seleccionar features válidas
    available_features = [col for col in feature_cols if col in df_result.columns]
    
    if len(available_features) == 0:
        df_result['cluster_id'] = '0'
        return df_result
    
    # Preparar datos para clustering
    X_cluster = df_result[available_features].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(X_cluster) == 0:
        df_result['cluster_id'] = '0'
        return df_result
    
    # Estandarizar y aplicar KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Asignar clusters
    df_result['cluster_id'] = '0'  # Default
    df_result.loc[X_cluster.index, 'cluster_id'] = cluster_labels.astype(str)
    
    return df_result

def create_lagged_features(df: pd.DataFrame, feature_cols: List[str], 
                          lags: List[int] = [1, 3, 7], id_col: str = 'id') -> pd.DataFrame:
    """
    Crea features con rezagos temporales
    
    Args:
        df: DataFrame con los datos
        feature_cols: Columnas para crear rezagos
        lags: Lista de rezagos a crear
        id_col: Columna de identificador
        
    Returns:
        DataFrame con features rezagadas
    """
    df_result = df.copy()
    
    for feature in feature_cols:
        if feature in df_result.columns:
            for lag in lags:
                lag_col_name = f'{feature}_lag_{lag}'
                df_result[lag_col_name] = df_result.groupby(id_col)[feature].shift(lag)
    
    return df_result

def calculate_momentum_features(df: pd.DataFrame, price_col: str = 'close',
                              periods: List[int] = [5, 10, 20], id_col: str = 'id') -> pd.DataFrame:
    """
    Calcula features de momentum
    
    Args:
        df: DataFrame con los datos
        price_col: Columna de precios
        periods: Períodos para calcular momentum
        id_col: Columna de identificador
        
    Returns:
        DataFrame con features de momentum
    """
    df_result = df.copy()
    
    for period in periods:
        # Rate of Change (ROC)
        roc_col = f'roc_{period}'
        df_result[roc_col] = df_result.groupby(id_col)[price_col].pct_change(period)
        
        # Momentum
        momentum_col = f'momentum_{period}'
        df_result[momentum_col] = df_result.groupby(id_col)[price_col].transform(
            lambda x: x / x.shift(period) - 1
        )
    
    return df_result
