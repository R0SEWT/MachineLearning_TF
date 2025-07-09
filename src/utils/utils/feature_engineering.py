"""
Módulo de feature engineering para análisis de criptomonedas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Importaciones adicionales para ML
try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    # --- Sección principal de procesamiento ---
    
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
    # --- Sección principal de procesamiento ---
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
    # --- Sección principal de procesamiento ---
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
    # --- Sección principal de procesamiento ---
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
    # --- Sección principal de procesamiento ---
    
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
    # --- Sección principal de procesamiento ---
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
    # --- Sección principal de procesamiento ---
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
    # --- Sección de validación y resultados ---
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
    df_result = df.copy()
    
    if not SKLEARN_AVAILABLE:
        print("   ⚠️  sklearn no disponible, asignando cluster por defecto")
        df_result['cluster_id'] = '0'
        return df_result
    
    # --- Sección principal de procesamiento ---
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
    # --- Sección de validación y resultados ---
    
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
    # --- Sección principal de procesamiento ---
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
    # --- Sección principal de procesamiento ---
        df_result[roc_col] = df_result.groupby(id_col)[price_col].pct_change(period)
        
        # Momentum
        momentum_col = f'momentum_{period}'
        df_result[momentum_col] = df_result.groupby(id_col)[price_col].transform(
            lambda x: x / x.shift(period) - 1
        )
    
    return df_result

def create_target_variables(df: pd.DataFrame, future_periods: List[int] = [7, 14, 30],
                           price_col: str = 'close', id_col: str = 'id') -> pd.DataFrame:
    """
    Crear variables objetivo para predicción de retornos futuros
    
    Args:
        df: DataFrame con los datos
        future_periods: Períodos futuros para calcular retornos
        price_col: Columna de precios
        id_col: Columna de identificador
        
    Returns:
        DataFrame con variables objetivo añadidas
    """
    df_result = df.copy()
    
    # Asegurar que esté ordenado por fecha
    df_result = df_result.sort_values(['id', 'date']).reset_index(drop=True)
    
    for period in future_periods:
        # Retorno futuro exacto
        future_return_col = f'future_return_{period}d'
        df_result[future_return_col] = df_result.groupby(id_col)[price_col].pct_change(period).shift(-period)
        
        # Clasificación binaria: retorno > 100%
        high_return_col = f'high_return_{period}d'
        df_result[high_return_col] = (df_result[future_return_col] > 1.0).astype(int)
        
        # Clasificación multi-clase por rangos de retorno
        category_col = f'return_category_{period}d'
        conditions = [
            df_result[future_return_col] < -0.2,  # Pérdida > 20%
            (df_result[future_return_col] >= -0.2) & (df_result[future_return_col] < 0.5),  # Estable
            (df_result[future_return_col] >= 0.5) & (df_result[future_return_col] < 1.0),   # Ganancia media
            df_result[future_return_col] >= 1.0   # Ganancia alta > 100%
        ]
        df_result[category_col] = np.select(conditions, [0, 1, 2, 3], default=1)
        
        # Variable de ganancia extrema (>200%)
        extreme_return_col = f'extreme_return_{period}d'
        df_result[extreme_return_col] = (df_result[future_return_col] > 2.0).astype(int)
    
    return df_result

def add_technical_indicators(df: pd.DataFrame, id_col: str = 'id', 
                           price_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
    """
    Agregar indicadores técnicos avanzados
    
    Args:
        df: DataFrame con los datos
        id_col: Columna de identificador
        price_col: Columna de precios
        volume_col: Columna de volumen
        
    Returns:
        DataFrame con indicadores técnicos añadidos
    """
    df_result = df.copy()
    
    # Asegurar orden correcto
    df_result = df_result.sort_values([id_col, 'date']).reset_index(drop=True)
    
    def calculate_for_group(group):
        """Calcular indicadores para cada token"""
        # RSI (Relative Strength Index)
        delta = group[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        group['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = group[price_col].ewm(span=12).mean()
        ema26 = group[price_col].ewm(span=26).mean()
        group['macd'] = ema12 - ema26
        group['macd_signal'] = group['macd'].ewm(span=9).mean()
        group['macd_histogram'] = group['macd'] - group['macd_signal']
        
        # Bollinger Bands
        sma20 = group[price_col].rolling(window=20).mean()
        std20 = group[price_col].rolling(window=20).std()
        group['bb_upper'] = sma20 + (std20 * 2)
        group['bb_lower'] = sma20 - (std20 * 2)
        group['bb_width'] = (group['bb_upper'] - group['bb_lower']) / sma20
        group['bb_position'] = (group[price_col] - group['bb_lower']) / (group['bb_upper'] - group['bb_lower'])
        
        # ATR (Average True Range)
        if 'high' in group.columns and 'low' in group.columns:
            high_low = group['high'] - group['low']
            high_close = abs(group['high'] - group[price_col].shift())
            low_close = abs(group['low'] - group[price_col].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        else:
            # Aproximación usando solo close price
            true_range = group[price_col].rolling(window=2).apply(lambda x: abs(x.iloc[1] - x.iloc[0]), raw=False)
        
        group['atr'] = true_range.rolling(window=14).mean()
        
        # Stochastic Oscillator (aproximado)
        if 'high' in group.columns and 'low' in group.columns:
            lowest_low = group['low'].rolling(window=14).min()
            highest_high = group['high'].rolling(window=14).max()
            group['stoch_k'] = 100 * (group[price_col] - lowest_low) / (highest_high - lowest_low)
        else:
            # Aproximación usando solo close
            lowest_close = group[price_col].rolling(window=14).min()
            highest_close = group[price_col].rolling(window=14).max()
            group['stoch_k'] = 100 * (group[price_col] - lowest_close) / (highest_close - lowest_close)
        
        group['stoch_d'] = group['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        group['williams_r'] = -100 * (group['stoch_k'] / 100)
        
        # Momentum indicators
        group['momentum_10'] = group[price_col] / group[price_col].shift(10) - 1
        group['momentum_20'] = group[price_col] / group[price_col].shift(20) - 1
        
        # Price position in recent range
        group['price_position_20'] = (group[price_col] - group[price_col].rolling(20).min()) / (
            group[price_col].rolling(20).max() - group[price_col].rolling(20).min())
        
        return group
    
    df_result = df_result.groupby(id_col).apply(calculate_for_group).reset_index(drop=True)
    
    return df_result

def add_volume_features(df: pd.DataFrame, id_col: str = 'id',
                       price_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
    """
    Agregar features relacionadas con volumen y liquidez
    
    Args:
        df: DataFrame con los datos
        id_col: Columna de identificador  
        price_col: Columna de precios
        volume_col: Columna de volumen
        
    Returns:
        DataFrame con features de volumen añadidas
    """
    df_result = df.copy()
    
    def calculate_volume_features(group):
        """Calcular features de volumen para cada token"""
        # Volumen promedio en diferentes ventanas
        group['volume_sma_7'] = group[volume_col].rolling(window=7).mean()
        group['volume_sma_30'] = group[volume_col].rolling(window=30).mean()
        
        # Ratio de volumen actual vs promedio
        group['volume_ratio_7'] = group[volume_col] / group['volume_sma_7']
        group['volume_ratio_30'] = group[volume_col] / group['volume_sma_30']
        
        # OBV (On Balance Volume)
        price_change = group[price_col].diff()
        obv = np.where(price_change > 0, group[volume_col], 
                      np.where(price_change < 0, -group[volume_col], 0))
        group['obv'] = obv.cumsum()
        
        # VWAP (Volume Weighted Average Price) aproximado
        group['vwap'] = (group[price_col] * group[volume_col]).rolling(window=20).sum() / group[volume_col].rolling(window=20).sum()
        
        # Volume Rate of Change
        group['volume_roc'] = group[volume_col].pct_change(periods=7)
        
        # Unusual volume detection
        volume_std = group[volume_col].rolling(window=30).std()
        volume_mean = group[volume_col].rolling(window=30).mean()
        group['volume_spike'] = (group[volume_col] > volume_mean + 2 * volume_std).astype(int)
        
        # Dollar volume (precio * volumen)
        group['dollar_volume'] = group[price_col] * group[volume_col]
        group['dollar_volume_sma_7'] = group['dollar_volume'].rolling(window=7).mean()
        
        return group
    
    df_result = df_result.groupby(id_col).apply(calculate_volume_features).reset_index(drop=True)
    
    return df_result

def add_momentum_features(df: pd.DataFrame, id_col: str = 'id',
                         price_col: str = 'close') -> pd.DataFrame:
    """
    Agregar features de momentum y aceleración
    
    Args:
        df: DataFrame con los datos
        id_col: Columna de identificador
        price_col: Columna de precios
        
    Returns:
        DataFrame con features de momentum añadidas
    """
    df_result = df.copy()
    
    def calculate_momentum_features(group):
        """Calcular features de momentum para cada token"""
        # Retornos en múltiples períodos
        for period in [1, 3, 5, 7, 14, 21, 30]:
            group[f'return_{period}d'] = group[price_col].pct_change(period)
        
        # Volatilidad en diferentes ventanas
        for window in [7, 14, 30]:
            group[f'volatility_{window}d'] = group[price_col].pct_change().rolling(window=window).std()
        
        # Aceleración (cambio en el momentum)
        group['price_acceleration_3d'] = group['return_3d'].diff()
        group['price_acceleration_7d'] = group['return_7d'].diff()
        
        # Maximum y minimum returns en ventanas
        group['max_return_7d'] = group[price_col].pct_change().rolling(window=7).max()
        group['min_return_7d'] = group[price_col].pct_change().rolling(window=7).min()
        group['max_return_30d'] = group[price_col].pct_change().rolling(window=30).max()
        group['min_return_30d'] = group[price_col].pct_change().rolling(window=30).min()
        
        # Días desde máximo/mínimo
        group['days_since_high'] = (group[price_col].rolling(window=30).apply(lambda x: len(x) - 1 - x.argmax(), raw=False))
        group['days_since_low'] = (group[price_col].rolling(window=30).apply(lambda x: len(x) - 1 - x.argmin(), raw=False))
        
        # Consistency score (qué tan consistente es la tendencia)
        returns_3d = group[price_col].pct_change(3)
        group['momentum_consistency'] = returns_3d.rolling(window=10).apply(lambda x: (x > 0).sum() / len(x), raw=False)
        
        # Breakout detection
        price_max_20 = group[price_col].rolling(window=20).max().shift(1)
        price_min_20 = group[price_col].rolling(window=20).min().shift(1)
        group['breakout_up'] = (group[price_col] > price_max_20).astype(int)
        group['breakout_down'] = (group[price_col] < price_min_20).astype(int)
        
        # Support/Resistance strength
        group['support_strength'] = group[price_col].rolling(window=20).min()
        group['resistance_strength'] = group[price_col].rolling(window=20).max()
        group['support_distance'] = (group[price_col] - group['support_strength']) / group['support_strength']
        group['resistance_distance'] = (group['resistance_strength'] - group[price_col]) / group[price_col]
        
        return group
    
    df_result = df_result.groupby(id_col).apply(calculate_momentum_features).reset_index(drop=True)
    
    return df_result

def add_narrative_features(df: pd.DataFrame, id_col: str = 'id',
                          narrative_col: str = 'narrative', price_col: str = 'close') -> pd.DataFrame:
    """
    Agregar features relacionadas con narrativas
    
    Args:
        df: DataFrame con los datos
        id_col: Columna de identificador
        narrative_col: Columna de narrativa
        price_col: Columna de precios
        
    Returns:
        DataFrame con features de narrativa añadidas
    """
    df_result = df.copy()
    
    # Performance promedio por narrativa
    narrative_performance = df_result.groupby(['date', narrative_col])[price_col].mean().reset_index()
    narrative_performance.columns = ['date', narrative_col, 'narrative_avg_price']
    df_result = df_result.merge(narrative_performance, on=['date', narrative_col], how='left')
    
    # Ranking dentro de la narrativa
    df_result['narrative_rank'] = df_result.groupby(['date', narrative_col])['market_cap'].rank(ascending=False)
    df_result['narrative_total_tokens'] = df_result.groupby(['date', narrative_col])[id_col].transform('count')
    df_result['narrative_percentile'] = df_result['narrative_rank'] / df_result['narrative_total_tokens']
    
    # Performance relativa vs narrativa
    df_result['relative_to_narrative'] = df_result[price_col] / df_result['narrative_avg_price']
    
    # Correlación con líderes de narrativa (aproximada)
    def calculate_narrative_correlation(group):
        """Calcular correlación con el líder de la narrativa"""
        if len(group) < 20:  # Necesitamos suficientes datos
            group['narrative_correlation'] = 0
            return group
            
        # Encontrar el token líder (mayor market cap promedio)
        leader_id = group.loc[group['market_cap'].idxmax(), id_col]
        leader_prices = group[group[id_col] == leader_id][price_col]
        
        # Calcular correlación rolling para cada token
        for token_id in group[id_col].unique():
            if token_id == leader_id:
                group.loc[group[id_col] == token_id, 'narrative_correlation'] = 1.0
            else:
                token_prices = group[group[id_col] == token_id][price_col]
                if len(token_prices) >= 20 and len(leader_prices) >= 20:
                    # Correlación simple (rolling sería más complejo)
                    corr = np.corrcoef(token_prices.iloc[-20:], leader_prices.iloc[-20:])[0, 1]
                    if np.isnan(corr):
                        corr = 0
                    group.loc[group[id_col] == token_id, 'narrative_correlation'] = corr
                else:
                    group.loc[group[id_col] == token_id, 'narrative_correlation'] = 0
        
        return group
    
    # Aplicar cálculo de correlación por narrativa
    df_result = df_result.groupby(narrative_col).apply(calculate_narrative_correlation).reset_index(drop=True)
    
    # One-hot encoding para narrativas
    narrative_dummies = pd.get_dummies(df_result[narrative_col], prefix='narrative')
    df_result = pd.concat([df_result, narrative_dummies], axis=1)
    
    return df_result

def add_timing_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Agregar features temporales y de estacionalidad
    
    Args:
        df: DataFrame con los datos
        date_col: Columna de fecha
        
    Returns:
        DataFrame con features temporales añadidas
    """
    df_result = df.copy()
    
    # Asegurar que date_col es datetime
    df_result[date_col] = pd.to_datetime(df_result[date_col])
    
    # Features básicas de tiempo
    df_result['day_of_week'] = df_result[date_col].dt.dayofweek
    df_result['day_of_month'] = df_result[date_col].dt.day
    df_result['month'] = df_result[date_col].dt.month
    df_result['quarter'] = df_result[date_col].dt.quarter
    df_result['is_weekend'] = (df_result['day_of_week'] >= 5).astype(int)
    df_result['is_month_end'] = (df_result[date_col].dt.day >= 28).astype(int)
    
    # Features cíclicas (sin y cos para capturar ciclicidad)
    df_result['day_of_week_sin'] = np.sin(2 * np.pi * df_result['day_of_week'] / 7)
    df_result['day_of_week_cos'] = np.cos(2 * np.pi * df_result['day_of_week'] / 7)
    df_result['month_sin'] = np.sin(2 * np.pi * df_result['month'] / 12)
    df_result['month_cos'] = np.cos(2 * np.pi * df_result['month'] / 12)
    
    # Días desde el inicio del dataset
    min_date = df_result[date_col].min()
    df_result['days_since_start'] = (df_result[date_col] - min_date).dt.days
    
    # Días desde listing (aproximado - primer aparición del token)
    df_result['days_since_listing'] = df_result.groupby('id')[date_col].transform(
        lambda x: (x - x.min()).dt.days
    )
    
    return df_result

def create_ml_features(df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
    """
    Crear todas las features para machine learning
    
    Args:
        df: DataFrame con los datos base
        include_targets: Si incluir variables objetivo
        
    Returns:
        DataFrame con todas las features para ML
    """
    print("🔧 Creando features para ML...")
    df_result = df.copy()
    
    # Asegurar que tenemos las columnas necesarias
    required_cols = ['close', 'date', 'id', 'market_cap', 'volume', 'narrative']
    missing_cols = [col for col in required_cols if col not in df_result.columns]
    if missing_cols:
        print(f"⚠️  Advertencia: Faltan columnas {missing_cols}")
    
    # Ordenar por id y fecha
    df_result = df_result.sort_values(['id', 'date']).reset_index(drop=True)
    
    # 1. Features técnicas
    print("   📊 Agregando indicadores técnicos...")
    df_result = add_technical_indicators(df_result)
    
    # 2. Features de volumen
    print("   📈 Agregando features de volumen...")
    df_result = add_volume_features(df_result)
    
    # 3. Features de momentum
    print("   🚀 Agregando features de momentum...")
    df_result = add_momentum_features(df_result)
    
    # 4. Features de narrativa
    print("   🎯 Agregando features de narrativa...")
    df_result = add_narrative_features(df_result)
    
    # 5. Features temporales
    print("   ⏰ Agregando features temporales...")
    df_result = add_timing_features(df_result)
    
    # 6. Variables objetivo (si se solicitan)
    if include_targets:
        print("   🎯 Creando variables objetivo...")
        df_result = create_target_variables(df_result)
    
    print(f"✅ Features creadas! Dimensiones: {df_result.shape}")
    print(f"   📊 Total de columnas: {len(df_result.columns)}")
    
    return df_result

def prepare_ml_dataset(df: pd.DataFrame, target_col: str = 'high_return_30d',
                      min_history_days: int = 60, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preparar dataset final para machine learning con validación temporal
    
    Args:
        df: DataFrame con todas las features
        target_col: Nombre de la columna objetivo
        min_history_days: Días mínimos de historia requeridos
        test_size: Proporción para conjunto de test
        
    Returns:
        Tuple con (X_train, X_test, y_train, y_test)
    """
    print("🎯 Preparando dataset para ML...")
    
    # Filtrar tokens con suficiente historia
    df_filtered = df.copy()
    token_counts = df_filtered.groupby('id').size()
    valid_tokens = token_counts[token_counts >= min_history_days].index
    df_filtered = df_filtered[df_filtered['id'].isin(valid_tokens)]
    
    print(f"   📊 Tokens con suficiente historia: {len(valid_tokens)}")
    print(f"   📅 Observaciones válidas: {len(df_filtered)}")
    
    # Eliminar filas con target NaN
    df_clean = df_filtered.dropna(subset=[target_col])
    print(f"   🧹 Observaciones después de limpiar: {len(df_clean)}")
    
    # Split temporal (las últimas fechas para test)
    df_clean = df_clean.sort_values('date')
    split_date = df_clean['date'].quantile(1 - test_size)
    
    train_mask = df_clean['date'] < split_date
    df_train = df_clean[train_mask]
    df_test = df_clean[~train_mask]
    
    print(f"   📅 Fecha de split: {split_date}")
    print(f"   🚂 Entrenamiento: {len(df_train)} observaciones")
    print(f"   🧪 Test: {len(df_test)} observaciones")
    
    # Seleccionar features (excluir columnas no útiles)
    exclude_cols = ['id', 'date', 'name', 'symbol', 'cmc_id', 'price'] + \
                   [col for col in df_clean.columns if col.startswith('future_') or 
                    col.startswith('high_return_') or col.startswith('return_category_') or
                    col.startswith('extreme_return_')]
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    # Preparar features con manejo de variables categóricas
    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    
    # Identificar y codificar variables categóricas
    categorical_cols = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or col in ['narrative', 'cluster_id']:
            categorical_cols.append(col)
    
    if categorical_cols:
        print(f"   🔤 Variables categóricas encontradas: {categorical_cols}")
        
        # One-hot encoding para variables categóricas
        if not SKLEARN_AVAILABLE:
            print("   ⚠️  sklearn no disponible, usando encoding básico")
            for col in categorical_cols:
                if col in X_train.columns:
                    # Encoding básico usando factorize
                    train_codes, train_uniques = pd.factorize(X_train[col].astype(str))
                    test_codes, _ = pd.factorize(X_test[col].astype(str))
                    X_train[col] = train_codes
                    X_test[col] = test_codes
                    print(f"      ✅ {col}: {len(train_uniques)} categorías únicas")
        else:
            for col in categorical_cols:
                if col in X_train.columns:
                    # Usar LabelEncoder para mantener compatibilidad con todos los algoritmos
                    le = LabelEncoder()
                    
                    # Combinar train y test para encoding consistente
                    combined_values = pd.concat([X_train[col], X_test[col]]).astype(str)
                    le.fit(combined_values)
                    
                    # Aplicar encoding
                    X_train[col] = le.transform(X_train[col].astype(str))
                    X_test[col] = le.transform(X_test[col].astype(str))
                    
                    print(f"      ✅ {col}: {len(le.classes_)} categorías únicas")
    
    # Convertir todas las columnas a numérico
    print("   🔢 Convirtiendo todas las columnas a numéricas...")
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Eliminar valores infinitos y NaN
    print("   🧹 Limpiando valores infinitos y NaN...")
    
    # Reemplazar infinitos con NaN y luego rellenar
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    # Contar valores problemáticos antes de limpiar
    inf_count_train = X_train.isnull().sum().sum()
    inf_count_test = X_test.isnull().sum().sum()
    
    if inf_count_train > 0 or inf_count_test > 0:
        print(f"      🔧 Valores problemáticos encontrados - Train: {inf_count_train}, Test: {inf_count_test}")
    
    # Rellenar NaN con 0 (estrategia conservadora)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Verificación final: asegurar que todo sea numérico
    print("   ✅ Verificación final de tipos de datos...")
    
    # Verificar que no hay columnas object
    object_cols_train = X_train.select_dtypes(include=['object']).columns
    object_cols_test = X_test.select_dtypes(include=['object']).columns
    
    if len(object_cols_train) > 0:
        print(f"      ⚠️  Columnas no numéricas en train: {list(object_cols_train)}")
        for col in object_cols_train:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
    
    if len(object_cols_test) > 0:
        print(f"      ⚠️  Columnas no numéricas en test: {list(object_cols_test)}")
        for col in object_cols_test:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
    
    # Verificación final (solo si todo es numérico)
    try:
        remaining_inf_train = np.isinf(X_train.values).sum()
        remaining_inf_test = np.isinf(X_test.values).sum()
        
        if remaining_inf_train > 0 or remaining_inf_test > 0:
            print(f"      🔧 Limpiando infinitos restantes - Train: {remaining_inf_train}, Test: {remaining_inf_test}")
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
    except Exception as e:
        print(f"      ⚠️  Error en verificación de infinitos: {e}")
        # Limpiar de forma segura
        X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Verificar tipos de datos
    non_numeric = X_train.select_dtypes(include=['object']).columns
    if len(non_numeric) > 0:
        print(f"   ⚠️  Columnas aún no numéricas: {list(non_numeric)}")
        # Convertir forzadamente a numérico
        for col in non_numeric:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
    
    y_train = df_train[target_col]
    y_test = df_test[target_col]
    
    print(f"   🔧 Features utilizadas: {len(feature_cols)}")
    print(f"   🔢 Todas las features son numéricas: {X_train.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all()}")
    print(f"   🎯 Distribución objetivo train: {y_train.value_counts().to_dict()}")
    print(f"   🎯 Distribución objetivo test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test
