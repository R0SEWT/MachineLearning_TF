
# 🏗️ Documentación del Sistema Modularizado

## 🎯 Visión General

El sistema de análisis de criptomonedas ha sido completamente **modularizado** para mejorar la mantenibilidad, reutilización y escalabilidad del código.

## 📋 Arquitectura del Sistema

```
crypto_eda_system/
├── 📁 utils/
│   ├── 🔧 config.py          # Configuraciones y constantes
│   ├── 📊 data_analysis.py   # Análisis estadístico y calidad
│   ├── 📈 visualizations.py  # Funciones de visualización
│   ├── 🔧 feature_engineering.py # Ingeniería de características
│   └── 📦 __init__.py        # Inicialización del paquete
├── 📓 EDA_crypto_modular.ipynb # Notebook principal modularizado
├── 🧪 test_modules.py        # Suite de pruebas
├── 🔍 quality_checker.py     # Analizador de calidad
└── 🔧 auto_formatter.py      # Reformateador automático
```

## 🌟 Características Principales

### ✅ **Modularidad**
- Separación clara de responsabilidades
- Código reutilizable entre proyectos
- Fácil mantenimiento y testing

### ✅ **Calidad de Código**
- Type hints en funciones críticas
- Documentación completa
- Estándares PEP 8
- Suite de pruebas automatizada

### ✅ **Funcionalidad Completa**
- Análisis estadístico avanzado
- Visualizaciones profesionales
- Ingeniería de características automática
- Pipeline completo de ML

### ✅ **Herramientas de Desarrollo**
- Tests automatizados
- Análisis de calidad
- Reformateo automático
- Documentación auto-generada

## 🔄 Flujo de Trabajo

1. **Configuración** (`config.py`) → Define parámetros y constantes
2. **Carga de Datos** → Import y validación inicial  
3. **Análisis de Calidad** (`data_analysis.py`) → Evaluación del dataset
4. **Visualización** (`visualizations.py`) → Gráficos y dashboards
5. **Feature Engineering** (`feature_engineering.py`) → Preparación para ML
6. **Testing** → Validación continua con `test_modules.py`


## 📁 VISUALIZATIONS

**Descripción:** Módulo de visualizaciones para el análisis EDA de criptomonedas

### 📊 Métricas del Código
- **Total de líneas:** 494
- **Líneas de código:** 375
- **Líneas de comentarios:** 41
- **Ratio comentarios/código:** 10.9%

### 🔧 Funciones Disponibles (5)


#### 🟡 `plot_narrative_distribution()`
- **Parámetros:** `df, colors, figsize`
- **Líneas:** 11-49 (39 líneas)
- **Complejidad:** media
- **Descripción:** Crea gráficos de distribución por narrativa

Args:
    df: DataFrame con los datos
    colors: Dicci...


#### 🔴 `plot_market_cap_analysis()`
- **Parámetros:** `df, colors, figsize`
- **Líneas:** 51-99 (49 líneas)
- **Complejidad:** compleja
- **Descripción:** Crea análisis visual del market cap por narrativa

Args:
    df: DataFrame con los datos
    colors:...


#### 🔴 `plot_temporal_analysis()`
- **Parámetros:** `df, colors, figsize`
- **Líneas:** 101-206 (106 líneas)
- **Complejidad:** compleja
- **Descripción:** Crea análisis temporal avanzado

Args:
    df: DataFrame con los datos
    colors: Diccionario de co...


#### 🔴 `plot_returns_analysis()`
- **Parámetros:** `df, colors, figsize`
- **Líneas:** 208-376 (169 líneas)
- **Complejidad:** compleja
- **Descripción:** Crea análisis de distribuciones de retornos

Args:
    df: DataFrame con los datos
    colors: Dicci...


#### 🔴 `plot_quality_dashboard()`
- **Parámetros:** `metrics, quality_eval, df, colors, figsize`
- **Líneas:** 378-494 (117 líneas)
- **Complejidad:** compleja
- **Descripción:** Crea dashboard de calidad consolidado

Args:
    metrics: Métricas del dataset
    quality_eval: Eva...


## 📁 DATA_ANALYSIS

**Descripción:** Módulo de utilidades para análisis estadístico y de calidad de datos

### 📊 Métricas del Código
- **Total de líneas:** 307
- **Líneas de código:** 239
- **Líneas de comentarios:** 15
- **Ratio comentarios/código:** 6.3%

### 🔧 Funciones Disponibles (9)


#### 🟡 `calculate_basic_metrics()`
- **Parámetros:** `df`
- **Líneas:** 12-43 (32 líneas)
- **Complejidad:** media
- **Descripción:** Calcula métricas básicas del dataset

Args:
    df: DataFrame con los datos
    
Returns:
    Dict c...


#### 🟢 `get_basic_metrics()`
- **Parámetros:** `df`
- **Líneas:** 46-56 (11 líneas)
- **Complejidad:** simple
- **Descripción:** Alias para calculate_basic_metrics - Calcula métricas básicas del dataset

Args:
    df: DataFrame c...


#### 🔴 `evaluate_data_quality()`
- **Parámetros:** `metrics, thresholds`
- **Líneas:** 58-130 (73 líneas)
- **Complejidad:** compleja
- **Descripción:** Evalúa la calidad del dataset basándose en métricas y umbrales

Args:
    metrics: Diccionario con m...


#### 🟡 `detect_outliers_iqr()`
- **Parámetros:** `series`
- **Líneas:** 132-152 (21 líneas)
- **Complejidad:** media
- **Descripción:** Detecta outliers usando el método IQR

Args:
    series: Serie de datos para analizar
    
Returns:
...


#### 🟡 `calculate_distribution_stats()`
- **Parámetros:** `series`
- **Líneas:** 154-184 (31 líneas)
- **Complejidad:** media
- **Descripción:** Calcula estadísticas de distribución para una serie

Args:
    series: Serie de datos
    
Returns:
...


#### 🟢 `calculate_market_dominance()`
- **Parámetros:** `df, group_col, value_col`
- **Líneas:** 186-202 (17 líneas)
- **Complejidad:** simple
- **Descripción:** Calcula la dominancia de mercado por grupo

Args:
    df: DataFrame con los datos
    group_col: Col...


#### 🟡 `filter_extreme_values()`
- **Parámetros:** `series, quantiles`
- **Líneas:** 204-227 (24 líneas)
- **Complejidad:** media
- **Descripción:** Filtra valores extremos basándose en cuantiles

Args:
    series: Serie de datos
    quantiles: Tupl...


#### 🟡 `calculate_correlation_matrix()`
- **Parámetros:** `df, tokens, date_col, value_col`
- **Líneas:** 229-259 (31 líneas)
- **Complejidad:** media
- **Descripción:** Calcula matriz de correlación para tokens seleccionados

Args:
    df: DataFrame con los datos
    t...


#### 🔴 `generate_summary_report()`
- **Parámetros:** `metrics, quality_eval`
- **Líneas:** 261-307 (47 líneas)
- **Complejidad:** compleja
- **Descripción:** Genera un resumen textual del análisis

Args:
    metrics: Métricas del dataset
    quality_eval: Ev...


## 📁 FEATURE_ENGINEERING

**Descripción:** Módulo de feature engineering para análisis de criptomonedas

### 📊 Métricas del Código
- **Total de líneas:** 331
- **Líneas de código:** 229
- **Líneas de comentarios:** 30
- **Ratio comentarios/código:** 13.1%

### 🔧 Funciones Disponibles (11)


#### 🟡 `calculate_returns()`
- **Parámetros:** `df, periods, price_col, id_col`
- **Líneas:** 9-30 (22 líneas)
- **Complejidad:** media
- **Descripción:** Calcula retornos para diferentes períodos

Args:
    df: DataFrame con los datos
    periods: Lista ...


#### 🟡 `calculate_moving_averages()`
- **Parámetros:** `df, windows, price_col, id_col`
- **Líneas:** 32-55 (24 líneas)
- **Complejidad:** media
- **Descripción:** Calcula promedios móviles para diferentes ventanas

Args:
    df: DataFrame con los datos
    window...


#### 🟡 `calculate_volatility()`
- **Parámetros:** `df, window, return_col, id_col`
- **Líneas:** 57-79 (23 líneas)
- **Complejidad:** media
- **Descripción:** Calcula volatilidad móvil

Args:
    df: DataFrame con los datos
    window: Ventana para cálculo de...


#### 🟡 `calculate_bollinger_bands()`
- **Parámetros:** `df, window, num_std, price_col, id_col`
- **Líneas:** 81-109 (29 líneas)
- **Complejidad:** media
- **Descripción:** Calcula bandas de Bollinger

Args:
    df: DataFrame con los datos
    window: Ventana para cálculo
...


#### 🟡 `calculate_future_returns()`
- **Parámetros:** `df, periods, price_col, id_col`
- **Líneas:** 111-132 (22 líneas)
- **Complejidad:** media
- **Descripción:** Calcula retornos futuros (targets para ML)

Args:
    df: DataFrame con los datos
    periods: Lista...


#### 🟡 `create_technical_features()`
- **Parámetros:** `df, config`
- **Líneas:** 134-164 (31 líneas)
- **Complejidad:** media
- **Descripción:** Crea todas las features técnicas basándose en configuración

Args:
    df: DataFrame con los datos
 ...


#### 🟢 `filter_tokens_by_history()`
- **Parámetros:** `df, min_days, date_col, id_col`
- **Líneas:** 166-183 (18 líneas)
- **Complejidad:** simple
- **Descripción:** Filtra tokens con histórico mínimo suficiente

Args:
    df: DataFrame con los datos
    min_days: M...


#### 🔴 `prepare_ml_dataset()`
- **Parámetros:** `df, target_col, categorical_cols, drop_cols`
- **Líneas:** 185-228 (44 líneas)
- **Complejidad:** compleja
- **Descripción:** Prepara el dataset final para machine learning

Args:
    df: DataFrame con todas las features
    t...


#### 🔴 `add_clustering_features()`
- **Parámetros:** `df, feature_cols, n_clusters, random_state`
- **Líneas:** 230-276 (47 líneas)
- **Complejidad:** compleja
- **Descripción:** Añade features de clustering al dataset

Args:
    df: DataFrame con los datos
    feature_cols: Col...


#### 🟡 `create_lagged_features()`
- **Parámetros:** `df, feature_cols, lags, id_col`
- **Líneas:** 278-301 (24 líneas)
- **Complejidad:** media
- **Descripción:** Crea features con rezagos temporales

Args:
    df: DataFrame con los datos
    feature_cols: Column...


#### 🟡 `calculate_momentum_features()`
- **Parámetros:** `df, price_col, periods, id_col`
- **Líneas:** 303-331 (29 líneas)
- **Complejidad:** media
- **Descripción:** Calcula features de momentum

Args:
    df: DataFrame con los datos
    price_col: Columna de precio...


## 📁 CONFIG

**Descripción:** Módulo de configuración para análisis EDA de criptomonedas
Contiene configuraciones, colores y constantes utilizadas en el análisis

### 📊 Métricas del Código
- **Total de líneas:** 90
- **Líneas de código:** 71
- **Líneas de comentarios:** 9
- **Ratio comentarios/código:** 12.7%

### 🔧 Funciones Disponibles (2)


#### 🟢 `setup_plotting_style()`
- **Parámetros:** `Sin parámetros`
- **Líneas:** 14-20 (7 líneas)
- **Complejidad:** simple
- **Descripción:** Configura el estilo visual para los gráficos


#### 🟢 `get_project_paths()`
- **Parámetros:** `Sin parámetros`
- **Líneas:** 33-45 (13 líneas)
- **Complejidad:** simple
- **Descripción:** Obtiene las rutas principales del proyecto


## 💡 Ejemplos de Uso

### 🚀 Uso Básico

```python
# Importar módulos
from utils.config import NARRATIVE_COLORS, setup_plotting_style
from utils.data_analysis import calculate_basic_metrics, evaluate_data_quality
from utils.visualizations import plot_narrative_distribution
from utils.feature_engineering import calculate_returns, create_technical_features

# Configurar entorno
setup_plotting_style()

# Cargar y analizar datos
import pandas as pd
df = pd.read_csv("data/crypto_ohlc_join.csv")

# Análisis básico
metrics = calculate_basic_metrics(df)
print(f"Dataset: {metrics['total_observations']:,} observaciones")
print(f"Tokens: {metrics['total_tokens']} únicos")

# Evaluación de calidad
from utils.config import QUALITY_THRESHOLDS
quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
print(f"Calidad: {quality['overall_score']}/10")

# Visualización
plot_narrative_distribution(df, NARRATIVE_COLORS)

# Feature engineering
df_with_returns = calculate_returns(df, periods=[1, 7, 30])
df_with_features = create_technical_features(df_with_returns, config)
```

### 🔧 Uso Avanzado

```python
# Pipeline completo de ML
from utils.feature_engineering import prepare_ml_dataset, add_clustering_features

# Preparar dataset para machine learning
ml_data = prepare_ml_dataset(
    df_with_features, 
    target_col='future_ret_30d',
    feature_columns=['ret_1d', 'ret_7d', 'vol_30d', 'rsi']
)

# Agregar clustering
df_clustered = add_clustering_features(
    ml_data, 
    feature_cols=['market_cap', 'volume', 'volatility'],
    n_clusters=4
)

# Análisis por cluster
for cluster in df_clustered['cluster'].unique():
    cluster_data = df_clustered[df_clustered['cluster'] == cluster]
    print(f"Cluster {cluster}: {len(cluster_data)} tokens")
```

### 🧪 Testing y Calidad

```python
# Ejecutar tests
python test_modules.py

# Verificar calidad
python quality_checker.py

# Aplicar mejoras automáticas
python auto_formatter.py
```


## 📚 Referencia de API

### Funciones Principales por Módulo

#### 🔧 config.py
- `setup_plotting_style()` → Configurar estilo visual
- `get_project_paths()` → Obtener rutas del proyecto
- **Constantes:** `NARRATIVE_COLORS`, `QUALITY_THRESHOLDS`, `ANALYSIS_CONFIG`

#### 📊 data_analysis.py
- `calculate_basic_metrics(df)` → Métricas básicas del dataset
- `evaluate_data_quality(metrics, thresholds)` → Evaluación de calidad
- `detect_outliers_iqr(series)` → Detección de outliers
- `calculate_market_dominance(df)` → Análisis de dominancia
- `generate_summary_report(df, thresholds)` → Reporte automático

#### 📈 visualizations.py
- `plot_narrative_distribution(df, colors)` → Gráfico de narrativas
- `plot_market_cap_analysis(df, colors)` → Análisis de market cap
- `plot_temporal_analysis(df, colors)` → Análisis temporal
- `plot_returns_analysis(df, colors)` → Análisis de retornos
- `plot_quality_dashboard(metrics, quality_eval)` → Dashboard de calidad

#### 🔧 feature_engineering.py
- `calculate_returns(df, periods)` → Calcular retornos
- `calculate_moving_averages(df, windows)` → Promedios móviles
- `calculate_volatility(df, window)` → Volatilidad
- `create_technical_features(df, config)` → Features técnicos
- `prepare_ml_dataset(df, target_col)` → Preparar para ML
- `add_clustering_features(df, feature_cols)` → Agregar clusters

