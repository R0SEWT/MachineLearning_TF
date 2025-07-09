# EDA Crypto - Análisis Modularizado

Este directorio contiene el análisis exploratorio de datos (EDA) para criptomonedas en versión modularizada, lo que permite mayor reutilización, mantenimiento y escalabilidad del código.

## 📁 Estructura de Archivos

```
EDA/
├── EDA_crypto.ipynb           # Notebook original mejorado
├── EDA_crypto_modular.ipynb   # Notebook completamente modularizado
├── README.md                  # Este archivo
└── utils/                     # Módulos de utilidades
    ├── __init__.py
    ├── config.py              # Configuraciones y constantes
    ├── data_analysis.py       # Funciones de análisis estadístico
    ├── visualizations.py      # Funciones de visualización
    └── feature_engineering.py # Funciones de ingeniería de características
```

## 🔧 Módulos Creados

### 1. `config.py`
Contiene todas las configuraciones centralizadas:
- **Colores de narrativas**: Paleta consistente para visualizaciones
- **Rutas del proyecto**: Gestión automática de paths
- **Configuraciones de análisis**: Parámetros para algoritmos
- **Umbrales de calidad**: Criterios de evaluación del dataset
- **Columnas esperadas**: Definición de estructura de datos

### 2. `data_analysis.py`
Funciones especializadas para análisis estadístico:
- `calculate_basic_metrics()`: Métricas básicas del dataset
- `evaluate_data_quality()`: Evaluación automática de calidad
- `detect_outliers_iqr()`: Detección de outliers por IQR
- `calculate_distribution_stats()`: Estadísticas de distribución
- `calculate_market_dominance()`: Análisis de dominancia de mercado
- `generate_summary_report()`: Generación de reportes automáticos

### 3. `visualizations.py`
Funciones especializadas para visualización:
- `plot_narrative_distribution()`: Distribución por narrativas
- `plot_market_cap_analysis()`: Análisis visual de market cap
- `plot_temporal_analysis()`: Patrones temporales avanzados
- `plot_returns_analysis()`: Análisis de distribuciones de retornos
- `plot_quality_dashboard()`: Dashboard ejecutivo de calidad

### 4. `feature_engineering.py`
Funciones para creación de características:
- `calculate_returns()`: Retornos para múltiples períodos
- `calculate_moving_averages()`: Promedios móviles
- `calculate_volatility()`: Volatilidad móvil
- `calculate_bollinger_bands()`: Bandas de Bollinger
- `create_technical_features()`: Pipeline completo de features
- `filter_tokens_by_history()`: Filtrado por histórico mínimo
- `prepare_ml_dataset()`: Preparación final para ML
- `add_clustering_features()`: Segmentación automática

## 🚀 Uso Rápido

### Importar Módulos
```python
import sys
sys.path.append('./utils')

from config import NARRATIVE_COLORS, get_project_paths, ANALYSIS_CONFIG
from data_analysis import calculate_basic_metrics, evaluate_data_quality
from visualizations import plot_quality_dashboard
from feature_engineering import create_technical_features
```

### Análisis Básico
```python
# Cargar configuración
paths = get_project_paths()
df = pd.read_csv(paths['data'])

# Calcular métricas
metrics = calculate_basic_metrics(df)
quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)

# Visualizar calidad
fig = plot_quality_dashboard(metrics, quality, df, NARRATIVE_COLORS)
```

### Feature Engineering
```python
# Crear características técnicas
df_features = create_technical_features(df, TECHNICAL_FEATURES)

# Preparar para ML
X, y = prepare_ml_dataset(df_features)
```

## ✅ Beneficios de la Modularización

### 🔄 Reutilización
- Las funciones pueden usarse en otros proyectos de crypto
- Código base para análisis similares
- Bibliotecas internas reutilizables

### 🛠 Mantenimiento
- Cada función tiene responsabilidad única
- Fácil localizar y corregir bugs
- Actualizaciones modulares sin afectar todo el código

### 📖 Legibilidad
- Notebooks más claros y enfocados en el flujo
- Separación entre lógica y presentación
- Documentación centralizada

### 🧪 Testeo
- Cada función puede probarse independientemente
- Tests unitarios más fáciles de implementar
- Validación modular de funcionalidad

### ⚙️ Configuración
- Parámetros centralizados y fáciles de modificar
- Configuraciones por ambiente (dev, prod)
- Gestión consistente de constantes

### 📈 Escalabilidad
- Fácil añadir nuevas funcionalidades
- Arquitectura extensible
- Soporte para múltiples datasets

## 🎯 Casos de Uso

### 1. Análisis Rápido de Nuevo Dataset
```python
# Solo cambiar la ruta del archivo
paths['data'] = 'nuevo_dataset.csv'
df = pd.read_csv(paths['data'])

# El resto del análisis es automático
metrics = calculate_basic_metrics(df)
quality = evaluate_data_quality(metrics, QUALITY_THRESHOLDS)
```

### 2. Personalización de Visualizaciones
```python
# Modificar colores para nuevas narrativas
NARRATIVE_COLORS['defi'] = '#ffa726'
NARRATIVE_COLORS['nft'] = '#ab47bc'

# Las visualizaciones se adaptan automáticamente
fig = plot_narrative_distribution(df, NARRATIVE_COLORS)
```

### 3. Pipeline de Feature Engineering
```python
# Configurar features personalizadas
custom_config = {
    'returns': [1, 3, 7, 14, 30],
    'moving_averages': [5, 10, 20, 50],
    'volatility_window': 14
}

# Aplicar pipeline
df_features = create_technical_features(df, custom_config)
```

## 🔬 Testing y Validación

### Tests Recomendados
```python
# Test de funciones básicas
def test_calculate_basic_metrics():
    sample_df = create_sample_dataframe()
    metrics = calculate_basic_metrics(sample_df)
    assert 'total_observations' in metrics
    assert metrics['total_observations'] > 0

# Test de visualizaciones
def test_plot_quality_dashboard():
    fig = plot_quality_dashboard(metrics, quality, df, NARRATIVE_COLORS)
    assert fig is not None
    assert len(fig.axes) == 6
```

## 📊 Configuraciones Disponibles

### Narrativas Soportadas
- `meme`: Meme coins
- `rwa`: Real World Assets
- `gaming`: Gaming tokens
- `ai`: Artificial Intelligence
- `defi`: Decentralized Finance
- `infrastructure`: Infrastructure tokens

### Parámetros Configurables
- **Outlier contamination**: 5% por defecto
- **Histórico mínimo**: 60 días
- **Ventana de volatilidad**: 30 días
- **Ventana de correlación**: 90 días
- **Número de clusters**: 4 por defecto

## 🚀 Próximos Pasos

### Mejoras Planificadas
1. **Testing**: Implementar suite completa de tests
2. **Logging**: Sistema de logs para debugging
3. **Performance**: Optimización de funciones críticas
4. **Validación**: Validación robusta de inputs
5. **Documentation**: Documentación API detallada
6. **CI/CD**: Integración continua

### Extensiones Posibles
1. **Análisis de sentimiento**: Módulo para datos de redes sociales
2. **Análisis técnico avanzado**: Más indicadores técnicos
3. **Risk metrics**: Métricas de riesgo especializadas
4. **Portfolio analysis**: Análisis de portafolios
5. **Real-time**: Soporte para datos en tiempo real

## 📞 Soporte

Para preguntas sobre el uso de estos módulos:
1. Revisar la documentación en cada archivo
2. Consultar los ejemplos en los notebooks
3. Verificar los tests unitarios
4. Contactar al equipo de desarrollo

---

**Versión**: 1.0  
**Última actualización**: 8 de julio de 2025  
**Autor**: Equipo ML-TF-G  
**Estado**: Producción
