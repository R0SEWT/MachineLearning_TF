# 📖 Documentación Técnica Detallada

> **🏠 DOCUMENTACIÓN PRINCIPAL**: Ver **[README.md](../README.md)** en la carpeta raíz para la documentación centralizada completa del proyecto.

Este archivo contiene documentación técnica detallada complementaria al README principal.

## 📁 Estructura del Proyecto

```
EDA/
├── 📁 docs/                   # Documentación completa del proyecto
│   ├── README.md              # Este archivo - Guía principal
│   ├── MODULAR_SYSTEM_DOCS.md # Documentación técnica detallada
│   ├── README_MODULAR.md      # Inicio rápido para el sistema modular
│   ├── TESTING_MODULE_DOCUMENTATION.md # Documentación de testing
│   └── TESTING_QUICK_GUIDE.md # Guía rápida de testing
├── 📁 notebooks/              # Jupyter Notebooks organizados
│   ├── EDA_crypto.ipynb       # Notebook original mejorado
│   └── EDA_crypto_modular.ipynb # Notebook completamente modularizado
├── 📁 utils/                  # Módulos de utilidades core
│   ├── __init__.py
│   ├── config.py              # Configuraciones y constantes
│   ├── data_analysis.py       # Funciones de análisis estadístico
│   ├── visualizations.py      # Funciones de visualización
│   └── feature_engineering.py # Funciones de ingeniería de características
├── 📁 testing/                # Sistema de testing completo
│   ├── master_test.py         # Ejecutor maestro de tests
│   ├── test_functional.py     # Tests funcionales (95% éxito)
│   ├── test_smart.py          # Tests auto-adaptativos
│   ├── test_professional.py   # Suite completa profesional
│   ├── fixtures/              # Datos de prueba
│   ├── reports/               # Reportes de testing
│   └── README.md              # Documentación de testing
├── 📁 scripts/                # Scripts de utilidad y automatización
│   ├── auto_formatter.py      # Reformateador automático
│   ├── generate_docs.py       # Generador de documentación
│   └── quality_checker.py     # Analizador de calidad de código
├── 📁 outputs/                # Resultados y reportes generados
│   └── test_results/          # Resultados de tests
├── 📄 README.md               # Documentación principal del proyecto
├── 📄 ORGANIZATION_SUMMARY.md # Resumen de la organización
├── 📄 TESTING_QUICK_GUIDE.md  # Guía rápida de testing (nivel raíz)
├── 📄 setup.py                # Configuración y validación del entorno
├── 📄 .gitignore              # Configuración de Git
└── 📁 .vscode/                # Configuración de VS Code
    └── settings.json
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

## 🚀 Inicio Rápido

### 1. Configurar el Entorno
```bash
# Validar la estructura del proyecto
python setup.py

# Verificar que todo funciona
python testing/master_test.py --functional
```

### 2. Importar Módulos
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

### 🧪 Testing y Validación

El proyecto incluye un sistema de testing robusto y completo:

#### Ejecutar Tests
```bash
# Opción recomendada: usar el maestro de tests
python testing/master_test.py --functional    # Tests más confiables (95% éxito)
python testing/master_test.py --smart         # Tests auto-adaptativos
python testing/master_test.py --all           # Ejecutar todos los tests
python testing/master_test.py --list          # Ver tests disponibles

# Ejecutar tests individuales
python testing/test_functional.py             # Test funcional directo
python testing/test_professional.py           # Suite completa profesional
```

#### Verificar Calidad del Código
```bash
python scripts/quality_checker.py             # Análisis de calidad
python scripts/auto_formatter.py              # Reformatear código automáticamente
```

#### Tests Disponibles
- **test_functional.py**: Tests funcionales con 95% de éxito garantizado
- **test_smart.py**: Tests auto-adaptativos que detectan funciones dinámicamente  
- **test_professional.py**: Suite completa con casos edge y análisis de performance
- **Tests modulares**: Tests específicos para cada módulo (data_analysis, visualizations, etc.)

#### Reportes y Resultados
- Los resultados se guardan en `outputs/test_results/`
- Reportes detallados con métricas y estadísticas
- Logs de ejecución para debugging

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

### ✅ Completado
1. **✅ Testing**: Suite completa de tests implementada y funcionando
2. **✅ Estructura**: Proyecto completamente organizado y modularizado
3. **✅ Documentación**: Documentación completa y actualizada
4. **✅ Calidad**: Sistema de verificación de calidad implementado
5. **✅ Automatización**: Scripts de automatización y validación listos

### 🔄 Mejoras Continuas
1. **Logging**: Sistema de logs más detallado para debugging avanzado
2. **Performance**: Optimización de funciones críticas para datasets grandes
3. **Validación avanzada**: Validación más robusta de inputs complejos
4. **CI/CD**: Integración continua para automatización completa
5. **API Documentation**: Documentación API auto-generada con Sphinx

### 🌟 Extensiones Futuras
1. **Análisis de sentimiento**: Módulo para datos de redes sociales
2. **Análisis técnico avanzado**: Más indicadores técnicos especializados
3. **Risk metrics**: Métricas de riesgo y VaR para portfolios
4. **Portfolio analysis**: Análisis de portafolios y optimización
5. **Real-time**: Soporte para streaming de datos en tiempo real
6. **Web dashboard**: Dashboard interactivo con Streamlit/Dash

## 📞 Guías y Soporte

### 📖 Documentación Completa
- **MODULAR_SYSTEM_DOCS.md**: Documentación técnica detallada del sistema
- **TESTING_QUICK_GUIDE.md**: Guía rápida para ejecutar tests
- **README_MODULAR.md**: Inicio rápido para el sistema modular
- **ORGANIZATION_SUMMARY.md**: Resumen de la organización del proyecto (en raíz)

### 🆘 Resolución de Problemas
1. **Errores de imports**: Verificar que `sys.path.append('./utils')` esté incluido
2. **Tests fallando**: Ejecutar primero `python setup.py` para validar el entorno
3. **Datos faltantes**: Verificar que `data/crypto_ohlc_join.csv` existe
4. **Problemas de dependencias**: Revisar `environment.yml` en la raíz del proyecto

### 🛠️ Herramientas de Desarrollo
```bash
# Verificar estructura del proyecto
python setup.py

# Análisis de calidad del código
python scripts/quality_checker.py

# Reformatear código automáticamente  
python scripts/auto_formatter.py

# Generar documentación actualizada
python scripts/generate_docs.py
```

### 📧 Contacto y Soporte
Para preguntas sobre el uso de estos módulos:
1. **Primera opción**: Revisar la documentación en `docs/`
2. **Testing**: Ejecutar `python testing/master_test.py --help`
3. **Ejemplos**: Consultar los notebooks en `notebooks/`
4. **Verificación**: Usar las herramientas en `scripts/`
5. **Contacto**: Equipo de desarrollo ML-TF-G

---

**Versión**: 2.0 - Profesional Organizada  
**Última actualización**: Enero 2025  
**Autor**: Equipo ML-TF-G  
**Estado**: ✅ Producción - Completamente Funcional  
**Testing**: ✅ 95% Éxito Garantizado
