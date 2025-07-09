# 🤖 Machine Learning Models - Sistema Completo

## 📖 Descripción

Directorio principal que contiene todo el sistema de Machine Learning para identificar criptomonedas de baja capitalización con alto potencial de retorno. Incluye pipeline de entrenamiento, optimización automática con Optuna, análisis de resultados y integración automática.

## 📁 Estructura de Archivos

```
code/Models/
├── 🧠 crypto_ml_trainer.py              # Pipeline principal de entrenamiento
├── 🚀 crypto_ml_trainer_optimized.py    # Trainer con hiperparámetros optimizados
├── 🔧 crypto_hyperparameter_optimizer.py # Sistema Optuna completo
├── ⚡ quick_optimization.py             # Scripts de optimización rápida
├── 📊 optuna_results_analyzer.py        # Analizador de resultados
├── 🔗 integrate_optimized_params.py     # Integrador automático
├── 🧪 test_ml_system.py                 # Tests del sistema ML
├── 📖 README_OPTIMIZATION.md            # Documentación optimización
├── 📄 OPTUNA_IMPLEMENTATION_COMPLETED.md # Reporte implementación
├── 📄 integration_report.md             # Reporte de integración
├── 💾 crypto_ml_trainer_backup_*.py     # Backups automáticos
├── 📓 Model_training.ipynb              # Entrenamiento legacy
└── 📁 catboost_info/                    # Información CatBoost
```

## 🚀 Inicio Rápido

### 1. Entrenamiento Básico
```bash
# Entrenar modelos con configuración por defecto
python crypto_ml_trainer.py

# Entrenar con hiperparámetros optimizados (recomendado)
python crypto_ml_trainer_optimized.py
```

### 2. Optimización de Hiperparámetros
```bash
# Optimización rápida de XGBoost (recomendado para empezar)
python quick_optimization.py --mode quick-xgb --trials 30 --timeout 600

# Optimización completa de todos los modelos
python crypto_hyperparameter_optimizer.py

# Análisis de resultados
python optuna_results_analyzer.py
```

### 3. Tests del Sistema
```bash
# Verificar que todo funciona correctamente
python test_ml_system.py
```

## 🤖 Modelos Implementados

### 1. 🔥 XGBoost
- **Algoritmo**: Extreme Gradient Boosting
- **Fortalezas**: Excelente performance general, manejo de overfitting
- **Optimización**: 9 hiperparámetros optimizables
- **Resultado**: AUC 0.9954 (CV), 0.8100 (Test) después de optimización

### 2. 💡 LightGBM  
- **Algoritmo**: Light Gradient Boosting Machine
- **Fortalezas**: Entrenamiento muy rápido, eficiente en memoria
- **Optimización**: 9 hiperparámetros optimizables
- **Resultado**: AUC 0.6871, entrenamiento con early stopping

### 3. 🐱 CatBoost
- **Algoritmo**: Categorical Boosting
- **Fortalezas**: Manejo automático de variables categóricas
- **Optimización**: 8+ hiperparámetros optimizables
- **Resultado**: AUC 0.7620 (consistentemente el mejor modelo)

### 4. 🎭 Ensemble Voting
- **Método**: Voto mayoritario de los 3 modelos
- **Objetivo**: Maximizar robustez y reducir overfitting
- **Configuración**: Soft voting con probabilidades

## 🔧 Sistema de Optimización (Optuna)

### Características Principales
- ✅ **Optimización automática** de hiperparámetros
- ✅ **Validación temporal** robusta (60/20/20 split)
- ✅ **Cross-validation** 3-fold estratificado
- ✅ **Persistencia completa** (SQLite + JSON + Pickle)
- ✅ **Visualizaciones interactivas** con Plotly
- ✅ **Integración automática** de mejores parámetros

### Espacios de Búsqueda Configurados

#### XGBoost
```python
{
    'n_estimators': 100-1000 (step 50),
    'max_depth': 3-12,
    'learning_rate': 0.01-0.3 (log scale),
    'subsample': 0.6-1.0,
    'colsample_bytree': 0.6-1.0,
    'reg_alpha': 0-10,
    'reg_lambda': 0-10,
    'min_child_weight': 1-10,
    'gamma': 0-5
}
```

#### LightGBM
```python
{
    'n_estimators': 100-1000 (step 50),
    'max_depth': 3-12,
    'learning_rate': 0.01-0.3 (log scale),
    'subsample': 0.6-1.0,
    'colsample_bytree': 0.6-1.0,
    'reg_alpha': 0-10,
    'reg_lambda': 0-10,
    'min_child_samples': 5-100,
    'num_leaves': 10-300
}
```

#### CatBoost
```python
{
    'iterations': 100-1000 (step 50),
    'depth': 3-10,
    'learning_rate': 0.01-0.3 (log scale),
    'subsample': 0.6-1.0,
    'colsample_bylevel': 0.6-1.0,
    'l2_leaf_reg': 1-10,
    'min_data_in_leaf': 1-100,
    'bootstrap_type': ['Bayesian', 'Bernoulli']
}
```

## 📊 Features y Variables

### Variables Objetivo
- **`high_return_30d`**: Retorno > 100% en 30 días (binaria)
- **`future_return_30d`**: Retorno exacto en 30 días (continua)
- **`return_category_30d`**: Categorización de retornos (multi-clase)

### Categorías de Features (76 total)

#### 1. 📈 Técnicas (20+)
- **Retornos**: 1d, 5d, 7d, 14d, 30d
- **Promedios móviles**: SMA 5, 10, 20, 50, 200
- **Bandas Bollinger**: upper, lower, width, position
- **RSI**: Relative Strength Index
- **MACD**: Signal, histogram, crossovers
- **Stochastic**: %K, %D

#### 2. 💰 Volumen (15+)
- **OBV**: On-Balance Volume
- **VWAP**: Volume Weighted Average Price
- **Volume ratios**: 7d, 30d
- **Dollar volume**: SMAs 7d, 30d
- **Volume spikes**: Detección automática

#### 3. 🚀 Momentum (10+)
- **ROC**: Rate of Change (múltiples períodos)
- **Momentum**: 5, 10, 20 períodos
- **Momentum consistency**: Estabilidad direccional
- **Breakouts**: Detección up/down

#### 4. 🎯 Narrativa (10+)
- **Narrative encoding**: Codificación categórica
- **Narrative rank**: Ranking dentro de narrativa
- **Narrative correlation**: Correlación con precio
- **Narrative percentile**: Percentil de performance
- **Relative to narrative**: Performance relativa

#### 5. ⏰ Temporales (15+)
- **Día de semana**: Efectos estacionales
- **Mes**: Tendencias mensuales
- **Trimestre**: Ciclos trimestrales
- **Es fin de semana**: Binaria
- **Es fin de mes**: Binaria
- **Días desde inicio**: Madurez del token

#### 6. 🎪 Soporte/Resistencia (10+)
- **Support/Resistance levels**: Niveles calculados
- **Distance to support/resistance**: Distancias
- **Strength indicators**: Fortaleza de niveles
- **Days since high/low**: Días desde extremos

## 📋 Comandos Principales

### 🔧 Optimización

```bash
# === OPTIMIZACIÓN RÁPIDA (RECOMENDADO) ===
# Solo XGBoost (5-10 min)
python quick_optimization.py --mode quick-xgb --trials 30 --timeout 600

# Solo LightGBM (5-10 min) 
python quick_optimization.py --mode quick-lgb --trials 30 --timeout 600

# Solo CatBoost (5-10 min)
python quick_optimization.py --mode quick-cat --trials 30 --timeout 600

# === OPTIMIZACIÓN ESTÁNDAR ===
# Todos los modelos (30-90 min)
python quick_optimization.py --mode full --trials 50 --timeout 1800

# === OPTIMIZACIÓN EXPERIMENTAL ===
# Búsqueda extensiva (2-6 horas)
python quick_optimization.py --mode experimental --trials 200 --timeout 3600

# === OPTIMIZACIÓN COMPLETA ===
# Sistema completo con todas las capacidades
python crypto_hyperparameter_optimizer.py
```

### 📊 Análisis

```bash
# Análisis completo de resultados
python optuna_results_analyzer.py

# Comparar estudios previos
python quick_optimization.py --mode compare

# Ver visualizaciones generadas
# Se crean en ../../optimization_results/analysis_visualizations/
```

### 🔗 Integración

```bash
# Integrar mejores parámetros encontrados
python integrate_optimized_params.py

# Esto genera:
# - crypto_ml_trainer_optimized.py (trainer actualizado)
# - crypto_ml_trainer_backup_*.py (backup del original)
# - integration_report.md (reporte de cambios)
```

### 🧪 Testing

```bash
# Verificar que todo funciona
python test_ml_system.py

# Ejecutar trainer optimizado
python crypto_ml_trainer_optimized.py
```

## 📈 Interpretación de Resultados

### Métricas Principales
- **CV Score**: Promedio de cross-validation (métrica objetivo)
- **Validation AUC**: Performance en conjunto de validación  
- **Test AUC**: Performance final en conjunto de test
- **Feature Importance**: Importancia de cada característica

### Ranking Automático
El sistema genera rankings automáticos:
1. 🥇 Mejor modelo por AUC
2. 🥈 Segundo mejor
3. 🥉 Tercer mejor

### Visualizaciones Generadas
- **Historia de optimización**: Convergencia por trial
- **Importancia de parámetros**: Qué hiperparámetros importan más
- **Gráficos de contorno**: Interacciones entre parámetros
- **Análisis temporal**: Evolución de experimentos

## 🎯 Detección de Oportunidades

### Método
1. **Ensemble prediction**: Predicciones de todos los modelos
2. **Probability threshold**: Filtro por probabilidad (por defecto >50%)
3. **Ranking**: Ordenamiento por probabilidad descendente
4. **Top-K selection**: Selección de mejores K oportunidades

### Output
```python
# Ejemplo de salida
🎯 Top 10 oportunidades detectadas:
    1. Índice 28531: 0.769 probabilidad
    2. Índice 28532: 0.754 probabilidad
    3. Índice 17529: 0.745 probabilidad
    ...
```

## 💾 Persistencia y Versionado

### Archivos Generados Automáticamente

#### Modelos Entrenados (`../../models/`)
- `xgboost_crypto_ml_[timestamp].model`
- `lightgbm_crypto_ml_[timestamp].txt`
- `catboost_crypto_ml_[timestamp].cbm`
- `feature_importance_[timestamp].json`

#### Resultados Optuna (`../../optimization_results/`)
- `optuna_studies.db` - Base de datos SQLite
- `optimization_summary_[timestamp].json` - Resúmenes
- `evaluation_results_[timestamp].json` - Evaluaciones
- `best_configs_[timestamp].json` - Mejores configuraciones
- `optuna_studies_[timestamp].pkl` - Estudios completos
- `analysis_visualizations/` - Gráficos HTML interactivos

## ⚙️ Configuración Avanzada

### Modificar Espacios de Búsqueda
Editar las funciones `optimize_*` en `crypto_hyperparameter_optimizer.py`:

```python
# Ejemplo: ampliar rango de n_estimators
'n_estimators': trial.suggest_int('n_estimators', 50, 2000, step=50)
```

### Cambiar Métricas de Optimización
```python
# En funciones objective()
scoring='roc_auc'      # Por defecto (AUC-ROC)
scoring='precision'    # Para maximizar precisión
scoring='f1'          # Para F1-score
scoring='recall'      # Para recall
```

### Configurar Timeouts y Trials
```python
# En main() de los scripts
N_TRIALS = 100              # Número de trials
TIMEOUT_PER_MODEL = 3600    # Timeout en segundos
```

## 🚨 Troubleshooting

### Problemas Comunes

#### "No se encontró archivo de datos"
```bash
# Verificar dataset principal
ls ../../data/crypto_ohlc_join.csv
```

#### "Import optuna could not be resolved"
```bash
# Instalar dependencias faltantes
conda activate ML-TF-G
pip install optuna plotly kaleido
```

#### "Optimization timeout reached"
- Normal si se alcanza el timeout configurado
- Los resultados parciales siguen siendo válidos
- Aumentar timeout si es necesario

#### Performance muy baja
- Verificar balance de clases en datos
- Revisar que las features son informativas
- Considerar cambiar métricas de optimización

### Logs y Debug
- Los estudios de Optuna se guardan automáticamente
- Usar `verbose=True` en funciones para más detalles
- Revisar `../../optimization_results/` para históricos

## 📚 Documentación Adicional

- **`README_OPTIMIZATION.md`**: Guía completa de optimización
- **`OPTUNA_IMPLEMENTATION_COMPLETED.md`**: Reporte técnico detallado
- **`integration_report.md`**: Reporte de integración automática
- **`../../README.md`**: Documentación principal del proyecto

## 🎯 Próximos Pasos

### Para Desarrollo
1. Ejecutar optimización rápida: `python quick_optimization.py --mode quick-xgb --trials 20`
2. Analizar resultados: `python optuna_results_analyzer.py`
3. Integrar mejores parámetros: `python integrate_optimized_params.py`

### Para Producción
1. Optimización exhaustiva: `python crypto_hyperparameter_optimizer.py`
2. Validación en datos nuevos
3. Implementar re-entrenamiento periódico
4. Monitorear performance en el tiempo

---

**🤖 Sistema completo de ML para maximizar retornos en criptomonedas de baja capitalización** 🚀💰📈
