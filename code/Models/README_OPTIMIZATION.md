# 🔧 Sistema de Optimización de Hiperparámetros con Optuna

## 📖 Descripción

Sistema completo de optimización de hiperparámetros para modelos de Machine Learning aplicados a criptomonedas de baja capitalización. Utiliza **Optuna** para encontrar automáticamente los mejores hiperparámetros para XGBoost, LightGBM y CatBoost.

## 🎯 Características

### ✅ Optimización Automática
- **XGBoost**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_weight, gamma
- **LightGBM**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_samples, num_leaves
- **CatBoost**: iterations, depth, learning_rate, subsample, colsample_bylevel, l2_leaf_reg, min_data_in_leaf, bootstrap_type

### ✅ Validación Robusta
- **Split temporal**: 60% entrenamiento, 20% validación, 20% test
- **Cross-validation**: 3-fold estratificado en entrenamiento
- **Métricas**: AUC-ROC como objetivo de optimización

### ✅ Persistencia Completa
- **Base de datos SQLite** para estudios de Optuna
- **JSON** para resúmenes y mejores parámetros
- **Pickle** para objetos completos de estudios
- **Versionado automático** con timestamps

### ✅ Visualizaciones Interactivas
- **Historia de optimización** por modelo
- **Importancia de parámetros** 
- **Gráficos de contorno** para correlaciones
- **Comparación temporal** de experimentos
- **Análisis de sensibilidad** de hiperparámetros

### ✅ Análisis Avanzado
- **Comparación entre modelos** y experimentos
- **Evolución temporal** de performance
- **Correlaciones** entre parámetros y performance
- **Exportación** de mejores configuraciones

## 📁 Archivos del Sistema

```
code/Models/
├── crypto_hyperparameter_optimizer.py    # Optimizador principal
├── quick_optimization.py                 # Scripts de optimización rápida
├── optuna_results_analyzer.py           # Analizador de resultados
└── README_OPTIMIZATION.md               # Esta documentación

optimization_results/                     # Resultados (auto-generados)
├── optuna_studies.db                    # Base de datos SQLite
├── optimization_summary_[timestamp].json # Resúmenes por experimento
├── evaluation_results_[timestamp].json   # Evaluaciones en test
├── best_configs_[timestamp].json        # Mejores configuraciones
└── visualizations/                      # Gráficos interactivos
    ├── xgboost/
    ├── lightgbm/
    └── catboost/
```

## 🚀 Cómo Usar

### 1. Optimización Completa (Recomendado)
```bash
cd /home/exodia/Documentos/MachineLearning_TF/code/Models
conda activate ML-TF-G

# Optimización completa (todos los modelos)
python crypto_hyperparameter_optimizer.py
```

### 2. Optimización Rápida por Modelo
```bash
# Solo XGBoost (30 trials, 10 min)
python quick_optimization.py --mode quick-xgb --trials 30 --timeout 600

# Solo LightGBM (30 trials, 10 min)
python quick_optimization.py --mode quick-lgb --trials 30 --timeout 600

# Solo CatBoost (30 trials, 10 min)
python quick_optimization.py --mode quick-cat --trials 30 --timeout 600
```

### 3. Optimización Personalizada
```bash
# Optimización completa personalizada
python quick_optimization.py --mode full --trials 100 --timeout 3600

# Optimización experimental (más trials y tiempo)
python quick_optimization.py --mode experimental --trials 200 --timeout 7200
```

### 4. Análisis de Resultados
```bash
# Analizar todos los resultados previos
python optuna_results_analyzer.py

# Comparar estudios existentes
python quick_optimization.py --mode compare
```

## ⚙️ Configuración Avanzada

### Modificar Espacios de Búsqueda

Editar `crypto_hyperparameter_optimizer.py` en las funciones `optimize_*`:

```python
# Ejemplo para XGBoost
params = {
    'n_estimators': trial.suggest_int('n_estimators', 50, 2000, step=50),  # Rango ampliado
    'max_depth': trial.suggest_int('max_depth', 2, 15),                    # Más profundidad
    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),  # Rango extendido
    # ... otros parámetros
}
```

### Cambiar Métricas de Optimización

```python
# En las funciones objective(), cambiar:
scoring='roc_auc'  # Por defecto
# A:
scoring='precision'  # Para precisión
scoring='recall'     # Para recall
scoring='f1'         # Para F1-score
```

### Configurar Timeout y Trials

```python
# Configuración en main()
N_TRIALS = 100              # Número de trials por modelo
TIMEOUT_PER_MODEL = 3600    # 1 hora por modelo
```

## 📊 Interpretación de Resultados

### Métricas Principales
- **CV Score**: Promedio de cross-validation (métrica objetivo)
- **Validation AUC**: Performance en conjunto de validación
- **Test AUC**: Performance final en conjunto de test

### Ranking de Modelos
El sistema automáticamente rankea los modelos por performance y muestra:
1. 🥇 Mejor modelo general
2. 🥈 Segundo mejor
3. 🥉 Tercer mejor

### Hiperparámetros Importantes
El análisis identifica automáticamente qué hiperparámetros tienen mayor impacto en la performance mediante correlaciones.

## 🎯 Estrategias de Optimización

### 1. Búsqueda Rápida (30-50 trials)
- **Tiempo**: 10-30 minutos por modelo
- **Objetivo**: Encontrar configuraciones decentes rápidamente
- **Uso**: Desarrollo y pruebas iniciales

### 2. Búsqueda Estándar (100 trials)
- **Tiempo**: 30-60 minutos por modelo  
- **Objetivo**: Buena optimización con tiempo razonable
- **Uso**: Entrenamiento regular de modelos

### 3. Búsqueda Extensiva (200+ trials)
- **Tiempo**: 1-3 horas por modelo
- **Objetivo**: Encontrar los mejores hiperparámetros posibles
- **Uso**: Modelos de producción críticos

### 4. Búsqueda Continua
- **Configurar**: Timeout sin límite de trials
- **Objetivo**: Optimización continua en background
- **Uso**: Servidores dedicados para optimización

## 📈 Visualizaciones Disponibles

### 1. Historia de Optimización
- Evolución del mejor valor encontrado por trial
- Identifica convergencia y plateau

### 2. Importancia de Parámetros
- Ranking de parámetros por impacto en performance
- Ayuda a enfocar futuras optimizaciones

### 3. Gráficos de Contorno
- Relación entre pares de parámetros
- Identifica interacciones entre hiperparámetros

### 4. Comparación Temporal
- Evolución de performance entre experimentos
- Muestra progreso a lo largo del tiempo

### 5. Análisis de Sensibilidad
- Cómo varía la performance con cada parámetro
- Identifica rangos óptimos de valores

## 🔄 Flujo de Trabajo Recomendado

1. **Primera optimización**: `quick-optimization.py --mode full --trials 50`
2. **Analizar resultados**: `optuna_results_analyzer.py`
3. **Identificar modelo prometedor**: Revisar ranking y visualizaciones
4. **Optimización enfocada**: Más trials en el mejor modelo
5. **Validación final**: Entrenar con mejores parámetros y evaluar
6. **Monitoreo continuo**: Ejecutar optimizaciones periódicas

## 🛠️ Troubleshooting

### Error: "No se encontró archivo de datos"
```bash
# Verificar que existe el dataset
ls /home/exodia/Documentos/MachineLearning_TF/data/crypto_ohlc_join.csv
```

### Error: "Import optuna could not be resolved"
```bash
# Instalar optuna en el ambiente
conda activate ML-TF-G
pip install optuna plotly
```

### Warning: "Optimization timeout reached"
- Normal si se alcanza el timeout configurado
- Los resultados parciales siguen siendo válidos
- Aumentar timeout o reducir trials si es necesario

### Performance muy baja
- Verificar que los datos están balanceados
- Revisar que las features son informativas
- Considerar cambiar la métrica de optimización

## 📝 Tips y Mejores Prácticas

### 🎯 Optimización Eficiente
1. **Empezar con pocos trials** para verificar funcionamiento
2. **Usar timeouts** para evitar experimentos infinitos
3. **Monitorear convergencia** en visualizaciones
4. **Ejecutar múltiples experimentos** para verificar reproducibilidad

### 📊 Análisis de Resultados
1. **No confiar solo en CV score** - validar en test set
2. **Revisar importancia de parámetros** antes de re-optimizar
3. **Comparar múltiples experimentos** para identificar tendencias
4. **Guardar configuraciones exitosas** para uso futuro

### 🔧 Configuración de Parámetros
1. **Usar rangos amplios** inicialmente
2. **Afinar rangos** basado en resultados previos
3. **Considerar interacciones** entre parámetros
4. **Validar robustez** con diferentes semillas

## 🎉 Próximos Pasos

Una vez completada la optimización:

1. **Integrar mejores parámetros** en `crypto_ml_trainer.py`
2. **Crear configuración automática** desde archivos JSON
3. **Implementar re-optimización periódica** con nuevos datos
4. **Desarrollar alertas** para caídas de performance
5. **Crear dashboard** para monitoreo continuo

---

**🔧 Sistema desarrollado para maximizar la performance de modelos ML en detección de oportunidades cripto** 🚀
