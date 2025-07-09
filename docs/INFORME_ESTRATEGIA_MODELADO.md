# 📊 INFORME: ESTRATEGIA DE MODELADO PARA CRIPTOMONEDAS DE BAJA CAPITALIZACIÓN

## 🎯 RESUMEN EJECUTIVO

Basándome en el análisis exploratorio de datos (EDA) realizado y los objetivos del proyecto, este informe define la estrategia completa de modelado para **identificar criptomonedas de baja capitalización con alto potencial de valorización** en los próximos meses.

**Objetivo Principal**: Desarrollar un sistema de ML que identifique tokens con market cap < $10M que puedan generar retornos superiores al 100% en 30 días.

---

## 📋 ANÁLISIS DEL DATASET ACTUAL

### 📊 **Características de los Datos**
- **55,685 observaciones** de datos históricos OHLC
- **Período**: Julio 2024 - Julio 2025 (12 meses de datos)
- **Variables disponibles**: close, date, id, cmc_id, market_cap, name, narrative, price, symbol, volume
- **Tokens únicos**: ~200+ criptomonedas diferentes
- **Narrativas**: meme, ai, gaming, rwa, infrastructure, defi

### 🎯 **Segmentación por Market Cap**
```
Baja Capitalización (< $10M):    ~70% de tokens
Media Capitalización ($10-100M): ~20% de tokens  
Alta Capitalización (> $100M):   ~10% de tokens
```

### 📈 **Distribución por Narrativas** (Basado en EDA)
```
meme           ~40%    # Mayor volatilidad y potencial explosivo
ai             ~25%    # Narrativa en crecimiento, fundamentals sólidos
gaming         ~15%    # Volatilidad media, dependiente de adopción
rwa            ~10%    # Menor volatilidad, crecimiento sostenido
infrastructure ~5%     # Tokens de utilidad, crecimiento a largo plazo
defi           ~5%     # Tokens DeFi diversos
```

---

## 🎯 DEFINICIÓN DEL PROBLEMA DE MODELADO

### 📌 **Problema Principal**
**Predicción de Retornos a 30 días**: Clasificar tokens que generarán retornos > 100% en 30 días

### 🎯 **Variables Objetivo Propuestas**

#### 1. **Clasificación Binaria** (Modelo Principal)
```python
target_binary = "high_return_30d"  # 1 si retorno > 100% en 30 días, 0 si no
```

#### 2. **Clasificación Multi-clase** (Modelo Secundario)
```python
target_multiclass = "return_category_30d"
# Categorías:
# 0: "PÉRDIDA" (retorno < -20%)
# 1: "ESTABLE" (retorno entre -20% y +50%)  
# 2: "GANANCIA_MEDIA" (retorno entre +50% y +100%)
# 3: "GANANCIA_ALTA" (retorno > +100%)
```

#### 3. **Regresión** (Modelo de Apoyo)
```python
target_regression = "future_return_30d"  # Retorno exacto en % a 30 días
```

### 🎯 **Métricas de Éxito**
- **Precisión**: >80% en identificar tokens con retorno >100%
- **Recall**: >70% para no perder oportunidades importantes
- **ROI Simulado**: >300% anual en backtesting
- **Sharpe Ratio**: >2.0 en estrategia de trading

---

## 🔧 INGENIERÍA DE CARACTERÍSTICAS

### 📊 **Features Financieras** (Basadas en EDA)

#### 1. **Indicadores Técnicos**
```python
# Momentum
- RSI (14, 30 días)
- MACD y señal
- Stochastic Oscillator
- Williams %R

# Tendencia  
- SMA (5, 10, 20, 50 días)
- EMA (12, 26 días)
- Bollinger Bands (20 días)
- ADX (Dirección de tendencia)

# Volatilidad
- ATR (Average True Range)
- Volatilidad histórica (7, 14, 30 días)
- Bollinger Band Width
- Volatilidad vs media móvil
```

#### 2. **Features de Volumen y Liquidez**
```python
# Volumen
- Volumen promedio (7, 14, 30 días)
- Ratio volumen actual / promedio
- OBV (On Balance Volume)
- VWAP (Volume Weighted Average Price)

# Liquidez
- Bid-ask spread estimado
- Market depth proxy
- Volume Rate of Change
- Accumulation/Distribution Line
```

#### 3. **Features de Market Cap y Valoración**
```python
# Capitalización
- Market cap actual
- Ratio market cap / volumen
- Percentil de market cap en narrativa
- Días desde market cap mínimo/máximo

# Comparativa con narrativa
- Performance vs promedio de narrativa
- Ranking dentro de narrativa
- Correlación con líderes de narrativa
```

### 🧠 **Features de Narrativa y Sentimiento**

#### 1. **Features de Narrativa**
```python
# Clasificación por narrativa
- narrative_encoded (one-hot encoding)
- narrative_momentum (performance promedio narrativa)
- narrative_volatility (volatilidad promedio narrativa)
- narrative_market_share

# Timing de narrativa
- dias_desde_listing
- posicion_en_narrativa_ranking
- correlation_with_narrative_leaders
```

#### 2. **Features de Timing y Estacionalidad**
```python
# Temporales
- day_of_week
- day_of_month  
- month
- quarter
- is_weekend
- is_month_end

# Ciclos cripto
- days_since_bitcoin_ath
- bitcoin_dominance_trend
- altcoin_season_indicator
```

### 📈 **Features de Momentum y Patrones**

#### 1. **Momentum Multi-período**
```python
# Retornos históricos
- return_1d, return_3d, return_7d, return_14d
- return_volatility_7d, return_volatility_30d
- max_return_7d, min_return_7d
- recovery_time_from_drawdown

# Aceleración
- price_acceleration_3d
- volume_acceleration_7d
- momentum_consistency_score
```

#### 2. **Detección de Patrones**
```python
# Patrones técnicos
- breakout_from_resistance
- support_level_strength  
- consolidation_period_length
- trend_strength_score

# Anomalías
- unusual_volume_spike
- price_gap_detection
- volatility_breakout
- correlation_breakdown
```

---

## 🤖 ARQUITECTURA DE MODELOS

### 🏗️ **Sistema de Modelos en Cascada**

#### 1. **Filtro Inicial - Modelo de Screening**
```python
# Objetivo: Filtrar tokens candidatos
# Input: Todos los tokens de baja cap
# Output: Top 20% de tokens prometedores
# Algoritmo: XGBoost optimizado para recall alto
# Métricas: Recall > 90%, Precisión > 30%
```

#### 2. **Clasificador Principal - Modelo de Predicción**
```python
# Objetivo: Predecir retornos altos (>100%)
# Input: Tokens filtrados del paso 1
# Output: Probabilidad de retorno >100% en 30 días
# Algoritmo: Ensemble (XGBoost + CatBoost + Neural Network)
# Métricas: Precisión > 80%, F1-Score > 0.75
```

#### 3. **Modelo de Regresión - Estimación de Retorno**
```python
# Objetivo: Estimar retorno exacto esperado
# Input: Tokens clasificados como "prometedores"
# Output: Retorno esperado en % (0-500%)
# Algoritmo: LightGBM con regularización
# Métricas: RMSE < 50%, R² > 0.6
```

#### 4. **Modelo de Riesgo - Evaluación de Downside**
```python
# Objetivo: Estimar riesgo de pérdida severa
# Input: Tokens prometedores
# Output: Probabilidad de pérdida > 50%
# Algoritmo: Isolation Forest + XGBoost
# Métricas: Precisión en detectar pérdidas > 85%
```

### 🧠 **Algoritmos Específicos Recomendados**

#### 1. **XGBoost** (Modelo Principal)
```python
# Ventajas para cripto:
- Excelente con features numéricas y categóricas mixtas
- Robusto a outliers (común en cripto)
- Feature importance interpretable
- Optimización con Optuna ya implementada

# Configuración recomendada:
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

#### 2. **CatBoost** (Modelo de Narrativas)
```python
# Ventajas para nuestro caso:
- Manejo nativo de features categóricas (narrativas)
- Menos overfitting en datos pequeños
- Excelente con datos temporales

# Configuración recomendada:
catboost_params = {
    'objective': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'cat_features': ['narrative', 'cluster_id']
}
```

#### 3. **Red Neuronal** (Modelo de Patrones Complejos)
```python
# Arquitectura recomendada:
# Input: 50-100 features normalizadas
# Hidden: 3 capas (128, 64, 32 neuronas)
# Output: 1 neurona (sigmoid para probabilidad)
# Dropout: 0.3 en cada capa
# Regularización: L1/L2 combinada
```

---

## 📊 ESTRATEGIA DE VALIDACIÓN

### ⏰ **Validación Temporal** (Crucial para cripto)

#### 1. **Walk-Forward Validation**
```python
# División temporal estricta:
# Entrenamiento: Mes 1-8 
# Validación: Mes 9-10
# Test: Mes 11-12

# Sin data leakage futuro
# Validación realista de trading
```

#### 2. **Backtesting Financiero**
```python
# Simulación de trading:
# Capital inicial: $10,000
# Max posiciones simultáneas: 5
# Stop-loss: -30%
# Take-profit: +150%
# Comisiones: 0.1% por transacción

# Métricas de trading:
# - Sharpe Ratio > 2.0
# - Maximum Drawdown < 30%
# - Win Rate > 60%
# - Average Return per Trade > 25%
```

### 🎯 **Métricas de Evaluación**

#### 1. **Métricas de Clasificación**
```python
# Precisión: % de predicciones correctas de "alta ganancia"
# Recall: % de oportunidades reales capturadas
# F1-Score: Balance entre precisión y recall
# AUC-ROC: Capacidad de ranking de oportunidades
```

#### 2. **Métricas Financieras**
```python
# ROI Total: Retorno total de la estrategia
# Sharpe Ratio: Retorno ajustado por riesgo
# Calmar Ratio: Retorno / Maximum Drawdown
# Hit Rate: % de trades ganadores
# Profit Factor: Ganancias totales / Pérdidas totales
```

---

## 🔍 DETECCIÓN DE OPORTUNIDADES

### 🚨 **Sistema de Alertas en Tiempo Real**

#### 1. **Trigger de Oportunidades**
```python
# Condiciones para alerta:
oportunidad_detectada = (
    modelo_probabilidad > 0.75 AND
    riesgo_estimado < 0.30 AND
    market_cap < 10_000_000 AND
    volumen_24h > promedio_7d * 2 AND
    RSI < 70  # No sobrecomprado
)
```

#### 2. **Scoring de Oportunidades**
```python
# Score compuesto (0-100):
opportunity_score = (
    probabilidad_ganancia * 40 +      # 40% peso
    retorno_esperado/100 * 30 +       # 30% peso  
    (1 - riesgo_estimado) * 20 +      # 20% peso
    momentum_score * 10               # 10% peso
)
```

### 📈 **Estrategia de Portfolio**

#### 1. **Diversificación por Narrativa**
```python
# Distribución recomendada:
portfolio_allocation = {
    'meme': 0.40,        # Alta volatilidad, alto retorno
    'ai': 0.30,          # Fundamentals sólidos
    'gaming': 0.15,      # Adopción creciente
    'rwa': 0.10,         # Estabilidad relativa
    'defi': 0.05         # Utilidad establecida
}
```

#### 2. **Gestión de Riesgo**
```python
# Reglas de trading:
max_position_size = 0.20    # 20% del capital por token
max_narrative_exposure = 0.50  # 50% en una narrativa
stop_loss = -0.30           # -30% stop loss
take_profit = 1.50          # +150% take profit
rebalance_frequency = "weekly"
```

---

## 🛠️ IMPLEMENTACIÓN TÉCNICA

### 📋 **Pipeline de Datos**

#### 1. **Preparación de Features** 
```python
# code/EDA/utils/feature_engineering.py
def create_ml_features(df):
    """Crear todas las features para ML"""
    
    # Features técnicas
    df = add_technical_indicators(df)
    df = add_volume_features(df)
    df = add_momentum_features(df)
    
    # Features de narrativa
    df = add_narrative_features(df)
    df = add_timing_features(df)
    
    # Target variables
    df = create_target_variables(df)
    
    return df

def create_target_variables(df):
    """Crear variables objetivo"""
    # Calcular retorno a 30 días
    df['future_return_30d'] = df.groupby('id')['close'].pct_change(periods=30).shift(-30)
    
    # Clasificación binaria: retorno > 100%
    df['high_return_30d'] = (df['future_return_30d'] > 1.0).astype(int)
    
    # Clasificación multi-clase
    conditions = [
        df['future_return_30d'] < -0.2,
        (df['future_return_30d'] >= -0.2) & (df['future_return_30d'] < 0.5),
        (df['future_return_30d'] >= 0.5) & (df['future_return_30d'] < 1.0),
        df['future_return_30d'] >= 1.0
    ]
    df['return_category_30d'] = np.select(conditions, [0, 1, 2, 3], default=1)
    
    return df
```

#### 2. **Entrenamiento de Modelos**
```python
# code/Models/model_training_enhanced.py
def train_ensemble_model(X_train, y_train, X_val, y_val):
    """Entrenar ensemble de modelos"""
    
    # Modelo 1: XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Modelo 2: CatBoost  
    cat_model = train_catboost(X_train, y_train, X_val, y_val)
    
    # Modelo 3: Neural Network
    nn_model = train_neural_network(X_train, y_train, X_val, y_val)
    
    # Ensemble con voting
    ensemble = VotingClassifier([
        ('xgb', xgb_model),
        ('cat', cat_model), 
        ('nn', nn_model)
    ], voting='soft')
    
    ensemble.fit(X_train, y_train)
    return ensemble
```

#### 3. **Sistema de Predicción**
```python
# code/Models/prediction_system.py
def predict_opportunities(df_current):
    """Predecir oportunidades actuales"""
    
    # Preparar features
    features = create_ml_features(df_current)
    
    # Filtrar baja capitalización
    low_cap = features[features['market_cap'] < 10_000_000]
    
    # Predicciones del ensemble
    probabilities = ensemble_model.predict_proba(low_cap)[:, 1]
    expected_returns = regression_model.predict(low_cap)
    risk_scores = risk_model.predict(low_cap)
    
    # Crear ranking de oportunidades
    opportunities = pd.DataFrame({
        'token': low_cap['id'],
        'narrative': low_cap['narrative'],
        'probability': probabilities,
        'expected_return': expected_returns,
        'risk_score': risk_scores,
        'opportunity_score': calculate_opportunity_score(
            probabilities, expected_returns, risk_scores
        )
    })
    
    return opportunities.sort_values('opportunity_score', ascending=False)
```

### 🚀 **Sistema de Alertas**
```python
# code/Models/alert_system.py
def check_opportunities():
    """Verificar nuevas oportunidades"""
    
    # Obtener datos actuales
    current_data = get_latest_data()
    
    # Generar predicciones
    opportunities = predict_opportunities(current_data)
    
    # Filtrar alertas de alta calidad
    high_quality = opportunities[
        (opportunities['probability'] > 0.75) &
        (opportunities['risk_score'] < 0.30) &
        (opportunities['opportunity_score'] > 80)
    ]
    
    # Enviar alertas
    for _, opp in high_quality.iterrows():
        send_alert(opp)
    
    return high_quality
```

---

## 📊 CRONOGRAMA DE IMPLEMENTACIÓN

### 🗓️ **Fase 1: Preparación de Datos** (1-2 semanas)
- [ ] Completar feature engineering avanzado
- [ ] Crear variables objetivo (retornos a 30 días)
- [ ] Implementar validación temporal
- [ ] Crear dataset de entrenamiento limpio

### 🗓️ **Fase 2: Desarrollo de Modelos** (2-3 semanas)  
- [ ] Implementar baseline models (XGBoost, CatBoost)
- [ ] Optimizar hiperparámetros con Optuna
- [ ] Desarrollar ensemble de modelos
- [ ] Crear modelo de estimación de riesgo

### 🗓️ **Fase 3: Validación y Backtesting** (1-2 semanas)
- [ ] Implementar walk-forward validation
- [ ] Ejecutar backtesting financiero completo
- [ ] Validar métricas de trading
- [ ] Ajustar estrategia de portfolio

### 🗓️ **Fase 4: Sistema de Producción** (1-2 semanas)
- [ ] Desarrollar pipeline de predicción automatizado
- [ ] Implementar sistema de alertas
- [ ] Crear dashboard de monitoreo
- [ ] Testing en entorno de producción

---

## 🎯 MÉTRICAS DE ÉXITO DEL PROYECTO

### 📊 **KPIs Principales**
```
✅ Precisión en predicción de retornos >100%: Meta > 80%
✅ Recall de oportunidades reales: Meta > 70%  
✅ ROI anual simulado: Meta > 300%
✅ Sharpe Ratio de estrategia: Meta > 2.0
✅ Maximum Drawdown: Meta < 30%
✅ Win Rate en trades: Meta > 60%
```

### 📈 **Métricas Secundarias**
```
• Tiempo promedio de detección de oportunidad: < 24 horas
• Número de oportunidades detectadas por mes: 10-20
• Precisión en estimación de retorno exacto: ±30%
• Falsos positivos por semana: < 5
• Disponibilidad del sistema: > 99%
```

---

## 🚨 RIESGOS Y MITIGACIONES

### ⚠️ **Riesgos Técnicos**
1. **Overfitting a datos históricos**
   - Mitigación: Validación temporal estricta, regularización agresiva

2. **Data leakage futuro**
   - Mitigación: Pipeline de features sin lookahead bias

3. **Cambios en dinámicas del mercado**
   - Mitigación: Reentrenamiento mensual, monitoreo de deriva

### ⚠️ **Riesgos Financieros**
1. **Volatilidad extrema en cripto**
   - Mitigación: Stop-loss automático, diversificación

2. **Liquidez limitada en low-cap**
   - Mitigación: Filtros de volumen mínimo, posiciones pequeñas

3. **Manipulación de mercado**
   - Mitigación: Detección de anomalías, filtros de calidad

---

## 🏆 CONCLUSIONES Y RECOMENDACIONES

### 🎯 **Estrategia Recomendada**

1. **Enfoque en clasificación binaria** para identificar tokens con potencial >100% retorno
2. **Sistema ensemble** combinando XGBoost, CatBoost y redes neuronales
3. **Validación temporal estricta** para evitar overfitting
4. **Backtesting financiero** como métrica principal de éxito
5. **Diversificación por narrativa** para gestión de riesgo
6. **Sistema de alertas automatizado** para ejecución en tiempo real

### 🚀 **Ventajas Competitivas**

- **Datos exclusivos**: Acceso a datos de múltiples narrativas
- **Features avanzadas**: Indicadores técnicos y de sentimiento
- **Validación robusta**: Backtesting financiero realista
- **Sistema completo**: Desde detección hasta ejecución
- **Gestión de riesgo**: Protección contra downside severo

### 📊 **Expectativas Realistas**

```
Escenario Conservador:  ROI anual 150-200%
Escenario Optimista:    ROI anual 300-500%  
Escenario Pesimista:    ROI anual 50-100%
```

**El éxito del sistema dependerá de la calidad de la implementación, la disciplina en la ejecución y la adaptación continua a las condiciones cambiantes del mercado cripto.**

---

**📅 Fecha del Informe**: Enero 2025  
**🎯 Estado**: Listo para Implementación  
**👥 Equipo**: ML-TF-G  
**📊 Próximo Paso**: Iniciar Fase 1 - Preparación de Datos
