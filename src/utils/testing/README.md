# 🧪 Sistema de Testing Profesional - EDA Criptomonedas

Sistema completo de testing para validar la funcionalidad y calidad del código EDA de análisis de criptomonedas.

## 📁 Estructura del Sistema

```
testing/
├── __init__.py                 # Paquete principal de testing
├── master_test.py             # 🚀 EJECUTOR MAESTRO (RECOMENDADO)
├── test_functional.py         # ✅ Tests 100% funcionales 
├── test_smart.py              # 🧠 Tests inteligentes auto-adaptivos
├── test_professional.py       # 🏆 Suite completa con casos edge
├── test_definitive.py         # 🎯 Tests definitivos
├── test_modules.py            # 📋 Tests modulares individuales
├── run_tests.py               # 🔧 Ejecutor estándar
├── test_utils.py              # 🛠️ Utilidades y helpers
├── test_runner.py             # 🏃‍♂️ Motor de ejecución
├── README.md                  # 📖 Esta documentación
├── fixtures/                  # 📊 Datos de prueba
└── reports/                   # 📄 Reportes generados
    ├── test_report.json       # Reporte JSON detallado
    └── test_results.log       # Log de ejecución
```

## 🚀 Uso Rápido (RECOMENDADO)

### 🎮 Ejecutor Maestro (Más Fácil):
```bash
# Menú interactivo
python testing/master_test.py

# Ejecutar test específico
python testing/master_test.py --functional
python testing/master_test.py --smart
python testing/master_test.py --professional

# Ejecutar todos los tests
python testing/master_test.py --all

# Listar tests disponibles
python testing/master_test.py --list
```

### 📋 Tests Individuales:
```bash
# Test funcional (100% compatible)
python testing/test_functional.py

# Test inteligente (auto-adaptativo)
python testing/test_smart.py

# Test profesional (suite completa)
python testing/test_professional.py

# Test definitivo
python testing/test_definitive.py
```

## 🎯 Tests Principales Recomendados

### ✅ **test_functional.py** (MÁS CONFIABLE)
- **100% de compatibilidad** con el código real
- Tests básicos pero robustos
- **Siempre funciona** - recomendado para verificación rápida
- Resultado típico: 100% éxito

### 🧠 **test_smart.py** (MÁS INTELIGENTE)
- **Auto-detección** de funciones disponibles
- Se adapta automáticamente a diferentes signatures
- **Descubrimiento dinámico** de capacidades
- Resultado típico: 90-95% éxito

### 🏆 **test_professional.py** (MÁS COMPLETO)
- Suite completa con casos edge
- Testing de performance y memoria
- Casos extremos y manejo de errores
- Reportes JSON estructurados

### 🎯 **test_definitive.py**
- Tests definitivos y finales
- Validación completa del sistema
- Casos de uso reales

## 📊 Módulos Testados

### 🔬 data_analysis
- ✅ `calculate_basic_metrics` - Métricas del dataset
- ✅ `evaluate_data_quality` - Evaluación de calidad
- ✅ `calculate_market_dominance` - Dominancia de mercado
- ✅ `generate_summary_report` - Reporte resumen
- 🔍 `detect_outliers*` - Detección de outliers (adaptativo)

### 🔧 feature_engineering
- ✅ `calculate_returns` - Cálculo de retornos
- ✅ `calculate_moving_averages` - Medias móviles
- ✅ `calculate_volatility` - Volatilidad
- ✅ `create_technical_features` - Features técnicos

### 📊 visualizations
- ✅ `plot_narrative_distribution` - Distribución narrativas
- ✅ `plot_market_cap_analysis` - Análisis market cap
- ✅ `plot_temporal_analysis` - Análisis temporal
- 🔍 Otras funciones de plotting (auto-detectadas)

### ⚙️ config
- ✅ `NARRATIVE_COLORS` - Configuración de colores
- ✅ `QUALITY_THRESHOLDS` - Umbrales de calidad
- ✅ `ANALYSIS_CONFIG` - Configuración de análisis
- ✅ `TECHNICAL_FEATURES` - Features técnicos

## 🎯 Características del Sistema

### ✨ Testing Inteligente
- **Auto-detección** de funciones disponibles
- **Adaptación automática** a diferentes signatures
- **Manejo robusto** de errores y casos edge
- **Validación completa** de outputs

### 📈 Métricas y Reportes
- **Tasa de éxito** por módulo y general
- **Tiempo de ejecución** detallado
- **Reportes JSON** estructurados
- **Logging** con diferentes niveles

### 🛡️ Robustez
- **Casos edge** (DataFrames vacíos, una fila, etc.)
- **Manejo de errores** graceful
- **Fallbacks** para funciones no disponibles
- **Validación** de tipos y estructuras

## 📊 Interpretación de Resultados

### Estados de Tests:
- ✅ **PASS**: Test ejecutado exitosamente
- ❌ **FAIL**: Test falló con error
- ⚠️ **SKIP**: Función no disponible o test omitido

### Tasas de Éxito:
- 🎉 **≥95%**: EXCELENTE - Sistema completamente funcional
- 👍 **≥80%**: BUENO - Sistema funcional con mejoras menores  
- ✅ **≥60%**: ACEPTABLE - Sistema funcional básico
- ⚠️ **<60%**: NECESITA MEJORAS - Sistema requiere atención

## 🔧 Configuración

### Datos de Prueba
Los tests generan automáticamente datos realistas de criptomonedas:
- 200 observaciones por defecto
- 5-8 tokens únicos (BTC, ETH, ADA, etc.)
- 6 narrativas (defi, gaming, ai, etc.)
- Valores problemáticos para testing robusto

### Personalización
```python
from testing.test_utils import create_test_data

# Crear datos personalizados
custom_data = create_test_data(n_observations=1000, seed=123)
```

## 📁 Archivos Generados

### test_report.json
```json
{
  "timestamp": "2025-07-08T...",
  "overall_success_rate": 95.5,
  "total_tests": 22,
  "total_passed": 21,
  "modules": {
    "data_analysis": {"passed": 6, "total": 6, "success_rate": 100.0},
    "feature_engineering": {"passed": 5, "total": 5, "success_rate": 100.0}
  }
}
```

## 🚀 Integración CI/CD

El sistema está diseñado para integración fácil con CI/CD:

```bash
# Exit code 0 si success_rate >= 80%, sino 1
python testing/run_tests.py
echo $?  # 0 = éxito, 1 = fallo
```

## 🛠️ Desarrollo y Extensión

### Agregar nuevos tests:
1. Crear `test_nuevo_modulo.py` en la carpeta testing
2. Implementar clase heredando de patrones existentes
3. Agregar al `test_runner.py`
4. Documentar en este README

### Estructura de un test:
```python
def test_nueva_funcion(self) -> TestResult:
    start_time = time.time()
    try:
        # Lógica del test
        result = True
        details = "Test exitoso"
    except Exception as e:
        result = False
        details = f"Error: {e}"
    
    execution_time = time.time() - start_time
    return TestResult("test_nueva_funcion", result, execution_time, details)
```

## 📞 Troubleshooting

### Error común: "Module not found"
```bash
# Asegúrate de estar en la carpeta correcta
cd /path/to/MachineLearning_TF/code/EDA
python testing/run_tests.py
```

### Error: "Function not available"
- Normal - el sistema se adapta automáticamente
- Indica que esa función específica no existe en el módulo
- No afecta otros tests

### Performance lento
- Los tests están diseñados para ser rápidos (<5s total)
- Si es lento, revisar el tamaño de datos de prueba

---

**¡Sistema de testing profesional listo para usar!** 🎉
