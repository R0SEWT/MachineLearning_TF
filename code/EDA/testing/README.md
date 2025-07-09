# 🧪 Sistema de Testing Profesional - EDA Criptomonedas

Sistema completo de testing para validar la funcionalidad y calidad del código EDA de análisis de criptomonedas.

## 📁 Estructura del Sistema

```
testing/
├── __init__.py                 # Paquete principal
├── test_runner.py             # Ejecutor principal de tests
├── test_utils.py              # Utilidades y helpers
├── test_data_analysis.py      # Tests para data_analysis
├── run_tests.py               # Script ejecutor
├── README.md                  # Esta documentación
├── fixtures/                  # Datos de prueba
└── reports/                   # Reportes generados
    └── test_report.json       # Reporte JSON detallado
```

## 🚀 Uso Rápido

### Ejecutar todos los tests:
```bash
# Desde la carpeta EDA:
python testing/run_tests.py

# O desde la carpeta testing:
cd testing
python run_tests.py
```

### Ejecutar tests específicos:
```python
from testing import run_all_tests

# Ejecutar suite completa
result = run_all_tests()
print(f"Éxito: {result['overall_success_rate']:.1f}%")
```

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
