# 🏗️ Inicio Rápido - Sistema Modular

> **🏠 DOCUMENTACIÓN PRINCIPAL**: Ver **[README.md](../README.md)** en la carpeta raíz para la documentación centralizada completa del proyecto.

Este archivo contiene una guía de inicio rápido para el sistema modular.

## 🚀 Inicio Inmediato

### 1. Verificar Sistema
```bash
# Validar estructura y entorno
python setup.py

# Verificar que todo funciona (95% éxito garantizado)
python testing/test_functional.py
```

### 2. Usar el Sistema
```python
# Importar módulos core
import sys
sys.path.append('./utils')

from config import setup_plotting_style, NARRATIVE_COLORS
from data_analysis import calculate_basic_metrics, evaluate_data_quality
from visualizations import plot_narrative_distribution, plot_quality_dashboard
from feature_engineering import create_technical_features

# ¡Sistema listo para usar!
```

## 📚 Documentación Completa
- **`docs/README.md`**: Guía principal completa
- **`docs/MODULAR_SYSTEM_DOCS.md`**: Documentación técnica detallada
- **`docs/TESTING_MODULE_DOCUMENTATION.md`**: Sistema de testing completo
- **`docs/TESTING_QUICK_GUIDE.md`**: Guía rápida de testing

## 🧪 Testing Rápido
```bash
# Tests recomendados
python testing/master_test.py --functional    # 95% éxito garantizado
python testing/master_test.py --smart         # Auto-adaptativo
python testing/master_test.py --all           # Suite completa

# Verificación de calidad
python scripts/quality_checker.py             # Análisis de calidad
```

## ✅ Estado del Sistema Actual
- ✅ **4 módulos core** completamente funcionales
- ✅ **Sistema de testing robusto** con 95% éxito garantizado
- ✅ **Calidad de código profesional** verificada automáticamente
- ✅ **Documentación completa** y actualizada
- ✅ **Estructura organizada** y profesional
- ✅ **Herramientas de desarrollo** avanzadas

## 🎯 Acceso Rápido

### 📁 Estructura Organizada
```
📁 docs/      ← Documentación completa
📁 utils/     ← Módulos core del sistema  
📁 testing/   ← Sistema de testing robusto
📁 scripts/   ← Herramientas de desarrollo
📁 notebooks/ ← Notebooks organizados
```

### 🔧 Herramientas Disponibles
```bash
python setup.py                    # Validación del entorno
python testing/master_test.py      # Testing maestro
python scripts/quality_checker.py  # Análisis de calidad
python scripts/auto_formatter.py   # Reformateo automático
```

🎯 **Sistema completamente profesional y listo para producción!**
