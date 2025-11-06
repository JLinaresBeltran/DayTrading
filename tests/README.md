# 🧪 Directorio de Tests

Este directorio contiene tests unitarios y de integración para el proyecto.

## 📁 Estructura

```
tests/
├── unit/              # Tests unitarios de módulos individuales
├── integration/       # Tests de integración entre componentes
└── README.md         # Este archivo
```

## 🎯 Tests Unitarios (`unit/`)

Tests para componentes individuales del sistema:
- `test_signal_generator.py` - Tests de generación de señales
- `test_hybrid_strategy.py` - Tests de estrategia híbrida
- `test_bb_columns.py` - Tests de columnas Bollinger Bands

## 🔗 Tests de Integración (`integration/`)

Tests que verifican la interacción entre múltiples módulos (por implementar).

## 🚀 Cómo ejecutar tests

### Ejecutar todos los tests
```bash
python -m pytest tests/
```

### Ejecutar tests unitarios
```bash
python -m pytest tests/unit/
```

### Ejecutar un test específico
```bash
python tests/unit/test_signal_generator.py
```

## 📝 Escribir nuevos tests

Al agregar nuevas funcionalidades, añade tests correspondientes:

1. **Tests unitarios**: Para funciones individuales
   ```python
   def test_generar_senales():
       df = crear_df_prueba()
       resultado = generar_señales(df)
       assert 'señal' in resultado.columns
   ```

2. **Tests de integración**: Para flujos completos
   ```python
   def test_pipeline_completo():
       # Descargar datos -> Calcular indicadores -> Generar señales -> Backtest
       pass
   ```

## ✅ Buenas prácticas

- Usar nombres descriptivos: `test_nombre_de_la_funcionalidad()`
- Testear casos límite (edge cases)
- Usar fixtures para datos de prueba compartidos
- Mantener tests rápidos y enfocados
- Agregar docstrings explicando qué se testea

---

**Última actualización**: 2025-11-06
