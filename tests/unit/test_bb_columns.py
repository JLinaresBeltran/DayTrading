import pandas as pd
import pandas_ta as ta
import numpy as np

# Crear datos de prueba
np.random.seed(42)
df = pd.DataFrame({
    'open': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 101,
    'low': np.random.randn(100).cumsum() + 99,
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 100)
})

# Probar con diferentes configuraciones de Bollinger Bands
configs = [
    {'length': 20, 'std': 2},
    {'length': 20, 'std': 2.5},
    {'length': 30, 'std': 2},
    {'length': 30, 'std': 2.5}
]

for config in configs:
    df_test = df.copy()
    df_test.ta.bbands(length=config['length'], std=config['std'], append=True)
    
    # Filtrar solo columnas de BB
    bb_cols = [col for col in df_test.columns if col.startswith('BB')]
    
    print(f"\nConfig: length={config['length']}, std={config['std']}")
    print(f"Columnas generadas: {bb_cols}")
