"""
Optimizador de parámetros de estrategias usando Grid Search.
"""

import json
import pandas as pd
from sklearn.model_selection import ParameterGrid
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.strategy import signal_generator
from src.backtest.engine import VectorizedBacktester


def run_strategy_backtest(client, params, symbol='BTCUSDT', interval='5m', start_date='1 year ago UTC',
                          df_cached=None, signal_function='generar_señales', use_stop_loss=False):
    """
    Ejecuta un backtest con parámetros específicos.

    Args:
        client: Cliente de Binance
        params: Diccionario con parámetros de estrategia
        symbol: Par de trading
        interval: Intervalo de velas
        start_date: Fecha de inicio
        df_cached: DataFrame pre-calculado con indicadores (opcional)
        signal_function: Nombre de la función de generación de señales (default: 'generar_señales')
        use_stop_loss: Si True, usa backtest con Stop Loss dinámico basado en ATR (default: False)

    Returns:
        Diccionario con resultados
    """
    # Usar DataFrame cacheado si está disponible, sino obtener y calcular
    if df_cached is not None:
        df = df_cached.copy()
    else:
        # Obtener datos
        df = obtener_datos_binance(client, symbol, interval, start_date)

        # Calcular indicadores con parámetros personalizados
        df = agregar_indicadores(df, config=params)

    # Obtener la función de generación de señales dinámicamente
    signal_func = getattr(signal_generator, signal_function)

    # Generar señales usando la función especificada
    df = signal_func(df, config=params)

    # Ejecutar backtest
    backtester = VectorizedBacktester(df, initial_capital=10000)

    if use_stop_loss:
        # Usar backtest con Stop Loss dinámico basado en ATR
        atr_multiplier = params.get('atr_multiplier', 2.0)
        atr_length = params.get('atr_length', 14)
        atr_column = f'ATRr_{atr_length}'

        backtester.run_backtest_with_stop_loss(
            atr_column=atr_column,
            atr_multiplier=atr_multiplier
        )
    else:
        # Usar backtest vectorizado estándar
        backtester.run_backtest()

    metrics = backtester.calculate_metrics()

    # Combinar parámetros con resultados
    result = {**params, **metrics}

    return result


def optimize_parameters(client, param_grid, symbol='BTCUSDT', interval='5m', start_date='1 year ago UTC',
                       signal_function='generar_señales', use_stop_loss=False):
    """
    Optimiza parámetros usando Grid Search.

    Args:
        client: Cliente de Binance
        param_grid: Diccionario con grid de parámetros
        symbol: Par de trading
        interval: Intervalo
        start_date: Fecha de inicio
        signal_function: Nombre de la función de generación de señales (default: 'generar_señales')
        use_stop_loss: Si True, usa backtest con Stop Loss dinámico basado en ATR (default: False)

    Returns:
        DataFrame con todos los resultados ordenados por Sharpe Ratio
    """
    print(f"Optimizando parámetros para {symbol}...")
    print(f"Estrategia: {signal_function}")
    print(f"Stop Loss ATR: {'ACTIVADO' if use_stop_loss else 'DESACTIVADO'}")
    print(f"Total de combinaciones a probar: {len(list(ParameterGrid(param_grid)))}\n")

    # Pre-calcular indicadores una sola vez con TODOS los períodos necesarios
    df_cached = None
    if 'donchian_period' in param_grid and isinstance(param_grid['donchian_period'], list):
        print("Pre-calculando todos los Canales de Donchian...")
        df_cached = obtener_datos_binance(client, symbol, interval, start_date)

        # Crear config con todos los períodos de Donchian
        config_base = {k: v[0] if isinstance(v, list) else v for k, v in param_grid.items()}
        config_base['donchian_period'] = param_grid['donchian_period']  # Pasar lista completa

        df_cached = agregar_indicadores(df_cached, config=config_base)
        print(f"✓ Indicadores pre-calculados\n")

    results = []

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        print(f"[{i}/{len(list(ParameterGrid(param_grid)))}] Probando: {params}")

        try:
            result = run_strategy_backtest(client, params, symbol, interval, start_date,
                                          df_cached=df_cached, signal_function=signal_function,
                                          use_stop_loss=use_stop_loss)
            results.append(result)
            print(f"   Sharpe: {result['sharpe_ratio']:.2f}, Retorno: {result['total_return_pct']:.2f}%\n")
        except Exception as e:
            print(f"   Error: {e}\n")

    # Convertir a DataFrame y ordenar
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)

    return results_df


def save_optimal_params(params, filepath='config/optimal_params.json'):
    """Guarda los parámetros óptimos en un archivo JSON."""
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\n✓ Parámetros óptimos guardados en {filepath}")


def load_optimal_params(filepath='config/optimal_params.json'):
    """Carga los parámetros óptimos desde un archivo JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    from src.data.binance_client import BinanceClientManager

    print("=== Test de Parameter Optimizer ===\n")

    # Crear cliente
    manager = BinanceClientManager()
    client = manager.get_public_client()

    # Definir grid para estrategia de Reversión a la Media (Bollinger Bands + RSI)
    param_grid = {
        'bb_length': [20, 30],
        'bb_std': [2, 2.5],
        'rsi_period': [14, 21],
        'rsi_overbought': [70, 80],
        'rsi_oversold': [20, 30],
        # Parámetros que no se usan en la nueva estrategia pero se incluyen para compatibilidad con indicadores
        'macd_fast': [12],
        'macd_slow': [26],
        'macd_signal': [9],
        'atr_length': [14],
        'stoch_k': [14],
        'stoch_d': [3],
        'stoch_smooth': [3]
    }

    # Optimizar (usa solo 7 días para el test)
    results = optimize_parameters(
        client=client,
        param_grid=param_grid,
        symbol='BTCUSDT',
        interval='5m',
        start_date='7 days ago UTC'
    )

    # Mostrar top 5
    print("\n" + "="*80)
    print("TOP 5 MEJORES CONFIGURACIONES")
    print("="*80)
    print(results.head()[['bb_length', 'bb_std', 'rsi_period', 'rsi_overbought', 'rsi_oversold',
                           'sharpe_ratio', 'total_return_pct', 'max_drawdown_pct', 'win_rate_pct', 'profit_factor']])

    # Guardar mejores parámetros
    best_params = results.iloc[0][['bb_length', 'bb_std', 'rsi_period', 'rsi_overbought', 'rsi_oversold',
                                     'macd_fast', 'macd_slow', 'macd_signal',
                                     'atr_length', 'stoch_k', 'stoch_d', 'stoch_smooth']].to_dict()

    # Convertir a int/float según corresponda
    for key in best_params:
        if key == 'bb_std':
            best_params[key] = float(best_params[key])
        elif isinstance(best_params[key], (float, int)):
            best_params[key] = int(best_params[key])

    save_optimal_params(best_params)

    print("\n✓ Test completado exitosamente")
