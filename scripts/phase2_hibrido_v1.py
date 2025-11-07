#!/usr/bin/env python3
"""
FASE 2: Backtesting y Optimización - ITERACIÓN 12 (MÓDULO HÍBRIDO v1)
Ejecuta backtests con ESTRATEGIA HÍBRIDA DE 4 CAPAS (LONG-ONLY).

ESTRATEGIA DE 4 CAPAS:
- CAPA 1: Filtro de régimen con EMA(200) - Solo opera en tendencia ALCISTA (precio > EMA_200)
- CAPA 2: Filtro de momentum con RSI(14) - Solo entra con momentum ALCISTA (RSI > nivel)
- CAPA 3: Señal de entrada (COMPRA) con cruce ALCISTA MACD
- CAPA 3: Señal de salida (VENTA/TP) con cruce BAJISTA MACD
- CAPA 4: Gestión de riesgo con Stop Loss ATR dinámico

Stop Loss Dinámico ATR:
- SL = entry_price - (ATR × atr_multiplier)
- Verificación: Se cierra si df['low'] toca/cruza SL
- Objetivo: Proteger capital con stops adaptativos a la volatilidad

Activo: ETH/USDT
Timeframe: 15m
Dataset: 1 año completo

Documentación completa: ESTRATEGIA_HIBRIDA_DAY_TRADING.md
"""

import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.strategy.signal_generator import generar_senales_hibrido_v1
from sklearn.model_selection import ParameterGrid


def run_backtest_with_stop_loss(df, initial_capital=10000, commission=0.00075,
                                  slippage=0.0005, atr_multiplier=2.0,
                                  capital_per_trade=15):
    """
    Ejecuta backtest con Stop Loss ATR dinámico.

    Args:
        df: DataFrame con señales y indicadores
        initial_capital: Capital inicial
        commission: Comisión por operación (0.075% = 0.00075)
        slippage: Deslizamiento (0.05% = 0.0005)
        atr_multiplier: Multiplicador del ATR para Stop Loss
        capital_per_trade: Capital fijo por operación

    Returns:
        dict con métricas de rendimiento
    """
    df = df.copy()

    # Buscar columna ATR (puede ser ATRr_14 o ATR_14)
    atr_col = None
    for col in df.columns:
        if 'ATR' in col.upper():
            atr_col = col
            break

    if atr_col is None:
        raise ValueError("No se encontró columna ATR en el DataFrame")

    # Inicializar variables de tracking
    capital = initial_capital
    position = 0  # 0 = sin posición, 1 = en posición LONG
    entry_price = 0
    stop_loss = 0
    trades = []
    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        signal = row['señal']
        price = row['close']
        atr = row[atr_col]

        # Tracking del equity en cada barra
        current_equity = capital
        if position == 1:
            # Si estamos en posición, calcular equity actual
            current_equity = capital + ((price - entry_price) / entry_price) * capital_per_trade

        equity_curve.append({
            'timestamp': row['timestamp'],
            'equity': current_equity,
            'price': price
        })

        # VERIFICAR STOP LOSS (si estamos en posición LONG)
        if position == 1:
            # Para LONG: verificamos si el low del candle tocó el stop loss
            if row['low'] <= stop_loss:
                # Stop Loss tocado - Cerrar posición con pérdida
                exit_price = stop_loss  # Asumimos ejecución exacta en SL

                # Calcular resultado del trade
                pnl_pct = ((exit_price - entry_price) / entry_price)
                pnl_gross = pnl_pct * capital_per_trade
                commission_cost = capital_per_trade * commission * 2  # Entrada + Salida
                slippage_cost = capital_per_trade * slippage * 2
                pnl_net = pnl_gross - commission_cost - slippage_cost

                # Registrar trade
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_net': pnl_net,
                    'exit_reason': 'STOP_LOSS',
                    'holding_periods': i - entry_index
                })

                # Actualizar capital
                capital += pnl_net
                position = 0
                continue  # Siguiente barra

        # GESTIÓN DE SEÑALES
        if signal == 1 and position == 0:
            # SEÑAL DE COMPRA - Abrir posición LONG
            position = 1
            entry_price = price * (1 + slippage)  # Aplicar slippage en entrada
            entry_time = row['timestamp']
            entry_index = i

            # Calcular Stop Loss dinámico basado en ATR
            stop_loss = entry_price - (atr * atr_multiplier)

        elif signal == -1 and position == 1:
            # SEÑAL DE VENTA - Cerrar posición LONG (Take Profit)
            exit_price = price * (1 - slippage)  # Aplicar slippage en salida

            # Calcular resultado del trade
            pnl_pct = ((exit_price - entry_price) / entry_price)
            pnl_gross = pnl_pct * capital_per_trade
            commission_cost = capital_per_trade * commission * 2
            slippage_cost = capital_per_trade * slippage * 2
            pnl_net = pnl_gross - commission_cost - slippage_cost

            # Registrar trade
            trades.append({
                'entry_time': entry_time,
                'exit_time': row['timestamp'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct * 100,
                'pnl_net': pnl_net,
                'exit_reason': 'TAKE_PROFIT',
                'holding_periods': i - entry_index
            })

            # Actualizar capital
            capital += pnl_net
            position = 0

    # Cerrar posición abierta al final (si existe)
    if position == 1:
        exit_price = df.iloc[-1]['close'] * (1 - slippage)
        pnl_pct = ((exit_price - entry_price) / entry_price)
        pnl_gross = pnl_pct * capital_per_trade
        commission_cost = capital_per_trade * commission * 2
        slippage_cost = capital_per_trade * slippage * 2
        pnl_net = pnl_gross - commission_cost - slippage_cost

        trades.append({
            'entry_time': entry_time,
            'exit_time': df.iloc[-1]['timestamp'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_net': pnl_net,
            'exit_reason': 'END_OF_DATA',
            'holding_periods': len(df) - entry_index
        })

        capital += pnl_net

    # Calcular métricas
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'profit_factor': 0,
            'final_capital': initial_capital
        }

    # Convertir trades a DataFrame para análisis
    trades_df = pd.DataFrame(trades)

    # Métricas básicas
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_net'] > 0])
    losing_trades = len(trades_df[trades_df['pnl_net'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # Retorno total
    total_return_pct = ((capital - initial_capital) / initial_capital) * 100

    # Win/Loss promedio
    wins = trades_df[trades_df['pnl_net'] > 0]['pnl_pct']
    losses = trades_df[trades_df['pnl_net'] < 0]['pnl_pct']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    # Profit Factor
    gross_profit = trades_df[trades_df['pnl_net'] > 0]['pnl_net'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_net'] < 0]['pnl_net'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

    # Sharpe Ratio (usando equity curve)
    equity_df = pd.DataFrame(equity_curve)
    equity_df['returns'] = equity_df['equity'].pct_change()
    returns = equity_df['returns'].dropna()

    if len(returns) > 0 and returns.std() != 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 4)  # Anualizado para 15m
    else:
        sharpe_ratio = 0

    # Max Drawdown
    equity_series = equity_df['equity']
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_return_pct': total_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'profit_factor': profit_factor,
        'final_capital': capital,
        'avg_holding_periods': trades_df['holding_periods'].mean(),
        'trades_df': trades_df,
        'equity_df': equity_df
    }


def optimize_hibrido_strategy(client, symbol='ETHUSDT', interval='15m',
                              start_date='1 year ago UTC'):
    """
    Optimiza la estrategia híbrida mediante grid search.

    Args:
        client: Cliente de Binance
        symbol: Par de trading
        interval: Timeframe
        start_date: Fecha de inicio de datos

    Returns:
        tuple (best_params, best_metrics, all_results)
    """
    print("\n📊 OPTIMIZACIÓN DE ESTRATEGIA HÍBRIDA DE 4 CAPAS")
    print("=" * 80)

    # 1. Descargar datos
    print("\n1. Descargando datos históricos...")
    print(f"   Símbolo: {symbol}")
    print(f"   Intervalo: {interval}")
    print(f"   Período: {start_date}")

    df = obtener_datos_binance(
        client=client,
        simbolo=symbol,
        intervalo=interval,
        inicio=start_date
    )
    print(f"   ✓ {len(df)} velas descargadas")
    print(f"   Rango: {df['timestamp'].iloc[0]} a {df['timestamp'].iloc[-1]}")

    # 2. Definir grid de parámetros a optimizar
    print("\n2. Definiendo grid de optimización...")
    param_grid = {
        # CAPA 1: Filtro de Régimen (EMA de Tendencia)
        'ema_trend': [200],  # EMA 200 estándar

        # CAPA 2: Filtro de Momentum (RSI) - PARÁMETRO A OPTIMIZAR
        'rsi_period': [14],                    # RSI estándar
        'rsi_momentum_level': [45, 50, 55],   # Nivel mínimo de RSI para entrar (3 valores)

        # CAPA 3: Señal de Entrada/Salida (MACD)
        'macd_fast': [12],      # MACD rápido (estándar)
        'macd_slow': [26],      # MACD lento (estándar)
        'macd_signal': [9],     # MACD señal (estándar)

        # CAPA 4: Stop Loss ATR (Gestión de Riesgo) - PARÁMETRO A OPTIMIZAR
        'atr_length': [14],                    # ATR estándar
        'atr_multiplier': [1.5, 2.0, 2.5],    # Multiplicador para SL (3 valores)

        # Parámetros auxiliares (para agregar_indicadores)
        'bb_length': [20],
        'bb_std': [2],
        'stoch_k': [14],
        'stoch_d': [3],
        'stoch_smooth': [3],
        'donchian_period': [20]
    }

    total_combinations = len(list(ParameterGrid(param_grid)))
    print(f"   ✓ Total de combinaciones: {total_combinations}")
    print(f"   Parámetros a optimizar:")
    print(f"      - rsi_momentum_level: {param_grid['rsi_momentum_level']}")
    print(f"      - atr_multiplier: {param_grid['atr_multiplier']}")

    # 3. Ejecutar optimización
    print("\n3. Ejecutando optimización (puede tardar varios minutos)...")
    print("   Estrategia: Híbrida 4 Capas LONG-ONLY")
    print("   Stop Loss: Dinámico basado en ATR")
    print("=" * 80)

    all_results = []
    best_sharpe = -999
    best_params = None
    best_metrics = None

    for i, params in enumerate(ParameterGrid(param_grid)):
        print(f"\n   [{i+1}/{total_combinations}] Testing params: "
              f"RSI_level={params['rsi_momentum_level']}, "
              f"ATR_mult={params['atr_multiplier']}")

        # Calcular indicadores
        df_temp = agregar_indicadores(df.copy(), config=params)

        # Generar señales con estrategia híbrida
        df_temp = generar_senales_hibrido_v1(df_temp, config=params)

        # Contar señales
        num_buy_signals = (df_temp['señal'] == 1).sum()
        num_sell_signals = (df_temp['señal'] == -1).sum()

        # Ejecutar backtest con Stop Loss ATR
        try:
            metrics = run_backtest_with_stop_loss(
                df=df_temp,
                initial_capital=10000,
                commission=0.00075,
                slippage=0.0005,
                atr_multiplier=params['atr_multiplier'],
                capital_per_trade=15
            )

            # Almacenar resultados
            result = {
                'params': params.copy(),
                'metrics': metrics,
                'num_buy_signals': num_buy_signals,
                'num_sell_signals': num_sell_signals
            }
            all_results.append(result)

            # Imprimir resumen
            print(f"      Trades: {metrics['total_trades']}, "
                  f"Win Rate: {metrics['win_rate']:.2f}%, "
                  f"Sharpe: {metrics['sharpe_ratio']:.4f}, "
                  f"Return: {metrics['total_return_pct']:.2f}%")

            # Actualizar mejor resultado
            if metrics['sharpe_ratio'] > best_sharpe and metrics['total_trades'] > 10:
                best_sharpe = metrics['sharpe_ratio']
                best_params = params.copy()
                best_metrics = metrics.copy()
                print(f"      ⭐ NUEVO MEJOR RESULTADO (Sharpe: {best_sharpe:.4f})")

        except Exception as e:
            print(f"      ❌ Error en backtest: {e}")
            continue

    print("\n" + "=" * 80)
    print("✅ OPTIMIZACIÓN COMPLETADA")
    print("=" * 80)

    return best_params, best_metrics, all_results


def main():
    print("=" * 80)
    print("FASE 2: BACKTESTING MULTI-TIMEFRAME - ESTRATEGIA HÍBRIDA v1")
    print("Estrategia: 4 Capas LONG-ONLY (EMA_200 + RSI + MACD + ATR)")
    print("Timeframes a probar: 5m, 15m, 1h")
    print("=" * 80)

    # 1. Crear cliente
    print("\n1. Conectando a Binance...")
    manager = BinanceClientManager()
    client = manager.get_public_client()
    print("   ✓ Cliente público creado")

    # 2. Definir timeframes a probar
    timeframes = ['5m', '15m', '1h']
    all_timeframe_results = {}

    print("\n" + "=" * 80)
    print("🔄 INICIANDO BACKTESTS MULTI-TIMEFRAME")
    print("=" * 80)

    # 3. Ejecutar optimización para cada timeframe
    for interval in timeframes:
        print("\n" + "🎯" * 40)
        print(f"TIMEFRAME: {interval.upper()}")
        print("🎯" * 40)

        try:
            best_params, best_metrics, all_results = optimize_hibrido_strategy(
                client=client,
                symbol='ETHUSDT',
                interval=interval,
                start_date='1 year ago UTC'
            )

            if best_params is not None:
                all_timeframe_results[interval] = {
                    'best_params': best_params,
                    'best_metrics': best_metrics,
                    'all_results': all_results
                }
                print(f"\n✅ {interval} completado - Sharpe: {best_metrics['sharpe_ratio']:.4f}, "
                      f"Win Rate: {best_metrics['win_rate']:.2f}%")
            else:
                print(f"\n❌ {interval} - No se encontraron resultados válidos")

        except Exception as e:
            print(f"\n❌ Error en timeframe {interval}: {e}")
            continue

    # 4. Comparar resultados entre timeframes
    if len(all_timeframe_results) == 0:
        print("\n❌ No se obtuvieron resultados válidos en ningún timeframe")
        return

    print("\n\n" + "=" * 80)
    print("📊 COMPARACIÓN DE RESULTADOS ENTRE TIMEFRAMES")
    print("=" * 80)

    # Tabla comparativa
    print(f"\n{'Timeframe':<12} {'Trades':<10} {'Win Rate':<12} {'Sharpe':<10} {'Return %':<12} {'Drawdown %':<12}")
    print("-" * 80)

    for interval in timeframes:
        if interval in all_timeframe_results:
            metrics = all_timeframe_results[interval]['best_metrics']
            print(f"{interval:<12} {metrics['total_trades']:<10} "
                  f"{metrics['win_rate']:<12.2f} {metrics['sharpe_ratio']:<10.4f} "
                  f"{metrics['total_return_pct']:<12.2f} {metrics['max_drawdown_pct']:<12.2f}")

    # 5. Determinar el mejor timeframe
    print("\n" + "=" * 80)
    print("🏆 MEJOR TIMEFRAME (por Sharpe Ratio)")
    print("=" * 80)

    best_timeframe = max(all_timeframe_results.items(),
                         key=lambda x: x[1]['best_metrics']['sharpe_ratio'])

    interval_winner = best_timeframe[0]
    best_params = best_timeframe[1]['best_params']
    best_metrics = best_timeframe[1]['best_metrics']

    print(f"\n🥇 GANADOR: {interval_winner.upper()}")
    print("\nParámetros Óptimos:")
    print(f"  CAPA 1 (Régimen): EMA_trend = {best_params['ema_trend']}")
    print(f"  CAPA 2 (Momentum): RSI_period = {best_params['rsi_period']}, "
          f"RSI_momentum_level = {best_params['rsi_momentum_level']}")
    print(f"  CAPA 3 (Timing): MACD({best_params['macd_fast']}, "
          f"{best_params['macd_slow']}, {best_params['macd_signal']})")
    print(f"  CAPA 4 (Riesgo): ATR_length = {best_params['atr_length']}, "
          f"ATR_multiplier = {best_params['atr_multiplier']}")

    print("\n📈 Métricas de Rendimiento:")
    print(f"  Total de Trades: {best_metrics['total_trades']}")
    print(f"  Trades Ganadores: {best_metrics['winning_trades']}")
    print(f"  Trades Perdedores: {best_metrics['losing_trades']}")
    print(f"  Win Rate: {best_metrics['win_rate']:.2f}%")
    print(f"  Promedio Ganancia: {best_metrics['avg_win_pct']:.2f}%")
    print(f"  Promedio Pérdida: {best_metrics['avg_loss_pct']:.2f}%")
    print(f"  Profit Factor: {best_metrics['profit_factor']:.2f}")
    print(f"\n  Retorno Total: {best_metrics['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {best_metrics['max_drawdown_pct']:.2f}%")
    print(f"  Capital Final: ${best_metrics['final_capital']:.2f}")
    print(f"  Holding Promedio: {best_metrics['avg_holding_periods']:.1f} períodos")

    # 6. Guardar resultados de todos los timeframes
    print("\n" + "=" * 80)
    print("💾 GUARDANDO RESULTADOS")
    print("=" * 80)

    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    os.makedirs(config_dir, exist_ok=True)

    # Guardar comparación completa
    comparison_config = {
        'strategy_name': 'hibrido_v1',
        'strategy_type': 'long_only',
        'description': 'Estrategia Híbrida de 4 Capas - Comparación Multi-Timeframe',
        'best_timeframe': interval_winner,
        'timeframe_results': {}
    }

    for interval, data in all_timeframe_results.items():
        comparison_config['timeframe_results'][interval] = {
            'optimized_params': data['best_params'],
            'performance_metrics': {
                'total_trades': data['best_metrics']['total_trades'],
                'win_rate': data['best_metrics']['win_rate'],
                'sharpe_ratio': data['best_metrics']['sharpe_ratio'],
                'total_return_pct': data['best_metrics']['total_return_pct'],
                'max_drawdown_pct': data['best_metrics']['max_drawdown_pct']
            }
        }

    # Guardar comparación
    comparison_file = os.path.join(config_dir, 'optimal_params_hibrido_v1_multi_timeframe.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison_config, f, indent=2)
    print(f"   ✓ Comparación guardada en: {comparison_file}")

    # Guardar mejor configuración individual
    best_config = {
        'strategy_name': 'hibrido_v1',
        'strategy_type': 'long_only',
        'description': f'Estrategia Híbrida de 4 Capas - Mejor Timeframe: {interval_winner}',
        'optimized_params': best_params,
        'performance_metrics': {
            'total_trades': best_metrics['total_trades'],
            'win_rate': best_metrics['win_rate'],
            'sharpe_ratio': best_metrics['sharpe_ratio'],
            'total_return_pct': best_metrics['total_return_pct'],
            'max_drawdown_pct': best_metrics['max_drawdown_pct']
        },
        'backtest_info': {
            'symbol': 'ETHUSDT',
            'interval': interval_winner,
            'period': '1 year ago UTC'
        }
    }

    best_file = os.path.join(config_dir, 'optimal_params_hibrido_v1.json')
    with open(best_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"   ✓ Mejor configuración guardada en: {best_file}")

    # 7. Mostrar resumen final
    print("\n" + "=" * 80)
    print("✅ BACKTEST MULTI-TIMEFRAME COMPLETADO")
    print("=" * 80)

    print(f"\n📌 RESUMEN:")
    print(f"  • Timeframes probados: {', '.join(timeframes)}")
    print(f"  • Mejor timeframe: {interval_winner.upper()}")
    print(f"  • Mejor Sharpe Ratio: {best_metrics['sharpe_ratio']:.4f}")
    print(f"  • Mejor Win Rate: {best_metrics['win_rate']:.2f}%")

    print("\n📚 Próximos pasos:")
    print("  1. Revisa los resultados en config/optimal_params_hibrido_v1_multi_timeframe.json")
    print(f"  2. La mejor configuración está en config/optimal_params_hibrido_v1.json ({interval_winner})")
    print("  3. Si los resultados son satisfactorios (Win Rate >40%, Sharpe >0.5):")
    print("     - Continúa a Fase 3: Paper Trading (python scripts/phase3_paper.py)")
    print("  4. Si los resultados no son satisfactorios:")
    print("     - Ajusta el grid de parámetros en este script")
    print("     - Considera añadir filtros adicionales (volumen, volatilidad)")
    print("\n  Documentación completa: ESTRATEGIA_HIBRIDA_DAY_TRADING.md")


if __name__ == "__main__":
    main()
