"""
Módulo para calcular métricas de rendimiento de estrategias de trading.
Incluye Sharpe Ratio, Max Drawdown, Win Rate, etc.
"""

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252):
    """
    Calcula el Ratio de Sharpe anualizado.

    Args:
        returns: Serie de retornos (pandas Series)
        risk_free_rate: Tasa libre de riesgo anual (default: 0)
        periods: Períodos por año (252 para diario, 252*78 para 5min en horas de trading)

    Returns:
        Sharpe Ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods)
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods)

    return round(sharpe, 2)


def calculate_max_drawdown(cumulative_returns):
    """
    Calcula el Máximo Drawdown (máxima caída desde el pico más alto).

    Args:
        cumulative_returns: Serie de retornos acumulados (pandas Series)

    Returns:
        Max Drawdown como decimal (ej. 0.15 = 15%)
    """
    if len(cumulative_returns) == 0:
        return 0.0

    # Calcular el valor del portafolio
    wealth_index = 1000 * (1 + cumulative_returns)

    # Calcular picos históricos
    previous_peaks = wealth_index.cummax()

    # Calcular drawdown
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    # Máximo drawdown (valor negativo)
    max_dd = drawdowns.min()

    return round(abs(max_dd), 4)


def calculate_win_rate(trades):
    """
    Calcula el porcentaje de operaciones ganadoras.

    Args:
        trades: Serie o lista de PnL de cada operación

    Returns:
        Win rate como decimal (ej. 0.65 = 65%)
    """
    if len(trades) == 0:
        return 0.0

    winning_trades = sum(1 for trade in trades if trade > 0)
    total_trades = len(trades)

    win_rate = winning_trades / total_trades

    return round(win_rate, 4)


def calculate_profit_factor(trades):
    """
    Calcula el Profit Factor (ganancias totales / pérdidas totales).

    Args:
        trades: Serie o lista de PnL de cada operación

    Returns:
        Profit Factor
    """
    if len(trades) == 0:
        return 0.0

    gross_profit = sum(trade for trade in trades if trade > 0)
    gross_loss = abs(sum(trade for trade in trades if trade < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    profit_factor = gross_profit / gross_loss

    return round(profit_factor, 2)


def calculate_sortino_ratio(returns, target_return=0.0, periods=252):
    """
    Calcula el Ratio de Sortino (similar a Sharpe pero solo penaliza volatilidad negativa).

    Args:
        returns: Serie de retornos
        target_return: Retorno objetivo
        periods: Períodos por año

    Returns:
        Sortino Ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods)

    return round(sortino, 2)


def calculate_calmar_ratio(cumulative_returns, annual_return):
    """
    Calcula el Ratio de Calmar (retorno anual / máximo drawdown).

    Args:
        cumulative_returns: Serie de retornos acumulados
        annual_return: Retorno anualizado

    Returns:
        Calmar Ratio
    """
    max_dd = calculate_max_drawdown(cumulative_returns)

    if max_dd == 0:
        return 0.0

    calmar = annual_return / max_dd

    return round(calmar, 2)


def calculate_all_metrics(df, initial_capital=10000):
    """
    Calcula todas las métricas de rendimiento de una estrategia.

    Args:
        df: DataFrame con columnas 'returns', 'strategy_returns', 'cumulative_returns'
        initial_capital: Capital inicial

    Returns:
        Diccionario con todas las métricas
    """
    # Retornos
    returns = df['strategy_returns'].dropna()
    cumulative_returns = df['cumulative_returns'].dropna()

    # Valor final del portafolio
    final_value = initial_capital * (1 + cumulative_returns.iloc[-1])
    net_profit = final_value - initial_capital
    total_return = (final_value / initial_capital) - 1

    # Sharpe Ratio
    sharpe = calculate_sharpe_ratio(returns)

    # Max Drawdown
    max_dd = calculate_max_drawdown(cumulative_returns)

    # Identificar trades (cuando cambia la señal)
    if 'señal' in df.columns:
        trades = _extract_trades(df, initial_capital)
        num_trades = len(trades)
        win_rate = calculate_win_rate(trades) if trades else 0.0
        profit_factor = calculate_profit_factor(trades) if trades else 0.0
    else:
        num_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        trades = []

    # Sortino Ratio
    sortino = calculate_sortino_ratio(returns)

    # Retorno anualizado
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    if days > 0:
        annual_return = ((final_value / initial_capital) ** (365 / days)) - 1
    else:
        annual_return = 0.0

    # Calmar Ratio
    calmar = calculate_calmar_ratio(cumulative_returns, annual_return)

    metrics = {
        'initial_capital': initial_capital,
        'final_value': round(final_value, 2),
        'net_profit': round(net_profit, 2),
        'total_return_pct': round(total_return * 100, 2),
        'annual_return_pct': round(annual_return * 100, 2),
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown_pct': round(max_dd * 100, 2),
        'num_trades': num_trades,
        'win_rate_pct': round(win_rate * 100, 2),
        'profit_factor': profit_factor,
        'avg_trade': round(net_profit / num_trades, 2) if num_trades > 0 else 0.0,
        'best_trade': round(max(trades), 2) if trades else 0.0,
        'worst_trade': round(min(trades), 2) if trades else 0.0
    }

    return metrics


def _extract_trades(df, initial_capital):
    """
    Extrae los PnL de cada trade individual.

    Args:
        df: DataFrame con señales y retornos
        initial_capital: Capital inicial

    Returns:
        Lista de PnL por trade
    """
    trades = []
    in_position = False
    entry_value = 0

    for i in range(len(df)):
        signal = df['señal'].iloc[i]
        portfolio_value = initial_capital * (1 + df['cumulative_returns'].iloc[i])

        # Entrar en posición
        if not in_position and signal != 0:
            in_position = True
            entry_value = portfolio_value

        # Salir de posición (cambio de señal o señal a neutral)
        elif in_position:
            current_signal = signal
            previous_signal = df['señal'].iloc[i-1] if i > 0 else 0

            # Si la señal cambia, cerrar trade
            if current_signal != previous_signal:
                exit_value = portfolio_value
                trade_pnl = exit_value - entry_value
                trades.append(trade_pnl)

                in_position = False

                # Si nueva señal no es neutral, abrir nuevo trade
                if current_signal != 0:
                    in_position = True
                    entry_value = portfolio_value

    return trades


def print_metrics(metrics):
    """
    Imprime las métricas de forma legible.

    Args:
        metrics: Diccionario de métricas (de calculate_all_metrics)
    """
    print("=" * 60)
    print("MÉTRICAS DE RENDIMIENTO")
    print("=" * 60)
    print(f"\n{'Capital Inicial:':<25} ${metrics['initial_capital']:,.2f}")
    print(f"{'Valor Final:':<25} ${metrics['final_value']:,.2f}")
    print(f"{'Ganancia Neta:':<25} ${metrics['net_profit']:,.2f}")
    print(f"{'Retorno Total:':<25} {metrics['total_return_pct']:.2f}%")
    print(f"{'Retorno Anualizado:':<25} {metrics['annual_return_pct']:.2f}%")

    print(f"\n{'Sharpe Ratio:':<25} {metrics['sharpe_ratio']}")
    print(f"{'Sortino Ratio:':<25} {metrics['sortino_ratio']}")
    print(f"{'Calmar Ratio:':<25} {metrics['calmar_ratio']}")
    print(f"{'Max Drawdown:':<25} {metrics['max_drawdown_pct']:.2f}%")

    print(f"\n{'Número de Trades:':<25} {metrics['num_trades']}")
    print(f"{'Win Rate:':<25} {metrics['win_rate_pct']:.2f}%")
    print(f"{'Profit Factor:':<25} {metrics['profit_factor']}")
    print(f"{'Trade Promedio:':<25} ${metrics['avg_trade']:,.2f}")
    print(f"{'Mejor Trade:':<25} ${metrics['best_trade']:,.2f}")
    print(f"{'Peor Trade:':<25} ${metrics['worst_trade']:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    # Test básico
    print("=== Test de Metrics ===\n")

    # Crear datos de prueba
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 año de retornos diarios
    cumulative = (1 + returns).cumprod() - 1

    print("1. Sharpe Ratio:")
    sharpe = calculate_sharpe_ratio(returns)
    print(f"   Sharpe: {sharpe}\n")

    print("2. Max Drawdown:")
    max_dd = calculate_max_drawdown(cumulative)
    print(f"   Max Drawdown: {max_dd * 100:.2f}%\n")

    print("3. Win Rate:")
    trades = [100, -50, 75, -30, 120, -40, 90]
    win_rate = calculate_win_rate(trades)
    print(f"   Win Rate: {win_rate * 100:.2f}%\n")

    print("4. Profit Factor:")
    pf = calculate_profit_factor(trades)
    print(f"   Profit Factor: {pf}\n")

    print("✓ Test completado exitosamente")
