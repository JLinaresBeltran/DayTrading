"""
Motor de backtesting vectorizado para estrategias de trading.
Simula operaciones con comisiones y calcula métricas de rendimiento.
"""

import pandas as pd
import numpy as np
from src.utils.metrics import calculate_all_metrics


class VectorizedBacktester:
    """
    Motor de backtesting vectorizado (rápido) para estrategias de trading.
    """

    def __init__(self, df, initial_capital=10000, commission=0.00075, slippage=0.0005):
        """
        Inicializa el backtester.

        Args:
            df: DataFrame con señales (columna 'señal' con valores 1, -1, 0)
            initial_capital: Capital inicial en USD
            commission: Comisión por operación (ej. 0.00075 = 0.075% Binance)
            slippage: Slippage estimado (ej. 0.0005 = 0.05%)
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        # Verificar columnas requeridas
        required_cols = ['timestamp', 'close', 'señal']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"DataFrame falta columnas: {missing}")

    def run_backtest(self):
        """
        Ejecuta el backtest vectorizado.

        Returns:
            DataFrame con resultados del backtest
        """
        df = self.df.copy()

        # 1. Calcular retornos del mercado
        df['returns'] = df['close'].pct_change()

        # 2. Determinar qué columna usar para la estrategia
        # Si existe 'position', usar esa (para estrategias Long-Only)
        # Si no, usar 'señal' (para estrategias bidireccionales)
        position_col = 'position' if 'position' in df.columns else 'señal'

        # 3. Calcular retornos de la estrategia (con lag de 1 para evitar look-ahead bias)
        df['strategy_returns'] = df[position_col].shift(1) * df['returns']

        # 4. Detectar trades (cuando cambia la posición/señal)
        df['signal_diff'] = df[position_col].diff().abs()
        df['trade'] = df['signal_diff'] > 0

        # 5. Aplicar comisiones y slippage en cada trade
        total_cost = self.commission + self.slippage
        df.loc[df['trade'], 'strategy_returns'] = df.loc[df['trade'], 'strategy_returns'] - total_cost

        # 6. Calcular retornos acumulados
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod() - 1

        # 7. Calcular valor del portafolio
        df['portfolio_value'] = self.initial_capital * (1 + df['cumulative_returns'])

        # 8. Calcular retornos acumulados del buy-and-hold (benchmark)
        df['buy_hold_cumulative'] = (1 + df['returns']).cumprod() - 1
        df['buy_hold_value'] = self.initial_capital * (1 + df['buy_hold_cumulative'])

        self.results = df

        return df

    def run_backtest_with_stop_loss(self, atr_column='ATRr_14', atr_multiplier=2.0):
        """
        Ejecuta el backtest con Stop Loss dinámico basado en ATR.

        ITERACIÓN 10.1 - CAPA 3 (Gestión de Riesgo):
        Esta función simula operaciones Long-Only con Stop Loss dinámico calculado
        usando ATR. El SL se calcula al momento de entrada y se mantiene fijo para
        esa operación específica.

        Lógica:
        1. Al detectar señal de COMPRA (señal == 1), abrimos posición LONG
        2. Calculamos SL = entry_price - (ATR * atr_multiplier)
        3. En cada vela subsiguiente, verificamos si low <= stop_loss_price
        4. Si se toca el SL, cerramos con pérdida
        5. Si aparece señal de VENTA (señal == -1), cerramos normalmente (Take Profit)
        6. Lo que ocurra primero (SL o TP), cierra la operación

        Args:
            atr_column: Nombre de la columna ATR en el DataFrame (default: 'ATRr_14')
            atr_multiplier: Multiplicador del ATR para calcular el SL (default: 2.0)

        Returns:
            DataFrame con resultados del backtest incluyendo columna 'exit_reason'
        """
        df = self.df.copy()

        # Verificar que existe la columna ATR
        if atr_column not in df.columns:
            raise ValueError(f"Columna '{atr_column}' no encontrada. Columnas disponibles: {df.columns.tolist()}")

        # Verificar que existe la columna 'low' para detectar toques de SL
        if 'low' not in df.columns:
            raise ValueError("Columna 'low' no encontrada. Se requiere para simular Stop Loss.")

        # Inicializar columnas de resultados
        df['position_active'] = 0  # 1 si hay posición abierta, 0 si no
        df['entry_price'] = np.nan
        df['stop_loss_price'] = np.nan
        df['exit_price'] = np.nan
        df['exit_reason'] = ''  # 'SL', 'TP', o ''
        df['pnl'] = 0.0

        # Variables de estado
        in_position = False
        position_type = None  # 'LONG' o 'SHORT'
        entry_price = 0
        stop_loss_price = 0
        entry_idx = None
        portfolio_value = self.initial_capital
        trades_log = []

        # Simulación trade-by-trade
        for i in range(len(df)):
            signal = df['señal'].iloc[i]
            current_price = df['close'].iloc[i]
            current_low = df['low'].iloc[i]
            current_atr = df[atr_column].iloc[i]

            # Si NO estamos en posición, buscamos señal de entrada
            if not in_position:
                # LONG: Señal de COMPRA (signal == 1)
                if signal == 1:
                    # Abrir posición LONG
                    in_position = True
                    position_type = 'LONG'
                    entry_price = current_price
                    entry_idx = i

                    # Calcular Stop Loss basado en ATR (LONG)
                    stop_loss_price = entry_price - (current_atr * atr_multiplier)

                    # Registrar en DataFrame
                    df.at[i, 'position_active'] = 1
                    df.at[i, 'entry_price'] = entry_price
                    df.at[i, 'stop_loss_price'] = stop_loss_price

                # SHORT: Señal de VENTA EN CORTO (signal == -1)
                elif signal == -1:
                    # Abrir posición SHORT
                    in_position = True
                    position_type = 'SHORT'
                    entry_price = current_price
                    entry_idx = i

                    # Calcular Stop Loss basado en ATR (SHORT) ⚠️ INVERTIDO
                    stop_loss_price = entry_price + (current_atr * atr_multiplier)

                    # Registrar en DataFrame
                    df.at[i, 'position_active'] = 1
                    df.at[i, 'entry_price'] = entry_price
                    df.at[i, 'stop_loss_price'] = stop_loss_price

            # Si ESTAMOS en posición, verificamos condiciones de salida
            else:
                # Marcar que seguimos en posición
                df.at[i, 'position_active'] = 1
                df.at[i, 'stop_loss_price'] = stop_loss_price

                # ==========================================
                # LÓGICA PARA POSICIÓN LONG
                # ==========================================
                if position_type == 'LONG':
                    # CONDICIÓN 1: Verificar si se tocó el Stop Loss (LONG)
                    if current_low <= stop_loss_price:
                        # Cerrar por Stop Loss
                        exit_price = stop_loss_price  # Salimos al precio del SL
                        exit_reason = 'SL'

                        # Calcular PnL (LONG)
                        pnl_pct = ((exit_price - entry_price) / entry_price) - (self.commission + self.slippage)
                        pnl_usd = portfolio_value * pnl_pct

                        # Actualizar portafolio
                        portfolio_value += pnl_usd

                        # Registrar salida
                        df.at[i, 'exit_price'] = exit_price
                        df.at[i, 'exit_reason'] = exit_reason
                        df.at[i, 'pnl'] = pnl_usd
                        df.at[i, 'position_active'] = 0

                        # Guardar trade
                        trades_log.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl_usd,
                            'position_type': position_type
                        })

                        # Resetear estado
                        in_position = False
                        position_type = None
                        entry_price = 0
                        stop_loss_price = 0
                        entry_idx = None

                    # CONDICIÓN 2: Verificar señal de VENTA (Take Profit LONG)
                    elif signal == -1:
                        # Cerrar por Take Profit
                        exit_price = current_price
                        exit_reason = 'TP'

                        # Calcular PnL (LONG)
                        pnl_pct = ((exit_price - entry_price) / entry_price) - (self.commission + self.slippage)
                        pnl_usd = portfolio_value * pnl_pct

                        # Actualizar portafolio
                        portfolio_value += pnl_usd

                        # Registrar salida
                        df.at[i, 'exit_price'] = exit_price
                        df.at[i, 'exit_reason'] = exit_reason
                        df.at[i, 'pnl'] = pnl_usd
                        df.at[i, 'position_active'] = 0

                        # Guardar trade
                        trades_log.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl_usd,
                            'position_type': position_type
                        })

                        # Resetear estado
                        in_position = False
                        position_type = None
                        entry_price = 0
                        stop_loss_price = 0
                        entry_idx = None

                # ==========================================
                # LÓGICA PARA POSICIÓN SHORT ⚠️ INVERTIDA
                # ==========================================
                elif position_type == 'SHORT':
                    current_high = df['high'].iloc[i]  # ⚠️ Usamos 'high' para SHORT

                    # CONDICIÓN 1: Verificar si se tocó el Stop Loss (SHORT) ⚠️ INVERTIDO
                    if current_high >= stop_loss_price:
                        # Cerrar por Stop Loss
                        exit_price = stop_loss_price  # Salimos al precio del SL
                        exit_reason = 'SL'

                        # Calcular PnL (SHORT) ⚠️ INVERTIDO
                        pnl_pct = ((entry_price - exit_price) / entry_price) - (self.commission + self.slippage)
                        pnl_usd = portfolio_value * pnl_pct

                        # Actualizar portafolio
                        portfolio_value += pnl_usd

                        # Registrar salida
                        df.at[i, 'exit_price'] = exit_price
                        df.at[i, 'exit_reason'] = exit_reason
                        df.at[i, 'pnl'] = pnl_usd
                        df.at[i, 'position_active'] = 0

                        # Guardar trade
                        trades_log.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl_usd,
                            'position_type': position_type
                        })

                        # Resetear estado
                        in_position = False
                        position_type = None
                        entry_price = 0
                        stop_loss_price = 0
                        entry_idx = None

                    # CONDICIÓN 2: Verificar señal de CUBRIR CORTO (Take Profit SHORT) ⚠️ INVERTIDO
                    elif signal == 1:
                        # Cerrar por Take Profit
                        exit_price = current_price
                        exit_reason = 'TP'

                        # Calcular PnL (SHORT) ⚠️ INVERTIDO
                        pnl_pct = ((entry_price - exit_price) / entry_price) - (self.commission + self.slippage)
                        pnl_usd = portfolio_value * pnl_pct

                        # Actualizar portafolio
                        portfolio_value += pnl_usd

                        # Registrar salida
                        df.at[i, 'exit_price'] = exit_price
                        df.at[i, 'exit_reason'] = exit_reason
                        df.at[i, 'pnl'] = pnl_usd
                        df.at[i, 'position_active'] = 0

                        # Guardar trade
                        trades_log.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl_usd,
                            'position_type': position_type
                        })

                        # Resetear estado
                        in_position = False
                        position_type = None
                        entry_price = 0
                        stop_loss_price = 0
                        entry_idx = None

        # Calcular valor del portafolio a lo largo del tiempo
        df['portfolio_value'] = self.initial_capital + df['pnl'].cumsum()

        # Calcular retornos
        df['returns'] = df['portfolio_value'].pct_change().fillna(0)
        df['strategy_returns'] = df['returns']  # Alias para compatibilidad con calculate_metrics()
        df['cumulative_returns'] = (df['portfolio_value'] / self.initial_capital) - 1

        # Calcular retornos del buy-and-hold (benchmark) para comparación
        market_returns = df['close'].pct_change().fillna(0)
        df['buy_hold_cumulative'] = (1 + market_returns).cumprod() - 1
        df['buy_hold_value'] = self.initial_capital * (1 + df['buy_hold_cumulative'])

        # Guardar trades para análisis
        self.trades_log = pd.DataFrame(trades_log)
        self.results = df

        return df

    def run_backtest_with_sl_tp(self, atr_column='ATRr_14', sl_multiplier=2.0, tp_multiplier=3.0):
        """
        Ejecuta el backtest con Stop Loss y Take Profit dinámicos basados en ATR.

        ITERACIÓN 23 - Gestión de Riesgo con SL y TP parametrizables:
        Esta función simula operaciones Long-Only con SL y TP dinámicos calculados
        usando ATR. Ambos niveles se calculan al momento de entrada y se mantienen
        fijos para esa operación específica.

        Lógica:
        1. Al detectar señal de COMPRA (señal == 1), abrimos posición LONG
        2. Calculamos SL = entry_price - (ATR × sl_multiplier)
        3. Calculamos TP = entry_price + (ATR × tp_multiplier)
        4. En cada vela subsiguiente, verificamos si:
           - low <= stop_loss_price → Cierra con pérdida (SL)
           - high >= take_profit_price → Cierra con ganancia (TP)
        5. Lo que ocurra primero (SL o TP), cierra la operación
        6. Si aparece señal de VENTA (-1) antes, cierra normalmente

        Args:
            atr_column: Nombre de la columna ATR en el DataFrame (default: 'ATRr_14')
            sl_multiplier: Multiplicador del ATR para calcular el SL (default: 2.0)
            tp_multiplier: Multiplicador del ATR para calcular el TP (default: 3.0)

        Returns:
            DataFrame con resultados del backtest incluyendo columna 'exit_reason'
        """
        df = self.df.copy()

        # Verificar que existe la columna ATR
        if atr_column not in df.columns:
            raise ValueError(f"Columna '{atr_column}' no encontrada. Columnas disponibles: {df.columns.tolist()}")

        # Verificar columnas requeridas
        if 'low' not in df.columns or 'high' not in df.columns:
            raise ValueError("Columnas 'low' y 'high' no encontradas. Se requieren para simular SL/TP.")

        # Inicializar columnas de resultados
        df['position_active'] = 0
        df['entry_price'] = np.nan
        df['stop_loss_price'] = np.nan
        df['take_profit_price'] = np.nan
        df['exit_price'] = np.nan
        df['exit_reason'] = ''
        df['pnl'] = 0.0

        # Variables de estado
        in_position = False
        position_type = None  # 'LONG' o 'SHORT'
        entry_price = 0
        stop_loss_price = 0
        take_profit_price = 0
        entry_idx = None
        portfolio_value = self.initial_capital
        trades_log = []

        # Simulación trade-by-trade
        for i in range(len(df)):
            signal = df['señal'].iloc[i]
            current_price = df['close'].iloc[i]
            current_low = df['low'].iloc[i]
            current_high = df['high'].iloc[i]
            current_atr = df[atr_column].iloc[i]

            # Si NO estamos en posición, buscamos señal de entrada
            if not in_position:
                # LONG: Señal de COMPRA (signal == 1)
                if signal == 1:
                    # Abrir posición LONG
                    in_position = True
                    position_type = 'LONG'
                    entry_price = current_price
                    entry_idx = i

                    # Calcular Stop Loss y Take Profit basados en ATR
                    stop_loss_price = entry_price - (current_atr * sl_multiplier)
                    take_profit_price = entry_price + (current_atr * tp_multiplier)

                    # Registrar en DataFrame
                    df.at[i, 'position_active'] = 1
                    df.at[i, 'entry_price'] = entry_price
                    df.at[i, 'stop_loss_price'] = stop_loss_price
                    df.at[i, 'take_profit_price'] = take_profit_price

                # SHORT: Señal de VENTA EN CORTO (signal == -1)
                elif signal == -1:
                    # Abrir posición SHORT
                    in_position = True
                    position_type = 'SHORT'
                    entry_price = current_price
                    entry_idx = i

                    # Calcular Stop Loss y Take Profit (invertidos para SHORT)
                    stop_loss_price = entry_price + (current_atr * sl_multiplier)
                    take_profit_price = entry_price - (current_atr * tp_multiplier)

                    # Registrar en DataFrame
                    df.at[i, 'position_active'] = 1
                    df.at[i, 'entry_price'] = entry_price
                    df.at[i, 'stop_loss_price'] = stop_loss_price
                    df.at[i, 'take_profit_price'] = take_profit_price

            # Si ESTAMOS en posición, verificamos condiciones de salida
            else:
                # Marcar que seguimos en posición
                df.at[i, 'position_active'] = 1
                df.at[i, 'stop_loss_price'] = stop_loss_price
                df.at[i, 'take_profit_price'] = take_profit_price

                # ==========================================
                # LÓGICA PARA POSICIÓN LONG
                # ==========================================
                if position_type == 'LONG':
                    # PRIORIDAD 1: Verificar si se tocó el Stop Loss
                    if current_low <= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_reason = 'SL'

                        # Calcular PnL
                        pnl_pct = ((exit_price - entry_price) / entry_price) - (self.commission + self.slippage)
                        pnl_usd = portfolio_value * pnl_pct
                        portfolio_value += pnl_usd

                        # Registrar salida
                        df.at[i, 'exit_price'] = exit_price
                        df.at[i, 'exit_reason'] = exit_reason
                        df.at[i, 'pnl'] = pnl_usd
                        df.at[i, 'position_active'] = 0

                        # Guardar trade
                        trades_log.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl_usd,
                            'position_type': position_type
                        })

                        # Resetear estado
                        in_position = False
                        position_type = None

                    # PRIORIDAD 2: Verificar si se tocó el Take Profit
                    elif current_high >= take_profit_price:
                        exit_price = take_profit_price
                        exit_reason = 'TP'

                        # Calcular PnL
                        pnl_pct = ((exit_price - entry_price) / entry_price) - (self.commission + self.slippage)
                        pnl_usd = portfolio_value * pnl_pct
                        portfolio_value += pnl_usd

                        # Registrar salida
                        df.at[i, 'exit_price'] = exit_price
                        df.at[i, 'exit_reason'] = exit_reason
                        df.at[i, 'pnl'] = pnl_usd
                        df.at[i, 'position_active'] = 0

                        # Guardar trade
                        trades_log.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl_usd,
                            'position_type': position_type
                        })

                        # Resetear estado
                        in_position = False
                        position_type = None

                    # PRIORIDAD 3: Verificar señal de VENTA manual
                    elif signal == -1:
                        exit_price = current_price
                        exit_reason = 'SIGNAL'

                        # Calcular PnL
                        pnl_pct = ((exit_price - entry_price) / entry_price) - (self.commission + self.slippage)
                        pnl_usd = portfolio_value * pnl_pct
                        portfolio_value += pnl_usd

                        # Registrar salida
                        df.at[i, 'exit_price'] = exit_price
                        df.at[i, 'exit_reason'] = exit_reason
                        df.at[i, 'pnl'] = pnl_usd
                        df.at[i, 'position_active'] = 0

                        # Guardar trade
                        trades_log.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl_usd,
                            'position_type': position_type
                        })

                        # Resetear estado
                        in_position = False
                        position_type = None

                # ==========================================
                # LÓGICA PARA POSICIÓN SHORT
                # ==========================================
                elif position_type == 'SHORT':
                    # PRIORIDAD 1: Verificar si se tocó el Stop Loss
                    if current_high >= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_reason = 'SL'

                        # Calcular PnL (SHORT)
                        pnl_pct = ((entry_price - exit_price) / entry_price) - (self.commission + self.slippage)
                        pnl_usd = portfolio_value * pnl_pct
                        portfolio_value += pnl_usd

                        # Registrar salida
                        df.at[i, 'exit_price'] = exit_price
                        df.at[i, 'exit_reason'] = exit_reason
                        df.at[i, 'pnl'] = pnl_usd
                        df.at[i, 'position_active'] = 0

                        # Guardar trade
                        trades_log.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl_usd,
                            'position_type': position_type
                        })

                        # Resetear estado
                        in_position = False
                        position_type = None

                    # PRIORIDAD 2: Verificar si se tocó el Take Profit
                    elif current_low <= take_profit_price:
                        exit_price = take_profit_price
                        exit_reason = 'TP'

                        # Calcular PnL (SHORT)
                        pnl_pct = ((entry_price - exit_price) / entry_price) - (self.commission + self.slippage)
                        pnl_usd = portfolio_value * pnl_pct
                        portfolio_value += pnl_usd

                        # Registrar salida
                        df.at[i, 'exit_price'] = exit_price
                        df.at[i, 'exit_reason'] = exit_reason
                        df.at[i, 'pnl'] = pnl_usd
                        df.at[i, 'position_active'] = 0

                        # Guardar trade
                        trades_log.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl_usd,
                            'position_type': position_type
                        })

                        # Resetear estado
                        in_position = False
                        position_type = None

                    # PRIORIDAD 3: Verificar señal de CUBRIR CORTO manual
                    elif signal == 1:
                        exit_price = current_price
                        exit_reason = 'SIGNAL'

                        # Calcular PnL (SHORT)
                        pnl_pct = ((entry_price - exit_price) / entry_price) - (self.commission + self.slippage)
                        pnl_usd = portfolio_value * pnl_pct
                        portfolio_value += pnl_usd

                        # Registrar salida
                        df.at[i, 'exit_price'] = exit_price
                        df.at[i, 'exit_reason'] = exit_reason
                        df.at[i, 'pnl'] = pnl_usd
                        df.at[i, 'position_active'] = 0

                        # Guardar trade
                        trades_log.append({
                            'entry_idx': entry_idx,
                            'exit_idx': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct * 100,
                            'pnl_usd': pnl_usd,
                            'position_type': position_type
                        })

                        # Resetear estado
                        in_position = False
                        position_type = None

        # Calcular valor del portafolio a lo largo del tiempo
        df['portfolio_value'] = self.initial_capital + df['pnl'].cumsum()

        # Calcular retornos
        df['returns'] = df['portfolio_value'].pct_change().fillna(0)
        df['strategy_returns'] = df['returns']
        df['cumulative_returns'] = (df['portfolio_value'] / self.initial_capital) - 1

        # Calcular retornos del buy-and-hold (benchmark)
        market_returns = df['close'].pct_change().fillna(0)
        df['buy_hold_cumulative'] = (1 + market_returns).cumprod() - 1
        df['buy_hold_value'] = self.initial_capital * (1 + df['buy_hold_cumulative'])

        # Guardar trades para análisis
        self.trades_log = pd.DataFrame(trades_log)
        self.results = df

        return df

    def run_backtest_bidirectional_v2(self, atr_column='ATRr_14', atr_multiplier=2.0):
        """
        Ejecuta backtest BIDIRECCIONAL con sistema de 4 señales para estrategia MTF.

        ITERACIÓN 001 - Motor de Backtest Bidireccional:
        Diseñado específicamente para la estrategia Multi-Timeframe Bidireccional.
        Procesa las 4 señales: 1 (Abrir Long), -1 (Abrir Short), 2 (Cerrar Long), -2 (Cerrar Short).

        DIFERENCIAS con run_backtest_with_stop_loss:
        1. Sistema de 4 señales explícitas (1, -1, 2, -2) en lugar de 2 (1, -1)
        2. Validación estricta de estados: solo abre si in_position==0, solo cierra si in_position correcta
        3. Stop Loss obligatorio (no tiene opción sin SL)
        4. No calcula Take Profit automático - las salidas son por señal (2, -2) o SL

        LÓGICA DE SEÑALES:
          signal == 1: Abrir Long  (solo si in_position == 0)
          signal == -1: Abrir Short (solo si in_position == 0)
          signal == 2: Cerrar Long  (solo si in_position == 1)
          signal == -2: Cerrar Short (solo si in_position == -1)
          signal == 0: Mantener (no hacer nada)

        STOP LOSS BIDIRECCIONAL:
          LONG: SL = entry_price - (ATR × multiplier), verifica low <= SL
          SHORT: SL = entry_price + (ATR × multiplier), verifica high >= SL

        PnL CORRECTO POR DIRECCIÓN:
          LONG: (exit - entry) / entry - costs
          SHORT: (entry - exit) / entry - costs

        Args:
            atr_column: Nombre de la columna ATR (default: 'ATRr_14')
            atr_multiplier: Multiplicador del ATR para SL (default: 2.0)

        Returns:
            DataFrame con resultados del backtest
        """
        df = self.df.copy()

        # Verificar columnas requeridas
        if atr_column not in df.columns:
            raise ValueError(f"Columna '{atr_column}' no encontrada. Disponibles: {df.columns.tolist()}")
        if 'low' not in df.columns or 'high' not in df.columns:
            raise ValueError("Columnas 'low' y 'high' requeridas para Stop Loss")

        # Inicializar columnas de resultados
        df['position_active'] = 0  # 0=plano, 1=long, -1=short
        df['entry_price'] = np.nan
        df['stop_loss_price'] = np.nan
        df['exit_price'] = np.nan
        df['exit_reason'] = ''  # 'SL', 'SIGNAL_2', 'SIGNAL_-2'
        df['pnl'] = 0.0

        # Variables de estado
        in_position = 0  # 0=plano, 1=long activo, -1=short activo
        entry_price = 0
        stop_loss_price = 0
        entry_idx = None
        portfolio_value = self.initial_capital
        trades_log = []

        # Simulación trade-by-trade
        for i in range(len(df)):
            signal = df['señal'].iloc[i]
            current_price = df['close'].iloc[i]
            current_low = df['low'].iloc[i]
            current_high = df['high'].iloc[i]
            current_atr = df[atr_column].iloc[i]

            # ==========================================
            # CASO 1: NO HAY POSICIÓN (in_position == 0)
            # ==========================================
            if in_position == 0:
                # SEÑAL 1: Abrir Long
                if signal == 1:
                    in_position = 1
                    entry_price = current_price
                    entry_idx = i
                    stop_loss_price = entry_price - (current_atr * atr_multiplier)

                    df.at[i, 'position_active'] = 1
                    df.at[i, 'entry_price'] = entry_price
                    df.at[i, 'stop_loss_price'] = stop_loss_price

                # SEÑAL -1: Abrir Short
                elif signal == -1:
                    in_position = -1
                    entry_price = current_price
                    entry_idx = i
                    stop_loss_price = entry_price + (current_atr * atr_multiplier)

                    df.at[i, 'position_active'] = -1
                    df.at[i, 'entry_price'] = entry_price
                    df.at[i, 'stop_loss_price'] = stop_loss_price

            # ==========================================
            # CASO 2: LONG ACTIVO (in_position == 1)
            # ==========================================
            elif in_position == 1:
                # Mantener indicadores de posición activa
                df.at[i, 'position_active'] = 1
                df.at[i, 'entry_price'] = entry_price
                df.at[i, 'stop_loss_price'] = stop_loss_price

                # PRIORIDAD 1: Verificar Stop Loss
                if current_low <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'SL'

                    # Calcular PnL (LONG)
                    pnl_pct = ((exit_price - entry_price) / entry_price) - (self.commission + self.slippage)
                    pnl_usd = portfolio_value * pnl_pct
                    portfolio_value += pnl_usd

                    # Registrar salida
                    df.at[i, 'exit_price'] = exit_price
                    df.at[i, 'exit_reason'] = exit_reason
                    df.at[i, 'pnl'] = pnl_usd
                    df.at[i, 'position_active'] = 0

                    # Guardar trade
                    trades_log.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss_price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_usd': pnl_usd,
                        'position_type': 'LONG'
                    })

                    # Resetear estado
                    in_position = 0

                # PRIORIDAD 2: Verificar Señal de Cierre (SEÑAL 2)
                elif signal == 2:
                    exit_price = current_price
                    exit_reason = 'SIGNAL_2'

                    # Calcular PnL (LONG)
                    pnl_pct = ((exit_price - entry_price) / entry_price) - (self.commission + self.slippage)
                    pnl_usd = portfolio_value * pnl_pct
                    portfolio_value += pnl_usd

                    # Registrar salida
                    df.at[i, 'exit_price'] = exit_price
                    df.at[i, 'exit_reason'] = exit_reason
                    df.at[i, 'pnl'] = pnl_usd
                    df.at[i, 'position_active'] = 0

                    # Guardar trade
                    trades_log.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss_price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_usd': pnl_usd,
                        'position_type': 'LONG'
                    })

                    # Resetear estado
                    in_position = 0

            # ==========================================
            # CASO 3: SHORT ACTIVO (in_position == -1)
            # ==========================================
            elif in_position == -1:
                # Mantener indicadores de posición activa
                df.at[i, 'position_active'] = -1
                df.at[i, 'entry_price'] = entry_price
                df.at[i, 'stop_loss_price'] = stop_loss_price

                # PRIORIDAD 1: Verificar Stop Loss
                if current_high >= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'SL'

                    # Calcular PnL (SHORT)
                    pnl_pct = ((entry_price - exit_price) / entry_price) - (self.commission + self.slippage)
                    pnl_usd = portfolio_value * pnl_pct
                    portfolio_value += pnl_usd

                    # Registrar salida
                    df.at[i, 'exit_price'] = exit_price
                    df.at[i, 'exit_reason'] = exit_reason
                    df.at[i, 'pnl'] = pnl_usd
                    df.at[i, 'position_active'] = 0

                    # Guardar trade
                    trades_log.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss_price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_usd': pnl_usd,
                        'position_type': 'SHORT'
                    })

                    # Resetear estado
                    in_position = 0

                # PRIORIDAD 2: Verificar Señal de Cierre (SEÑAL -2)
                elif signal == -2:
                    exit_price = current_price
                    exit_reason = 'SIGNAL_-2'

                    # Calcular PnL (SHORT)
                    pnl_pct = ((entry_price - exit_price) / entry_price) - (self.commission + self.slippage)
                    pnl_usd = portfolio_value * pnl_pct
                    portfolio_value += pnl_usd

                    # Registrar salida
                    df.at[i, 'exit_price'] = exit_price
                    df.at[i, 'exit_reason'] = exit_reason
                    df.at[i, 'pnl'] = pnl_usd
                    df.at[i, 'position_active'] = 0

                    # Guardar trade
                    trades_log.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss_price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_usd': pnl_usd,
                        'position_type': 'SHORT'
                    })

                    # Resetear estado
                    in_position = 0

        # Calcular valor del portafolio acumulado
        df['portfolio_value'] = self.initial_capital + df['pnl'].cumsum()

        # Calcular retornos de la estrategia (para métricas)
        df['strategy_returns'] = df['portfolio_value'].pct_change().fillna(0)
        df['cumulative_returns'] = (df['portfolio_value'] / self.initial_capital) - 1

        # Calcular retornos del buy-and-hold (benchmark)
        market_returns = df['close'].pct_change().fillna(0)
        df['buy_hold_cumulative'] = (1 + market_returns).cumprod() - 1
        df['buy_hold_value'] = self.initial_capital * (1 + df['buy_hold_cumulative'])

        # Guardar trades para análisis
        self.trades_log = pd.DataFrame(trades_log)
        self.results = df

        return df

    def run_backtest_quant_flow(self, atr_column='ATRr_14', atr_multiplier=2.0):
        """
        Ejecuta backtest para estrategia QUANT-FLOW con gestión avanzada de Take Profit.

        CARACTERÍSTICAS ESPECÍFICAS:
        1. Stop Loss Inicial: entry ± (ATR × multiplier)
        2. TP1 (Breakeven): Cuando ganancia = 1.5R, mueve SL a entry
        3. TP2 (Trailing): Después de 1.5R, SL sigue min/max de últimas 3 velas
        4. Weekend Exit: Cierre forzoso Viernes 23:00 UTC

        LÓGICA DE GESTIÓN:
        - Long:
          * SL inicial = entry - (ATR × multiplier)
          * TP1 @ 1.5R: SL → entry (breakeven)
          * TP2 > 1.5R: SL → max(SL, min(últimas 3 velas))
        - Short:
          * SL inicial = entry + (ATR × multiplier)
          * TP1 @ 1.5R: SL → entry (breakeven)
          * TP2 > 1.5R: SL → min(SL, max(últimas 3 velas))

        Args:
            atr_column: Columna del ATR (default: 'ATRr_14')
            atr_multiplier: Multiplicador del ATR para SL (default: 2.0)

        Returns:
            DataFrame con resultados del backtest
        """
        df = self.df.copy()

        # Verificar columnas requeridas
        if atr_column not in df.columns:
            raise ValueError(f"Columna '{atr_column}' no encontrada")
        if 'timestamp' not in df.columns:
            raise ValueError("Columna 'timestamp' requerida para Time Exit")

        # Asegurar que timestamp sea datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Calcular día de la semana y hora para Weekend Exit
        df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=Lun, 4=Vie
        df['hour'] = df['timestamp'].dt.hour

        # Inicializar columnas de resultados
        df['position_active'] = 0
        df['entry_price'] = np.nan
        df['stop_loss_price'] = np.nan
        df['exit_price'] = np.nan
        df['exit_reason'] = ''
        df['pnl'] = 0.0
        df['tp_stage'] = ''  # 'INITIAL', 'TP1_BREAKEVEN', 'TP2_TRAILING'

        # Variables de estado
        in_position = 0  # 0=plano, 1=long, -1=short
        entry_price = 0
        stop_loss_price = 0
        entry_idx = None
        entry_atr = 0
        risk_initial = 0
        tp_stage = 'INITIAL'
        portfolio_value = self.initial_capital
        trades_log = []

        # Simulación trade-by-trade
        for i in range(len(df)):
            signal = df['señal'].iloc[i]
            current_price = df['close'].iloc[i]
            current_low = df['low'].iloc[i]
            current_high = df['high'].iloc[i]
            current_atr = df[atr_column].iloc[i]
            current_dayofweek = df['dayofweek'].iloc[i]
            current_hour = df['hour'].iloc[i]

            # ==========================================
            # WEEKEND EXIT: Cerrar posición Viernes 23:00 UTC
            # ==========================================
            if in_position != 0 and current_dayofweek == 4 and current_hour == 23:
                exit_price = current_price
                exit_reason = 'WEEKEND_EXIT'

                # Calcular PnL según dirección
                if in_position == 1:  # Long
                    pnl_pct = ((exit_price - entry_price) / entry_price) - (self.commission + self.slippage)
                else:  # Short
                    pnl_pct = ((entry_price - exit_price) / entry_price) - (self.commission + self.slippage)

                pnl_usd = portfolio_value * pnl_pct
                portfolio_value += pnl_usd

                # Registrar salida
                df.at[i, 'exit_price'] = exit_price
                df.at[i, 'exit_reason'] = exit_reason
                df.at[i, 'pnl'] = pnl_usd
                df.at[i, 'position_active'] = 0

                # Guardar trade
                trades_log.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stop_loss': stop_loss_price,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'position_type': 'LONG' if in_position == 1 else 'SHORT',
                    'tp_stage': tp_stage
                })

                # Resetear estado
                in_position = 0
                tp_stage = 'INITIAL'
                continue

            # ==========================================
            # CASO 1: NO HAY POSICIÓN
            # ==========================================
            if in_position == 0:
                if signal == 1:  # Abrir Long
                    in_position = 1
                    entry_price = current_price
                    entry_idx = i
                    entry_atr = current_atr
                    stop_loss_price = entry_price - (entry_atr * atr_multiplier)
                    risk_initial = entry_price - stop_loss_price
                    tp_stage = 'INITIAL'

                    df.at[i, 'position_active'] = 1
                    df.at[i, 'entry_price'] = entry_price
                    df.at[i, 'stop_loss_price'] = stop_loss_price
                    df.at[i, 'tp_stage'] = tp_stage

                elif signal == -1:  # Abrir Short
                    in_position = -1
                    entry_price = current_price
                    entry_idx = i
                    entry_atr = current_atr
                    stop_loss_price = entry_price + (entry_atr * atr_multiplier)
                    risk_initial = stop_loss_price - entry_price
                    tp_stage = 'INITIAL'

                    df.at[i, 'position_active'] = -1
                    df.at[i, 'entry_price'] = entry_price
                    df.at[i, 'stop_loss_price'] = stop_loss_price
                    df.at[i, 'tp_stage'] = tp_stage

            # ==========================================
            # CASO 2: LONG ACTIVO
            # ==========================================
            elif in_position == 1:
                # Mantener indicadores
                df.at[i, 'position_active'] = 1
                df.at[i, 'entry_price'] = entry_price
                df.at[i, 'stop_loss_price'] = stop_loss_price
                df.at[i, 'tp_stage'] = tp_stage

                # Calcular ganancia actual en R (múltiplos del riesgo)
                profit_current = current_price - entry_price
                profit_r = profit_current / risk_initial if risk_initial > 0 else 0

                # GESTIÓN DINÁMICA DE STOP LOSS
                # TP1: Si ganancia >= 1.5R → Mover SL a breakeven
                if tp_stage == 'INITIAL' and profit_r >= 1.5:
                    stop_loss_price = entry_price
                    tp_stage = 'TP1_BREAKEVEN'
                    df.at[i, 'stop_loss_price'] = stop_loss_price
                    df.at[i, 'tp_stage'] = tp_stage

                # TP2: Si en breakeven → Activar trailing stop
                elif tp_stage == 'TP1_BREAKEVEN':
                    # Trailing: SL sigue mínimos de últimas 3 velas
                    if i >= 3:
                        trailing_low = df['low'].iloc[i-3:i].min()
                        stop_loss_price = max(stop_loss_price, trailing_low)
                        df.at[i, 'stop_loss_price'] = stop_loss_price
                    tp_stage = 'TP2_TRAILING'
                    df.at[i, 'tp_stage'] = tp_stage

                elif tp_stage == 'TP2_TRAILING':
                    # Continuar trailing
                    if i >= 3:
                        trailing_low = df['low'].iloc[i-3:i].min()
                        stop_loss_price = max(stop_loss_price, trailing_low)
                        df.at[i, 'stop_loss_price'] = stop_loss_price

                # VERIFICAR STOP LOSS
                if current_low <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'SL' if tp_stage == 'INITIAL' else f'SL_{tp_stage}'

                    pnl_pct = ((exit_price - entry_price) / entry_price) - (self.commission + self.slippage)
                    pnl_usd = portfolio_value * pnl_pct
                    portfolio_value += pnl_usd

                    df.at[i, 'exit_price'] = exit_price
                    df.at[i, 'exit_reason'] = exit_reason
                    df.at[i, 'pnl'] = pnl_usd
                    df.at[i, 'position_active'] = 0

                    trades_log.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss_price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_usd': pnl_usd,
                        'position_type': 'LONG',
                        'tp_stage': tp_stage
                    })

                    in_position = 0
                    tp_stage = 'INITIAL'

            # ==========================================
            # CASO 3: SHORT ACTIVO
            # ==========================================
            elif in_position == -1:
                # Mantener indicadores
                df.at[i, 'position_active'] = -1
                df.at[i, 'entry_price'] = entry_price
                df.at[i, 'stop_loss_price'] = stop_loss_price
                df.at[i, 'tp_stage'] = tp_stage

                # Calcular ganancia actual en R
                profit_current = entry_price - current_price
                profit_r = profit_current / risk_initial if risk_initial > 0 else 0

                # GESTIÓN DINÁMICA DE STOP LOSS
                # TP1: Si ganancia >= 1.5R → Mover SL a breakeven
                if tp_stage == 'INITIAL' and profit_r >= 1.5:
                    stop_loss_price = entry_price
                    tp_stage = 'TP1_BREAKEVEN'
                    df.at[i, 'stop_loss_price'] = stop_loss_price
                    df.at[i, 'tp_stage'] = tp_stage

                # TP2: Si en breakeven → Activar trailing stop
                elif tp_stage == 'TP1_BREAKEVEN':
                    # Trailing: SL sigue máximos de últimas 3 velas
                    if i >= 3:
                        trailing_high = df['high'].iloc[i-3:i].max()
                        stop_loss_price = min(stop_loss_price, trailing_high)
                        df.at[i, 'stop_loss_price'] = stop_loss_price
                    tp_stage = 'TP2_TRAILING'
                    df.at[i, 'tp_stage'] = tp_stage

                elif tp_stage == 'TP2_TRAILING':
                    # Continuar trailing
                    if i >= 3:
                        trailing_high = df['high'].iloc[i-3:i].max()
                        stop_loss_price = min(stop_loss_price, trailing_high)
                        df.at[i, 'stop_loss_price'] = stop_loss_price

                # VERIFICAR STOP LOSS
                if current_high >= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = 'SL' if tp_stage == 'INITIAL' else f'SL_{tp_stage}'

                    pnl_pct = ((entry_price - exit_price) / entry_price) - (self.commission + self.slippage)
                    pnl_usd = portfolio_value * pnl_pct
                    portfolio_value += pnl_usd

                    df.at[i, 'exit_price'] = exit_price
                    df.at[i, 'exit_reason'] = exit_reason
                    df.at[i, 'pnl'] = pnl_usd
                    df.at[i, 'position_active'] = 0

                    trades_log.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss_price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_usd': pnl_usd,
                        'position_type': 'SHORT',
                        'tp_stage': tp_stage
                    })

                    in_position = 0
                    tp_stage = 'INITIAL'

        # Calcular métricas finales
        df['portfolio_value'] = self.initial_capital + df['pnl'].cumsum()
        df['strategy_returns'] = df['portfolio_value'].pct_change().fillna(0)
        df['cumulative_returns'] = (df['portfolio_value'] / self.initial_capital) - 1

        # Buy-and-hold benchmark
        market_returns = df['close'].pct_change().fillna(0)
        df['buy_hold_cumulative'] = (1 + market_returns).cumprod() - 1
        df['buy_hold_value'] = self.initial_capital * (1 + df['buy_hold_cumulative'])

        # Guardar trades
        self.trades_log = pd.DataFrame(trades_log)
        self.results = df

        return df

    def calculate_metrics(self):
        """
        Calcula métricas de rendimiento del backtest.

        Returns:
            Diccionario con métricas
        """
        if not hasattr(self, 'results'):
            raise ValueError("Debe ejecutar run_backtest() primero")

        # Si usamos backtest con Stop Loss, usar trades_log real en lugar de _extract_trades
        if hasattr(self, 'trades_log') and not self.trades_log.empty:
            # Calcular métricas usando trades_log REAL (para backtests con SL)
            metrics = self._calculate_metrics_from_trades_log()
        else:
            # Calcular métricas usando método estándar (para backtests vectorizados)
            metrics = calculate_all_metrics(self.results, self.initial_capital)

        # Añadir métricas adicionales específicas del backtest
        metrics['commission'] = self.commission
        metrics['slippage'] = self.slippage

        # Comparar con buy-and-hold
        buy_hold_return = (self.results['buy_hold_value'].iloc[-1] / self.initial_capital) - 1
        metrics['buy_hold_return_pct'] = round(buy_hold_return * 100, 2)
        metrics['excess_return_pct'] = round(metrics['total_return_pct'] - (buy_hold_return * 100), 2)

        return metrics

    def _calculate_metrics_from_trades_log(self):
        """
        Calcula métricas usando el trades_log real (para backtests con Stop Loss).

        Returns:
            Diccionario con métricas
        """
        df = self.results
        trades_log = self.trades_log

        # Valor final y retornos
        final_value = df['portfolio_value'].iloc[-1]
        net_profit = final_value - self.initial_capital
        total_return = (final_value / self.initial_capital) - 1

        # Retornos para Sharpe/Sortino
        returns = df['strategy_returns'].dropna()
        cumulative_returns = df['cumulative_returns'].dropna()

        # Sharpe Ratio
        from src.utils.metrics import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown, calculate_calmar_ratio
        sharpe = calculate_sharpe_ratio(returns)

        # Max Drawdown
        max_dd = calculate_max_drawdown(cumulative_returns)

        # Métricas de trades desde trades_log REAL
        if len(trades_log) > 0:
            trades_pnl = trades_log['pnl_usd'].tolist()
            num_trades = len(trades_pnl)

            # Win Rate: % de trades con PnL > 0
            winning_trades = sum(1 for pnl in trades_pnl if pnl > 0)
            win_rate = winning_trades / num_trades if num_trades > 0 else 0.0

            # Profit Factor: Gross Profit / Gross Loss
            gross_profit = sum(pnl for pnl in trades_pnl if pnl > 0)
            gross_loss = abs(sum(pnl for pnl in trades_pnl if pnl < 0))
            if gross_loss == 0:
                profit_factor = float('inf') if gross_profit > 0 else 0.0
            else:
                profit_factor = gross_profit / gross_loss

            avg_trade = net_profit / num_trades
            best_trade = max(trades_pnl) if trades_pnl else 0.0
            worst_trade = min(trades_pnl) if trades_pnl else 0.0
        else:
            num_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade = 0.0
            best_trade = 0.0
            worst_trade = 0.0

        # Sortino Ratio
        sortino = calculate_sortino_ratio(returns)

        # Retorno anualizado
        days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
        if days > 0:
            annual_return = ((final_value / self.initial_capital) ** (365 / days)) - 1
        else:
            annual_return = 0.0

        # Calmar Ratio
        calmar = calculate_calmar_ratio(cumulative_returns, annual_return)

        metrics = {
            'initial_capital': self.initial_capital,
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
            'profit_factor': round(profit_factor, 2),
            'avg_trade': round(avg_trade, 2),
            'best_trade': round(best_trade, 2),
            'worst_trade': round(worst_trade, 2)
        }

        return metrics

    def get_trades_summary(self):
        """
        Obtiene un resumen de todas las operaciones realizadas.

        Returns:
            DataFrame con información de cada trade
        """
        if not hasattr(self, 'results'):
            raise ValueError("Debe ejecutar run_backtest() primero")

        df = self.results

        trades = []
        in_position = False
        entry_idx = None
        entry_price = None
        entry_signal = None

        for i in range(len(df)):
            current_signal = df['señal'].iloc[i]

            # Entrar en posición
            if not in_position and current_signal != 0:
                in_position = True
                entry_idx = i
                entry_price = df['close'].iloc[i]
                entry_signal = current_signal

            # Salir de posición
            elif in_position:
                prev_signal = df['señal'].iloc[i-1] if i > 0 else 0

                if current_signal != prev_signal:
                    # Cerrar trade
                    exit_price = df['close'].iloc[i]
                    exit_idx = i

                    # Calcular PnL
                    if entry_signal == 1:  # LONG
                        pnl_pct = ((exit_price - entry_price) / entry_price) - (self.commission + self.slippage)
                    else:  # SHORT
                        pnl_pct = ((entry_price - exit_price) / entry_price) - (self.commission + self.slippage)

                    pnl_usd = self.initial_capital * pnl_pct

                    trades.append({
                        'entry_time': df['timestamp'].iloc[entry_idx],
                        'exit_time': df['timestamp'].iloc[exit_idx],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'LONG' if entry_signal == 1 else 'SHORT',
                        'pnl_pct': pnl_pct * 100,
                        'pnl_usd': pnl_usd,
                        'duration': df['timestamp'].iloc[exit_idx] - df['timestamp'].iloc[entry_idx]
                    })

                    in_position = False

                    # Abrir nuevo trade si la señal no es neutral
                    if current_signal != 0:
                        in_position = True
                        entry_idx = i
                        entry_price = df['close'].iloc[i]
                        entry_signal = current_signal

        return pd.DataFrame(trades)


if __name__ == "__main__":
    # Test básico
    from src.data.binance_client import BinanceClientManager
    from src.data.data_fetcher import obtener_datos_binance
    from src.indicators.technical import agregar_indicadores
    from src.strategy.signal_generator import generar_señales
    from src.utils.metrics import print_metrics

    print("=== Test de Backtesting Engine ===\n")

    # Preparar datos
    print("1. Preparando datos...")
    manager = BinanceClientManager()
    client = manager.get_public_client()

    df = obtener_datos_binance(client, 'BTCUSDT', '5m', '30 days ago UTC')
    df = agregar_indicadores(df)
    df = generar_señales(df)

    # Ejecutar backtest
    print("\n2. Ejecutando backtest...")
    backtester = VectorizedBacktester(
        df=df,
        initial_capital=10000,
        commission=0.00075,
        slippage=0.0005
    )

    results = backtester.run_backtest()

    # Calcular métricas
    print("\n3. Métricas de rendimiento:")
    metrics = backtester.calculate_metrics()
    print_metrics(metrics)

    # Resumen de trades
    print("\n4. Últimos 5 trades:")
    trades_df = backtester.get_trades_summary()
    if len(trades_df) > 0:
        print(trades_df.tail())
    else:
        print("   No se ejecutaron trades")

    print("\n✓ Test completado exitosamente")
