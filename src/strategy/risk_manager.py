"""
Módulo para gestión de riesgo en trading.
Calcula tamaños de posición, Stop Loss y Take Profit basados en ATR.
"""

import json
import math


class RiskManager:
    """
    Gestor de riesgo para trading algorítmico.
    Utiliza ATR para calcular niveles de Stop Loss y Take Profit dinámicos.
    """

    def __init__(self, config_path='config/config.json'):
        """
        Inicializa el gestor de riesgo.

        Args:
            config_path: Ruta al archivo de configuración
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.trading_config = config['trading']

        # Parámetros de riesgo
        self.capital_per_trade = self.trading_config.get('capital_per_trade', 15)
        self.atr_sl_multiplier = self.trading_config.get('atr_sl_multiplier', 2.0)
        self.atr_tp_multiplier = self.trading_config.get('atr_tp_multiplier', 3.0)
        self.max_open_positions = self.trading_config.get('max_open_positions', 3)
        self.max_daily_loss_pct = self.trading_config.get('max_daily_loss_pct', 0.05)

        # Variables de estado
        self.daily_pnl = 0.0
        self.initial_capital = None
        self.open_positions = 0

    def calculate_position_size(self, current_price, symbol_info):
        """
        Calcula el tamaño de la posición basado en el capital por operación.

        Args:
            current_price: Precio actual del activo
            symbol_info: Información del símbolo de Binance

        Returns:
            Cantidad formateada según requisitos de Binance
        """
        # Calcular cantidad bruta
        quantity = self.capital_per_trade / current_price

        # Aplicar step size de Binance
        step_size = symbol_info['step_size']
        precision = self._get_precision(step_size)

        # Redondear hacia abajo para cumplir con step size
        quantity = math.floor(quantity / step_size) * step_size

        # Formatear con precisión correcta
        formatted_quantity = f"{quantity:.{precision}f}"

        return formatted_quantity

    def calculate_sl_tp(self, entry_price, atr_value, side='BUY', price_precision=2):
        """
        Calcula precios de Stop Loss y Take Profit basados en ATR.

        Args:
            entry_price: Precio de entrada de la operación
            atr_value: Valor actual de ATR
            side: 'BUY' o 'SELL'
            price_precision: Decimales para precios (de symbol_info)

        Returns:
            Tupla (sl_price, tp_price)
        """
        if side == 'BUY':
            # Para compras: SL por debajo, TP por encima
            sl_price = entry_price - (atr_value * self.atr_sl_multiplier)
            tp_price = entry_price + (atr_value * self.atr_tp_multiplier)
        elif side == 'SELL':
            # Para ventas: SL por encima, TP por debajo
            sl_price = entry_price + (atr_value * self.atr_sl_multiplier)
            tp_price = entry_price - (atr_value * self.atr_tp_multiplier)
        else:
            raise ValueError(f"Side debe ser 'BUY' o 'SELL', recibido: {side}")

        # Redondear a la precisión correcta
        sl_price = round(sl_price, price_precision)
        tp_price = round(tp_price, price_precision)

        return sl_price, tp_price

    def calculate_risk_reward_ratio(self, entry_price, sl_price, tp_price):
        """
        Calcula el ratio riesgo/recompensa de la operación.

        Args:
            entry_price: Precio de entrada
            sl_price: Precio de stop loss
            tp_price: Precio de take profit

        Returns:
            Ratio riesgo/recompensa (ej. 1.5 significa 1:1.5)
        """
        riesgo = abs(entry_price - sl_price)
        recompensa = abs(tp_price - entry_price)

        if riesgo == 0:
            return 0

        ratio = recompensa / riesgo
        return round(ratio, 2)

    def puede_abrir_posicion(self, initial_capital=None):
        """
        Verifica si se puede abrir una nueva posición según límites de riesgo.

        Args:
            initial_capital: Capital inicial (para calcular pérdida diaria)

        Returns:
            Tupla (bool, str) - (puede_operar, razón)
        """
        # Verificar límite de posiciones abiertas
        if self.open_positions >= self.max_open_positions:
            return False, f"Máximo de posiciones abiertas alcanzado ({self.max_open_positions})"

        # Verificar pérdida diaria máxima
        if initial_capital:
            self.initial_capital = initial_capital
            max_daily_loss = initial_capital * self.max_daily_loss_pct

            if self.daily_pnl < -max_daily_loss:
                return False, f"Pérdida diaria máxima alcanzada (${abs(self.daily_pnl):.2f} de ${max_daily_loss:.2f})"

        return True, "OK"

    def registrar_operacion(self, pnl):
        """
        Registra el resultado de una operación cerrada.

        Args:
            pnl: Profit and Loss de la operación (positivo = ganancia, negativo = pérdida)
        """
        self.daily_pnl += pnl

        if self.open_positions > 0:
            self.open_positions -= 1

    def abrir_posicion(self):
        """Incrementa el contador de posiciones abiertas."""
        self.open_positions += 1

    def resetear_dia(self):
        """Resetea estadísticas diarias (llamar al inicio de cada día de trading)."""
        self.daily_pnl = 0.0

    def get_estadisticas(self):
        """
        Obtiene estadísticas actuales de gestión de riesgo.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'capital_per_trade': self.capital_per_trade,
            'open_positions': self.open_positions,
            'max_open_positions': self.max_open_positions,
            'daily_pnl': self.daily_pnl,
            'atr_sl_multiplier': self.atr_sl_multiplier,
            'atr_tp_multiplier': self.atr_tp_multiplier,
            'max_daily_loss_pct': self.max_daily_loss_pct * 100
        }

    def _get_precision(self, step_size):
        """Calcula la precisión decimal basada en step_size."""
        step_str = f"{step_size:.10f}".rstrip('0')
        if '.' in step_str:
            return len(step_str.split('.')[-1])
        return 0


if __name__ == "__main__":
    # Test básico del módulo
    print("=== Test de Risk Manager ===\n")

    # Crear gestor de riesgo
    risk_manager = RiskManager()

    # Información de símbolo simulada
    symbol_info = {
        'symbol': 'BTCUSDT',
        'price_precision': 2,
        'quantity_precision': 5,
        'step_size': 0.00001,
        'min_notional': 10
    }

    # Datos de prueba
    current_price = 60500.0
    atr_value = 150.0  # ATR de ejemplo

    print("1. Calcular tamaño de posición:")
    quantity = risk_manager.calculate_position_size(current_price, symbol_info)
    print(f"   Precio actual: ${current_price}")
    print(f"   Capital por operación: ${risk_manager.capital_per_trade}")
    print(f"   Cantidad calculada: {quantity} BTC")
    print(f"   Valor total: ${float(quantity) * current_price:.2f}\n")

    print("2. Calcular Stop Loss y Take Profit (COMPRA):")
    sl_price, tp_price = risk_manager.calculate_sl_tp(
        entry_price=current_price,
        atr_value=atr_value,
        side='BUY',
        price_precision=symbol_info['price_precision']
    )
    print(f"   Precio de entrada: ${current_price}")
    print(f"   ATR: ${atr_value}")
    print(f"   Stop Loss (2x ATR): ${sl_price}")
    print(f"   Take Profit (3x ATR): ${tp_price}")

    rr_ratio = risk_manager.calculate_risk_reward_ratio(current_price, sl_price, tp_price)
    print(f"   Ratio Riesgo/Recompensa: 1:{rr_ratio}\n")

    print("3. Verificar límites de riesgo:")
    puede_operar, razon = risk_manager.puede_abrir_posicion(initial_capital=1000)
    print(f"   Puede operar: {puede_operar}")
    print(f"   Razón: {razon}\n")

    print("4. Simular operaciones:")
    # Abrir posición
    risk_manager.abrir_posicion()
    print(f"   Posición abierta. Total posiciones: {risk_manager.open_positions}")

    # Cerrar con ganancia
    pnl = 25.50
    risk_manager.registrar_operacion(pnl)
    print(f"   Operación cerrada con PnL: +${pnl}")
    print(f"   PnL diario acumulado: ${risk_manager.daily_pnl:.2f}")
    print(f"   Posiciones abiertas: {risk_manager.open_positions}\n")

    print("5. Estadísticas del gestor de riesgo:")
    stats = risk_manager.get_estadisticas()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✓ Test completado exitosamente")
