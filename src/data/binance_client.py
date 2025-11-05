"""
Módulo para configurar y gestionar el cliente de Binance.
Soporta tanto API REST como WebSockets.
"""

import json
from binance.client import Client
from binance.exceptions import BinanceAPIException


class BinanceClientManager:
    """Gestor del cliente de Binance con soporte para testnet."""

    def __init__(self, config_path='config/config.json'):
        """
        Inicializa el gestor del cliente de Binance.

        Args:
            config_path: Ruta al archivo de configuración JSON
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.binance_config = self.config['binance']
        self.client = None

    def get_public_client(self):
        """
        Obtiene un cliente público de Binance (sin autenticación).
        Usado para datos históricos y de mercado.

        Returns:
            Cliente de Binance configurado
        """
        try:
            self.client = Client("", "")
            return self.client
        except BinanceAPIException as e:
            raise Exception(f"Error al crear cliente público: {e}")

    def get_authenticated_client(self):
        """
        Obtiene un cliente autenticado de Binance.
        Usado para trading real (Fases 4 y 5).

        Returns:
            Cliente de Binance autenticado
        """
        use_testnet = self.binance_config.get('testnet', True)

        if use_testnet:
            api_key = self.binance_config.get('testnet_api_key', '')
            api_secret = self.binance_config.get('testnet_api_secret', '')

            if not api_key or not api_secret:
                raise ValueError("Testnet está activado pero las credenciales no están configuradas")

            # Cliente de testnet
            self.client = Client(api_key, api_secret, testnet=True)
        else:
            api_key = self.binance_config.get('api_key', '')
            api_secret = self.binance_config.get('api_secret', '')

            if not api_key or not api_secret:
                raise ValueError("Las credenciales de API no están configuradas")

            # Cliente de producción
            self.client = Client(api_key, api_secret)

        # Verificar conexión
        try:
            account_info = self.client.get_account()
            print(f"✓ Conectado a Binance {'Testnet' if use_testnet else 'Producción'}")
            print(f"✓ Balance USDT: {self._get_usdt_balance(account_info)}")
            return self.client
        except BinanceAPIException as e:
            raise Exception(f"Error al autenticar con Binance: {e}")

    def _get_usdt_balance(self, account_info):
        """Extrae el balance de USDT de la información de cuenta."""
        for balance in account_info['balances']:
            if balance['asset'] == 'USDT':
                return float(balance['free'])
        return 0.0

    def get_symbol_info(self, symbol='BTCUSDT'):
        """
        Obtiene información del símbolo (precisión de precios, lotes, etc.).

        Args:
            symbol: Par de trading (ej. 'BTCUSDT')

        Returns:
            Diccionario con información del símbolo
        """
        if not self.client:
            self.client = self.get_public_client()

        exchange_info = self.client.get_symbol_info(symbol)

        # Extraer información relevante
        filters = {f['filterType']: f for f in exchange_info['filters']}

        return {
            'symbol': exchange_info['symbol'],
            'status': exchange_info['status'],
            'base_asset': exchange_info['baseAsset'],
            'quote_asset': exchange_info['quoteAsset'],
            'price_precision': exchange_info['quotePrecision'],
            'quantity_precision': exchange_info['baseAssetPrecision'],
            'min_notional': float(filters.get('MIN_NOTIONAL', {}).get('minNotional', 10)),
            'min_qty': float(filters.get('LOT_SIZE', {}).get('minQty', 0.00001)),
            'step_size': float(filters.get('LOT_SIZE', {}).get('stepSize', 0.00001))
        }

    def format_quantity(self, quantity, symbol='BTCUSDT'):
        """
        Formatea la cantidad según los requisitos del símbolo.

        Args:
            quantity: Cantidad a formatear
            symbol: Par de trading

        Returns:
            Cantidad formateada como string
        """
        symbol_info = self.get_symbol_info(symbol)
        step_size = symbol_info['step_size']

        # Redondear a la precisión correcta
        precision = len(str(step_size).split('.')[-1].rstrip('0'))
        formatted = f"{quantity:.{precision}f}"

        return formatted


if __name__ == "__main__":
    # Test básico del cliente
    print("=== Test de BinanceClientManager ===\n")

    manager = BinanceClientManager()

    # Cliente público
    print("1. Cliente Público:")
    public_client = manager.get_public_client()
    print(f"✓ Cliente público creado\n")

    # Información del símbolo
    print("2. Información de BTCUSDT:")
    symbol_info = manager.get_symbol_info('BTCUSDT')
    print(f"   Precisión de precio: {symbol_info['price_precision']}")
    print(f"   Precisión de cantidad: {symbol_info['quantity_precision']}")
    print(f"   Min Notional: ${symbol_info['min_notional']}")
    print(f"   Min Qty: {symbol_info['min_qty']}")
    print(f"   Step Size: {symbol_info['step_size']}\n")

    # Formatear cantidad
    print("3. Formateo de Cantidad:")
    test_qty = 0.12345678
    formatted = manager.format_quantity(test_qty, 'BTCUSDT')
    print(f"   Cantidad original: {test_qty}")
    print(f"   Cantidad formateada: {formatted}\n")

    print("✓ Test completado exitosamente")
