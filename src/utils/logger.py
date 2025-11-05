"""
Módulo para configurar el sistema de logging del bot.
Soporta logging a archivo con rotación y salida a consola.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(name='TradingBot', log_file=None, level=logging.INFO, max_bytes=10*1024*1024, backup_count=5):
    """
    Configura y retorna un logger con handlers de archivo y consola.

    Args:
        name: Nombre del logger
        log_file: Ruta del archivo de log (si None, usa logs/bot.log)
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Tamaño máximo del archivo de log antes de rotar (default: 10MB)
        backup_count: Número de archivos de backup a mantener

    Returns:
        Logger configurado
    """
    # Crear directorio de logs si no existe
    if log_file is None:
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'bot.log')
    else:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicación de handlers
    if logger.handlers:
        logger.handlers.clear()

    # Formato de log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler de archivo con rotación
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Añadir handlers al logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_trade_logger(log_file='logs/trades.log'):
    """
    Configura un logger especializado para registrar operaciones de trading.

    Args:
        log_file: Archivo donde se guardarán los trades

    Returns:
        Logger para trades
    """
    logger = setup_logger(
        name='TradeLogger',
        log_file=log_file,
        level=logging.INFO
    )

    return logger


def log_trade(logger, trade_type, symbol, price, quantity, sl=None, tp=None, pnl=None):
    """
    Registra una operación de trading con formato estructurado.

    Args:
        logger: Logger a usar
        trade_type: Tipo de operación ('BUY', 'SELL', 'CLOSE')
        symbol: Par de trading ('BTCUSDT')
        price: Precio de ejecución
        quantity: Cantidad operada
        sl: Precio de stop loss (opcional)
        tp: Precio de take profit (opcional)
        pnl: Profit/Loss de la operación (para cierre)
    """
    trade_info = f"{trade_type} {symbol} - Precio: ${price:.2f}, Cantidad: {quantity}"

    if sl is not None:
        trade_info += f", SL: ${sl:.2f}"

    if tp is not None:
        trade_info += f", TP: ${tp:.2f}"

    if pnl is not None:
        pnl_sign = '+' if pnl >= 0 else ''
        trade_info += f", PnL: {pnl_sign}${pnl:.2f}"

    logger.info(trade_info)


def log_signal(logger, signal_type, price, indicators):
    """
    Registra una señal de trading con indicadores relevantes.

    Args:
        logger: Logger a usar
        signal_type: Tipo de señal ('COMPRA', 'VENTA', 'NEUTRAL')
        price: Precio actual
        indicators: Diccionario con valores de indicadores
    """
    ind_str = ', '.join([f"{k}: {v:.2f}" for k, v in indicators.items()])
    logger.info(f"SEÑAL: {signal_type} @ ${price:.2f} | {ind_str}")


def log_error(logger, error_msg, exception=None):
    """
    Registra un error con información detallada.

    Args:
        logger: Logger a usar
        error_msg: Mensaje de error
        exception: Excepción capturada (opcional)
    """
    if exception:
        logger.error(f"{error_msg}: {str(exception)}", exc_info=True)
    else:
        logger.error(error_msg)


def setup_from_config(config_path='config/config.json'):
    """
    Configura el logger desde el archivo de configuración.

    Args:
        config_path: Ruta al archivo de configuración

    Returns:
        Logger configurado
    """
    import json

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        logging_config = config.get('logging', {})

        # Mapeo de niveles de string a constantes
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        level = level_map.get(logging_config.get('level', 'INFO'), logging.INFO)
        log_file = logging_config.get('log_file', 'logs/bot.log')
        max_bytes = logging_config.get('max_bytes', 10*1024*1024)
        backup_count = logging_config.get('backup_count', 5)

        logger = setup_logger(
            name='TradingBot',
            log_file=log_file,
            level=level,
            max_bytes=max_bytes,
            backup_count=backup_count
        )

        logger.info("Logger configurado desde config.json")
        return logger

    except Exception as e:
        # Fallback a configuración por defecto
        logger = setup_logger()
        logger.warning(f"No se pudo cargar configuración de logging: {e}. Usando valores por defecto.")
        return logger


class TradingLogger:
    """
    Clase wrapper para facilitar el uso de logging en el bot.
    """

    def __init__(self, config_path='config/config.json'):
        """Inicializa el logger del bot."""
        self.main_logger = setup_from_config(config_path)
        self.trade_logger = setup_trade_logger()

    def info(self, msg):
        """Log de información general."""
        self.main_logger.info(msg)

    def warning(self, msg):
        """Log de advertencia."""
        self.main_logger.warning(msg)

    def error(self, msg, exception=None):
        """Log de error."""
        log_error(self.main_logger, msg, exception)

    def critical(self, msg):
        """Log de error crítico."""
        self.main_logger.critical(msg)

    def trade(self, trade_type, symbol, price, quantity, sl=None, tp=None, pnl=None):
        """Log de operación de trading."""
        log_trade(self.trade_logger, trade_type, symbol, price, quantity, sl, tp, pnl)

    def signal(self, signal_type, price, indicators):
        """Log de señal de trading."""
        log_signal(self.main_logger, signal_type, price, indicators)

    def separator(self):
        """Imprime una línea separadora en los logs."""
        self.main_logger.info("=" * 60)


if __name__ == "__main__":
    # Test básico del módulo
    print("=== Test de Logger ===\n")

    # Crear logger
    logger = TradingLogger()

    # Test de mensajes
    logger.separator()
    logger.info("Bot iniciado")
    logger.info("Conectando a Binance...")

    # Test de señal
    logger.signal(
        signal_type='COMPRA',
        price=60500.50,
        indicators={
            'RSI': 45.6,
            'MACD': 120.3,
            'EMA_21': 60450.0,
            'EMA_50': 60200.0
        }
    )

    # Test de trade
    logger.trade(
        trade_type='BUY',
        symbol='BTCUSDT',
        price=60500.50,
        quantity=0.00025,
        sl=60200.0,
        tp=60900.0
    )

    # Test de cierre
    logger.trade(
        trade_type='SELL',
        symbol='BTCUSDT',
        price=60900.0,
        quantity=0.00025,
        pnl=10.0
    )

    # Test de advertencia
    logger.warning("Balance bajo detectado")

    # Test de error
    try:
        raise ValueError("Error de prueba")
    except Exception as e:
        logger.error("Error al ejecutar orden", exception=e)

    logger.separator()
    logger.info("Test completado")

    print("\n✓ Logs guardados en logs/bot.log y logs/trades.log")
    print("✓ Test completado exitosamente")
