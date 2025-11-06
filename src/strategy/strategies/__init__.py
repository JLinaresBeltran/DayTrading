"""
Módulo de estrategias de trading.
Cada estrategia está en su propio archivo para mejor organización.
"""

from .ema_cross import generar_señales
from .triple_layer import generar_senales_triple_capa
from .momentum import generar_senales_momentum_v1
from .hybrid import generar_senales_hibrido_v1
from .short_only import generar_senales_bajista_v1
from .advanced import generar_señales_avanzadas, generar_señales_con_filtro_tendencia

__all__ = [
    'generar_señales',
    'generar_senales_triple_capa',
    'generar_senales_momentum_v1',
    'generar_senales_hibrido_v1',
    'generar_senales_bajista_v1',
    'generar_señales_avanzadas',
    'generar_señales_con_filtro_tendencia',
]
