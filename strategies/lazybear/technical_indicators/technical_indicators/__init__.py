"""LazyBear technical indicators package"""
from .strategy_151_custom_bollinger_bands_width import CustomBollingerBandsWidthStrategy
from .strategy_038_wow import Strategy038Wow
from .strategy_041_price_volume_trend import Strategy041PriceVolumeTrend
from .strategy_042_on_balance_volume_oscillator import Strategy042OnBalanceVolumeOscillator

__all__ = [
    'CustomBollingerBandsWidthStrategy',
    'Strategy038Wow',
    'Strategy041PriceVolumeTrend', 
    'Strategy042OnBalanceVolumeOscillator'
]