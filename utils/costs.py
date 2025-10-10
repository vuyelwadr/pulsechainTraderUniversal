"""Centralised trade cost configuration."""

from decimal import Decimal

# Hard-coded per-trade fee rate (15 bps = 1.5%).
TRADE_FEE_RATE = Decimal("0.015")


def apply_buy_fee(quote_amount: Decimal) -> Decimal:
    """
    Apply the configured fee to a quote-side trade amount (buy).

    Returns the quote amount remaining after deducting fees.
    """
    return quote_amount * (Decimal("1") - TRADE_FEE_RATE)


def apply_sell_fee(quote_amount: Decimal) -> Decimal:
    """
    Apply the configured fee to the proceeds of a sell trade.

    Returns the quote amount remaining after deducting fees.
    """
    return quote_amount * (Decimal("1") - TRADE_FEE_RATE)


def fee_amount(quote_amount: Decimal) -> Decimal:
    """Return the absolute fee charged on the given quote currency amount."""
    return quote_amount * TRADE_FEE_RATE
