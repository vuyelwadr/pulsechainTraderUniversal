# Compatibility wrapper after repo reorg
from bot.pulsechain_trading_bot import *  # noqa: F401,F403

if __name__ == "__main__":
    from bot.pulsechain_trading_bot import main as _main

    _main()
