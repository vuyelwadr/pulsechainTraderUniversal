"""Generic PulseChain trading bot configuration."""
import os
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ASSET_SYMBOL = "PDAI"
# PDAI address (user-specified) on PulseChain fork
DEFAULT_ASSET_ADDRESS = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
DEFAULT_ASSET_DECIMALS = 18

DEFAULT_BRIDGE_SYMBOL = "WPLS"
DEFAULT_WPLS_ADDRESS = "0xA1077a294dDE1B09bB078844df40758a5D0f9a27"
DEFAULT_BRIDGE_DECIMALS = 18

DEFAULT_QUOTE_SYMBOL = "DAI"
# Quote is same as asset (PDAI) - just different label
DEFAULT_DAI_ADDRESS = "0xefD766cCb38EaF1dfd701853BFCe31359239F305"
DEFAULT_QUOTE_DECIMALS = 18

# Core PulseX contracts
DEFAULT_PULSEX_FACTORY_V2 = "0x29eA7545DEf87022BAdc76323F373EA1e707C523"

class Config:
    """Configuration settings for the trading bot."""

    # Blockchain settings
    RPC_URL = os.getenv("RPC_URL", "https://rpc.pulsechain.com")
    _RPC_URLS_DEFAULT = [
        "https://rpc.pulsechainrpc.com",
        "https://pulsechain-rpc.publicnode.com",
        "https://rpc.pulsechain.com",
        "https://rpc-pulsechain.g4mm4.io",
        "https://rpc.owlracle.info/pulse/70d38ce1826c4a60bb2a8e05a6c8b20f",
        "https://evex.cloud/pulserpc",
    ]
    _ENV_RPC_URLS = os.getenv("RPC_URLS", "").strip()
    RPC_URLS = [u.strip() for u in _ENV_RPC_URLS.split(',') if u.strip()] or _RPC_URLS_DEFAULT
    CHAIN_ID = int(os.getenv("CHAIN_ID", "369"))

    # Token configuration
    ASSET_SYMBOL = os.getenv("ASSET_SYMBOL", DEFAULT_ASSET_SYMBOL).upper()
    ASSET_NAME = os.getenv("ASSET_NAME", ASSET_SYMBOL)
    ASSET_DECIMALS = int(os.getenv("ASSET_DECIMALS", str(DEFAULT_ASSET_DECIMALS)))
    ASSET_ADDRESS = os.getenv("ASSET_ADDRESS", DEFAULT_ASSET_ADDRESS)

    BRIDGE_SYMBOL = os.getenv("BRIDGE_SYMBOL", DEFAULT_BRIDGE_SYMBOL).upper()
    BRIDGE_DECIMALS = int(os.getenv("BRIDGE_DECIMALS", str(DEFAULT_BRIDGE_DECIMALS)))
    BRIDGE_ADDRESS = os.getenv("BRIDGE_ADDRESS", DEFAULT_WPLS_ADDRESS)

    QUOTE_SYMBOL = os.getenv("QUOTE_SYMBOL", DEFAULT_QUOTE_SYMBOL).upper()
    QUOTE_DECIMALS = int(os.getenv("QUOTE_DECIMALS", str(DEFAULT_QUOTE_DECIMALS)))
    QUOTE_ADDRESS = os.getenv("QUOTE_ADDRESS", DEFAULT_DAI_ADDRESS)

    # Backwards compatibility aliases (legacy code expects HEX/WPLS/DAI names)
    HEX_SYMBOL = ASSET_SYMBOL
    HEX_ADDRESS = ASSET_ADDRESS
    HEX_DECIMALS = ASSET_DECIMALS

    WPLS_SYMBOL = BRIDGE_SYMBOL
    WPLS_ADDRESS = BRIDGE_ADDRESS
    WPLS_DECIMALS = BRIDGE_DECIMALS

    DAI_SYMBOL = QUOTE_SYMBOL
    DAI_ADDRESS = QUOTE_ADDRESS
    DAI_DECIMALS = QUOTE_DECIMALS

    PULSEX_ROUTER_V2 = "0x165C3410fC91EF562C50559f7d2289fEbed552d9"
    PULSEX_FACTORY_V2 = os.getenv("PULSEX_FACTORY_V2", DEFAULT_PULSEX_FACTORY_V2)

    # Wallet settings
    PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
    WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")

    # Trading settings
    DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
    INITIAL_BALANCE = Decimal(os.getenv("INITIAL_BALANCE", "1000"))
    SLIPPAGE_TOLERANCE = Decimal(os.getenv("SLIPPAGE_TOLERANCE", "0.05"))
    MAX_TRADE_AMOUNT_PCT = Decimal(os.getenv("MAX_TRADE_AMOUNT_PCT", "0.1"))
    MIN_NATIVE_RESERVE_DAI = Decimal(os.getenv("MIN_NATIVE_RESERVE_DAI", "10"))
    GAS_PRICE_MULTIPLIER = Decimal(os.getenv("GAS_PRICE_MULTIPLIER", "1.25"))
    GAS_LIMIT_BUFFER = Decimal(os.getenv("GAS_LIMIT_BUFFER", "1.2"))
    GAS_FEE_PER_TRADE = Decimal(os.getenv("GAS_FEE_PER_TRADE", "0.01"))  # Estimated gas cost per trade in DAI (~$0.01 based on PulseChain gas prices)

    # Strategy settings
    MA_SHORT_PERIOD = int(os.getenv("MA_SHORT_PERIOD", "10"))
    MA_LONG_PERIOD = int(os.getenv("MA_LONG_PERIOD", "30"))

    # Data settings
    DATA_FETCH_INTERVAL = int(os.getenv("DATA_FETCH_INTERVAL", "60"))
    BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "365"))

    DATA_DIR = "data"
    HTML_DIR = "html_reports"

    OHLCV_RANGE = os.getenv("OHLCV_RANGE", "30d").lower()
    _OHLCV_FILE_MAP = {
        '30d': '{asset}_ohlcv_{quote}_30day_5m.csv',
        '90d': '{asset}_ohlcv_{quote}_90day_5m.csv',
        '1y': '{asset}_ohlcv_{quote}_365day_5m.csv',
        '365d': '{asset}_ohlcv_{quote}_365day_5m.csv',
    }

    @classmethod
    def data_path(cls, template: str) -> str:
        """Render a data file path using the active asset/quote symbols."""
        return os.path.join(
            cls.DATA_DIR,
            template.format(
            asset=cls.ASSET_SYMBOL.lower(),
            quote=cls.QUOTE_SYMBOL.lower(),
            ),
        )

    @classmethod
    def ohlcv_candidates(cls) -> list:
        pref = cls.OHLCV_RANGE if cls.OHLCV_RANGE in cls._OHLCV_FILE_MAP else '30d'
        order = {
            '30d': ['30d', '90d'],
            '90d': ['90d', '30d'],
            '1y': ['1y', '90d', '30d'],
        }
        seq = order.get(pref, ['30d', '90d'])
        asset = cls.ASSET_SYMBOL.lower()
        quote = cls.QUOTE_SYMBOL.lower()
        return [
            os.path.join(cls.DATA_DIR, cls._OHLCV_FILE_MAP[k].format(asset=asset, quote=quote))
            for k in seq
        ]

    @classmethod
    def resolve_ohlcv_path(cls) -> str:
        for path in cls.ohlcv_candidates():
            if os.path.exists(path):
                return path
        return ""

    @classmethod
    def validate(cls):
        if not cls.DEMO_MODE:
            if not cls.PRIVATE_KEY or not cls.WALLET_ADDRESS:
                raise ValueError("PRIVATE_KEY and WALLET_ADDRESS must be set for live trading")

        if cls.MA_SHORT_PERIOD >= cls.MA_LONG_PERIOD:
            raise ValueError("MA_SHORT_PERIOD must be less than MA_LONG_PERIOD")

        return True

TOKENS = {
    Config.ASSET_SYMBOL: {
        "address": Config.ASSET_ADDRESS,
        "decimals": Config.ASSET_DECIMALS,
        "symbol": Config.ASSET_SYMBOL,
    },
    Config.BRIDGE_SYMBOL: {
        "address": Config.BRIDGE_ADDRESS,
        "decimals": Config.BRIDGE_DECIMALS,
        "symbol": Config.BRIDGE_SYMBOL,
    },
    Config.QUOTE_SYMBOL: {
        "address": Config.QUOTE_ADDRESS,
        "decimals": Config.QUOTE_DECIMALS,
        "symbol": Config.QUOTE_SYMBOL,
    },
}

PULSEX_ROUTER_ABI = [
    {
        "inputs": [],
        "name": "factory",
        "outputs": [
            {"internalType": "address", "name": "", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"},
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "swapExactETHForTokens",
        "outputs": [
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"},
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "swapExactTokensForETH",
        "outputs": [
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"},
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "swapExactTokensForTokens",
        "outputs": [
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountOut", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"}
        ],
        "name": "getAmountsIn",
        "outputs": [
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"}
        ],
        "name": "getAmountsOut",
        "outputs": [
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "payable": False, "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "payable": False, "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "payable": False, "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "payable": False, "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "payable": False, "stateMutability": "view", "type": "function"},
    {
        "constant": False,
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function",
    },
]
