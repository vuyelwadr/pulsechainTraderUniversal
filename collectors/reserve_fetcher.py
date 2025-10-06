"""
Reserve Fetcher for PulseX Liquidity Pools
Fetches and caches pool reserves for accurate slippage calculations
"""
import time
from typing import Dict, Optional, Tuple
from decimal import Decimal
import logging

import os, sys
# Ensure repo root on sys.path so sibling packages import correctly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bot.config import Config
from collectors.rpc_load_balancer import RPCLoadBalancer

logger = logging.getLogger(__name__)

# Uniswap V2 Pair ABI for getReserves
PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"internalType": "uint112", "name": "reserve0", "type": "uint112"},
            {"internalType": "uint112", "name": "reserve1", "type": "uint112"},
            {"internalType": "uint32", "name": "blockTimestampLast", "type": "uint32"}
        ],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token0",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token1",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

class ReserveCache:
    """Caches pool reserves with TTL for performance"""

    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default
        self.cache: Dict[str, Dict] = {}
        self.ttl_seconds = ttl_seconds
        self.rpc_balancer = RPCLoadBalancer()

    def get_reserves(self, pair_address: str) -> Optional[Tuple[int, int]]:
        """
        Get cached reserves, or None if expired/stale

        Returns:
            (reserve0, reserve1) or None
        """
        if pair_address not in self.cache:
            return None

        entry = self.cache[pair_address]
        if time.time() - entry['timestamp'] > self.ttl_seconds:
            # Expired
            del self.cache[pair_address]
            return None

        return entry['reserves']

    def set_reserves(self, pair_address: str, reserve0: int, reserve1: int):
        """Cache reserves with current timestamp"""
        self.cache[pair_address] = {
            'reserves': (reserve0, reserve1),
            'timestamp': time.time()
        }

    def fetch_and_cache_reserves(self, pair_address: str) -> Optional[Tuple[int, int]]:
        """
        Fetch reserves from blockchain and cache them

        Returns:
            (reserve0, reserve1) or None if failed
        """
        try:
            # Get a Web3 instance from load balancer
            w3 = self.rpc_balancer.get_round_robin_endpoint().w3

            # Create pair contract
            pair_contract = w3.eth.contract(address=pair_address, abi=PAIR_ABI)

            # Get reserves
            reserves = pair_contract.functions.getReserves().call()
            reserve0, reserve1, _ = reserves

            # Cache the result
            self.set_reserves(pair_address, reserve0, reserve1)

            logger.info(f"Fetched reserves for {pair_address}: {reserve0}, {reserve1}")
            return (reserve0, reserve1)

        except Exception as e:
            logger.error(f"Failed to fetch reserves for {pair_address}: {e}")
            return None

class ReserveFetcher:
    """Fetches and manages pool reserves for trading pairs"""

    def __init__(self):
        self.config = Config()
        self.cache = ReserveCache()
        self.pair_addresses = {}  # Cache pair addresses

    def get_pair_address(self, token0: str, token1: str) -> Optional[str]:
        """
        Get pair address for token pair, with caching

        Args:
            token0: First token address
            token1: Second token address

        Returns:
            Pair address or None
        """
        # Sort addresses for consistent key
        key = tuple(sorted([token0.lower(), token1.lower()]))

        if key in self.pair_addresses:
            return self.pair_addresses[key]

        try:
            # Get factory contract
            w3 = self.cache.rpc_balancer.get_round_robin_endpoint().w3
            factory_contract = w3.eth.contract(
                address=self.config.PULSEX_FACTORY_V2,
                abi=[
                    {
                        "constant": True,
                        "inputs": [
                            {"internalType": "address", "name": "tokenA", "type": "address"},
                            {"internalType": "address", "name": "tokenB", "type": "address"}
                        ],
                        "name": "getPair",
                        "outputs": [{"internalType": "address", "name": "pair", "type": "address"}],
                        "payable": False,
                        "stateMutability": "view",
                        "type": "function"
                    }
                ]
            )

            pair_address = factory_contract.functions.getPair(token0, token1).call()

            if pair_address and int(pair_address, 16) != 0:
                pair_address = w3.to_checksum_address(pair_address)
                self.pair_addresses[key] = pair_address
                return pair_address

        except Exception as e:
            logger.error(f"Failed to get pair address for {token0}/{token1}: {e}")

        return None

    def get_reserves_for_path(self, path: list) -> Dict[str, Tuple[int, int]]:
        """
        Get reserves for all pairs in a trading path

        Args:
            path: List of token addresses [token0, token1, token2, ...]

        Returns:
            Dict mapping pair_address -> (reserve0, reserve1)
        """
        reserves = {}

        # Get reserves for each consecutive pair
        for i in range(len(path) - 1):
            token0 = path[i]
            token1 = path[i + 1]

            pair_address = self.get_pair_address(token0, token1)
            if not pair_address:
                logger.warning(f"No pair found for {token0}/{token1}")
                continue

            # Try cache first
            pair_reserves = self.cache.get_reserves(pair_address)
            if pair_reserves is None:
                # Fetch and cache
                pair_reserves = self.cache.fetch_and_cache_reserves(pair_address)

            if pair_reserves:
                reserves[pair_address] = pair_reserves

        return reserves

    def get_all_trading_reserves(self) -> Dict[str, Tuple[int, int]]:
        """
        Get reserves for all configured trading pairs

        Returns:
            Dict mapping pair_address -> (reserve0, reserve1)
        """
        # Define trading paths
        paths = [
            [self.config.DAI_ADDRESS, self.config.WPLS_ADDRESS],  # DAI/WPLS
            [self.config.ASSET_ADDRESS, self.config.WPLS_ADDRESS],  # PDAI/WPLS
            [self.config.ASSET_ADDRESS, self.config.DAI_ADDRESS],   # PDAI/DAI (direct if exists)
        ]

        all_reserves = {}
        for path in paths:
            path_reserves = self.get_reserves_for_path(path)
            all_reserves.update(path_reserves)

        return all_reserves

# Global instance for easy access
_reserve_fetcher = None

def get_reserve_fetcher() -> ReserveFetcher:
    """Get global reserve fetcher instance"""
    global _reserve_fetcher
    if _reserve_fetcher is None:
        _reserve_fetcher = ReserveFetcher()
    return _reserve_fetcher

def fetch_trading_reserves() -> Dict[str, Tuple[int, int]]:
    """
    Convenience function to fetch all trading reserves
    Call this once at optimizer startup
    """
    fetcher = get_reserve_fetcher()
    return fetcher.get_all_trading_reserves()