"""Real trade execution utilities for HEX Trading Bot.

Handles PulseX swaps, gas estimation, and native PLS reserve management
when the bot operates in live (non-demo) mode.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Dict, List, Optional

from eth_account import Account
from eth_typing import ChecksumAddress
from web3 import Web3

from bot.config import Config, TOKENS

# Increase decimal precision for on-chain math (supports large token decimals)
getcontext().prec = 38

MAX_UINT256 = (1 << 256) - 1


@dataclass
class SwapResult:
    """Structured result for executed swaps."""

    tx_hash: str
    gas_used: int
    gas_price_wei: int
    fee_pls: Decimal
    fee_dai: Decimal
    direction: str
    amount_in: Decimal
    amount_out: Decimal
    metadata: Dict[str, Decimal]


class TradeExecutor:
    """Execute real swaps via PulseX router with gas + reserve safeguards."""

    def __init__(self, data_handler, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config: Config = data_handler.config
        self.asset_symbol = self.config.ASSET_SYMBOL
        self.bridge_symbol = self.config.BRIDGE_SYMBOL
        self.quote_symbol = self.config.QUOTE_SYMBOL
        self.w3: Web3 = data_handler.w3
        self.router = data_handler.router_contract
        self.dai_contract = data_handler.dai_contract
        self.hex_contract = data_handler.hex_contract
        self.wpls_contract = data_handler.wpls_contract
        self.quote_contract = self.dai_contract
        self.asset_contract = self.hex_contract
        self.bridge_contract = self.wpls_contract

        if not self.config.PRIVATE_KEY:
            raise ValueError("PRIVATE_KEY must be set for live trading execution")

        self.account = Account.from_key(self.config.PRIVATE_KEY)
        configured_address = self.config.WALLET_ADDRESS or self.account.address
        self.wallet_address: ChecksumAddress = Web3.to_checksum_address(configured_address)

        if self.wallet_address.lower() != self.account.address.lower():
            self.logger.warning(
                "Configured wallet address %s does not match private key derived address %s."
                " Using private key address.",
                self.wallet_address,
                self.account.address,
            )
            self.wallet_address = Web3.to_checksum_address(self.account.address)

        self.min_pls_reserve_dai: Decimal = Decimal(self.config.MIN_NATIVE_RESERVE_DAI)
        self.slippage: Decimal = Decimal(self.config.SLIPPAGE_TOLERANCE)
        if self.slippage <= 0 or self.slippage >= 1:
            self.slippage = Decimal("0.05")
        self.gas_price_multiplier: Decimal = Decimal(self.config.GAS_PRICE_MULTIPLIER)
        if self.gas_price_multiplier <= 0:
            self.gas_price_multiplier = Decimal("1.25")
        self.gas_limit_buffer: Decimal = Decimal(self.config.GAS_LIMIT_BUFFER)
        if self.gas_limit_buffer <= 1:
            self.gas_limit_buffer = Decimal("1.2")

        self.token_decimals: Dict[str, int] = {
            Web3.to_checksum_address(meta['address']).lower(): meta['decimals']
            for meta in TOKENS.values()
        }
        self.token_contracts: Dict[str, any] = {
            Web3.to_checksum_address(self.config.DAI_ADDRESS).lower(): self.dai_contract,
            Web3.to_checksum_address(self.config.HEX_ADDRESS).lower(): self.hex_contract,
            Web3.to_checksum_address(self.config.WPLS_ADDRESS).lower(): self.wpls_contract,
        }

    # ------------------------------------------------------------------
    # Balance helpers
    # ------------------------------------------------------------------

    def sync_balances(self) -> Dict[str, Decimal]:
        dai = self._get_token_balance(self.dai_contract, self.config.DAI_ADDRESS)
        hex_bal = self._get_token_balance(self.hex_contract, self.config.HEX_ADDRESS)
        pls = self._get_pls_balance()
        pls_price = self.get_bridge_price_in_quote()
        pls_dai = pls * pls_price
        return {
            'dai': dai,
            'hex': hex_bal,
            'pls': pls,
            'pls_dai': pls_dai,
        }

    def _get_token_balance(self, contract, token_address: str) -> Decimal:
        decimals = self._decimals_for(token_address)
        raw = contract.functions.balanceOf(self.wallet_address).call()
        return self._units_to_decimal(raw, decimals)

    def _get_pls_balance(self) -> Decimal:
        balance_wei = self.w3.eth.get_balance(self.wallet_address)
        return self._units_to_decimal(balance_wei, 18)

    # ------------------------------------------------------------------
    # Pricing utilities
    # ------------------------------------------------------------------

    def get_bridge_price_in_quote(self) -> Decimal:
        path = [
            Web3.to_checksum_address(self.config.WPLS_ADDRESS),
            Web3.to_checksum_address(self.config.DAI_ADDRESS),
        ]
        amount_in = self._decimal_to_units(Decimal('1'), 18)
        amounts = self.router.functions.getAmountsOut(amount_in, path).call()
        if not amounts or len(amounts) < 2:
            raise RuntimeError("Unable to fetch PLS→DAI rate from router")
        dai_units = Decimal(amounts[-1])
        return dai_units / Decimal(10 ** self._decimals_for(self.config.DAI_ADDRESS))

    # ------------------------------------------------------------------
    # Reserve management
    # ------------------------------------------------------------------

    def ensure_pls_reserve(self, target_dai: Optional[Decimal] = None) -> Optional[SwapResult]:
        target = Decimal(target_dai) if target_dai is not None else self.min_pls_reserve_dai
        if target <= 0:
            return None

        balances = self.sync_balances()
        if balances['pls_dai'] >= target:
            return None

        deficit = (target - balances['pls_dai']) * Decimal('1.05')  # add 5% buffer
        dai_balance = balances['dai']
        if dai_balance <= deficit:
            raise RuntimeError(
                f"Insufficient DAI to top up PLS reserve. Need {deficit:.4f} DAI, have {dai_balance:.4f} DAI"
            )

        self.logger.info(
            "PLS reserve below target (%.4f DAI). Swapping %.4f DAI → PLS for gas buffer",
            balances['pls_dai'],
            deficit,
        )
        return self._swap_tokens_for_native(
            amount_in=deficit,
            path=[self.config.DAI_ADDRESS, self.config.WPLS_ADDRESS],
        )

    # ------------------------------------------------------------------
    # Public trade helpers
    # ------------------------------------------------------------------

    def swap_dai_for_hex(self, amount_dai: Decimal) -> SwapResult:
        return self._swap_tokens_for_tokens(
            amount_in=amount_dai,
            path=[self.config.DAI_ADDRESS, self.config.WPLS_ADDRESS, self.config.HEX_ADDRESS],
            direction=f"{self.quote_symbol.lower()}->{self.asset_symbol.lower()}",
        )

    def swap_hex_for_dai(self, amount_hex: Decimal) -> SwapResult:
        return self._swap_tokens_for_tokens(
            amount_in=amount_hex,
            path=[self.config.HEX_ADDRESS, self.config.WPLS_ADDRESS, self.config.DAI_ADDRESS],
            direction=f"{self.asset_symbol.lower()}->{self.quote_symbol.lower()}",
        )

    # ------------------------------------------------------------------
    # Core swap execution
    # ------------------------------------------------------------------

    def _swap_tokens_for_tokens(self, amount_in: Decimal, path: List[str], direction: str) -> SwapResult:
        if amount_in <= 0:
            raise ValueError("amount_in must be positive for token swap")

        checksum_path = [Web3.to_checksum_address(addr) for addr in path]
        token_in = checksum_path[0]
        token_out = checksum_path[-1]
        decimals_in = self._decimals_for(token_in)
        decimals_out = self._decimals_for(token_out)

        amount_in_units = self._decimal_to_units(amount_in, decimals_in)
        if amount_in_units <= 0:
            raise ValueError("amount_in converts to zero units after decimal scaling")

        pre_in_balance = self._get_token_balance_units(token_in)
        pre_out_balance = self._get_token_balance_units(token_out)

        self._ensure_allowance(token_in, amount_in_units)

        expected_amounts = self.router.functions.getAmountsOut(amount_in_units, checksum_path).call()
        min_out_units = self._apply_slippage(expected_amounts[-1])
        if min_out_units <= 0:
            raise RuntimeError("Calculated minimum output is zero; aborting swap")

        tx_func = self.router.functions.swapExactTokensForTokens(
            amount_in_units,
            min_out_units,
            checksum_path,
            self.wallet_address,
            int(time.time()) + 600,
        )
        tx, tx_hash, receipt = self._build_sign_send(tx_func, value=0)

        post_in_balance = self._get_token_balance_units(token_in)
        post_out_balance = self._get_token_balance_units(token_out)

        actual_in_units = pre_in_balance - post_in_balance
        actual_out_units = post_out_balance - pre_out_balance
        if actual_in_units <= 0 or actual_out_units <= 0:
            self.logger.warning("Swap executed but balance delta is non-positive (in: %s, out: %s)", actual_in_units, actual_out_units)

        amount_in_decimal = self._units_to_decimal(actual_in_units, decimals_in)
        amount_out_decimal = self._units_to_decimal(actual_out_units, decimals_out)

        return self._compose_swap_result(
            tx_hash=tx_hash,
            receipt=receipt,
            transaction=tx,
            direction=direction,
            amount_in=amount_in_decimal,
            amount_out=amount_out_decimal,
            token_out_decimals=decimals_out,
        )

    def _swap_tokens_for_native(self, amount_in: Decimal, path: List[str]) -> SwapResult:
        if amount_in <= 0:
            raise ValueError("amount_in must be positive for token→PLS swap")

        checksum_path = [Web3.to_checksum_address(addr) for addr in path]
        token_in = checksum_path[0]
        decimals_in = self._decimals_for(token_in)

        amount_in_units = self._decimal_to_units(amount_in, decimals_in)
        if amount_in_units <= 0:
            raise ValueError("amount_in converts to zero units after decimal scaling")

        pre_in_balance = self._get_token_balance_units(token_in)
        pre_pls_balance = self._get_pls_balance_units()

        self._ensure_allowance(token_in, amount_in_units)

        expected_amounts = self.router.functions.getAmountsOut(amount_in_units, checksum_path).call()
        min_out_units = self._apply_slippage(expected_amounts[-1])
        if min_out_units <= 0:
            raise RuntimeError("Calculated minimum PLS output is zero; aborting swap")

        tx_func = self.router.functions.swapExactTokensForETH(
            amount_in_units,
            min_out_units,
            checksum_path,
            self.wallet_address,
            int(time.time()) + 600,
        )
        tx, tx_hash, receipt = self._build_sign_send(tx_func, value=0)

        post_in_balance = self._get_token_balance_units(token_in)
        post_pls_balance = self._get_pls_balance_units()

        actual_in_units = pre_in_balance - post_in_balance
        actual_out_units = post_pls_balance - pre_pls_balance
        amount_in_decimal = self._units_to_decimal(actual_in_units, decimals_in)
        amount_out_decimal = self._units_to_decimal(actual_out_units, 18)

        return self._compose_swap_result(
            tx_hash=tx_hash,
            receipt=receipt,
            transaction=tx,
            direction='token->pls',
            amount_in=amount_in_decimal,
            amount_out=amount_out_decimal,
            token_out_decimals=18,
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _compose_swap_result(self, tx_hash: str, receipt, transaction: Dict, direction: str,
                              amount_in: Decimal, amount_out: Decimal, token_out_decimals: int) -> SwapResult:
        gas_price_wei = transaction.get('gasPrice', self.w3.eth.gas_price)
        fee_pls = self._units_to_decimal(receipt.gasUsed * gas_price_wei, 18)
        fee_dai = fee_pls * self.get_pls_price_in_dai()
        metadata = {
            'gas_used': Decimal(receipt.gasUsed),
            'gas_price_gwei': Decimal(gas_price_wei) / Decimal(10 ** 9),
            'block_number': Decimal(receipt.blockNumber),
        }
        return SwapResult(
            tx_hash=tx_hash,
            gas_used=receipt.gasUsed,
            gas_price_wei=gas_price_wei,
            fee_pls=fee_pls,
            fee_dai=fee_dai,
            direction=direction,
            amount_in=amount_in,
            amount_out=amount_out,
            metadata=metadata,
        )

    def _build_sign_send(self, tx_func, value: int = 0):
        gas_price = int(Decimal(self.w3.eth.gas_price) * self.gas_price_multiplier)
        nonce = self.w3.eth.get_transaction_count(self.wallet_address)

        estimate_params = {
            'from': self.wallet_address,
            'value': value,
        }
        gas_estimate = tx_func.estimate_gas(estimate_params)
        gas_limit = int(Decimal(gas_estimate) * self.gas_limit_buffer)

        transaction = tx_func.build_transaction({
            'from': self.wallet_address,
            'value': value,
            'nonce': nonce,
            'gasPrice': gas_price,
            'gas': gas_limit,
        })

        signed = self.account.sign_transaction(transaction)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt.status != 1:
            raise RuntimeError(f"Transaction {tx_hash.hex()} failed with status {receipt.status}")
        tx_hash_hex = tx_hash.hex()
        self.logger.info(
            "Swap submitted tx=%s gas_used=%s gas_price=%s gwei",
            tx_hash_hex,
            receipt.gasUsed,
            gas_price / 1e9,
        )
        return transaction, tx_hash_hex, receipt

    def _ensure_allowance(self, token_address: ChecksumAddress, required_amount: int):
        token_lower = token_address.lower()
        contract = self.token_contracts.get(token_lower)
        if contract is None:
            raise ValueError(f"Unsupported token for swap allowance: {token_address}")
        allowance = contract.functions.allowance(self.wallet_address, self.config.PULSEX_ROUTER_V2).call()
        if allowance >= required_amount:
            return
        self.logger.info(
            "Approving router to spend %s (current allowance %s, required %s)",
            token_address,
            allowance,
            required_amount,
        )
        tx_func = contract.functions.approve(self.config.PULSEX_ROUTER_V2, MAX_UINT256)
        tx, tx_hash, receipt = self._build_sign_send(tx_func)
        if receipt.status != 1:
            raise RuntimeError(f"Approval transaction {tx_hash} failed with status {receipt.status}")

    def _apply_slippage(self, expected_units: int) -> int:
        if expected_units <= 0:
            return 0
        tolerance = (Decimal('1') - self.slippage)
        min_units = int(Decimal(expected_units) * tolerance)
        return max(min_units, 1)

    def _get_token_balance_units(self, token_address: ChecksumAddress) -> int:
        token_lower = token_address.lower()
        contract = self.token_contracts.get(token_lower)
        if contract is None:
            raise ValueError(f"Unknown token address {token_address}")
        return contract.functions.balanceOf(self.wallet_address).call()

    def _get_pls_balance_units(self) -> int:
        return self.w3.eth.get_balance(self.wallet_address)

    def _decimals_for(self, token_address: str) -> int:
        checksum = Web3.to_checksum_address(token_address).lower()
        decimals = self.token_decimals.get(checksum)
        if decimals is None:
            # Fetch on demand and cache
            contract = self.token_contracts.get(checksum)
            if contract is None:
                raise ValueError(f"Unknown token address {token_address}")
            decimals = contract.functions.decimals().call()
            self.token_decimals[checksum] = decimals
        return decimals

    @staticmethod
    def _decimal_to_units(amount: Decimal, decimals: int) -> int:
        factor = Decimal(10) ** decimals
        return int((amount * factor).to_integral_value())

    @staticmethod
    def _units_to_decimal(amount: int, decimals: int) -> Decimal:
        factor = Decimal(10) ** decimals
        return Decimal(amount) / factor
