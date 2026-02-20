"""IBKRBroker — IB Gateway wrapper using ib_insync."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd
from ib_insync import IB, CFD, Contract, Forex, LimitOrder, MarketOrder, Order, StopOrder, Trade

from .config import IB_TO_PAIR, PAIR_TO_CFD, PAIR_TO_IB, LiveConfig

logger = logging.getLogger(__name__)


class IBKRBroker:
    """Wraps ib_insync for all IB Gateway interaction."""

    def __init__(self, config: LiveConfig) -> None:
        self.ib = IB()
        self.config = config
        self._contracts: dict[str, Contract] = {}       # Forex for data
        self._trade_contracts: dict[str, Contract] = {}  # CFD for orders
        self._bars: dict[str, object] = {}  # pair -> BarDataList
        self._reconnect_delay = 5.0
        self._max_reconnect_delay = 60.0

    # ── Connection ─────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to IB Gateway with exponential backoff retry."""
        delay = self._reconnect_delay
        while True:
            try:
                await self.ib.connectAsync(
                    host=self.config.ib_host,
                    port=self.config.ib_port,
                    clientId=self.config.ib_client_id,
                    readonly=False,
                )
                # Avoid handler accumulation on reconnect (#10)
                self.ib.disconnectedEvent -= self._on_disconnect
                self.ib.disconnectedEvent += self._on_disconnect
                logger.info(
                    "Connected to IB Gateway at %s:%s (clientId=%s)",
                    self.config.ib_host,
                    self.config.ib_port,
                    self.config.ib_client_id,
                )
                self._reconnect_delay = 5.0  # reset on success
                return
            except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as exc:
                logger.warning("IB connect failed (%s), retrying in %.0fs...", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

    async def disconnect(self) -> None:
        """Disconnect from IB Gateway."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    def _on_disconnect(self) -> None:
        logger.warning("IB Gateway disconnected — will reconnect on next heartbeat")

    # ── Contracts ──────────────────────────────────────────────────

    def get_contract(self, pair: str) -> Contract:
        """Get or create a Forex contract for market data (e.g. EUR_USD)."""
        if pair not in self._contracts:
            ib_symbol = PAIR_TO_IB[pair]
            contract = Forex(pair=ib_symbol, exchange="IDEALPRO")
            self._contracts[pair] = contract
        return self._contracts[pair]

    async def get_trade_contract(self, pair: str) -> Contract:
        """Get or create a qualified CFD contract for order placement.

        European IB accounts cannot trade leveraged spot forex (IDEALPRO).
        CFDs via SMART exchange are used instead for order execution.
        Contracts are qualified on first use and cached.
        """
        if pair not in self._trade_contracts:
            symbol, currency = PAIR_TO_CFD[pair]
            contract = CFD(symbol, currency=currency)
            details = await self.ib.reqContractDetailsAsync(contract)
            if details:
                self._trade_contracts[pair] = details[0].contract
                logger.info("Qualified CFD contract for %s: %s", pair, details[0].contract.localSymbol)
            else:
                # Fallback: set exchange manually
                contract.exchange = "SMART"
                self._trade_contracts[pair] = contract
                logger.warning("Could not qualify CFD for %s, using SMART", pair)
        return self._trade_contracts[pair]

    # ── Market Data ────────────────────────────────────────────────

    async def request_historical_bars(
        self,
        pair: str,
        duration: str = "30 D",
        bar_size: str = "1 hour",
    ) -> pd.DataFrame:
        """Request historical OHLC bars and return as DataFrame.

        Returns DataFrame with columns: open, high, low, close, volume
        and a UTC DatetimeIndex.
        """
        contract = self.get_contract(pair)
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=2,  # UTC datetime
        )
        return self._bars_to_dataframe(bars)

    async def subscribe_bars(self, pair: str, callback) -> None:
        """Subscribe to real-time bars with keepUpToDate.

        The callback receives (bars, hasNewBar) — when hasNewBar is True,
        the previous candle has finalized.
        Bar size and duration are determined by config.timeframe.
        """
        contract = self.get_contract(pair)
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=self.config.bar_duration,
            barSizeSetting=self.config.bar_size,
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=2,
            keepUpToDate=True,
        )
        self._bars[pair] = bars

        def on_update(new_bars, has_new_bar):
            asyncio.ensure_future(callback(pair, new_bars, has_new_bar))

        bars.updateEvent += on_update
        logger.info(
            "Subscribed to %s bars for %s (%d historical bars)",
            self.config.timeframe, pair, len(bars),
        )

    def get_live_bars(self, pair: str) -> pd.DataFrame | None:
        """Return current live bars as DataFrame, or None if not subscribed."""
        bars = self._bars.get(pair)
        if bars is None:
            return None
        return self._bars_to_dataframe(bars)

    # ── Orders ─────────────────────────────────────────────────────

    async def place_market_order(
        self, pair: str, direction: str, units: float
    ) -> Trade:
        """Place a market order via CFD. direction is 'long' or 'short'."""
        contract = await self.get_trade_contract(pair)
        action = "BUY" if direction == "long" else "SELL"
        order = MarketOrder(action, units, outsideRth=True, tif="GTC")
        trade = self.ib.placeOrder(contract, order)
        logger.info("Market %s %s %.0f units (CFD)", action, pair, units)
        return trade

    async def place_limit_order(
        self, pair: str, direction: str, units: float, price: float,
        oca_group: str | None = None,
    ) -> Trade:
        """Place a limit order via CFD (used for take-profit)."""
        contract = await self.get_trade_contract(pair)
        action = "BUY" if direction == "long" else "SELL"
        order = LimitOrder(action, units, price, outsideRth=True, tif="GTC")
        if oca_group:
            order.ocaGroup = oca_group
            order.ocaType = 1  # Cancel other orders in group on fill
        trade = self.ib.placeOrder(contract, order)
        logger.info("Limit %s %s %.0f units @ %.5f (CFD) oca=%s", action, pair, units, price, oca_group or "none")
        return trade

    async def place_stop_order(
        self, pair: str, direction: str, units: float, price: float,
        oca_group: str | None = None,
    ) -> Trade:
        """Place a stop order via CFD (used for stop-loss)."""
        contract = await self.get_trade_contract(pair)
        action = "BUY" if direction == "long" else "SELL"
        order = StopOrder(action, units, price, outsideRth=True, tif="GTC")
        if oca_group:
            order.ocaGroup = oca_group
            order.ocaType = 1  # Cancel other orders in group on fill
        trade = self.ib.placeOrder(contract, order)
        logger.info("Stop %s %s %.0f units @ %.5f (CFD) oca=%s", action, pair, units, price, oca_group or "none")
        return trade

    async def modify_order(self, trade: Trade, new_price: float) -> None:
        """Modify an existing order's price (for BE moves)."""
        order = trade.order
        if isinstance(order, StopOrder):
            order.auxPrice = new_price
        elif isinstance(order, LimitOrder):
            order.lmtPrice = new_price
        self.ib.placeOrder(trade.contract, order)
        logger.info("Modified order %s → %.5f", order.orderId, new_price)

    async def cancel_order(self, trade: Trade) -> None:
        """Cancel an open order."""
        self.ib.cancelOrder(trade.order)
        logger.info("Cancelled order %s", trade.order.orderId)

    async def cancel_all_orders(self) -> None:
        """Cancel ALL open orders across all client IDs (global cancel).

        Called on startup to clean up stale SL/TP orders from previous runs.
        """
        self.ib.reqGlobalCancel()
        await asyncio.sleep(2)
        remaining = len(self.ib.openOrders())
        if remaining:
            logger.warning("Global cancel: %d orders still open", remaining)
        else:
            logger.info("Global cancel: all stale orders cleared")

    # ── Account ────────────────────────────────────────────────────

    async def get_account_balance(self) -> float:
        """Return account net liquidation value in USD.

        For non-USD accounts (e.g. SEK), finds the NetLiquidation in the
        account's base currency and divides by the USD exchange rate
        (which IB reports as local_currency_per_USD).
        """
        accounts = self.ib.managedAccounts()
        if accounts:
            summary = await self.ib.accountSummaryAsync()

            # Try USD directly first
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == "USD":
                    val = float(item.value)
                    if val > 0:
                        logger.info("Account balance (USD): $%.2f", val)
                        return val

            # Non-USD account: find NetLiquidation in any currency + USD exchange rate
            nlv = None
            nlv_currency = None
            usd_rate = None  # local_currency per 1 USD

            for item in summary:
                if item.tag == "NetLiquidation" and item.currency not in ("BASE", "USD"):
                    val = float(item.value)
                    if val > 0:
                        nlv = val
                        nlv_currency = item.currency
                if item.tag == "ExchangeRate" and item.currency == "USD":
                    usd_rate = float(item.value)

            if nlv is not None and usd_rate is not None and usd_rate > 0:
                usd_value = nlv / usd_rate
                logger.info(
                    "Account balance: %.2f %s ÷ %.4f (SEK/USD) = $%.2f USD",
                    nlv, nlv_currency, usd_rate, usd_value,
                )
                return usd_value
            if nlv is not None:
                logger.warning(
                    "Got NLV %.2f %s but no USD rate — using config default",
                    nlv, nlv_currency,
                )

        # Fallback: cached account values
        for av in self.ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                val = float(av.value)
                if val > 0:
                    return val
            # Try non-USD cached values too
            if av.tag == "NetLiquidation" and av.currency not in ("BASE", "USD"):
                nlv = float(av.value)
                # Find USD rate in cached values
                for av2 in self.ib.accountValues():
                    if av2.tag == "ExchangeRate" and av2.currency == "USD":
                        rate = float(av2.value)
                        if rate > 0 and nlv > 0:
                            return nlv / rate

        logger.warning("Could not retrieve account balance, using config default")
        return self.config.starting_balance

    async def get_open_positions(self) -> dict[str, float]:
        """Return open positions as {pair: signed_units} (positive=long).

        Handles both Forex (IDEALPRO) and CFD (SMART) positions.
        """
        positions: dict[str, float] = {}
        for pos in self.ib.positions():
            # CFD positions: symbol=EUR, currency=USD → EUR_USD
            if pos.contract.secType == "CFD":
                pair_key = f"{pos.contract.symbol}_{pos.contract.currency}"
                if pair_key in PAIR_TO_IB:
                    positions[pair_key] = pos.position
                    continue
            # Forex positions: symbol=EUR, currency=USD
            symbol = pos.contract.symbol + pos.contract.currency
            pair = IB_TO_PAIR.get(symbol)
            if pair:
                positions[pair] = pos.position
        return positions

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _bars_to_dataframe(bars) -> pd.DataFrame:
        """Convert IB BarDataList to our standard OHLC DataFrame."""
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        records = []
        for bar in bars:
            records.append({
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume if bar.volume > 0 else 0,
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date").sort_index()
        # Remove duplicate indices (can happen with keepUpToDate)
        df = df[~df.index.duplicated(keep="last")]
        return df
