"""Sequential portfolio backtest engine.

Simulates trading the v6-prime model's signals through time with realistic:
  - Capital constraints (no infinite money)
  - Position sizing (% of available capital)
  - Reserve buffer (untouchable safety capital)
  - Fees (taker entry, maker TP / taker SL)
  - Slippage (per side, applied to entry and exit)
  - Max hold time (timeout exit)
  - Pyramiding control (stack vs skip on same-direction overlap)

The engine processes predictions chronologically and maintains:
  - Cash balance
  - Open positions list
  - Closed trades log
  - Equity curve

Results include all trade-level metrics needed for evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
@dataclass
class BacktestConfig:
    """All trading rules and parameters for one backtest run."""

    # Capital
    starting_capital: float = 5000.0
    reserve_pct: float = 0.30          # 30% reserve, untouchable
    position_size_pct: float = 0.20    # 20% of available per trade (when fixed)
    sizing_mode: str = "fixed_pct"     # "fixed_pct" or "dynamic" or "fixed_100"

    # Execution
    entry_at: str = "close"            # "close" of signal bar (5min latency assumption)
    fee_taker: float = 0.0026          # Kraken taker
    fee_maker: float = 0.0016          # Kraken maker
    slippage_per_side: float = 0.0005  # 0.05% per side

    # Position management
    max_hold_bars: int = 14 * 24       # 14 days at 1h
    allow_pyramiding: bool = False     # Stack vs skip same-direction
    direction: str = "long"            # "long" or "short" or "both"

    # Filter (applied to predictions before considering)
    min_confidence: float = 0.50       # sigmoid > this to consider
    min_asymmetry: float = 0.0         # tp_pct / sl_pct > this

    # Leverage
    leverage: float = 1.0              # 1.0 = spot/no leverage, N = N× exposure
    funding_rate_per_8h: float = 0.0001  # 0.01% per 8h funding (typical avg)
    liquidation_threshold: float = 0.95  # Liquidate at 95% loss of margin


# ═══════════════════════════════════════════════════════════════════
# Position
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Position:
    entry_bar: int
    entry_date: pd.Timestamp
    entry_price: float
    size_dollars: float        # Margin locked (the dollars you commit)
    exposure_dollars: float    # Actual notional exposure = size × leverage
    btc_amount: float          # BTC controlled by exposure (not just margin)
    direction: int             # +1 long, -1 short
    tp_level: float            # Absolute price for TP
    sl_level: float            # Absolute price for SL
    tp_pct: float              # Original tp_pct (informational)
    sl_pct: float              # Original sl_pct (informational)
    leverage: float            # Leverage used (1.0 = spot)
    entry_fee: float           # $ fee paid on entry


# ═══════════════════════════════════════════════════════════════════
# Trade record (closed)
# ═══════════════════════════════════════════════════════════════════
@dataclass
class ClosedTrade:
    entry_bar: int
    exit_bar: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    size_dollars: float
    exposure_dollars: float
    leverage: float
    direction: int
    tp_pct: float
    sl_pct: float
    exit_reason: str           # "tp", "sl", "timeout", "liquidated"
    pnl_dollars: float         # Net P&L after fees, slippage, funding
    pnl_pct: float             # P&L as % of margin (not exposure)
    fees_total: float
    funding_total: float       # Cumulative funding paid
    slippage_total: float
    hold_bars: int


# ═══════════════════════════════════════════════════════════════════
# Portfolio
# ═══════════════════════════════════════════════════════════════════
class Portfolio:
    """Stateful portfolio that processes signals chronologically."""

    def __init__(self, config: BacktestConfig):
        self.cfg = config
        self.cash = config.starting_capital
        self.equity = config.starting_capital
        self.open_positions: list[Position] = []
        self.closed_trades: list[ClosedTrade] = []
        self.equity_history: list[tuple[pd.Timestamp, float]] = []  # (date, equity)
        self.signals_skipped_no_capital = 0
        self.signals_skipped_pyramid = 0
        self.signals_skipped_filter = 0

    def available_capital(self) -> float:
        """Capital available for new positions = (equity × (1 - reserve_pct)) - committed."""
        target_available = self.equity * (1.0 - self.cfg.reserve_pct)
        committed = sum(p.size_dollars for p in self.open_positions)
        return max(0.0, target_available - committed)

    def compute_position_size(self) -> float:
        """Decide how big the next position should be."""
        avail = self.available_capital()

        if self.cfg.sizing_mode == "fixed_pct":
            # Fixed % of equity-based available
            target_available = self.equity * (1.0 - self.cfg.reserve_pct)
            size = target_available * self.cfg.position_size_pct
            return min(size, avail)  # Can't exceed available cash

        elif self.cfg.sizing_mode == "dynamic":
            # Divide all available by (number of expected concurrent + 1)
            n_open = len(self.open_positions)
            divisor = n_open + 1
            return avail / divisor

        elif self.cfg.sizing_mode == "fixed_100":
            return avail

        else:
            return avail * self.cfg.position_size_pct

    def update_equity(self, current_price: float):
        """Mark-to-market: equity = cash + margin value of open positions.

        For leveraged positions, the margin value = original margin + unrealized P&L.
        Unrealized P&L = (current_price - entry_price) × btc_amount × direction.
        """
        position_value = 0.0
        for p in self.open_positions:
            unrealized_pnl = (current_price - p.entry_price) * p.btc_amount * p.direction
            position_value += p.size_dollars + unrealized_pnl
        self.equity = self.cash + position_value

    def try_open_position(
        self, bar_idx: int, date: pd.Timestamp, price: float,
        direction: int, tp_pct: float, sl_pct: float,
    ) -> bool:
        """Attempt to open a new position. Returns True if opened."""

        # Pyramiding check: skip if same-direction position already open and pyramiding off
        if not self.cfg.allow_pyramiding:
            for p in self.open_positions:
                if p.direction == direction:
                    self.signals_skipped_pyramid += 1
                    return False

        # Compute size (margin)
        size_dollars = self.compute_position_size()
        if size_dollars < 10.0:  # Minimum $10 position to avoid dust
            self.signals_skipped_no_capital += 1
            return False

        # Leverage: exposure = margin × leverage
        leverage = self.cfg.leverage
        exposure_dollars = size_dollars * leverage

        # Apply entry slippage
        if direction == 1:
            entry_price = price * (1 + self.cfg.slippage_per_side)
        else:
            entry_price = price * (1 - self.cfg.slippage_per_side)

        # Entry fee (taker — market order). Fee is on EXPOSURE not margin.
        entry_fee = exposure_dollars * self.cfg.fee_taker

        # BTC amount controlled by exposure (not just margin)
        net_exposure = exposure_dollars - entry_fee
        btc_amount = net_exposure / entry_price

        # Compute TP/SL levels (in absolute price space, unchanged by leverage)
        if direction == 1:
            tp_level = entry_price * (1 + tp_pct)
            sl_level = entry_price * (1 - sl_pct)
        else:
            tp_level = entry_price * (1 - tp_pct)
            sl_level = entry_price * (1 + sl_pct)

        position = Position(
            entry_bar=bar_idx,
            entry_date=date,
            entry_price=entry_price,
            size_dollars=size_dollars,
            exposure_dollars=exposure_dollars,
            btc_amount=btc_amount,
            direction=direction,
            tp_level=tp_level,
            sl_level=sl_level,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            leverage=leverage,
            entry_fee=entry_fee,
        )
        self.open_positions.append(position)
        self.cash -= size_dollars  # Lock margin (not exposure)
        return True

    def try_close_positions(self, bar_idx: int, date: pd.Timestamp, price: float):
        """Check if any open positions hit TP/SL/timeout. Close those that did."""
        still_open = []
        for p in self.open_positions:
            exit_reason = None
            exit_price = price

            # Liquidation check (for leveraged positions)
            # Unrealized loss in dollars; if it exceeds liquidation_threshold × margin, liquidate
            unrealized_pnl = (price - p.entry_price) * p.btc_amount * p.direction
            if unrealized_pnl < -p.size_dollars * self.cfg.liquidation_threshold:
                exit_reason = "liquidated"
                # Liquidation price = the price where loss = threshold × margin
                # solve: (liq_price - entry) × btc × dir = -threshold × margin
                threshold_loss = -self.cfg.liquidation_threshold * p.size_dollars
                liq_price = p.entry_price + (threshold_loss / (p.btc_amount * p.direction))
                exit_price = liq_price

            # Check TP/SL
            if exit_reason is None:
                if p.direction == 1:
                    if price >= p.tp_level:
                        exit_reason = "tp"
                        exit_price = p.tp_level
                    elif price <= p.sl_level:
                        exit_reason = "sl"
                        exit_price = p.sl_level
                else:
                    if price <= p.tp_level:
                        exit_reason = "tp"
                        exit_price = p.tp_level
                    elif price >= p.sl_level:
                        exit_reason = "sl"
                        exit_price = p.sl_level

            # Check timeout
            if exit_reason is None and (bar_idx - p.entry_bar) >= self.cfg.max_hold_bars:
                exit_reason = "timeout"
                exit_price = price

            if exit_reason is None:
                still_open.append(p)
                continue

            # Close the position
            self._close_position(p, bar_idx, date, exit_price, exit_reason)

        self.open_positions = still_open

    def _close_position(
        self, p: Position, bar_idx: int, date: pd.Timestamp,
        exit_price: float, exit_reason: str,
    ):
        """Compute P&L on leveraged position, deduct fees + funding + slippage."""
        # Apply exit slippage (opposite direction from entry slippage)
        if p.direction == 1:
            actual_exit_price = exit_price * (1 - self.cfg.slippage_per_side)
        else:
            actual_exit_price = exit_price * (1 + self.cfg.slippage_per_side)

        # Exit fee rate depends on exit type
        if exit_reason == "tp":
            exit_fee_rate = self.cfg.fee_maker      # TP via limit order
        else:
            exit_fee_rate = self.cfg.fee_taker      # SL/timeout/liquidation = market

        # Gross P&L on exposure (leveraged amount)
        # = (exit - entry) × btc_amount × direction
        # btc_amount already represents the leveraged BTC controlled
        gross_pnl = (actual_exit_price - p.entry_price) * p.btc_amount * p.direction

        # Exit fee on exit notional value
        exit_notional = p.btc_amount * actual_exit_price
        exit_fee = exit_notional * exit_fee_rate

        # Funding cost (for leveraged perp positions)
        # Funding paid every 8h (8 bars at 1h) on the FULL exposure
        hold_bars = bar_idx - p.entry_bar
        funding_periods = hold_bars / 8.0  # Number of 8h periods held
        funding_total = p.exposure_dollars * self.cfg.funding_rate_per_8h * funding_periods

        # Net P&L = gross P&L - exit fee - funding (entry fee already deducted from btc_amount)
        # Actually, entry fee was deducted when computing btc_amount, so it's already implicit
        # in btc_amount × prices. But for accounting clarity we track it separately.
        net_pnl = gross_pnl - exit_fee - funding_total

        # If liquidated, cap net_pnl at -margin (can't lose more than margin)
        if exit_reason == "liquidated":
            net_pnl = max(net_pnl, -p.size_dollars)

        # P&L as % of margin (not exposure) — this is the "return on capital deployed"
        pnl_pct = net_pnl / p.size_dollars

        slippage_total = p.exposure_dollars * self.cfg.slippage_per_side * 2  # both sides
        fees_total = p.entry_fee + exit_fee

        trade = ClosedTrade(
            entry_bar=p.entry_bar,
            exit_bar=bar_idx,
            entry_date=p.entry_date,
            exit_date=date,
            entry_price=p.entry_price,
            exit_price=actual_exit_price,
            size_dollars=p.size_dollars,
            exposure_dollars=p.exposure_dollars,
            leverage=p.leverage,
            direction=p.direction,
            tp_pct=p.tp_pct,
            sl_pct=p.sl_pct,
            exit_reason=exit_reason,
            pnl_dollars=net_pnl,
            pnl_pct=pnl_pct,
            fees_total=fees_total,
            funding_total=funding_total,
            slippage_total=slippage_total,
            hold_bars=hold_bars,
        )
        self.closed_trades.append(trade)
        # Return margin + net P&L to cash
        self.cash += p.size_dollars + net_pnl
        self.equity_history.append((date, self.cash + sum(
            pp.size_dollars for pp in self.open_positions
        )))


# ═══════════════════════════════════════════════════════════════════
# Engine entry point
# ═══════════════════════════════════════════════════════════════════
def run_backtest(
    dates: np.ndarray,           # (N,) np.datetime64 or pd.Timestamp
    close_prices: np.ndarray,    # (N,) close price at each bar
    probs: np.ndarray,           # (N,) sigmoid output, 0-1
    tp_pct: np.ndarray,          # (N,) per-sample TP %
    sl_pct: np.ndarray,          # (N,) per-sample SL %
    config: BacktestConfig,
) -> tuple[Portfolio, dict]:
    """Run a backtest over the predictions.

    For each bar:
      1. Check open positions for exit
      2. Check if signal fires (probs > min_confidence AND tp/sl > min_asymmetry)
      3. If signal fires AND capital available, open new position
    """
    portfolio = Portfolio(config)

    # Pre-compute filter mask
    safe_ratio = tp_pct / np.clip(sl_pct, 1e-8, None)
    asymmetry_mask = safe_ratio > config.min_asymmetry

    if config.direction == "long":
        signal_mask = (probs > config.min_confidence) & asymmetry_mask
        directions = np.full(len(probs), 1, dtype=np.int8)
    elif config.direction == "short":
        signal_mask = (probs < (1 - config.min_confidence)) & asymmetry_mask
        directions = np.full(len(probs), -1, dtype=np.int8)
    else:  # both
        long_signal = (probs > config.min_confidence) & asymmetry_mask
        short_signal = (probs < (1 - config.min_confidence)) & asymmetry_mask
        signal_mask = long_signal | short_signal
        directions = np.where(probs > 0.5, 1, -1).astype(np.int8)

    n_signals = int(signal_mask.sum())

    # Iterate chronologically
    for i in range(len(dates)):
        date = pd.Timestamp(dates[i])
        price = float(close_prices[i])

        # Mark to market
        portfolio.update_equity(price)

        # Try to close existing positions
        portfolio.try_close_positions(i, date, price)

        # Try to open new position if signal
        if signal_mask[i]:
            portfolio.try_open_position(
                bar_idx=i,
                date=date,
                price=price,
                direction=int(directions[i]),
                tp_pct=float(tp_pct[i]),
                sl_pct=float(sl_pct[i]),
            )

    # Force-close any remaining open positions at the last price
    if portfolio.open_positions:
        final_date = pd.Timestamp(dates[-1])
        final_price = float(close_prices[-1])
        for p in list(portfolio.open_positions):
            portfolio._close_position(p, len(dates) - 1, final_date, final_price, "end_of_test")
        portfolio.open_positions = []

    portfolio.update_equity(float(close_prices[-1]))

    summary = {
        "n_signals": n_signals,
        "n_trades_executed": len(portfolio.closed_trades),
        "skipped_no_capital": portfolio.signals_skipped_no_capital,
        "skipped_pyramid": portfolio.signals_skipped_pyramid,
    }
    return portfolio, summary
