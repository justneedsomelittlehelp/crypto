"""Backtest metrics computation.

Takes a Portfolio (after run_backtest) and produces standard
trading performance metrics: returns, drawdown, Sharpe, etc.
"""

import numpy as np
import pandas as pd

from src.backtest.engine import Portfolio


def compute_metrics(portfolio: Portfolio, starting_capital: float) -> dict:
    """Compute full performance metrics from a finished backtest."""
    trades = portfolio.closed_trades

    if not trades:
        return {
            "n_trades": 0,
            "starting_capital": starting_capital,
            "final_equity": portfolio.equity,
            "total_return_pct": 0.0,
        }

    # Sort trades by exit date
    trades_sorted = sorted(trades, key=lambda t: t.exit_date)

    # Build trade-level dataframe for easier stats
    df = pd.DataFrame([
        {
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "size_dollars": t.size_dollars,
            "exposure_dollars": getattr(t, "exposure_dollars", t.size_dollars),
            "leverage": getattr(t, "leverage", 1.0),
            "pnl_dollars": t.pnl_dollars,
            "pnl_pct": t.pnl_pct,
            "exit_reason": t.exit_reason,
            "hold_bars": t.hold_bars,
            "fees": t.fees_total,
            "funding": getattr(t, "funding_total", 0.0),
        }
        for t in trades_sorted
    ])

    # Total return
    final_equity = portfolio.equity
    total_return_pct = (final_equity / starting_capital - 1) * 100

    # Per-trade stats
    n_trades = len(df)
    n_wins = int((df["pnl_dollars"] > 0).sum())
    n_losses = int((df["pnl_dollars"] < 0).sum())
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0
    avg_win_dollars = float(df.loc[df["pnl_dollars"] > 0, "pnl_dollars"].mean()) if n_wins > 0 else 0.0
    avg_loss_dollars = float(df.loc[df["pnl_dollars"] < 0, "pnl_dollars"].mean()) if n_losses > 0 else 0.0
    avg_win_pct = float(df.loc[df["pnl_pct"] > 0, "pnl_pct"].mean() * 100) if n_wins > 0 else 0.0
    avg_loss_pct = float(df.loc[df["pnl_pct"] < 0, "pnl_pct"].mean() * 100) if n_losses > 0 else 0.0
    avg_trade_pct = float(df["pnl_pct"].mean() * 100)
    total_fees = float(df["fees"].sum())
    total_funding = float(df["funding"].sum()) if "funding" in df.columns else 0.0

    # Hold time
    avg_hold_bars = float(df["hold_bars"].mean())
    avg_hold_days = avg_hold_bars / 24  # 1h bars

    # Time period
    first_date = df["entry_date"].min()
    last_date = df["exit_date"].max()
    span_days = (last_date - first_date).total_seconds() / 86400
    span_years = span_days / 365.25

    # Annualized return (compound)
    if span_years > 0:
        ann_return_pct = ((final_equity / starting_capital) ** (1 / span_years) - 1) * 100
    else:
        ann_return_pct = 0.0

    # Build equity curve from closed trades chronologically
    # Equity at trade close = starting_capital + cumulative pnl
    cum_pnl = df["pnl_dollars"].cumsum()
    equity_curve = starting_capital + cum_pnl

    # Drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = float(drawdown.min() * 100)

    # Sharpe (annualized)
    # Use per-trade returns; estimate trades per year
    if n_trades > 1 and span_years > 0:
        trades_per_year = n_trades / span_years
        per_trade_returns = df["pnl_pct"].values
        sharpe_per_trade = float(per_trade_returns.mean() / per_trade_returns.std()) if per_trade_returns.std() > 0 else 0.0
        sharpe_annualized = sharpe_per_trade * np.sqrt(trades_per_year)
    else:
        sharpe_per_trade = 0.0
        sharpe_annualized = 0.0
        trades_per_year = 0.0

    # Max consecutive losses
    is_loss = (df["pnl_dollars"] < 0).astype(int).values
    max_consec = 0
    cur = 0
    for x in is_loss:
        if x == 1:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    # Exit reason breakdown
    exit_counts = df["exit_reason"].value_counts().to_dict()

    # Monthly returns
    df["month"] = df["exit_date"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        pnl_dollars=("pnl_dollars", "sum"),
        n_trades=("pnl_dollars", "count"),
    ).reset_index()
    monthly["month"] = monthly["month"].astype(str)
    monthly_returns = monthly.to_dict(orient="records")

    return {
        # Capital
        "starting_capital": float(starting_capital),
        "final_equity": float(final_equity),
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(ann_return_pct, 2),

        # Trade counts
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": round(win_rate, 4),

        # Per-trade
        "avg_trade_pct": round(avg_trade_pct, 3),
        "avg_win_dollars": round(avg_win_dollars, 2),
        "avg_loss_dollars": round(avg_loss_dollars, 2),
        "avg_win_pct": round(avg_win_pct, 3),
        "avg_loss_pct": round(avg_loss_pct, 3),
        "total_fees": round(total_fees, 2),
        "total_funding": round(total_funding, 2),

        # Time
        "first_trade_date": str(first_date.date()),
        "last_trade_date": str(last_date.date()),
        "span_days": round(span_days, 1),
        "span_years": round(span_years, 2),
        "avg_hold_days": round(avg_hold_days, 2),
        "trades_per_year": round(trades_per_year, 1),

        # Risk
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "max_consec_losses": max_consec,
        "sharpe_per_trade": round(sharpe_per_trade, 3),
        "sharpe_annualized": round(sharpe_annualized, 3),

        # Exits
        "exit_reasons": exit_counts,

        # Engine stats
        "skipped_no_capital": portfolio.signals_skipped_no_capital,
        "skipped_pyramid": portfolio.signals_skipped_pyramid,

        # Monthly breakdown (compact, last 12 entries)
        "monthly_summary": monthly_returns[-12:],
        "n_months": len(monthly_returns),
    }
