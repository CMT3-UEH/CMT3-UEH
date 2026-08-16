"""Quản lý đầu tư với một tài sản rủi ro: phân bổ vốn, định cỡ vị thế, chiến lược giải ngân."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.metrics import (
    annual_return,
    annual_volatility,
    max_drawdown,
    sharpe_ratio,
)
from src.config import FEE_RATE, RISK_FREE_ANNUAL, TRADING_DAYS


# Đường phân bổ vốn
@dataclass
class AllocationResult:
    y_optimal: float                 # tỷ trọng tối ưu vào tài sản rủi ro
    risk_aversion: float
    expected_return: float           # của danh mục hoàn chỉnh
    volatility: float
    sharpe: float
    utility: float


def capital_allocation_line(
    mu_asset: float, sigma_asset: float, rf: float = RISK_FREE_ANNUAL,
    y_max: float = 1.5,
) -> pd.DataFrame:
    """Toạ độ CAL: mỗi điểm là một mức phân bổ y vào cổ phiếu, (1-y) vào tiền gửi."""
    y = np.linspace(0, y_max, 60)
    return pd.DataFrame({
        "y": y,
        "expected_return": rf + y * (mu_asset - rf),
        "volatility": y * sigma_asset,
    })


def optimal_allocation(
    mu_asset: float,
    sigma_asset: float,
    risk_aversion: float = 3.0,
    rf: float = RISK_FREE_ANNUAL,
    allow_leverage: bool = False,
) -> AllocationResult:
    """y* = (E[R] - rf) / (A * sigma^2) — tối đa hoá U = E[R] - 0.5*A*sigma^2."""
    y = (mu_asset - rf) / (risk_aversion * sigma_asset ** 2) if sigma_asset else 0.0
    if not allow_leverage:
        y = float(np.clip(y, 0.0, 1.0))
    er = rf + y * (mu_asset - rf)
    vol = y * sigma_asset
    utility = er - 0.5 * risk_aversion * vol ** 2
    sharpe = (er - rf) / vol if vol > 0 else float("nan")
    return AllocationResult(y, risk_aversion, er, vol, sharpe, utility)


# Kelly và định cỡ vị thế
def kelly_fraction(mu_excess: float, sigma: float) -> float:
    """f* = (mu - rf) / sigma^2 — tỷ trọng tối đa hoá tốc độ tăng trưởng log."""
    return float(mu_excess / sigma ** 2) if sigma else float("nan")


def position_size_by_risk(
    capital: float, price: float, stop_distance: float, risk_pct: float = 0.02
) -> dict[str, float]:
    """Số cổ phiếu nên mua nếu chỉ chấp nhận mất ``risk_pct`` vốn khi chạm cắt lỗ."""
    if stop_distance <= 0 or price <= 0:
        return {"shares": 0, "value": 0.0, "risk_amount": 0.0, "weight": 0.0}
    risk_amount = capital * risk_pct
    shares = int(risk_amount / stop_distance)
    value = shares * price
    return {
        "shares": shares,
        "value": value,
        "risk_amount": risk_amount,
        "weight": value / capital if capital else 0.0,
        "stop_price": price - stop_distance,
    }


# Backtest chiến lược giải ngân
def period_end_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Phiên giao dịch cuối cùng của mỗi kỳ (tháng, quý, năm)."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period(freq)).last().values)


def backtest_mixed_portfolio(
    asset_returns: pd.Series,
    weight: float,
    rf_annual: float = RISK_FREE_ANNUAL,
    rebalance: str = "M",
    fee: float = FEE_RATE,
) -> pd.Series:
    """NAV của danh mục gồm ``weight`` cổ phiếu và phần còn lại gửi tiết kiệm."""
    idx = asset_returns.index
    if not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError("Chuỗi lợi suất phải có chỉ mục thời gian duy nhất và tăng dần.")

    rf_d = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    marks = pd.Series(0, index=idx)
    if rebalance:
        marks.loc[period_end_dates(idx, rebalance)] = 1

    nav, w = 1.0, weight
    out = []
    for t, r in asset_returns.items():
        nav_risky = nav * w * (1 + r)
        nav_safe = nav * (1 - w) * (1 + rf_d)
        nav = nav_risky + nav_safe
        w = nav_risky / nav if nav else 0.0
        if marks.loc[t] == 1 and abs(w - weight) > 1e-9:
            nav -= nav * abs(w - weight) * fee
            w = weight
        out.append(nav)
    return pd.Series(out, index=idx, name=f"{weight:.0%} cổ phiếu")


def compare_allocations(
    asset_returns: pd.Series,
    weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    rf_annual: float = RISK_FREE_ANNUAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (bảng NAV theo thời gian, bảng chỉ tiêu hiệu quả) cho từng mức phân bổ."""
    navs, rows = {}, {}
    for w in weights:
        nav = backtest_mixed_portfolio(asset_returns, w, rf_annual)
        navs[nav.name] = nav
        r = nav.pct_change().dropna()
        rows[nav.name] = {
            "Lợi suất năm": annual_return(r),
            "Độ biến động": annual_volatility(r),
            "Sharpe": sharpe_ratio(r, rf_annual),
            "Sụt giảm tối đa": max_drawdown(r),
            "NAV cuối kỳ": nav.iloc[-1],
        }
    return pd.DataFrame(navs), pd.DataFrame(rows).T


def dca_vs_lumpsum(
    prices: pd.Series, monthly_amount: float = 10_000_000, fee: float = FEE_RATE
) -> pd.DataFrame:
    """So sánh bình quân giá (DCA) hằng tháng với mua một lần toàn bộ số vốn."""
    if prices.empty:
        raise ValueError("Chuỗi giá rỗng.")

    buy_dates = set(period_end_dates(prices.index, "M"))
    n_months = len(buy_dates)
    total_capital = monthly_amount * n_months

    # DCA: mua vào phiên giao dịch cuối mỗi tháng
    shares_dca, invested = 0.0, 0.0
    dca_curve, invested_curve = [], []
    for t, p in prices.items():
        if t in buy_dates:
            shares_dca += monthly_amount * (1 - fee) / p
            invested += monthly_amount
        dca_curve.append(shares_dca * p)
        invested_curve.append(invested)

    # Mua một lần: dồn toàn bộ vốn vào phiên đầu tiên
    shares_ls = total_capital * (1 - fee) / float(prices.iloc[0])

    return pd.DataFrame({
        "DCA hàng tháng": pd.Series(dca_curve, index=prices.index),
        "Mua một lần": pd.Series(shares_ls * prices.values, index=prices.index),
        "Vốn đã giải ngân (DCA)": pd.Series(invested_curve, index=prices.index),
        "Vốn đã giải ngân (mua một lần)": pd.Series(total_capital, index=prices.index),
    })
