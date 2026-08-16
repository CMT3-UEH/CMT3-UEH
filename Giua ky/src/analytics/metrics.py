"""Các độ đo hiệu quả đầu tư và rủi ro."""

import numpy as np
import pandas as pd

from src.config import RISK_FREE_ANNUAL, TRADING_DAYS


# Lợi suất
def to_returns(prices: pd.Series, log: bool = False) -> pd.Series:
    """Chuỗi lợi suất theo phiên. ``log=True`` cho lợi suất logarit."""
    prices = prices.astype(float)
    r = np.log(prices / prices.shift(1)) if log else prices.pct_change()
    return r.dropna()


def cumulative_return(returns: pd.Series) -> pd.Series:
    """Đường tăng trưởng 1 đồng vốn (NAV)."""
    return (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def annual_return(returns: pd.Series) -> float:
    """CAGR suy ra từ chuỗi lợi suất ngày."""
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return float("nan")
    return growth ** (TRADING_DAYS / n) - 1.0


def annual_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def daily_rf(rf_annual: float = RISK_FREE_ANNUAL) -> float:
    """Quy đổi lãi suất phi rủi ro năm về một phiên (lãi kép)."""
    return (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0


# Tỷ số hiệu quả điều chỉnh rủi ro
def sharpe_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    excess = returns - daily_rf(rf_annual)
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    excess = returns - daily_rf(rf_annual)
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def drawdown_series(returns: pd.Series) -> pd.Series:
    nav = cumulative_return(returns)
    return nav / nav.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("nan")
    return annual_return(returns) / mdd


def hit_ratio(returns: pd.Series) -> float:
    """Tỷ lệ phiên tăng."""
    if len(returns) == 0:
        return float("nan")
    return float((returns > 0).mean())


# Rủi ro đuôi
def var_historical(returns: pd.Series, level: float = 0.95) -> float:
    """VaR theo phương pháp lịch sử (trả về số âm = mức lỗ)."""
    return float(np.percentile(returns, (1 - level) * 100))


def var_parametric(returns: pd.Series, level: float = 0.95) -> float:
    """VaR tham số giả định phân phối chuẩn."""
    from scipy.stats import norm

    mu, sd = returns.mean(), returns.std(ddof=1)
    return float(mu + norm.ppf(1 - level) * sd)


def cvar_historical(returns: pd.Series, level: float = 0.95) -> float:
    """CVaR / Expected Shortfall: lỗ trung bình khi đã vượt ngưỡng VaR."""
    var = var_historical(returns, level)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else float("nan")


# Chỉ tiêu có đơn vị phần trăm, so khớp chính xác theo tên để đổi nhãn là lỗi lộ ra ngay
PCT_METRICS = frozenset({
    "Lợi suất tích luỹ",
    "Lợi suất năm (CAGR)",
    "Độ biến động năm",
    "Sụt giảm tối đa",
    "Tỷ lệ phiên tăng",
    "VaR 95% (ngày)",
    "CVaR 95% (ngày)",
})


def summary_table(
    returns: pd.Series,
    rf_annual: float = RISK_FREE_ANNUAL,
    label: str = "Tài sản",
) -> pd.DataFrame:
    rows = {
        "Lợi suất tích luỹ": total_return(returns),
        "Lợi suất năm (CAGR)": annual_return(returns),
        "Độ biến động năm": annual_volatility(returns),
        "Tỷ số Sharpe": sharpe_ratio(returns, rf_annual),
        "Tỷ số Sortino": sortino_ratio(returns, rf_annual),
        "Sụt giảm tối đa": max_drawdown(returns),
        "Tỷ số Calmar": calmar_ratio(returns),
        "Tỷ lệ phiên tăng": hit_ratio(returns),
        "VaR 95% (ngày)": var_historical(returns, 0.95),
        "CVaR 95% (ngày)": cvar_historical(returns, 0.95),
        "Độ lệch (Skewness)": float(returns.skew()),
        "Độ nhọn (Kurtosis)": float(returns.kurtosis()),
    }
    return pd.DataFrame({label: rows})


def rolling_metric(
    returns: pd.Series, window: int = 126, func: str = "sharpe",
    rf_annual: float = RISK_FREE_ANNUAL,
) -> pd.Series:
    """Độ đo trượt theo cửa sổ (mặc định 126 phiên ~ 6 tháng)."""
    if func == "sharpe":
        f = lambda x: sharpe_ratio(pd.Series(x), rf_annual)   # noqa: E731
    elif func == "vol":
        f = lambda x: pd.Series(x).std(ddof=1) * np.sqrt(TRADING_DAYS)  # noqa: E731
    else:
        raise ValueError(f"func không hỗ trợ: {func}")
    return returns.rolling(window).apply(f, raw=True)


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Bảng lợi suất theo tháng × năm (%), kèm cột cả năm."""
    monthly = (1 + returns).resample("ME").prod() - 1
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = [f"T{m}" for m in pivot.columns]
    yearly = (1 + returns).resample("YE").prod() - 1
    pivot["Cả năm"] = yearly.values[: len(pivot)]
    return pivot
