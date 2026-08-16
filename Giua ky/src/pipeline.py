"""Lớp tổng hợp: nạp dữ liệu đã xử lý và chạy toàn bộ phân tích một lần."""

from dataclasses import dataclass, field

import pandas as pd

from src.analytics import allocation as AL
from src.analytics import apt as APT
from src.analytics import capm as CAPM
from src.analytics import metrics as M
from src.analytics import montecarlo as MC
from src.analytics import technical as TA
from src.config import (
    BENCHMARK,
    MACRO_TICKERS,
    PROCESSED_DIR,
    RAW_DIR,
    RISK_FREE_ANNUAL,
    TICKER,
    TRADING_DAYS,
)


@dataclass
class Analysis:
    ticker: str
    panel: pd.DataFrame                  # dữ liệu gốc theo ngày
    prices: pd.Series
    benchmark: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series
    log_returns: pd.Series
    tech: pd.DataFrame
    capm: CAPM.CAPMResult
    apt: APT.APTResult
    summary: pd.DataFrame
    benchmark_summary: pd.DataFrame
    overview: pd.DataFrame = field(default_factory=pd.DataFrame)
    ratio: pd.DataFrame = field(default_factory=pd.DataFrame)
    shareholders: pd.DataFrame = field(default_factory=pd.DataFrame)
    rf_annual: float = RISK_FREE_ANNUAL

    @property
    def last_price(self) -> float:
        return float(self.prices.iloc[-1])

    @property
    def last_date(self) -> pd.Timestamp:
        return self.prices.index[-1]

    @property
    def mu_annual(self) -> float:
        """Lợi suất kép hằng năm (CAGR) — dùng để mô tả hiệu quả thực tế."""
        return M.annual_return(self.returns)

    @property
    def mu_arithmetic(self) -> float:
        """Trung bình số học của lợi suất, năm hoá."""
        return float(self.returns.mean()) * TRADING_DAYS

    @property
    def sigma_annual(self) -> float:
        return M.annual_volatility(self.returns)

    def price_change(self, days: int) -> float:
        """Biến động giá trong ``days`` phiên gần nhất."""
        if len(self.prices) <= days:
            return float("nan")
        return float(self.prices.iloc[-1] / self.prices.iloc[-1 - days] - 1)

    def yearly_returns(self) -> pd.Series:
        return (1 + self.returns).resample("YE").prod() - 1

    def simulate(
        self, horizon: int = 252, n_sims: int = 10_000, method: str = "gbm", seed: int = 42
    ) -> MC.SimulationResult:
        mu, sd = float(self.log_returns.mean()), float(self.log_returns.std(ddof=1))
        if method == "gbm":
            return MC.simulate_gbm(self.last_price, mu, sd, horizon, n_sims, seed)
        if method == "t":
            return MC.simulate_student_t(self.last_price, mu, sd, 4.0, horizon, n_sims, seed)
        return MC.simulate_bootstrap(
            self.last_price, self.log_returns.values, horizon, n_sims, 5, seed
        )

    def allocation(self, risk_aversion: float = 3.0) -> AL.AllocationResult:
        return AL.optimal_allocation(
            self.mu_arithmetic, self.sigma_annual, risk_aversion, self.rf_annual
        )


def _read_csv(path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return pd.DataFrame()


def load_analysis(
    ticker: str = TICKER,
    rf_annual: float = RISK_FREE_ANNUAL,
    start: str | None = None,
    end: str | None = None,
) -> Analysis:
    """Nạp dữ liệu từ ``data/`` và chạy toàn bộ mô hình."""
    panel_path = PROCESSED_DIR / f"panel_{ticker}.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Chưa có dữ liệu {panel_path}. Hãy chạy: python fetch_data.py"
        )
    panel = pd.read_csv(panel_path, parse_dates=["time"], index_col="time")
    if start is not None or end is not None:
        panel = panel.loc[start:end]
    if len(panel) < 60:
        raise ValueError("Giai đoạn được chọn quá ngắn để ước lượng mô hình (cần ≥ 60 phiên).")

    prices = panel["stock_close"].astype(float)
    benchmark = panel["benchmark_close"].astype(float)
    returns = M.to_returns(prices)
    bench_ret = M.to_returns(benchmark)
    log_ret = M.to_returns(prices, log=True)

    ohlc = panel.rename(columns=lambda c: c.replace("stock_", ""))
    tech = TA.add_indicators(ohlc[["open", "high", "low", "close", "volume"]])

    capm_res = CAPM.estimate_capm(returns, bench_ret, rf_annual)

    macro_cols = [c for c in MACRO_TICKERS if c in panel.columns]
    factors = APT.build_factors(bench_ret, panel[macro_cols].reset_index())
    apt_res = APT.estimate_apt(returns, factors, rf_annual)

    return Analysis(
        ticker=ticker,
        panel=panel,
        prices=prices,
        benchmark=benchmark,
        returns=returns,
        benchmark_returns=bench_ret,
        log_returns=log_ret,
        tech=tech,
        capm=capm_res,
        apt=apt_res,
        summary=M.summary_table(returns, rf_annual, ticker),
        benchmark_summary=M.summary_table(bench_ret, rf_annual, BENCHMARK),
        overview=_read_csv(RAW_DIR / f"overview_{ticker}.csv"),
        ratio=_read_csv(RAW_DIR / f"ratio_summary_{ticker}.csv"),
        shareholders=_read_csv(RAW_DIR / f"shareholders_{ticker}.csv"),
        rf_annual=rf_annual,
    )
