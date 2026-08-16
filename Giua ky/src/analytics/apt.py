"""Lý thuyết định giá kinh doanh chênh lệch (APT) với nhân tố vĩ mô."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analytics.metrics import daily_rf
from src.config import RISK_FREE_ANNUAL, TRADING_DAYS

FACTOR_LABELS = {
    "market": "Thị trường (VNINDEX)",
    "USDVND": "Tỷ giá USD/VND",
    "OIL": "Giá dầu WTI",
    "GOLD": "Giá vàng",
    "SP500": "Chứng khoán Mỹ (S&P 500)",
}


@dataclass
class APTResult:
    params: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    r_squared: float
    adj_r_squared: float
    n_obs: int
    factor_premia: pd.Series          # phần bù rủi ro trung bình năm của từng nhân tố
    contributions: pd.Series          # đóng góp vào lợi suất kỳ vọng (năm)
    expected_return: float
    vif: pd.Series
    data: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    model: object = field(repr=False, default=None)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Hệ số nhạy (b)": self.params,
            "t-stat": self.tvalues,
            "p-value": self.pvalues,
        })
        df.index = [FACTOR_LABELS.get(i, i) for i in df.index]
        return df


def build_factors(
    benchmark_ret: pd.Series, macro: pd.DataFrame, lag_foreign: int = 1
) -> pd.DataFrame:
    """Tạo bảng nhân tố: lợi suất thị trường + biến động % các biến vĩ mô."""
    factors = pd.DataFrame({"market": benchmark_ret})
    if macro is not None and not macro.empty:
        m = macro.copy()
        if "time" in m.columns:
            m["time"] = pd.to_datetime(m["time"])
            m = m.set_index("time")
        for col in m.columns:
            r = m[col].astype(float).pct_change()
            factors[col] = r.shift(lag_foreign) if lag_foreign else r
    return factors.dropna(how="all")


def _vif(X: pd.DataFrame) -> pd.Series:
    """Hệ số phóng đại phương sai — kiểm tra đa cộng tuyến giữa các nhân tố."""
    out = {}
    for col in X.columns:
        others = X.drop(columns=[col])
        r2 = sm.OLS(X[col], sm.add_constant(others)).fit().rsquared
        out[col] = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
    return pd.Series(out)


def estimate_apt(
    stock_ret: pd.Series,
    factors: pd.DataFrame,
    rf_annual: float = RISK_FREE_ANNUAL,
) -> APTResult:
    df = pd.concat([stock_ret.rename("stock"), factors], axis=1).dropna()
    rf_d = daily_rf(rf_annual)
    y = df["stock"] - rf_d
    X_raw = df.drop(columns=["stock"])
    X = sm.add_constant(X_raw)

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    # Phần bù mỗi nhân tố = trung bình số học năm hoá. Riêng thị trường phải trừ Rf
    # (Rm − Rf) như CAPM, nếu không lợi suất kỳ vọng bị cộng Rf hai lần.
    premia = X_raw.mean() * TRADING_DAYS
    if "market" in premia.index:
        premia["market"] = premia["market"] - rf_annual
    betas = model.params.drop("const")
    contributions = betas * premia
    expected = rf_annual + float(contributions.sum())

    return APTResult(
        params=model.params,
        tvalues=model.tvalues,
        pvalues=model.pvalues,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        n_obs=int(model.nobs),
        factor_premia=premia,
        contributions=contributions,
        expected_return=expected,
        vif=_vif(X_raw),
        data=df,
        model=model,
    )


def interpret(result: APTResult, ticker: str) -> list[str]:
    msgs = []
    betas = result.params.drop("const")
    sig = [f for f in betas.index if result.pvalues[f] < 0.05]
    if sig:
        parts = []
        for f in sig:
            direction = "cùng chiều" if betas[f] > 0 else "ngược chiều"
            parts.append(f"{FACTOR_LABELS.get(f, f)} ({direction}, b = {betas[f]:.2f})")
        msgs.append(
            f"Các nhân tố tác động có ý nghĩa thống kê tới {ticker}: " + "; ".join(parts) + "."
        )
    else:
        msgs.append("Không nhân tố nào đạt ý nghĩa thống kê ở mức 5%.")

    msgs.append(
        f"R² hiệu chỉnh = {result.adj_r_squared:.2f} — mô hình đa nhân tố giải thích "
        f"{result.adj_r_squared * 100:.0f}% biến động lợi suất của {ticker}."
    )
    msgs.append(
        f"Lợi suất kỳ vọng theo APT: {result.expected_return * 100:.2f}%/năm, "
        "tính bằng lãi suất phi rủi ro cộng tổng phần bù rủi ro của các nhân tố."
    )
    if (result.vif > 5).any():
        bad = ", ".join(result.vif[result.vif > 5].index)
        msgs.append(
            f"Cảnh báo đa cộng tuyến (VIF > 5) ở nhân tố: {bad}. "
            "Hệ số của các nhân tố này cần diễn giải thận trọng."
        )
    else:
        msgs.append("VIF của mọi nhân tố đều dưới 5 — không có đa cộng tuyến nghiêm trọng.")
    return msgs
