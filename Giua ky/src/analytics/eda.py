"""Phân tích khám phá dữ liệu (EDA) và kiểm định giả thiết của chuỗi thời gian."""

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")


# Tính dừng
def stationarity_tests(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Kiểm định ADF và KPSS cho từng chuỗi."""
    from statsmodels.tsa.stattools import adfuller, kpss

    rows = {}
    for name, s in series.items():
        s = pd.Series(s).dropna()
        adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
        kpss_stat, kpss_p = kpss(s, regression="c", nlags="auto")[:2]
        if adf_p < 0.05 and kpss_p >= 0.05:
            verdict = "Dừng — I(0)"
        elif adf_p >= 0.05 and kpss_p < 0.05:
            verdict = "Không dừng — I(1)"
        else:
            verdict = "Hai kiểm định không đồng thuận"
        rows[name] = {
            "ADF thống kê": adf_stat, "ADF p-value": adf_p,
            "KPSS thống kê": kpss_stat, "KPSS p-value": kpss_p,
            "Kết luận": verdict,
        }
    return pd.DataFrame(rows).T


# Phân phối
def normality_tests(returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Bộ kiểm định phân phối chuẩn kèm số quan sát ở vùng đuôi."""
    rows = {}
    for name, r in returns.items():
        r = pd.Series(r).dropna()
        z = (r - r.mean()) / r.std(ddof=1)
        jb_stat, jb_p = sps.jarque_bera(r)[:2]
        # Shapiro-Wilk giới hạn 5000 quan sát
        sw_stat, sw_p = sps.shapiro(r.sample(min(5000, len(r)), random_state=0))
        ks_stat, ks_p = sps.kstest(z, "norm")
        n = len(r)
        rows[name] = {
            "Số quan sát": n,
            "Độ lệch (Skewness)": float(r.skew()),
            "Độ nhọn vượt chuẩn": float(r.kurtosis()),
            "Jarque–Bera": jb_stat, "JB p-value": jb_p,
            "Shapiro–Wilk W": sw_stat, "SW p-value": sw_p,
            "Kolmogorov–Smirnov D": ks_stat, "KS p-value": ks_p,
            "Số phiên vượt ±3σ": int((z.abs() > 3).sum()),
            "Kỳ vọng nếu chuẩn (±3σ)": round(n * 0.0027, 1),
            "Số phiên vượt ±5σ": int((z.abs() > 5).sum()),
        }
    return pd.DataFrame(rows).T


def qq_data(returns: pd.Series, dist: str = "norm", df: float = 4.0) -> pd.DataFrame:
    """Toạ độ biểu đồ Q-Q: phân vị lý thuyết so với phân vị thực nghiệm."""
    r = pd.Series(returns).dropna()
    z = ((r - r.mean()) / r.std(ddof=1)).sort_values()
    p = (np.arange(1, len(z) + 1) - 0.5) / len(z)
    if dist == "t":
        theo = sps.t.ppf(p, df) / np.sqrt(df / (df - 2))    # chuẩn hoá phương sai 1
    else:
        theo = sps.norm.ppf(p)
    return pd.DataFrame({"ly_thuyet": theo, "thuc_nghiem": z.values})


# Tự tương quan và cụm biến động
def autocorrelation_tests(returns: dict[str, pd.Series],
                          lags: tuple[int, ...] = (1, 5, 10, 20)) -> pd.DataFrame:
    """Ljung–Box trên lợi suất và trên bình phương lợi suất."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rows = {}
    for name, r in returns.items():
        r = pd.Series(r).dropna()
        lb = acorr_ljungbox(r, lags=list(lags), return_df=True)
        lb2 = acorr_ljungbox(r ** 2, lags=list(lags), return_df=True)
        for lag in lags:
            rows[f"{name} — độ trễ {lag}"] = {
                "LB lợi suất": lb.loc[lag, "lb_stat"],
                "p-value": lb.loc[lag, "lb_pvalue"],
                "LB bình phương lợi suất": lb2.loc[lag, "lb_stat"],
                "p-value ": lb2.loc[lag, "lb_pvalue"],
            }
    return pd.DataFrame(rows).T


def arch_test(returns: pd.Series, lags: int = 10) -> dict[str, float]:
    """Kiểm định Engle ARCH-LM: phương sai có thay đổi theo thời gian không."""
    from statsmodels.stats.diagnostic import het_arch

    r = pd.Series(returns).dropna()
    lm, lm_p, f_stat, f_p = het_arch(r, nlags=lags)
    return {"LM": lm, "LM p-value": lm_p, "F": f_stat, "F p-value": f_p}


def acf_values(returns: pd.Series, nlags: int = 5) -> pd.DataFrame:
    """Hệ số tự tương quan của lợi suất và của bình phương lợi suất."""
    from statsmodels.tsa.stattools import acf

    r = pd.Series(returns).dropna()
    return pd.DataFrame({
        "Lợi suất": acf(r, nlags=nlags, fft=True)[1:],
        "Bình phương lợi suất": acf(r ** 2, nlags=nlags, fft=True)[1:],
    }, index=[f"Độ trễ {i}" for i in range(1, nlags + 1)])


# Quan sát ngoại lai
def outlier_summary(returns: dict[str, pd.Series], iqr_k: float = 3.0) -> pd.DataFrame:
    """Đếm quan sát ngoại lai theo hai tiêu chí: khoảng tứ phân vị và điểm z."""
    rows = {}
    for name, r in returns.items():
        r = pd.Series(r).dropna()
        q1, q3 = r.quantile(0.25), r.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        z = (r - r.mean()) / r.std(ddof=1)
        rows[name] = {
            f"Ngoại lai IQR ({iqr_k:g}×)": int(((r < lo) | (r > hi)).sum()),
            "Tỷ lệ IQR": float(((r < lo) | (r > hi)).mean()),
            "Số quan sát |z| > 4": int((z.abs() > 4).sum()),
            "Phiên giảm mạnh nhất": float(r.min()),
            "Phiên tăng mạnh nhất": float(r.max()),
            "Phân vị 1%": float(r.quantile(0.01)),
            "Phân vị 99%": float(r.quantile(0.99)),
        }
    return pd.DataFrame(rows).T


def robustness_check(stock_ret: pd.Series, market_ret: pd.Series,
                     rf_annual: float, quantile: float = 0.005) -> pd.DataFrame:
    """So sánh ước lượng CAPM: dữ liệu gốc, sau winsorize, sau loại ngoại lai."""
    from src.analytics.capm import estimate_capm

    df = pd.concat([stock_ret.rename("s"), market_ret.rename("m")], axis=1).dropna()
    lo, hi = df["s"].quantile(quantile), df["s"].quantile(1 - quantile)

    wins = df.copy()
    wins["s"] = wins["s"].clip(lo, hi)
    trimmed = df[(df["s"] >= lo) & (df["s"] <= hi)]

    out = {}
    for label, d in (("Dữ liệu gốc", df),
                     (f"Winsorize {quantile:.1%}", wins),
                     (f"Loại ngoại lai {quantile:.1%}", trimmed)):
        res = estimate_capm(d["s"], d["m"], rf_annual)
        out[label] = {
            "Số quan sát": f"{res.n_obs:,}",
            "Beta": round(res.beta, 4),
            "Alpha năm": round(res.alpha_annual, 4),
            "p-value α": round(res.p_alpha, 4),
            "R²": round(res.r_squared, 4),
        }
    return pd.DataFrame(out).T


# Thống kê theo giai đoạn
DEFAULT_PERIODS = (
    ("2006–2009 (khủng hoảng)", "2006-01-01", "2009-12-31"),
    ("2010–2015 (tích luỹ)", "2010-01-01", "2015-12-31"),
    ("2016–2019 (tăng trưởng)", "2016-01-01", "2019-12-31"),
    ("2020–2021 (Covid-19)", "2020-01-01", "2021-12-31"),
    ("2022 đến nay", "2022-01-01", None),
)


def period_statistics(stock_ret: pd.Series, market_ret: pd.Series,
                      periods=DEFAULT_PERIODS) -> pd.DataFrame:
    """Thống kê mô tả và beta theo từng chu kỳ thị trường."""
    from src.analytics.metrics import annual_return, annual_volatility

    rows = {}
    for label, start, end in periods:
        s = stock_ret.loc[start:end]
        m = market_ret.loc[start:end]
        if len(s) < 60:
            continue
        df = pd.concat([s.rename("s"), m.rename("m")], axis=1).dropna()
        beta = float(df["s"].cov(df["m"]) / df["m"].var())
        rows[label] = {
            "Số phiên": f"{len(df):,}",
            "Lợi suất năm": annual_return(s),
            "Biến động năm": annual_volatility(s),
            "Độ lệch": float(s.skew()),
            "Độ nhọn": float(s.kurtosis()),
            "Lợi suất VNINDEX": annual_return(m),
            "Tương quan": float(df["s"].corr(df["m"])),
            "Beta": beta,
        }
    return pd.DataFrame(rows).T


# Kiểm toán chất lượng dữ liệu
def data_quality_report(panel: pd.DataFrame, macro_cols: list[str]) -> pd.DataFrame:
    """Bảng kiểm toán chất lượng dữ liệu để công bố kèm báo cáo."""
    n = len(panel)
    close = panel["stock_close"]
    ohlc_bad = flat = 0
    if {"stock_high", "stock_low", "stock_open"}.issubset(panel.columns):
        op, hi, lo = panel["stock_open"], panel["stock_high"], panel["stock_low"]
        ohlc_bad = int(((hi < lo) | (hi < op.combine(close, max)) |
                        (lo > op.combine(close, min))).sum())
        flat = int(((op == hi) & (hi == lo) & (lo == close)).sum())

    rows = {
        "Số phiên trong mẫu phân tích": f"{n:,}",
        "Khoảng thời gian": f"{panel.index[0]:%d/%m/%Y} – {panel.index[-1]:%d/%m/%Y}",
        "Ngày bị trùng lặp": int(panel.index.duplicated().sum()),
        "Chỉ mục thời gian tăng dần": "Có" if panel.index.is_monotonic_increasing else "Không",
        "Ô thiếu ở giá cổ phiếu": int(close.isna().sum()),
        "Ô thiếu ở chỉ số VNINDEX": int(panel["benchmark_close"].isna().sum()),
        "Giá nhỏ hơn hoặc bằng 0": int((close <= 0).sum()),
        "Dòng vi phạm quan hệ OHLC": ohlc_bad,
        "Phiên có giá không đổi (o=h=l=c)": f"{flat:,} ({flat / n:.2%})",
    }
    for col in macro_cols:
        if col in panel.columns:
            filled = int((panel[col].diff() == 0).sum())
            rows[f"Tỷ lệ điền tiến — {col}"] = f"{filled / n:.2%}"
    return pd.DataFrame({"Giá trị": rows})
