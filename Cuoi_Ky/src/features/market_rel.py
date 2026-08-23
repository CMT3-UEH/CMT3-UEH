# Đặc trưng quan hệ giữa FPT và thị trường.

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS


def _rolling_beta_alpha(
    r: pd.Series, rm: pd.Series, window: int
) -> tuple[pd.Series, pd.Series]:
    # Beta và alpha của hồi quy r theo rm trên cửa sổ trượt, tính bằng công thức đóng.
    cov = r.rolling(window).cov(rm)
    var = rm.rolling(window).var(ddof=1)
    beta = cov / var.replace(0.0, np.nan)
    alpha = r.rolling(window).mean() - beta * rm.rolling(window).mean()
    return beta, alpha


def market_relative_features(
    close: pd.Series, benchmark_close: pd.Series
) -> pd.DataFrame:
    # Sức mạnh tương đối, beta, alpha và tương quan trượt so với chỉ số.
    r = close.pct_change()
    rm = benchmark_close.pct_change()
    out = pd.DataFrame(index=close.index)

    # Sức mạnh tương đối: chênh lệch lợi suất tích luỹ so với chỉ số.
    for k in (20, 60, 120, 250):
        out[f"rel_ret_{k}"] = close.pct_change(k) - benchmark_close.pct_change(k)

    # Tỷ số giá FPT/chỉ số, chuẩn hoá theo mặt bằng 120 phiên.
    ty_so = close / benchmark_close.replace(0.0, np.nan)
    ma120 = ty_so.rolling(120).mean()
    sd120 = ty_so.rolling(120).std(ddof=1)
    out["rel_ratio_z"] = (ty_so - ma120) / sd120.replace(0.0, np.nan)
    out["rel_ratio_mom"] = ty_so.pct_change(20)

    # Beta và alpha trượt — cùng đại lượng đã ước lượng bằng OLS ở phần giữa kỳ,
    # ở đây tính bằng công thức đóng để chạy nhanh trên toàn chuỗi.
    for w in (60, 120):
        beta, alpha = _rolling_beta_alpha(r, rm, w)
        out[f"rel_beta_{w}"] = beta
        out[f"rel_alpha_{w}"] = alpha * TRADING_DAYS      # quy về đơn vị năm

    # Tương quan trượt: beta cao mà tương quan thấp thì beta không đáng tin.
    out["rel_corr_60"] = r.rolling(60).corr(rm)

    # Lợi suất vượt trội của phiên gần nhất và trung bình 5 phiên.
    vuot = r - rm
    out["rel_excess_1"] = vuot
    out["rel_excess_5"] = vuot.rolling(5).mean()

    return out.replace([np.inf, -np.inf], np.nan)
