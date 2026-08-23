# Đặc trưng đa khung thời gian — nhìn nhịp tuần và nhịp tháng mà không cần cửa sổ dài.

import numpy as np
import pandas as pd

from .technical import rsi, sma


def _resample_ohlcv(px: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    cols = {k: v for k, v in agg.items() if k in px.columns}
    return px.resample(rule).agg(cols).dropna(subset=["close"])


def _bo_khung(bars: pd.DataFrame, prefix: str, moms: tuple[int, ...]) -> pd.DataFrame:
    # Bộ đặc trưng chuẩn tính trên một khung thời gian bất kỳ.
    c = bars["close"]
    out = pd.DataFrame(index=bars.index)
    for k in moms:
        out[f"{prefix}_mom_{k}"] = c.pct_change(k)
    out[f"{prefix}_ma_dist"] = c / sma(c, moms[-1]) - 1.0
    out[f"{prefix}_rsi"] = rsi(c, 14) / 100.0
    out[f"{prefix}_vol"] = c.pct_change().rolling(moms[-1]).std(ddof=1)
    out[f"{prefix}_up_ratio"] = (c.pct_change() > 0).rolling(moms[-1]).mean()
    return out


def multiframe_features(px: pd.DataFrame) -> pd.DataFrame:
    # Đặc trưng tuần và tháng, đã đẩy lùi một kỳ rồi ghép về lịch ngày.
    khung = pd.DataFrame(index=px.index)

    for rule, prefix, moms in (("W-FRI", "wk", (1, 4, 12)), ("ME", "mo", (1, 3, 12))):
        bars = _resample_ohlcv(px, rule)
        if len(bars) < max(moms) + 2:
            continue
        feats = _bo_khung(bars, prefix, moms)

        # Đẩy lùi một kỳ: thông tin của kỳ đang diễn ra chưa được phép nhìn thấy.
        feats = feats.shift(1)

        # Ghép về lịch ngày bằng cách lấy giá trị của kỳ đã chốt gần nhất.
        khung = khung.join(feats.reindex(px.index, method="ffill"))

    return khung.replace([np.inf, -np.inf], np.nan)
