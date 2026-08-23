# Gán nhãn cho bài toán học.

import numpy as np
import pandas as pd

from src.config import HORIZONS


def forward_return(px: pd.DataFrame, horizon: int, price: str = "open") -> pd.Series:
    # Lợi suất của vị thế mở ở phiên t+1 và đóng ở phiên t+1+h.
    p = px[price]
    vao = p.shift(-1)
    ra = p.shift(-(1 + horizon))
    return (ra / vao - 1.0).rename(f"y_{horizon}")


def label_end_time(index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    # Thời điểm nhãn của mỗi quan sát được biết kết quả.
    idx = pd.DatetimeIndex(index)
    vi_tri = np.minimum(np.arange(len(idx)) + 1 + horizon, len(idx) - 1)
    return pd.Series(idx[vi_tri], index=idx, name="t1")


def forward_returns(px: pd.DataFrame, horizons=HORIZONS, price: str = "open") -> pd.DataFrame:
    # Bảng nhãn đa tầm dự báo, mỗi cột là một tầm dự báo.
    return pd.concat([forward_return(px, h, price) for h in horizons], axis=1)


def build_labels(px: pd.DataFrame, horizons=HORIZONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Toàn bộ nhãn của dự án, kèm bảng `t1` tương ứng cho từng tầm dự báo.
    y = forward_returns(px, horizons)
    t1 = pd.DataFrame(
        {f"t1_{h}": label_end_time(px.index, h) for h in horizons}, index=px.index
    )
    return y, t1
