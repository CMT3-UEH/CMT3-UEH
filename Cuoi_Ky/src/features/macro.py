# Đặc trưng từ nhân tố vĩ mô — cùng bộ nhân tố đã dùng cho mô hình APT giữa kỳ.

import numpy as np
import pandas as pd

from src.config import MACRO_TICKERS


def macro_features(panel: pd.DataFrame) -> pd.DataFrame:
    # Lợi suất và biến động của các nhân tố vĩ mô, đã trễ một phiên.
    out = pd.DataFrame(index=panel.index)

    for ten in MACRO_TICKERS:
        if ten not in panel.columns:
            continue
        s = pd.to_numeric(panel[ten], errors="coerce")
        for k in (1, 5, 20):
            out[f"mac_{ten}_r{k}"] = s.pct_change(k)
        out[f"mac_{ten}_vol"] = s.pct_change().rolling(60).std(ddof=1)

    # Đẩy lùi một phiên cho toàn nhóm — xem phần giải thích ở đầu tệp.
    out = out.shift(1)
    return out.replace([np.inf, -np.inf], np.nan)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    # Đặc trưng lịch, mã hoá vòng tròn để tháng 12 nằm cạnh tháng 1.
    idx = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=idx)
    out["cal_dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
    out["cal_dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)
    out["cal_month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    out["cal_month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    out["cal_turn_of_month"] = ((idx.day <= 3) | (idx.day >= 26)).astype(float)
    return out
