# Đặc trưng chế độ thị trường — đầu vào cho cổng điều tiết tỷ trọng nắm giữ.

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS


def _drawdown(close: pd.Series, window: int) -> pd.Series:
    # Mức sụt so với đỉnh cao nhất trong cửa sổ quá khứ (giá trị âm).
    dinh = close.rolling(window, min_periods=20).max()
    return close / dinh.replace(0.0, np.nan) - 1.0


def _percentile_rank(s: pd.Series, window: int) -> pd.Series:
    # Vị trí phân vị của giá trị hiện tại trong cửa sổ quá khứ, khoảng [0, 1].
    return s.rolling(window, min_periods=window // 2).rank(pct=True)


def regime_features(close: pd.Series, benchmark_close: pd.Series) -> pd.DataFrame:
    # Trạng thái biến động, sụt giảm và xu hướng của thị trường lẫn cổ phiếu.
    out = pd.DataFrame(index=close.index)

    for ten, s in (("stock", close), ("mkt", benchmark_close)):
        r = s.pct_change()

        # Biến động thực hiện ngắn và dài hạn, cùng vị trí phân vị của nó.
        vol20 = r.rolling(20).std(ddof=1) * np.sqrt(TRADING_DAYS)
        vol120 = r.rolling(120).std(ddof=1) * np.sqrt(TRADING_DAYS)
        out[f"reg_{ten}_vol20"] = vol20
        out[f"reg_{ten}_vol_pct"] = _percentile_rank(vol20, TRADING_DAYS)
        out[f"reg_{ten}_vol_ratio"] = vol20 / vol120.replace(0.0, np.nan)

        # Sụt giảm hiện hành so với đỉnh một năm.
        out[f"reg_{ten}_dd"] = _drawdown(s, TRADING_DAYS)

        # Xu hướng dài hạn: khoảng cách tới MA200 và số phiên đã nằm trên nó.
        ma200 = s.rolling(200).mean()
        out[f"reg_{ten}_ma200_dist"] = s / ma200.replace(0.0, np.nan) - 1.0
        out[f"reg_{ten}_above_ma200"] = (s > ma200).astype(float)

        # Tỷ lệ phiên tăng trong 60 phiên gần nhất — đo độ rộng của xu hướng.
        out[f"reg_{ten}_up_ratio_60"] = (r > 0).rolling(60).mean()

    # Độ lệch và độ nhọn trượt: đuôi trái dày lên thường báo hiệu chế độ căng thẳng.
    r = close.pct_change()
    out["reg_skew_60"] = r.rolling(60).skew()
    out["reg_kurt_60"] = r.rolling(60).kurt()

    return out.replace([np.inf, -np.inf], np.nan)
