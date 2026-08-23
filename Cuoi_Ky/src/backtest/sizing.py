# Chuyển tín hiệu thô thành vị thế thực tế.

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS


def realised_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    # Biến động thực hiện thường niên hoá, chỉ dùng dữ liệu quá khứ.
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def no_trade_band(
    target: pd.Series,
    band: float | pd.Series = 0.10,
) -> pd.Series:
    # Vùng không giao dịch: chỉ đổi vị thế khi lệch quá ngưỡng.
    tgt = target.astype(float)
    nguong = (
        pd.Series(band, index=tgt.index) if np.isscalar(band)
        else pd.Series(band).reindex(tgt.index)
    ).astype(float)

    ra = np.empty(len(tgt))
    hien_tai = 0.0
    gia_tri = tgt.to_numpy()
    nguong_arr = nguong.to_numpy()

    for i in range(len(tgt)):
        muon = gia_tri[i]
        if np.isfinite(muon) and abs(muon - hien_tai) > nguong_arr[i]:
            hien_tai = muon
        ra[i] = hien_tai

    return pd.Series(ra, index=tgt.index, name="w_target")
