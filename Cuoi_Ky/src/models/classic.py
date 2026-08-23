# Baseline nhóm A — mô hình cổ điển và chiến lược dựa trên quy tắc.

import numpy as np
import pandas as pd

from src.backtest.sizing import realised_vol
from src.config import VOL_TARGET
from src.features.technical import rsi, sma


def mua_va_nam_giu(px: pd.DataFrame) -> pd.Series:
    # A1 — mốc tham chiếu quan trọng nhất: mua rồi để yên.
    return pd.Series(1.0, index=px.index)


def nam_giu_dieu_tiet_bien_dong(
    px: pd.DataFrame, target: float = VOL_TARGET, window: int = 20
) -> pd.Series:
    # A3 — nắm giữ nhưng điều tiết tỷ trọng theo biến động.
    vol = realised_vol(px["close"].pct_change(), window)
    return (target / vol.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)


def giao_cat_trung_binh(px: pd.DataFrame, nhanh: int = 20, cham: int = 50) -> pd.Series:
    # A4 — nắm giữ khi đường trung bình nhanh nằm trên đường chậm.
    c = px["close"]
    return (sma(c, nhanh) > sma(c, cham)).astype(float).fillna(0.0)


def rsi_dao_chieu(
    px: pd.DataFrame, window: int = 14, vao: float = 30.0, ra: float = 70.0
) -> pd.Series:
    # A5 — mua khi quá bán, thoát khi quá mua.
    r = rsi(px["close"], window).to_numpy()
    w = np.zeros(len(r))
    dang_nam = False
    for i in range(len(r)):
        if not np.isnan(r[i]):
            if not dang_nam and r[i] <= vao:
                dang_nam = True
            elif dang_nam and r[i] >= ra:
                dang_nam = False
        w[i] = 1.0 if dang_nam else 0.0
    return pd.Series(w, index=px.index)


def momentum_tuyet_doi(
    px: pd.DataFrame, nhin_lai: int = 252, bo_qua: int = 21, loc_ma: int = 200
) -> pd.Series:
    # A6 — nhân tố momentum tuyệt đối 12–1, kèm bộ lọc xu hướng dài hạn.
    c = px["close"]
    mom = c.shift(bo_qua) / c.shift(nhin_lai) - 1.0
    tren_ma = c > sma(c, loc_ma)
    return ((mom > 0) & tren_ma).astype(float).fillna(0.0)


def but_pha_kenh_gia(px: pd.DataFrame, vao: int = 20, ra: int = 10) -> pd.Series:
    # A7 — bứt phá kênh Donchian: mua khi vượt đỉnh, thoát khi thủng đáy.
    dinh = px["high"].rolling(vao).max().shift(1).to_numpy()
    day = px["low"].rolling(ra).min().shift(1).to_numpy()
    c = px["close"].to_numpy()

    w = np.zeros(len(c))
    dang_nam = False
    for i in range(len(c)):
        if not dang_nam and np.isfinite(dinh[i]) and c[i] > dinh[i]:
            dang_nam = True
        elif dang_nam and np.isfinite(day[i]) and c[i] < day[i]:
            dang_nam = False
        w[i] = 1.0 if dang_nam else 0.0
    return pd.Series(w, index=px.index)


def bien_do_bollinger(px: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.Series:
    # A7b — biến thể hồi quy về trung bình: mua ở dải dưới, thoát ở đường giữa.
    c = px["close"]
    mid = sma(c, window)
    sd = c.rolling(window).std(ddof=1)
    duoi = (mid - n_std * sd).to_numpy()
    m = mid.to_numpy()
    cv = c.to_numpy()

    w = np.zeros(len(cv))
    dang_nam = False
    for i in range(len(cv)):
        if not dang_nam and np.isfinite(duoi[i]) and cv[i] < duoi[i]:
            dang_nam = True
        elif dang_nam and np.isfinite(m[i]) and cv[i] > m[i]:
            dang_nam = False
        w[i] = 1.0 if dang_nam else 0.0
    return pd.Series(w, index=px.index)


def loc_alpha_capm(ds: pd.DataFrame, window: int = 60) -> pd.Series:
    # A8 — nắm giữ khi alpha trượt so với VNINDEX đang dương.
    cot = f"rel_alpha_{window}"
    if cot not in ds.columns:
        raise KeyError(f"Bảng dữ liệu không có cột {cot}")
    return (ds[cot] > 0).astype(float).fillna(0.0)


def theo_xu_huong_thi_truong(ds: pd.DataFrame) -> pd.Series:
    # A8b — chỉ nắm giữ khi cả cổ phiếu lẫn thị trường đều trên MA200.
    a = ds.get("reg_stock_above_ma200")
    b = ds.get("reg_mkt_above_ma200")
    if a is None or b is None:
        raise KeyError("Bảng dữ liệu thiếu cột trạng thái so với MA200")
    return ((a > 0) & (b > 0)).astype(float).fillna(0.0)


# Lưới cấu hình dò trên tập kiểm định. Số lượng cấu hình ở đây được đếm vào tổng
# số phép thử khi tính Deflated Sharpe Ratio ở phần đánh giá.
LUOI_CO_DIEN = {
    "A3 · Nắm giữ điều tiết biến động": [
        {"target": t} for t in (0.10, 0.15, 0.20, 0.25)
    ],
    "A4 · Giao cắt trung bình động": [
        {"nhanh": a, "cham": b}
        for a, b in ((10, 30), (20, 50), (20, 100), (50, 200))
    ],
    "A5 · RSI đảo chiều": [
        {"vao": v, "ra": r} for v, r in ((30, 70), (25, 65), (35, 65), (20, 60))
    ],
    "A6 · Momentum tuyệt đối 12–1": [
        {"nhin_lai": n, "loc_ma": m}
        for n in (126, 252) for m in (100, 200)
    ],
    "A7 · Bứt phá kênh Donchian": [
        {"vao": a, "ra": b} for a, b in ((20, 10), (55, 20), (40, 20), (20, 20))
    ],
    "A7b · Hồi quy dải Bollinger": [
        {"window": w, "n_std": s} for w in (20, 40) for s in (1.5, 2.0)
    ],
    "A8 · Lọc alpha CAPM": [{"window": w} for w in (60, 120)],
}

HAM_CO_DIEN = {
    "A3 · Nắm giữ điều tiết biến động": nam_giu_dieu_tiet_bien_dong,
    "A4 · Giao cắt trung bình động": giao_cat_trung_binh,
    "A5 · RSI đảo chiều": rsi_dao_chieu,
    "A6 · Momentum tuyệt đối 12–1": momentum_tuyet_doi,
    "A7 · Bứt phá kênh Donchian": but_pha_kenh_gia,
    "A7b · Hồi quy dải Bollinger": bien_do_bollinger,
    "A8 · Lọc alpha CAPM": loc_alpha_capm,
}


def dem_cau_hinh() -> int:
    # Tổng số cấu hình nhóm A được dò trên tập kiểm định.
    return sum(len(v) for v in LUOI_CO_DIEN.values())
