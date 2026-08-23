# Đặc trưng vi cấu trúc phiên — thông tin nằm trong hình dạng nến, không phải trong giá đóng
# cửa.

import numpy as np
import pandas as pd

from src.config import PRICE_LIMIT


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def microstructure_features(px: pd.DataFrame) -> pd.DataFrame:
    # Đặc trưng từ quan hệ giữa open/high/low/close trong cùng một phiên.
    o, h, l, c = px["open"], px["high"], px["low"], px["close"]
    prev_c = c.shift(1)
    rng = h - l

    out = pd.DataFrame(index=px.index)

    # Giá đóng cửa nằm ở đâu trong biên độ ngày: 1 = đóng ở đỉnh, 0 = đóng ở đáy.
    out["ms_close_pos"] = _safe_div(c - l, rng)

    # Thân nến so với biên độ: đo mức độ quyết đoán của phiên.
    out["ms_body_ratio"] = _safe_div(c - o, rng)

    # Khoảng trống qua đêm và biên độ ngày, chuẩn hoá theo giá phiên trước.
    out["ms_gap"] = _safe_div(o - prev_c, prev_c)
    out["ms_range"] = _safe_div(rng, prev_c)

    # Râu trên và râu dưới: dấu hiệu lực mua bán bị từ chối trong phiên.
    than_tren = np.maximum(c, o)
    than_duoi = np.minimum(c, o)
    out["ms_upper_wick"] = _safe_div(h - than_tren, rng)
    out["ms_lower_wick"] = _safe_div(than_duoi - l, rng)

    # Biên độ hiện tại so với mặt bằng 60 phiên gần nhất.
    r60_mean = out["ms_range"].rolling(60).mean()
    r60_std = out["ms_range"].rolling(60).std(ddof=1)
    out["ms_range_z"] = _safe_div(out["ms_range"] - r60_mean, r60_std)

    # Khoảng trống trung bình 5 phiên: đo áp lực tích tụ ngoài giờ giao dịch.
    out["ms_gap_ma5"] = out["ms_gap"].rolling(5).mean()

    # Phiên chạm trần hoặc sàn — không phải lúc nào cũng khớp được lệnh.
    doi_gia = _safe_div(c - prev_c, prev_c)
    cham_tran = doi_gia >= PRICE_LIMIT * 0.98
    cham_san = doi_gia <= -PRICE_LIMIT * 0.98
    out["ms_limit_up_20"] = cham_tran.rolling(20).sum()
    out["ms_limit_down_20"] = cham_san.rolling(20).sum()

    # Phiên không khớp lệnh: giá mở, cao, thấp, đóng trùng nhau.
    out["ms_locked"] = ((rng == 0) & (px["volume"] > 0)).astype(float)

    return out


def is_tradable(px: pd.DataFrame) -> pd.Series:
    # Phiên có thể vào lệnh được hay không.
    o, h, l, c = px["open"], px["high"], px["low"], px["close"]
    prev_c = c.shift(1)
    doi_gia = _safe_div(c - prev_c, prev_c).abs()
    khoa = (h == l) & (o == c) & (doi_gia >= PRICE_LIMIT * 0.98)
    khong_khoi_luong = px["volume"].fillna(0) <= 0
    return ~(khoa.fillna(False) | khong_khoi_luong)
