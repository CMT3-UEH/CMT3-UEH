# Ghép mọi nhóm đặc trưng thành một ma trận đầu vào duy nhất.

import numpy as np
import pandas as pd

from src.config import HORIZONS

from .labeling import build_labels
from .macro import calendar_features, macro_features
from .market_rel import market_relative_features
from .microstructure import is_tradable, microstructure_features
from .multiframe import multiframe_features
from .regime import regime_features
from .technical import technical_features


# Cột không phải đặc trưng: giá dùng cho backtest, nhãn, và cờ trạng thái.
PRICE_COLS = ["open", "high", "low", "close", "volume", "benchmark_close"]
FLAG_COLS = ["tradable"]


def extract_prices(panel: pd.DataFrame) -> pd.DataFrame:
    # Tách bảng giá của cổ phiếu ra khỏi bảng tổng hợp, đổi về tên cột chuẩn.
    px = panel[[c for c in panel.columns if c.startswith("stock_")]].copy()
    px.columns = [c.replace("stock_", "") for c in px.columns]
    px["benchmark_close"] = panel["benchmark_close"]
    return px.astype(float)


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    # Ma trận đặc trưng đầy đủ, chưa cắt bỏ phần khởi động.
    px = extract_prices(panel)
    close, bm = px["close"], px["benchmark_close"]

    nhom = [
        technical_features(px),
        microstructure_features(px),
        multiframe_features(px),
        market_relative_features(close, bm),
        regime_features(close, bm),
        macro_features(panel),
        calendar_features(px.index),
    ]
    X = pd.concat(nhom, axis=1)

    trung = X.columns[X.columns.duplicated()].tolist()
    if trung:
        raise ValueError(f"Tên đặc trưng bị trùng: {trung}")

    return X.replace([np.inf, -np.inf], np.nan)


def build_dataset(panel: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    # Bảng làm việc hoàn chỉnh: giá + đặc trưng + nhãn + cờ giao dịch được.
    px = extract_prices(panel)
    X = build_features(panel)
    y, t1 = build_labels(px, horizons)

    # Cột rỗng hoàn toàn thường là do thiếu một nguồn dữ liệu; giữ lại sẽ khiến
    # phần cắt khởi động bên dưới xoá sạch bảng.
    rong = [c for c in X.columns if X[c].isna().all()]
    if rong:
        print(f"Bỏ {len(rong)} đặc trưng rỗng hoàn toàn: {', '.join(rong[:8])}"
              f"{' ...' if len(rong) > 8 else ''}")
        X = X.drop(columns=rong)

    ds = pd.concat([px, X, y, t1], axis=1)
    ds["tradable"] = is_tradable(px)

    # Cắt phần khởi động: giữ từ phiên đầu tiên mà mọi đặc trưng đều có giá trị.
    du_lieu_du = ds[list(X.columns)].notna().all(axis=1)
    if not du_lieu_du.any():
        thieu = ds[list(X.columns)].isna().sum().sort_values(ascending=False)
        raise ValueError(
            "Không có phiên nào đủ toàn bộ đặc trưng. Thiếu nhiều nhất:\n"
            f"{thieu.head(10)}"
        )
    return ds.loc[du_lieu_du.idxmax():]


def feature_columns(ds: pd.DataFrame) -> list[str]:
    # Danh sách cột đặc trưng, loại bỏ giá, nhãn và cờ.
    bo = set(PRICE_COLS) | set(FLAG_COLS)
    bo |= {c for c in ds.columns if c.startswith(("y_", "t1_", "tb_"))}
    return [c for c in ds.columns if c not in bo]


def feature_groups(cols: list[str]) -> dict[str, list[str]]:
    # Gom đặc trưng theo nhóm — dùng cho ablation theo nhóm đặc trưng.
    quy_tac = {
        "giá": ("mom_", "ma_dist", "ma_slope", "ma_above", "rev_", "macd_"),
        "biến động": ("vol_", "atr_"),
        "thanh khoản": ("liq_",),
        "vi cấu trúc": ("ms_",),
        "đa khung": ("wk_", "mo_"),
        "quan hệ thị trường": ("rel_",),
        "chế độ": ("reg_",),
        "vĩ mô": ("mac_",),
        "lịch": ("cal_",),
    }
    nhom: dict[str, list[str]] = {k: [] for k in quy_tac}
    nhom["khác"] = []
    for c in cols:
        for ten, tien_to in quy_tac.items():
            if c.startswith(tien_to):
                nhom[ten].append(c)
                break
        else:
            nhom["khác"].append(c)
    return {k: v for k, v in nhom.items() if v}


def assert_causal(panel: pd.DataFrame, n_kiem: int = 40, seed: int = 0) -> None:
    # Kiểm tra bằng thực nghiệm rằng không đặc trưng nào nhìn thấy tương lai.
    day_du = build_features(panel)
    rng = np.random.default_rng(seed)
    n = len(panel)
    diem = rng.choice(np.arange(int(n * 0.5), n - 1), size=min(n_kiem, n // 4),
                      replace=False)

    loi: dict[str, int] = {}
    for i in sorted(diem):
        cat = build_features(panel.iloc[: i + 1])
        t = panel.index[i]
        a, b = day_du.loc[t], cat.loc[t]
        lech = (a - b).abs() > 1e-9 * (1.0 + a.abs())
        lech &= a.notna() | b.notna()
        for col in lech[lech].index:
            loi[col] = loi.get(col, 0) + 1

    if loi:
        chi_tiet = ", ".join(f"{k} ({v}/{len(diem)} lần)" for k, v in sorted(loi.items()))
        raise AssertionError(
            f"Đặc trưng dùng thông tin tương lai: {chi_tiet}"
        )
    print(f"Kiểm tra tính nhân quả: {len(day_du.columns)} đặc trưng đạt "
          f"trên {len(diem)} điểm cắt ngẫu nhiên.")
